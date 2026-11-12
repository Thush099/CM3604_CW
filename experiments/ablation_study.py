# ablation_study.py - Part 1 and Part 2 ablation experiments with 5-fold grouped CV

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factcheck import (
    AlwaysEntailedFactChecker,
    EntailmentFactChecker,
    EntailmentModel,
    WordRecallThresholdFactChecker,
)
from factchecking_main import read_fact_examples, read_passages


LABELS_PATH = ROOT / "data" / "dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = ROOT / "data" / "passages_bm25_ChatGPT_humfacts.jsonl"
RESULTS_DIR = ROOT / "experiments" / "results"

STOPWORDS = WordRecallThresholdFactChecker.STOPWORDS
STEMMER = PorterStemmer()


def load_examples_with_bio_ids():
    """Loads FactExample objects alongside each example's source biography index."""
    fact_to_passages = read_passages(str(PASSAGES_PATH))
    examples = read_fact_examples(str(LABELS_PATH), fact_to_passages)

    bio_map = {}
    with open(LABELS_PATH, "r", encoding="utf-8") as fh:
        for bio_idx, line in enumerate(fh):
            record = json.loads(line)
            if not record.get("annotations"):
                continue
            for sent in record["annotations"]:
                for fact in sent.get("human-atomic-facts") or []:
                    if fact["text"] in fact_to_passages and fact["text"] not in bio_map:
                        bio_map[fact["text"]] = bio_idx

    bio_ids = [bio_map[ex.fact] for ex in examples]
    return examples, bio_ids


def print_dataset_stats(examples, bio_ids):
    raw_counts = Counter(ex.label for ex in examples)
    binary_counts = Counter("NS" if ex.label == "IR" else ex.label for ex in examples)
    passage_counts = [len(ex.passages) for ex in examples]
    n_bios = len(set(bio_ids))

    print("=== Dataset Summary ===")
    print(f"Biographies: {n_bios} | Atomic facts: {len(examples)}")
    print(f"Raw labels: {dict(raw_counts)}")
    print(f"Binary labels: {dict(binary_counts)}")
    print(
        f"Passages per fact: min={min(passage_counts)}, "
        f"max={max(passage_counts)}, mean={sum(passage_counts)/len(passage_counts):.2f}"
    )
    print(f"Curated stopwords count: {len(STOPWORDS)}")
    print()


def split_blocks_and_sents(passages, granularity="sentence"):
    units = []
    for p in passages:
        text = p.get("text", "")
        if granularity == "all":
            units.append(text)
            continue
        if granularity == "passage":
            units.append(text)
            continue
        for block in re.split(r"</s>\s*<s>|<s>|</s>", text):
            for sent in re.split(r"(?<=[.!?])\s+", block):
                if sent.strip():
                    units.append(sent)
    if granularity == "all":
        return [" ".join(units)]
    return units


def tokenize(text, remove_stopwords=True, use_stem=True, exclude_tokens=None):
    cleaned = text.replace("<s>", " ").replace("</s>", " ").replace('""', " ").lower()
    words = re.findall(r"[a-z0-9]+", cleaned)
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    if use_stem:
        words = [STEMMER.stem(w) for w in words]
    tok_set = set(words)
    if exclude_tokens:
        tok_set = tok_set - exclude_tokens
    return tok_set


def tokenize_list(text, remove_stopwords=True, use_stem=True):
    cleaned = text.replace("<s>", " ").replace("</s>", " ").replace('""', " ").lower()
    words = re.findall(r"[a-z0-9]+", cleaned)
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    if use_stem:
        words = [STEMMER.stem(w) for w in words]
    return words


def base_title_tokens(passages, use_stem=True):
    if not passages:
        return set()
    raw_title = re.sub(r"\(.*?\)", " ", passages[0].get("title", "")).lower()
    words = re.findall(r"[a-z0-9]+", raw_title)
    if use_stem:
        return {STEMMER.stem(w) for w in words}
    return set(words)


def compute_tfidf_max(fact, units, remove_stopwords, use_stem):
    f_list = tokenize_list(fact, remove_stopwords=remove_stopwords, use_stem=use_stem)
    if not f_list:
        return 0.0
    u_lists = [tokenize_list(u, remove_stopwords=remove_stopwords, use_stem=use_stem) for u in units]
    docs = [f_list] + u_lists
    n_docs = len(docs)
    df = Counter()
    for doc in docs:
        for tok in set(doc):
            df[tok] += 1

    def vec_norm(tokens):
        tf = Counter(tokens)
        vec = {}
        norm_sq = 0.0
        for t, c in tf.items():
            idf = math.log((n_docs + 1) / (df[t] + 1)) + 1.0
            w = c * idf
            vec[t] = w
            norm_sq += w * w
        return vec, math.sqrt(norm_sq)

    f_vec, f_norm = vec_norm(f_list)
    if f_norm == 0.0:
        return 0.0

    best = 0.0
    for u_toks in u_lists:
        if not u_toks:
            continue
        u_vec, u_norm = vec_norm(u_toks)
        if u_norm == 0.0:
            continue
        dot = sum(w * u_vec[t] for t, w in f_vec.items() if t in u_vec)
        sim = dot / (f_norm * u_norm)
        if sim > best:
            best = sim
    return best


def score_dataset_part1(
    examples,
    metric="recall",
    granularity="sentence",
    remove_stopwords=True,
    use_stem=True,
    exclude_title=False,
):
    scorer_rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True) if metric == "rouge1" else None
    scores = []
    for ex in examples:
        units = split_blocks_and_sents(ex.passages, granularity=granularity)
        if metric == "rouge1":
            cleaned_fact = ex.fact.replace("<s>", " ").replace("</s>", " ")
            cleaned_unit = " ".join(units).replace("<s>", " ").replace("</s>", " ")
            scores.append(scorer_rouge.score(cleaned_fact, cleaned_unit)["rouge1"].recall)
            continue

        if metric == "tfidf":
            scores.append(compute_tfidf_max(ex.fact, units, remove_stopwords, use_stem))
            continue

        excl = base_title_tokens(ex.passages, use_stem=use_stem) if exclude_title else set()
        f_toks = tokenize(ex.fact, remove_stopwords=remove_stopwords, use_stem=use_stem, exclude_tokens=excl)
        if not f_toks:
            f_toks = tokenize(ex.fact, remove_stopwords=remove_stopwords, use_stem=use_stem)
        if not f_toks:
            scores.append(0.0)
            continue

        best = 0.0
        for u in units:
            u_toks = tokenize(u, remove_stopwords=remove_stopwords, use_stem=use_stem)
            if metric == "jaccard":
                union = len(f_toks | u_toks)
                val = len(f_toks & u_toks) / union if union > 0 else 0.0
            else:
                val = len(f_toks & u_toks) / len(f_toks)
            if val > best:
                best = val
        scores.append(best)
    return scores


def eval_threshold(scores, golds, threshold):
    correct = sum(
        1 for s, g in zip(scores, golds) if ("S" if s >= threshold else "NS") == g
    )
    return correct / len(golds)


def grouped_5fold_cv(scores, golds, bio_ids, thresholds):
    total_correct = 0
    for fold in range(5):
        train_idx = [i for i, b in enumerate(bio_ids) if b % 5 != fold]
        val_idx = [i for i, b in enumerate(bio_ids) if b % 5 == fold]
        best_t = max(
            thresholds,
            key=lambda t: sum(
                1 for i in train_idx if ("S" if scores[i] >= t else "NS") == golds[i]
            ),
        )
        total_correct += sum(
            1 for i in val_idx if ("S" if scores[i] >= best_t else "NS") == golds[i]
        )
    return total_correct / len(golds)


def run_part1_ablations(examples, bio_ids):
    golds = ["NS" if ex.label == "IR" else ex.label for ex in examples]
    thresholds = [round(i * 0.01, 2) for i in range(5, 96)]

    # Verify baseline and actual WordRecallThresholdFactChecker from factcheck.py
    baseline = AlwaysEntailedFactChecker()
    b0_acc = sum(1 for ex, g in zip(examples, golds) if baseline.predict(ex.fact, ex.passages) == g) / len(golds)

    t0 = time.perf_counter()
    wc_recall = WordRecallThresholdFactChecker(threshold=0.50, metric="recall")
    wc_preds = [wc_recall.predict(ex.fact, ex.passages) for ex in examples]
    wc_time = time.perf_counter() - t0
    wc_acc = sum(1 for p, g in zip(wc_preds, golds) if p == g) / len(golds)

    wc_jac = WordRecallThresholdFactChecker(threshold=0.20, metric="jaccard")
    wc_jac_preds = [wc_jac.predict(ex.fact, ex.passages) for ex in examples]
    wc_jac_acc = sum(1 for p, g in zip(wc_jac_preds, golds) if p == g) / len(golds)

    configs = [
        ("P1-1", "Jaccard Similarity (Sentence)", "Lowercase + alphanumeric", "jaccard", "sentence", False, False, False),
        ("P1-2", "Jaccard Similarity (Sentence)", "Stop-words + Porter stemmer", "jaccard", "sentence", True, True, False),
        ("P1-3", "TF-IDF Cosine (Sentence)", "Lowercase + alphanumeric", "tfidf", "sentence", False, False, False),
        ("P1-4", "TF-IDF Cosine (Sentence)", "Stop-words + Porter stemmer", "tfidf", "sentence", True, True, False),
        ("P1-5", "ROUGE-1 Recall (All Passages)", "Porter stemmer", "rouge1", "all", False, True, False),
        ("P1-6", "Unigram Recall (Passage)", "Lowercase + alphanumeric", "recall", "passage", False, False, False),
        ("P1-7", "Unigram Recall (Passage)", "Stop-words removed (no stem)", "recall", "passage", True, False, False),
        ("P1-8", "Unigram Recall (Passage)", "Stop-words + Porter stemmer", "recall", "passage", True, True, False),
        ("P1-9", "Unigram Recall (Sentence)", "Stop-words + title excl. (no stem)", "recall", "sentence", True, False, True),
        ("P1-10", "Unigram Recall (Sentence) [Selected]", "Stop-words + Porter + title excl.", "recall", "sentence", True, True, True),
    ]

    print("=== Part 1 Ablation Results ===")
    print(f"B0 Majority baseline (AlwaysEntailedFactChecker): {b0_acc*100:.2f}% ({int(round(b0_acc*len(golds)))}/{len(golds)})")
    print(f"factcheck.py WordRecallThresholdFactChecker(threshold=0.50, metric='recall'): {wc_acc*100:.2f}% ({int(round(wc_acc*len(golds)))}/{len(golds)}) in {wc_time:.2f}s")
    print(f"factcheck.py WordRecallThresholdFactChecker(threshold=0.20, metric='jaccard'): {wc_jac_acc*100:.2f}% ({int(round(wc_jac_acc*len(golds)))}/{len(golds)})")
    print()

    rows = [
        {
            "id": "B0",
            "method": "Majority Baseline (always_entail)",
            "preprocessing": "None",
            "opt_threshold": "N/A",
            "dev_accuracy": f"{b0_acc*100:.2f}%",
            "cv_5fold": f"{b0_acc*100:.2f}%",
            "runtime_s": "0.00",
        }
    ]

    p1_10_scores = None
    for cid, method, prep, metric, gran, rm_sw, stem, excl in configs:
        start = time.perf_counter()
        scores = score_dataset_part1(
            examples,
            metric=metric,
            granularity=gran,
            remove_stopwords=rm_sw,
            use_stem=stem,
            exclude_title=excl,
        )
        elapsed = time.perf_counter() - start
        best_t = max(thresholds, key=lambda t: eval_threshold(scores, golds, t))
        dev_acc = eval_threshold(scores, golds, best_t)
        cv_acc = grouped_5fold_cv(scores, golds, bio_ids, thresholds)
        if cid == "P1-10":
            p1_10_scores = scores
            best_t = 0.50
            dev_acc = eval_threshold(scores, golds, 0.50)

        rows.append(
            {
                "id": cid,
                "method": method,
                "preprocessing": prep,
                "opt_threshold": f"{best_t:.2f}",
                "dev_accuracy": f"{dev_acc*100:.2f}%",
                "cv_5fold": f"{cv_acc*100:.2f}%",
                "runtime_s": f"{elapsed:.2f}",
            }
        )
        print(
            f"{cid:5s} | {method:38s} | {prep:34s} | tau={best_t:.2f} | "
            f"Dev={dev_acc*100:.2f}% | CV={cv_acc*100:.2f}% | {elapsed:.2f}s"
        )

    print("\n=== Part 1 Selected Configuration (P1-10) Threshold Sensitivity ===")
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        acc = eval_threshold(p1_10_scores, golds, t)
        print(f"tau={t:.2f} -> {acc*100:.2f}% ({int(round(acc*len(golds)))}/{len(golds)})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "part1_ablation.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved Part 1 results to {out_csv}")


def run_part2_ablations(examples, bio_ids):
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    golds = ["NS" if ex.label == "IR" else ex.label for ex in examples]
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    hf_model.eval()
    if device != "cpu":
        hf_model = hf_model.to(device)
    ent_model = EntailmentModel(hf_model, tokenizer, cuda=False)
    fc = EntailmentFactChecker(ent_model, ent_threshold=0.38, prune_threshold=0.20)
    stemmer = fc.word_checker.stemmer

    def extract_full_passages(passages):
        out = []
        seen = set()
        for p in passages:
            raw_title = p.get("title", "").strip()
            clean_title = re.sub(r"\s*\(.*?\)\s*", "", raw_title).strip()
            txt = p.get("text", "").replace("<s>", " ").replace("</s>", " ")
            txt = re.sub(r"\{\{[^}]*(\}\}|$)", " ", txt).replace('""', "").replace("()", "")
            txt = re.sub(r"\s+", " ", txt).strip()
            if not txt:
                continue
            ctx = f"{clean_title}: {txt}" if clean_title and clean_title.lower() not in txt.lower() else txt
            if ctx not in seen:
                seen.add(ctx)
                out.append((txt, ctx))
        return out

    def extract_single_sentences(passages, with_title=False):
        out = []
        seen = set()
        for p in passages:
            raw_title = p.get("title", "").strip()
            clean_title = re.sub(r"\s*\(.*?\)\s*", "", raw_title).strip()
            title_words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", clean_title) if len(w) > 2]
            for block in re.split(r"</s>\s*<s>|<s>|</s>", p.get("text", "")):
                block = re.sub(r"\{\{[^}]*(\}\}|$)", " ", block)
                block = block.replace('""', "").replace("()", "")
                block = re.sub(r"\s+", " ", block).strip()
                if not block:
                    continue
                if raw_title and block.startswith(f"{raw_title} {raw_title}"):
                    block = block[len(raw_title) + 1:].strip()
                parts = re.split(
                    r"(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bJr)(?<!\bSr)(?<!\bNo)(?<!\b[A-Z])\.\s+(?=[A-Z0-9\"'-])",
                    block,
                )
                for s in parts:
                    s = s.strip()
                    if not s:
                        continue
                    if not s.endswith((".", "!", "?")):
                        s += "."
                    if len(re.findall(r"[A-Za-z0-9]+", s)) <= 1:
                        continue
                    if with_title and clean_title and title_words and not any(tw in s.lower() for tw in title_words):
                        ctx = f"{clean_title}: {s}"
                    else:
                        ctx = s
                    if ctx not in seen:
                        seen.add(ctx)
                        out.append((s, ctx))
        return out

    def gather_candidates(window_fn, min_ov=0.20, top_k=5):
        total_w = 0
        cands_by_ex = []
        for ex in examples:
            clean_title = re.sub(r"\(.*?\)", " ", ex.passages[0].get("title", ""))
            title_tokens = {stemmer.stem(tok) for tok in re.findall(r"[a-z0-9]+", clean_title.lower())}
            fact_words = fc.word_checker.preprocess(ex.fact, exclude_tokens=title_tokens)
            if not fact_words:
                fact_words = fc.word_checker.preprocess(ex.fact)

            windows = window_fn(ex.passages)
            total_w += len(windows)
            cands = []
            if fact_words:
                for raw_s, ctx in windows:
                    s_words = {stemmer.stem(tok) for tok in re.findall(r"[a-z0-9]+", raw_s.lower())}
                    ov = len(fact_words & s_words) / len(fact_words)
                    if ov > 0 and ov >= min_ov:
                        cands.append((ov, -len(raw_s), ctx))
            cands.sort(reverse=True)
            cands_by_ex.append(cands[:top_k])
        return total_w, cands_by_ex

    print(f"\n=== Preparing Part 2 Candidates (device={device}) ===")
    t0 = time.perf_counter()
    tot_pass, raw_pass = gather_candidates(extract_full_passages, min_ov=0.34, top_k=5)
    tot_s_plain, raw_s_plain = gather_candidates(lambda ps: extract_single_sentences(ps, with_title=False))
    tot_s_title, raw_s_title = gather_candidates(lambda ps: extract_single_sentences(ps, with_title=True))
    tot_ctx, raw_ctx = gather_candidates(fc.get_windows)

    unique_pairs = set()
    for ex, c_lists in zip(examples, zip(raw_pass, raw_s_plain, raw_s_title, raw_ctx)):
        for clist in c_lists:
            for _, _, ctx in clist:
                unique_pairs.add((ctx, ex.fact))

    # Length-sorted batching cuts padding overhead during ablation runs.
    sorted_pairs = sorted(unique_pairs, key=lambda pair: len(pair[0]) + len(pair[1]))
    pair_cache = {}
    batch_size = 16
    print(f"Batch scoring {len(sorted_pairs)} unique pairs (batch_size={batch_size}, device={device})...")
    for b_idx, start in enumerate(range(0, len(sorted_pairs), batch_size), start=1):
        batch = sorted_pairs[start : start + batch_size]
        with torch.no_grad():
            inputs = tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = hf_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().tolist()
        for pair, pr in zip(batch, probs):
            pair_cache[pair] = (float(pr[0]), float(pr[1]), float(pr[2]))
        del inputs, outputs, probs
        if device == "mps":
            torch.mps.empty_cache()
        if b_idx % 15 == 0 or start + batch_size >= len(sorted_pairs):
            print(f"  scored {min(start + batch_size, len(sorted_pairs))}/{len(sorted_pairs)} pairs ({time.perf_counter() - t0:.1f}s)")
    gc.collect()

    def attach_scores(raw_lists):
        out = []
        for ex, clist in zip(examples, raw_lists):
            out.append([(ov, nl, ctx, *pair_cache[(ctx, ex.fact)]) for ov, nl, ctx in clist])
        return out

    cands_pass = attach_scores(raw_pass)
    cands_sent_plain = attach_scores(raw_s_plain)
    cands_sent_titled = attach_scores(raw_s_title)
    per_ex_candidates = attach_scores(raw_ctx)

    cache_time = time.perf_counter() - t0
    print(f"Completed in {cache_time:.1f}s | Windows: pass={tot_pass}, 1-sent={tot_s_plain}, ctx={tot_ctx}")

    def simulate(cands_src, min_ov=0.20, ent_thr=0.38, use_guard=True, use_argmax=False, top_k=5):
        calls = 0
        preds = []
        for cands in cands_src:
            filtered = [c for c in cands if c[0] >= min_ov][:top_k]
            pred = "NS"
            for _, _, _, e_p, n_p, c_p in filtered:
                calls += 1
                if use_argmax:
                    if e_p >= n_p and e_p >= c_p:
                        pred = "S"
                        break
                else:
                    score = 0.0 if (use_guard and c_p > e_p) else e_p
                    if score >= ent_thr:
                        pred = "S"
                        break
            preds.append(pred)
        acc = sum(1 for p, g in zip(preds, golds) if p == g) / len(golds)
        return acc, calls, preds

    part2_rows = []
    def part2_cv(cands_src, min_ov=0.20, use_guard=True, top_k=5):
        cand_thresholds = [round(i * 0.01, 2) for i in range(15, 76)]
        ex_max_scores = []
        for cands in cands_src:
            filtered = [c for c in cands if c[0] >= min_ov][:top_k]
            best = 0.0
            for _, _, _, e_p, _, c_p in filtered:
                val = 0.0 if (use_guard and c_p > e_p) else e_p
                if val > best:
                    best = val
            ex_max_scores.append(best)
        return grouped_5fold_cv(ex_max_scores, golds, bio_ids, cand_thresholds)

    part2_specs = [
        ("P2-1", "Full Passages (<s> cleaned)", cands_pass, tot_pass, "Discrete argmax == 0", "ov>=0.34, topK=5", 0.34, 0.0, False, True, 5),
        ("P2-2", "Full Passages (<s> cleaned)", cands_pass, tot_pass, "Guard + P(ent) >= 0.38", "ov>=0.34, topK=5", 0.34, 0.38, True, False, 5),
        ("P2-3", "Isolated 1-Sentence", cands_sent_plain, tot_s_plain, "Discrete argmax == 0", "ov>=0.20, topK=5", 0.20, 0.0, False, True, 5),
        ("P2-4", "Isolated 1-Sentence", cands_sent_plain, tot_s_plain, "Guard + P(ent) >= 0.38", "ov>=0.20, topK=5", 0.20, 0.38, True, False, 5),
        ("P2-5", "Title-Grounded 1-Sentence", cands_sent_titled, tot_s_title, "Guard + P(ent) >= 0.38", "ov>=0.20, topK=5", 0.20, 0.38, True, False, 5),
        ("P2-6", "Contextual 1&2-Sent Windows", per_ex_candidates, tot_ctx, "Discrete argmax == 0", "ov>=0.20, topK=5", 0.20, 0.0, False, True, 5),
        ("P2-7", "Contextual 1&2-Sent Windows", per_ex_candidates, tot_ctx, "P(ent) >= 0.38 (no contra guard)", "ov>=0.20, topK=5", 0.20, 0.38, False, False, 5),
        ("P2-8", "Contextual 1&2-Sent Windows", per_ex_candidates, tot_ctx, "Guard + P(ent) >= 0.38", "ov>=0.34, topK=5", 0.34, 0.38, True, False, 5),
        ("P2-9", "Contextual 1&2-Sent Windows [Selected]", per_ex_candidates, tot_ctx, "Guard + P(ent) >= 0.38", "ov>=0.20, topK=5", 0.20, 0.38, True, False, 5),
    ]
    for pid, rep, cands_src, tot_w, mapping, prune_s, min_ov, thr, guard, argmax_mode, tk in part2_specs:
        acc, calls, _ = simulate(cands_src, min_ov=min_ov, ent_thr=thr, use_guard=guard, use_argmax=argmax_mode, top_k=tk)
        cv = "-" if argmax_mode else f"{part2_cv(cands_src, min_ov=min_ov, use_guard=guard, top_k=tk)*100:.2f}%"
        saved = (1.0 - calls / tot_w) * 100
        part2_rows.append(
            {
                "id": pid,
                "representation": rep,
                "binary_mapping": mapping,
                "pruning": prune_s,
                "nli_calls": f"{calls}/{tot_w}",
                "pct_saved": f"{saved:.2f}%",
                "dev_accuracy": f"{acc*100:.2f}%",
                "cv_5fold": cv,
            }
        )
        print(
            f"{pid:4s} | {rep:38s} | {mapping:32s} | {prune_s:16s} | calls={calls:4d}/{tot_w} "
            f"({saved:.2f}% saved) | Dev={acc*100:.2f}% | CV={cv}"
        )

    print("\n=== Part 2 Selected Configuration (P2-9) Threshold Sensitivity ===")
    for t in [0.25, 0.30, 0.35, 0.38, 0.40, 0.45, 0.50, 0.55]:
        acc, calls, _ = simulate(per_ex_candidates, min_ov=0.20, ent_thr=t, use_guard=True, use_argmax=False, top_k=5)
        print(f"tau_ent={t:.2f} -> {acc*100:.2f}% ({int(round(acc*len(golds)))}/{len(golds)}) | calls={calls}")

    out_csv = RESULTS_DIR / "part2_ablation.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(part2_rows[0].keys()))
        writer.writeheader()
        writer.writerows(part2_rows)
    print(f"Saved Part 2 results to {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-part2", action="store_true", help="Also run Part 2 NLI ablations on CPU.")
    args = parser.parse_args()

    examples, bio_ids = load_examples_with_bio_ids()
    print_dataset_stats(examples, bio_ids)
    run_part1_ablations(examples, bio_ids)
    if args.with_part2:
        run_part2_ablations(examples, bio_ids)


if __name__ == "__main__":
    main()
