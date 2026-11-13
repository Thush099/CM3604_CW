"""Part 1 preprocessing, scoring metric, threshold, and 5-fold grouped CV ablations."""

import argparse
import csv
import json
import math
import re
import time
from collections import Counter

from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer

from factcheck import AlwaysEntailedFactChecker, WordRecallThresholdFactChecker
from factchecking_main import read_fact_examples, read_passages

STOPWORDS = WordRecallThresholdFactChecker.STOPWORDS
STEMMER = PorterStemmer()


def load_examples_with_bio_ids(labels_path: str, passages_path: str):
    """Load FactExample objects and map each example to its source biography index."""
    fact_to_passages = read_passages(passages_path)
    examples = read_fact_examples(labels_path, fact_to_passages)

    fact_to_bio = {}
    with open(labels_path, "r", encoding="utf-8") as fh:
        for bio_idx, line in enumerate(fh):
            record = json.loads(line)
            for sent in record.get("annotations") or []:
                for fact in sent.get("human-atomic-facts") or []:
                    if fact["text"] in fact_to_passages and fact["text"] not in fact_to_bio:
                        fact_to_bio[fact["text"]] = bio_idx

    bio_ids = [fact_to_bio[ex.fact] for ex in examples]
    return examples, bio_ids


def split_units(passages, granularity: str = "sentence"):
    """Split passages at sentence, passage, or concatenated document granularity."""
    if granularity in ("all", "passage"):
        units = [p.get("text", "") for p in passages]
        return [" ".join(units)] if granularity == "all" else units

    units = []
    for p in passages:
        for block in re.split(r"</s>\s*<s>|<s>|</s>", p.get("text", "")):
            for sent in re.split(r"(?<=[.!?])\s+", block):
                if sent.strip():
                    units.append(sent)
    return units


def tokenize(text: str, remove_stopwords: bool = True, use_stem: bool = True, exclude=None):
    """Normalize text into a list of alphanumeric tokens with optional filtering/stemming."""
    cleaned = text.replace("<s>", " ").replace("</s>", " ").replace('""', " ").lower()
    words = re.findall(r"[a-z0-9]+", cleaned)
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    if use_stem:
        words = [STEMMER.stem(w) for w in words]
    if exclude:
        words = [w for w in words if w not in exclude]
    return words


def title_tokens(passages, use_stem: bool = True):
    """Extract normalized subject-title tokens, excluding parenthetical qualifiers."""
    if not passages:
        return set()
    raw_title = re.sub(r"\(.*?\)", " ", passages[0].get("title", "")).lower()
    words = re.findall(r"[a-z0-9]+", raw_title)
    return {STEMMER.stem(w) for w in words} if use_stem else set(words)


def tfidf_max_cosine(fact: str, units, remove_stopwords: bool, use_stem: bool) -> float:
    """Compute max local TF-IDF cosine similarity between fact and passage units."""
    f_tokens = tokenize(fact, remove_stopwords=remove_stopwords, use_stem=use_stem)
    if not f_tokens:
        return 0.0
    u_tokens_list = [tokenize(u, remove_stopwords=remove_stopwords, use_stem=use_stem) for u in units]
    docs = [f_tokens] + u_tokens_list
    n_docs = len(docs)

    df = Counter()
    for doc in docs:
        for tok in set(doc):
            df[tok] += 1

    def to_vec(tokens):
        tf = Counter(tokens)
        vec = {t: c * (math.log((n_docs + 1) / (df[t] + 1)) + 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return vec, norm

    f_vec, f_norm = to_vec(f_tokens)
    if f_norm == 0.0:
        return 0.0

    best = 0.0
    for u_toks in u_tokens_list:
        if not u_toks:
            continue
        u_vec, u_norm = to_vec(u_toks)
        if u_norm == 0.0:
            continue
        dot = sum(w * u_vec[t] for t, w in f_vec.items() if t in u_vec)
        best = max(best, dot / (f_norm * u_norm))
    return best


def score_examples(
    examples,
    metric: str = "recall",
    granularity: str = "sentence",
    remove_stopwords: bool = True,
    use_stem: bool = True,
    exclude_title: bool = False,
):
    """Score every example under the specified Part 1 configuration."""
    rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True) if metric == "rouge1" else None
    scores = []

    for ex in examples:
        units = split_units(ex.passages, granularity=granularity)
        if rouge is not None:
            fact_clean = ex.fact.replace("<s>", " ").replace("</s>", " ")
            text_clean = " ".join(units).replace("<s>", " ").replace("</s>", " ")
            scores.append(rouge.score(fact_clean, text_clean)["rouge1"].recall)
            continue

        if metric == "tfidf":
            scores.append(tfidf_max_cosine(ex.fact, units, remove_stopwords, use_stem))
            continue

        excl = title_tokens(ex.passages, use_stem=use_stem) if exclude_title else set()
        f_set = set(tokenize(ex.fact, remove_stopwords, use_stem, exclude=excl))
        if not f_set:
            f_set = set(tokenize(ex.fact, remove_stopwords, use_stem))
        if not f_set:
            scores.append(0.0)
            continue

        best = 0.0
        for unit in units:
            u_set = set(tokenize(unit, remove_stopwords, use_stem))
            if metric == "jaccard":
                union = len(f_set | u_set)
                val = len(f_set & u_set) / union if union else 0.0
            else:
                val = len(f_set & u_set) / len(f_set)
            if val > best:
                best = val
        scores.append(best)
    return scores


def eval_accuracy(scores, golds, threshold: float) -> float:
    """Evaluate thresholded accuracy against binary gold labels."""
    correct = sum(1 for s, g in zip(scores, golds) if ("S" if s >= threshold else "NS") == g)
    return correct / len(golds)


def grouped_5fold_cv(scores, golds, bio_ids, thresholds) -> float:
    """5-fold CV grouped by biography index so claims from one bio stay together."""
    correct = 0
    for fold in range(5):
        train_idx = [i for i, b in enumerate(bio_ids) if b % 5 != fold]
        val_idx = [i for i, b in enumerate(bio_ids) if b % 5 == fold]
        best_t = max(
            thresholds,
            key=lambda t: sum(1 for i in train_idx if ("S" if scores[i] >= t else "NS") == golds[i]),
        )
        correct += sum(1 for i in val_idx if ("S" if scores[i] >= best_t else "NS") == golds[i])
    return correct / len(golds)


def run_ablations(labels_path: str, passages_path: str, save_csv: str = "") -> None:
    """Run Part 1 metric/preprocessing ablations and threshold sensitivity sweep."""
    examples, bio_ids = load_examples_with_bio_ids(labels_path, passages_path)
    golds = ["NS" if ex.label == "IR" else ex.label for ex in examples]
    thresholds = [round(i * 0.01, 2) for i in range(5, 96)]

    baseline = AlwaysEntailedFactChecker()
    b0_acc = sum(1 for ex, g in zip(examples, golds) if baseline.predict(ex.fact, ex.passages) == g) / len(golds)

    word_checker = WordRecallThresholdFactChecker(threshold=0.50, metric="recall")
    wc_acc = sum(1 for ex, g in zip(examples, golds) if word_checker.predict(ex.fact, ex.passages) == g) / len(golds)

    configs = [
        ("P1-1", "Jaccard (Sentence)", "Lowercase + alphanumeric", "jaccard", "sentence", False, False, False),
        ("P1-2", "Jaccard (Sentence)", "Stopwords + Porter stemmer", "jaccard", "sentence", True, True, False),
        ("P1-3", "TF-IDF Cosine (Sentence)", "Lowercase + alphanumeric", "tfidf", "sentence", False, False, False),
        ("P1-4", "TF-IDF Cosine (Sentence)", "Stopwords + Porter stemmer", "tfidf", "sentence", True, True, False),
        ("P1-5", "ROUGE-1 Recall (All Passages)", "Porter stemmer", "rouge1", "all", False, True, False),
        ("P1-6", "Unigram Recall (Passage)", "Lowercase + alphanumeric", "recall", "passage", False, False, False),
        ("P1-7", "Unigram Recall (Passage)", "Stopwords (no stem)", "recall", "passage", True, False, False),
        ("P1-8", "Unigram Recall (Passage)", "Stopwords + Porter stemmer", "recall", "passage", True, True, False),
        ("P1-9", "Unigram Recall (Sentence)", "Stopwords + title excl. (no stem)", "recall", "sentence", True, False, True),
        ("P1-10", "Unigram Recall (Sentence) [factcheck.py]", "Stopwords + Porter + title excl.", "recall", "sentence", True, True, True),
    ]

    print(f"Majority baseline (always_entail): {b0_acc * 100:.2f}% ({int(round(b0_acc * len(golds)))}/{len(golds)})")
    print(f"factcheck.py WordRecallThresholdFactChecker(0.50): {wc_acc * 100:.2f}% ({int(round(wc_acc * len(golds)))}/{len(golds)})\n")

    rows = []
    selected_scores = None
    for cid, method, prep, metric, gran, rm_sw, stem, excl in configs:
        t0 = time.perf_counter()
        scores = score_examples(examples, metric, gran, rm_sw, stem, excl)
        elapsed = time.perf_counter() - t0
        best_t = max(thresholds, key=lambda t: eval_accuracy(scores, golds, t))
        if cid == "P1-10":
            selected_scores = scores
            best_t = 0.50
        dev_acc = eval_accuracy(scores, golds, best_t)
        cv_acc = grouped_5fold_cv(scores, golds, bio_ids, thresholds)

        rows.append(
            {
                "id": cid,
                "method": method,
                "preprocessing": prep,
                "threshold": f"{best_t:.2f}",
                "dev_accuracy": f"{dev_acc * 100:.2f}%",
                "cv_5fold": f"{cv_acc * 100:.2f}%",
                "runtime_s": f"{elapsed:.2f}",
            }
        )
        print(
            f"{cid:5s} | {method:40s} | {prep:33s} | tau={best_t:.2f} | "
            f"Dev={dev_acc * 100:.2f}% | CV={cv_acc * 100:.2f}% | {elapsed:.2f}s"
        )

    print("\n=== Threshold Sensitivity (P1-10 Sentence Unigram Recall) ===")
    for t in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        acc = eval_accuracy(selected_scores, golds, t)
        print(f"tau={t:.2f} -> {acc * 100:.2f}% ({int(round(acc * len(golds)))}/{len(golds)})")

    if save_csv:
        with open(save_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved ablation results to {save_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 1 lexical overlap ablations.")
    parser.add_argument("--labels_path", default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument("--passages_path", default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument("--save", default="", help="Optional CSV output path (ignored by git).")
    args = parser.parse_args()
    run_ablations(args.labels_path, args.passages_path, save_csv=args.save)


if __name__ == "__main__":
    main()
