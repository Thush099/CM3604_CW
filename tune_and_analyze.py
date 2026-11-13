"""Part 2 threshold/pruning check and Part 3 FP/FN error export and analysis."""

import argparse
import json
import re
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from factcheck import EntailmentFactChecker, EntailmentModel
from factchecking_main import print_eval_stats, read_fact_examples, read_passages


def score_unique_candidates(examples, checker, hf_model, tokenizer, device: str):
    """Score top-5 candidate windows down to overlap >= 0.15 once per unique (premise, fact)."""
    stemmer = checker.word_checker.stemmer
    total_windows = 0
    per_ex_cands = []
    unique_pairs = set()

    for ex in examples:
        clean_title = re.sub(r"\(.*?\)", " ", ex.passages[0].get("title", "")) if ex.passages else ""
        title_tokens = {stemmer.stem(tok) for tok in re.findall(r"[a-z0-9]+", clean_title.lower())}
        fact_words = checker.word_checker.preprocess(ex.fact, exclude_tokens=title_tokens)
        if not fact_words:
            fact_words = checker.word_checker.preprocess(ex.fact)

        windows = checker.get_windows(ex.passages)
        total_windows += len(windows)
        cands = []
        if fact_words:
            for raw_sent, context in windows:
                sent_words = {stemmer.stem(tok) for tok in re.findall(r"[a-z0-9]+", raw_sent.lower())}
                overlap = len(fact_words & sent_words) / len(fact_words)
                if overlap >= 0.15:
                    cands.append((overlap, -len(raw_sent), context))
        cands.sort(reverse=True)
        per_ex_cands.append(cands)
        for min_ov in (0.15, 0.20, 0.25, 0.30, 0.34):
            for ov, _, ctx in [c for c in cands if c[0] >= min_ov][:5]:
                unique_pairs.add((ctx, ex.fact))

    sorted_pairs = sorted(unique_pairs, key=lambda pair: len(pair[0]) + len(pair[1]))
    pair_probs = {}
    batch_size = 16 if device != "cpu" else 8

    for start in range(0, len(sorted_pairs), batch_size):
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
            logits = hf_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
        for pair, pr in zip(batch, probs):
            pair_probs[pair] = (float(pr[0]), float(pr[1]), float(pr[2]))
        del inputs, logits, probs

    return total_windows, per_ex_cands, pair_probs


def simulate_setting(examples, per_ex_cands, pair_probs, golds, prune_thr: float, ent_thr: float, use_guard: bool, use_argmax: bool):
    """Evaluate cached NLI probabilities under a pruning and binary decision configuration."""
    calls = 0
    preds = []
    for ex, cands in zip(examples, per_ex_cands):
        filtered = [c for c in cands if c[0] >= prune_thr][:5]
        pred = "NS"
        for _, _, ctx in filtered:
            calls += 1
            e_p, n_p, c_p = pair_probs[(ctx, ex.fact)]
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


def run_tune_and_analyze(labels_path: str, passages_path: str, export_path: str, use_cuda: bool) -> None:
    """Sweep Part 2 thresholds/pruning and export Part 3 FP/FN errors."""
    fact_to_passages = read_passages(passages_path)
    examples = read_fact_examples(labels_path, fact_to_passages)
    golds = ["NS" if ex.label == "IR" else ex.label for ex in examples]

    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    hf_model.eval()

    if use_cuda and torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if device != "cpu":
        hf_model = hf_model.to(device)

    ent_model = EntailmentModel(hf_model, tokenizer, cuda=(device == "cuda"))
    checker = EntailmentFactChecker(ent_model)

    t0 = time.perf_counter()
    total_windows, per_ex_cands, pair_probs = score_unique_candidates(
        examples, checker, hf_model, tokenizer, device
    )
    print(f"Scored {len(pair_probs)} unique candidate pairs on {device} in {time.perf_counter() - t0:.1f}s")
    print(f"Total contextual 1&2-sentence windows across dataset: {total_windows}\n")

    print("=== Part 2 Decision Rule & Threshold Sweep (prune_threshold=0.20) ===")
    acc_arg, calls_arg, _ = simulate_setting(examples, per_ex_cands, pair_probs, golds, 0.20, 0.0, False, True)
    print(f"Discrete argmax == 0       | Acc={acc_arg * 100:.2f}% | Calls={calls_arg}/{total_windows}")
    acc_noguard, calls_ng, _ = simulate_setting(examples, per_ex_cands, pair_probs, golds, 0.20, 0.38, False, False)
    print(f"P(ent) >= 0.38 (no guard)  | Acc={acc_noguard * 100:.2f}% | Calls={calls_ng}/{total_windows}")
    for thr in (0.25, 0.30, 0.35, 0.38, 0.40, 0.45, 0.50):
        acc, calls, _ = simulate_setting(examples, per_ex_cands, pair_probs, golds, 0.20, thr, True, False)
        marker = " [selected]" if thr == 0.38 else ""
        print(f"Guard + P(ent) >= {thr:.2f}     | Acc={acc * 100:.2f}% | Calls={calls}/{total_windows}{marker}")

    print("\n=== Part 2 Pruning Threshold Check (ent_threshold=0.38) ===")
    for p_thr in (0.15, 0.20, 0.25, 0.30, 0.34):
        acc, calls, _ = simulate_setting(examples, per_ex_cands, pair_probs, golds, p_thr, 0.38, True, False)
        saved = (1.0 - calls / total_windows) * 100
        marker = " [selected]" if p_thr == 0.20 else ""
        print(f"prune_threshold={p_thr:.2f} | Acc={acc * 100:.2f}% | Calls={calls}/{total_windows} ({saved:.2f}% saved){marker}")

    # Run Part 3 error audit using the exact selected configuration (0.20 / 0.38)
    print("\n=== Part 3 Confusion Matrix & FP/FN Audit (EntailmentFactChecker) ===")
    indexer = ["S", "NS"]
    confusion = [[0, 0], [0, 0]]
    fps = []
    fns = []

    for idx, (ex, cands) in enumerate(zip(examples, per_ex_cands)):
        filtered = [c for c in cands if c[0] >= 0.20][:5]
        evaluated = []
        best_score = 0.0
        pred = "NS"
        for ov, _, ctx in filtered:
            e_p, n_p, c_p = pair_probs[(ctx, ex.fact)]
            score = 0.0 if c_p > e_p else e_p
            evaluated.append({"overlap": round(ov, 3), "premise": ctx, "p_ent": round(e_p, 4), "p_neu": round(n_p, 4), "p_con": round(c_p, 4), "score": round(score, 4)})
            if score > best_score:
                best_score = score
            if best_score >= 0.38:
                pred = "S"
                break

        gold_bin = "NS" if ex.label == "IR" else ex.label
        confusion[indexer.index(gold_bin)][indexer.index(pred)] += 1

        if pred != gold_bin:
            record = {
                "index": idx,
                "title": ex.passages[0].get("title", "") if ex.passages else "",
                "raw_label": ex.label,
                "gold": gold_bin,
                "pred": pred,
                "fact": ex.fact,
                "best_score": round(best_score, 4),
                "evaluated": evaluated,
            }
            if gold_bin == "NS":
                fps.append(record)
            else:
                fns.append(record)

    print_eval_stats(confusion, indexer)
    print(f"\nFalse Positives ({len(fps)}):")
    for rec in fps:
        top_p = rec["evaluated"][-1]["premise"][:90] if rec["evaluated"] else "<pruned>"
        print(f"  [{rec['index']:3d}] ({rec['raw_label']} -> {rec['pred']}, score={rec['best_score']:.3f}) {rec['fact']} | {top_p}")

    print(f"\nFalse Negatives ({len(fns)}, showing first 10):")
    for rec in fns[:10]:
        top_p = rec["evaluated"][0]["premise"][:90] if rec["evaluated"] else "<all pruned (0 calls)>"
        print(f"  [{rec['index']:3d}] ({rec['raw_label']} -> {rec['pred']}, max={rec['best_score']:.3f}) {rec['fact']} | {top_p}")

    if export_path:
        with open(export_path, "w", encoding="utf-8") as fh:
            for err_type, records in (("FP", fps), ("FN", fns)):
                for rec in records:
                    fh.write(json.dumps({"error_type": err_type, **rec}) + "\n")
        print(f"\nExported {len(fps)} FPs and {len(fns)} FNs to {export_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2 tuning checks and Part 3 FP/FN analysis.")
    parser.add_argument("--labels_path", default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument("--passages_path", default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument("--export", default="errors_entailment.jsonl", help="JSONL file for FP/FN dump (empty to skip).")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA GPU if available.")
    args = parser.parse_args()
    run_tune_and_analyze(args.labels_path, args.passages_path, args.export, args.cuda)


if __name__ == "__main__":
    main()
