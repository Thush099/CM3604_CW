# error_analysis.py - dump Part 3 FP/FN errors and verify error_analysis_sheet.csv

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factcheck import EntailmentFactChecker, EntailmentModel
from factchecking_main import print_eval_stats, read_fact_examples, read_passages

LABELS_PATH = ROOT / "data" / "dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = ROOT / "data" / "passages_bm25_ChatGPT_humfacts.jsonl"
RESULTS_DIR = ROOT / "experiments" / "results"
SHEET_PATH = ROOT / "experiments" / "error_analysis_sheet.csv"


def inspect_three_class(hf_model, tokenizer, premise: str, hypothesis: str):
    """Returns full (entailment, neutral, contradiction) probabilities."""
    with torch.no_grad():
        inputs = tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        outputs = hf_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        res = (float(probs[0].item()), float(probs[1].item()), float(probs[2].item()))
    del inputs, outputs, probs
    return res


def main():
    fact_to_passages = read_passages(str(PASSAGES_PATH))
    examples = read_fact_examples(str(LABELS_PATH), fact_to_passages)

    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    ent_model = EntailmentModel(hf_model, tokenizer, cuda=False)
    checker = EntailmentFactChecker(ent_model)

    call_log = []
    orig_check = ent_model.check_entailment

    def tracked_check(premise, hypothesis):
        score = orig_check(premise, hypothesis)
        call_log.append((premise, hypothesis, score))
        return score

    ent_model.check_entailment = tracked_check

    indexer = ["S", "NS"]
    confusion = [[0, 0], [0, 0]]
    total_windows = 0
    total_calls = 0
    fps = []
    fns = []

    t0 = time.perf_counter()
    for idx, ex in enumerate(examples):
        windows = checker.get_windows(ex.passages)
        total_windows += len(windows)
        call_log.clear()

        pred = checker.predict(ex.fact, ex.passages)
        total_calls += len(call_log)
        gold_bin = "NS" if ex.label == "IR" else ex.label

        confusion[indexer.index(gold_bin)][indexer.index(pred)] += 1
        title = ex.passages[0].get("title", "") if ex.passages else ""

        if pred != gold_bin:
            record = {
                "example_index": idx,
                "entity_title": title,
                "fact": ex.fact,
                "raw_gold_label": ex.label,
                "binary_gold_label": gold_bin,
                "predicted_label": pred,
                "nli_calls_for_example": len(call_log),
                "evaluated_premises": [
                    {"premise": p, "score": s} for p, _, s in call_log
                ],
                "total_candidate_windows": len(windows),
            }
            if gold_bin == "NS" and pred == "S":
                fps.append(record)
            elif gold_bin == "S" and pred == "NS":
                fns.append(record)

    elapsed = time.perf_counter() - t0
    print_eval_stats(confusion, indexer)
    print(f"\nElapsed: {elapsed:.2f}s | Windows: {total_windows} | NLI calls: {total_calls} ({(1 - total_calls / total_windows) * 100:.2f}% saved)")
    print(f"False Positives: {len(fps)} | False Negatives: {len(fns)}")

    # Diagnostic inspection of the 3 deep-dive examples (indices 8, 67, 104)
    print("\n=== Diagnostic Probabilities for Representative Examples ===")
    for deep_idx in [104, 67, 8]:
        ex = examples[deep_idx]
        windows = checker.get_windows(ex.passages)
        best_e = -1.0
        best_tuple = None
        for _, ctx in windows:
            e_p, n_p, c_p = inspect_three_class(hf_model, tokenizer, ctx, ex.fact)
            if e_p > best_e:
                best_e = e_p
                best_tuple = (ctx, e_p, n_p, c_p)
        print(f"Index {deep_idx} ({ex.fact}):")
        print(f"  Max unpruned window P(ent)={best_tuple[1]:.3f}, P(neu)={best_tuple[2]:.3f}, P(con)={best_tuple[3]:.3f}")
        print(f"  Premise: {best_tuple[0][:220]}")

        # For index 67 (WKNR), also check single-sentence containing WKNR
        if deep_idx == 67:
            for raw_u, ctx in windows:
                if "WKNR" in raw_u and "sportsline" not in raw_u.lower() and raw_u.count(".") <= 1:
                    e_p, n_p, c_p = inspect_three_class(hf_model, tokenizer, ctx, ex.fact)
                    print(f"  Single-sentence WKNR P(ent)={e_p:.3f}, P(neu)={n_p:.3f}, P(con)={c_p:.3f} | {ctx}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "errors_entailment.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "elapsed_s": round(elapsed, 2),
                "total_windows": total_windows,
                "total_nli_calls": total_calls,
                "confusion_matrix": confusion,
                "false_positives": fps,
                "false_negatives": fns,
            },
            fh,
            indent=2,
        )
    print(f"Saved detailed error dump to {out_json}")

    if SHEET_PATH.exists():
        print("\n=== Verifying error_analysis_sheet.csv ===")
        by_idx = {r["example_index"]: r for r in fps + fns}
        cat_counts = Counter()
        fp_counts = Counter()
        fn_counts = Counter()
        with open(SHEET_PATH, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                eidx = int(row["example_index"])
                assert eidx in by_idx, f"Example {eidx} in sheet is NOT an error of factcheck.py!"
                actual = by_idx[eidx]
                assert actual["binary_gold_label"] == row["binary_gold_label"], f"Gold mismatch at {eidx}"
                assert actual["predicted_label"] == row["predicted_label"], f"Pred mismatch at {eidx}"
                assert actual["fact"] == row["fact_text"], f"Fact text mismatch at {eidx}"
                cat = row["fine_grained_category"]
                cat_counts[cat] += 1
                if row["broad_error_type"] == "False Positive":
                    fp_counts[cat] += 1
                else:
                    fn_counts[cat] += 1
        print("Verified all rows in error_analysis_sheet.csv match current factcheck.py errors!")
        for cat, cnt in cat_counts.most_common():
            print(f"  {cat}: FP={fp_counts[cat]}, FN={fn_counts[cat]}, Total={cnt} ({cnt/sum(cat_counts.values())*100:.1f}%)")


if __name__ == "__main__":
    main()
