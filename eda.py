"""Dataset exploration for the CM3604 FActScore biographical fact-checking data."""

import argparse
import json
from collections import Counter
from statistics import mean, median

from factchecking_main import read_fact_examples, read_passages


def explore_dataset(labels_path: str, passages_path: str) -> None:
    """Print biography counts, fact counts, label distribution, and passage lengths."""
    fact_to_passages = read_passages(passages_path)
    examples = read_fact_examples(labels_path, fact_to_passages)

    total_bios = 0
    annotated_bios = 0
    facts_per_bio = []
    with open(labels_path, "r", encoding="utf-8") as fh:
        for line in fh:
            total_bios += 1
            record = json.loads(line)
            annotations = record.get("annotations")
            if annotations is None:
                continue
            annotated_bios += 1
            bio_facts = sum(
                1
                for sent in annotations
                for fact in (sent.get("human-atomic-facts") or [])
                if fact["text"] in fact_to_passages
            )
            facts_per_bio.append(bio_facts)

    raw_counts = Counter(ex.label for ex in examples)
    binary_counts = Counter("NS" if ex.label == "IR" else ex.label for ex in examples)
    n_facts = len(examples)

    passage_counts = [len(ex.passages) for ex in examples]
    passage_char_lens = [len(p["text"]) for ex in examples for p in ex.passages]
    passage_word_lens = [len(p["text"].split()) for ex in examples for p in ex.passages]
    fact_word_lens = [len(ex.fact.split()) for ex in examples]
    tagged_passages = sum(
        1 for ex in examples for p in ex.passages if "<s>" in p["text"] or "</s>" in p["text"]
    )

    print("=== Dataset Overview ===")
    print(f"Biographies: {annotated_bios} annotated / {total_bios} total")
    print(f"Atomic facts: {n_facts}")
    print(
        f"Facts per bio: min={min(facts_per_bio)}, max={max(facts_per_bio)}, "
        f"mean={mean(facts_per_bio):.2f}"
    )

    print("\n=== Label Distribution ===")
    for label in ("S", "NS", "IR"):
        count = raw_counts[label]
        print(f"Raw {label:2s}: {count:3d} ({count / n_facts * 100:5.2f}%)")
    for label in ("S", "NS"):
        count = binary_counts[label]
        print(f"Binary {label:2s} (IR -> NS): {count:3d} ({count / n_facts * 100:5.2f}%)")

    print("\n=== Passages & Length Statistics ===")
    print(
        f"Passages per fact: total={sum(passage_counts)}, min={min(passage_counts)}, "
        f"max={max(passage_counts)}, mean={mean(passage_counts):.2f}"
    )
    print(f"Passages containing <s>/</s> tags: {tagged_passages}/{len(passage_char_lens)}")
    print(
        f"Passage length (chars): min={min(passage_char_lens)}, max={max(passage_char_lens)}, "
        f"mean={mean(passage_char_lens):.1f}, median={median(passage_char_lens):.1f}"
    )
    print(
        f"Passage length (words): min={min(passage_word_lens)}, max={max(passage_word_lens)}, "
        f"mean={mean(passage_word_lens):.1f}, median={median(passage_word_lens):.1f}"
    )
    print(
        f"Fact length (words): min={min(fact_word_lens)}, max={max(fact_word_lens)}, "
        f"mean={mean(fact_word_lens):.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore CM3604 fact-checking dataset.")
    parser.add_argument("--labels_path", default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument("--passages_path", default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    args = parser.parse_args()
    explore_dataset(args.labels_path, args.passages_path)


if __name__ == "__main__":
    main()
