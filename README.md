# CM3604 — Deep Learning Coursework: Fact-Checking LLM Outputs

Group coursework for **CM3604 (Deep Learning)**, Academic Year 2026/2027 (Robert Gordon University).  
This repository implements automated fact-checking models to verify atomic claims extracted from ChatGPT-generated biographies against BM25-retrieved Wikipedia passages, based on the [FActScore](https://arxiv.org/abs/2305.14251) evaluation pipeline.

## Group Members
- **Thushanth Mahendran**
- **Kavindu Sudasinghe**
- **Yasith**

---

## Project Structure

```text
CM3604_CW/
├── data/
│   ├── dev_labeled_ChatGPT.jsonl            # 221 labeled atomic facts (31 biographies)
│   └── passages_bm25_ChatGPT_humfacts.jsonl # Retrieved Wikipedia passages per fact
├── factcheck.py                             # Part 1 & Part 2 fact-checking implementations
├── factchecking_main.py                     # Coursework evaluation driver (unmodified)
├── requirements.txt                         # Python 3.10 package dependencies
└── README.md
```
*(Note: The `.jsonl` dataset files inside `data/` are excluded via `.gitignore` due to file size; place both files from the Moodle `Framework Code` archive into `data/` before running).*

---

## Setup & Installation

Ensure **Python 3.10** is installed, then create a virtual environment and install the required packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Models

### 1. Baselines
```bash
python factchecking_main.py --mode random
python factchecking_main.py --mode always_entail
```

### 2. Part 1: Word Overlap (`WordRecallThresholdFactChecker`)
 Cleans Wikipedia markup (`<s>`, `</s>`, empty quotes), removes stop words, stems tokens with `PorterStemmer`, filters biography subject name tokens, and computes sentence-level unigram recall (`threshold = 0.50`):

```bash
python factchecking_main.py --mode word_overlap
```

### 3. Part 2: Textual Entailment (`EntailmentModel` & `EntailmentFactChecker`)
Uses `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` with 1- and 2-sentence context windows, entity-title grounding, and word-overlap candidate pruning (`prune_threshold = 0.20`, `ent_threshold = 0.38`):

```bash
python factchecking_main.py --mode entailment
```
*(Add `--cuda` only if running on a machine with a CUDA-enabled NVIDIA GPU).*

---

## Results Summary (`dev_labeled_ChatGPT.jsonl` — 221 Facts)

| Mode / Method | Accuracy | Target | Precision (`S`) | Recall (`S`) | F1 (`S`) | F1 (`NS`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `always_entail` (Majority Baseline) | `50.68%` (112/221) | — | 0.5068 | 1.0000 | 0.6727 | — |
| **Part 1:** `word_overlap` | **`84.16%`** (186/221) | `>= 75%` | 0.8235 | 0.8750 | 0.8485 | 0.8341 |
| **Part 2:** `entailment` (DeBERTa-v3 + Pruning) | **`88.24%`** (195/221) | `>= 83%` | 0.9300 | 0.8304 | 0.8774 | 0.8870 |
