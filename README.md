# CM3604 - Deep Learning Coursework

Group coursework for CM3604 Deep Learning (Robert Gordon University).

## Group Members
- Thushanth Mahendran
- Kavindu Sudasinghe
- Yasith Kirihena

## Setup

Make sure Python 3.10 is installed, then install the required packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `dev_labeled_ChatGPT.jsonl` and `passages_bm25_ChatGPT_humfacts.jsonl` inside the `data/` folder before running.

## Running the Code

Part 1 (Word Overlap):
```bash
python factchecking_main.py --mode word_overlap
```

Part 2 (Textual Entailment):
```bash
python factchecking_main.py --mode entailment
```

## Experiments & Report
- `experiments/ablation_study.py`: Part 1 preprocessing/metric ablations and 5-fold grouped cross-validation (`--with-part2` includes DeBERTa ablations).
- `experiments/error_analysis.py`: Exports Part 3 False Positives/Negatives and validates `experiments/error_analysis_sheet.csv`.
- `report/CM3604_Coursework_Report.pdf`: Written report (compiled from `report/report.md` via `scripts/render_report_pdf.py`).

## Results
- Majority baseline (`always_entail`): 112/221 (50.68%)
- Part 1 (`word_overlap`): 186/221 (84.16%)
- Part 2 (`entailment`): 195/221 (88.24%)
