# CM3604 - Deep Learning Coursework

Automated fact-checking of ChatGPT-generated biographies against Wikipedia passages (FActScore dataset), Robert Gordon University.

## Group Members
- Thushanth Mahendran
- Kavindu Sudasinghe
- Yasith Kirihena

## Setup

Create a Python 3.10 virtual environment and install the required packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Evaluator

```bash
python factchecking_main.py --mode always_entail
python factchecking_main.py --mode word_overlap
python factchecking_main.py --mode entailment
```

Pass `--cuda` with `--mode entailment` if running on a CUDA-enabled GPU.

## Helper Scripts

- `eda.py`: Explores biography/fact counts, S/NS/IR label distribution, and passage length statistics.
- `experiments_part1.py`: Runs Part 1 preprocessing, metric, and threshold ablations with 5-fold grouped cross-validation.
- `tune_and_analyze.py`: Checks Part 2 windowing/threshold/pruning ablations and exports Part 3 FP/FN errors for analysis.

## Results

| Mode | Accuracy | Correct / Total |
| :--- | :---: | :---: |
| `always_entail` | 50.68% | 112 / 221 |
| `word_overlap` | 84.16% | 186 / 221 |
| `entailment` | 88.24% | 195 / 221 |
