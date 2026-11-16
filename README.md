# CL2 SentiMix Project — Mid-Submission

Sentiment analysis of Hinglish tweets (SemEval-2020 Task 9, Hinglish subset).  
Two deliverables: **A** (Normalisation + LID), **B** (LR sentiment baseline).

## Requirements

```
python >= 3.11
pip install -r requirements.txt
```

## Setup (one-time)

```bash
# 1. Sparse-clone CL-1 reference code
mkdir -p external && cd external
git clone --depth 1 --filter=blob:none --sparse https://github.com/i-yush23/CL-1-project.git
cd CL-1-project && git sparse-checkout set hinglish && cd ../..

# 2. Download and unpack SentiMix data
mkdir -p data/sentimix
curl -L -o data/sentimix/Semeval_2020_task9_data.zip \
  "https://zenodo.org/api/records/3974927/files/Semeval_2020_task9_data.zip/content"
unzip -q data/sentimix/Semeval_2020_task9_data.zip -d data/sentimix/
```

## Running (in order)

```bash
# Data summary (creates results/data_summary.csv)
PYTHONPATH=. .venv/bin/python src/data_io.py

# Build variant dictionary + normalisation stats
PYTHONPATH=. .venv/bin/python src/normalise.py

# Deliverable A — LID experiments (conditions: raw / cleaned / normalised)
PYTHONPATH=. .venv/bin/python src/run_lid_experiments.py

# LID error analysis (raw vs normalised)
PYTHONPATH=. .venv/bin/python src/run_lid_error_analysis.py

# Deliverable B — LR sentiment baseline ablation
PYTHONPATH=. .venv/bin/python src/run_baseline.py
```

Add `--smoke` to any script for a 500-tweet smoke test.

## Results

| Task | Metric | Value |
|------|--------|-------|
| LID raw | Accuracy / Macro-F1 | 0.835 / 0.841 |
| LID cleaned | Accuracy / Macro-F1 | 0.835 / 0.841 |
| LID normalised | Accuracy / Macro-F1 | 0.833 / 0.839 |
| Baseline R1 (best) | Dev Accuracy / Macro-F1 | 0.618 / 0.619 |

All result tables and confusion matrix plots are in `results/`.
