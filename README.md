# Hinglish SentiMix Sentiment Analysis

- Code-mixed Hindi-English sentiment classification and token-level language identification on Hinglish SentiMix tweets.
- Download the data archive Semeval_2020_task9_data.zip (md5: 4d93553c213b3a966b6d55b7a7c03be7) from Zenodo record 3974927 and sparse-clone CL-1 (hinglish/ only) into external/.
- Pipeline execution order:
  - python src/normalise.py
  - python src/run_lid_experiments.py
  - python src/run_lid_error_analysis.py
  - python src/run_baseline.py
  - python src/make_report.py
  - python tests/check_normalise.py
- Smoke test runs with --smoke write only to results_smoke/.
