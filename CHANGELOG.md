# Changelog

| Issue | Fix | Files | Commit |
|---|---|---|---|
| Task 0 Baseline | Tagged v1-reviewed and archived v1 results | CHANGELOG.md, results_v1/ | f395672 |
| I1 (Task 1) | Guarded variant dictionary (Hin/Eng share, min len 3, difflib >= 0.7, 5x freq guard, NEVER_SOURCE blocklist, cycle-free chains, CL-1 rejections), tests/check_normalise.py | src/normalise.py, tests/check_normalise.py, results/variant_dict.json | 0ba03ee, a29b29d, 5d43aeb |
| I2 (Task 3) | 5-fold CV predicted train LID tags, out-of-fold dev tags, R4_gold upper bound, R4_insample diagnostic | src/run_baseline.py, results/lid_cv_accuracy.csv, results/baseline_ablation.csv | 6de93a9 |
| I3 (Task 5) | NOTES data grounding via make_report.py, exact McNemar test, bootstrap CIs, paired deltas, optimistic C-tuning caveat | src/make_report.py, notes_template.md, NOTES.md, results/summary.md, results/derived_stats.csv | 0e9b6fd |
| I4 (Task 2) | Smoke isolation to results_smoke/, cache invalidation via keyed filenames with get_cache_key(smoke) | src/lid.py, src/run_lid_experiments.py, src/run_baseline.py, src/normalise.py | ddd831e |
| I5 (Task 2) | LID cleanup (remove unused phash_label), O-rule (_is_language_independent) routing non-letters, URLs, mentions | src/lid.py, src/run_lid_experiments.py, results/lid_rule_stats.csv | ddd831e |
| I6 (Task 1) | Restrict elongation collapse strictly to ASCII letters [a-zA-Z], leaving digits and emoji untouched | src/normalise.py, tests/check_normalise.py | 0ba03ee |
| I7 (Task 1) | Rename variant_groups to entries, track accepted entries by source (learned, cl1_variants, cl1_slang), 20 seeded random examples | src/normalise.py, results/normalisation_stats.csv, results/normalisation_stats_examples.csv | 0ba03ee, a29b29d |
| I8 (Task 4, 4b) | Ranked LID error analysis with helped/hurt counts per gold tag, change_type breakdown (none, case_only, cleaning, dictionary), dictionary examples CSV | src/run_lid_error_analysis.py, results/lid_error_summary.csv, results/lid_error_examples.csv, results/lid_error_examples_dictionary.csv, results/lid_rule_gap.csv | c8bc0a0, 9ce813c |
| I9 (Task 3) | Functional make_f1..make_f5 API, deduplicated NEGATION_WORDS frozenset, F5 MaxAbsScaler + per-character emoji extraction, cleaned src/cl1.py imports | src/features.py, src/cl1.py, src/run_baseline.py | 6de93a9 |
| I10 (Task 3, 5) | Refactored run_baseline.py into functions <=40 lines (6de93a9); rewritten README.md to <=30 lines plain python with bullet-only execution order | src/run_baseline.py, README.md | 6de93a9, 0e9b6fd |
