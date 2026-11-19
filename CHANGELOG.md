# Changelog

| Issue | Fix | Files | Commit |
|---|---|---|---|
| Task 0 Baseline | Tagged v1-reviewed and archived v1 results | CHANGELOG.md, results_v1/ | eb1357d |
| Task 1 (I1, I6, I7) | Guarded variant dictionary (Hin/Eng share, min len 3, difflib >= 0.7, cycle-free chains, CL-1 rejections), ASCII-only elongation collapse, 1-to-1 assert, stats with 20 seeded-random groups, tests/check_normalise.py | src/normalise.py, src/run_lid_experiments.py, tests/check_normalise.py, results/variant_dict.json, results/normalisation_stats.csv, results/normalisation_stats_examples.csv | 813927b |
| Task 1 Steps 1-5 & 7 | Unpack (variant_dict, rejections) across all callers, keyed cache paths in all scripts, NEVER_SOURCE guard, entries by source in stats, check_normalise test suite with forbidden keys and space checks | src/normalise.py, src/run_lid_experiments.py, src/run_baseline.py, src/run_lid_error_analysis.py, tests/check_normalise.py, results/normalisation_stats.csv, results/variant_dict.json | a29b29d |
| Task 1 Step 6 | Refactor build_variant_dictionary into small helpers (<40 lines: _group_by_phonetic_hash, _learn_variants, _add_cl1_entries) with verified identical output | src/normalise.py | 5d43aeb |
| Task 2 (I4, I5) | LID cleanup (remove phash_label), O-rule (_is_language_independent), dev rule stats (results/lid_rule_stats.csv), smoke isolation to results_smoke/ | src/lid.py, src/run_lid_experiments.py, results/lid_rule_stats.csv, results/lid_* | pending |
