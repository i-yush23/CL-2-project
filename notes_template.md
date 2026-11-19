# NOTES: CL-2 mid-submission (Hinglish SentiMix)

## Scope
Deliverable A (normalisation and token-level LID) and Deliverable B (feature-based logistic-regression sentiment baseline), Hinglish train and dev only. The SentiMix test split was never loaded, evaluated or inspected.

## Data
Gold tags are kept as in the data. EMT (emoji) is mapped to O; the LID label set is Hin, Eng, O.

{{table:data_summary}}

## CL-1 reuse
Used only through src/cl1.py: {{cl1_files_used}}. CL-1's own LID model was not reused because its label set and CSV interface do not match the SentiMix token tags; a small own character n-gram Naive Bayes LID was trained on the SentiMix gold tags instead (documented fallback).

## Normalisation
clean_token lowercases, maps URLs to a placeholder, maps mentions to a placeholder, strips the hashtag sign, collapses elongated ASCII letters, and leaves emoji and digits untouched. The variant dictionary is learned from train only: forms are grouped by CL-1 phonetic hash, and a form may be merged only if its Hindi share is at least {{HIN_SHARE_MIN}} and its English share is at most {{ENG_SHARE_MAX}}, its similarity to the canonical form is at least {{SIMILARITY_MIN}}, and the canonical form is at least {{MIN_FREQ_RATIO}} times as frequent as the variant. CL-1 entries are added only if they pass the same guards. Normalisation is strictly one-to-one with the input tokens.

The dictionary has {{entries}} entries ({{entries_learned}} learned from train, {{entries_cl1_variants}} from CL-1 variants, {{entries_cl1_slang}} from CL-1 slang) over {{distinct_canonicals}} canonical forms. It rewrites {{dev_tokens_changed}} of {{dev_tokens_total}} dev tokens ({{dev_changed_pct}} percent).

{{table:normalisation_stats}}

## LID
{{table:lid_comparison}}

Headline accuracy includes O, which is {{o_share_pct}} percent of dev tokens and is classified perfectly (precision {{o_prec}}, recall {{o_rec}}). Accuracy on Hin and Eng tokens only is {{hin_eng_acc_raw}} (raw) and {{hin_eng_acc_norm}} (normalised).

O-rule: tokens with no letter (punctuation, digits, emoji), URLs and mentions are routed directly to O; this covers {{rule_tokens}} dev tokens, all of them gold O. The remaining {{gap_tokens}} gold-O dev tokens are written in non-Latin scripts or accented characters (see results/lid_rule_gap.csv); the Naive Bayes learns them as O because those characters occur only in O-tagged training tokens.

{{table:lid_rule_stats}}

### LID error analysis
Comparing the cleaned model with the normalised model on dev, {{cn_helped}} tokens went from wrong to right and {{cn_hurt}} from right to wrong (net {{cn_net}}; exact McNemar p = {{cn_p}}). Tokens the dictionary rewrote flipped {{cn_dict_helped}} helped and {{cn_dict_hurt}} hurt; tokens left unchanged flipped {{cn_none_helped}} helped and {{cn_none_hurt}} hurt (net {{cn_none_net}}, p = {{cn_none_p}}), so the difference comes from retraining on the normalised training data, not from rewriting dev tokens. By gold tag the net change is {{cn_hin_net}} for Hindi and {{cn_eng_net}} for English. Cleaning-only changes net {{cn_clean_net}} and case-only changes net {{cn_case_net}}. Against the raw model the net is {{rn_net}} (p = {{rn_p}}). We make no claim that normalisation helps or hurts LID: the per-token effect of the dictionary is negligible and the small accuracy drop is a training-set effect. A mechanism (fewer spelling variants seen in training) is a hypothesis we did not test.

{{table:lid_error_summary}}

## LR baseline
Features: F1 word n-grams (TF-IDF), F2 character n-grams (TF-IDF), F3 language-prefixed unigrams, F4 language-mixing numbers (language fractions, switches, CMI), F5 emoji polarity counts and a negation count. Vectorisers and scalers are fit on train only. Train language tags are 5-fold out-of-fold LID predictions (token accuracy {{oof_acc}}); dev tags come from the LID trained on all of train (token accuracy {{dev_lid_acc}}). R4_gold uses gold tags (upper bound); R4_insample uses in-sample train tags (diagnostic).

{{table:baseline_ablation}}

{{table:baseline_deltas}}

Only {{significant_steps}} has a 95% bootstrap interval that excludes zero; the intervals for {{null_steps}} include zero, so we report no measurable gain from those steps. Tag quality: R4 {{r4_f1}}, R4_gold {{r4g_f1}}, R4_insample {{r4i_f1}}; the differences are smaller than the width of the R4 interval ({{r4_ci_width}}), so tag quality and the in-sample versus out-of-fold mismatch have no measurable effect here, and F3 to F5 add nothing for tags to act on.

Best configuration by dev macro-F1: {{best_config}} (macro-F1 {{best_f1}}, accuracy {{best_acc}}). The weakest class is {{weak_class}} (F1 {{weak_f1}}).

{{table:best_per_class}}

Caveat: C was tuned on dev macro-F1, so dev scores are slightly optimistic, and the bootstrap intervals reflect dev-sample noise for the chosen C and do not include the effect of the selection. No test-set evaluation was done.

## Before and after fixes
{{table:before_after}}

## Open problems
- Residual false merges in the variant dictionary: the blocklist (NEVER_SOURCE, BLOCKED_MERGES) was hand-curated after inspecting the train-built dictionary; no manual error-rate audit was done.
- The variant-dictionary origin map is stored as a function attribute in normalise.py.
- The LID cache key does not cover LID hyperparameters.
- The LID has no word-level memory and the gold tags are noisy, which limits Hin/Eng accuracy.
- The negation feature is a token count, not a binary flag.
- Conditions cleaned and normalised are near-identical because the dictionary rewrites few tokens.
- The transformer, the full corpus study and the theory analysis are out of scope for this slice.
