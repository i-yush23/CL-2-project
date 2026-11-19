## data_summary

| split | category | item | count |
| --- | --- | --- | --- |
| train | overall | tweet_count | 14000 |
| train | overall | token_count | 365560 |
| train | sentiment | negative | 4102 |
| train | sentiment | neutral | 5264 |
| train | sentiment | positive | 4634 |
| train | language_tag | EMT | 520 |
| train | language_tag | Eng | 121412 |
| train | language_tag | Hin | 169893 |
| train | language_tag | O | 73735 |
| dev | overall | tweet_count | 3000 |
| dev | overall | token_count | 78678 |
| dev | sentiment | negative | 890 |
| dev | sentiment | neutral | 1128 |
| dev | sentiment | positive | 982 |
| dev | language_tag | EMT | 100 |
| dev | language_tag | Eng | 25615 |
| dev | language_tag | Hin | 37004 |
| dev | language_tag | O | 15959 |

## normalisation_stats

| metric | value |
| --- | --- |
| entries | 1031 |
| distinct_canonicals | 371 |
| entries_learned | 993 |
| entries_cl1_variants | 38 |
| entries_cl1_slang | 0 |
| train_tokens_total | 365560 |
| train_tokens_changed | 7895 |
| dev_tokens_total | 78678 |
| dev_tokens_changed | 1633 |
| cl1_rejection_cl1_slang_already_learned | 1 |
| cl1_rejection_cl1_slang_explicit_drop | 6 |
| cl1_rejection_cl1_slang_src_eng_majority | 11 |
| cl1_rejection_cl1_slang_src_too_short | 7 |
| cl1_rejection_cl1_slang_tgt_fails_hin_guard | 11 |
| cl1_rejection_cl1_slang_tgt_not_single_alpha | 7 |
| cl1_rejection_cl1_slang_tgt_too_short | 1 |
| cl1_rejection_cl1_variants_already_learned | 11 |
| cl1_rejection_cl1_variants_explicit_drop | 4 |
| cl1_rejection_cl1_variants_src_eng_majority | 8 |
| cl1_rejection_cl1_variants_src_too_short | 8 |
| cl1_rejection_cl1_variants_tgt_fails_hin_guard | 3 |
| cl1_rejection_cl1_variants_tgt_too_short | 1 |

## lid_comparison

| condition | accuracy | macro_f1 | hin_eng_acc |
| --- | --- | --- | --- |
| raw | 0.8627 | 0.8813 | 0.8275 |
| cleaned | 0.8626 | 0.8812 | 0.8273 |
| normalised | 0.8623 | 0.8810 | 0.8270 |

## lid_rule_stats

| condition | rule_tokens | gold_O | gold_Hin | gold_Eng |
| --- | --- | --- | --- | --- |
| raw | 15954 | 15954 | 0 | 0 |
| cleaned | 15954 | 15954 | 0 | 0 |
| normalised | 15954 | 15954 | 0 | 0 |

## lid_error_summary

| pair | kind | gold | count | change_type |
| --- | --- | --- | --- | --- |
| raw_vs_norm | helped | Hin | 22 | all |
| raw_vs_norm | helped | Eng | 41 | all |
| raw_vs_norm | helped | O | 0 | all |
| raw_vs_norm | helped | all | 63 | all |
| raw_vs_norm | hurt | Hin | 72 | all |
| raw_vs_norm | hurt | Eng | 24 | all |
| raw_vs_norm | hurt | O | 0 | all |
| raw_vs_norm | hurt | all | 96 | all |
| raw_vs_norm | net | all | -33 | all |
| raw_vs_norm | helped | Hin | 14 | none |
| raw_vs_norm | helped | Eng | 25 | none |
| raw_vs_norm | helped | O | 0 | none |
| raw_vs_norm | helped | all | 39 | none |
| raw_vs_norm | hurt | Hin | 49 | none |
| raw_vs_norm | hurt | Eng | 19 | none |
| raw_vs_norm | hurt | O | 0 | none |
| raw_vs_norm | hurt | all | 68 | none |
| raw_vs_norm | net | Hin | -35 | none |
| raw_vs_norm | net | Eng | 6 | none |
| raw_vs_norm | net | O | 0 | none |
| raw_vs_norm | net | all | -29 | none |
| raw_vs_norm | helped | Hin | 6 | case_only |
| raw_vs_norm | helped | Eng | 13 | case_only |
| raw_vs_norm | helped | O | 0 | case_only |
| raw_vs_norm | helped | all | 19 | case_only |
| raw_vs_norm | hurt | Hin | 7 | case_only |
| raw_vs_norm | hurt | Eng | 5 | case_only |
| raw_vs_norm | hurt | O | 0 | case_only |
| raw_vs_norm | hurt | all | 12 | case_only |
| raw_vs_norm | net | Hin | -1 | case_only |
| raw_vs_norm | net | Eng | 8 | case_only |
| raw_vs_norm | net | O | 0 | case_only |
| raw_vs_norm | net | all | 7 | case_only |
| raw_vs_norm | helped | Hin | 1 | cleaning |
| raw_vs_norm | helped | Eng | 3 | cleaning |
| raw_vs_norm | helped | O | 0 | cleaning |
| raw_vs_norm | helped | all | 4 | cleaning |
| raw_vs_norm | hurt | Hin | 15 | cleaning |
| raw_vs_norm | hurt | Eng | 0 | cleaning |
| raw_vs_norm | hurt | O | 0 | cleaning |
| raw_vs_norm | hurt | all | 15 | cleaning |
| raw_vs_norm | net | Hin | -14 | cleaning |
| raw_vs_norm | net | Eng | 3 | cleaning |
| raw_vs_norm | net | O | 0 | cleaning |
| raw_vs_norm | net | all | -11 | cleaning |
| raw_vs_norm | helped | Hin | 1 | dictionary |
| raw_vs_norm | helped | Eng | 0 | dictionary |
| raw_vs_norm | helped | O | 0 | dictionary |
| raw_vs_norm | helped | all | 1 | dictionary |
| raw_vs_norm | hurt | Hin | 1 | dictionary |
| raw_vs_norm | hurt | Eng | 0 | dictionary |
| raw_vs_norm | hurt | O | 0 | dictionary |
| raw_vs_norm | hurt | all | 1 | dictionary |
| raw_vs_norm | net | Hin | 0 | dictionary |
| raw_vs_norm | net | Eng | 0 | dictionary |
| raw_vs_norm | net | O | 0 | dictionary |
| raw_vs_norm | net | all | 0 | dictionary |
| clean_vs_norm | helped | Hin | 20 | all |
| clean_vs_norm | helped | Eng | 36 | all |
| clean_vs_norm | helped | O | 0 | all |
| clean_vs_norm | helped | all | 56 | all |
| clean_vs_norm | hurt | Hin | 57 | all |
| clean_vs_norm | hurt | Eng | 22 | all |
| clean_vs_norm | hurt | O | 0 | all |
| clean_vs_norm | hurt | all | 79 | all |
| clean_vs_norm | net | all | -23 | all |
| clean_vs_norm | helped | Hin | 13 | none |
| clean_vs_norm | helped | Eng | 25 | none |
| clean_vs_norm | helped | O | 0 | none |
| clean_vs_norm | helped | all | 38 | none |
| clean_vs_norm | hurt | Hin | 46 | none |
| clean_vs_norm | hurt | Eng | 18 | none |
| clean_vs_norm | hurt | O | 0 | none |
| clean_vs_norm | hurt | all | 64 | none |
| clean_vs_norm | net | Hin | -33 | none |
| clean_vs_norm | net | Eng | 7 | none |
| clean_vs_norm | net | O | 0 | none |
| clean_vs_norm | net | all | -26 | none |
| clean_vs_norm | helped | Hin | 6 | case_only |
| clean_vs_norm | helped | Eng | 11 | case_only |
| clean_vs_norm | helped | O | 0 | case_only |
| clean_vs_norm | helped | all | 17 | case_only |
| clean_vs_norm | hurt | Hin | 8 | case_only |
| clean_vs_norm | hurt | Eng | 4 | case_only |
| clean_vs_norm | hurt | O | 0 | case_only |
| clean_vs_norm | hurt | all | 12 | case_only |
| clean_vs_norm | net | Hin | -2 | case_only |
| clean_vs_norm | net | Eng | 7 | case_only |
| clean_vs_norm | net | O | 0 | case_only |
| clean_vs_norm | net | all | 5 | case_only |
| clean_vs_norm | helped | Hin | 0 | cleaning |
| clean_vs_norm | helped | Eng | 0 | cleaning |
| clean_vs_norm | helped | O | 0 | cleaning |
| clean_vs_norm | helped | all | 0 | cleaning |
| clean_vs_norm | hurt | Hin | 2 | cleaning |
| clean_vs_norm | hurt | Eng | 0 | cleaning |
| clean_vs_norm | hurt | O | 0 | cleaning |
| clean_vs_norm | hurt | all | 2 | cleaning |
| clean_vs_norm | net | Hin | -2 | cleaning |
| clean_vs_norm | net | Eng | 0 | cleaning |
| clean_vs_norm | net | O | 0 | cleaning |
| clean_vs_norm | net | all | -2 | cleaning |
| clean_vs_norm | helped | Hin | 1 | dictionary |
| clean_vs_norm | helped | Eng | 0 | dictionary |
| clean_vs_norm | helped | O | 0 | dictionary |
| clean_vs_norm | helped | all | 1 | dictionary |
| clean_vs_norm | hurt | Hin | 1 | dictionary |
| clean_vs_norm | hurt | Eng | 0 | dictionary |
| clean_vs_norm | hurt | O | 0 | dictionary |
| clean_vs_norm | hurt | all | 1 | dictionary |
| clean_vs_norm | net | Hin | 0 | dictionary |
| clean_vs_norm | net | Eng | 0 | dictionary |
| clean_vs_norm | net | O | 0 | dictionary |
| clean_vs_norm | net | all | 0 | dictionary |

## baseline_ablation

| config | groups | best_C | dev_accuracy | dev_macro_f1 | macro_f1_ci_low | macro_f1_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | F1 | 1.0000 | 0.6067 | 0.6053 | 0.5894 | 0.6226 |
| R1 | F1+F2 | 1.0000 | 0.6167 | 0.6184 | 0.6019 | 0.6343 |
| R2 | F1+F2+F3 | 0.3000 | 0.6143 | 0.6154 | 0.5990 | 0.6319 |
| R3 | F1+F2+F3+F4 | 1.0000 | 0.6147 | 0.6173 | 0.6001 | 0.6331 |
| R4 | F1+F2+F3+F4+F5 | 1.0000 | 0.6123 | 0.6150 | 0.5979 | 0.6305 |
| R1_raw | F1+F2 | 0.3000 | 0.6170 | 0.6173 | 0.6005 | 0.6337 |
| R4_gold | F1+F2+F3+F4+F5 | 1.0000 | 0.6130 | 0.6160 | 0.5997 | 0.6321 |
| R4_insample | F1+F2+F3+F4+F5 | 1.0000 | 0.6147 | 0.6173 | 0.6006 | 0.6327 |

## baseline_deltas

| pair | mean_delta_macro_f1 | ci_low | ci_high |
| --- | --- | --- | --- |
| R1-R0 | 0.0128 | 0.0011 | 0.0241 |
| R2-R1 | -0.0029 | -0.0113 | 0.0059 |
| R3-R2 | 0.0019 | -0.0074 | 0.0109 |
| R4-R3 | -0.0023 | -0.0061 | 0.0013 |
| R1-R1_raw | 0.0012 | -0.0093 | 0.0114 |

## best_per_class

| class | precision | recall | f1_score | support |
| --- | --- | --- | --- | --- |
| negative | 0.5981 | 0.7090 | 0.6488 | 890 |
| neutral | 0.5741 | 0.4840 | 0.5253 | 1128 |
| positive | 0.6771 | 0.6853 | 0.6812 | 982 |

## before_after

| metric | v1 | now |
| --- | --- | --- |
| variant_entries | 5300 | 1031 |
| lid_accuracy_raw | 0.8348 | 0.8627 |
| lid_macro_f1_raw | 0.8406 | 0.8813 |
| lid_accuracy_cleaned | 0.8347 | 0.8626 |
| lid_macro_f1_cleaned | 0.8405 | 0.8812 |
| lid_accuracy_normalised | 0.8332 | 0.8623 |
| lid_macro_f1_normalised | 0.8394 | 0.8810 |
| dev_macro_f1_R0 | 0.6057 | 0.6053 |
| dev_macro_f1_R1 | 0.6192 | 0.6184 |
| dev_macro_f1_R2 | 0.6170 | 0.6154 |
| dev_macro_f1_R3 | 0.6181 | 0.6173 |
| dev_macro_f1_R4 | 0.6168 | 0.6150 |
| dev_macro_f1_R1_raw | 0.6173 | 0.6173 |
