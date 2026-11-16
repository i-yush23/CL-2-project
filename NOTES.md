# Project Notes & Log

## Phase 0: Setup & Data Inspection
- Created workspace layout, sparse-cloned `external/CL-1-project` with `hinglish/` only (`git sparse-checkout list` confirmed).
- Downloaded `Semeval_2020_task9_data.zip` from Zenodo (MD5: 4d93553c213b3a966b6d55b7a7c03be7), unpacked Hinglish train and dev sets.
- Distinct sentiment labels: `['negative', 'neutral', 'positive']`. Distinct language tags: `['EMT', 'Eng', 'Hin', 'O']` (where `EMT` is emoticon/emoji).
- CoNLL format parsed with `rstrip("\r\n").split("\t")` yielding 0 malformed lines.

## Phase 1: CL-1 Code Analysis & Adapter
- (a) Normalisation: `get_phonetic_hash(word: str) -> str` (`features/phonetic_matcher.py`) converts words into uppercase phonetic skeletons by compressing repeats, mapping phonemes, and dropping non-initial vowels; `load_transliteration_variants()` (77 pairs) and `load_slang_map()` (44 pairs) map noisy/slang forms to canonical words.
- (b) LID: `LIDModel` (`statistical_models/lid_model.py`) trains on CSV (`token,label`) via `train(data_path: str)` and infers via `predict(tokens: list[str]) -> list[str]` using unigram + phonetic hash log-priors with context smoothing window. Since it hardcodes `UNI` and expects CSV, we follow Sec. 5 to implement a tailored token-level NB in `src/lid.py`.
- (c) CMI: `CmiCalculator.calculate(text: str) -> dict` (`metrics/cmi_calculator.py`) evaluates mixing percentage. In Phase 6, we use the specified formula `100 * (1 - max_lang / (n - u))` ignoring `O` and `EMT`.
- (d) Emoji polarity: `load_emoji_lexicon() -> dict[str, list[str]]` (`features/emoji_analyzer.py`) provides 3 categories: POSITIVE (73 emoji), NEGATIVE (53 emoji), NEUTRAL (42 emoji). `extract_emojis(text: str) -> list[str]` extracts emoji characters via regex.
- Created `src/cl1.py` adapter; verified successful imports and data loading.

## Phase 2: Data I/O & Evaluation Utilities
- Implemented `src/data_io.py` (`parse_conll_file`, `load_dataset`, `summarize_split`, `run_data_summary`).
- Train set: 14,000 tweets, 365,560 tokens (Eng: 121,412, Hin: 169,893, O: 73,735, EMT: 520; neg: 4,102, neu: 5,264, pos: 4,634).
- Dev set: 3,000 tweets, 78,678 tokens (Eng: 25,615, Hin: 37,004, O: 15,959, EMT: 100; neg: 890, neu: 1,128, pos: 982).
- Implemented `src/eval_utils.py` for accuracy, macro-F1, per-class PRF1 tables, and confusion matrix plotting to PNG.
- Saved summary to `results/data_summary.csv`.

## Phase 3: Normaliser Design
- Implemented `src/normalise.py` with `clean_token`, `build_variant_dictionary`, `normalise_tokens`, and stats helpers.
- Variant dictionary built from train only: 5,300 entries (including ~4,900 learned + 77 transliteration + 44 slang from CL-1 where not conflicting).
- Train changed: 70,484 tokens of 365,560 (~19.3%). Dev changed: 14,850 of 78,678 (~18.9%).
- Groups require: canonical must appear ≥3 times; variant must appear ≥2 times; majority gold tag = 'Hin'.
- Open issue: short-word collisions (jo/ji, nana/noone) due to coarse phonetic hash — acceptable trade-off for Hinglish.

## Phase 4: LID Training & Evaluation
- Implemented `src/lid.py` (char n-gram NB, 1-4 grams, Laplace alpha=1, context window=2, weight=0.1). Wrote own NB (not retraining CL-1 LIDModel) because CL-1 LIDModel expects a CSV path and hardcodes UNI label; documented here.
- Label mapping: Hin->Hin, Eng->Eng, O->O, EMT->O (emoticons are language-neutral).
- Full dev results: raw acc=0.835 F1=0.841; cleaned acc=0.835 F1=0.841; normalised acc=0.833 F1=0.839.
- Cleaning alone has near-zero effect; normalisation slightly hurts (−0.0012 macro-F1).

## Phase 5: LID Error Analysis
- Helped (wrong→right): 693 tokens; Hurt (right→wrong): 820 tokens — net slight negative.
- Helped examples: 'g'->'gaya' (Hin), 'u'->'you' (Eng), 'mtlb'->'matlab' (Hin); normalisation correctly redirected abbreviated spellings.
- Hurt examples: 'U'->'you' classified Eng but gold is Hin (pronoun in Hindi context); 'luv'->'love' reclassified as Eng but gold was Hin; collider words where canonical maps to wrong language.
- Conclusion: normalisation helps recall of Hin abbreviations but hurts Hin items merged into an Eng canonical via slang_map (u/ur/r → English words), and over-merges some Hinglish forms to canonical forms seen more in English training context.

## Phase 6: Feature Design (src/features.py)
- F1: TF-IDF word 1-2-grams (min_df=2, sublinear_tf). F2: TF-IDF char 2-5-grams (char_wb). F3: language-prefixed unigrams. F4: Hin/Eng/O fractions + switch count + CMI (MaxAbsScaler). F5: emoji polarity counts + negation flag (window=3).
- CMI formula used: 100 * (1 - max_lang / (n - u)), u = O + EMT tokens.
- Negation word list: {not, no, never, nahi, nahin, mat, na, naa, ...}.

## Phase 7: Baseline Results (src/run_baseline.py)
- Incremental ablation on dev set (3,000 tweets), LR(C tuned over {0.1,0.3,1,3,10}, max_iter=2000, class_weight=balanced):

| Config | Groups         | Best C | Dev Acc | Dev Macro-F1 |
|--------|---------------|--------|---------|--------------|
| R0     | F1            | 1.0    | 0.606   | 0.606        |
| R1     | F1+F2         | 1.0    | 0.618   | 0.619        |
| R2     | F1+F2+F3      | 1.0    | 0.615   | 0.617        |
| R3     | F1+F2+F3+F4   | 0.3    | 0.617   | 0.618        |
| R4     | F1+F2+F3+F4+F5| 0.3    | 0.616   | 0.617        |
| R1_raw | F1+F2 (raw)   | 0.3    | 0.617   | 0.617        |

- Best config: R1 (F1+F2). Adding char n-grams over word n-grams gives +1.4pp macro-F1. F3 (language-prefixed) and F4 (mixing features) add marginal gains. F5 (emoji+negation) slightly hurts — emoji too sparse.
- Raw vs normalised (R1_raw vs R1): negligible difference (0.617 vs 0.619) on sentiment; normalisation helps LID more than it helps downstream sentiment.
- Best config per-class: neg P/R/F1=0.609/0.717/0.658; neu=0.576/0.472/0.519; pos=0.665/0.697/0.681.
- Neutral class is hardest (lowest F1 0.519) — typical for 3-class sentiment on code-mixed data.

## Phase 8: Final Summary
- CL-1 files reused: `hinglish/features/phonetic_matcher.py` (get_phonetic_hash), `hinglish/features/emoji_analyzer.py` (emoji lexicon), `hinglish/data/transliteration_variants.json`, `hinglish/data/slang_map.json`, `hinglish/data/emoji_lexicon.json`.
- CL-1 LIDModel NOT reused for retraining (hardcodes CSV format + UNI label); wrote own char n-gram NB in `src/lid.py`.
- Open issues: (1) Phonetic hash too coarse → false merges for short words. (2) slang_map maps Hin forms to English canonicals (u→you) hurting Hin LID recall. (3) Neutral class underperforms — consider adding Hindi-specific sentiment lexicons.
