"""
DELIVERABLE B: LR sentiment baseline — incremental feature ablation.

Configurations:
  R0:          F1 (word n-grams)
  R1:          F1+F2
  R2:          F1+F2+F3  (CV-predicted LID tags on train, full-train LID tags on dev)
  R3:          F1+F2+F3+F4
  R4:          F1+F2+F3+F4+F5
  R1_raw:      F1+F2 on RAW (no cleaning/normalisation) — shows normalisation effect
  R4_gold:     R4 with gold LID tags on train+dev (upper bound)
  R4_insample: R4 with in-sample LID tags on train (diagnostic)

For each config: LR (max_iter=2000, class_weight=balanced),
tune C in {0.1, 0.3, 1, 3, 10} on dev macro-F1, report best.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

from src.data_io import load_dataset
from src.normalise import clean_token, build_variant_dictionary, normalise_tokens, get_cache_key
from src.lid import NaiveBayesLID, predict_tweets, TAG_MAP, build_flat_token_label_lists
from src.features import F1Builder, F2Builder, F3Builder, F4Builder, F5Builder
from src.eval_utils import (
    compute_metrics,
    get_classification_report_df,
    plot_confusion_matrix,
    save_dataframe_csv,
)

CACHE_DIR = "results/cache"
SENTIMENT_LABELS = ['negative', 'neutral', 'positive']
C_GRID = [0.1, 0.3, 1, 3, 10]
SEED = 42
N_FOLDS = 5
N_BOOT = 1000


# ─── Data helpers ─────────────────────────────────────────────────────────────

def get_labels(tweets: list[dict]) -> list[str]:
    return [t['sentiment'] for t in tweets]


def add_cleaned(tweets: list[dict]) -> None:
    for tweet in tweets:
        tweet['cleaned_tokens'] = [clean_token(t) for t in tweet['tokens']]


def add_normalised(tweets: list[dict], variant_dict: dict) -> None:
    for tweet in tweets:
        tweet['normalised_tokens'] = normalise_tokens(tweet['tokens'], variant_dict)


def add_lid_tags(tweets: list[dict], model: NaiveBayesLID,
                 tokens_key: str = 'normalised_tokens') -> None:
    """Predict LID tags and store as tweet['lid_tags']."""
    preds = predict_tweets(model, tweets, tokens_key)
    for tweet, pred in zip(tweets, preds):
        tweet['lid_tags'] = pred


def add_gold_lid_tags(tweets: list[dict]) -> None:
    """Use gold tags (mapped) as lid_tags for upper-bound run."""
    for tweet in tweets:
        tweet['lid_tags'] = [TAG_MAP.get(t, 'O') for t in tweet['tags']]


# ─── Step 1: CV LID tagging ───────────────────────────────────────────────────

def _cv_fold_preds(train_tweets: list[dict]) -> dict[int, list[str]]:
    """Run 5-fold CV LID; return {tweet_idx: predicted_tags}."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_tags: dict[int, list[str]] = {}
    for train_idx, val_idx in kf.split(range(len(train_tweets))):
        fold_train = [train_tweets[i] for i in train_idx]
        fold_val   = [train_tweets[i] for i in val_idx]
        toks, tags = build_flat_token_label_lists(fold_train, 'normalised_tokens')
        lid = NaiveBayesLID()
        lid.fit(toks, tags)
        preds = predict_tweets(lid, fold_val, 'normalised_tokens')
        for i, pred in zip(val_idx, preds):
            cv_tags[i] = pred
    return cv_tags


def _attach_cv_tags(train_tweets: list[dict], cv_tags: dict[int, list[str]]) -> float:
    """Store OOF tags on tweets; return train OOF token accuracy."""
    all_gold, all_pred = [], []
    for i, tweet in enumerate(train_tweets):
        pred = cv_tags.get(i, ['O'] * len(tweet['tags']))
        tweet['cv_lid_tags'] = pred
        all_gold.extend([TAG_MAP.get(t, 'O') for t in tweet['tags']])
        all_pred.extend(pred)
    return accuracy_score(all_gold, all_pred)


def run_lid_cv(
    train_tweets: list[dict],
    dev_tweets: list[dict],
    lid_model: NaiveBayesLID,
    out_dir: str,
) -> None:
    """
    5-fold CV LID on train (normalised tokens). Tags each tweet's OOF fold.
    Stores result in tweet['cv_lid_tags']. Writes lid_cv_accuracy.csv.
    """
    cv_tags = _cv_fold_preds(train_tweets)
    cv_train_acc = _attach_cv_tags(train_tweets, cv_tags)

    # Dev tags from full-train LID model
    dev_preds = predict_tweets(lid_model, dev_tweets, 'normalised_tokens')
    dev_gold, dev_pred_flat = [], []
    for tweet, pred in zip(dev_tweets, dev_preds):
        tweet['lid_tags'] = pred
        dev_gold.extend([TAG_MAP.get(t, 'O') for t in tweet['tags']])
        dev_pred_flat.extend(pred)
    cv_dev_acc = accuracy_score(dev_gold, dev_pred_flat)

    df = pd.DataFrame([
        {'split': 'train_oof', 'token_accuracy': round(cv_train_acc, 6)},
        {'split': 'dev',       'token_accuracy': round(cv_dev_acc, 6)},
    ])
    save_dataframe_csv(df, os.path.join(out_dir, 'lid_cv_accuracy.csv'))
    print(f"  LID CV train OOF acc={cv_train_acc:.4f}  dev acc={cv_dev_acc:.4f}")


# ─── Feature building ─────────────────────────────────────────────────────────

def _build_text_features(
    train, dev, groups, tokens_key, parts_tr, parts_dv, builders
) -> None:
    """Fit and collect F1, F2 (text n-grams) in-place."""
    if 'F1' in groups:
        b = F1Builder(); b.fit(train, tokens_key)
        parts_tr.append(b.transform(train, tokens_key))
        parts_dv.append(b.transform(dev, tokens_key))
        builders['F1'] = b
    if 'F2' in groups:
        b = F2Builder(); b.fit(train, tokens_key)
        parts_tr.append(b.transform(train, tokens_key))
        parts_dv.append(b.transform(dev, tokens_key))
        builders['F2'] = b


def _build_lang_features(
    train, dev, groups, tokens_key, tags_key, parts_tr, parts_dv, builders
) -> None:
    """Fit and collect F3, F4, F5 (language-aware) in-place."""
    if 'F3' in groups:
        b = F3Builder(); b.fit(train, tokens_key, tags_key)
        parts_tr.append(b.transform(train, tokens_key, tags_key))
        parts_dv.append(b.transform(dev, tokens_key, tags_key))
        builders['F3'] = b
    if 'F4' in groups:
        b = F4Builder(); b.fit(train, tags_key)
        parts_tr.append(b.transform(train, tags_key))
        parts_dv.append(b.transform(dev, tags_key))
        builders['F4'] = b
    if 'F5' in groups:
        b = F5Builder(); b.fit(train, tokens_key)
        parts_tr.append(b.transform(train, tokens_key))
        parts_dv.append(b.transform(dev, tokens_key))
        builders['F5'] = b


def build_features(
    train: list[dict],
    dev: list[dict],
    use_groups: list[str],
    tokens_key: str = 'normalised_tokens',
    tags_key: str = 'lid_tags',
) -> tuple[sp.csr_matrix, sp.csr_matrix, dict]:
    """Fit builders on train, transform train+dev. Returns (X_train, X_dev, builders)."""
    parts_tr, parts_dv, builders = [], [], {}
    _build_text_features(train, dev, use_groups, tokens_key, parts_tr, parts_dv, builders)
    _build_lang_features(train, dev, use_groups, tokens_key, tags_key, parts_tr, parts_dv, builders)
    return sp.hstack(parts_tr).tocsr(), sp.hstack(parts_dv).tocsr(), builders


# ─── Tuning & evaluation ──────────────────────────────────────────────────────

def tune_and_eval(
    X_train: sp.csr_matrix,
    y_train: list[str],
    X_dev: sp.csr_matrix,
    y_dev: list[str],
) -> tuple[float, float, float, object, list[str]]:
    """
    Grid-search over C. Returns (best_C, dev_accuracy, dev_macro_f1, model, y_pred).
    """
    best_c, best_f1, best_model = None, -1.0, None
    for c in C_GRID:
        lr = LogisticRegression(
            C=c, max_iter=2000, class_weight='balanced',
            random_state=SEED, solver='lbfgs',
        )
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_dev)
        m = compute_metrics(y_dev, y_pred, labels=SENTIMENT_LABELS)
        if m['macro_f1'] > best_f1:
            best_f1 = m['macro_f1']
            best_c = c
            best_model = lr

    y_pred_best = best_model.predict(X_dev)
    m = compute_metrics(y_dev, y_pred_best, labels=SENTIMENT_LABELS)
    return best_c, m['accuracy'], m['macro_f1'], best_model, y_pred_best


# ─── Paired bootstrap ─────────────────────────────────────────────────────────

def paired_bootstrap(
    y_true: list[str],
    preds_a: list[str],
    preds_b: list[str],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """
    Paired bootstrap (resample by tweet index) for macro-F1 difference A - B.
    Returns (mean_diff, ci_low_2.5, ci_high_97.5).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    ya = np.array(y_true)
    pa = np.array(preds_a)
    pb = np.array(preds_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fa = f1_score(ya[idx], pa[idx], labels=SENTIMENT_LABELS,
                      average='macro', zero_division=0)
        fb = f1_score(ya[idx], pb[idx], labels=SENTIMENT_LABELS,
                      average='macro', zero_division=0)
        diffs.append(fa - fb)
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def bootstrap_ci(
    y_true: list[str],
    y_pred: list[str],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float]:
    """
    Bootstrap CI (2.5, 97.5 percentiles) for macro-F1 of a single model.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    ya = np.array(y_true)
    pa = np.array(y_pred)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(f1_score(ya[idx], pa[idx], labels=SENTIMENT_LABELS,
                              average='macro', zero_division=0))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ─── Single config runner ─────────────────────────────────────────────────────

def run_config(
    name: str,
    groups: list[str],
    train: list[dict],
    dev: list[dict],
    y_train: list[str],
    y_dev: list[str],
    tokens_key: str = 'normalised_tokens',
    tags_key: str = 'lid_tags',
    smoke_test: bool = False,
) -> dict:
    print(f"\n--- Config: {name} | groups={groups} | tokens={tokens_key} | tags={tags_key} ---")
    tr = train[:500] if smoke_test else train
    y_tr = y_train[:500] if smoke_test else y_train

    X_train, X_dev, _ = build_features(tr, dev, groups, tokens_key, tags_key)
    best_c, acc, f1, model, y_pred = tune_and_eval(X_train, y_tr, X_dev, y_dev)
    ci_low, ci_high = bootstrap_ci(y_dev, y_pred)
    print(f"  Best C={best_c}  Acc={acc:.4f}  Macro-F1={f1:.4f}  CI=[{ci_low:.4f},{ci_high:.4f}]")

    return {
        'config': name,
        'groups': '+'.join(groups),
        'best_C': best_c,
        'dev_accuracy': round(acc, 4),
        'dev_macro_f1': round(f1, 4),
        'macro_f1_ci_low':  round(ci_low, 4),
        'macro_f1_ci_high': round(ci_high, 4),
        'model': model,
        'y_pred': y_pred,
    }


# ─── Load / build cached data ─────────────────────────────────────────────────

def load_or_build_variant_dict(cache_dir: str, smoke: bool,
                                train_tweets: list[dict]) -> dict:
    cache_key = get_cache_key(smoke)
    path = os.path.join(cache_dir, f"variant_dict_{cache_key}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return data[0] if isinstance(data, tuple) else data
    variant_dict, rejections = build_variant_dictionary(train_tweets)
    with open(path, 'wb') as f:
        pickle.dump((variant_dict, rejections), f)
    return variant_dict


def load_lid_model(cache_dir: str, smoke: bool) -> NaiveBayesLID:
    cache_key = get_cache_key(smoke)
    for name in [f"lid_normalised_{cache_key}.pkl", "lid_normalised.pkl",
                 f"lid_raw_{cache_key}.pkl", "lid_raw.pkl"]:
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"  Loaded LID model from {path}")
            return model
    raise FileNotFoundError("No cached LID model found in " + cache_dir)


# ─── Save ablation and deltas ─────────────────────────────────────────────────

def save_ablation(results: list[dict], out_dir: str) -> None:
    cols = ['config', 'groups', 'best_C', 'dev_accuracy', 'dev_macro_f1',
            'macro_f1_ci_low', 'macro_f1_ci_high']
    df = pd.DataFrame([{k: v for k, v in r.items() if k in cols} for r in results])
    save_dataframe_csv(df, os.path.join(out_dir, 'baseline_ablation.csv'))
    print("\n=== Ablation Results ===")
    print(df.to_string(index=False))


def save_deltas(results: list[dict], y_dev: list[str], out_dir: str) -> None:
    """Paired bootstrap deltas for consecutive pairs and R1 vs R1_raw."""
    pmap = {r['config']: r['y_pred'] for r in results}
    pairs = [
        ('R1-R0',     'R1',     'R0'),
        ('R2-R1',     'R2',     'R1'),
        ('R3-R2',     'R3',     'R2'),
        ('R4-R3',     'R4',     'R3'),
        ('R1-R1_raw', 'R1',     'R1_raw'),
    ]
    rows = []
    for label, a_name, b_name in pairs:
        if a_name not in pmap or b_name not in pmap:
            continue
        mean_d, lo, hi = paired_bootstrap(y_dev, pmap[a_name], pmap[b_name])
        rows.append({'pair': label, 'mean_delta_macro_f1': round(mean_d, 4),
                     'ci_low': round(lo, 4), 'ci_high': round(hi, 4)})
    df = pd.DataFrame(rows)
    save_dataframe_csv(df, os.path.join(out_dir, 'baseline_deltas.csv'))
    print("\n=== Deltas ===")
    print(df.to_string(index=False))


# ─── Per-class report and confusion matrix ────────────────────────────────────

def save_best_reports(
    results: list[dict],
    train_tweets: list[dict],
    dev_tweets: list[dict],
    y_dev: list[str],
    out_dir: str,
    smoke_test: bool,
) -> None:
    exclude = {'R4_gold', 'R4_insample', 'R1_raw'}
    main_results = [r for r in results if r['config'] not in exclude]
    best = max(main_results, key=lambda r: r['dev_macro_f1'])
    name = best['config']
    groups = best['groups'].split('+')
    print(f"\nBest config: {name}. Computing per-class breakdown...")

    tr = train_tweets[:500] if smoke_test else train_tweets
    X_tr, X_dv, _ = build_features(tr, dev_tweets, groups,
                                    'normalised_tokens', 'lid_tags')
    y_pred = best['model'].predict(X_dv)

    rdf = get_classification_report_df(y_dev, y_pred, labels=SENTIMENT_LABELS)
    save_dataframe_csv(rdf, os.path.join(out_dir, f'baseline_{name}_per_class.csv'))
    print(rdf.to_string(index=False))

    plot_confusion_matrix(
        y_dev, y_pred, labels=SENTIMENT_LABELS,
        output_path=os.path.join(out_dir, f'baseline_{name}_confusion.png'),
        title=f"Sentiment Baseline — {name} Confusion Matrix",
    )
    print(f"Confusion matrix saved to {out_dir}/baseline_{name}_confusion.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def _prepare_data(smoke: bool, cache_dir: str, out_dir: str):
    """Load, clean, normalise tweets; run CV LID; return (train, dev, y_train, y_dev)."""
    train_tweets, dev_tweets = load_dataset()
    if smoke:
        train_tweets = train_tweets[:1000]
        dev_tweets   = dev_tweets[:300]

    add_cleaned(train_tweets)
    add_cleaned(dev_tweets)

    variant_dict = load_or_build_variant_dict(cache_dir, smoke, train_tweets)
    add_normalised(train_tweets, variant_dict)
    add_normalised(dev_tweets, variant_dict)

    lid_model = load_lid_model(cache_dir, smoke)

    print("Running 5-fold CV LID tagging on train...")
    run_lid_cv(train_tweets, dev_tweets, lid_model, out_dir)
    # dev now has 'lid_tags' from full-train LID model

    # In-sample tags on train (from full-train LID model, for R4_insample)
    add_lid_tags(train_tweets, lid_model, 'normalised_tokens')
    # Rename to insample_lid_tags to avoid overwriting cv_lid_tags
    for tweet in train_tweets:
        tweet['insample_lid_tags'] = tweet.pop('lid_tags')
    # Restore cv_lid_tags as the working lid_tags on train
    for tweet in train_tweets:
        tweet['lid_tags'] = tweet['cv_lid_tags']

    return train_tweets, dev_tweets, get_labels(train_tweets), get_labels(dev_tweets)


def _run_ablation(train, dev, y_train, y_dev, smoke) -> list[dict]:
    """Run all configs and return list of result dicts."""
    results = []
    main_configs = [
        ('R0', ['F1']),
        ('R1', ['F1', 'F2']),
        ('R2', ['F1', 'F2', 'F3']),
        ('R3', ['F1', 'F2', 'F3', 'F4']),
        ('R4', ['F1', 'F2', 'F3', 'F4', 'F5']),
    ]
    for name, groups in main_configs:
        r = run_config(name, groups, train, dev, y_train, y_dev,
                       'normalised_tokens', 'lid_tags', smoke_test=smoke)
        results.append(r)

    # R1_raw: F1+F2 on raw tokens (no cleaning/normalisation)
    r = run_config('R1_raw', ['F1', 'F2'], train, dev, y_train, y_dev,
                   'tokens', 'lid_tags', smoke_test=smoke)
    results.append(r)

    # R4_gold: upper bound with gold LID tags
    for tweet in train:
        tweet['gold_lid_tags'] = [TAG_MAP.get(t, 'O') for t in tweet['tags']]
    for tweet in dev:
        tweet['gold_lid_tags'] = [TAG_MAP.get(t, 'O') for t in tweet['tags']]
    r = run_config('R4_gold', ['F1', 'F2', 'F3', 'F4', 'F5'], train, dev,
                   y_train, y_dev, 'normalised_tokens', 'gold_lid_tags', smoke_test=smoke)
    results.append(r)

    # R4_insample: diagnostic with in-sample train tags + predicted dev tags
    # dev uses the same full-train LID tags (lid_tags); alias on dev for key consistency
    for tweet in dev:
        tweet['insample_lid_tags'] = tweet['lid_tags']
    r = run_config('R4_insample', ['F1', 'F2', 'F3', 'F4', 'F5'], train, dev,
                   y_train, y_dev, 'normalised_tokens', 'insample_lid_tags', smoke_test=smoke)
    results.append(r)

    return results


def main(smoke_test: bool = False) -> None:
    out_dir   = "results_smoke" if smoke_test else "results"
    cache_dir = os.path.join(out_dir, "cache") if smoke_test else CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading data...")
    train_tweets, dev_tweets, y_train, y_dev = _prepare_data(smoke_test, cache_dir, out_dir)

    results = _run_ablation(train_tweets, dev_tweets, y_train, y_dev, smoke_test)

    save_ablation(results, out_dir)
    save_deltas(results, y_dev, out_dir)
    save_best_reports(results, train_tweets, dev_tweets, y_dev, out_dir, smoke_test)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke:
        print("SMOKE TEST MODE")
    main(smoke_test=smoke)
