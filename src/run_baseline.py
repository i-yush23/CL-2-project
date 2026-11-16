"""
DELIVERABLE B: LR sentiment baseline — incremental feature ablation.

Configurations:
  R0: F1
  R1: F1+F2
  R2: F1+F2+F3
  R3: F1+F2+F3+F4
  R4: F1+F2+F3+F4+F5
  R1_raw: F1+F2 on raw (uncleaned/unnormalised) text — shows normalisation effect

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

from src.data_io import load_dataset
from src.normalise import clean_token, build_variant_dictionary, normalise_tokens
from src.lid import NaiveBayesLID, predict_tweets, TAG_MAP
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
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)


def get_labels(tweets: list[dict]) -> list[str]:
    return [t['sentiment'] for t in tweets]


def add_cleaned(tweets: list[dict]) -> None:
    for tweet in tweets:
        tweet['cleaned_tokens'] = [clean_token(t) for t in tweet['tokens']]


def add_normalised(tweets: list[dict], variant_dict: dict) -> None:
    for tweet in tweets:
        tweet['normalised_tokens'] = normalise_tokens(tweet['tokens'], variant_dict)


def add_lid_tags(tweets: list[dict], model: NaiveBayesLID, tokens_key: str = 'normalised_tokens') -> None:
    """Predict LID tags and store as tweet['lid_tags']."""
    preds = predict_tweets(model, tweets, tokens_key)
    for tweet, pred in zip(tweets, preds):
        tweet['lid_tags'] = pred


def add_gold_lid_tags(tweets: list[dict]) -> None:
    """Use gold tags (mapped) as lid_tags for upper-bound run."""
    for tweet in tweets:
        tweet['lid_tags'] = [TAG_MAP.get(t, 'O') for t in tweet['tags']]


def build_features(
    train: list[dict],
    dev: list[dict],
    use_groups: list[str],
    tokens_key: str = 'normalised_tokens',
    tags_key: str = 'lid_tags',
) -> tuple[sp.csr_matrix, sp.csr_matrix, dict]:
    """
    Fit feature builders on train, transform train+dev.
    Returns (X_train, X_dev, builders_dict).
    use_groups: subset of ['F1','F2','F3','F4','F5'].
    """
    parts_train, parts_dev = [], []
    builders = {}

    if 'F1' in use_groups:
        b = F1Builder()
        b.fit(train, tokens_key)
        parts_train.append(b.transform(train, tokens_key))
        parts_dev.append(b.transform(dev, tokens_key))
        builders['F1'] = b

    if 'F2' in use_groups:
        b = F2Builder()
        b.fit(train, tokens_key)
        parts_train.append(b.transform(train, tokens_key))
        parts_dev.append(b.transform(dev, tokens_key))
        builders['F2'] = b

    if 'F3' in use_groups:
        b = F3Builder()
        b.fit(train, tokens_key, tags_key)
        parts_train.append(b.transform(train, tokens_key, tags_key))
        parts_dev.append(b.transform(dev, tokens_key, tags_key))
        builders['F3'] = b

    if 'F4' in use_groups:
        b = F4Builder()
        b.fit(train, tags_key)
        parts_train.append(b.transform(train, tags_key))
        parts_dev.append(b.transform(dev, tags_key))
        builders['F4'] = b

    if 'F5' in use_groups:
        b = F5Builder()
        b.fit(train, tokens_key)
        parts_train.append(b.transform(train, tokens_key))
        parts_dev.append(b.transform(dev, tokens_key))
        builders['F5'] = b

    X_train = sp.hstack(parts_train).tocsr()
    X_dev = sp.hstack(parts_dev).tocsr()
    return X_train, X_dev, builders


def tune_and_eval(
    X_train: sp.csr_matrix,
    y_train: list[str],
    X_dev: sp.csr_matrix,
    y_dev: list[str],
) -> tuple[float, float, float, object]:
    """
    Grid search over C. Returns (best_C, dev_accuracy, dev_macro_f1, best_model).
    """
    best_c, best_f1, best_model = None, -1.0, None
    for c in C_GRID:
        lr = LogisticRegression(
            C=c,
            max_iter=2000,
            class_weight='balanced',
            random_state=SEED,
            solver='lbfgs',
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
    return best_c, m['accuracy'], m['macro_f1'], best_model


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
    print(f"\n--- Config: {name} | groups={groups} | tokens={tokens_key} ---")

    if smoke_test:
        tr = train[:500]
        y_tr = y_train[:500]
    else:
        tr = train
        y_tr = y_train

    X_train, X_dev, _ = build_features(tr, dev, groups, tokens_key, tags_key)
    best_c, acc, f1, model = tune_and_eval(X_train, y_tr, X_dev, y_dev)
    print(f"  Best C={best_c}  Acc={acc:.4f}  Macro-F1={f1:.4f}")

    return {
        'config': name,
        'groups': '+'.join(groups),
        'best_C': best_c,
        'dev_accuracy': round(acc, 4),
        'dev_macro_f1': round(f1, 4),
        'model': model,
    }


def main(smoke_test: bool = False) -> None:
    print("Loading data...")
    train_tweets, dev_tweets = load_dataset()

    add_cleaned(train_tweets)
    add_cleaned(dev_tweets)

    variant_dict_path = os.path.join(CACHE_DIR, "variant_dict.pkl")
    if os.path.exists(variant_dict_path):
        with open(variant_dict_path, 'rb') as f:
            variant_dict = pickle.load(f)
    else:
        variant_dict = build_variant_dictionary(train_tweets)
        with open(variant_dict_path, 'wb') as f:
            pickle.dump(variant_dict, f)

    add_normalised(train_tweets, variant_dict)
    add_normalised(dev_tweets, variant_dict)

    # Load LID model (normalised condition)
    lid_model_path = os.path.join(CACHE_DIR, "lid_normalised.pkl")
    if os.path.exists(lid_model_path):
        with open(lid_model_path, 'rb') as f:
            lid_model = pickle.load(f)
        print(f"Loaded LID model from {lid_model_path}")
    else:
        # Fallback: use raw model
        lid_model_path = os.path.join(CACHE_DIR, "lid_raw.pkl")
        with open(lid_model_path, 'rb') as f:
            lid_model = pickle.load(f)
        print(f"Loaded fallback LID model from {lid_model_path}")

    # Predict LID tags on train and dev (using normalised tokens)
    add_lid_tags(train_tweets, lid_model, 'normalised_tokens')
    add_lid_tags(dev_tweets, lid_model, 'normalised_tokens')

    y_train = get_labels(train_tweets)
    y_dev = get_labels(dev_tweets)

    results = []

    # Main ablation on cleaned+normalised text with predicted LID tags
    configs = [
        ('R0', ['F1']),
        ('R1', ['F1', 'F2']),
        ('R2', ['F1', 'F2', 'F3']),
        ('R3', ['F1', 'F2', 'F3', 'F4']),
        ('R4', ['F1', 'F2', 'F3', 'F4', 'F5']),
    ]
    for name, groups in configs:
        row = run_config(name, groups, train_tweets, dev_tweets,
                         y_train, y_dev, 'normalised_tokens', 'lid_tags',
                         smoke_test=smoke_test)
        results.append(row)

    # R1 on raw text — shows normalisation effect
    # Raw tweets don't have lid_tags; use gold tags as proxy for F3/F4/F5
    add_gold_lid_tags(train_tweets)
    add_gold_lid_tags(dev_tweets)
    row_raw = run_config('R1_raw', ['F1', 'F2'], train_tweets, dev_tweets,
                          y_train, y_dev, 'tokens', 'lid_tags',
                          smoke_test=smoke_test)
    results.append(row_raw)

    # Save ablation table (without model column)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results])
    save_dataframe_csv(df, "results/baseline_ablation.csv")
    print("\n=== Ablation Results ===")
    print(df.to_string(index=False))

    # For the best config (highest dev macro-F1 among main configs), save per-class + confusion matrix
    main_results = [r for r in results if r['config'] != 'R1_raw']
    best_result = max(main_results, key=lambda r: r['dev_macro_f1'])
    best_name = best_result['config']
    best_groups = best_result['groups'].split('+')
    print(f"\nBest config: {best_name}. Computing per-class breakdown...")

    # Re-fit with best config to get predictions
    if smoke_test:
        tr_best = train_tweets[:500]
        y_tr_best = y_train[:500]
    else:
        tr_best = train_tweets
        y_tr_best = y_train

    X_tr, X_dv, _ = build_features(tr_best, dev_tweets, best_groups,
                                    'normalised_tokens', 'lid_tags')
    best_model = best_result['model']
    y_pred_best = best_model.predict(X_dv)

    report_df = get_classification_report_df(y_dev, y_pred_best, labels=SENTIMENT_LABELS)
    save_dataframe_csv(report_df, f"results/baseline_{best_name}_per_class.csv")
    print(report_df.to_string(index=False))

    plot_confusion_matrix(
        y_dev, y_pred_best, labels=SENTIMENT_LABELS,
        output_path=f"results/baseline_{best_name}_confusion.png",
        title=f"Sentiment Baseline — {best_name} Confusion Matrix",
    )
    print(f"Confusion matrix saved to results/baseline_{best_name}_confusion.png")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke:
        print("SMOKE TEST MODE")
    main(smoke_test=smoke)
