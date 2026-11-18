"""
DELIVERABLE A: LID experiments
Runs three conditions: (a) raw, (b) cleaned, (c) cleaned+normalised tokens.
Trains on train set, evaluates on dev set.
Saves per-condition CSVs/PNGs and a comparison table.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import pandas as pd

from src.data_io import load_dataset
from src.normalise import (
    clean_token,
    build_variant_dictionary,
    normalise_tokens,
    get_cache_key,
)
from src.lid import (
    NaiveBayesLID,
    TAG_MAP,
    LID_LABELS,
    build_flat_token_label_lists,
    predict_tweets,
)
from src.eval_utils import (
    compute_metrics,
    get_classification_report_df,
    plot_confusion_matrix,
    save_dataframe_csv,
)

CACHE_DIR = "results/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)


def add_cleaned_tokens(tweets: list[dict]) -> None:
    """Add 'cleaned_tokens' field to each tweet in-place."""
    for tweet in tweets:
        tweet['cleaned_tokens'] = [clean_token(t) for t in tweet['tokens']]


def add_normalised_tokens(tweets: list[dict], variant_dict: dict) -> None:
    """Add 'normalised_tokens' field to each tweet in-place."""
    for tweet in tweets:
        normed = normalise_tokens(tweet['tokens'], variant_dict)
        assert len(normed) == len(tweet['tokens']), "Length mismatch in normalise_tokens"
        tweet['normalised_tokens'] = normed


def flatten_gold_labels(tweets: list[dict]) -> list[str]:
    """Flatten all gold tags (mapped) for a split."""
    return [TAG_MAP.get(tag, 'O') for tweet in tweets for tag in tweet['tags']]


def run_condition(
    name: str,
    train_tweets: list[dict],
    dev_tweets: list[dict],
    tokens_key: str,
    smoke_test: bool = False,
) -> dict:
    """Train LID, evaluate on dev, save results. Returns metrics dict."""
    print(f"\n--- Condition: {name} (tokens_key={tokens_key}) ---")

    cache_key = get_cache_key(smoke_test)
    model_path = os.path.join(CACHE_DIR, f"lid_{name}_{cache_key}.pkl")

    if os.path.exists(model_path):
        print("  Loading cached model...")
        model = NaiveBayesLID.load(model_path)
    else:
        train_toks, train_lbls = build_flat_token_label_lists(train_tweets, tokens_key)
        if smoke_test:
            # Count tokens in first 500 tweets
            n_smoke = sum(len(t[tokens_key]) for t in train_tweets[:500])
            train_toks = train_toks[:n_smoke]
            train_lbls = train_lbls[:n_smoke]
        model = NaiveBayesLID()
        model.fit(train_toks, train_lbls)
        model.save(model_path)

    # Evaluate
    if smoke_test:
        eval_tweets = dev_tweets[:100]
    else:
        eval_tweets = dev_tweets

    y_true = flatten_gold_labels(eval_tweets)
    tweet_preds = predict_tweets(model, eval_tweets, tokens_key)
    y_pred = [lbl for preds in tweet_preds for lbl in preds]

    metrics = compute_metrics(y_true, y_pred, labels=LID_LABELS)
    print(f"  Accuracy: {metrics['accuracy']:.4f}  Macro-F1: {metrics['macro_f1']:.4f}")

    # Save per-class report
    report_df = get_classification_report_df(y_true, y_pred, labels=LID_LABELS)
    report_df.insert(0, 'condition', name)
    save_dataframe_csv(report_df, f"results/lid_{name}.csv")

    # Save confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, labels=LID_LABELS,
        output_path=f"results/lid_{name}.png",
        title=f"LID Confusion Matrix - {name}",
    )

    return {'condition': name, **metrics}


def main(smoke_test: bool = False) -> None:
    print("Loading data...")
    train_tweets, dev_tweets = load_dataset()

    print("Adding cleaned tokens...")
    add_cleaned_tokens(train_tweets)
    add_cleaned_tokens(dev_tweets)

    print("Building variant dictionary (train only)...")
    cache_key = get_cache_key(smoke_test)
    variant_dict_path = os.path.join(CACHE_DIR, f"variant_dict_{cache_key}.pkl")
    if os.path.exists(variant_dict_path):
        with open(variant_dict_path, 'rb') as f:
            import pickle
            variant_dict = pickle.load(f)
        print(f"  Loaded cached variant dict ({len(variant_dict)} entries).")
    else:
        variant_dict, _ = build_variant_dictionary(train_tweets)
        with open(variant_dict_path, 'wb') as f:
            import pickle
            pickle.dump(variant_dict, f)
        print(f"  Built variant dict ({len(variant_dict)} entries).")

    print("Adding normalised tokens...")
    add_normalised_tokens(train_tweets, variant_dict)
    add_normalised_tokens(dev_tweets, variant_dict)

    comparison_rows = []

    # Run all three conditions
    for name, key in [
        ('raw', 'tokens'),
        ('cleaned', 'cleaned_tokens'),
        ('normalised', 'normalised_tokens'),
    ]:
        # Clear cache for smoke test to retrain on subset
        if smoke_test:
            cache_key = get_cache_key(smoke_test)
            model_path = os.path.join(CACHE_DIR, f"lid_{name}_{cache_key}.pkl")
            if os.path.exists(model_path):
                os.remove(model_path)

        metrics = run_condition(name, train_tweets, dev_tweets, key, smoke_test=smoke_test)
        comparison_rows.append(metrics)

    comparison_df = pd.DataFrame(comparison_rows)
    save_dataframe_csv(comparison_df, "results/lid_comparison.csv")
    print("\n=== LID Comparison ===")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke:
        print("SMOKE TEST MODE (500-tweet subset)")
    main(smoke_test=smoke)
