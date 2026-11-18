"""
PHASE 5: LID error analysis — condition (a) raw vs (c) normalised.
Finds dev tokens where prediction changed from wrong->right (helped)
and right->wrong (hurt) after normalisation.
Saves top 40 of each kind to results/lid_error_examples.csv.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pickle
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from src.data_io import load_dataset
from src.normalise import clean_token, build_variant_dictionary, normalise_tokens, get_cache_key
from src.lid import NaiveBayesLID, TAG_MAP, LID_LABELS, predict_tweets

CACHE_DIR = "results/cache"


def load_model(name: str, cache_key: str = None) -> NaiveBayesLID:
    if cache_key:
        path = os.path.join(CACHE_DIR, f"lid_{name}_{cache_key}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    path = os.path.join(CACHE_DIR, f"lid_{name}.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_error_analysis() -> None:
    print("Loading data and models...")
    train_tweets, dev_tweets = load_dataset()

    # Rebuild cleaned / normalised tokens on dev
    for tweet in dev_tweets:
        tweet['cleaned_tokens'] = [clean_token(t) for t in tweet['tokens']]

    cache_key = get_cache_key(False)
    variant_dict_path = os.path.join(CACHE_DIR, f"variant_dict_{cache_key}.pkl")
    if not os.path.exists(variant_dict_path):
        variant_dict_path = os.path.join(CACHE_DIR, "variant_dict.pkl")
    with open(variant_dict_path, 'rb') as f:
        data = pickle.load(f)
        variant_dict = data[0] if isinstance(data, tuple) else data

    for tweet in dev_tweets:
        tweet['normalised_tokens'] = normalise_tokens(tweet['tokens'], variant_dict)

    model_raw = load_model('raw', cache_key)
    model_norm = load_model('normalised', cache_key)

    preds_raw  = predict_tweets(model_raw,  dev_tweets, 'tokens')
    preds_norm = predict_tweets(model_norm, dev_tweets, 'normalised_tokens')

    helped = []   # wrong in raw, right in normalised
    hurt   = []   # right in raw, wrong in normalised

    for tweet, raw_preds, norm_preds in zip(dev_tweets, preds_raw, preds_norm):
        for i, (raw_tok, norm_tok, gold_tag, raw_pred, norm_pred) in enumerate(
            zip(tweet['tokens'], tweet['normalised_tokens'],
                tweet['tags'], raw_preds, norm_preds)
        ):
            gold_lbl = TAG_MAP.get(gold_tag, 'O')
            raw_correct  = (raw_pred  == gold_lbl)
            norm_correct = (norm_pred == gold_lbl)

            if not raw_correct and norm_correct:
                helped.append({
                    'tweet_id':     tweet['uid'],
                    'surface_form': raw_tok,
                    'normalised':   norm_tok,
                    'gold':         gold_lbl,
                    'pred_raw':     raw_pred,
                    'pred_norm':    norm_pred,
                    'kind':         'helped',
                })
            elif raw_correct and not norm_correct:
                hurt.append({
                    'tweet_id':     tweet['uid'],
                    'surface_form': raw_tok,
                    'normalised':   norm_tok,
                    'gold':         gold_lbl,
                    'pred_raw':     raw_pred,
                    'pred_norm':    norm_pred,
                    'kind':         'hurt',
                })

    print(f"Tokens helped by normalisation: {len(helped)}")
    print(f"Tokens hurt by normalisation:   {len(hurt)}")

    # Take up to 40 of each, prioritise tokens where surface != normalised
    def rank(rows):
        changed = [r for r in rows if r['surface_form'] != r['normalised']]
        unchanged = [r for r in rows if r['surface_form'] == r['normalised']]
        return (changed + unchanged)[:40]

    rows = rank(helped) + rank(hurt)
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/lid_error_examples.csv", index=False)
    print(f"Saved {len(df)} rows to results/lid_error_examples.csv")

    # Print summary breakdown by gold label
    print("\nHelped breakdown by gold label:")
    print(pd.DataFrame(helped)['gold'].value_counts().to_string())
    print("\nHurt breakdown by gold label:")
    print(pd.DataFrame(hurt)['gold'].value_counts().to_string())


if __name__ == "__main__":
    run_error_analysis()
