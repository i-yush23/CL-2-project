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

from collections import Counter
import pickle
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from src.data_io import load_dataset
from src.normalise import clean_token, build_variant_dictionary, normalise_tokens, get_cache_key
from src.lid import NaiveBayesLID, TAG_MAP, LID_LABELS, predict_tweets, _is_language_independent

CACHE_DIR = "results/cache"


def load_model(name: str, cache_key: str = None, cache_dir: str = CACHE_DIR) -> NaiveBayesLID:
    if cache_key:
        path = os.path.join(cache_dir, f"lid_{name}_{cache_key}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    path = os.path.join(cache_dir, f"lid_{name}.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_change_type(surf: str, norm: str) -> str:
    if surf == norm:
        return 'none'
    elif surf.lower() == norm:
        return 'case_only'
    elif clean_token(surf) == norm:
        return 'cleaning'
    else:
        return 'dictionary'


def run_error_analysis() -> None:
    smoke = "--smoke" in sys.argv
    out_dir = "results_smoke" if smoke else "results"
    cache_dir = os.path.join(out_dir, "cache") if smoke else CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data and models...")
    train_tweets, dev_tweets = load_dataset()
    if smoke:
        dev_tweets = dev_tweets[:100]

    # Rebuild cleaned / normalised tokens on dev
    for tweet in dev_tweets:
        tweet['cleaned_tokens'] = [clean_token(t) for t in tweet['tokens']]

    cache_key = get_cache_key(smoke)
    variant_dict_path = os.path.join(cache_dir, f"variant_dict_{cache_key}.pkl")
    if not os.path.exists(variant_dict_path):
        variant_dict_path = os.path.join(cache_dir, "variant_dict.pkl")
    with open(variant_dict_path, 'rb') as f:
        data = pickle.load(f)
        variant_dict = data[0] if isinstance(data, tuple) else data

    for tweet in dev_tweets:
        tweet['normalised_tokens'] = normalise_tokens(tweet['tokens'], variant_dict)

    model_raw = load_model('raw', cache_key, cache_dir=cache_dir)
    model_clean = load_model('cleaned', cache_key, cache_dir=cache_dir)
    model_norm = load_model('normalised', cache_key, cache_dir=cache_dir)

    preds_raw   = predict_tweets(model_raw,   dev_tweets, 'tokens')
    preds_clean = predict_tweets(model_clean, dev_tweets, 'cleaned_tokens')
    preds_norm  = predict_tweets(model_norm,  dev_tweets, 'normalised_tokens')

    pairs = [
        ('raw_vs_norm', preds_raw, preds_norm),
        ('clean_vs_norm', preds_clean, preds_norm),
    ]

    summary_rows = []
    example_dfs = []
    dict_example_dfs = []

    for pair_name, preds_a, preds_c in pairs:
        helped_counts = Counter()
        hurt_counts = Counter()
        helped_by_ct = {ct: Counter() for ct in ['none', 'case_only', 'cleaning', 'dictionary']}
        hurt_by_ct   = {ct: Counter() for ct in ['none', 'case_only', 'cleaning', 'dictionary']}
        helped_instances = []
        hurt_instances = []

        for tweet, pa_list, pc_list in zip(dev_tweets, preds_a, preds_c):
            for surf, norm, tag, pa, pc in zip(
                tweet['tokens'], tweet['normalised_tokens'], tweet['tags'], pa_list, pc_list
            ):
                gold = TAG_MAP.get(tag, 'O')
                ct = get_change_type(surf, norm)
                is_helped = (pa != gold and pc == gold)
                is_hurt = (pa == gold and pc != gold)

                if is_helped:
                    helped_counts[gold] += 1
                    helped_by_ct[ct][gold] += 1
                    helped_instances.append((surf, norm, gold, pa, pc))
                elif is_hurt:
                    hurt_counts[gold] += 1
                    hurt_by_ct[ct][gold] += 1
                    hurt_instances.append((surf, norm, gold, pa, pc))

        # Existing summary rows: change_type='all'
        for lbl in LID_LABELS:
            summary_rows.append({'pair': pair_name, 'kind': 'helped', 'gold': lbl, 'count': helped_counts[lbl], 'change_type': 'all'})
        summary_rows.append({'pair': pair_name, 'kind': 'helped', 'gold': 'all', 'count': sum(helped_counts.values()), 'change_type': 'all'})

        for lbl in LID_LABELS:
            summary_rows.append({'pair': pair_name, 'kind': 'hurt', 'gold': lbl, 'count': hurt_counts[lbl], 'change_type': 'all'})
        summary_rows.append({'pair': pair_name, 'kind': 'hurt', 'gold': 'all', 'count': sum(hurt_counts.values()), 'change_type': 'all'})

        net_val = sum(helped_counts.values()) - sum(hurt_counts.values())
        summary_rows.append({'pair': pair_name, 'kind': 'net', 'gold': 'all', 'count': net_val, 'change_type': 'all'})

        # Per-change_type summary rows
        for ct in ['none', 'case_only', 'cleaning', 'dictionary']:
            h_ct = helped_by_ct[ct]
            u_ct = hurt_by_ct[ct]
            for lbl in LID_LABELS:
                summary_rows.append({'pair': pair_name, 'kind': 'helped', 'gold': lbl, 'count': h_ct[lbl], 'change_type': ct})
            summary_rows.append({'pair': pair_name, 'kind': 'helped', 'gold': 'all', 'count': sum(h_ct.values()), 'change_type': ct})

            for lbl in LID_LABELS:
                summary_rows.append({'pair': pair_name, 'kind': 'hurt', 'gold': lbl, 'count': u_ct[lbl], 'change_type': ct})
            summary_rows.append({'pair': pair_name, 'kind': 'hurt', 'gold': 'all', 'count': sum(u_ct.values()), 'change_type': ct})

            for lbl in LID_LABELS:
                summary_rows.append({'pair': pair_name, 'kind': 'net', 'gold': lbl, 'count': h_ct[lbl] - u_ct[lbl], 'change_type': ct})
            summary_rows.append({'pair': pair_name, 'kind': 'net', 'gold': 'all', 'count': sum(h_ct.values()) - sum(u_ct.values()), 'change_type': ct})

        # Examples: top 40 per pair and kind BY COUNT (ties broken alphabetically by surface)
        for kind, instances in [('helped', helped_instances), ('hurt', hurt_instances)]:
            counts = Counter(instances)
            rows = []
            for (surf, norm, gold, pa, pc), count in counts.items():
                ct = get_change_type(surf, norm)
                rows.append({
                    'pair': pair_name,
                    'kind': kind,
                    'surface': surf,
                    'normalised': norm,
                    'gold': gold,
                    'pred_a': pa,
                    'pred_c': pc,
                    'count': count,
                    'change_type': ct,
                    'token_changed': (ct != 'none'),
                })
            df_k = pd.DataFrame(rows)
            if not df_k.empty:
                df_k = df_k.sort_values(
                    by=['count', 'surface', 'normalised', 'gold', 'pred_a', 'pred_c'],
                    ascending=[False, True, True, True, True, True]
                )
                example_dfs.append(df_k.head(40))

                dict_df_k = df_k[df_k['change_type'] == 'dictionary'].head(40)
                dict_example_dfs.append(dict_df_k)

    summary_df = pd.DataFrame(summary_rows, columns=['pair', 'kind', 'gold', 'count', 'change_type'])
    summary_path = os.path.join(out_dir, "lid_error_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {len(summary_df)} rows to {summary_path}")

    examples_df = pd.concat(example_dfs, ignore_index=True) if example_dfs else pd.DataFrame()
    examples_path = os.path.join(out_dir, "lid_error_examples.csv")
    examples_df.to_csv(examples_path, index=False)
    print(f"Saved {len(examples_df)} rows to {examples_path}")

    dict_examples_df = pd.concat(dict_example_dfs, ignore_index=True) if dict_example_dfs else pd.DataFrame()
    dict_examples_path = os.path.join(out_dir, "lid_error_examples_dictionary.csv")
    dict_examples_df.to_csv(dict_examples_path, index=False)
    print(f"Saved {len(dict_examples_df)} rows to {dict_examples_path}")

    # Write results/lid_rule_gap.csv
    unrouted_gold_O = []
    for tweet, preds in zip(dev_tweets, preds_raw):
        for tok, tag, pred in zip(tweet['tokens'], tweet['tags'], preds):
            gold = TAG_MAP.get(tag, 'O')
            if gold == 'O' and not _is_language_independent(tok):
                unrouted_gold_O.append((tok, pred))

    rule_gap_counts = Counter(unrouted_gold_O)
    rule_gap_rows = [
        {'surface': tok, 'pred': pred, 'count': c}
        for (tok, pred), c in rule_gap_counts.items()
    ]
    rule_gap_df = pd.DataFrame(rule_gap_rows)
    if not rule_gap_df.empty:
        rule_gap_df = rule_gap_df.sort_values(
            by=['count', 'surface', 'pred'],
            ascending=[False, True, True]
        ).head(30)
    rule_gap_path = os.path.join(out_dir, "lid_rule_gap.csv")
    rule_gap_df.to_csv(rule_gap_path, index=False)
    print(f"Saved {len(rule_gap_df)} rows to {rule_gap_path}")


if __name__ == "__main__":
    run_error_analysis()
