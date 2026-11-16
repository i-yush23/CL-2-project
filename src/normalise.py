import re
import json
import os
import sys
from collections import defaultdict, Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cl1 import get_phonetic_hash, load_transliteration_variants, load_slang_map

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'^#(.+)$')
ELONGATION_RE = re.compile(r'(.)\1{2,}')  # 3+ repeats -> 2 (keep at most 2)


def clean_token(tok: str) -> str:
    """
    Lowercase, map URLs -> <url>, @mentions -> <user>,
    strip '#' from hashtags, collapse elongations (3+ chars to 2).
    Emoji are left untouched (they are not purely alpha so the regex is safe).
    """
    if URL_RE.fullmatch(tok):
        return '<url>'
    if MENTION_RE.fullmatch(tok):
        return '<user>'
    m = HASHTAG_RE.match(tok)
    if m:
        tok = m.group(1)
    # Lowercase before elongation so case doesn't prevent collapsing
    tok = tok.lower()
    # Collapse 3+ identical consecutive characters to 2
    tok = ELONGATION_RE.sub(r'\1\1', tok)
    return tok


def _majority_tag(tag_list: list[str]) -> str:
    """Return the most common tag in a list."""
    return Counter(tag_list).most_common(1)[0][0]


def build_variant_dictionary(train_tweets: list[dict]) -> dict[str, str]:
    """
    Build a canonical-form dictionary from train tweets only.

    Strategy:
    1. For each train token, group by its phonetic hash.
    2. Within each group, only merge tokens whose majority gold tag is 'Hin'.
       (Never merge English forms into Hindi canonical forms.)
    3. The canonical form is the most frequent surface form in the group.
    4. Merge in CL-1 transliteration_variants.json and slang_map.json
       as background, only where there is no learned entry for the same key.

    Returns: dict mapping lowercase surface form -> canonical lowercase form.
    """
    # Gather (cleaned_token, gold_tag) pairs from train
    hash_to_tokens: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tweet in train_tweets:
        for tok, tag in zip(tweet['tokens'], tweet['tags']):
            cleaned = clean_token(tok)
            if not cleaned or cleaned in ('<url>', '<user>'):
                continue
            if not cleaned.isalpha():
                continue  # Skip punctuation, numbers, emoji
            hash_to_tokens[get_phonetic_hash(cleaned)].append((cleaned, tag))

    learned: dict[str, str] = {}
    MIN_FREQ = 3        # canonical must appear at least this many times
    MIN_VARIANT_FREQ = 2  # variant must appear at least this many times

    for ph, token_tag_pairs in hash_to_tokens.items():
        if len(token_tag_pairs) < 2:
            continue
        # Determine majority tag for each distinct surface form
        form_tags: dict[str, list[str]] = defaultdict(list)
        for form, tag in token_tag_pairs:
            form_tags[form].append(tag)

        # Keep only forms with majority Hin tag and minimum frequency
        hin_forms = {f: tags for f, tags in form_tags.items()
                     if _majority_tag(tags) == 'Hin' and len(tags) >= MIN_VARIANT_FREQ}
        if len(hin_forms) < 2:
            continue  # Need 2+ Hin variants to merge

        # Pick canonical = most frequent Hin form with minimum frequency
        form_freq = Counter({f: len(tags) for f, tags in hin_forms.items()})
        canonical, canon_freq = form_freq.most_common(1)[0]
        if canon_freq < MIN_FREQ:
            continue  # Skip if canonical is too rare

        for form in hin_forms:
            if form != canonical:
                learned[form] = canonical

    # Load CL-1 background dictionaries
    cl1_variants = load_transliteration_variants()
    cl1_slang = load_slang_map()

    # Merge background: add CL-1 entries only where learned has no entry
    for src, tgt in cl1_variants.items():
        if src not in learned:
            learned[src] = tgt
    for src, tgt in cl1_slang.items():
        if src not in learned:
            learned[src] = tgt

    return learned


def normalise_tokens(tokens: list[str], variant_dict: dict[str, str]) -> list[str]:
    """
    Return normalised tokens, strictly 1-to-1 with input (never split/merge).
    For each token: clean it, then look up in variant_dict.
    """
    out = []
    for tok in tokens:
        cleaned = clean_token(tok)
        normalised = variant_dict.get(cleaned, cleaned)
        out.append(normalised)
    return out


def compute_normalisation_stats(
    train_tweets: list[dict],
    dev_tweets: list[dict],
    variant_dict: dict[str, str],
) -> dict:
    """
    Count how many tokens changed on train and dev sets.
    Returns dict with keys: variant_groups, train_changed, dev_changed.
    """
    variant_groups = len(variant_dict)

    def count_changes(tweets):
        total, changed = 0, 0
        for tweet in tweets:
            for tok in tweet['tokens']:
                total += 1
                cleaned = clean_token(tok)
                if variant_dict.get(cleaned, cleaned) != cleaned:
                    changed += 1
        return total, changed

    train_total, train_changed = count_changes(train_tweets)
    dev_total, dev_changed = count_changes(dev_tweets)
    return {
        'variant_groups': variant_groups,
        'train_total': train_total,
        'train_changed': train_changed,
        'dev_total': dev_total,
        'dev_changed': dev_changed,
    }


def save_variant_dict(variant_dict: dict[str, str], path: str = "results/variant_dict.json") -> None:
    """Save variant dictionary to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(variant_dict, f, ensure_ascii=False, indent=2)


def save_normalisation_stats(
    stats: dict,
    variant_dict: dict[str, str],
    path: str = "results/normalisation_stats.csv",
) -> None:
    """Save normalisation stats CSV with 20 example variant groups."""
    import pandas as pd
    rows = []
    rows.append({
        'metric': 'variant_groups',
        'value': stats['variant_groups'],
    })
    rows.append({
        'metric': 'train_tokens_total',
        'value': stats['train_total'],
    })
    rows.append({
        'metric': 'train_tokens_changed',
        'value': stats['train_changed'],
    })
    rows.append({
        'metric': 'dev_tokens_total',
        'value': stats['dev_total'],
    })
    rows.append({
        'metric': 'dev_tokens_changed',
        'value': stats['dev_changed'],
    })

    df_stats = pd.DataFrame(rows)
    example_rows = [{'source': k, 'canonical': v}
                    for k, v in list(variant_dict.items())[:20]]
    df_examples = pd.DataFrame(example_rows)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_stats.to_csv(path, index=False)
    example_path = path.replace('.csv', '_examples.csv')
    df_examples.to_csv(example_path, index=False)
    print("Stats:\n", df_stats.to_string(index=False))
    print("\n20 example variant entries:\n", df_examples.head(20).to_string(index=False))


if __name__ == "__main__":
    from src.data_io import load_dataset
    train_tweets, dev_tweets = load_dataset()
    print("Building variant dictionary from train only...")
    variant_dict = build_variant_dictionary(train_tweets)
    save_variant_dict(variant_dict)
    print(f"Variant dictionary has {len(variant_dict)} entries.")
    stats = compute_normalisation_stats(train_tweets, dev_tweets, variant_dict)
    save_normalisation_stats(stats, variant_dict)
