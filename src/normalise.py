import re
import json
import os
import sys
import hashlib
import random
import difflib
from collections import defaultdict, Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cl1 import get_phonetic_hash, load_transliteration_variants, load_slang_map

# ── Named constants for variant dictionary building ──────────────────────────
MIN_CANON_FREQ = 3       # canonical must appear at least this many times
MIN_VARIANT_FREQ = 2     # variant must appear at least this many times
MIN_FREQ_RATIO = 5       # canonical frequency must be >= 5 * source frequency
MIN_TOKEN_LEN = 3        # minimum length for source and canonical
HIN_SHARE_MIN = 0.8      # form must have >= this Hin share to qualify
ENG_SHARE_MAX = 0.1      # form must have <= this Eng share to qualify
ENG_BLOCKLIST_THRESH = 0.3  # forms with Eng share >= this are blocklisted
SIMILARITY_MIN = 0.7     # difflib ratio threshold between source and canonical
ALLOW_SHORT = {"kr", "jb", "kb", "pr", "pe", "naa", "hn"}
CL1_EXPLICIT_DROP = {"masti", "mast", "ayyo", "imho", "brb", "lmao", "asap", "q", "k", "h", "ha"}
BLOCKED_MERGES = {('pyaar', 'par'), ('pyar', 'par'), ('haan', 'hain'), ('khana', 'kahan'), ('jane', 'jaan')}
# hand-curated after inspection of the train-built dictionary
NEVER_SOURCE = {
    'pyara', 'pyare', 'pyari', 'piyar', 'kahani', 'kahaani', 'khani', 'khaani',
    'khaana', 'nahane', 'kaur', 'kyari', 'kayar', 'jaana', 'jaanna', 'jaanne', 'jaanay',
}
SEED = 42

# ── Regexes ──────────────────────────────────────────────────────────────────
URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'^#(.+)$')
LETTER_ELONGATION_RE = re.compile(r'([a-zA-Z])\1{2,}')  # only ASCII letters


def clean_token(tok: str) -> str:
    """Clean a single token: lowercase, URLs, mentions, hashtags, elongation.

    Elongation collapsing applies only to ASCII letters (not digits/emoji/punct).
    """
    if URL_RE.fullmatch(tok):
        return '<url>'
    if MENTION_RE.fullmatch(tok):
        return '<user>'
    m = HASHTAG_RE.match(tok)
    if m:
        tok = m.group(1)
    tok = tok.lower()
    tok = LETTER_ELONGATION_RE.sub(r'\1\1', tok)
    return tok


def _compute_tag_shares(train_tweets: list[dict]) -> dict[str, dict[str, float]]:
    """Compute per-form tag shares over all train occurrences.

    Returns {form: {'Hin': share, 'Eng': share, ...}}.
    """
    form_tag_counts: dict[str, Counter] = defaultdict(Counter)
    for tweet in train_tweets:
        for tok, tag in zip(tweet['tokens'], tweet['tags']):
            cleaned = clean_token(tok)
            if cleaned and cleaned not in ('<url>', '<user>') and cleaned.isalpha():
                form_tag_counts[cleaned][tag] += 1

    shares = {}
    for form, counts in form_tag_counts.items():
        total = sum(counts.values())
        shares[form] = {tag: cnt / total for tag, cnt in counts.items()}
    return shares


def _passes_hin_guard(form: str, tag_shares: dict) -> bool:
    """Check if a form passes the Hindi guard for variant dictionary."""
    if form not in tag_shares:
        return False
    shares = tag_shares[form]
    return (shares.get('Hin', 0) >= HIN_SHARE_MIN and
            shares.get('Eng', 0) <= ENG_SHARE_MAX)


def _is_eng_blocklisted(form: str, tag_shares: dict) -> bool:
    """Check if a form is blocklisted due to high English share."""
    if form not in tag_shares:
        return False
    return tag_shares[form].get('Eng', 0) >= ENG_BLOCKLIST_THRESH


def _resolve_chains(d: dict[str, str]) -> dict[str, str]:
    """Resolve transitive chains to fixed point with cycle guard."""
    # Find cycles and assign canonical representative
    canonical_repr = {}
    for k in list(d.keys()):
        curr = k
        path = []
        while curr in d and curr not in path:
            path.append(curr)
            curr = d[curr]
        if curr in path:
            idx = path.index(curr)
            cycle = path[idx:]
            canon = min(cycle)
            for node in cycle:
                canonical_repr[node] = canon

    # Trace each key to terminal target or cycle canonical
    resolved = {}
    for k in d:
        curr = k
        visited = set()
        while curr in d and curr not in visited:
            visited.add(curr)
            if curr in canonical_repr:
                curr = canonical_repr[curr]
                break
            curr = d[curr]
        if curr in canonical_repr:
            curr = canonical_repr[curr]
        if k != curr:
            resolved[k] = curr

    # Final pass: ensure target is not itself mapped
    for k, v in list(resolved.items()):
        target = v
        visited = {k}
        while target in resolved and target not in visited:
            visited.add(target)
            target = resolved[target]
        resolved[k] = target

    resolved = {k: v for k, v in resolved.items() if k != v}

    for k, v in resolved.items():
        assert resolved.get(v, v) == v, f"Chain not resolved: {k} -> {v} -> {resolved.get(v, v)}"
    return resolved


def build_variant_dictionary(train_tweets: list[dict]) -> tuple[dict[str, str], dict[str, int]]:
    """Build canonical-form dictionary from train tweets only.

    Returns (variant_dict, rejection_counts) where rejection_counts
    tracks CL-1 rejections by reason.
    """
    tag_shares = _compute_tag_shares(train_tweets)

    # Group tokens by phonetic hash
    hash_to_forms: dict[str, Counter] = defaultdict(Counter)
    for tweet in train_tweets:
        for tok, tag in zip(tweet['tokens'], tweet['tags']):
            cleaned = clean_token(tok)
            if (cleaned and cleaned not in ('<url>', '<user>') and
                    cleaned.isalpha() and len(cleaned) >= MIN_TOKEN_LEN):
                hash_to_forms[get_phonetic_hash(cleaned)][cleaned] += 1

    learned: dict[str, str] = {}

    for ph, form_counts in hash_to_forms.items():
        # Filter to forms passing Hin guard with minimum frequency
        passing = {f: c for f, c in form_counts.items()
                   if _passes_hin_guard(f, tag_shares) and
                   not _is_eng_blocklisted(f, tag_shares) and
                   c >= MIN_VARIANT_FREQ}
        if len(passing) < 2:
            continue

        # Canonical = most frequent passing form
        canonical = max(passing, key=lambda f: passing[f])
        if passing[canonical] < MIN_CANON_FREQ:
            continue
        if len(canonical) < MIN_TOKEN_LEN:
            continue

        for form in passing:
            if form == canonical:
                continue
            if len(form) < MIN_TOKEN_LEN:
                continue
            if form in NEVER_SOURCE:
                continue
            # Frequency guard: canonical frequency must dominate source frequency
            if passing[canonical] < MIN_FREQ_RATIO * passing[form]:
                continue
            if (form, canonical) in BLOCKED_MERGES:
                continue
            # Similarity check
            ratio = difflib.SequenceMatcher(None, form, canonical).ratio()
            if ratio < SIMILARITY_MIN:
                continue
            learned[form] = canonical

    source_origin: dict[str, str] = {k: 'learned' for k in learned}

    # Load CL-1 background dictionaries with guards
    cl1_variants = load_transliteration_variants()
    cl1_slang = load_slang_map()
    rejections: dict[str, int] = defaultdict(int)

    for label, cl1_dict in [('cl1_variants', cl1_variants), ('cl1_slang', cl1_slang)]:
        for src, tgt in cl1_dict.items():
            if src in learned:
                rejections[f'{label}_already_learned'] += 1
                continue
            if src in NEVER_SOURCE:
                rejections[f'{label}_never_source'] += 1
                continue
            if src in CL1_EXPLICIT_DROP or tgt in CL1_EXPLICIT_DROP:
                rejections[f'{label}_explicit_drop'] += 1
                continue
            # Min length 3 applies unless in ALLOW_SHORT
            if len(src) < MIN_TOKEN_LEN and src not in ALLOW_SHORT:
                rejections[f'{label}_src_too_short'] += 1
                continue
            if len(tgt) < MIN_TOKEN_LEN and tgt not in ALLOW_SHORT:
                rejections[f'{label}_tgt_too_short'] += 1
                continue
            # Target must be single alphabetic token with no spaces
            if ' ' in tgt or not tgt.isalpha():
                rejections[f'{label}_tgt_not_single_alpha'] += 1
                continue
            if _is_eng_blocklisted(src, tag_shares):
                rejections[f'{label}_src_eng_majority'] += 1
                continue
            # Target must pass the same Hin-share guard as learned entries
            if not _passes_hin_guard(tgt, tag_shares):
                rejections[f'{label}_tgt_fails_hin_guard'] += 1
                continue
            learned[src] = tgt
            source_origin[src] = label

    # Resolve chains
    learned = _resolve_chains(learned)
    build_variant_dictionary.source_origin = {
        k: source_origin.get(k, 'learned') for k in learned
    }

    return learned, dict(rejections)


def normalise_tokens(tokens: list[str], variant_dict: dict[str, str]) -> list[str]:
    """Return normalised tokens, strictly 1-to-1 with input."""
    out = []
    for tok in tokens:
        cleaned = clean_token(tok)
        normalised = variant_dict.get(cleaned, cleaned)
        out.append(normalised)
    assert len(out) == len(tokens), "normalise_tokens must be 1-to-1"
    return out


def compute_normalisation_stats(
    train_tweets: list[dict],
    dev_tweets: list[dict],
    variant_dict: dict[str, str],
    rejections: dict[str, int],
) -> dict:
    """Compute normalisation statistics excluding case-only changes."""
    entries = len(variant_dict)
    distinct_canonicals = len(set(variant_dict.values()))

    origin = getattr(build_variant_dictionary, 'source_origin', {})
    entries_learned = sum(1 for k in variant_dict if origin.get(k) == 'learned')
    entries_cl1_variants = sum(1 for k in variant_dict if origin.get(k) == 'cl1_variants')
    entries_cl1_slang = sum(1 for k in variant_dict if origin.get(k) == 'cl1_slang')

    def count_changes(tweets):
        total, changed = 0, 0
        for tweet in tweets:
            for tok in tweet['tokens']:
                total += 1
                cleaned = clean_token(tok)
                normalised = variant_dict.get(cleaned, cleaned)
                if normalised != cleaned:
                    changed += 1
        return total, changed

    train_total, train_changed = count_changes(train_tweets)
    dev_total, dev_changed = count_changes(dev_tweets)
    return {
        'entries': entries,
        'distinct_canonicals': distinct_canonicals,
        'entries_learned': entries_learned,
        'entries_cl1_variants': entries_cl1_variants,
        'entries_cl1_slang': entries_cl1_slang,
        'train_total': train_total,
        'train_changed': train_changed,
        'dev_total': dev_total,
        'dev_changed': dev_changed,
        'rejections': rejections,
    }


def save_variant_dict(variant_dict: dict[str, str],
                      path: str = "results/variant_dict.json") -> None:
    """Save variant dictionary to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(variant_dict, f, ensure_ascii=False, indent=2)


def save_normalisation_stats(
    stats: dict,
    variant_dict: dict[str, str],
    path: str = "results/normalisation_stats.csv",
) -> None:
    """Save normalisation stats and 20 seeded-random example groups."""
    import pandas as pd
    rows = [
        {'metric': 'entries', 'value': stats['entries']},
        {'metric': 'distinct_canonicals', 'value': stats['distinct_canonicals']},
        {'metric': 'entries_learned', 'value': stats['entries_learned']},
        {'metric': 'entries_cl1_variants', 'value': stats['entries_cl1_variants']},
        {'metric': 'entries_cl1_slang', 'value': stats['entries_cl1_slang']},
        {'metric': 'train_tokens_total', 'value': stats['train_total']},
        {'metric': 'train_tokens_changed', 'value': stats['train_changed']},
        {'metric': 'dev_tokens_total', 'value': stats['dev_total']},
        {'metric': 'dev_tokens_changed', 'value': stats['dev_changed']},
    ]
    for reason, cnt in sorted(stats.get('rejections', {}).items()):
        rows.append({'metric': f'cl1_rejection_{reason}', 'value': cnt})

    df_stats = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_stats.to_csv(path, index=False)

    # 20 seeded-random groups (canonical -> members)
    groups: dict[str, list[str]] = defaultdict(list)
    for src, tgt in variant_dict.items():
        groups[tgt].append(src)

    rng = random.Random(SEED)
    group_keys = sorted(groups.keys())
    sampled = rng.sample(group_keys, min(20, len(group_keys)))

    example_rows = []
    for canon in sampled:
        members = sorted(groups[canon])
        example_rows.append({
            'canonical': canon,
            'members': ', '.join(members),
            'member_count': len(members),
        })
    df_examples = pd.DataFrame(example_rows)
    example_path = path.replace('.csv', '_examples.csv')
    df_examples.to_csv(example_path, index=False)
    print("Stats:\n", df_stats.to_string(index=False))
    print(f"\n20 random example groups:\n", df_examples.to_string(index=False))


def get_cache_key(smoke: bool = False) -> str:
    """Hash of normalisation constants + smoke flag for cache keying."""
    key_str = (
        f"{MIN_CANON_FREQ}_{MIN_VARIANT_FREQ}_{MIN_FREQ_RATIO}_{MIN_TOKEN_LEN}_"
        f"{HIN_SHARE_MIN}_{ENG_SHARE_MAX}_{ENG_BLOCKLIST_THRESH}_{SIMILARITY_MIN}_{smoke}_"
        f"{','.join(sorted(ALLOW_SHORT))}_{','.join(sorted(CL1_EXPLICIT_DROP))}_"
        f"{','.join(sorted(f'{s}:{c}' for s, c in BLOCKED_MERGES))}_{','.join(sorted(NEVER_SOURCE))}"
    )
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


if __name__ == "__main__":
    from src.data_io import load_dataset
    train_tweets, dev_tweets = load_dataset()
    print("Building variant dictionary from train only...")
    variant_dict, rejections = build_variant_dictionary(train_tweets)
    save_variant_dict(variant_dict)
    print(f"Variant dictionary has {len(variant_dict)} entries.")
    stats = compute_normalisation_stats(train_tweets, dev_tweets, variant_dict, rejections)
    save_normalisation_stats(stats, variant_dict)
