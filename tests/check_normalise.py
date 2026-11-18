"""Tests for normalise.py: I1 bad merges absent, I6 fixes, idempotence, 1-to-1."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from src.normalise import clean_token, normalise_tokens, build_variant_dictionary
from src.data_io import load_dataset

# ── I6: clean_token must not corrupt digits or emoji ─────────────────────────
assert clean_token('10000') == '10000', f"digits corrupted: {clean_token('10000')}"
assert clean_token('😀😀😀') == '😀😀😀', f"emoji corrupted: {clean_token('😀😀😀')}"
assert clean_token('soooo') == 'soo', "letter elongation not collapsed"
assert clean_token('!!!') == '!!!', f"punct corrupted: {clean_token('!!!')}"
assert clean_token('aaa111') == 'aa111', f"mixed corrupted: {clean_token('aaa111')}"
print("PASS: clean_token I6 fixes")

# ── Load variant dict ────────────────────────────────────────────────────────
vd_path = "results/variant_dict.json"
if os.path.exists(vd_path):
    with open(vd_path) as f:
        vd = json.load(f)
else:
    print("Building variant dict for tests...")
    train, _ = load_dataset()
    vd, _ = build_variant_dictionary(train)

# ── I1: bad merges must be absent ────────────────────────────────────────────
BAD_MERGES = [
    ('hello', 'haal'), ('sorry', 'sir'), ('busy', 'bas'),
    ('do', 'diya'), ('he', 'hai'), ('hi', 'hai'),
    ('men', 'main'), ('too', 'tu'), ('jao', 'ji'),
    ('kya', 'ki'), ('mudde', 'modi'),
    ('pyaar', 'par'), ('haan', 'hain'), ('khana', 'kahan'), ('jane', 'jaan'),
]
for src, tgt in BAD_MERGES:
    actual = vd.get(src)
    assert actual != tgt, f"Bad merge still present: {src} -> {tgt}"

DROPPED_WORDS = ['masti', 'mast', 'ayyo', 'imho', 'brb', 'lmao', 'asap', 'q', 'k', 'h', 'ha']
for w in DROPPED_WORDS:
    assert w not in vd, f"Dropped word still in dict keys: {w}"

for k, v in vd.items():
    assert ' ' not in v, f"Canonical contains a space: {k} -> {v}"
    assert v.isalpha(), f"Canonical is not alphabetic: {k} -> {v}"

print("PASS: I1 bad merges and dropped words absent, no values with spaces")

# ── Idempotence: d[d[k]] == d[k] ────────────────────────────────────────────
for k, v in vd.items():
    assert vd.get(v, v) == v, f"Chain not resolved: {k} -> {v} -> {vd.get(v, v)}"
print("PASS: idempotence (no chains)")

# ── 1-to-1 length preservation ───────────────────────────────────────────────
test_tokens = ['hello', 'world', '😀', '10000', 'bhaiii', '@user', '#tag']
result = normalise_tokens(test_tokens, vd)
assert len(result) == len(test_tokens), f"Length mismatch: {len(result)} vs {len(test_tokens)}"
print("PASS: 1-to-1 length preservation")

# ── Full dataset 1-to-1 check ───────────────────────────────────────────────
train, dev = load_dataset()
for split_name, tweets in [("train", train), ("dev", dev)]:
    for i, tweet in enumerate(tweets):
        normed = normalise_tokens(tweet['tokens'], vd)
        assert len(normed) == len(tweet['tokens']), \
            f"{split_name} tweet {i}: length {len(normed)} != {len(tweet['tokens'])}"
print("PASS: full dataset 1-to-1 check")

print("\nAll normalise tests passed!")
