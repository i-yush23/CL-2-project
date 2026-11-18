"""Tests for normalise.py: plain asserts matching Task 1 specification."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from src.normalise import clean_token, normalise_tokens, build_variant_dictionary

# ── Clean token assertions ───────────────────────────────────────────────────
assert clean_token('😂😂😂') == '😂😂😂', f"emoji corrupted: {clean_token('😂😂😂')}"
assert clean_token('10000') == '10000', f"digits corrupted: {clean_token('10000')}"

# ── Load results/variant_dict.json ───────────────────────────────────────────
vd_path = os.path.join(PROJECT_ROOT, "results", "variant_dict.json")
assert os.path.exists(vd_path), f"Missing {vd_path}"
with open(vd_path, "r", encoding="utf-8") as f:
    vd = json.load(f)

# ── Forbidden keys check ─────────────────────────────────────────────────────
FORBIDDEN_KEYS = [
    'hello', 'sorry', 'busy', 'do', 'he', 'hi', 'men', 'man', 'too', 'jao',
    'kya', 'mudde', 'ayush', 'masti', 'mast', 'ayyo', 'imho', 'brb', 'lmao',
    'asap', 'q', 'k', 'h', 'ha', 'haan',
    # Step 3 forms
    'pyara', 'pyare', 'pyari', 'piyar', 'kahani', 'kahaani', 'khani', 'khaani',
    'khaana', 'nahane', 'kaur', 'kyari', 'kayar', 'jaana', 'jaanna', 'jaanne', 'jaanay'
]
for k in FORBIDDEN_KEYS:
    assert k not in vd, f"Forbidden key present in variant_dict: {k} -> {vd.get(k)}"

# ── Idempotence: d[d[k]] == d[k] for every key whose target is also a key ────
for k, v in vd.items():
    if v in vd:
        assert vd[v] == v, f"Chain not resolved: {k} -> {v} -> {vd[v]}"
    assert vd.get(v, v) == v, f"Idempotence failed: {k} -> {v} -> {vd.get(v, v)}"

# ── No key or value contains a space; all keys and values are alphabetic ──────
for k, v in vd.items():
    assert ' ' not in k, f"Key contains space: {k!r}"
    assert ' ' not in v, f"Value contains space: {v!r}"
    assert k.isalpha(), f"Key is not alphabetic: {k!r}"
    assert v.isalpha(), f"Value is not alphabetic: {v!r}"

# ── normalise_tokens returns same length on mixed list ────────────────────────
mixed_input = ['U', 'luvvv', '@a', '#b', '😂😂😂', 'http://x', '']
mixed_output = normalise_tokens(mixed_input, vd)
assert len(mixed_output) == len(mixed_input), (
    f"Length mismatch: {len(mixed_output)} != {len(mixed_input)}"
)

# ── build_variant_dictionary returns a 2-tuple on tiny hand-made train list ──
dummy_train = [
    {'tokens': ['bhai', 'bhaii', 'achha'], 'tags': ['Hin', 'Hin', 'Hin']},
    {'tokens': ['bhai', 'bhaiii', 'theek'], 'tags': ['Hin', 'Hin', 'Hin']},
    {'tokens': ['good', 'morning', 'bhai'], 'tags': ['Eng', 'Eng', 'Hin']},
]
res = build_variant_dictionary(dummy_train)
assert isinstance(res, tuple) and len(res) == 2, (
    f"Expected 2-tuple from build_variant_dictionary, got: {type(res)} len={len(res) if isinstance(res, tuple) else 'N/A'}"
)
assert isinstance(res[0], dict), f"First element must be dict, got {type(res[0])}"
assert isinstance(res[1], dict), f"Second element must be dict, got {type(res[1])}"

print("ALL CHECKS PASSED")
