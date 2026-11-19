"""
Feature builders for the LR sentiment baseline (Deliverable B).

Feature groups:
  F1: word 1-2-grams, TF-IDF
  F2: char 2-5-grams (char_wb), TF-IDF
  F3: language-prefixed unigrams (Hin_word, Eng_word), TF-IDF
  F4: language-mixing scalars (Hindi/Eng/Other fraction, switch count, CMI)
  F5: emoji polarity counts + negation flag

All vectorisers/scalers are fit on TRAIN only.
Each function returns a scipy sparse matrix (or dense for F4/F5, stacked as sparse).
"""

import re
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

from src.cl1 import load_cl1_emoji_lexicon, extract_emojis

# Negation trigger words (Hinglish + English), deduplicated
NEGATION_WORDS = frozenset([
    "not", "no", "never", "nahi", "nahin", "mat", "na", "naa",
    "nhin", "nhii", "maat",
])
NEGATION_WINDOW = 3   # tokens after a negation word are flagged


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _tweet_to_text(tweet: dict, tokens_key: str = 'normalised_tokens') -> str:
    """Join tokens into a single string for TF-IDF vectorisers."""
    return " ".join(tweet[tokens_key])


def _build_lang_prefixed_string(tokens: list[str], tags: list[str]) -> str:
    """Given parallel token + tag lists, return language-prefixed string."""
    parts = []
    for tok, tag in zip(tokens, tags):
        if tag == 'Hin':
            parts.append(f'Hin_{tok}')
        elif tag == 'Eng':
            parts.append(f'Eng_{tok}')
    return " ".join(parts)


def _cmi(tags: list[str]) -> float:
    """
    Code Mixing Index as defined in brief:
    CMI = 100 * (1 - max_lang_tokens / (n - u))
    where n = total tokens, u = O+EMT+NE count,
    max_lang_tokens = max(Hin count, Eng count).
    Returns 0 if n == u.
    """
    n = len(tags)
    u = sum(1 for t in tags if t in ('O', 'EMT', 'UNI', 'NE'))
    hin = sum(1 for t in tags if t == 'Hin')
    eng = sum(1 for t in tags if t == 'Eng')
    if n == u or (hin + eng) == 0:
        return 0.0
    return 100.0 * (1.0 - max(hin, eng) / (n - u))


def _switch_count(tags: list[str]) -> int:
    """Count adjacent (Hin, Eng) or (Eng, Hin) transitions (ignoring O/EMT/NE)."""
    content_tags = [t for t in tags if t not in ('O', 'EMT', 'UNI', 'NE')]
    switches = 0
    for i in range(1, len(content_tags)):
        if content_tags[i] != content_tags[i - 1]:
            switches += 1
    return switches


def _negation_flag(tokens: list[str]) -> int:
    """
    Count tokens within NEGATION_WINDOW positions after any negation word.
    Returns the count (can be used as a feature).
    """
    count = 0
    in_neg = 0
    for tok in tokens:
        if tok.lower() in NEGATION_WORDS:
            in_neg = NEGATION_WINDOW
        elif in_neg > 0:
            count += 1
            in_neg -= 1
    return count


# ─── Feature group builders ───────────────────────────────────────────────────

class F1Builder:
    """Word 1-2 unigrams+bigrams, TF-IDF. Fit on train."""

    def __init__(self):
        self.vec = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer='word',
            min_df=2,
            sublinear_tf=True,
        )

    def fit(self, tweets: list[dict], tokens_key: str = 'normalised_tokens') -> 'F1Builder':
        texts = [_tweet_to_text(t, tokens_key) for t in tweets]
        self.vec.fit(texts)
        return self

    def transform(self, tweets: list[dict], tokens_key: str = 'normalised_tokens') -> sp.csr_matrix:
        texts = [_tweet_to_text(t, tokens_key) for t in tweets]
        return self.vec.transform(texts)


class F2Builder:
    """Character 2-5-grams (char_wb), TF-IDF. Fit on train."""

    def __init__(self):
        self.vec = TfidfVectorizer(
            ngram_range=(2, 5),
            analyzer='char_wb',
            min_df=2,
            sublinear_tf=True,
        )

    def fit(self, tweets: list[dict], tokens_key: str = 'normalised_tokens') -> 'F2Builder':
        texts = [_tweet_to_text(t, tokens_key) for t in tweets]
        self.vec.fit(texts)
        return self

    def transform(self, tweets: list[dict], tokens_key: str = 'normalised_tokens') -> sp.csr_matrix:
        texts = [_tweet_to_text(t, tokens_key) for t in tweets]
        return self.vec.transform(texts)


class F3Builder:
    """Language-prefixed unigrams, TF-IDF. Fit on train."""

    def __init__(self):
        self.vec = TfidfVectorizer(
            ngram_range=(1, 1),
            analyzer='word',
            min_df=2,
            sublinear_tf=True,
        )

    def fit(self, tweets: list[dict], tokens_key: str = 'normalised_tokens',
            tags_key: str = 'lid_tags') -> 'F3Builder':
        texts = [_build_lang_prefixed_string(t[tokens_key], t[tags_key]) for t in tweets]
        self.vec.fit(texts)
        return self

    def transform(self, tweets: list[dict], tokens_key: str = 'normalised_tokens',
                  tags_key: str = 'lid_tags') -> sp.csr_matrix:
        texts = [_build_lang_prefixed_string(t[tokens_key], t[tags_key]) for t in tweets]
        return self.vec.transform(texts)


class F4Builder:
    """Language-mixing scalars: Hin/Eng/O fractions, switch count, CMI."""

    def __init__(self):
        self.scaler = MaxAbsScaler()

    def _extract(self, tweet: dict, tags_key: str = 'lid_tags') -> list[float]:
        tags = tweet[tags_key]
        n = len(tags)
        if n == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        hin_frac   = sum(1 for t in tags if t == 'Hin') / n
        eng_frac   = sum(1 for t in tags if t == 'Eng') / n
        other_frac = sum(1 for t in tags if t not in ('Hin', 'Eng')) / n
        switches   = float(_switch_count(tags))
        cmi_val    = _cmi(tags)
        return [hin_frac, eng_frac, other_frac, switches, cmi_val]

    def fit(self, tweets: list[dict], tags_key: str = 'lid_tags') -> 'F4Builder':
        X = np.array([self._extract(t, tags_key) for t in tweets])
        self.scaler.fit(X)
        return self

    def transform(self, tweets: list[dict], tags_key: str = 'lid_tags') -> sp.csr_matrix:
        X = np.array([self._extract(t, tags_key) for t in tweets])
        X_scaled = self.scaler.transform(X)
        return sp.csr_matrix(X_scaled)


class F5Builder:
    """Emoji polarity counts + negation flag."""

    def __init__(self):
        lexicon = load_cl1_emoji_lexicon()
        # Flatten multi-codepoint emoji entries via extract_emojis so that
        # matching is consistent with iterating over token characters.
        self.pos_set = frozenset(c for e in lexicon.get('POSITIVE', []) for c in extract_emojis(e))
        self.neg_set = frozenset(c for e in lexicon.get('NEGATIVE', []) for c in extract_emojis(e))
        self.neu_set = frozenset(c for e in lexicon.get('NEUTRAL', []) for c in extract_emojis(e))
        self.scaler = MaxAbsScaler()

    def _extract(self, tweet: dict, tokens_key: str = 'normalised_tokens') -> list[float]:
        toks = tweet[tokens_key]
        pos = neg = neu = 0
        for tok in toks:
            for ch in extract_emojis(tok):
                if ch in self.pos_set:
                    pos += 1
                elif ch in self.neg_set:
                    neg += 1
                elif ch in self.neu_set:
                    neu += 1
        neg_flag = float(_negation_flag(toks))
        return [float(pos), float(neg), float(neu), neg_flag]

    def fit(self, tweets: list[dict], tokens_key: str = 'normalised_tokens') -> 'F5Builder':
        X = np.array([self._extract(t, tokens_key) for t in tweets])
        self.scaler.fit(X)
        return self

    def transform(self, tweets: list[dict], tokens_key: str = 'normalised_tokens') -> sp.csr_matrix:
        X = np.array([self._extract(t, tokens_key) for t in tweets])
        X_scaled = self.scaler.transform(X)
        return sp.csr_matrix(X_scaled)



# ─── Functional make_fN API ───────────────────────────────────────────────────

def make_f1(
    train: list[dict], dev: list[dict],
    tokens_key: str = 'normalised_tokens',
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Word 1-2-gram TF-IDF, fit on train."""
    b = F1Builder()
    b.fit(train, tokens_key)
    return b.transform(train, tokens_key), b.transform(dev, tokens_key)


def make_f2(
    train: list[dict], dev: list[dict],
    tokens_key: str = 'normalised_tokens',
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Char_wb 2-5-gram TF-IDF, fit on train."""
    b = F2Builder()
    b.fit(train, tokens_key)
    return b.transform(train, tokens_key), b.transform(dev, tokens_key)


def make_f3(
    train: list[dict], dev: list[dict],
    tokens_key: str = 'normalised_tokens',
    tags_key: str = 'lid_tags',
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Language-prefixed unigrams TF-IDF, fit on train."""
    b = F3Builder()
    b.fit(train, tokens_key, tags_key)
    return b.transform(train, tokens_key, tags_key), b.transform(dev, tokens_key, tags_key)


def make_f4(
    train: list[dict], dev: list[dict],
    tags_key: str = 'lid_tags',
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Language-mixing scalars, MaxAbsScaler fit on train."""
    b = F4Builder()
    b.fit(train, tags_key)
    return b.transform(train, tags_key), b.transform(dev, tags_key)


def make_f5(
    train: list[dict], dev: list[dict],
    tokens_key: str = 'normalised_tokens',
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Emoji polarity counts + negation count, MaxAbsScaler fit on train."""
    b = F5Builder()
    b.fit(train, tokens_key)
    return b.transform(train, tokens_key), b.transform(dev, tokens_key)
