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

from src.cl1 import load_cl1_emoji_lexicon

# Negation trigger words (Hinglish + English)
NEGATION_WORDS = {
    "not", "no", "never", "nahi", "nahin", "nahi", "mat", "na", "naa",
    "nhin", "nhin", "nhii", "nhii", "maat", "maat",
}
NEGATION_WINDOW = 3   # tokens after a negation word are flagged


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _tweet_to_text(tweet: dict, tokens_key: str = 'normalised_tokens') -> str:
    """Join tokens into a single string for TF-IDF vectorisers."""
    return " ".join(tweet[tokens_key])


def _tweet_to_lang_prefixed(tweet: dict, tags_key: str = 'lid_tags') -> str:
    """Build a string of language-prefixed unigrams for F3."""
    parts = []
    for tok, tag in zip(tweet[tags_key[:-4] + '_tokens'] if tags_key.endswith('tags') else
                        tweet['normalised_tokens'],
                        tweet[tags_key]):
        if tag == 'Hin':
            parts.append(f'Hin_{tok}')
        elif tag == 'Eng':
            parts.append(f'Eng_{tok}')
        # O tokens are skipped — they add noise not signal
    return " ".join(parts)


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
    where n = total tokens, u = O+EMT count,
    max_lang_tokens = max(Hin count, Eng count).
    Returns 0 if n == u.
    """
    n = len(tags)
    u = sum(1 for t in tags if t in ('O', 'EMT', 'UNI'))
    hin = sum(1 for t in tags if t == 'Hin')
    eng = sum(1 for t in tags if t == 'Eng')
    if n == u or (hin + eng) == 0:
        return 0.0
    return 100.0 * (1.0 - max(hin, eng) / (n - u))


def _switch_count(tags: list[str]) -> int:
    """Count adjacent (Hin, Eng) or (Eng, Hin) transitions (ignoring O/EMT)."""
    content_tags = [t for t in tags if t not in ('O', 'EMT', 'UNI')]
    switches = 0
    for i in range(1, len(content_tags)):
        if content_tags[i] != content_tags[i - 1]:
            switches += 1
    return switches


def _build_emoji_set(polarity: str, lexicon: dict) -> set:
    """Return set of emoji for a given polarity key (POSITIVE/NEGATIVE/NEUTRAL)."""
    return set(lexicon.get(polarity, []))


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
        hin_frac = sum(1 for t in tags if t == 'Hin') / n
        eng_frac = sum(1 for t in tags if t == 'Eng') / n
        o_frac   = sum(1 for t in tags if t in ('O', 'EMT')) / n
        switches = float(_switch_count(tags))
        cmi_val  = _cmi(tags)
        return [hin_frac, eng_frac, o_frac, switches, cmi_val]

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
        self.pos_set = _build_emoji_set('POSITIVE', lexicon)
        self.neg_set = _build_emoji_set('NEGATIVE', lexicon)
        self.neu_set = _build_emoji_set('NEUTRAL', lexicon)

    def _extract(self, tweet: dict, tokens_key: str = 'normalised_tokens') -> list[float]:
        toks = tweet[tokens_key]
        pos  = float(sum(1 for t in toks if t in self.pos_set))
        neg  = float(sum(1 for t in toks if t in self.neg_set))
        neu  = float(sum(1 for t in toks if t in self.neu_set))
        neg_flag = float(_negation_flag(toks))
        return [pos, neg, neu, neg_flag]

    def fit(self, tweets: list[dict], tokens_key: str = 'normalised_tokens') -> 'F5Builder':
        # No fitting needed; return self for API consistency
        return self

    def transform(self, tweets: list[dict], tokens_key: str = 'normalised_tokens') -> sp.csr_matrix:
        X = np.array([self._extract(t, tokens_key) for t in tweets])
        return sp.csr_matrix(X)
