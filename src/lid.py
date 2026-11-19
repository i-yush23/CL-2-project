"""
Token-level Naive Bayes LID using character n-grams + a context feature.

Design reference: external/CL-1-project/hinglish/statistical_models/lid_model.py
(which uses word + phonetic-hash NB with context smoothing).
Here we use character n-grams (1-4) as features (via sklearn CountVectorizer)
and train directly on the SentiMix gold tags, so we can retrain on any tag set.

Label mapping (documented in NOTES.md):
  SentiMix tag -> LID label
  Hin          -> Hin
  Eng          -> Eng
  O            -> O    (punctuation, digits, universal)
  EMT          -> O    (emoticon/emoji treated as language-neutral)

For Phase 4 we train in three conditions:
  (a) raw tokens
  (b) cleaned tokens (via clean_token)
  (c) cleaned + normalised tokens
"""

import os
import pickle
import math
from collections import Counter

from src.normalise import URL_RE, MENTION_RE

# Mapping from SentiMix gold tags to our LID label set
TAG_MAP = {
    'Hin': 'Hin',
    'Eng': 'Eng',
    'O': 'O',
    'EMT': 'O',
}
LID_LABELS = ['Hin', 'Eng', 'O']
ALPHA = 1.0  # Laplace smoothing
NGRAM_RANGE = (1, 4)   # character n-gram sizes


def _char_ngrams(word: str, n_min: int, n_max: int) -> list[str]:
    """Extract character n-grams from a word (padded with _ boundary markers)."""
    padded = f'_{word}_'
    ngrams = []
    for n in range(n_min, n_max + 1):
        for i in range(len(padded) - n + 1):
            ngrams.append(padded[i:i+n])
    return ngrams


def _is_language_independent(tok: str) -> bool:
    """Return True if token contains no letters, or is a URL / mention."""
    if not tok:
        return True
    if tok in ('<url>', '<user>') or URL_RE.fullmatch(tok) or MENTION_RE.fullmatch(tok):
        return True
    return not any(c.isalpha() for c in tok)


class NaiveBayesLID:
    """
    Character n-gram Naive Bayes LID with context smoothing, no phonetic fallback.
    Trains on (token, label) pairs; labels must be in LID_LABELS.
    Context smoothing: use the majority label of neighbour tokens
    to break ties (window=2, weight=0.1).
    """

    def __init__(self, ngram_range: tuple[int, int] = NGRAM_RANGE, alpha: float = ALPHA):
        self.ngram_range = ngram_range
        self.alpha = alpha
        self.labels: list[str] = []
        self.log_prior: dict[str, float] = {}
        self.ngram_counts: dict[str, dict[str, int]] = {}
        self.vocab: set[str] = set()
        self.total_ngrams: dict[str, int] = {}

    def fit(self, tokens: list[str], labels_seq: list[str]) -> 'NaiveBayesLID':
        """Train on flat lists of tokens and their labels."""
        label_counts: Counter = Counter(labels_seq)
        total = sum(label_counts.values())
        self.labels = sorted(label_counts.keys())
        self.log_prior = {lbl: math.log(cnt / total) for lbl, cnt in label_counts.items()}

        # Count n-grams per label
        for tok, lbl in zip(tokens, labels_seq):
            if lbl not in self.labels:
                continue
            tok_lower = tok.lower()
            ngrams = _char_ngrams(tok_lower, *self.ngram_range)
            for ng in ngrams:
                if ng not in self.ngram_counts:
                    self.ngram_counts[ng] = {}
                self.ngram_counts[ng][lbl] = self.ngram_counts[ng].get(lbl, 0) + 1
                self.vocab.add(ng)
            self.total_ngrams[lbl] = self.total_ngrams.get(lbl, 0) + len(ngrams)
        return self

    def _score_token(self, tok: str) -> dict[str, float]:
        """Log-probability score per label for a single token."""
        tok_lower = tok.lower()
        ngrams = _char_ngrams(tok_lower, *self.ngram_range)
        V = len(self.vocab)
        scores = {}
        for lbl in self.labels:
            log_p = self.log_prior[lbl]
            for ng in ngrams:
                cnt = self.ngram_counts.get(ng, {}).get(lbl, 0)
                log_p += math.log((cnt + self.alpha) / (self.total_ngrams[lbl] + self.alpha * V + 1))
            scores[lbl] = log_p
        return scores

    def predict(self, tokens: list[str]) -> list[str]:
        """Predict LID labels for a sequence with context smoothing."""
        if not tokens:
            return []

        CONTEXT_W = 2
        CONTEXT_WT = 0.10

        raw_scores = []
        for tok in tokens:
            if _is_language_independent(tok):
                raw_scores.append({lbl: (1.0 if lbl == 'O' else 0.0) for lbl in self.labels})
            else:
                raw_scores.append(self._score_token(tok))

        labels_out = []
        for i, scores in enumerate(raw_scores):
            # Gather context window votes
            neighbour_votes = Counter()
            for j in range(max(0, i - CONTEXT_W), min(len(raw_scores), i + CONTEXT_W + 1)):
                if j == i:
                    continue
                best_nb = max(raw_scores[j], key=raw_scores[j].get)
                neighbour_votes[best_nb] += 1

            count = sum(neighbour_votes.values())
            if count == 0:
                labels_out.append(max(scores, key=scores.get))
                continue

            # Blend own score (softmax-normalised) with context
            own_max = max(scores.values())
            blended = {}
            for lbl in self.labels:
                own_norm = math.exp(scores[lbl] - own_max)
                ctx_w = neighbour_votes[lbl] / count
                blended[lbl] = (1 - CONTEXT_WT) * own_norm + CONTEXT_WT * ctx_w
            labels_out.append(max(blended, key=blended.get))

        return labels_out

    def save(self, path: str) -> None:
        """Save model to pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'NaiveBayesLID':
        """Load model from pickle."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def build_flat_token_label_lists(
    tweets: list[dict],
    tokens_key: str = 'tokens',
) -> tuple[list[str], list[str]]:
    """
    Flatten tweets into two parallel lists: tokens and mapped LID labels.
    tokens_key selects 'tokens', 'cleaned_tokens', or 'normalised_tokens'.
    """
    tokens_flat = []
    labels_flat = []
    for tweet in tweets:
        for tok, gold_tag in zip(tweet[tokens_key], tweet['tags']):
            lbl = TAG_MAP.get(gold_tag, 'O')
            tokens_flat.append(tok)
            labels_flat.append(lbl)
    return tokens_flat, labels_flat


def predict_tweets(model: NaiveBayesLID, tweets: list[dict], tokens_key: str) -> list[list[str]]:
    """Run model.predict for each tweet; return list of per-tweet label lists."""
    preds = []
    for tweet in tweets:
        preds.append(model.predict(tweet[tokens_key]))
    return preds
