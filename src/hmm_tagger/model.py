"""Counting, smoothing and the HMMTagger class."""
from collections import Counter

import numpy as np

from .data import UNK_CLASSES, unk_class
from .viterbi import viterbi


def smooth(trans_counts, emit_counts, k_trans, k_emit):
    """Add-k smoothing, returned as log-probabilities (log_A, log_B)."""
    n_tags = emit_counts.shape[0]
    start, end = n_tags, n_tags + 1

    # only legal moves get probability: START -> tag, tag -> tag, tag -> END
    allowed = np.zeros(trans_counts.shape, dtype=bool)
    allowed[:n_tags, :n_tags] = True
    allowed[:n_tags, end] = True
    allowed[start, :n_tags] = True

    A = np.where(allowed, trans_counts + k_trans, 0.0)
    A = A / A.sum(axis=1, keepdims=True).clip(min=1e-12)   # the END row stays all zeros
    B = emit_counts + k_emit
    B = B / B.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore"):
        return np.log(A), np.log(B)


class HMMTagger:
    """First-order HMM POS tagger.

    v1 setting: shape_unk=False.   v2 setting: shape_unk=True (default).
    """

    def __init__(self, lowercase=True, min_freq=2, shape_unk=True, k_trans=1.0, k_emit=0.001):
        self.lowercase = lowercase
        self.min_freq = min_freq
        self.shape_unk = shape_unk
        self.k_trans = k_trans
        self.k_emit = k_emit

    # ---- helpers -------------------------------------------------------
    def _norm(self, word):
        return word.lower() if self.lowercase else word

    def _encode_word(self, raw_word):
        w = self._norm(raw_word)
        if w in self.word2id:
            return self.word2id[w]
        bucket = unk_class(raw_word) if self.shape_unk else "<UNK>"
        return self.word2id[bucket]

    def encode(self, words):
        """List of raw words -> list of vocabulary ids."""
        return [self._encode_word(w) for w in words]

    # ---- training ------------------------------------------------------
    def fit(self, sentences):
        """sentences: list of sentences, each a list of (word, tag) pairs."""
        self.tags = sorted({t for s in sentences for _, t in s})
        self.tag2id = {t: i for i, t in enumerate(self.tags)}
        n = len(self.tags)
        self.n_tags, self.start, self.end = n, n, n + 1

        counts = Counter(self._norm(w) for s in sentences for w, _ in s)
        self.train_words = set(counts)                       # every word form seen in training
        unk_tokens = UNK_CLASSES if self.shape_unk else ["<UNK>"]
        self.vocab = unk_tokens + sorted(w for w, c in counts.items() if c >= self.min_freq)
        self.word2id = {w: i for i, w in enumerate(self.vocab)}

        self.trans_counts = np.zeros((n + 2, n + 2))
        self.emit_counts = np.zeros((n, len(self.vocab)))
        for s in sentences:
            prev = self.start
            for word, tag in s:
                t = self.tag2id[tag]
                self.trans_counts[prev, t] += 1
                self.emit_counts[t, self._encode_word(word)] += 1   # rare words land in UNK buckets
                prev = t
            self.trans_counts[prev, self.end] += 1

        self.log_A, self.log_B = smooth(self.trans_counts, self.emit_counts, self.k_trans, self.k_emit)
        return self

    # ---- prediction ----------------------------------------------------
    def predict_with_score(self, words):
        path, score = viterbi(self.encode(words), self.log_A, self.log_B)
        return [self.tags[i] for i in path], score

    def predict(self, words):
        """Tag one sentence given as a list of words."""
        return self.predict_with_score(words)[0]

    def predict_many(self, sentences):
        """sentences: list of (word, tag) sentences; the gold tags are ignored."""
        return [self.predict([w for w, _ in s]) for s in sentences]