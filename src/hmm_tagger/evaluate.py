"""Metrics: token accuracy, macro F1 and unknown-word accuracy."""
import numpy as np
from sklearn.metrics import f1_score


def gold_tags(sentences):
    return [[t for _, t in s] for s in sentences]


def _flat(list_of_lists):
    return np.array([x for xs in list_of_lists for x in xs])


def token_accuracy(gold, pred):
    return float((_flat(gold) == _flat(pred)).mean())


def macro_f1(gold, pred, labels):
    return float(f1_score(_flat(gold), _flat(pred), average="macro", labels=labels, zero_division=0))


def unseen_mask(sentences, train_words, lowercase=True):
    """True for every token whose (lowercased) form never appears in training."""
    norm = (lambda w: w.lower()) if lowercase else (lambda w: w)
    return np.array([norm(w) not in train_words for s in sentences for w, _ in s])


def unknown_word_accuracy(gold, pred, mask):
    return float((_flat(gold) == _flat(pred))[mask].mean())