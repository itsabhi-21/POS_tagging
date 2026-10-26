"""Baselines and ablations for the HMM POS tagger.

Run from the repository root:  python experiments/ablations.py
"""
import json
import random
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")                      # lets the script run without a display
import matplotlib.pyplot as plt
import numpy as np

from hmm_tagger import HMMTagger
from hmm_tagger.data import load_ewt
from hmm_tagger.evaluate import gold_tags, token_accuracy, unknown_word_accuracy, unseen_mask

SEED = 42
data = load_ewt("data")
train, dev, test = data["train"], data["dev"], data["test"]
results = {}                               # everything we report ends up in this dict


def evaluate(pred, sentences, train_words):
    """Accuracy overall and on unknown words, for predictions on one split."""
    gold = gold_tags(sentences)
    mask = unseen_mask(sentences, train_words)
    return {"accuracy": token_accuracy(gold, pred), "unknown_accuracy": unknown_word_accuracy(gold, pred, mask)}


v1 = HMMTagger(shape_unk=False).fit(train)     # final settings from the report
v2 = HMMTagger(shape_unk=True).fit(train)

# sanity check: these must match the published numbers (v2 test accuracy 0.9224, v1 0.9003)
for name, model in [("v1", v1), ("v2", v2)]:
    pred = model.predict_many(test)
    print(name, "test", evaluate(pred, test, model.train_words))
