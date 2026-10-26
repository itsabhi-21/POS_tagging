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

# ---------- 1. Most-frequent-tag baseline ----------
tag_by_word = defaultdict(Counter)
for s in train:
    for w, t in s:
        tag_by_word[w.lower()][t] += 1
overall_tag = Counter(t for s in train for _, t in s).most_common(1)[0][0]    # fallback for unseen words


def mft_predict(sentences):
    return [[tag_by_word[w.lower()].most_common(1)[0][0] if w.lower() in tag_by_word else overall_tag
             for w, _ in s] for s in sentences]


results["mft_baseline"] = {
    "fallback_tag": overall_tag,
    "dev": evaluate(mft_predict(dev), dev, v1.train_words),
    "test": evaluate(mft_predict(test), test, v1.train_words),
}
print("MFT baseline:", results["mft_baseline"])
