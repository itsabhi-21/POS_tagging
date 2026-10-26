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

# ---------- 2. Greedy decoding vs Viterbi ----------
def greedy_predict(model, sentences):
    """Left to right: pick the best tag at each word given only the previous predicted tag."""
    out = []
    for s in sentences:
        ids = model.encode([w for w, _ in s])
        prev, tags_out = model.start, []
        for wid in ids:
            t = int((model.log_A[prev, :model.n_tags] + model.log_B[:, wid]).argmax())
            tags_out.append(model.tags[t])
            prev = t
        out.append(tags_out)
    return out


results["greedy_vs_viterbi"] = {}
for name, model in [("v1", v1), ("v2", v2)]:
    results["greedy_vs_viterbi"][name] = {
        "greedy_dev": evaluate(greedy_predict(model, dev), dev, model.train_words),
        "viterbi_dev": evaluate(model.predict_many(dev), dev, model.train_words),
        "greedy_test": evaluate(greedy_predict(model, test), test, model.train_words),
        "viterbi_test": evaluate(model.predict_many(test), test, model.train_words),
    }
    print(name, "greedy vs Viterbi:", results["greedy_vs_viterbi"][name])

# ---------- 3. Smoothing ablation (dev set) ----------
results["smoothing_ablation"] = []
for k in [0, 1e-6, 1e-3, 1e-1, 1, 10]:
    model = HMMTagger(shape_unk=True, k_emit=k).fit(train)
    scored = [model.predict_with_score([w for w, _ in s]) for s in dev]
    pred = [tags for tags, _ in scored]
    no_path = sum(np.isneginf(score) for _, score in scored)     # sentences where every path had probability 0
    row = {"k_emit": k, "sentences_without_valid_path": int(no_path), **evaluate(pred, dev, model.train_words)}
    results["smoothing_ablation"].append(row)
    print(row)
