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

# ---------- 4. Learning curve (dev set) ----------
results["learning_curve"] = []
for frac in [0.1, 0.25, 0.5, 1.0]:
    subset = random.Random(SEED).sample(train, int(frac * len(train)))
    row = {"fraction": frac, "train_sentences": len(subset)}
    for name, shape in [("v1", False), ("v2", True)]:
        model = HMMTagger(shape_unk=shape).fit(subset)
        row[name] = evaluate(model.predict_many(dev), dev, model.train_words)["accuracy"]
    results["learning_curve"].append(row)
    print(row)

plt.figure(figsize=(6, 4))
fr = [r["fraction"] * 100 for r in results["learning_curve"]]
plt.plot(fr, [r["v1"] * 100 for r in results["learning_curve"]], "o-", label="HMM v1")
plt.plot(fr, [r["v2"] * 100 for r in results["learning_curve"]], "o-", label="HMM v2")
plt.xlabel("Training data used (%)")
plt.ylabel("Dev accuracy (%)")
plt.title("Learning curve")
plt.legend()
plt.savefig("figures/learning_curve.png", dpi=150)
plt.close()

# ---------- 5. Bootstrap confidence interval for v2 - v1 (test set) ----------
def per_sentence_correct(model):
    correct = np.array([sum(p == g for p, (_, g) in zip(model.predict([w for w, _ in s]), s)) for s in test])
    return correct


lengths = np.array([len(s) for s in test])
c1, c2 = per_sentence_correct(v1), per_sentence_correct(v2)

rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(test), size=(1000, len(test)))               # 1000 resamples of the test sentences
diff = c2[idx].sum(axis=1) / lengths[idx].sum(axis=1) - c1[idx].sum(axis=1) / lengths[idx].sum(axis=1)
lo, hi = np.percentile(diff, [2.5, 97.5])
results["bootstrap_v2_minus_v1"] = {"mean_diff": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
                                    "excludes_zero": bool(lo > 0 or hi < 0)}
print(results["bootstrap_v2_minus_v1"])
