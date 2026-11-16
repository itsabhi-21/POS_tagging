"""One experiment = one command. Examples (from repo root):

  python -m bilstm_tagger.run --subset 200 --epochs 30 --patience 100 --lr 1e-2 --dropout 0 --word_dropout 0
  python -m bilstm_tagger.run --epochs 2 --patience 100
  python -m bilstm_tagger.run --encoder lstm --bidirectional --use_char --seed 1
  python -m bilstm_tagger.run --no-use_char --seed 1          # word-only ablation

Test is touched ONLY when --eval_test is passed (do that once, at the very end).
"""
import argparse
import json
import os
import random

import torch

from hmm_tagger.data import load_ewt
from hmm_tagger.evaluate import gold_tags, token_accuracy, macro_f1, unseen_mask, unknown_word_accuracy

from .dataset import make_batches
from .model import BiLSTMTagger
from .train import fit, predict, set_seed
from .vocab import build_vocabs


def metrics(gold, pred, sents, train_words, labels):
    mask = unseen_mask(sents, train_words)          # same definition as the HMM evaluation
    return {"accuracy": token_accuracy(gold, pred),
            "macro_f1": macro_f1(gold, pred, labels),
            "unknown_accuracy": unknown_word_accuracy(gold, pred, mask),
            "n_unknown_tokens": int(mask.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="lstm", choices=["rnn", "lstm", "gru", "cnn"])
    ap.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_char", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.33)
    ap.add_argument("--word_dropout", type=float, default=0.1)
    ap.add_argument("--subset", type=int, default=0, help="first N train sentences only (sanity check)")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="results/dl")
    ap.add_argument("--eval_test", action="store_true", help="evaluate on TEST (final run only)")
    a = ap.parse_args()

    set_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    data = load_ewt(a.data_dir)
    full_train, dev, test = data["train"], data["dev"], data["test"]
    train = full_train[:a.subset] if a.subset else full_train
    vocabs = build_vocabs(train, tag_sents=full_train)
    train_words = {w.lower() for s in train for w, _ in s}    # same as HMM.train_words
    labels = sorted(vocabs.tag2id)

    direction = "win" if a.encoder == "cnn" else ("bi" if a.bidirectional else "uni")
    name = f"{a.encoder}-{direction}-{'char' if a.use_char else 'word'}-seed{a.seed}"
    ckpt = os.path.join(a.out_dir, f"{name}.pt")
    print(f"run: {name} | device {device} | train {len(train)} sents | vocab {len(vocabs.word2id)} words")

    dev_batches = make_batches(dev, vocabs, a.batch_size)
    make_train = lambda: make_batches(train, vocabs, a.batch_size, shuffle=True,
                                      word_dropout=a.word_dropout, rng=rng)

    model = BiLSTMTagger(len(vocabs.word2id), len(vocabs.char2id), len(vocabs.tag2id),
                         hidden=a.hidden, dropout=a.dropout, use_char=a.use_char,
                         encoder=a.encoder, bidirectional=a.bidirectional)
    best_dev = fit(model, make_train, dev_batches, device, epochs=a.epochs, lr=a.lr,
                   patience=a.patience, ckpt=ckpt)

    result = {"name": name, "args": vars(a), "best_dev_accuracy": best_dev,
              "dev": metrics(gold_tags(dev), predict(model, dev_batches, device, vocabs.id2tag),
                             dev, train_words, labels)}
    if a.eval_test:
        test_batches = make_batches(test, vocabs, a.batch_size)       # no shuffle -> original order
        preds = predict(model, test_batches, device, vocabs.id2tag)
        result["test"] = metrics(gold_tags(test), preds, test, train_words, labels)
        json.dump(preds, open(os.path.join(a.out_dir, f"preds_{name}_test.json"), "w"))
    json.dump(result, open(os.path.join(a.out_dir, f"{name}.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()