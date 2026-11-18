# src/bilstm_tagger/train.py
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

PAD_TAG = -100


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_epoch(model, batches, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for b in batches:
            words, chars = b["words"].to(device), b["chars"].to(device)
            tags, lengths = b["tags"].to(device), b["lengths"]
            logits = model(words, chars, lengths)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   tags.view(-1), ignore_index=PAD_TAG)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            mask = tags != PAD_TAG
            correct += ((logits.argmax(-1) == tags) & mask).sum().item()
            count += mask.sum().item()
            loss_sum += loss.item()
    return loss_sum / max(len(batches), 1), correct / max(count, 1)


def fit(model, train_batches_fn, dev_batches, device,
        epochs=30, lr=1e-3, patience=4, ckpt="results/bilstm_best.pt"):
    model.to(device)
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, bad = 0.0, 0
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_batches_fn(), device, opt)
        dv_loss, dv_acc = run_epoch(model, dev_batches, device)
        print(f"ep {ep:02d} | train {tr_loss:.3f}/{tr_acc:.4f} | dev {dv_loss:.3f}/{dv_acc:.4f}")
        if dv_acc > best:
            best, bad = dv_acc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print("early stop"); break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return best


@torch.no_grad()
def predict(model, batches, device, id2tag):
    """Return predicted tags as list[list[str]] (same format as the HMM tagger)."""
    model.eval()
    preds = []
    for b in batches:
        logits = model(b["words"].to(device), b["chars"].to(device), b["lengths"])
        ids = logits.argmax(-1).cpu()
        for row, n in zip(ids, b["lengths"]):
            preds.append([id2tag[i] for i in row[:n].tolist()])
    return preds