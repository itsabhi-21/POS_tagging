"""Sentences -> padded tensor batches (dicts with words, chars, tags, lengths)."""
import random
import torch

from .vocab import PAD_ID, UNK_ID
from .train import PAD_TAG


def make_batches(sents, vocabs, batch_size=32, shuffle=False, word_dropout=0.0,
                 max_word_len=20, rng=None):
    """shuffle=False keeps ORIGINAL order (needed for dev/test predictions).
    word_dropout: chance of replacing a word id with <UNK> (train only). Chars stay intact."""
    rng = rng or random
    order = list(range(len(sents)))
    if shuffle:
        rng.shuffle(order)
    batches = []
    for i in range(0, len(order), batch_size):
        chunk = [sents[j] for j in order[i:i + batch_size]]
        B = len(chunk)
        T = max(len(s) for s in chunk)
        L = min(max(len(w) for s in chunk for w, _ in s), max_word_len)
        words = torch.full((B, T), PAD_ID, dtype=torch.long)
        chars = torch.full((B, T, L), PAD_ID, dtype=torch.long)
        tags = torch.full((B, T), PAD_TAG, dtype=torch.long)
        lengths = torch.tensor([len(s) for s in chunk], dtype=torch.long)
        for b, s in enumerate(chunk):
            for t, (w, tag) in enumerate(s):
                wid = vocabs.word_id(w)
                if word_dropout > 0 and rng.random() < word_dropout:
                    wid = UNK_ID
                words[b, t] = wid
                cids = vocabs.char_ids(w, max_word_len)
                chars[b, t, :len(cids)] = torch.tensor(cids)
                tags[b, t] = vocabs.tag2id[tag]
        batches.append({"words": words, "chars": chars, "tags": tags, "lengths": lengths})
    return batches