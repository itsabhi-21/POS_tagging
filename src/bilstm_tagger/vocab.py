"""Vocabularies built from the TRAIN split only."""
from collections import Counter

PAD, UNK = "<PAD>", "<UNK>"
PAD_ID, UNK_ID = 0, 1


class Vocabs:
    def __init__(self, word2id, char2id, tag2id, lowercase=True):
        self.word2id, self.char2id, self.tag2id = word2id, char2id, tag2id
        self.id2tag = {i: t for t, i in tag2id.items()}
        self.lowercase = lowercase

    def norm(self, w):
        return w.lower() if self.lowercase else w

    def word_id(self, w):
        return self.word2id.get(self.norm(w), UNK_ID)

    def char_ids(self, w, max_len=20):
        # chars keep original casing: capitals/digits are clues for unseen words
        return [self.char2id.get(c, UNK_ID) for c in w[:max_len]]


def build_vocabs(train_sents, min_freq=2, lowercase=True, tag_sents=None):
    """min_freq=2 matches the HMM: words seen only once in train become <UNK>."""
    norm = (lambda w: w.lower()) if lowercase else (lambda w: w)
    wc = Counter(norm(w) for s in train_sents for w, _ in s)
    words = [PAD, UNK] + sorted(w for w, c in wc.items() if c >= min_freq)
    chars = [PAD, UNK] + sorted({c for s in train_sents for w, _ in s for c in w})
    # tag set always comes from the FULL train split (matters for small subsets)
    tags = sorted({t for s in (tag_sents or train_sents) for _, t in s})
    return Vocabs({w: i for i, w in enumerate(words)},
                  {c: i for i, c in enumerate(chars)},
                  {t: i for i, t in enumerate(tags)}, lowercase)
