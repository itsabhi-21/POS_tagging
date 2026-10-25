"""Loading the UD English EWT treebank and the unknown-word bucket rules."""
import os
import re
import urllib.request

from conllu import parse_incr

BASE_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/"
SPLITS = ("train", "dev", "test")


def download_ewt(data_dir="data"):
    """Download the three official EWT files if they are not there yet."""
    os.makedirs(data_dir, exist_ok=True)
    for split in SPLITS:
        fname = f"en_ewt-ud-{split}.conllu"
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(BASE_URL + fname, path)
    return data_dir


def load_conllu(path):
    """Return sentences as lists of (word, upos) pairs, keeping the original casing.

    Multiword tokens (id like 1-2) and empty nodes (id like 8.1) are not real
    tokens, so they are skipped to keep words and tags aligned.
    """
    sentences = []
    with open(path, encoding="utf-8") as f:
        for sent in parse_incr(f):
            pairs = [(tok["form"], tok["upos"]) for tok in sent if isinstance(tok["id"], int)]
            if pairs:
                sentences.append(pairs)
    return sentences


def load_ewt(data_dir="data"):
    """Download (if needed) and load all three splits as a dict."""
    download_ewt(data_dir)
    return {s: load_conllu(os.path.join(data_dir, f"en_ewt-ud-{s}.conllu")) for s in SPLITS}


# Words that look like web addresses or e-mails
URL_RE = re.compile(r"(https?://|www\.|@|\.com\b|\.org\b|\.net\b)", re.I)

UNK_CLASSES = ["<UNK>", "<UNK-SYM>", "<UNK-URL>", "<UNK-NUM>", "<UNK-ALLCAP>", "<UNK-CAP>",
               "<UNK-HYPHEN>", "<UNK-ING>", "<UNK-ED>", "<UNK-LY>", "<UNK-S>"]


def unk_class(raw_word):
    """Put an unseen word into a bucket based on its shape (original casing)."""
    w = raw_word
    if not any(ch.isalnum() for ch in w):
        return "<UNK-SYM>"
    if URL_RE.search(w):
        return "<UNK-URL>"
    if any(ch.isdigit() for ch in w):
        return "<UNK-NUM>"
    if len(w) > 1 and w.isupper():
        return "<UNK-ALLCAP>"
    if w[0].isupper():
        return "<UNK-CAP>"
    if "-" in w:
        return "<UNK-HYPHEN>"
    lw = w.lower()
    if lw.endswith("ing") and len(lw) > 4:
        return "<UNK-ING>"
    if lw.endswith("ed") and len(lw) > 3:
        return "<UNK-ED>"
    if lw.endswith("ly") and len(lw) > 3:
        return "<UNK-LY>"
    if lw.endswith("s") and len(lw) > 3:
        return "<UNK-S>"
    return "<UNK>"