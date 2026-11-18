import pytest

from hmm_tagger.data import UNK_CLASSES, unk_class


@pytest.mark.parametrize("word, bucket", [
    ("!!", "<UNK-SYM>"),
    ("www.example.com", "<UNK-URL>"),
    ("someone@mail", "<UNK-URL>"),
    ("1999", "<UNK-NUM>"),
    ("NASA", "<UNK-ALLCAP>"),
    ("London", "<UNK-CAP>"),
    ("well-known", "<UNK-HYPHEN>"),
    ("running", "<UNK-ING>"),
    ("jumped", "<UNK-ED>"),
    ("quickly", "<UNK-LY>"),
    ("tables", "<UNK-S>"),
    ("xyz", "<UNK>"),
])
def test_unk_class_buckets(word, bucket):
    assert unk_class(word) == bucket


def test_short_words_do_not_match_suffix_rules():
    assert unk_class("ing") == "<UNK>"
    assert unk_class("is") == "<UNK>"


def test_every_bucket_is_a_known_class():
    for w in ["!!", "a.com", "42", "UN", "Bob", "x-y", "going", "liked", "only", "cats", "q"]:
        assert unk_class(w) in UNK_CLASSES
