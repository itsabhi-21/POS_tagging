import math

import numpy as np

from hmm_tagger.evaluate import gold_tags, macro_f1, token_accuracy, unknown_word_accuracy, unseen_mask


def test_token_accuracy():
    gold = [["DET", "NOUN"], ["VERB"]]
    pred = [["DET", "VERB"], ["VERB"]]
    assert token_accuracy(gold, pred) == 2 / 3


def test_macro_f1_perfect():
    gold = [["DET", "NOUN", "VERB"]]
    assert macro_f1(gold, gold, ["DET", "NOUN", "VERB"]) == 1.0


def test_unseen_mask_is_case_insensitive(toy_sentences):
    train_words = {"the", "dog"}
    mask = unseen_mask(toy_sentences[:1], train_words)
    assert mask.tolist() == [False, False, True]


def test_unknown_word_accuracy():
    gold = [["DET", "NOUN", "VERB"]]
    pred = [["DET", "NOUN", "NOUN"]]
    mask = np.array([False, True, True])
    assert unknown_word_accuracy(gold, pred, mask) == 0.5


def test_unknown_word_accuracy_with_no_unknown_tokens():
    gold = [["DET", "NOUN"]]
    assert math.isnan(unknown_word_accuracy(gold, gold, np.array([False, False])))


def test_gold_tags(toy_sentences):
    assert gold_tags(toy_sentences)[0] == ["DET", "NOUN", "VERB"]
