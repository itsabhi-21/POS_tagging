import pytest

from hmm_tagger import HMMTagger
from hmm_tagger.evaluate import gold_tags, macro_f1, token_accuracy, unknown_word_accuracy, unseen_mask

pytestmark = pytest.mark.slow


def evaluate(model, sentences):
    gold, pred = gold_tags(sentences), model.predict_many(sentences)
    mask = unseen_mask(sentences, model.train_words)
    return (token_accuracy(gold, pred), macro_f1(gold, pred, model.tags),
            unknown_word_accuracy(gold, pred, mask))


def test_v2_reproduces_published_numbers(ewt):
    m = HMMTagger(shape_unk=True).fit(ewt["train"])
    acc, f1, unk = evaluate(m, ewt["test"])
    assert acc == pytest.approx(0.9224, abs=6e-5)
    assert f1 == pytest.approx(0.8553, abs=6e-5)
    assert unk == pytest.approx(0.7572, abs=6e-5)


def test_v1_reproduces_published_numbers(ewt):
    m = HMMTagger(shape_unk=False).fit(ewt["train"])
    acc, f1, unk = evaluate(m, ewt["test"])
    assert acc == pytest.approx(0.9003, abs=6e-5)
    assert f1 == pytest.approx(0.8304, abs=6e-5)
    assert unk == pytest.approx(0.5351, abs=6e-5)