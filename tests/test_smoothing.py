import numpy as np

from hmm_tagger import HMMTagger


def test_rows_sum_to_one(toy_sentences):
    m = HMMTagger(min_freq=1, shape_unk=False, k_trans=0.5, k_emit=0.1).fit(toy_sentences)
    assert np.allclose(np.exp(m.log_A[: m.n_tags + 1]).sum(axis=1), 1)   # tag rows and START row
    assert np.allclose(np.exp(m.log_B).sum(axis=1), 1)


def test_no_minus_infinity_in_emissions(toy_sentences):
    m = HMMTagger(min_freq=1, shape_unk=False, k_emit=0.001).fit(toy_sentences)
    assert np.isfinite(m.log_B).all()


def test_zero_smoothing_gives_minus_infinity(toy_sentences):
    m = HMMTagger(min_freq=1, shape_unk=False, k_emit=0.0).fit(toy_sentences)
    assert np.isinf(m.log_B).any()