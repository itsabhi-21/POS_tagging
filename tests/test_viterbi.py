import itertools

import numpy as np

from hmm_tagger import HMMTagger
from hmm_tagger.viterbi import viterbi


def brute_force(word_ids, log_A, log_B):
    """Try every tag sequence and keep the best one."""
    n = log_B.shape[0]
    start, end = n, n + 1
    best_score, best_path = -np.inf, None
    for path in itertools.product(range(n), repeat=len(word_ids)):
        score = log_A[start, path[0]] + log_B[path[0], word_ids[0]]
        for t in range(1, len(path)):
            score += log_A[path[t - 1], path[t]] + log_B[path[t], word_ids[t]]
        score += log_A[path[-1], end]
        if score > best_score:
            best_score, best_path = score, list(path)
    return best_path, best_score


def test_viterbi_matches_brute_force_on_random_hmm():
    rng = np.random.default_rng(0)
    n_tags, vocab = 3, 5
    log_A = np.log(rng.dirichlet(np.ones(n_tags + 2), size=n_tags + 2))
    log_B = np.log(rng.dirichlet(np.ones(vocab), size=n_tags))
    for _ in range(20):
        ids = rng.integers(0, vocab, size=rng.integers(1, 6)).tolist()
        v_path, v_score = viterbi(ids, log_A, log_B)
        b_path, b_score = brute_force(ids, log_A, log_B)
        assert v_path == b_path
        assert np.isclose(v_score, b_score)


def test_viterbi_matches_brute_force_on_real_model(ewt):
    m = HMMTagger().fit(ewt["train"])
    short = [s for s in ewt["dev"] if len(s) == 3][:5]        # 17**3 = 4913 sequences each
    for s in short:
        ids = m.encode([w for w, _ in s])
        v_path, v_score = viterbi(ids, m.log_A, m.log_B)
        b_path, b_score = brute_force(ids, m.log_A, m.log_B)
        assert v_path == b_path
        assert np.isclose(v_score, b_score)


def test_empty_sentence():
    log_A = np.zeros((5, 5))
    log_B = np.zeros((3, 4))
    assert viterbi([], log_A, log_B) == ([], 0.0)
    