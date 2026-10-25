"""Log-space Viterbi decoding."""
import numpy as np


def viterbi(word_ids, log_A, log_B):
    """Best tag sequence for one sentence.

    log_A has shape (n_tags + 2, n_tags + 2); the last two rows/columns are START and END.
    log_B has shape (n_tags, vocab_size).
    Returns (list of tag ids, log-probability of that path).
    """
    n_tags = log_B.shape[0]
    start, end = n_tags, n_tags + 1
    T = len(word_ids)
    if T == 0:
        return [], 0.0

    delta = np.full((T, n_tags), -np.inf)      # best score of any path ending in each tag
    back = np.zeros((T, n_tags), dtype=int)    # previous tag on that best path

    delta[0] = log_A[start, :n_tags] + log_B[:, word_ids[0]]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A[:n_tags, :n_tags]   # rows = from tag, cols = to tag
        back[t] = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + log_B[:, word_ids[t]]

    final = delta[-1] + log_A[:n_tags, end]    # the last tag must be able to move to END
    last = int(final.argmax())

    path = [last]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return path[::-1], float(final[last])