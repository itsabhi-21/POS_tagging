from hmm_tagger import HMMTagger


def test_toy_counts(toy_sentences):
    m = HMMTagger(min_freq=1, shape_unk=False).fit(toy_sentences)
    n_tokens = sum(len(s) for s in toy_sentences)

    assert m.emit_counts.sum() == n_tokens                              # one emission per token
    assert m.trans_counts.sum() == n_tokens + len(toy_sentences)        # transitions incl. one END per sentence
    assert m.trans_counts[m.start].sum() == len(toy_sentences)          # every sentence starts exactly once
    assert m.trans_counts[:, m.start].sum() == 0                        # nothing moves into START
    assert m.trans_counts[m.end].sum() == 0                             # nothing moves out of END


def test_toy_transition_value(toy_sentences):
    m = HMMTagger(min_freq=1, shape_unk=False).fit(toy_sentences)
    det, noun = m.tag2id["DET"], m.tag2id["NOUN"]
    assert m.trans_counts[det, noun] == 2                               # DET is followed by NOUN twice


def test_real_counts(ewt):
    m = HMMTagger().fit(ewt["train"])
    n_tokens = sum(len(s) for s in ewt["train"])
    assert m.emit_counts.sum() == n_tokens
    assert m.trans_counts.sum() == n_tokens + len(ewt["train"])
    assert len(m.tags) == 17