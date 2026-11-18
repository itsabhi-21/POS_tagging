import pytest
import torch
from bilstm_tagger.vocab import build_vocabs, PAD, UNK, PAD_ID, UNK_ID
from bilstm_tagger.dataset import make_batches
from bilstm_tagger.train import PAD_TAG
from bilstm_tagger.model import BiLSTMTagger


@pytest.fixture
def sample_sents():
    return [
        [("The", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
        [("A", "DET"), ("cat", "NOUN"), ("meowed", "VERB"), ("loudly", "ADV"), (".", "PUNCT")],
        [("The", "DET"), ("cat", "NOUN"), ("ran", "VERB")],
    ]


def test_vocab(sample_sents):
    """Test vocabulary construction: min_freq threshold, PAD/UNK indices, and casing."""
    vocabs = build_vocabs(sample_sents, min_freq=2, lowercase=True)

    # 1. Special token indices
    assert vocabs.word2id[PAD] == PAD_ID
    assert vocabs.word2id[UNK] == UNK_ID
    assert vocabs.char2id[PAD] == PAD_ID
    assert vocabs.char2id[UNK] == UNK_ID

    # 2. Rarity threshold (min_freq=2)
    # 'the' occurs twice, 'cat' occurs twice -> must be in vocab
    assert "the" in vocabs.word2id
    assert "cat" in vocabs.word2id
    # 'dog', 'barks', 'meowed', 'loudly' occur only once -> must map to UNK_ID
    assert "dog" not in vocabs.word2id
    assert vocabs.word_id("dog") == UNK_ID
    assert vocabs.word_id("meowed") == UNK_ID

    # 3. Lowercase normalization for words
    assert vocabs.word_id("The") == vocabs.word_id("the")
    assert vocabs.word_id("CAT") == vocabs.word_id("cat")

    # 4. Character vocabulary preserves casing
    assert "T" in vocabs.char2id
    assert "t" in vocabs.char2id
    assert vocabs.char2id["T"] != vocabs.char2id["t"]

    # 5. Char ID length truncation
    long_word = "supercalifragilisticexpialidocious"
    cids = vocabs.char_ids(long_word, max_len=20)
    assert len(cids) == 20

    # 6. Tag mappings
    assert len(vocabs.tag2id) == len(vocabs.id2tag)
    assert vocabs.id2tag[vocabs.tag2id["NOUN"]] == "NOUN"


def test_batch_alignment(sample_sents):
    """Test batch construction: shape alignment, PAD_TAG masking, and word dropout."""
    vocabs = build_vocabs(sample_sents, min_freq=1, lowercase=True)
    batches = make_batches(sample_sents, vocabs, batch_size=2, shuffle=False)

    assert len(batches) == 2
    b1 = batches[0]

    # Batch 1 has 2 sentences of length 3 and 5
    assert b1["words"].shape == (2, 5)
    assert b1["tags"].shape == (2, 5)
    assert b1["lengths"].tolist() == [3, 5]

    # Sentence 1 has length 3, so positions 3 and 4 must be PAD_TAG (-100)
    assert (b1["tags"][0, 3:] == PAD_TAG).all()
    # Sentence 2 has length 5, so none should be PAD_TAG
    assert (b1["tags"][1] != PAD_TAG).all()

    # Padded word indices must be PAD_ID (0)
    assert (b1["words"][0, 3:] == PAD_ID).all()

    # Word dropout replaces word id with UNK_ID but keeps character ids intact
    batches_drop = make_batches(sample_sents, vocabs, batch_size=3, shuffle=False, word_dropout=1.0)
    b_drop = batches_drop[0]
    # At word_dropout=1.0, every real word is UNK_ID
    assert (b_drop["words"][0, :3] == UNK_ID).all()
    # But characters must NOT be all UNK_ID / PAD_ID
    assert (b_drop["chars"][0, 0] != PAD_ID).any()


def test_padding_invariance():
    """Test padding invariance: altering padded positions must not alter real-token logits."""
    torch.manual_seed(42)
    n_words, n_chars, n_tags = 50, 40, 5
    model = BiLSTMTagger(n_words=n_words, n_chars=n_chars, n_tags=n_tags,
                         word_dim=32, char_dim=16, char_filters=16,
                         hidden=32, dropout=0.0, use_char=True, encoder="lstm")
    model.eval()

    # Batch of 2 sentences: length 3 and length 5
    lengths = torch.tensor([3, 5], dtype=torch.long)
    words = torch.tensor([
        [10, 20, 30, 0, 0],
        [5, 15, 25, 35, 45]
    ], dtype=torch.long)
    chars = torch.randint(1, n_chars, (2, 5, 10))
    chars[0, 3:] = PAD_ID

    with torch.no_grad():
        orig_logits = model(words, chars, lengths)

    # Perturb the padding positions of sentence 0 (change to arbitrary non-zero tokens)
    words_perturbed = words.clone()
    words_perturbed[0, 3:] = torch.tensor([42, 43])
    chars_perturbed = chars.clone()
    chars_perturbed[0, 3:] = torch.randint(1, n_chars, (2, 10))

    with torch.no_grad():
        perturbed_logits = model(words_perturbed, chars_perturbed, lengths)

    # Real tokens (positions 0, 1, 2 of sentence 0) must remain strictly invariant
    diff_real = (orig_logits[0, :3] - perturbed_logits[0, :3]).abs().max().item()
    assert diff_real < 1e-5, f"Padding affected real tokens! Max diff: {diff_real}"

    # Sentence 1 (which had no padding) must be identical
    diff_sent1 = (orig_logits[1] - perturbed_logits[1]).abs().max().item()
    assert diff_sent1 < 1e-5
