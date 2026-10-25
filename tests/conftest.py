import pytest

from hmm_tagger.data import load_ewt


@pytest.fixture(scope="session")
def ewt():
    """The real train/dev/test data (downloaded once into data/)."""
    return load_ewt("data")


@pytest.fixture
def toy_sentences():
    return [
        [("The", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
        [("A", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
        [("Dogs", "NOUN"), ("bark", "VERB")],
    ]