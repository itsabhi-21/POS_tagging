# HMM POS Tagger

A Hidden Markov Model (HMM) and Viterbi-based
Part-of-Speech tagger for the Universal Dependencies
English EWT treebank.

## Features

- HMM-based POS tagging
- Viterbi decoding in log space
- Unknown-word shape buckets
- Token accuracy, macro F1, and unseen-word evaluation
- Automated tests with pytest

## Installation

```bash
python -m pip install -e .
```

## Usage

```python
from hmm_tagger import HMMTagger
from hmm_tagger.data import load_ewt

data = load_ewt("data")
tagger = HMMTagger().fit(data["train"])

print(tagger.predict("The movie was really good".split()))
```

## Tests

```bash
python -m pytest -v
```