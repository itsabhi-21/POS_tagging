# Part-of-Speech Tagging: HMM Baseline vs. Bidirectional LSTM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Universal Dependencies](https://img.shields.io/badge/UD-English_EWT_v2.18-green.svg)](https://github.com/UniversalDependencies/UD_English-EWT)

A controlled comparative study of sequence tagging algorithms on the Universal Dependencies (UD) English Web Treebank (EWT). This project evaluates and contrasts:
1. **Phase 1 (Applied Machine Learning):** A first-order generative **Hidden Markov Model (HMM)** with Laplace emission smoothing and orthographic shape buckets, decoded globally via **Viterbi** in log-space.
2. **Phase 2 (Deep Learning):** A discriminative **Bidirectional LSTM (BiLSTM)** tagger combining learned word embeddings with a character-level **1D Convolutional Neural Network (CNN)** for automated subword morphology extraction.

Both phases share an identical evaluation protocol: the same data splits, identical 17 Universal POS (UPOS) tags, the same definition of an unknown word, and unified scoring functions.

---

## Benchmark Results (UD English EWT Test Set)

Evaluated across **25,094 tokens** (including **1,882 unknown tokens**, 7.50% OOV rate):

| Model Architecture | Token Accuracy (%) | Macro F1 (%) | Unknown (OOV) Acc (%) | Known Words Acc (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Most-Frequent-Tag (MFT)** | 86.16% | 76.12% | 28.59% | 90.83% |
| **HMM v1 (No Shape Buckets)** | 90.03% | 81.65% | 53.51% | 92.99% |
| **HMM v2 + Viterbi (Phase 1 Baseline)** | 92.24% | 85.53% | 75.72% | 93.58% |
| **BiLSTM (Word-Only)** | 92.32 ± 0.17% | 86.56 ± 0.21% | 55.15 ± 0.72% | ≈ 95.33% |
| **BiLSTM (Word + Char CNN, Final)** | **95.33 ± 0.17%** | **91.57 ± 0.42%** | **81.40 ± 0.66%** | **≈ 96.46%** |

> *BiLSTM metrics report mean ± standard deviation over 5 random seeds (seeds 1–5).*

### Key Empirical Findings
- **Overall Improvement:** The word + character BiLSTM outperforms the tuned HMM baseline by **+3.09 percentage points** in token accuracy, **+6.04 points** in macro F1, and **+5.68 points** on unknown words.
- **The Ablation Paradox:** A word-only BiLSTM matches the HMM in aggregate accuracy (92.32% vs. 92.24%), but masks two opposing shifts: bidirectional context reduces known-word errors by ~406 tokens, while the absence of subword features causes unknown-word errors to nearly double (457 → 844).
- **The Character CNN Advantage:** Adding the 1D character CNN ($k=3$, 50 filters) recovers roughly **+26.25 percentage points** on unknown tokens without manual feature engineering, while shaving another 262 known-word errors by recovering casing information.
- **Error Reduction on Syntax:** Bidirectional lookahead eliminates common HMM function-word confusions such as `VERB → AUX` (104 down to ≤32) and `SCONJ → ADP` (63 down to ≤32). The remaining error budget is dominated by nominal capitalization ambiguity (`PROPN ↔ NOUN`).

---

## Repository Structure

```text
POS_tagging/
├── data/                               # UD English EWT treebank splits (.conllu)
│   ├── en_ewt-ud-train.conllu
│   ├── en_ewt-ud-dev.conllu
│   └── en_ewt-ud-test.conllu
├── docs/                               # Detailed ablation analysis & guides
│   └── ablations.md
├── experiments/                        # Reproducible ablation experiment scripts
│   ├── POS_tagger/
│   │   └── ablations.py                # HMM baselines, smoothing, learning curves
│   └── BiLSTM/
│       └── bilstm_ablations.py         # BiLSTM evaluations, per-tag F1, error analysis
├── figures/                            # High-resolution (300 DPI) evaluation plots
│   ├── dl_model_comparison_overview.png
│   ├── dl_per_tag_f1_comparison.png
│   ├── dl_error_budget_decomposition.png
│   ├── dl_top_confusions_comparison.png
│   ├── dl_confusion_hmm_vs_bilstm.png
│   ├── dl_hyperparameter_grid_search.png
│   ├── dl_seed_stability_boxplot.png
│   ├── bilstm_learning_curves_early_stopping.png
│   ├── bilstm_accuracy_vs_sentence_length.png
│   ├── bilstm_architecture_ladder.png
│   ├── bilstm_char_cnn_morphology_breakdown.png
│   └── bilstm_syntactic_cluster_pca.png
├── notebooks/                          # Interactive experiment notebooks
│   ├── HMM_POS_tagger.ipynb            # Phase 1 notebook
│   └── DL_Project.ipynb                # Phase 2 notebook (BiLSTM training & tuning)
├── reports/                            # Final project reports
│   ├── AML_Phase1_Final_Report.docx    # Phase 1 report
│   ├── DL_Phase2_Final_Report.docx     # Phase 2 report (IEEE format, editable)
│   └── DL_Phase2_Final_Report.pdf      # Phase 2 report (IEEE publication format)
├── results/                            # Serialized quantitative metrics (JSON)
│   ├── ablations.json                  # Phase 1 results
│   └── dl_ablations.json               # Phase 2 results
├── src/
│   ├── hmm_tagger/                     # Modular HMM package
│   │   ├── model.py                    # HMM transition & emission estimation
│   │   ├── viterbi.py                  # Log-space dynamic programming decoder
│   │   ├── evaluate.py                 # Token acc, macro F1, OOV masks
│   │   └── data.py                     # CoNLL-U parsing and caching
│   └── bilstm_tagger/                  # Modular BiLSTM package
│       ├── model.py                    # BiLSTM + Char-CNN PyTorch architecture
│       └── train.py                    # Training loop with word dropout & early stopping
├── tests/                              # Pytest test suite
└── pyproject.toml
```

---

## Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone git@github.com:itsabhi-21/POS_tagging.git
cd POS_tagging

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

To run deep learning experiments, install PyTorch:
```bash
pip install torch
```

---

## Usage

### 1. Using the HMM + Viterbi Tagger (Phase 1)
```python
from hmm_tagger import HMMTagger
from hmm_tagger.data import load_ewt

# Load data and fit model with shape-based unknown word handling
data = load_ewt("data")
tagger = HMMTagger(shape_unk=True).fit(data["train"])

# Tag an input sentence
sentence = "The movie was really good".split()
tags = tagger.predict(sentence)
print(list(zip(sentence, tags)))
# Output: [('The', 'DET'), ('movie', 'NOUN'), ('was', 'AUX'), ('really', 'ADV'), ('good', 'ADJ')]
```

### 2. Using the BiLSTM Tagger (Phase 2)
```python
import torch
from bilstm_tagger.model import BiLSTMTagger

# Initialize model (word embeddings + character-level 1D CNN)
model = BiLSTMTagger(
    n_words=8867,
    n_chars=110,
    n_tags=17,
    word_dim=100,
    char_dim=30,
    char_filters=50,
    hidden=256,
    dropout=0.33,
    use_char=True
)
```

---

## Reproducing the Experiments

### Run Phase 1 HMM Ablations
Reproduces the Most-Frequent-Tag baseline, Greedy vs. Viterbi decoding, Laplace smoothing sweeps, and dataset fraction learning curves:
```bash
python experiments/POS_tagger/ablations.py
```
*Outputs: `results/ablations.json`, `figures/ablation_summary.png`, `figures/learning_curve.png`.*

### Run Phase 2 BiLSTM Comparative Evaluations
Evaluates the HMM v2 baseline against the BiLSTM architectures, computes per-tag F1 deltas across all 17 UPOS categories, decomposes error budgets, and extracts confusion distributions:
```bash
python experiments/BiLSTM/bilstm_ablations.py
```
*Outputs: `results/dl_ablations.json`.*

### Running Automated Tests
```bash
pytest -v
```

---

## Team & Course Information
- **Project 16:** Part-of-Speech Tagging via Hidden Markov Models and Bidirectional LSTMs
- **Team Members:** Abhinav Choudhary, Abhishek Rana, Varun Sharma, Lakshya Yadav, and Misthi Sharma
- **Course:** Deep Learning (DL) & Applied Machine Learning (AML) — November 2026
