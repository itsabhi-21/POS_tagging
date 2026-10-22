# Part-of-Speech Tagging with Hidden Markov Models

**Project 16: Part of Speech Tagging via Hidden Markov Models and Bidirectional LSTMs (AML phase)**

A part-of-speech (POS) tagger for English web text built from scratch with a **Hidden Markov Model (HMM)**: probabilities estimated by counting, add-k smoothing, and **Viterbi decoding** in log space. It is evaluated on the UD English EWT treebank and serves as the baseline for the BiLSTM tagger of the next phase (to be added to this repository).

---

## Highlights

- Complete HMM + Viterbi tagger using only NumPy (no tagging library).
- Two versions are compared: **v1** with a single unknown-word token, and **v2** with 11 shape-based unknown-word buckets.
- v2 improves unknown-word accuracy from **53.5% to 75.7%** and overall accuracy from **90.0% to 92.2%** on the test set.
- All settings are chosen on the development set; the test set is used only to report final numbers.
- Full evaluation: token accuracy, macro F1, unknown-word accuracy, per-tag error analysis, confusion matrix and decoded sentence examples.

## Results (test set)

| Metric | HMM v1 (single `<UNK>`) | HMM v2 (shape-based `<UNK>`) |
|---|---|---|
| Token accuracy | 90.03% | **92.24%** |
| Macro F1 | 83.04% | **85.53%** |
| Known-word accuracy | 92.99% | 93.58% |
| Unknown-word accuracy | 53.51% | **75.72%** |

What changed between v1 and v2:

- The gain comes almost entirely from **unknown words** (about 7.5% of test tokens never appear in training).
- Biggest per-tag F1 gains: **NUM** 0.711 to 0.918 and **PROPN** 0.743 to 0.848.
- The rare tag **X** (42 test tokens) stays almost unsolved (F1 about 0.07). It affects macro F1 much more than accuracy.
- Remaining confusions such as VERB vs AUX and NOUN vs PROPN involve mostly ambiguity that a first-order HMM cannot resolve. This is the motivation for the BiLSTM phase.

## Repository structure

```
.
├── notebooks/
│   └── hmm_pos_tagger.ipynb          # full pipeline, runs top to bottom
├── results/
│   ├── hmm_results.json              # settings, test metrics, per-tag F1
│   └── hmm_test_predictions.json     # gold tags and HMM v2 predictions for every test sentence
├── report/
│   └── AML_Phase1_Final_Report.docx  # written report with full analysis
├── requirements.txt
└── README.md
```

## Getting started

**Google Colab (easiest):** click the *Open in Colab* badge above and choose *Runtime > Run all*. No GPU is needed, and the dataset is downloaded automatically.

**Locally:**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
jupyter notebook notebooks/hmm_pos_tagger.ipynb
```

On the first run the notebook downloads the three UD English EWT files into a `data/` folder (not stored in this repository) and writes `hmm_results.json` and `hmm_test_predictions.json` next to the notebook.

## What the notebook does

1. **Data loading.** Reads the official train/dev/test `.conllu` files, keeps normal tokens only (multiword tokens and empty nodes are skipped so words and tags stay aligned), and uses the 17 UPOS tags.
2. **Vocabulary.** Built from the training split only. Rare words become unknown-word tokens.
3. **Counting.** Transition counts (with `<START>` and `<END>` states) and emission counts.
4. **Smoothing.** Add-k smoothing with separate values for transitions and emissions, tuned on the dev set.
5. **Viterbi decoding.** Log-space dynamic programming with backpointers.
6. **Evaluation.** Dev tuning, then a single test-set evaluation with token accuracy, macro F1, known/unknown-word accuracy, per-tag report, top confusions, confusion matrix and example sentences.
7. **HMM v2.** Replaces the single `<UNK>` with shape-based buckets, compares settings on dev, and re-evaluates on test.
8. **Export.** Saves the results files used for the later BiLSTM comparison.

## Method in brief

**Model.** A first-order HMM. Hidden states are POS tags and observations are words. The model is defined by three tables: how a sentence starts, tag-to-tag transitions, and tag-to-word emissions. All three are estimated by counting, because every training word already has a gold tag.

**Decoding.** The Viterbi algorithm finds the single best tag sequence for a sentence in time proportional to N² x T (N = 17 tags, T = number of words), instead of trying all N^T sequences. All computation uses log-probabilities to avoid underflow.

**Unknown words.**

| Version | Handling |
|---|---|
| v1 | Words seen fewer than twice in training, and any unseen test word, map to one `<UNK>` token. |
| v2 | `<UNK>` is split into 11 buckets by word shape: symbols, URL-like, contains a digit, ALLCAPS, Capitalised, hyphenated, and the endings -ing, -ed, -ly, -s, plus a default bucket. Each bucket learns its own emission distribution. |

**Final settings** (chosen on dev):

| Setting | v1 | v2 |
|---|---|---|
| Lowercase words | yes | yes |
| Rarity threshold (`min_freq`) | 2 | 2 |
| `k_trans` | 1.0 | 1.0 |
| `k_emit` | 0.001 | 0.001 |

## Limitations

- **Limited context:** each tag depends only on the previous tag, and each word only on its own tag.
- **Hand-written bucket rules** reflect English spelling and the style of this corpus.
- **Test-set exposure:** the idea of the shape buckets came from error analysis of v1, which was done on the test set. All choices among settings were made on dev, and the dev and test gains are similar, but the v2 test number may be slightly optimistic.
- **Single split:** results come from the official split only.

## Data

UD English EWT (release v2.18), official train/dev/test split, licensed CC BY-SA 4.0.
https://github.com/UniversalDependencies/UD_English-EWT

The dataset files are not included in this repository; the notebook downloads them.

## Roadmap

- [x] AML phase: HMM + Viterbi baseline, evaluation and report
- [ ] DL phase: BiLSTM tagger with word and character or subword embeddings
- [ ] Fair comparison of HMM and BiLSTM on the same split, tags and metrics

## References

- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.
- Huang, Z., Xu, W., and Yu, K. (2015). Bidirectional LSTM-CRF models for sequence tagging. arXiv:1508.01991 (for the DL phase).

## Team

Abhinav Choudhary 
Abhishek Rana
Varun Sharma
Mishti Sharma
Lakshya Yadav
