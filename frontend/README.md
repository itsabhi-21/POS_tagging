# 🏷️ POS Tagging Interactive Streamlit Frontend

An interactive, multi-tab Streamlit dashboard to showcase the entire Universal Dependencies Part-of-Speech Tagging project findings: comparing baseline statistical HMMs (v1 & v2) against our Deep Learning BiLSTM + Character-level CNN tagger.

---

## 🚀 Quickstart: Running the App

From the repository root, run:

```bash
# Using virtual environment
.venv/bin/streamlit run frontend/app.py
```

Or activate your environment first:
```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🌟 App Features & Tabs

1. **🎯 Live Interactive POS Tagger**:
   - Type any custom English sentence or load challenge sentences (e.g., rare words like *"unicyclist"*, polysemy with *"plan to back"*, participial adjectives like *"running water"*).
   - Real-time side-by-side comparison of **HMM v1**, **HMM v2**, and **BiLSTM + Char-CNN** with color-coded UPOS badge chips.
   - Detailed token breakdown table displaying vocabulary status (Known vs OOV), model agreement, and tag definitions.

2. **📊 Benchmark Leaderboard**:
   - Comprehensive comparative results on the UD English EWT test split (25,094 tokens).
   - High-level metric highlight cards for Token Accuracy, Macro F1, and Out-of-Vocabulary (OOV) accuracy.
   - Interactive bar chart visualizer.

3. **🔬 Statistical Ablations & Rigorous Validation**:
   - The 5 empirical ablations:
     - Most Frequent Tag (MFT) baseline (86.16% vs 28.59% OOV).
     - Greedy vs. Viterbi Dynamic Programming (+2.33 pp gain).
     - Lidstone smoothing sensitivity analysis ($k = 10^{-5}$ to $1.0$).
     - Data efficiency learning curve across 10% to 100% training subsets.
     - 1,000-sample bootstrap statistical hypothesis testing ($p < 0.001$).
     - Random seed stability boxplots (seeds 1 to 5: 95.33% ± 0.17%).

4. **🖼️ Publication Figure Gallery (300 DPI)**:
   - High-resolution display of all 12 scientific figures with detailed research explanations and key takeaways.
   - Filter figures by category (Overall Benchmarks, Architecture Deep-Dive, Training Dynamics, Error Analysis, Scientific Rigor).

5. **👥 Team & Architecture Blueprint**:
   - Neural network architectural flow diagram (Word 100d + Char-CNN 30d $\rightarrow$ 2-layer BiLSTM 512d $\rightarrow$ 17 UPOS Linear Head).
   - 5-member team workload matrix (Abhishek Rana, Abhinav, Members 3, 4, 5).
   - Direct download links for IEEE Phase 2 Final Report (PDF & DOCX) and Presentation slides.
