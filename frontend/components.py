"""UI components, color palettes, and data loading helpers for the POS Tagging Streamlit App."""

import json
import os
from pathlib import Path
import streamlit as st

# Universal Dependencies 17 UPOS Tag definitions & color palette
TAG_COLORS = {
    "NOUN": "#2563EB",   # Blue
    "PROPN": "#DC2626",  # Red
    "VERB": "#16A34A",   # Green
    "ADJ": "#9333EA",    # Purple
    "ADV": "#D97706",    # Amber
    "PRON": "#0284C7",   # Sky blue
    "DET": "#0D9488",    # Teal
    "ADP": "#4F46E5",    # Indigo
    "NUM": "#059669",    # Emerald
    "AUX": "#0891B2",    # Cyan
    "CCONJ": "#7C3AED",  # Violet
    "SCONJ": "#65A30D",  # Lime
    "PART": "#C026D3",   # Fuchsia
    "INTJ": "#E11D48",   # Rose
    "PUNCT": "#64748B",  # Slate
    "SYM": "#EA580C",    # Orange
    "X": "#475569",      # Stone
}

TAG_DESCRIPTIONS = {
    "NOUN": "Noun (common: car, table, science)",
    "PROPN": "Proper Noun (named entity: California, Google)",
    "VERB": "Verb (main verb: run, eat, predict)",
    "ADJ": "Adjective (modifier: big, green, running)",
    "ADV": "Adverb (modifier: quickly, strictly, very)",
    "PRON": "Pronoun (reference: he, she, they, it)",
    "DET": "Determiner (article/demonstrative: the, a, this)",
    "ADP": "Adposition / Preposition (in, on, under, with)",
    "NUM": "Numeral (cardinal number: 1, 42, 2024)",
    "AUX": "Auxiliary Verb (is, was, has, will, can)",
    "CCONJ": "Coordinating Conjunction (and, but, or)",
    "SCONJ": "Subordinating Conjunction (if, because, that)",
    "PART": "Particle (possessive 's, not, phrasal verb particle)",
    "INTJ": "Interjection (oh, wow, hey, yes)",
    "PUNCT": "Punctuation symbol (. , ! ? \" -)",
    "SYM": "Symbol ($, %, +, =)",
    "X": "Other / Foreign word / Unclassified",
}

SAMPLE_SENTENCES = {
    "1. Morphological OOV (Rare Words)": "He photographed the unicyclist in the park .",
    "2. Contextual Polysemy (plan / back)": "The government will plan to back the agreement .",
    "3. Participial Modifiers (running / freezing)": "The running water was freezing cold .",
    "4. Named Entity Casing Disambiguation": "Apple announced its new product in California yesterday .",
    "5. Complex Structure & Compounds": "State-of-the-art deep learning models outperform traditional baselines significantly .",
}


def render_tag_badge(word: str, tag: str) -> str:
    """Render an HTML badge chip for a tagged word."""
    color = TAG_COLORS.get(tag, "#475569")
    desc = TAG_DESCRIPTIONS.get(tag, tag)
    return (
        f'<span title="{desc}" style="'
        f'display: inline-block; margin: 3px 4px; padding: 4px 10px; '
        f'background-color: {color}15; border: 1.5px solid {color}; '
        f'border-radius: 8px; font-family: ui-monospace, monospace; font-size: 0.95rem;">'
        f'<strong style="color: #1E293B;">{word}</strong> '
        f'<span style="background-color: {color}; color: #FFFFFF; '
        f'padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; '
        f'margin-left: 4px; letter-spacing: 0.5px;">{tag}</span>'
        f'</span>'
    )


def render_tag_sentence_html(words: list[str], tags: list[str]) -> str:
    """Render an entire tagged sentence as HTML badges."""
    badges = [render_tag_badge(w, t) for w, t in zip(words, tags)]
    return (
        '<div style="line-height: 2.2; padding: 14px; background: #F8FAFC; '
        'border-radius: 12px; border: 1px solid #E2E8F0; margin: 10px 0;">'
        + "".join(badges) +
        '</div>'
    )


@st.cache_data
def load_ablation_results():
    """Load JSON files containing Phase 1 and Phase 2 ablation numbers."""
    base_dir = Path(__file__).resolve().parent.parent
    ablations_path = base_dir / "results" / "ablations.json"
    dl_ablations_path = base_dir / "results" / "dl_ablations.json"

    data = {"hmm": {}, "dl": {}}
    if ablations_path.exists():
        with open(ablations_path, "r") as f:
            data["hmm"] = json.load(f)

    if dl_ablations_path.exists():
        with open(dl_ablations_path, "r") as f:
            data["dl"] = json.load(f)

    return data


def get_figure_paths():
    """Return dictionary of available figures in the figures directory."""
    base_dir = Path(__file__).resolve().parent.parent
    fig_dir = base_dir / "figures"
    
    figures = {
        "Model Comparison Overview": {
            "path": fig_dir / "dl_model_comparison_overview.png",
            "category": "Overall Benchmarks",
            "description": "Comprehensive comparison of Token Accuracy, Macro F1, and Out-of-Vocabulary (OOV) Accuracy across all four model architectures on the UD English EWT test split (25,094 tokens)."
        },
        "Architecture Performance Ladder": {
            "path": fig_dir / "bilstm_architecture_ladder.png",
            "category": "Architecture Deep-Dive",
            "description": "Step-by-step performance ladder illustrating the incremental gain: HMM v1 (90.03%) -> HMM v2 (92.24%) -> BiLSTM Word-Only (92.32%) -> BiLSTM + Char-CNN (95.33%)."
        },
        "Learning Curves & Early Stopping": {
            "path": fig_dir / "bilstm_learning_curves_early_stopping.png",
            "category": "Training Dynamics",
            "description": "Training and validation loss alongside Macro F1 curves over epochs, showing convergence and early stopping rollback to prevent overfitting."
        },
        "Accuracy vs. Sentence Length": {
            "path": fig_dir / "bilstm_accuracy_vs_sentence_length.png",
            "category": "Linguistic Evaluation",
            "description": "Accuracy evaluated across sentence length bins (1-10, 11-20, 21-30, 31+ tokens). Shows BiLSTM retains high accuracy (>94.8%) across long compound sentences."
        },
        "Confusion Matrix: HMM vs. BiLSTM": {
            "path": fig_dir / "dl_confusion_hmm_vs_bilstm.png",
            "category": "Error Analysis",
            "description": "Side-by-side normalized confusion matrices across all 17 UPOS tags, illustrating sharp reduction in off-diagonal confusions for the BiLSTM tagger."
        },
        "Per-Tag F1 Score Comparison": {
            "path": fig_dir / "dl_per_tag_f1_comparison.png",
            "category": "Overall Benchmarks",
            "description": "Detailed F1 score breakdown across all 17 Universal POS tags comparing the HMM baseline against the BiLSTM + Char-CNN tagger."
        },
        "Error Budget Decomposition": {
            "path": fig_dir / "dl_error_budget_decomposition.png",
            "category": "Error Analysis",
            "description": "Decomposition of error sources: shows how the proportion of unknown-word errors collapses from 44% in HMM v1 down to a minor fraction in BiLSTM."
        },
        "Top Error Confusions Comparison": {
            "path": fig_dir / "dl_top_confusions_comparison.png",
            "category": "Error Analysis",
            "description": "Top 5 confusion pairs (NOUN vs PROPN, VERB vs ADJ, ADP vs PART, NOUN vs VERB, ADJ vs ADV) and how each model handles syntactic ambiguities."
        },
        "Char-CNN Morphology Breakdown": {
            "path": fig_dir / "bilstm_char_cnn_morphology_breakdown.png",
            "category": "Architecture Deep-Dive",
            "description": "OOV accuracy breakdown by morphological suffix category (-ing, -ed, -ly, -tion, capitalization, numeric), showing the Char-CNN's subword extraction power."
        },
        "Syntactic Cluster PCA Projection": {
            "path": fig_dir / "bilstm_syntactic_cluster_pca.png",
            "category": "Representational Analysis",
            "description": "2D PCA projection of the 512-dimensional BiLSTM hidden states for 1,000 test tokens, showing clear geometric clustering by grammatical category."
        },
        "Seed Stability Boxplot": {
            "path": fig_dir / "dl_seed_stability_boxplot.png",
            "category": "Scientific Rigor",
            "description": "Performance distributions across 5 random seeds showing tight variance (95.33% ± 0.17%), proving architectural robustness and reproducibility."
        },
        "Hyperparameter Search Heatmap": {
            "path": fig_dir / "dl_hyperparameter_grid_search.png",
            "category": "Scientific Rigor",
            "description": "Grid search exploration across learning rates (1e-4 to 1e-2) and hidden dimensions (64 to 256), identifying optimal convergence parameters."
        },
    }
    return figures
