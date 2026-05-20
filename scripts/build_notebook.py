#!/usr/bin/env python3
"""Generate the structured Assignment 3 Jupyter notebook with polished markdown."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docs.notebook_markdown import INTRO, SECTIONS  # noqa: E402

NOTEBOOK_PATH = ROOT / "notebooks" / "assignment3_nlp_transformers.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")],
        "outputs": [],
        "execution_count": None,
    }


def build():
    cells = [md(INTRO), code(
        """import sys
from pathlib import Path
ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from src.config import *
from src.data_loading import load_raw_catalog, prepare_classification_dataset
from src.preprocessing import preprocess_dataframe
from src.train import prepare_tensors, train_baseline, train_transformer
from src.evaluate import evaluate_classifier, save_evaluation, build_comparison_table
from src.visualization import *
from src.attention_viz import visualize_sample_attention
from src.hyperparameter_experiments import run_experiments

%matplotlib inline
sns.set_theme(style="whitegrid")
print("TensorFlow:", tf.__version__)
print("Project root:", ROOT)"""
    )]

    for num in range(1, 27):
        if num in SECTIONS:
            cells.append(md(SECTIONS[num]))

    # Code blocks keyed to sections
    cells.append(code(
        """raw_df = load_raw_catalog()
print("Raw catalog shape:", raw_df.shape)
display(raw_df.head())
print("\\nMissing values:")
display(raw_df.isnull().sum())"""
    ))

    cells.append(code(
        """df = prepare_classification_dataset(raw_df)
df = preprocess_dataframe(df)
print("Working set shape:", df.shape)
print("Number of classes:", df["category"].nunique())
display(df[["Title", "Authors", "Subjects", "category", "text_clean"]].head())

plot_class_distribution(df)
plot_text_length_distribution(df)
tokens = []
for t in df["text_clean"].head(3000):
    tokens.extend(t.split())
plot_word_frequency(tokens)"""
    ))

    cells.append(code(
        """splits, vectorizer, meta = prepare_tensors(df)
X_train, X_val, X_test, y_train, y_val, y_test = splits
print("Train:", X_train.shape, "| Val:", X_val.shape, "| Test:", X_test.shape)
print("Vocabulary size:", meta["vocab_size"], "| Classes:", meta["num_classes"])"""
    ))

    cells.append(code(
        """baseline_model, baseline_hist, baseline_result, baseline_test, _ = train_baseline(
    splits, vectorizer, meta
)
plot_training_history(baseline_hist, "baseline")
baseline_metrics, baseline_pred, _ = evaluate_classifier(
    baseline_model, *baseline_test, meta["class_names"]
)
save_evaluation("baseline", baseline_metrics)
plot_confusion_matrix(baseline_test[1], baseline_pred, meta["class_names"], "baseline")
baseline_result"""
    ))

    cells.append(code(
        """transformer_model, transformer_hist, transformer_result, transformer_test, _ = train_transformer(
    splits, vectorizer, meta
)
plot_training_history(transformer_hist, "transformer")
transformer_metrics, transformer_pred, _ = evaluate_classifier(
    transformer_model, *transformer_test, meta["class_names"]
)
save_evaluation("transformer", transformer_metrics)
plot_confusion_matrix(transformer_test[1], transformer_pred, meta["class_names"], "transformer")
transformer_result"""
    ))

    cells.append(code(
        """comparison = build_comparison_table([
    {"Model": "Baseline NN", "Accuracy": baseline_metrics["accuracy"],
     "F1_macro": baseline_metrics["f1_macro"],
     "Train_time_sec": baseline_result["train_time_sec"]},
    {"Model": "Custom Transformer", "Accuracy": transformer_metrics["accuracy"],
     "F1_macro": transformer_metrics["f1_macro"],
     "Train_time_sec": transformer_result["train_time_sec"]},
])
display(comparison)

sample_idx = np.random.choice(len(X_test), 5, replace=False)
probs = transformer_model.predict(X_test[sample_idx], verbose=0)
for i, idx in enumerate(sample_idx):
    pred = meta["class_names"][np.argmax(probs[i])]
    true = meta["class_names"][y_test[idx]]
    print(f"True: {true[:55]}")
    print(f"Pred: {pred[:55]} | Correct: {pred == true}\\n")"""
    ))

    cells.append(code(
        """# Attention visualization on one test example
rev_index = {v: k for k, v in vectorizer.word_index.items()}
sample_x = X_test[:1]
token_ids = [rev_index.get(i, "<unk>") for i in sample_x[0] if i != 0][:30]
visualize_sample_attention(transformer_model, sample_x, token_ids)
print("Saved attention heatmap to outputs/figures/attention_heatmap.png")"""
    ))

    cells.append(code(
        """hp_results = run_experiments(epochs=3)
pd.DataFrame(hp_results)"""
    ))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build()
