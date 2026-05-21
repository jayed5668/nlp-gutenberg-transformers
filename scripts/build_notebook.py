#!/usr/bin/env python3
"""Build notebook with rich markdown + visualisations."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docs.assignment_steps_markdown import (  # noqa: E402
    COMPARISON,
    INTRO,
    STEP1,
    STEP1_VIZ,
    STEP2,
    STEP3,
    STEP4,
    STEP18,
    STEP19,
    STEP20,
)

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
    cells = [
        md(INTRO),
        code(
            """# Setup
import sys
from pathlib import Path
ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import display, Image, Markdown

%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["font.size"] = 11
sns.set_theme(style="whitegrid", palette="muted")
print("Ready — project root:", ROOT)"""
        ),
        md(STEP1),
        code(
            """from src.data_loading import load_raw_catalog
from src.multilabel_data import prepare_multilabel_dataset
from src.preprocessing import preprocess_dataframe
from src.visualization import (
    plot_missing_values,
    plot_language_distribution,
    plot_labels_per_book,
    plot_multilabel_cooccurrence,
    plot_class_distribution,
    plot_text_length_distribution,
    plot_word_frequency,
)

raw_df = load_raw_catalog()
print("Shape:", raw_df.shape)
display(raw_df.head())"""
        ),
        code(
            """# Visualisation 1 — missing values in raw catalog
plot_missing_values(raw_df)"""
        ),
        code(
            """# Visualisation 2 — languages in full catalog
plot_language_distribution(raw_df)"""
        ),
        code(
            """# Prepare my working multi-label subset
df, class_names = prepare_multilabel_dataset(raw_df)
df = preprocess_dataframe(df, text_col="text")
df["num_labels"] = df["label_list"].apply(len)
df["category"] = df["label_list"].str[0]
print("Working set:", df.shape, "| categories:", len(class_names))
display(df[["Title", "label_list", "num_labels", "text_clean"]].head(8))"""
        ),
        code(
            """# Visualisation 3 — multi-label intensity
plot_labels_per_book(df)"""
        ),
        code(
            """# Visualisation 4 — which categories co-occur?
plot_multilabel_cooccurrence(df["label_list"].tolist(), class_names)"""
        ),
        code(
            """# Visualisation 5–7 — class balance, text length, vocabulary
plot_class_distribution(df)
plot_text_length_distribution(df, col="text")
tokens = []
for t in df["text_clean"].head(4000):
    tokens.extend(t.split())
plot_word_frequency(tokens, top_n=20)"""
        ),
        md(STEP1_VIZ),
        md(STEP2),
        code(
            """from src.multilabel_train import prepare_multilabel_tensors, train_conv1d, train_lstm
from src.multilabel_eval import predict_multilabel
from src.visualization import (
    plot_training_history_multilabel,
    plot_multilabel_metrics_bar,
    plot_per_label_f1,
    plot_sample_predictions,
)

splits, vectorizer, meta = prepare_multilabel_tensors(df)
X_train, X_val, X_test, y_train, y_val, y_test = splits
texts_test = meta["texts_test"]
print("Test samples:", len(X_test))"""
        ),
        code(
            """# Train Conv1D — I watch validation AUC in the next plot
conv_model, conv_hist, conv_res = train_conv1d(splits, meta, epochs=4)
plot_training_history_multilabel(conv_hist, "conv1d")"""
        ),
        code(
            """# Train BiLSTM
lstm_model, lstm_hist, lstm_res = train_lstm(splits, meta, epochs=4)
plot_training_history_multilabel(lstm_hist, "lstm")"""
        ),
        code(
            """# Visualisation — compare test metrics
plot_multilabel_metrics_bar([conv_res, lstm_res])

conv_pred, conv_prob = predict_multilabel(conv_model, X_test)
lstm_pred, lstm_prob = predict_multilabel(lstm_model, X_test)
plot_per_label_f1(y_test, lstm_pred, class_names, name="lstm_per_label_f1.png")
plot_sample_predictions(texts_test, y_test, lstm_prob, class_names, n=4)

# Advanced error-analysis plots
from src.visualization import (
    plot_multilabel_truth_pred_heatmap,
    plot_probability_heatmap,
    plot_threshold_sweep,
)
plot_multilabel_truth_pred_heatmap(y_test, lstm_pred, class_names, n_samples=20)
plot_probability_heatmap(lstm_prob, class_names, n_samples=20)
plot_threshold_sweep(y_test, lstm_prob)
plot_per_label_f1(y_test, conv_pred, class_names, name="conv1d_per_label_f1.png")"""
        ),
        code(
            """# Table for my report
pd.DataFrame([
    {"Model": "Conv1D", **conv_res["test_metrics"], "sec": conv_res["train_time_sec"]},
    {"Model": "BiLSTM", **lstm_res["test_metrics"], "sec": lstm_res["train_time_sec"]},
])"""
        ),
        md(STEP18),
        md(STEP19),
        code(
            """from src.hyperparameter_experiments import run_multilabel_experiments
from src.visualization import plot_hyperparameter_results

hp_results = run_multilabel_experiments(epochs=3)
display(pd.DataFrame([{"experiment": r["experiment"], **r["test_metrics"]} for r in hp_results]))
plot_hyperparameter_results(hp_results)"""
        ),
        md(STEP3),
        code(
            """# Optional: skip if offline — needs HuggingFace download (~250 MB)
RUN_BERT = True  # set False to load saved metrics only

if RUN_BERT:
    from src.bert_classifier import train_bert
    bert_model, bert_tok, bert_res = train_bert()
else:
    bert_res = json.loads((ROOT / "outputs/metrics/bert_multilabel.json").read_text())

display(pd.Series(bert_res["test_metrics"]))"""
        ),
        code(
            """# Add BERT to comparison chart
all_results = [conv_res, lstm_res, bert_res]
plot_multilabel_metrics_bar(all_results)"""
        ),
        md(STEP4),
        code(
            """from src.text_generation import train_and_generate
from src.visualization import plot_generated_text_card

gen_model, gen_result = train_and_generate(category="Category: History - American")
plot_generated_text_card(
    gen_result["seed"],
    gen_result["generated_text"],
    gen_result["prompt_category"],
)
print(gen_result["generated_text"])"""
        ),
        md(STEP20),
        code(
            """from src.generation_eval import generate_and_classify
from src.visualization import plot_generation_comparison

GEN_CATEGORIES = [
    "Category: History - American",
    "Category: Children & Young Adult Reading",
    "Category: Poetry",
]
gen_eval = generate_and_classify(GEN_CATEGORIES, lstm_model, vectorizer, class_names)
display(pd.DataFrame(gen_eval))
plot_generation_comparison(gen_eval)"""
        ),
        md(COMPARISON),
        code(
            """# Final dashboard — all saved metrics
from src.visualization import plot_multilabel_metrics_bar

metrics_dir = ROOT / "outputs" / "metrics"
loaded = []
for path in sorted(metrics_dir.glob("*multilabel.json")):
    data = json.loads(path.read_text())
    if "test_metrics" in data:
        loaded.append(data)
display(pd.DataFrame([
    {"model": d["model"], **d["test_metrics"], "train_sec": d.get("train_time_sec")}
    for d in loaded
]))
if loaded:
    plot_multilabel_metrics_bar(loaded)

# Show saved figures gallery
fig_dir = ROOT / "outputs" / "figures"
for img in sorted(fig_dir.glob("*.png")):
    display(Markdown(f"**{img.name}**"))
    display(Image(filename=str(img), width=700))"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NOTEBOOK_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
