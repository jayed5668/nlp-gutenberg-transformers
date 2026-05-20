#!/usr/bin/env python3
"""Build the main Assignment 3 notebook with first-person documentation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docs.assignment_steps_markdown import COMPARISON, STEP1, STEP2, STEP3, STEP4  # noqa: E402

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


INTRO = """
# Assignment 3 — NLP with Transformers (Project Gutenberg)

**Student:** Naimur Rahman Jayed · **Course:** Deep Learning Minor · **Institution:** Inholland

---

## How to read this notebook

I structured this notebook exactly like the assignment brief:

| Step | Topic |
|------|--------|
| **1** | Data preparation & embeddings |
| **2** | Conv1D and LSTM multi-label classification |
| **3** | DistilBERT (pretrained Transformer) classification |
| **4** | Category-conditioned text generation |

Each step has **markdown explanations** (what / how / why) and **code cells** you can run in order. Figures and metrics are saved under `outputs/`.
"""


def build():
    cells = [
        md(INTRO),
        code(
            """# Environment setup
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

%matplotlib inline
sns.set_theme(style="whitegrid")
print("Project root:", ROOT)"""
        ),
        md(STEP1),
        code(
            """from src.data_loading import load_raw_catalog
from src.multilabel_data import prepare_multilabel_dataset
from src.preprocessing import preprocess_dataframe
from src.visualization import plot_class_distribution, plot_text_length_distribution

raw_df = load_raw_catalog()
print("Full catalog shape:", raw_df.shape)
display(raw_df.head())

df, class_names = prepare_multilabel_dataset(raw_df)
df = preprocess_dataframe(df, text_col="text")
df["num_labels"] = df["label_list"].apply(len)
print("Working set:", df.shape)
print("Label categories:", len(class_names))
display(df[["Title", "label_list", "text_clean", "num_labels"]].head())

# Primary tag for EDA plot
df["category"] = df["label_list"].str[0]
plot_class_distribution(df)
plot_text_length_distribution(df, col="text")"""
        ),
        md(STEP2),
        code(
            """from src.multilabel_train import prepare_multilabel_tensors, train_conv1d, train_lstm
import pandas as pd

splits, vectorizer, meta = prepare_multilabel_tensors(df)
conv_model, conv_hist, conv_res = train_conv1d(splits, meta, epochs=4)
lstm_model, lstm_hist, lstm_res = train_lstm(splits, meta, epochs=4)

rows = []
for name, res in [("Conv1D", conv_res), ("BiLSTM", lstm_res)]:
    m = res["test_metrics"]
    rows.append({"Model": name, **m, "train_sec": res["train_time_sec"]})
display(pd.DataFrame(rows))"""
        ),
        md(STEP3),
        code(
            """# DistilBERT — downloads weights on first run (~250 MB). Allow a few minutes.
from src.bert_classifier import train_bert

# Set epochs=2 in src/config.py for stronger results if you have time
bert_model, bert_tok, bert_res = train_bert()
print(json.dumps(bert_res["test_metrics"], indent=2))"""
        ),
        md(STEP4),
        code(
            """from src.text_generation import train_and_generate

gen_model, gen_result = train_and_generate(category="Category: History - American")
print("Prompt category:", gen_result["prompt_category"])
print("Seed:", gen_result["seed"])
print("\\n--- Generated text ---\\n")
print(gen_result["generated_text"])"""
        ),
        md(COMPARISON),
        code(
            """# Load saved metrics for final comparison table
from pathlib import Path
import pandas as pd

metrics_dir = ROOT / "outputs" / "metrics"
rows = []
for path in sorted(metrics_dir.glob("*multilabel.json")):
    data = json.loads(path.read_text())
    row = {"file": path.name, "model": data.get("model"), "train_sec": data.get("train_time_sec")}
    if "test_metrics" in data:
        row.update(data["test_metrics"])
    rows.append(row)
display(pd.DataFrame(rows))"""
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
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build()
