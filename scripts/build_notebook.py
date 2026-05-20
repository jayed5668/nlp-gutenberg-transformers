#!/usr/bin/env python3
"""Generate the structured Assignment 3 Jupyter notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


SECTIONS = [
    ("# 1. Title and Project Overview", """
## 1.1 Assignment Title
**Assignment 3 — Natural Language Processing with Deep Learning and Transformers**

## 1.2 Project Goal
Build and compare NLP models that classify Project Gutenberg catalog entries into bookshelf categories using metadata text (title, authors, subjects).

## 1.3 Problem Statement
Digital libraries contain large volumes of unstructured text metadata. Manual categorization is slow and inconsistent. This project automates category prediction from book metadata.

## 1.4 Objectives
- Explore and preprocess the Gutenberg catalog dataset
- Train a baseline neural network classifier
- Implement a custom Transformer encoder classifier
- Evaluate models with standard classification metrics
- Compare baseline vs Transformer performance

## 1.5 Expected Outcomes
- Cleaned dataset and vocabulary
- Trained baseline and Transformer models
- Evaluation metrics, confusion matrices, and training curves
- Written analysis of results and limitations
"""),
    ("# 2. Introduction to NLP", """
## 2.1 What is NLP?
Natural Language Processing (NLP) enables computers to understand, process, and generate human language.

## 2.2 Real-World Applications
Search engines, chatbots, machine translation, sentiment analysis, and document classification.

## 2.3 Challenges in NLP
Ambiguity, context dependence, spelling variation, rare words, and class imbalance.

## 2.4 Traditional NLP vs Deep Learning NLP
Traditional methods use hand-crafted features (BoW, TF-IDF). Deep learning learns representations automatically.

## 2.5 Introduction to Transformers
Transformers use self-attention to model relationships between all tokens in parallel.

## 2.6 Why Transformers are Important
They capture long-range dependencies better than RNNs and scale effectively with pretraining.
"""),
    ("# 3. Dataset Overview", """
## 3.1 Dataset Description
Project Gutenberg catalog metadata (`pg_catalog.csv`) with book identifiers, type, language, title, authors, subjects, and bookshelf tags.

## 3.2 Dataset Source
[Project Gutenberg](https://www.gutenberg.org/) public catalog export.

## 3.3 Dataset Structure
Columns: `Text#`, `Type`, `Issued`, `Title`, `Language`, `Authors`, `Subjects`, `LoCC`, `Bookshelves`.

## 3.4 Input and Target Variables
- **Input:** combined text from `Title`, `Authors`, `Subjects`
- **Target:** primary bookshelf category (first tag in `Bookshelves`)

## 3.5 Example Samples
Shown in the code cell below.

## 3.6 Initial Dataset Inspection
Load raw data and display shape, dtypes, and missing values.
"""),
]

# Additional section headers for notebook navigation
MORE_HEADERS = [
    "# 4. Exploratory Data Analysis (EDA)",
    "# 5. Text Cleaning and Preprocessing",
    "# 6. Text Representation Techniques",
    "# 7. Tokenization and Vocabulary Building",
    "# 8. Train, Validation, and Test Split",
    "# 9. Baseline Deep Learning Model",
    "# 10. Introduction to Transformers",
    "# 11. Transformer Architecture Implementation",
    "# 12. Model Training",
    "# 13. Model Evaluation",
    "# 14. Text Generation / Prediction",
    "# 15. Attention Visualization",
    "# 16. Hyperparameter Experiments",
    "# 17. Transfer Learning Experiments",
    "# 18. Comparison of Models",
    "# 19. Computational Performance Analysis",
    "# 20. Challenges Faced During the Project",
    "# 21. Ethical Considerations in NLP",
    "# 22. Real-World Applications",
    "# 23. Future Improvements",
    "# 24. Conclusion",
    "# 25. GitHub / GitLab Repository Structure",
    "# 26. References",
]


def build():
    cells = [
        md("# Assignment 3 — NLP with Transformers\n\n**Student project — Deep Learning Minor**"),
        code(
            """import sys
from pathlib import Path
ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from src.config import *
from src.data_loading import load_raw_catalog, prepare_classification_dataset
from src.preprocessing import clean_text, preprocess_dataframe, TextVectorizer
from src.train import prepare_tensors, train_baseline, train_transformer
from src.evaluate import evaluate_classifier, save_evaluation, build_comparison_table
from src.visualization import *

%matplotlib inline
sns.set_theme(style='whitegrid')
print('TensorFlow:', tf.__version__)
print('Project root:', ROOT)"""
        ),
    ]

    for title, body in SECTIONS:
        cells.append(md(title + "\n" + body))

    cells.append(
        code(
            """raw_df = load_raw_catalog()
print('Raw shape:', raw_df.shape)
display(raw_df.head())
display(raw_df.info())
display(raw_df.isnull().sum())"""
        )
    )

    for header in MORE_HEADERS:
        cells.append(md(header + "\n\n*See code and outputs in following cells.*"))

    # EDA block
    cells.append(
        code(
            """df = prepare_classification_dataset(raw_df)
df = preprocess_dataframe(df)
print('Processed shape:', df.shape)
print('Classes:', df['category'].nunique())
display(df[['Title','Authors','Subjects','category','text_clean']].head())

plot_class_distribution(df)
plot_text_length_distribution(df)
tokens = []
for t in df['text_clean'].head(3000):
    tokens.extend(t.split())
plot_word_frequency(tokens)"""
        )
    )

    cells.append(
        code(
            """splits, vectorizer, meta = prepare_tensors(df)
X_train, X_val, X_test, y_train, y_val, y_test = splits
print('Train:', X_train.shape, 'Val:', X_val.shape, 'Test:', X_test.shape)
print('Vocab size:', meta['vocab_size'], 'Classes:', meta['num_classes'])"""
        )
    )

    cells.append(
        code(
            """baseline_model, baseline_hist, baseline_result, baseline_test, _ = train_baseline(
    splits, vectorizer, meta
)
plot_training_history(baseline_hist, 'baseline')
baseline_metrics, baseline_pred, _ = evaluate_classifier(
    baseline_model, *baseline_test, meta['class_names']
)
save_evaluation('baseline', baseline_metrics)
plot_confusion_matrix(baseline_test[1], baseline_pred, meta['class_names'], 'baseline')
baseline_result"""
        )
    )

    cells.append(
        code(
            """transformer_model, transformer_hist, transformer_result, transformer_test, _ = train_transformer(
    splits, vectorizer, meta
)
plot_training_history(transformer_hist, 'transformer')
transformer_metrics, transformer_pred, _ = evaluate_classifier(
    transformer_model, *transformer_test, meta['class_names']
)
save_evaluation('transformer', transformer_metrics)
plot_confusion_matrix(transformer_test[1], transformer_pred, meta['class_names'], 'transformer')
transformer_result"""
        )
    )

    cells.append(
        code(
            """comparison = build_comparison_table([
    {'Model': 'Baseline NN', 'Accuracy': baseline_metrics['accuracy'],
     'F1_macro': baseline_metrics['f1_macro'],
     'Train_time_sec': baseline_result['train_time_sec']},
    {'Model': 'Custom Transformer', 'Accuracy': transformer_metrics['accuracy'],
     'F1_macro': transformer_metrics['f1_macro'],
     'Train_time_sec': transformer_result['train_time_sec']},
])
display(comparison)

# Example predictions
sample_idx = np.random.choice(len(X_test), 5, replace=False)
probs = transformer_model.predict(X_test[sample_idx], verbose=0)
for i, idx in enumerate(sample_idx):
    pred = meta['class_names'][np.argmax(probs[i])]
    true = meta['class_names'][y_test[idx]]
    print(f'True: {true[:50]} | Pred: {pred[:50]} | OK={pred==true}')"""
        )
    )

    cells.append(
        md(
            """## 24. Conclusion

This notebook implemented a full NLP pipeline on the Project Gutenberg catalog:
EDA, preprocessing, baseline embedding classifier, custom Transformer encoder, and evaluation.

## 25. Repository Structure

```
Assignment -3/
├── data/raw/          # pg_catalog.csv
├── data/processed/    # cleaned CSV
├── notebooks/         # this notebook
├── src/               # reusable modules
├── outputs/figures/   # plots
├── outputs/metrics/   # JSON metrics
└── models/            # saved Keras models
```

## 26. References
- Vaswani et al. (2017) — Attention Is All You Need
- Project Gutenberg Catalog: https://www.gutenberg.org/
- TensorFlow documentation: https://www.tensorflow.org/
"""
        )
    )

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
