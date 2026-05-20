# NLP with Transformers — Project Gutenberg Catalog

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

> **Deep Learning Minor · Inholland University of Applied Sciences**  
> Assignment 3 — Natural Language Processing with Deep Learning and Transformers

---

## Table of Contents

1. [Overview](#overview)
2. [What I Built](#what-i-built)
3. [Results](#results)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Step-by-Step Workflow](#step-by-step-workflow)
7. [Documentation](#documentation)
8. [GitHub Setup](#github-setup)
9. [References](#references)

---

## Overview

In this project I classified **Project Gutenberg bookshelf categories** from book metadata. Each record contains a title, authors, and subjects; my models learn to predict the primary category tag (for example *Category: Novels* or *Category: Biographies*).

I compared two approaches:

| Approach | Description |
|----------|-------------|
| **Baseline NN** | Embedding layer + global pooling + dense layers |
| **Custom Transformer** | Encoder with multi-head self-attention and positional encoding |

The full narrative, EDA, and evaluation are in the [Jupyter notebook](notebooks/assignment3_nlp_transformers.ipynb) and the [written report](docs/PROJECT_REPORT.md).

---

## What I Built

- End-to-end data pipeline (`src/data_loading.py`, `preprocessing.py`)
- Exploratory analysis with saved figures (`outputs/figures/`)
- Baseline and Transformer classifiers (`src/baseline_model.py`, `transformer_model.py`)
- Training with early stopping and checkpoints (`scripts/run_pipeline.py`)
- Evaluation metrics and confusion matrices (`outputs/metrics/`)
- Attention heatmap visualization (`src/attention_viz.py`)
- Hyperparameter comparison experiments (`src/hyperparameter_experiments.py`)

---

## Results

| Model | Test accuracy | Macro F1 | Training time |
|-------|---------------|----------|---------------|
| Baseline NN | **70.9%** | **0.682** | ~6 s |
| Custom Transformer | 69.4% | 0.669 | ~6 s |

My baseline performed slightly better on short metadata texts. Confusion matrices and per-class reports are in `outputs/metrics/` and `outputs/figures/`.

---

## Project Structure

```
Assignment -3/
│
├── README.md                 # You are here
├── STEPS.md                  # Step-by-step commit log
├── requirements.txt
│
├── data/
│   ├── raw/pg_catalog.csv    # Original Gutenberg catalog (77k rows)
│   └── processed/            # Cleaned subset (generated)
│
├── notebooks/
│   └── assignment3_nlp_transformers.ipynb   # Main submission notebook
│
├── docs/
│   ├── PROJECT_REPORT.md     # Full written report (sections 1–26)
│   └── notebook_markdown.py  # Notebook narrative source
│
├── src/                      # Reusable Python modules
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── baseline_model.py
│   ├── transformer_model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualization.py
│   ├── attention_viz.py
│   └── hyperparameter_experiments.py
│
├── scripts/
│   ├── run_pipeline.py       # Train + evaluate everything
│   ├── build_notebook.py     # Regenerate notebook from markdown
│   ├── setup_github.sh       # Create repo + push
│   └── push_step.sh          # Push latest commit
│
├── outputs/
│   ├── figures/              # EDA plots, training curves, attention map
│   └── metrics/              # JSON evaluation results
│
└── models/                   # Saved Keras weights (generated locally)
```

---

## Quick Start

```bash
cd "Assignment -3"

# 1. Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

# 3. Train models and save outputs
python scripts/run_pipeline.py

# 4. Regenerate notebook (optional)
python scripts/build_notebook.py

# 5. Open notebook
jupyter notebook notebooks/assignment3_nlp_transformers.ipynb
```

---

## Step-by-Step Workflow

I organised development into numbered Git commits (see [STEPS.md](STEPS.md)):

| Step | Topic |
|------|--------|
| 01 | Project setup & dataset overview |
| 02 | Exploratory data analysis |
| 03 | Text cleaning & tokenization |
| 04 | Train / validation / test split |
| 05 | Baseline neural network |
| 06 | Transformer architecture |
| 07 | Training pipeline |
| 08 | Evaluation & predictions |
| 09 | Analysis support |
| 10 | Notebook & report |
| 11+ | Attention viz, hyperparameters, polished docs |

---

## Documentation

| File | Purpose |
|------|---------|
| [notebooks/assignment3_nlp_transformers.ipynb](notebooks/assignment3_nlp_transformers.ipynb) | Executable notebook with first-person explanations |
| [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) | Complete report matching the assignment section outline |
| [STEPS.md](STEPS.md) | Development checklist and Git commit map |

---

## GitHub Setup

**Repository name:** `nlp-gutenberg-transformers`

```bash
gh auth login          # one-time
./scripts/setup_github.sh
```

After pushing: `https://github.com/jayed5668/nlp-gutenberg-transformers`

To push a new step:

```bash
git add .
git commit -m "step-XX: description"
./scripts/push_step.sh
```

---

## Dataset

| Property | Value |
|----------|--------|
| **Source** | [Project Gutenberg Catalog](https://www.gutenberg.org/) |
| **File** | `data/raw/pg_catalog.csv` |
| **Rows** | 77,070 |
| **Task** | Multi-class classification (12 top English text categories) |
| **Input** | `Title` + `Authors` + `Subjects` |
| **Target** | First tag in `Bookshelves` |

---

## References

1. Vaswani, A. et al. (2017). *Attention Is All You Need.*
2. Project Gutenberg — https://www.gutenberg.org/
3. TensorFlow — https://www.tensorflow.org/
4. Hugging Face Transformers — https://huggingface.co/docs/transformers

---

<p align="center">
  <sub>Assignment 3 · Deep Learning Minor · Inholland</sub>
</p>
