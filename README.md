<p align="center">
  <strong>NLP with Transformers — Project Gutenberg Catalog</strong><br>
  <sub>Deep Learning Minor · Inholland · Assignment 3</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=flat-square&logo=tensorflow" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/PyTorch-Transformers-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Task-Multi--label%20NLP-2ea44f?style=flat-square" alt="Task"/>
</p>

---

## Hi — this is my Assignment 3 project

I built an end-to-end NLP pipeline on the [**Project Gutenberg**](https://www.gutenberg.org/) catalog. My goal was to **classify book metadata into bookshelf categories** (multi-label), compare **Conv1D**, **LSTM**, and **DistilBERT**, and finally **generate short text** in a chosen literary style — exactly as required in the official brief.

> **GitHub:** [jayed5668/nlp-gutenberg-transformers](https://github.com/jayed5668/nlp-gutenberg-transformers)  
> **Main notebook:** [`notebooks/assignment3_nlp_transformers.ipynb`](notebooks/assignment3_nlp_transformers.ipynb)

---

## Table of contents

- [Assignment steps I completed](#assignment-steps-i-completed)
- [My results](#my-results)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [How I developed this (step commits)](#how-i-developed-this-step-commits)
- [Documentation](#documentation)
- [Dataset](#dataset)
- [References](#references)

---

## Assignment steps I completed

| Step | What I implemented | My code |
|:----:|-------------------|---------|
| **1** | Data prep, multi-label tags, text cleaning, embedding choices | `src/multilabel_data.py`, `src/preprocessing.py` |
| **2** | **Conv1D** + **BiLSTM** multi-label classifiers (Keras) | `src/conv1d_model.py`, `src/lstm_model.py` |
| **3** | **DistilBERT** fine-tuning (pretrained Transformer) | `src/bert_classifier.py` |
| **4** | Category-conditioned **text generation** (char-LSTM) | `src/text_generation.py` |

Each step is explained in the notebook with **first-person markdown** (what I did, how, and why), including **IEEE-style references**.

---

## My results

Multi-label classification on a balanced subset of English Gutenberg metadata:

| Model | F1 (micro) | F1 (macro) | Hamming loss |
|-------|------------|------------|--------------|
| Conv1D + Dense | 0.41 | 0.27 | 0.11 |
| **BiLSTM** | **0.57** | **0.48** | **0.09** |
| DistilBERT (1 epoch, 2k rows) | 0.23 | 0.06 | 0.13 |

On my laptop, the **BiLSTM** was the strongest classical model. DistilBERT needs more epochs/data to show its usual advantage — I document that limitation in the notebook.

**Notebook visualisations:** missing values, language mix, label co-occurrence, training curves, per-category F1, sample predictions, model comparison bars, **truth vs predicted heatmaps**, **threshold sweep**, **hyperparameter comparison**, multi-category **generation + classifier check**, and text-generation cards — all inline in the notebook and saved under `outputs/figures/`.

Figures: `outputs/figures/` · Metrics: `outputs/metrics/`

---

## Repository layout

```
Assignment -3/
├── README.md                          ← project overview (this file)
├── STEPS.md                           ← Git step checklist
├── requirements.txt
│
├── notebooks/
│   └── assignment3_nlp_transformers.ipynb   ← ⭐ main submission
│
├── docs/
│   ├── PROJECT_REPORT.md              ← full written report (26 sections)
│   └── assignment_steps_markdown.py   ← notebook narrative source
│
├── data/raw/pg_catalog.csv            ← Gutenberg catalog
│
├── src/                               ← all models & pipelines
│   ├── multilabel_data.py
│   ├── conv1d_model.py · lstm_model.py
│   ├── bert_classifier.py
│   ├── text_generation.py
│   └── …
│
├── scripts/
│   ├── run_pipeline.py                ← original single-label pipeline
│   ├── run_step3_bert.py
│   └── build_notebook.py
│
└── outputs/figures/ · outputs/metrics/
```

---

## Quick start

```bash
git clone https://github.com/jayed5668/nlp-gutenberg-transformers.git
cd nlp-gutenberg-transformers

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

# Regenerate notebook (optional)
python scripts/build_notebook.py

# Train classical models (Step 2)
python -c "from src.multilabel_train import prepare_multilabel_tensors, train_conv1d, train_lstm; s,m=prepare_multilabel_tensors(); train_conv1d(s,m); train_lstm(s,m)"

# DistilBERT (Step 3 — needs internet, ~2–5 min)
python scripts/run_step3_bert.py

# Open notebook
jupyter notebook notebooks/assignment3_nlp_transformers.ipynb
```

---

## How I developed this (step commits)

I committed each assignment part separately so progress is visible on GitHub:

| Commit | Content |
|--------|---------|
| `step-01` … `step-10` | Project setup, EDA, baseline Transformer, evaluation |
| `step-11+` | Attention maps, hyperparameters, documentation |
| `step-14` | Multi-label data + Conv1D + LSTM |
| `step-15` | DistilBERT + text generation |
| `step-16` | Polished README & notebook |
| `step-17` | Rich EDA + training visualisations in notebook |
| `step-18` | Error-analysis heatmaps, threshold sweep, generation eval |
| `step-19` | Notebook Steps 5–7 (hyperparameters, multi-style generation) |

Push after each step:

```bash
git add .
git commit -m "step-XX: description"
git push origin main
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Notebook](notebooks/assignment3_nlp_transformers.ipynb) | Runnable story of Steps 1–4 plus evaluation & generation analysis |
| [PROJECT_REPORT.md](docs/PROJECT_REPORT.md) | Long-form report (sections 1–26) |
| [STEPS.md](STEPS.md) | Developer checklist |

---

## Dataset

| Field | Value |
|-------|--------|
| Source | [gutenberg.org](https://www.gutenberg.org/) |
| File | `data/raw/pg_catalog.csv` (77,070 rows) |
| Input | `Title` + `Authors` + `Subjects` |
| Labels | Multi-hot bookshelf `Category:` tags (12 classes in my subset) |

---

## References

1. A. Vaswani et al., “Attention is all you need,” *NeurIPS*, 2017. https://arxiv.org/abs/1706.03762  
2. V. Sanh et al., “DistilBERT,” *arXiv:1910.01108*, 2019. https://arxiv.org/abs/1910.01108  
3. F. Chollet, *Deep Learning with Python*, Manning, 2021.  
4. Inholland, *Minor Deep Learning — Assignment 3*, 2026.

---

<p align="center"><sub>© My Deep Learning Minor submission — please cite Gutenberg data if you reuse this work.</sub></p>
