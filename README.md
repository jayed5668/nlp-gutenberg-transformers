# Assignment 3 — NLP with Transformers (Project Gutenberg Catalog)

Deep Learning Minor — Inholland University of Applied Sciences

## 1. Title and Project Overview

### 1.1 Assignment Title
Assignment 3: Natural Language Processing with Deep Learning and Transformers

### 1.2 Project Goal
Classify Project Gutenberg bookshelf categories from book metadata text using baseline neural networks and a custom Transformer encoder.

### 1.3 Problem Statement
Large digital libraries store unstructured metadata. Automating category assignment reduces manual effort and improves discoverability.

### 1.4 Objectives
- Perform EDA on `pg_catalog.csv`
- Clean and vectorize text
- Train baseline and Transformer classifiers
- Evaluate with accuracy, precision, recall, F1, and confusion matrices
- Document findings following the course report structure

### 1.5 Expected Outcomes
Processed dataset, trained models, metrics JSON files, figures, and a structured Jupyter notebook.

---

## 25. Repository Structure

```
Assignment -3/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/pg_catalog.csv
│   └── processed/
├── notebooks/
│   └── assignment3_nlp_transformers.ipynb
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── baseline_model.py
│   ├── transformer_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
├── scripts/
│   ├── build_notebook.py
│   └── run_pipeline.py
├── outputs/
│   ├── figures/
│   └── metrics/
└── models/
```

## Reproducibility

```bash
cd "Assignment -3"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_notebook.py
python scripts/run_pipeline.py
jupyter notebook notebooks/assignment3_nlp_transformers.ipynb
```

## Dataset

- **Source:** [Project Gutenberg Catalog](https://www.gutenberg.org/)
- **File:** `data/raw/pg_catalog.csv` (77,070 records)
- **Task:** Predict primary bookshelf category from `Title`, `Authors`, and `Subjects` (English text entries)

## 26. References

- Vaswani, A. et al. (2017). *Attention Is All You Need.*
- Project Gutenberg. https://www.gutenberg.org/
- TensorFlow Documentation. https://www.tensorflow.org/
