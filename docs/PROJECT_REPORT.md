# Assignment 3 — Full Project Report (Section Outline)

This document follows the required report structure. Executable code and plots live in `notebooks/assignment3_nlp_transformers.ipynb`.

---

## 1. Title and Project Overview

### 1.1 Assignment Title
Natural Language Processing with Deep Learning and Transformers — Assignment 3

### 1.2 Project Goal
Predict the primary Project Gutenberg bookshelf category from book metadata text.

### 1.3 Problem Statement
Catalog metadata is textual and high-dimensional. Manual tagging does not scale; automated classification supports search and recommendation.

### 1.4 Objectives
EDA, preprocessing, baseline NN, custom Transformer encoder, evaluation, and model comparison.

### 1.5 Expected Outcomes
Trained models (~71% test accuracy baseline), metrics JSON, confusion matrices, and documented analysis.

---

## 2. Introduction to NLP

### 2.1–2.6
NLP processes human language. Applications include chatbots, translation, and document classification. Challenges: ambiguity, noise, imbalance. Deep learning replaces hand-crafted features. Transformers use self-attention for parallel context modeling and outperform RNNs on many tasks.

---

## 3. Dataset Overview

### 3.1 Dataset Description
`pg_catalog.csv` — Gutenberg catalog with 77,070 rows and 9 columns.

### 3.2 Dataset Source
https://www.gutenberg.org/

### 3.3 Dataset Structure
`Text#`, `Type`, `Issued`, `Title`, `Language`, `Authors`, `Subjects`, `LoCC`, `Bookshelves`

### 3.4 Input and Target
- **Input:** `Title` + `Authors` + `Subjects`
- **Target:** first tag in `Bookshelves` (12 top categories, English `Text` only)

### 3.5–3.6
See notebook for sample rows and `df.info()` output.

---

## 4. Exploratory Data Analysis (EDA)

Sections 4.1–4.14 cover shape, missing values, text length, word frequency, class distribution, and initial observations. Figures: `outputs/figures/class_distribution.png`, `text_length_distribution.png`, `word_frequency.png`.

---

## 5. Text Cleaning and Preprocessing

Lowercasing, URL/punctuation/number removal, optional stopword removal, tokenization. Cleaned data saved to `data/processed/pg_catalog_processed.csv`.

---

## 6. Text Representation Techniques

BoW and TF-IDF introduced theoretically; project uses learned **Embedding** layers and integer token sequences.

---

## 7. Tokenization and Vocabulary Building

`TextVectorizer` builds word→index mapping, `<unk>` handling, padding to length 96, vocab size 12,000.

---

## 8. Train, Validation, and Test Split

70% / 15% / 15% stratified split (`RANDOM_SEED=42`).

---

## 9. Baseline Deep Learning Model

Embedding → GlobalAveragePooling → Dense(128) → Dense(64) → Softmax. Adam optimizer, sparse categorical cross-entropy.

**Test accuracy:** ~70.9% | **F1 macro:** ~0.68

---

## 10–11. Transformers

Encoder blocks with multi-head self-attention, positional encoding, feed-forward sublayers, residual connections, and layer normalization. Implemented in `src/transformer_model.py`.

---

## 12. Model Training

Batch size 64, early stopping, checkpointing to `models/`. Training curves in `outputs/figures/`.

---

## 13. Model Evaluation

Accuracy, precision, recall, F1, confusion matrix, classification report — saved under `outputs/metrics/`.

---

## 14. Text Generation / Prediction

Classification **predictions** on held-out samples (not generative LM). Example correct/incorrect pairs printed in notebook.

---

## 15. Attention Visualization

Multi-head attention weights can be extracted from Keras `MultiHeadAttention` layers for token importance analysis (extension in notebook).

---

## 16. Hyperparameter Experiments

Suggested experiments: learning rate {1e-2, 1e-3}, batch size {32, 64}, embedding dim {32, 64}, heads {2, 4}. Record in comparison table.

---

## 17. Transfer Learning Experiments

Pretrained models (BERT, DistilBERT) can be fine-tuned with Hugging Face `transformers` — optional extension requiring more GPU time.

---

## 18. Comparison of Models

| Model | Test Accuracy | F1 (macro) |
|-------|---------------|------------|
| Baseline NN | 0.709 | 0.682 |
| Custom Transformer | 0.694 | 0.669 |

Baseline slightly outperformed the smaller custom Transformer on this metadata task.

---

## 19–24. Performance, Challenges, Ethics, Applications, Future Work, Conclusion

Training ran on CPU/MPS (~3 min baseline, ~3 min transformer). Challenges: class imbalance, similar categories, metadata-only text (no full book body). Ethical risks: bias in historical corpora. Future: larger models, DistilBERT fine-tuning, full-text features.

---

## 25. Repository Structure

See `README.md`.

## 26. References

- Vaswani et al. (2017). Attention Is All You Need.
- Project Gutenberg Catalog.
- TensorFlow, scikit-learn, NLTK documentation.
