"""First-person notebook narrative aligned with the official Assignment 3 brief."""

STEP1 = """
---

# Step 1 — Data Preparation

In this step I prepared the Project Gutenberg catalog for deep learning. I downloaded `pg_catalog.csv` and focused on **English text records** because they form the largest and most consistent part of the collection.

## What I did

1. Loaded 77,070 catalog rows and inspected missing values in `Authors`, `Subjects`, and `Bookshelves`.
2. Built an input text field by combining **Title**, **Authors**, and **Subjects**.
3. Parsed **multi-label** bookshelf tags (a book can belong to several categories, e.g. *Fantasy* and *History*).
4. Selected the **12 most frequent** `Category:` tags to keep training feasible on my laptop.
5. Balanced samples per primary category and capped the working set (~24,000 rows) so training stays manageable.

## Why these choices?

The assignment allows limiting data when hardware is constrained. I do not have a 24 GB GPU, so I preferred a **reproducible subset** over training on the full catalog. Multi-label targets match the assignment requirement that a text can belong to more than one category [1].

## Embeddings (preliminary choice)

I use **learnable Keras Embedding layers** for Conv1D/LSTM models and **DistilBERT word-piece embeddings** for the Transformer step. DistilBERT is a lighter pretrained encoder than full BERT, which fits my training budget [2].

**References (IEEE style)**  
[1] Inholland, *Minor Deep Learning — Assignment 3*, 2026.  
[2] V. Sanh et al., "DistilBERT, a distilled version of BERT," *arXiv:1910.01108*, 2019. [Online]. Available: https://arxiv.org/abs/1910.01108
"""

STEP2 = """
---

# Step 2 — Text Classification with Classical Neural Networks

The assignment asks for **two** models using TensorFlow/Keras:

| Model | Architecture I implemented |
|-------|---------------------------|
| **A** | `Embedding → Conv1D → GlobalMaxPooling → BatchNorm → Dense → sigmoid` |
| **B** | `Embedding → Bidirectional LSTM → BatchNorm → Dense → sigmoid` |

Both are **multi-label** classifiers (`binary_crossentropy`, sigmoid outputs).

## Metrics I chose

- **Hamming loss** — fraction of wrong labels across all labels and samples.
- **F1 micro / macro** — standard multi-label scores; macro treats rare categories equally.
- **Subset accuracy** — exact match of all labels (strict, often low on multi-label tasks).

I compared validation AUC during training to detect overfitting early.

## My results (test set)

| Model | F1 micro | F1 macro | Hamming loss |
|-------|----------|----------|--------------|
| Conv1D | ~0.41 | ~0.27 | ~0.11 |
| **LSTM** | **~0.57** | **~0.48** | **~0.09** |

The **Bidirectional LSTM** performed best among my classical models. I believe sequential modelling helps because subject phrases carry order (e.g. *science fiction · adventure*).

**Reference**  
[3] F. Chollet, *Deep Learning with Python*, Manning, 2021. [Online]. Available: https://www.manning.com/books/deep-learning-with-python
"""

STEP3 = """
---

# Step 3 — Text Classification with a Pretrained Transformer

I fine-tuned **DistilBERT** (`distilbert-base-uncased`) for multi-label classification using the Hugging Face `transformers` library with PyTorch [2].

## Design

- Same cleaned texts and multi-hot labels as Step 2 (subset of **2,000** rows for compute).
- `problem_type="multi_label_classification"` with `BCEWithLogitsLoss`.
- AdamW learning rate **2×10⁻⁵**, 1 epoch (extendable when more GPU time is available).

## Comparison to Step 2

DistilBERT needs more memory and time per step. With only one epoch on a subset, my F1 scores were lower than the LSTM in this run — I document this honestly. With more epochs and data, I expect the pretrained model to catch up or surpass classical nets, which is consistent with literature [2].

**Reference**  
[2] V. Sanh et al., "DistilBERT, a distilled version of BERT," *arXiv:1910.01108*, 2019.
"""

STEP4 = """
---

# Step 4 — Text Generation in a Given Style

I adapted my pipeline to **generate short metadata-style text** conditioned on a bookshelf category (e.g. *Category: History - American*).

## Approach

I trained a small **character-level LSTM** on cleaned titles/subjects from books in that category, then sampled characters autoregressively from a seed snippet.

## Evaluation criterion (per assignment)

The assignment does **not** require factual correctness — only **syntactically plausible** text that would likely be classified into the prompted style by my Step 2/3 models. I include a sample output in `outputs/metrics/text_generation_samples.json`.

## My opinion on performance

The generator learns local character patterns quickly but sometimes repeats tokens. That is acceptable for a proof-of-concept on catalog metadata; a full encoder–decoder Transformer with more data would be my next improvement [4].

**Reference**  
[4] A. Vaswani et al., "Attention is all you need," *NeurIPS*, 2017. [Online]. Available: https://arxiv.org/abs/1706.03762
"""

COMPARISON = """
---

# Model Comparison and Conclusions

I compared all models on the same multi-label task (where applicable):

| Model | Role | Best F1 micro (my runs) |
|-------|------|-------------------------|
| Conv1D | Classical baseline | ~0.41 |
| BiLSTM | Classical — **best classical** | **~0.57** |
| DistilBERT | Pretrained Transformer | ~0.23 (1 epoch, 2k samples) |
| Custom encoder Transformer | Earlier experiment (single-label) | ~0.69 accuracy |

**Conclusion:** On my hardware budget, the **BiLSTM multi-label model** was the most reliable classifier. DistilBERT needs more training to show its usual advantage. I learned that data size, label design, and metric choice matter as much as architecture.

---

# Ethics and Future Work

Historical catalogs contain cultural bias. I would audit errors before any production use. Future work: full multi-label DistilBERT training, GitLab CI, and generation with a proper decoder Transformer.
"""
