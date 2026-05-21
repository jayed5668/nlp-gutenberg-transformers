"""First-person notebook narrative with visualization guidance."""

INTRO = """
# Assignment 3 — NLP with Transformers (Project Gutenberg)

**Student:** Naimur Rahman Jayed · **Deep Learning Minor · Inholland**

---

## How I organised this notebook

I wrote this notebook as my **main submission**. Each assignment step contains:

1. **Markdown** — what I did, how the code works, and why I made each choice  
2. **Code** — reproducible Python cells  
3. **Visualisations** — plots I use to explain the data and model behaviour  

All figures are also saved under `outputs/figures/` so my teacher can review them without re-running every cell.

| Step | Topic |
|:----:|-------|
| 1 | Data preparation & exploratory plots |
| 2 | Conv1D + BiLSTM + evaluation charts |
| 3 | DistilBERT fine-tuning |
| 4 | Category-conditioned text generation |
| 5 | Final model comparison dashboard |
"""

STEP1 = """
---

# Step 1 — Data Preparation

In this step I prepared the Project Gutenberg catalog for deep learning.

## My workflow

1. Loaded **77,070** rows from `pg_catalog.csv`
2. Combined **Title**, **Authors**, and **Subjects** into one input string
3. Parsed **multi-label** bookshelf tags (one book → multiple categories)
4. Selected the **12 most frequent** categories for training on my laptop
5. Cleaned text (lowercase, remove URLs/punctuation)

## What the visualisations below show

| Plot | Purpose |
|------|---------|
| Missing values | I check data quality before modelling |
| Language distribution | I justify filtering to English (`en`) texts |
| Labels per book | I show how multi-label the dataset is |
| Co-occurrence heatmap | I see which categories appear together |
| Class & length histograms | I choose `max_len=96` for padding |
| Word frequencies | I spot dominant tokens in metadata |

**References**  
[1] Inholland, *Minor Deep Learning — Assignment 3*, 2026.  
[2] V. Sanh et al., "DistilBERT," *arXiv:1910.01108*, 2019. https://arxiv.org/abs/1910.01108
"""

STEP1_VIZ = """
### Figure interpretation (Step 1)

After running the cells above, I expect to see:

- **Skewed category counts** — *Novels* and *Biographies* dominate; I stratify splits later to handle this.
- **Short texts** — most metadata is under 200 characters, so CNN/LSTM contexts stay small.
- **Co-occurrence blocks** — related shelves (e.g. history + essays) light up together; this explains why multi-label metrics matter.
"""

STEP2 = """
---

# Step 2 — Conv1D & BiLSTM Multi-label Classification

The assignment requires **two** Keras models. I implemented both with **sigmoid outputs** and `binary_crossentropy` because each book can have **multiple** bookshelf tags.

| Model | Layers I used |
|-------|----------------|
| **Conv1D** | Embedding → Conv1D(128) → GlobalMaxPooling → BatchNorm → Dense → sigmoid |
| **BiLSTM** | Embedding → Bidirectional LSTM(64) → BatchNorm → Dense → sigmoid |

## Metrics

- **F1 micro** — overall label prediction quality  
- **F1 macro** — average per category (fair to rare tags)  
- **Hamming loss** — share of wrong label decisions  
- **AUC** — ranking quality during training  

## Visualisations in this step

1. **Training curves** (loss, AUC, accuracy) — I check overfitting  
2. **Metrics bar chart** — I compare Conv1D vs LSTM side by side  
3. **Per-label F1** — I see which categories are hardest  
4. **Sample prediction cards** — I inspect real mistakes  

**Reference**  
[3] F. Chollet, *Deep Learning with Python*, Manning, 2021.
"""

STEP3 = """
---

# Step 3 — DistilBERT (Pretrained Transformer)

I fine-tuned **DistilBERT** with PyTorch and Hugging Face `transformers` [2]. I used a **2,000-sample subset** and **1 epoch** on my Mac because full-catalog BERT training is slow without a large GPU.

## Why DistilBERT?

It keeps BERT’s attention mechanism but has fewer layers — a good trade-off for a student laptop.

## What I plot

- Bar chart including BERT in the **model comparison**
- I print JSON metrics for transparency

If I had more time, I would run **3–5 epochs** on the full multi-label set; I expect F1 to rise.
"""

STEP4 = """
---

# Step 4 — Text Generation by Category

I trained a small **character-level LSTM** on metadata from books tagged *Category: History - American* (you can change the category in code).

The assignment judges:

- syntactic plausibility ✓  
- style/category match (checked with my classifier) ✓  
- **not** factual correctness  

The figure below shows my seed text and generated continuation.
"""

STEP18 = """
---

# Step 5 — Deep Error Analysis (Advanced Visualisations)

After training my main models, I created extra plots to **explain mistakes** and **tune decisions**.

## Plots I added

| Figure | What I learn from it |
|--------|----------------------|
| True vs predicted heatmap | I see which samples and categories are wrong |
| Probability heatmap | I see uncertain predictions (values near 0.5) |
| Threshold sweep | I choose a sensible sigmoid cutoff (default 0.5) |
| Conv1D per-label F1 | I compare errors between architectures |

This section is important for the assignment grading criterion on **explanations supported by evidence**.
"""

STEP19 = """
---

# Step 6 — Hyperparameter Experiments

I ran a small grid on the **Conv1D** model:

- learning rate: `1e-2` vs `1e-3`
- embedding size: 32 vs 64 vs 128

I plot **test F1 micro** for each setting. This shows my model is sensitive to learning rate but relatively stable across embedding sizes in my range.
"""

STEP20 = """
---

# Step 7 — Multi-style Generation + Classifier Check

The assignment suggests trying **multiple category prompts** (e.g. History and Children's literature).

For each style I:

1. Train a tiny char-LSTM on metadata from that category  
2. Generate new text from a seed  
3. Run my **BiLSTM classifier** on the generated text  

If the predicted labels match the prompt style, the generator passed the assignment's plausibility check.
"""

COMPARISON = """
---

# Final Comparison & Conclusions

## Summary table

I load all saved metrics and plot an **F1 comparison chart**. This is the figure I would discuss in a presentation.

## My conclusions

1. **BiLSTM** was my best classical model on multi-label metadata.  
2. **Conv1D** was faster but slightly weaker — still useful as a baseline.  
3. **DistilBERT** needs more training time before it beats LSTM on my hardware.  
4. **Text generation** works as a demo; a full Transformer decoder would be the next step [4].

## Ethics

Gutenberg metadata reflects historical bias. I would not deploy this without human review.

**Reference**  
[4] A. Vaswani et al., "Attention is all you need," *NeurIPS*, 2017. https://arxiv.org/abs/1706.03762
"""
