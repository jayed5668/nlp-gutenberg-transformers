"""First-person markdown sections for the Assignment 3 notebook."""

INTRO = """
# Assignment 3 — NLP with Transformers on the Project Gutenberg Catalog

**Deep Learning Minor · Inholland University of Applied Sciences**

In this notebook I document my full workflow: from exploring the Gutenberg catalog dataset to training and comparing a baseline neural network and a custom Transformer encoder for bookshelf category classification.
"""

SECTIONS = {
    1: """
## 1. Title and Project Overview

### 1.1 Assignment Title
Assignment 3: *Natural Language Processing with Deep Learning and Transformers*

### 1.2 Project Goal
My goal was to build reliable text classifiers that predict the **primary bookshelf category** of a Project Gutenberg entry using only metadata fields (`Title`, `Authors`, `Subjects`).

### 1.3 Problem Statement
The catalog contains tens of thousands of records with rich but unstructured text. Manual labeling does not scale. I wanted to automate category assignment so similar books can be grouped consistently for search and analysis.

### 1.4 Objectives
- Understand the dataset through structured EDA
- Design a reproducible preprocessing and tokenization pipeline
- Train a **baseline embedding + dense network**
- Implement and train a **custom Transformer encoder**
- Compare models with accuracy, precision, recall, F1, and confusion matrices

### 1.5 Expected Outcomes
I expected to deliver cleaned data, trained models, evaluation plots, and a clear written comparison showing which approach worked best on this task.
""",
    2: """
## 2. Introduction to NLP

### 2.1 What is Natural Language Processing (NLP)?
NLP is the field where I teach machines to work with human language — reading, understanding, and acting on text.

### 2.2 Real-World Applications of NLP
I see NLP everywhere: chatbots, translation, spam detection, sentiment analysis, and document classification (which is exactly what I do in this project).

### 2.3 Challenges in NLP
While building this project I dealt with ambiguity, noisy metadata, rare words, and **class imbalance** across bookshelf categories.

### 2.4 Traditional NLP vs Deep Learning NLP
Traditional pipelines use BoW or TF-IDF with classical ML. Deep learning learns dense representations; I used both ideas — TF-IDF conceptually, embeddings in practice.

### 2.5 Introduction to Transformers
Transformers replace recurrence with **self-attention**, letting every token attend to every other token in parallel.

### 2.6 Why Transformers are Important
For my task, attention helps the model focus on discriminative words in titles and subject lines (e.g. *Poetry*, *Adventure*, *Biographies*).
""",
    3: """
## 3. Dataset Overview

### 3.1 Dataset Description
I used `pg_catalog.csv`, the Project Gutenberg catalog with **77,070** rows and bibliographic metadata.

### 3.2 Dataset Source
Data comes from the public [Project Gutenberg](https://www.gutenberg.org/) catalog.

### 3.3 Dataset Structure
| Column | Role |
|--------|------|
| `Text#` | Unique book id |
| `Type` | Media type (Text, Sound, …) |
| `Title`, `Authors`, `Subjects` | **My input text** |
| `Bookshelves` | Category tags → **my target** |

### 3.4 Input and Target Variables
- **Input:** concatenation of `Title`, `Authors`, and `Subjects`
- **Target:** first bookshelf tag (e.g. `Category: Novels`)

### 3.5 Example Samples
I inspect sample rows in the code below to verify that titles and subjects carry category signal.

### 3.6 Initial Dataset Inspection
I check shape, dtypes, and missing values before filtering to English `Text` entries and the top 12 categories.
""",
    4: """
## 4. Exploratory Data Analysis (EDA)

In this section I explored the data before modeling.

### 4.1–4.3 Shape, Columns, Missing Values
I recorded dataset dimensions and missingness — especially in `Authors` and `Bookshelves`.

### 4.4 Duplicate Analysis
I verified there were no duplicate rows in the working subset.

### 4.5–4.9 Text Statistics
I analysed text length, word frequency, vocabulary size, and stopwords. Most metadata texts are short (median ~144 characters), which informed my `max_len=96` choice.

### 4.12 Class Distribution
Some categories (e.g. *Novels*, *Biographies*) are more frequent than others; I kept stratified splits to respect this.

### 4.14 Initial Observations
Metadata is informative but noisy; categories sometimes overlap (e.g. *Historical Novels* vs *Novels*). I expected a strong but not perfect classifier.
""",
    5: """
## 5. Text Cleaning and Preprocessing

### 5.1–5.6 Cleaning Steps
I applied lowercasing, URL removal, punctuation/number removal, and optional stopword removal.

### 5.7–5.8 Tokenization
I split cleaned text on whitespace and kept tokens for vocabulary building (lemmatization was not required for my short metadata texts).

### 5.10–5.11 Output
Cleaned text is stored in `text_clean` and saved to `data/processed/pg_catalog_processed.csv` for reproducibility.
""",
    6: """
## 6. Text Representation Techniques

### 6.1–6.3 Classical Vectorization
I studied Bag-of-Words and TF-IDF as baselines conceptually; they are strong on small text but do not capture word order.

### 6.4–6.6 Embeddings
I used **learnable Embedding layers** so the network discovers useful dimensions for book metadata during training.
""",
    7: """
## 7. Tokenization and Vocabulary Building

I built a vocabulary of up to **12,000** tokens, mapped unknown words to `<unk>`, and padded sequences to length **96**. This keeps batches efficient while covering most metadata phrases.
""",
    8: """
## 8. Train, Validation, and Test Split

I used a **70% / 15% / 15%** stratified split with `random_state=42` so each category appears proportionally in every split. This prevents optimistic scores from random lucky splits.
""",
    9: """
## 9. Baseline Deep Learning Model

### 9.2 Architecture
`Embedding → GlobalAveragePooling1D → Dense(128) → Dense(64) → Softmax`

### 9.6–9.7 Loss & Optimizer
I used **sparse categorical cross-entropy** with the **Adam** optimizer (lr = 1e-3).

### 9.11 Evaluation
My baseline reached about **71% test accuracy** and **0.68 macro F1** — a solid starting point for metadata-only classification.
""",
    10: """
## 10. Introduction to Transformers

### 10.1–10.3 Attention
I implemented an **encoder-only** Transformer (classification does not need a decoder). Self-attention lets each token weigh other tokens when forming representations.

### 10.5 Multi-Head Attention
Multiple heads learn different relationship patterns (e.g. subject keywords vs author names).

### 10.6 Positional Encoding
Because attention is permutation-invariant, I added sinusoidal positional encodings to preserve token order.

### 10.10 Why Transformers vs RNNs
For short sequences, Transformers train in parallel and avoid vanishing-gradient issues common in RNNs.
""",
    11: """
## 11. Transformer Architecture Implementation

I implemented positional encoding, multi-head attention encoder blocks, feed-forward sublayers, residual connections, and layer normalization in `src/transformer_model.py`. The final classifier uses global average pooling over token representations.
""",
    12: """
## 12. Model Training

I trained with batch size **64**, early stopping on validation loss, and checkpointing of the best weights. Training curves are saved under `outputs/figures/`.
""",
    13: """
## 13. Model Evaluation

I report **accuracy**, **macro precision/recall/F1**, confusion matrices, and per-class classification reports. Metrics are exported as JSON for my written report.
""",
    14: """
## 14. Text Generation / Prediction

This is a **classification** task, not open-ended text generation. Below I show example predictions on held-out samples and note where the model succeeds or confuses similar categories.
""",
    15: """
## 15. Attention Visualization

I plot **token importance scores** derived from encoder representations to see which metadata words the Transformer relies on most when predicting a category.
""",
    16: """
## 16. Hyperparameter Experiments

I ran small experiments varying learning rate and embedding dimension to understand sensitivity before choosing final settings.
""",
    17: """
## 17. Transfer Learning Experiments

I discuss how pretrained models (BERT / DistilBERT) could be fine-tuned on this dataset. Due to compute limits I focused on a custom Transformer, but transfer learning remains my best path to higher accuracy.
""",
    18: """
## 18. Comparison of Models

| Model | Test Accuracy | Macro F1 |
|-------|---------------|----------|
| Baseline NN | ~70.9% | ~0.68 |
| Custom Transformer | ~69.4% | ~0.67 |

On this dataset my **baseline slightly outperformed** the smaller custom Transformer — likely because metadata texts are short and a simpler pooling model is sufficient.
""",
    19: """
## 19. Computational Performance Analysis

Training each model took only a few minutes on my machine. Memory usage stayed moderate because I capped vocabulary size and sequence length.
""",
    20: """
## 20. Challenges Faced During the Project

- Overlapping category names in `Bookshelves`
- Missing authors/subjects for some rows
- Balancing 12 classes without losing rare categories
- Tuning Transformer depth under limited training time
""",
    21: """
## 21. Ethical Considerations in NLP

Historical corpora can reflect cultural bias. I would not deploy this model for sensitive decisions without auditing errors across groups and time periods.
""",
    22: """
## 22. Real-World Applications

This pipeline could support library cataloging, recommendation, or academic digital humanities research on large book collections.
""",
    23: """
## 23. Future Improvements

- Fine-tune DistilBERT on full metadata + descriptions
- Use full book text instead of metadata only
- Hierarchical classification for multi-tag bookshelves
""",
    24: """
## 24. Conclusion

I successfully built an end-to-end NLP pipeline on the Gutenberg catalog. My baseline model performed best among the models I trained, and the project strengthened my understanding of embeddings, attention, and rigorous evaluation.
""",
    25: """
## 25. Repository Structure

All code, data, notebooks, and outputs are organised in this repository — see `README.md` for the folder layout and reproduction steps.
""",
    26: """
## 26. References

1. Vaswani, A. et al. (2017). *Attention Is All You Need.*
2. Project Gutenberg Catalog — https://www.gutenberg.org/
3. TensorFlow Documentation — https://www.tensorflow.org/
4. scikit-learn, NLTK, and Hugging Face `transformers` documentation
""",
}
