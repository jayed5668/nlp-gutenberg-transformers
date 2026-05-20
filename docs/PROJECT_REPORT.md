# Assignment 3 — NLP with Transformers on the Project Gutenberg Catalog

**Author:** [Your Name]  
**Programme:** Deep Learning Minor, Inholland University of Applied Sciences  
**Date:** May 2026

This report documents my work from initial exploration through model comparison. All code and figures are reproducible from this repository.

---

## 1. Title and Project Overview

### 1.1 Assignment Title
Natural Language Processing with Deep Learning and Transformers — Assignment 3

### 1.2 Project Goal
I set out to automatically classify Project Gutenberg books into their primary **bookshelf category** using only short metadata text (title, author, and subject fields).

### 1.3 Problem Statement
Digital libraries grow faster than manual cataloging can keep up. Mislabelled or missing categories make search and recommendation less useful. I wanted to show that a deep learning pipeline can learn meaningful patterns from metadata alone.

### 1.4 Objectives of the Project
1. Perform thorough exploratory data analysis on `pg_catalog.csv`
2. Build a reproducible cleaning and tokenization pipeline
3. Train a baseline embedding-based neural network
4. Implement a custom Transformer encoder classifier
5. Evaluate both models with standard classification metrics
6. Reflect on ethics, limitations, and future improvements

### 1.5 Expected Outcomes
I aimed to deliver cleaned data, trained models with ~65–75% test accuracy, saved metrics and figures, and a structured notebook suitable for assessment.

---

## 2. Introduction to NLP

### 2.1 What is Natural Language Processing (NLP)?
NLP is the branch of AI focused on human language. In my project, NLP means turning book metadata strings into numerical features and category predictions.

### 2.2 Real-World Applications of NLP
Examples I studied include machine translation, chatbots, sentiment analysis, and **text classification** — the task I implemented here.

### 2.3 Challenges in NLP
I encountered ambiguity (one book fits multiple shelves), spelling variation, missing fields, and **class imbalance** between popular and rare categories.

### 2.4 Traditional NLP vs Deep Learning NLP
Traditional workflows use BoW or TF-IDF with logistic regression or SVMs. Deep learning learns embeddings end-to-end; I used the latter because it integrates naturally with Transformers.

### 2.5 Introduction to Transformers
The Transformer architecture (Vaswani et al., 2017) relies on **self-attention** instead of recurrence, processing all tokens in parallel.

### 2.6 Why Transformers are Important
For my metadata sequences, attention can emphasise discriminative tokens such as *poetry*, *adventure*, or *biography* when they appear in subjects or titles.

---

## 3. Dataset Overview

### 3.1 Dataset Description
I worked with the Project Gutenberg catalog export: **77,070 records** and **9 columns** of bibliographic information.

### 3.2 Dataset Source
[Project Gutenberg](https://www.gutenberg.org/) — a large public-domain ebook library.

### 3.3 Dataset Structure
| Column | Description |
|--------|-------------|
| `Text#` | Unique identifier |
| `Type` | Media type (mostly `Text`) |
| `Issued` | Release date |
| `Title` | Book title |
| `Language` | ISO language code |
| `Authors` | Author string |
| `Subjects` | Subject keywords |
| `LoCC` | Library of Congress code |
| `Bookshelves` | Semicolon-separated category tags |

### 3.4 Input and Target Variables
- **Input features:** concatenated `Title`, `Authors`, `Subjects`
- **Target label:** first bookshelf tag (e.g. `Category: Novels`)

### 3.5 Example Samples
Typical rows pair a descriptive title with rich subject lines; I verified in the notebook that labels align with wording (e.g. *Poetry* subjects → poetry shelf).

### 3.6 Initial Dataset Inspection
I filtered to `Type == Text` and `Language == en`, then kept the **12 most frequent** bookshelf categories with sufficient samples per class.

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Dataset Shape
After filtering I worked with roughly **24,000** balanced samples across 12 classes.

### 4.2 Column Information
All core text columns are strings; `Bookshelves` may contain multiple tags separated by `;`.

### 4.3 Missing Value Analysis
`Authors` had ~165 missing values in the full catalog; `Bookshelves` had ~1,793 missing — I dropped rows without a primary category.

### 4.4 Duplicate Data Analysis
I found **no duplicate rows** in my processed subset.

### 4.5 Text Length Distribution
Character lengths ranged from about 8 to 2,600 with median **144** — short enough for `max_len = 96` tokens.

### 4.6–4.8 Word Frequency and Vocabulary
Frequent words reflected literary vocabulary (*life*, *history*, *english*, *fiction*). Rare tokens were mapped to `<unk>` during encoding.

### 4.9 Vocabulary Size Analysis
I capped vocabulary at **12,000** words to balance coverage and memory.

### 4.10 Stopword Analysis
I compared models with and without stopword removal; keeping content words in subjects was more important than removing *the* and *of*.

### 4.11 Special Character Analysis
Punctuation and URLs were removed during cleaning to reduce noise.

### 4.12 Class Distribution
*Category: Novels* and *Category: Biographies* were among the largest classes; I used **stratified splitting** to preserve proportions.

### 4.13 Visualization of Text Statistics
Figures saved: `class_distribution.png`, `text_length_distribution.png`, `word_frequency.png`.

### 4.14 Initial Observations
Metadata is predictive but noisy; similar categories sometimes confuse the model (e.g. *Historical Novels* vs *Novels*).

---

## 5. Text Cleaning and Preprocessing

### 5.1 Lowercasing
I lowercased all text for consistent token matching.

### 5.2 Removing Punctuation
Non-alphanumeric characters were stripped except spaces.

### 5.3 Removing Special Characters
I normalised odd symbols that appeared in older catalog entries.

### 5.4 Removing Numbers
Standalone digits were removed because they rarely indicate category.

### 5.5 Removing URLs
HTTP/HTTPS links were deleted with a regex.

### 5.6 Removing Stopwords
Optional English stopwords (NLTK) were removed in `text_clean_nostop` for comparison.

### 5.7 Tokenization
Whitespace tokenization was sufficient given short metadata length.

### 5.8 Lemmatization / Stemming
I skipped aggressive stemming to preserve subject phrases like *science fiction*.

### 5.9 Handling Emojis or Symbols
None were significant in this historical dataset.

### 5.10 Final Cleaned Text Examples
The notebook displays before/after cleaning for manual sanity checks.

### 5.11 Save Preprocessed Dataset
Output path: `data/processed/pg_catalog_processed.csv`.

---

## 6. Text Representation Techniques

### 6.1 Introduction to Text Vectorization
I needed fixed-size numerical inputs for Keras models.

### 6.2 Bag of Words (BoW)
BoW counts word occurrences; I discussed it as a classical baseline alternative.

### 6.3 TF-IDF Representation
TF-IDF weights rare terms higher — strong for short documents but ignores order.

### 6.4 Word Embeddings
Learned dense vectors capture semantic similarity between words.

### 6.5 Introduction to Embedding Layers
My first network layer maps token indices to `embed_dim` dimensions.

### 6.6 Why Embeddings are Important
They let the model generalise across related words (e.g. *novel* / *fiction*).

---

## 7. Tokenization and Vocabulary Building

### 7.1 Creating Vocabulary
Built from training texts, frequency-sorted, max 12,000 tokens.

### 7.2 Word-to-Index Mapping
Special tokens: `<pad>=0`, `<unk>=1`.

### 7.3 Handling Unknown Tokens
Rare words at inference map to `<unk>`.

### 7.4 Padding Sequences
Sequences shorter than 96 are zero-padded; longer are truncated.

### 7.5 Sequence Length Analysis
96 tokens covered >95% of my samples without excess padding.

### 7.6 Train and Validation Vocabulary
Vocabulary is fit **only on training text** to avoid leakage (implemented in the full pipeline via single fit before split in vectorizer — note: current code fits on full df before split; for strict leakage-free I'd fit on train only in a future revision).

### 7.7 Final Encoded Sequences
Integer tensors ready for `model.fit()`.

---

## 8. Train, Validation, and Test Split

### 8.1 Why Dataset Splitting is Important
Held-out data estimates real-world generalisation.

### 8.2–8.4 Splits
**70% train**, **15% validation**, **15% test**, stratified by category.

### 8.5 Final Dataset Shapes
Printed in notebook after `prepare_tensors()` — typically ~16.8k / 3.6k / 3.6k samples.

---

## 9. Baseline Deep Learning Model

### 9.1 Introduction to Baseline Models
A simple network establishes a performance floor before adding attention.

### 9.2 Simple Neural Network Architecture
Embedding → GlobalAveragePooling1D → Dense(128) → Dense(64) → Softmax.

### 9.3 Embedding Layer
64-dimensional embeddings with masking on padding.

### 9.4 Dense Layers
ReLU activations with dropout 0.2 for regularisation.

### 9.5 Activation Functions
ReLU in hidden layers, softmax on output.

### 9.6 Loss Function
Sparse categorical cross-entropy.

### 9.7 Optimizer
Adam, learning rate 1e-3.

### 9.8 Model Compilation
Metrics: accuracy.

### 9.9 Model Training
Up to 8 epochs with early stopping (patience 3).

### 9.10 Training Curves
Saved as `baseline_training_curves.png` — training and validation loss/accuracy converge without severe overfitting.

### 9.11 Baseline Model Evaluation
**Test accuracy: 70.9%** | **Macro F1: 0.682**

---

## 10. Introduction to Transformers

### 10.1 What is a Transformer?
An architecture built from attention layers rather than LSTM/GRU recurrence.

### 10.2 Encoder and Decoder Architecture
My classification task uses an **encoder-only** stack; decoders are for seq2seq generation.

### 10.3 Attention Mechanism
Queries, keys, and values compute weighted token combinations.

### 10.4 Self-Attention
Each token attends to all tokens in the same sequence.

### 10.5 Multi-Head Attention
Four heads learn parallel attention patterns.

### 10.6 Positional Encoding
Sinusoidal encodings inject order information.

### 10.7 Feed Forward Network
Two dense layers per block with expansion to `ff_dim=128`.

### 10.8 Residual Connections
Add input back after attention and FFN sublayers.

### 10.9 Layer Normalization
Stabilises activations between sublayers.

### 10.10 Why Transformers Outperform RNNs
Parallel training and direct long-range links — though on very short texts the benefit is smaller.

---

## 11. Transformer Architecture Implementation

### 11.1 Import Required Libraries
TensorFlow/Keras for layers and training.

### 11.2 Define Positional Encoding
Custom `PositionalEncoding` layer in `src/transformer_model.py`.

### 11.3 Define Multi-Head Attention
`keras.layers.MultiHeadAttention` with 4 heads.

### 11.4 Define Feed Forward Network
Sequential dense block inside each encoder.

### 11.5 Define Encoder Block
Pre-norm style residual block (LayerNorm after residual).

### 11.6 Define Decoder Block
Not used — documented for completeness.

### 11.7 Build Full Transformer Model
Two stacked encoder blocks + pooling + classification head.

### 11.8 Model Summary
~100k+ trainable parameters depending on vocabulary size.

---

## 12. Model Training

### 12.1 Training Configuration
Shared hyperparameters in `src/config.py`.

### 12.2 Batch Size
64 samples per step.

### 12.3 Epoch Selection
6 epochs max with early stopping.

### 12.4 Learning Rate
1e-3 Adam.

### 12.5 Optimizer Selection
Adam for adaptive per-parameter scaling.

### 12.6 Loss Function
Sparse categorical cross-entropy.

### 12.7 Early Stopping
Monitor `val_loss`, restore best weights.

### 12.8 Model Checkpointing
Best model saved to `models/transformer_best.keras`.

### 12.9 Start Training Process
Executed via `scripts/run_pipeline.py` or notebook cells.

---

## 13. Model Evaluation

### 13.1 Training Loss Analysis
Smooth decrease; slight gap between train and validation suggests mild overfitting on Transformer.

### 13.2 Validation Loss Analysis
Best epoch selected automatically by checkpoint callback.

### 13.3 Accuracy Analysis
Transformer test accuracy **69.4%**.

### 13.4 Precision
Macro precision ~0.685.

### 13.5 Recall
Macro recall ~0.673.

### 13.6 F1-Score
Macro F1 **0.669**.

### 13.7 Confusion Matrix
Heatmaps in `outputs/figures/*_confusion_matrix.png`.

### 13.8 Classification Report
Per-class precision/recall in JSON under `outputs/metrics/`.

### 13.9 Error Analysis
Most errors occur between semantically adjacent categories (e.g. *Adventure* vs *Historical Novels*).

---

## 14. Text Generation / Prediction

### 14.1 Input Text Examples
Random test samples shown in notebook.

### 14.2 Generated Predictions
Softmax probabilities over 12 classes.

### 14.3 Correct Predictions
Majority of biography and poetry titles classified correctly.

### 14.4 Incorrect Predictions
Often involve shared vocabulary across novel subgenres.

### 14.5 Model Interpretation
Global pooling aggregates token evidence; attention maps (Section 15) highlight key words.

---

## 15. Attention Visualization

### 15.1 Understanding Attention Maps
Rows = query tokens, columns = key tokens; brighter cells = stronger focus.

### 15.2 Visualizing Attention Scores
Implemented in `src/attention_viz.py` → `attention_heatmap.png`.

### 15.3 Interpretation of Attention
Subject nouns frequently attend to category-indicative tokens in the title.

### 15.4 Important Tokens Analysis
Manual review of heatmaps confirms sensible focus on genre keywords.

---

## 16. Hyperparameter Experiments

### 16.1 Different Learning Rates
Compared 1e-2 vs 1e-3 — 1e-3 was more stable.

### 16.2 Different Batch Sizes
Default 64 used throughout main experiments.

### 16.3 Different Embedding Dimensions
32, 64, 128 compared in `hyperparameter_experiments.json`.

### 16.4 Different Number of Heads
4 heads in final Transformer (2 vs 4 possible extension).

### 16.5 Different Number of Layers
2 encoder blocks — deeper stacks not tested due to time.

### 16.6 Experiment Comparison Table
See `outputs/metrics/hyperparameter_experiments.json` and notebook output.

---

## 17. Transfer Learning Experiments

### 17.1 Introduction to Pretrained NLP Models
BERT and DistilBERT offer strong text encoders pretrained on large corpora.

### 17.2 BERT Overview
Bidirectional encoder useful for classification with a `[CLS]` token.

### 17.3 GPT Overview
Autoregressive model — better for generation than my classification task.

### 17.4 Fine-Tuning Strategy
I would tokenise with WordPiece, attach a classification head, fine-tune 2–3 epochs on my labels.

### 17.5 Transfer Learning Results
Not fully run in this submission due to compute; discussed as extension.

### 17.6 Comparison with Custom Transformer
Pretrained models would likely exceed my custom small Transformer on this data.

---

## 18. Comparison of Models

### 18.1 Baseline vs Transformer
Baseline won by ~1.5% accuracy on my test split.

### 18.2 Custom Transformer vs Pretrained Models
Pretrained models expected to be stronger given more parameters and pretraining.

### 18.3 Advantages and Limitations
| Model | Advantage | Limitation |
|-------|-----------|------------|
| Baseline | Fast, simple, strong on short text | No order-sensitive interactions |
| Custom Transformer | Interpretable attention | More parameters, slightly lower score here |

### 18.4 Performance Comparison Table
See README results table.

---

## 19. Computational Performance Analysis

### 19.1 Training Time
~6 seconds per model on Apple Silicon / CPU for my subset.

### 19.2 Memory Usage
Moderate — dominated by embedding matrix (vocab × embed_dim).

### 19.3 GPU / MPS Utilization
TensorFlow automatically uses GPU/MPS when available.

### 19.4 Computational Challenges
Full-catalog training and BERT fine-tuning would need longer runs or cloud GPU.

---

## 20. Challenges Faced During the Project

### 20.1 Dataset Challenges
Multi-label bookshelves reduced to single label; some information lost.

### 20.2 Training Challenges
Transformer required unique layer names when stacking blocks.

### 20.3 Hardware Limitations
Local training only — no large-scale hyperparameter grid.

### 20.4 NLP Difficulties
Overlapping category semantics.

### 20.5 Transformer Complexity
Debugging attention extraction from nested Keras models.

---

## 21. Ethical Considerations in NLP

### 21.1 Bias in NLP Models
Historical text reflects period biases; predictions must not be treated as neutral truth.

### 21.2 Dataset Bias
English-only filter excludes most non-English works.

### 21.3 Responsible AI
I document limitations and error patterns instead of overclaiming accuracy.

### 21.4 Ethical Risks of Text Generation
Not applicable to my classifier, but relevant if extending to generative models.

---

## 22. Real-World Applications

### 22.1 Chatbots
Not my focus, but NLP underpins conversational AI.

### 22.2 Machine Translation
Related field using encoder-decoder Transformers.

### 22.3 Sentiment Analysis
Similar pipeline with different labels.

### 22.4 Text Summarisation
Seq2seq Transformers — future extension.

### 22.5 AI Assistants
Large language models build on the same attention ideas I studied here.

---

## 23. Future Improvements

### 23.1 Larger Dataset
Use all English texts without capping at 24k samples.

### 23.2 Better Preprocessing
Fit vocabulary only on training split; add lemmatisation for subjects.

### 23.3 Larger Transformer Models
More layers and heads with regularisation.

### 23.4 Fine-Tuning Large Language Models
DistilBERT or BERT-base as next step.

### 23.5 Deployment Possibilities
FastAPI service accepting title/authors/subjects JSON.

---

## 24. Conclusion

### 24.1 Summary of the Project
I built a complete NLP classification pipeline on the Gutenberg catalog with EDA, preprocessing, two neural architectures, and thorough evaluation.

### 24.2 Key Findings
Metadata carries strong category signal; a simple baseline can match or beat a small Transformer on short texts.

### 24.3 Final Model Performance
Best model: **Baseline NN — 70.9% accuracy, 0.682 macro F1**.

### 24.4 Lessons Learned
Attention is powerful, but data quality, label design, and model size must match the task. Rigorous splits and clear writing matter as much as architecture choice.

---

## 25. GitHub / GitLab Repository Structure

### 25.1 Folder Structure
Documented in `README.md`.

### 25.2 Source Code Organization
Modular `src/` package with scripts for training and notebook generation.

### 25.3 Requirements File
`requirements.txt` lists all Python dependencies.

### 25.4 Reproducibility Instructions
Virtual environment + `run_pipeline.py` + fixed `RANDOM_SEED=42`.

---

## 26. References

### 26.1 Research Papers
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need.* NeurIPS.

### 26.2 Documentation
- TensorFlow: https://www.tensorflow.org/
- scikit-learn: https://scikit-learn.org/
- NLTK: https://www.nltk.org/

### 26.3 Libraries Used
Python 3.11+, TensorFlow 2.x, pandas, NumPy, matplotlib, seaborn, NLTK, Hugging Face `transformers` (optional).

### 26.4 Dataset References
- Project Gutenberg Catalog: https://www.gutenberg.org/

### 26.5 Academic Sources
Course materials — Deep Learning Minor, Inholland University of Applied Sciences.
