"""DistilBERT multi-label classifier with PyTorch (Assignment Step 3)."""
import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    BERT_EPOCHS,
    BERT_MAX_SAMPLES,
    MODELS_DIR,
    OUTPUTS_METRICS,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
)
from .multilabel_data import labels_to_multihot, prepare_multilabel_dataset
from .preprocessing import preprocess_dataframe


class TextMultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_bert(epochs: int = BERT_EPOCHS, batch_size: int = 8, max_samples: int = BERT_MAX_SAMPLES):
    df, class_names = prepare_multilabel_dataset()
    df = preprocess_dataframe(df, text_col="text")
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=RANDOM_SEED).reset_index(drop=True)

    texts = df["text_clean"].tolist()
    y = labels_to_multihot(df["label_list"].tolist(), class_names)

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, y, test_size=TEST_SIZE + VAL_SIZE, random_state=RANDOM_SEED
    )
    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - relative_val, random_state=RANDOM_SEED
    )

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(class_names),
        problem_type="multi_label_classification",
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        TextMultiLabelDataset(X_train, y_train, tokenizer), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TextMultiLabelDataset(X_val, y_val, tokenizer), batch_size=batch_size
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    start = time.time()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = model(**batch)
                val_losses.append(loss_fn(outputs.logits, labels).item())
        print(f"Epoch {epoch + 1}/{epochs} — val loss: {np.mean(val_losses):.4f}")

    model.eval()
    test_enc = tokenizer(
        X_test, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in test_enc.items()}).logits
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "hamming_loss": float(hamming_loss(y_test, preds)),
        "f1_micro": float(f1_score(y_test, preds, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
    }
    result = {
        "model": model_name,
        "train_time_sec": time.time() - start,
        "test_metrics": metrics,
        "num_train_samples": len(X_train),
    }
    OUTPUTS_METRICS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_METRICS / "bert_multilabel.json", "w") as f:
        json.dump(result, f, indent=2)
    save_dir = MODELS_DIR / "distilbert_multilabel"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model, tokenizer, result
