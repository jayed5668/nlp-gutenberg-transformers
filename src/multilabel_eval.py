"""Evaluation helpers for multi-label models."""
import numpy as np
from sklearn.metrics import f1_score, hamming_loss


def predict_multilabel(model, X, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    probs = model.predict(X, verbose=0)
    preds = (probs >= threshold).astype(int)
    return preds, probs


def multilabel_metrics(y_true, y_pred) -> dict:
    return {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float((y_pred == y_true).all(axis=1).mean()),
    }
