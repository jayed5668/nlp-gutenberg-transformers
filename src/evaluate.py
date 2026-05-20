"""Model evaluation metrics and reports."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

from .config import OUTPUTS_METRICS


def evaluate_classifier(model: keras.Model, X_test, y_test, class_names: list[str]) -> dict:
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm,
    }
    return metrics, y_pred, y_prob


def save_evaluation(name: str, metrics: dict):
    OUTPUTS_METRICS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_METRICS / f"{name}_evaluation.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def build_comparison_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
