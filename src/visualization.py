"""Plotting utilities for EDA and model evaluation."""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .config import OUTPUTS_FIGURES


def _save(fig, name: str):
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_FIGURES / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(df: pd.DataFrame, col: str = "category"):
    counts = df[col].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Class Distribution (Top Categories)")
    ax.set_xlabel("Count")
    return _save(fig, "class_distribution.png")


def plot_text_length_distribution(df: pd.DataFrame, col: str = "text"):
    lengths = df[col].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=50, color="teal", edgecolor="white")
    ax.set_title("Text Length Distribution")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Frequency")
    return _save(fig, "text_length_distribution.png")


def plot_word_frequency(tokens: list[str], top_n: int = 20):
    freq = Counter(tokens)
    words, counts = zip(*freq.most_common(top_n))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(words[::-1], counts[::-1], color="coral")
    ax.set_title("Most Common Words")
    return _save(fig, "word_frequency.png")


def plot_training_history(history, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    return _save(fig, f"{name}_training_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=[c[:30] for c in class_names],
        yticklabels=[c[:30] for c in class_names],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return _save(fig, f"{name}_confusion_matrix.png")
