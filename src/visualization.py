"""Plotting utilities for EDA and model evaluation."""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

from .config import OUTPUTS_FIGURES


def _save(fig, name: str, show: bool = True):
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_FIGURES / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_missing_values(df: pd.DataFrame, name: str = "missing_values.png", show: bool = True):
    missing = df.isnull().sum().sort_values(ascending=True)
    missing = missing[missing > 0]
    if missing.empty:
        missing = pd.Series({"(no missing cols)": 0})
    fig, ax = plt.subplots(figsize=(9, max(4, len(missing) * 0.35)))
    missing.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Missing Values per Column (Raw Catalog)")
    ax.set_xlabel("Count")
    return _save(fig, name, show=show)


def plot_labels_per_book(df: pd.DataFrame, name: str = "labels_per_book.png", show: bool = True):
    counts = df["num_labels"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#55A868", edgecolor="white")
    ax.set_xlabel("Number of bookshelf tags per book")
    ax.set_ylabel("Number of books")
    ax.set_title("How Many Categories Does Each Book Have?")
    return _save(fig, name)


def plot_multilabel_cooccurrence(
    label_lists: list[list[str]], class_names: list[str], name: str = "label_cooccurrence.png"
):
    n = len(class_names)
    idx = {c: i for i, c in enumerate(class_names)}
    mat = np.zeros((n, n))
    for tags in label_lists:
        ids = [idx[t] for t in tags if t in idx]
        for i in ids:
            for j in ids:
                mat[i, j] += 1
    short = [c.replace("Category: ", "")[:22] for c in class_names]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(mat, xticklabels=short, yticklabels=short, cmap="YlOrRd", ax=ax)
    ax.set_title("Category Co-occurrence (How Often Tags Appear Together)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return _save(fig, name)


def plot_class_distribution(df: pd.DataFrame, col: str = "category", name: str = "class_distribution.png"):
    counts = df[col].value_counts().head(15)
    labels = [c.replace("Category: ", "") for c in counts.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], counts.values[::-1], color="steelblue")
    ax.set_title("Primary Category Distribution (My Training Subset)")
    ax.set_xlabel("Number of books")
    return _save(fig, name)


def plot_text_length_distribution(df: pd.DataFrame, col: str = "text", name: str = "text_length_distribution.png"):
    lengths = df[col].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=50, color="teal", edgecolor="white", alpha=0.85)
    ax.axvline(lengths.median(), color="crimson", linestyle="--", label=f"median = {lengths.median():.0f}")
    ax.set_title("Text Length Distribution (Characters)")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Frequency")
    ax.legend()
    return _save(fig, name)


def plot_word_frequency(tokens: list[str], top_n: int = 20, name: str = "word_frequency.png"):
    freq = Counter(tokens)
    words, counts = zip(*freq.most_common(top_n))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(words[::-1], counts[::-1], color="coral")
    ax.set_title(f"Top {top_n} Most Frequent Tokens (After Cleaning)")
    ax.set_xlabel("Count")
    return _save(fig, name)


def plot_language_distribution(df: pd.DataFrame, name: str = "language_distribution.png"):
    counts = df["Language"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="#8172B3")
    ax.set_title("Top Languages in Full Catalog")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    return _save(fig, name)


def plot_training_history(history, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train", marker="o")
    axes[0].plot(history.history["val_loss"], label="val", marker="o")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    if "accuracy" in history.history:
        axes[1].plot(history.history["accuracy"], label="train", marker="o")
        axes[1].plot(history.history["val_accuracy"], label="val", marker="o")
        axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")
    fig.suptitle(f"Training Curves — {name}", y=1.02)
    fig.tight_layout()
    return _save(fig, f"{name}_training_curves.png")


def plot_training_history_multilabel(history, name: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    if "auc" in history.history:
        axes[1].plot(history.history["auc"], label="train")
        axes[1].plot(history.history["val_auc"], label="val")
        axes[1].set_title("AUC (multi-label)")
        axes[1].legend()
    if "binary_accuracy" in history.history:
        axes[2].plot(history.history["binary_accuracy"], label="train")
        axes[2].plot(history.history["val_binary_accuracy"], label="val")
        axes[2].set_title("Binary accuracy")
        axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.suptitle(f"Training — {name}", y=1.02)
    fig.tight_layout()
    return _save(fig, f"{name}_training_curves.png")


def plot_multilabel_metrics_bar(results: list[dict], name: str = "model_metrics_comparison.png"):
    """results: list of {'model': str, 'test_metrics': dict}"""
    models = [r["model"] for r in results]
    f1_micro = [r["test_metrics"]["f1_micro"] for r in results]
    f1_macro = [r["test_metrics"]["f1_macro"] for r in results]
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, f1_micro, w, label="F1 micro", color="#4C72B0")
    ax.bar(x + w / 2, f1_macro, w, label="F1 macro", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-label Test Performance — Model Comparison")
    ax.set_ylabel("Score")
    ax.legend()
    return _save(fig, name)


def plot_per_label_f1(y_true, y_pred, class_names: list[str], name: str = "per_label_f1.png"):
    scores = []
    for i in range(len(class_names)):
        scores.append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
    short = [c.replace("Category: ", "")[:24] for c in class_names]
    fig, ax = plt.subplots(figsize=(10, 5))
    order = np.argsort(scores)
    ax.barh(np.array(short)[order], np.array(scores)[order], color="#55A868")
    ax.set_xlim(0, 1)
    ax.set_title("Per-category F1 Score (Test Set)")
    ax.set_xlabel("F1")
    return _save(fig, name)


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
    ax.set_title("Confusion Matrix (Single-label)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return _save(fig, f"{name}_confusion_matrix.png")


def plot_sample_predictions(
    texts: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    indices: list[int] | None = None,
    n: int = 4,
    threshold: float = 0.5,
    name: str = "sample_predictions.png",
):
    if indices is None:
        indices = list(range(min(n, len(texts))))
    fig, axes = plt.subplots(len(indices), 1, figsize=(11, 2.8 * len(indices)))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        true_labs = [class_names[i].replace("Category: ", "") for i, v in enumerate(y_true[idx]) if v == 1]
        pred_labs = [
            class_names[i].replace("Category: ", "")
            for i, p in enumerate(y_prob[idx])
            if p >= threshold
        ]
        snippet = texts[idx][:120] + ("…" if len(texts[idx]) > 120 else "")
        ax.axis("off")
        ax.text(
            0, 0.85,
            f"Sample {idx + 1}: {snippet}",
            fontsize=9,
            wrap=True,
            transform=ax.transAxes,
        )
        ax.text(0, 0.45, f"True: {', '.join(true_labs) or '(none)'}", fontsize=9, color="#2166AC", transform=ax.transAxes)
        ax.text(0, 0.15, f"Pred: {', '.join(pred_labs) or '(none)'}", fontsize=9, color="#B2182B", transform=ax.transAxes)
    fig.suptitle("Example Predictions (Multi-label)", y=1.01)
    fig.tight_layout()
    return _save(fig, name)


def plot_generated_text_card(seed: str, generated: str, category: str, name: str = "text_generation_sample.png"):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    ax.text(0, 0.95, f"Prompt category: {category}", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0, 0.72, f"Seed: {seed}", fontsize=10, transform=ax.transAxes)
    ax.text(0, 0.45, f"Generated:\n{generated[:400]}", fontsize=9, family="monospace", transform=ax.transAxes, va="top")
    ax.set_title("Step 4 — Generated Text Sample", pad=12)
    return _save(fig, name)
