#!/usr/bin/env python3
"""Run full Assignment 3 pipeline (EDA → train → evaluate)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collections import Counter

from src.data_loading import load_raw_catalog, prepare_classification_dataset
from src.evaluate import evaluate_classifier, save_evaluation
from src.preprocessing import preprocess_dataframe
from src.train import prepare_tensors, train_baseline, train_transformer
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_text_length_distribution,
    plot_training_history,
    plot_word_frequency,
)


def main():
    print("Loading and preparing dataset...")
    raw = load_raw_catalog()
    df = prepare_classification_dataset(raw)
    df = preprocess_dataframe(df)
    plot_class_distribution(df)
    plot_text_length_distribution(df)
    tokens = []
    for text in df["text_clean"].head(5000):
        tokens.extend(text.split())
    plot_word_frequency(tokens)

    print("Encoding and splitting...")
    splits, vectorizer, meta = prepare_tensors(df)
    class_names = meta["class_names"]

    print("Training baseline...")
    b_model, b_hist, b_res, b_test, _ = train_baseline(splits, vectorizer, meta)
    plot_training_history(b_hist, "baseline")
    b_metrics, b_pred, _ = evaluate_classifier(b_model, *b_test, class_names)
    save_evaluation("baseline", b_metrics)
    plot_confusion_matrix(b_test[1], b_pred, class_names, "baseline")

    print("Training custom transformer...")
    t_model, t_hist, t_res, t_test, _ = train_transformer(splits, vectorizer, meta)
    plot_training_history(t_hist, "transformer")
    t_metrics, t_pred, _ = evaluate_classifier(t_model, *t_test, class_names)
    save_evaluation("transformer", t_metrics)
    plot_confusion_matrix(t_test[1], t_pred, class_names, "transformer")

    print("\n=== Results ===")
    print(f"Baseline test accuracy: {b_res['test_accuracy']:.4f}")
    print(f"Transformer test accuracy: {t_res['test_accuracy']:.4f}")
    print(f"Baseline F1 (macro): {b_metrics['f1_macro']:.4f}")
    print(f"Transformer F1 (macro): {t_metrics['f1_macro']:.4f}")
    print("Figures saved to outputs/figures/")
    print("Metrics saved to outputs/metrics/")


if __name__ == "__main__":
    main()
