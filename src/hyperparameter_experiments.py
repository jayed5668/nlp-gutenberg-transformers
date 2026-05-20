"""Small hyperparameter grid for baseline model comparison."""
import json
import time
from pathlib import Path

import numpy as np
from tensorflow import keras

from .baseline_model import build_baseline_model, compile_model
from .config import BATCH_SIZE, MAX_SEQUENCE_LENGTH, OUTPUTS_METRICS
from .train import prepare_tensors


EXPERIMENTS = [
    {"name": "lr_1e-2", "learning_rate": 1e-2, "embed_dim": 64},
    {"name": "lr_1e-3", "learning_rate": 1e-3, "embed_dim": 64},
    {"name": "embed_32", "learning_rate": 1e-3, "embed_dim": 32},
    {"name": "embed_128", "learning_rate": 1e-3, "embed_dim": 128},
]


def run_experiments(epochs: int = 4) -> list[dict]:
    splits, vectorizer, meta = prepare_tensors()
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    results = []

    for exp in EXPERIMENTS:
        model = build_baseline_model(
            meta["vocab_size"],
            meta["num_classes"],
            MAX_SEQUENCE_LENGTH,
            embed_dim=exp["embed_dim"],
        )
        compile_model(model, learning_rate=exp["learning_rate"])
        start = time.time()
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        results.append(
            {
                "experiment": exp["name"],
                "learning_rate": exp["learning_rate"],
                "embed_dim": exp["embed_dim"],
                "val_accuracy": float(max(history.history["val_accuracy"])),
                "test_accuracy": float(test_acc),
                "train_time_sec": time.time() - start,
            }
        )

    OUTPUTS_METRICS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_METRICS / "hyperparameter_experiments.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
