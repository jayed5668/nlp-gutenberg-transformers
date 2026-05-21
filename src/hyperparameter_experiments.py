"""Hyperparameter grid for Conv1D multi-label model."""
import json
import time

from .config import BATCH_SIZE, EPOCHS_BASELINE, MAX_SEQUENCE_LENGTH, OUTPUTS_METRICS
from .conv1d_model import build_conv1d_classifier
from .multilabel_eval import multilabel_metrics, predict_multilabel
from .multilabel_train import prepare_multilabel_tensors


EXPERIMENTS = [
    {"name": "lr_1e-2", "learning_rate": 1e-2, "embed_dim": 64},
    {"name": "lr_1e-3", "learning_rate": 1e-3, "embed_dim": 64},
    {"name": "embed_32", "learning_rate": 1e-3, "embed_dim": 32},
    {"name": "embed_128", "learning_rate": 1e-3, "embed_dim": 128},
]


def run_multilabel_experiments(epochs: int = 3):
    splits, _, meta = prepare_multilabel_tensors()
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    results = []

    for exp in EXPERIMENTS:
        model = build_conv1d_classifier(
            meta["vocab_size"], meta["num_classes"], MAX_SEQUENCE_LENGTH, embed_dim=exp["embed_dim"]
        )
        import tensorflow as tf
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=exp["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
        )
        start = time.time()
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=BATCH_SIZE, verbose=0,
        )
        pred, _ = predict_multilabel(model, X_test)
        metrics = multilabel_metrics(y_test, pred)
        results.append({
            "experiment": exp["name"],
            **exp,
            "train_time_sec": time.time() - start,
            "test_metrics": metrics,
        })

    OUTPUTS_METRICS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_METRICS / "hyperparameter_multilabel.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_experiments(epochs: int = 4):
    """Backward-compatible alias."""
    return run_multilabel_experiments(epochs)
