"""Training helpers and dataset splitting."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

from .baseline_model import build_baseline_model, compile_model
from .config import (
    BATCH_SIZE,
    DROPOUT,
    EMBED_DIM,
    EPOCHS_BASELINE,
    EPOCHS_TRANSFORMER,
    FF_DIM,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    MODELS_DIR,
    NUM_HEADS,
    NUM_TRANSFORMER_LAYERS,
    OUTPUTS_METRICS,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
    VOCAB_SIZE,
)
from .data_loading import prepare_classification_dataset
from .preprocessing import TextVectorizer, preprocess_dataframe
from .transformer_model import build_transformer_classifier


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - relative_val, random_state=RANDOM_SEED, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_tensors(df: pd.DataFrame | None = None):
    if df is None:
        df = prepare_classification_dataset()
    df = preprocess_dataframe(df)
    vectorizer = TextVectorizer(max_vocab=VOCAB_SIZE, max_len=MAX_SEQUENCE_LENGTH)
    vectorizer.fit(df["text_clean"].tolist(), df["category"].tolist())
    X = vectorizer.encode_texts(df["text_clean"].tolist())
    y = vectorizer.encode_labels(df["category"].tolist())
    splits = split_data(X, y)
    meta = {
        "vocab_size": len(vectorizer.word_index),
        "num_classes": vectorizer.num_classes,
        "class_names": vectorizer.label_encoder.classes_.tolist(),
        "shapes": {
            "train": splits[0].shape,
            "val": splits[1].shape,
            "test": splits[3].shape,
        },
    }
    return splits, vectorizer, meta


def _callbacks(model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_accuracy", save_best_only=True
        ),
    ]


def train_baseline(splits=None, vectorizer=None, meta=None, epochs=EPOCHS_BASELINE):
    if splits is None:
        splits, vectorizer, meta = prepare_tensors()
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    model = build_baseline_model(
        meta["vocab_size"], meta["num_classes"], MAX_SEQUENCE_LENGTH, EMBED_DIM, DROPOUT
    )
    compile_model(model, LEARNING_RATE)
    path = MODELS_DIR / "baseline_best.keras"
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=_callbacks(path),
        verbose=1,
    )
    elapsed = time.time() - start
    result = {
        "model": "baseline",
        "train_time_sec": elapsed,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "test_accuracy": float(model.evaluate(X_test, y_test, verbose=0)[1]),
    }
    _save_metrics("baseline_metrics.json", result)
    return model, history, result, (X_test, y_test), vectorizer


def train_transformer(splits=None, vectorizer=None, meta=None, epochs=EPOCHS_TRANSFORMER):
    if splits is None:
        splits, vectorizer, meta = prepare_tensors()
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    model = build_transformer_classifier(
        meta["vocab_size"],
        meta["num_classes"],
        MAX_SEQUENCE_LENGTH,
        EMBED_DIM,
        NUM_HEADS,
        NUM_TRANSFORMER_LAYERS,
        FF_DIM,
        DROPOUT,
    )
    compile_model(model, LEARNING_RATE)
    path = MODELS_DIR / "transformer_best.keras"
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=_callbacks(path),
        verbose=1,
    )
    elapsed = time.time() - start
    result = {
        "model": "custom_transformer",
        "train_time_sec": elapsed,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "test_accuracy": float(model.evaluate(X_test, y_test, verbose=0)[1]),
    }
    _save_metrics("transformer_metrics.json", result)
    return model, history, result, (X_test, y_test), vectorizer


def _save_metrics(filename: str, payload: dict):
    OUTPUTS_METRICS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_METRICS / filename, "w") as f:
        json.dump(payload, f, indent=2)
