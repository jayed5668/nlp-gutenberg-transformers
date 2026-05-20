"""Train Conv1D and LSTM multi-label models."""
import json
import time

import numpy as np
from sklearn.metrics import f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from tensorflow import keras

from .config import (
    BATCH_SIZE,
    EPOCHS_BASELINE,
    MAX_SEQUENCE_LENGTH,
    MODELS_DIR,
    OUTPUTS_METRICS,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
    VOCAB_SIZE,
)
from .conv1d_model import build_conv1d_classifier
from .lstm_model import build_lstm_classifier
from .multilabel_data import labels_to_multihot, prepare_multilabel_dataset
from .preprocessing import TextVectorizer, preprocess_dataframe


def _split_multilabel(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE, random_state=RANDOM_SEED
    )
    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - relative_val, random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_multilabel_tensors(df=None):
    df, class_names = prepare_multilabel_dataset(df)
    df = preprocess_dataframe(df, text_col="text")
    vectorizer = TextVectorizer(max_vocab=VOCAB_SIZE, max_len=MAX_SEQUENCE_LENGTH)
    # Fit vocabulary on text only
    vectorizer.fit(df["text_clean"].tolist(), df["label_list"].str[0].tolist())
    X = vectorizer.encode_texts(df["text_clean"].tolist())
    y = labels_to_multihot(df["label_list"].tolist(), class_names)
    splits = _split_multilabel(X, y)
    meta = {
        "vocab_size": len(vectorizer.word_index),
        "num_classes": len(class_names),
        "class_names": class_names,
    }
    return splits, vectorizer, meta


def _evaluate_multilabel(model, X_test, y_test, threshold=0.5) -> dict:
    probs = model.predict(X_test, verbose=0)
    preds = (probs >= threshold).astype(int)
    return {
        "hamming_loss": float(hamming_loss(y_test, preds)),
        "f1_micro": float(f1_score(y_test, preds, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "subset_accuracy": float((preds == y_test).all(axis=1).mean()),
    }


def _train_one(build_fn, splits, meta, name: str, epochs=EPOCHS_BASELINE):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    model = build_fn(meta["vocab_size"], meta["num_classes"], MAX_SEQUENCE_LENGTH)
    path = MODELS_DIR / f"{name}_multilabel_best.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(path), monitor="val_auc", mode="max", save_best_only=True),
    ]
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    metrics = _evaluate_multilabel(model, X_test, y_test)
    result = {
        "model": name,
        "train_time_sec": time.time() - start,
        "test_metrics": metrics,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    OUTPUTS_METRICS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_METRICS / f"{name}_multilabel.json", "w") as f:
        json.dump(result, f, indent=2)
    return model, history, result


def train_conv1d(splits=None, meta=None, epochs=5):
    if splits is None:
        splits, _, meta = prepare_multilabel_tensors()
    return _train_one(build_conv1d_classifier, splits, meta, "conv1d", epochs)


def train_lstm(splits=None, meta=None, epochs=5):
    if splits is None:
        splits, _, meta = prepare_multilabel_tensors()
    return _train_one(build_lstm_classifier, splits, meta, "lstm", epochs)
