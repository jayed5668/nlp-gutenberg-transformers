"""Baseline embedding + dense classifier."""
import tensorflow as tf
from tensorflow import keras


def build_baseline_model(
    vocab_size: int,
    num_classes: int,
    max_len: int,
    embed_dim: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    inputs = keras.Input(shape=(max_len,), dtype="int32", name="token_ids")
    x = keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inputs)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="baseline_text_classifier")


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
