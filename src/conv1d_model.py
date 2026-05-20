"""Conv1D multi-label text classifier (Assignment Step 2)."""
from tensorflow import keras


def build_conv1d_classifier(
    vocab_size: int,
    num_labels: int,
    max_len: int,
    embed_dim: int = 64,
    filters: int = 128,
    kernel_size: int = 5,
    dropout: float = 0.3,
) -> keras.Model:
    inputs = keras.Input(shape=(max_len,), dtype="int32")
    x = keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inputs)
    x = keras.layers.Conv1D(filters, kernel_size, activation="relu", padding="same")(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_labels, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="conv1d_multilabel")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )
    return model
