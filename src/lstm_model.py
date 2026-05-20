"""LSTM multi-label text classifier (Assignment Step 2)."""
from tensorflow import keras


def build_lstm_classifier(
    vocab_size: int,
    num_labels: int,
    max_len: int,
    embed_dim: int = 64,
    lstm_units: int = 64,
    dropout: float = 0.3,
) -> keras.Model:
    inputs = keras.Input(shape=(max_len,), dtype="int32")
    x = keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inputs)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units, return_sequences=False)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_labels, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="lstm_multilabel")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )
    return model
