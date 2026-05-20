"""Custom Transformer encoder for text classification."""
import numpy as np
import tensorflow as tf
from tensorflow import keras


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_len: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = np.zeros((max_len, embed_dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, :, :])

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


def transformer_encoder_block(
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    dropout: float = 0.1,
    name: str = "transformer_encoder_block",
) -> keras.Model:
    inputs = keras.Input(shape=(None, embed_dim))
    attn = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout
    )(inputs, inputs)
    attn = keras.layers.Dropout(dropout)(attn)
    out1 = keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn)

    ffn = keras.Sequential(
        [
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
            keras.layers.Dropout(dropout),
        ]
    )
    ffn_out = ffn(out1)
    out2 = keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_out)
    return keras.Model(inputs, out2, name=name)


def build_transformer_classifier(
    vocab_size: int,
    num_classes: int,
    max_len: int,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    ff_dim: int = 128,
    dropout: float = 0.2,
) -> keras.Model:
    token_ids = keras.Input(shape=(max_len,), dtype="int32", name="token_ids")
    x = keras.layers.Embedding(vocab_size, embed_dim)(token_ids)
    x = PositionalEncoding(max_len, embed_dim)(x)

    for i in range(num_layers):
        block = transformer_encoder_block(
            embed_dim, num_heads, ff_dim, dropout, name=f"encoder_block_{i}"
        )
        x = block(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(token_ids, outputs, name="custom_transformer_classifier")
