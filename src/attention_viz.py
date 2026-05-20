"""Token importance visualization for the Transformer encoder."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import OUTPUTS_FIGURES


def build_token_importance_model(classifier: keras.Model) -> keras.Model:
    """Return model outputs per-token L2 norms after the last encoder block."""
    last_encoder = classifier.get_layer("encoder_block_1")
    inputs = classifier.input
    x = classifier.layers[1](inputs)
    x = classifier.layers[2](x)
    x = last_encoder(x)
    importance = keras.layers.Lambda(lambda t: keras.ops.norm(t, axis=-1))(x)
    return keras.Model(inputs, importance, name="token_importance")


def plot_token_importance(
    importance: np.ndarray,
    tokens: list[str],
    title: str = "Token importance (encoder representation norm)",
    filename: str = "attention_heatmap.png",
) -> Path:
    scores = importance[0][: len(tokens)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(tokens)), scores, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Importance score")
    ax.set_title(title)
    fig.tight_layout()
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_FIGURES / filename
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_sample_attention(
    classifier: keras.Model,
    X_sample: np.ndarray,
    tokens: list[str],
    filename: str = "attention_heatmap.png",
) -> Path:
    """Visualise which tokens carry the strongest encoder signal."""
    extractor = build_token_importance_model(classifier)
    scores = extractor.predict(X_sample, verbose=0)
    return plot_token_importance(scores, tokens, filename=filename)
