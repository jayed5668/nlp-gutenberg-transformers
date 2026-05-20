"""Category-conditioned text generation (Assignment Step 4)."""
import json
import random

import numpy as np
from tensorflow import keras

from .config import MODELS_DIR, OUTPUTS_METRICS, RANDOM_SEED
from .multilabel_data import prepare_multilabel_dataset
from .preprocessing import preprocess_dataframe


def train_and_generate(
    category: str = "Category: History - American",
    max_texts: int = 80,
    max_chars: int = 60,
    epochs: int = 5,
):
    """Train a compact char-LSTM on metadata snippets and generate sample text."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df, class_names = prepare_multilabel_dataset()
    df = preprocess_dataframe(df)
    subset = df[df["label_list"].apply(lambda tags: category in tags)]
    if len(subset) < 20:
        category = class_names[0]
        subset = df[df["label_list"].apply(lambda tags: category in tags)]
    texts = [t[:max_chars] for t in subset["text_clean"].head(max_texts).tolist()]

    chars = sorted(set("".join(texts)))
    char2id = {"<pad>": 0}
    for i, c in enumerate(chars, start=1):
        char2id[c] = i
    id2char = {i: c for c, i in char2id.items()}
    vocab_size = len(char2id)
    seq_len = 40

    X, y = [], []
    for text in texts:
        ids = [char2id.get(c, 1) for c in text]
        for i in range(1, len(ids)):
            window = ids[max(0, i - seq_len) : i]
            padded = [0] * (seq_len - len(window)) + window
            X.append(padded)
            y.append(ids[i])
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    inp = keras.Input(shape=(seq_len,), dtype="int32")
    x = keras.layers.Embedding(vocab_size, 32)(inp)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dense(vocab_size, activation="softmax")(x)
    model = keras.Model(inp, x)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(X, y, epochs=epochs, batch_size=64, verbose=0, validation_split=0.1)

    seed = texts[0][:25] if texts else category
    generated = list(seed)
    for _ in range(80):
        window = [char2id.get(c, 1) for c in generated[-seq_len:]]
        padded = [0] * (seq_len - len(window)) + window
        pred = model.predict(np.array([padded]), verbose=0)[0]
        generated.append(id2char[int(np.argmax(pred))])

    result = {
        "prompt_category": category,
        "seed": seed,
        "generated_text": "".join(generated),
        "note": "Syntactic/style demonstration on catalog metadata (assignment Step 4).",
    }
    OUTPUTS_METRICS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_METRICS / "text_generation_samples.json", "w") as f:
        json.dump(result, f, indent=2)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "text_generator.keras")
    return model, result
