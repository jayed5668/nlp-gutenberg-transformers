"""Text cleaning and tokenization utilities."""
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import MAX_SEQUENCE_LENGTH, RANDOM_SEED, VOCAB_SIZE


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
PUNCT_PATTERN = re.compile(r"[^\w\s]")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = PUNCT_PATTERN.sub(" ", text)
    text = NUMBER_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    if remove_stopwords:
        try:
            from nltk.corpus import stopwords

            stops = set(stopwords.words("english"))
            text = " ".join(w for w in text.split() if w not in stops)
        except Exception:
            pass
    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    out = df.copy()
    out["text_clean"] = out[text_col].apply(lambda t: clean_text(t, remove_stopwords=False))
    out["text_clean_nostop"] = out[text_col].apply(lambda t: clean_text(t, remove_stopwords=True))
    return out


class TextVectorizer:
    """Word-level vocabulary with padding for Keras models."""

    def __init__(self, max_vocab: int = VOCAB_SIZE, max_len: int = MAX_SEQUENCE_LENGTH):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word_index: dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.label_encoder = LabelEncoder()

    def fit(self, texts: list[str], labels: list[str]) -> "TextVectorizer":
        freq: dict[str, int] = {}
        for text in texts:
            for token in text.split():
                freq[token] = freq.get(token, 0) + 1
        sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        for token, _ in sorted_tokens[: self.max_vocab - 2]:
            self.word_index[token] = len(self.word_index)
        self.label_encoder.fit(labels)
        return self

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        sequences = []
        for text in texts:
            ids = [self.word_index.get(tok, 1) for tok in text.split()[: self.max_len]]
            if len(ids) < self.max_len:
                ids += [0] * (self.max_len - len(ids))
            else:
                ids = ids[: self.max_len]
            sequences.append(ids)
        return np.array(sequences, dtype=np.int32)

    def encode_labels(self, labels: list[str]) -> np.ndarray:
        return self.label_encoder.transform(labels)

    @property
    def num_classes(self) -> int:
        return len(self.label_encoder.classes_)

    def decode_label(self, index: int) -> str:
        return self.label_encoder.inverse_transform([index])[0]
