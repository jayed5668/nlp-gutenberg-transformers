"""Multi-label bookshelf classification dataset (Assignment Step 1–2)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from .config import (
    DATA_PROCESSED,
    DATA_RAW,
    MAX_SAMPLES,
    MIN_SAMPLES_PER_CLASS,
    RANDOM_SEED,
    TEXT_COLUMNS,
    TOP_N_CATEGORIES,
)
from .data_loading import build_text_row, load_raw_catalog


def parse_bookshelf_tags(bookshelves: str) -> list[str]:
    if pd.isna(bookshelves) or not str(bookshelves).strip():
        return []
    return [t.strip() for t in str(bookshelves).split(";") if t.strip()]


def prepare_multilabel_dataset(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """English text rows with multi-hot labels for top bookshelf categories."""
    if df is None:
        df = load_raw_catalog()

    work = df.copy()
    work = work[work["Type"] == "Text"]
    work = work[work["Language"] == "en"]
    work = work.dropna(subset=["Title", "Bookshelves"])
    work["tags"] = work["Bookshelves"].apply(parse_bookshelf_tags)
    work = work[work["tags"].map(len) > 0]

    mlb = MultiLabelBinarizer()
    all_tags = work["tags"].tolist()
    mlb.fit(all_tags)
    counts = mlb.transform(all_tags).sum(axis=0)
    tag_counts = pd.Series(counts, index=mlb.classes_).sort_values(ascending=False)
    selected = tag_counts.head(TOP_N_CATEGORIES).index.tolist()

    work["tags"] = work["tags"].apply(lambda ts: [t for t in ts if t in selected])
    work = work[work["tags"].map(len) > 0]
    work["text"] = work.apply(build_text_row, axis=1)
    work = work[work["text"].str.len() > 10]

    # Balance by primary tag (first tag) for manageable training
    work["primary"] = work["tags"].str[0]
    balanced = []
    for cat in selected:
        subset = work[work["primary"] == cat]
        if len(subset) == 0:
            subset = work[work["tags"].apply(lambda ts, c=cat: c in ts)]
        if len(subset) == 0:
            continue
        n = min(len(subset), max(MIN_SAMPLES_PER_CLASS, 400))
        balanced.append(subset.sample(n=min(n, len(subset)), random_state=RANDOM_SEED))
    work = pd.concat(balanced, ignore_index=True)

    if len(work) > MAX_SAMPLES:
        work = work.sample(MAX_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

    work["label_list"] = work["tags"]
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    out = work[["text", "label_list", "Title", "Authors", "Subjects", "Bookshelves"]].copy()
    out.to_csv(DATA_PROCESSED.parent / "pg_catalog_multilabel.csv", index=False)
    return work, selected


def labels_to_multihot(label_lists: list[list[str]], class_names: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(class_names)}
    y = np.zeros((len(label_lists), len(class_names)), dtype=np.float32)
    for row, tags in enumerate(label_lists):
        for tag in tags:
            if tag in idx:
                y[row, idx[tag]] = 1.0
    return y
