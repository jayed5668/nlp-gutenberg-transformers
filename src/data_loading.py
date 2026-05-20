"""Load and prepare the Project Gutenberg catalog dataset."""
import pandas as pd

from .config import (
    DATA_PROCESSED,
    DATA_RAW,
    MIN_SAMPLES_PER_CLASS,
    MAX_SAMPLES,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEXT_COLUMNS,
    TOP_N_CATEGORIES,
)


def load_raw_catalog(path=None) -> pd.DataFrame:
    path = path or DATA_RAW
    return pd.read_csv(path)


def extract_primary_category(bookshelves: str) -> str | None:
    if pd.isna(bookshelves) or not str(bookshelves).strip():
        return None
    return str(bookshelves).split(";")[0].strip()


def build_text_row(row: pd.Series) -> str:
    parts = [str(row[c]) for c in TEXT_COLUMNS if pd.notna(row.get(c))]
    return " ".join(parts).strip()


def prepare_classification_dataset(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Filter English text entries and top bookshelf categories."""
    if df is None:
        df = load_raw_catalog()

    work = df.copy()
    work = work[work["Type"] == "Text"]
    work = work[work["Language"] == "en"]
    work[TARGET_COLUMN] = work["Bookshelves"].apply(extract_primary_category)
    work = work.dropna(subset=[TARGET_COLUMN, "Title"])

    counts = work[TARGET_COLUMN].value_counts()
    top_categories = counts.head(TOP_N_CATEGORIES).index.tolist()
    work = work[work[TARGET_COLUMN].isin(top_categories)]

    balanced = []
    for cat in top_categories:
        subset = work[work[TARGET_COLUMN] == cat]
        n = min(len(subset), max(MIN_SAMPLES_PER_CLASS, len(subset)))
        balanced.append(subset.sample(n=min(n, len(subset)), random_state=RANDOM_SEED))
    work = pd.concat(balanced, ignore_index=True)

    if len(work) > MAX_SAMPLES:
        work = work.sample(MAX_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

    work["text"] = work.apply(build_text_row, axis=1)
    work = work[work["text"].str.len() > 10].reset_index(drop=True)

    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(DATA_PROCESSED, index=False)
    return work
