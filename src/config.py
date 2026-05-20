"""Project configuration for Assignment 3 — NLP with Transformers."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "pg_catalog.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "pg_catalog_processed.csv"
OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_METRICS = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

# Task: classify primary bookshelf category from metadata text
TEXT_COLUMNS = ["Title", "Authors", "Subjects"]
TARGET_COLUMN = "category"
TOP_N_CATEGORIES = 12
MIN_SAMPLES_PER_CLASS = 200
MAX_SAMPLES = 24000
MAX_SEQUENCE_LENGTH = 96
VOCAB_SIZE = 12000
EMBED_DIM = 64
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS_BASELINE = 8
EPOCHS_TRANSFORMER = 6
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BERT_MAX_SAMPLES = 2000
BERT_EPOCHS = 1
