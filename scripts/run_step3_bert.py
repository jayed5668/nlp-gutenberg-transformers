#!/usr/bin/env python3
"""Optional: train DistilBERT multi-label (Step 3). Requires network download."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bert_classifier import train_bert

if __name__ == "__main__":
    train_bert()
