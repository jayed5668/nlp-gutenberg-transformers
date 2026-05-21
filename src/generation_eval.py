"""Evaluate generated text with the trained multi-label classifier."""
import json

import numpy as np

from .config import OUTPUTS_METRICS
from .multilabel_eval import predict_multilabel
from .preprocessing import clean_text
from .text_generation import train_and_generate


def generate_and_classify(
    categories: list[str],
    lstm_model,
    vectorizer,
    class_names: list[str],
    threshold: float = 0.5,
) -> list[dict]:
    results = []
    for cat in categories:
        _, gen = train_and_generate(category=cat)
        text_clean = clean_text(gen["generated_text"])
        seq = vectorizer.encode_texts([text_clean])
        _, probs = predict_multilabel(lstm_model, seq, threshold=threshold)
        top = [
            class_names[i].replace("Category: ", "")
            for i in np.argsort(probs[0])[::-1][:3]
            if probs[0][i] >= threshold
        ]
        results.append({
            "category": cat,
            "seed": gen["seed"],
            "generated_text": gen["generated_text"],
            "classifier_labels": ", ".join(top) if top else "(below threshold)",
            "matches_prompt": cat.replace("Category: ", "") in ", ".join(top),
        })

    OUTPUTS_METRICS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_METRICS / "generation_classification.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
