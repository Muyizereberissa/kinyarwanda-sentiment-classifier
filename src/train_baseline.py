"""Baselines: majority class and TF-IDF + logistic regression."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import LABELS, MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.data import load_splits
from src.metrics_utils import (
    append_experiment_row,
    compute_metrics,
    save_confusion_matrix,
    save_metrics_json,
)


class MajorityClassifier:
    def __init__(self, label: str):
        self.label = label

    def predict(self, texts):
        return [self.label] * len(texts)


def majority_baseline(train_labels, val_texts, val_labels, test_texts, test_labels):
    majority = Counter(train_labels).most_common(1)[0][0]
    clf = MajorityClassifier(majority)
    val_pred = clf.predict(val_texts)
    test_pred = clf.predict(test_texts)
    val_m = compute_metrics(val_labels, val_pred)
    test_m = compute_metrics(test_labels, test_pred)
    return clf, majority, val_m, test_m, val_pred, test_pred


def train_tfidf_lr(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    test_texts,
    test_labels,
    analyzer: str = "word",
    ngram_range=(1, 2),
    max_features: int = 50_000,
):
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    pipe.fit(train_texts, train_labels)
    val_pred = pipe.predict(val_texts)
    test_pred = pipe.predict(test_texts)
    val_m = compute_metrics(val_labels, val_pred)
    test_m = compute_metrics(test_labels, test_pred)
    return pipe, val_m, test_m, val_pred, test_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true", help="Run E0, E1, E2 baselines")
    args = parser.parse_args()

    splits = load_splits()
    train = splits["train"]
    val = splits["validation"]
    test = splits["test"]

    train_texts = train["text"].tolist()
    train_labels = train["label"].tolist()
    val_texts = val["text"].tolist()
    val_labels = val["label"].tolist()
    test_texts = test["text"].tolist()
    test_labels = test["label"].tolist()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # E0: majority
    _, majority_label, val_m0, test_m0, _, test_pred0 = majority_baseline(
        train_labels, val_texts, val_labels, test_texts, test_labels
    )
    save_metrics_json(
        {"validation": val_m0, "test": test_m0, "majority_label": majority_label},
        "baseline_majority",
    )
    append_experiment_row(
        {
            "id": "E0",
            "model": "Majority class",
            "key_change": f"always '{majority_label}'",
            "macro_f1_val": val_m0["macro_f1"],
            "macro_f1_test": test_m0["macro_f1"],
        }
    )
    save_confusion_matrix(
        test_labels,
        test_pred0,
        RESULTS_DIR / "confusion_matrix_E0_majority.png",
        title="E0 Majority (test)",
    )

    experiments = [
        ("E1", "word", (1, 2), "baseline_tfidf_word.joblib"),
        ("E2", "char", (3, 5), "baseline_tfidf_char.joblib"),
    ]

    best_test_f1 = -1.0
    best_path = None

    for eid, analyzer, ngrams, fname in experiments:
        pipe, val_m, test_m, _, test_pred = train_tfidf_lr(
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            test_texts,
            test_labels,
            analyzer=analyzer,
            ngram_range=ngrams,
        )
        out = MODELS_DIR / fname
        joblib.dump(pipe, out)
        save_metrics_json({"validation": val_m, "test": test_m}, f"baseline_tfidf_{analyzer}")
        append_experiment_row(
            {
                "id": eid,
                "model": "TF-IDF + Logistic Regression",
                "key_change": f"{analyzer} ngrams {ngrams}",
                "macro_f1_val": val_m["macro_f1"],
                "macro_f1_test": test_m["macro_f1"],
            }
        )
        save_confusion_matrix(
            test_labels,
            test_pred,
            RESULTS_DIR / f"confusion_matrix_{eid}.png",
            title=f"{eid} TF-IDF {analyzer} (test)",
        )
        if test_m["macro_f1"] > best_test_f1:
            best_test_f1 = test_m["macro_f1"]
            best_path = out

    if best_path:
        joblib.dump(joblib.load(best_path), MODELS_DIR / "baseline_best.joblib")
        meta = {"best_baseline": str(best_path.name), "macro_f1_test": best_test_f1}
        (MODELS_DIR / "baseline_best_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    print("Baselines complete. See results/experiments.csv")


if __name__ == "__main__":
    main()
