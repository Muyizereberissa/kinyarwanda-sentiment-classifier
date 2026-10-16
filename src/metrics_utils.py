from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import ID2LABEL, LABELS, RESULTS_DIR


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
        ),
        "per_class": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }


def save_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    out_path: Path,
    title: str = "Confusion matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def append_experiment_row(row: Dict[str, Any], path: Optional[Path] = None) -> None:
    path = path or (RESULTS_DIR / "experiments.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)


def save_metrics_json(metrics: Dict[str, Any], name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out
