"""Sample misclassified test examples for the report and demo video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR
from src.data import load_splits
from src.predict import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-success", type=int, default=8)
    parser.add_argument("--n-errors", type=int, default=12)
    args = parser.parse_args()

    test = load_splits()["test"]
    predictor = Predictor()
    rows = []
    for _, row in test.iterrows():
        pred, scores = predictor.predict_one(row["text"])
        rows.append(
            {
                "text": row["text"],
                "true_label": row["label"],
                "pred_label": pred,
                "confidence": scores.get(pred, 0.0),
                "correct": pred == row["label"],
            }
        )
    df = pd.DataFrame(rows)
    successes = df[df["correct"]].head(args.n_success)
    errors = df[~df["correct"]].head(args.n_errors)

    out_md = RESULTS_DIR / "error_analysis.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Error analysis (test set samples)",
        "",
        "## Correct predictions",
        "",
    ]
    for _, r in successes.iterrows():
        lines.append(f"- **{r['pred_label']}** ({r['confidence']:.2f}): {r['text'][:200]}")
    lines.extend(["", "## Misclassifications", ""])
    for _, r in errors.iterrows():
        lines.append(
            f"- True **{r['true_label']}** → Pred **{r['pred_label']}** "
            f"({r['confidence']:.2f}): {r['text'][:200]}"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    (RESULTS_DIR / "error_analysis_samples.json").write_text(
        json.dumps({"successes": successes.to_dict(orient="records"), "errors": errors.to_dict(orient="records")}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
