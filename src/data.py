"""Load AfriSenti Kinyarwanda (kin) splits from the official SemEval 2023 TSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests

from src.config import AFRISENTI_KIN_BASE_URL, PROCESSED_DIR, RAW_DIR
from src.preprocess import preprocess_text


SPLIT_FILES = {
    "train": "train.tsv",
    "validation": "dev.tsv",
    "test": "test.tsv",
}


def download_raw_splits(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split, filename in SPLIT_FILES.items():
        dest = RAW_DIR / filename
        if dest.exists() and not force:
            continue
        url = f"{AFRISENTI_KIN_BASE_URL}/{filename}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_text(response.text, encoding="utf-8")


def _read_tsv(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tweet = row.get("tweet") or row.get("text") or ""
            label = (row.get("label") or "").strip().lower()
            rows.append((tweet, label))
    return rows


def build_processed_csv(force_download: bool = False) -> Dict[str, Path]:
    download_raw_splits(force=force_download)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for split, filename in SPLIT_FILES.items():
        raw_path = RAW_DIR / filename
        records = []
        for tweet, label in _read_tsv(raw_path):
            records.append(
                {
                    "text": preprocess_text(tweet),
                    "label": label,
                    "text_raw": tweet,
                }
            )
        df = pd.DataFrame(records)
        out = PROCESSED_DIR / f"{split}.csv"
        df.to_csv(out, index=False)
        paths[split] = out
    return paths


def load_splits() -> Dict[str, pd.DataFrame]:
    if not (PROCESSED_DIR / "train.csv").exists():
        build_processed_csv()
    splits = {}
    for split in SPLIT_FILES:
        splits[split] = pd.read_csv(PROCESSED_DIR / f"{split}.csv")
    return splits


def split_summary(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        counts = df["label"].value_counts().to_dict()
        rows.append({"split": name, "n": len(df), **counts})
    return pd.DataFrame(rows)
