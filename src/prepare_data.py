"""Download AfriSenti kin TSVs and write processed CSV splits."""

from __future__ import annotations

import argparse

from src.data import build_processed_csv, load_splits, split_summary


def main():
    parser = argparse.ArgumentParser(description="Prepare AfriSenti Kinyarwanda data")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    paths = build_processed_csv(force_download=args.force_download)
    splits = load_splits()
    summary = split_summary(splits)
    print("Processed files:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("\nSplit summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
