"""Upload models/afriberta-best to the Hugging Face Hub.

Usage:
  hf auth login
  python scripts/upload_model.py --repo Muyizereberissa/kinyarwanda-sentiment-afriberta
"""

from __future__ import annotations

import argparse

from huggingface_hub import HfApi

from src.config import MODELS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="Hub repo id, e.g. Muyizereberissa/kinyarwanda-sentiment-afriberta",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    folder = MODELS_DIR / "afriberta-best"
    if not (folder / "model.safetensors").exists():
        raise SystemExit(f"Missing checkpoint at {folder}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo,
        repo_type="model",
        commit_message="Upload fine-tuned AfriBERTa Kinyarwanda sentiment model",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo}")
    print(f"On the Space, set environment variable MODEL_ID={args.repo}")


if __name__ == "__main__":
    main()
