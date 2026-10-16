#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m src.prepare_data
python -m src.train_baseline
python -m src.train_lstm --epochs 8
python -m src.train_transformer --epochs 3
python -m src.train_transformer --epochs 3 --class-weights
python -m src.error_analysis

echo "Done. Launch app: streamlit run app/streamlit_app.py"
