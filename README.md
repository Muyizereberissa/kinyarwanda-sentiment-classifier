# Kinyarwanda Sentiment Classifier

Three-class sentiment classification (**negative**, **neutral**, **positive**) for Kinyarwanda Twitter-style text, using the [AfriSenti](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment) `kin` subset from SemEval 2023 Task 12 (Muhammad et al., 2023).

**Capstone deliverables (fill in after submission):**

| Resource | Link |
|----------|------|
| GitHub | *this repository* |
| Live demo | *Hugging Face Spaces URL* |
| Demo video | *YouTube/Drive URL* |
| PDF report | `report/CAPSTONE_REPORT.md` → export to PDF |

## Problem

Monitor or analyze public opinion in Kinyarwanda social media (e.g. civic engagement, brand feedback) by automatically labeling tweet sentiment. The model is trained only on Kinyarwanda AfriSenti data—not a generic API wrapper.

## Dataset

| Split | File | Approx. size |
|-------|------|----------------|
| Train | `data/raw/train.tsv` | 3,302 |
| Validation | `data/raw/dev.tsv` | 827 |
| Test | `data/raw/test.tsv` | 1,034 |

**Total:** ~5,163 labeled tweets. Official splits from the SemEval 2023 repository ([`afrisent-semeval-2023`](https://github.com/afrisenti-semeval/afrisent-semeval-2023)).

Download and preprocess:

```bash
python -m src.prepare_data
```

Processed CSVs are written to `data/processed/`.

## Models & experiments

| ID | Model | Purpose |
|----|--------|---------|
| E0 | Majority class | Trivial baseline |
| E1 | TF-IDF (word 1–2) + logistic regression | Strong classical baseline |
| E2 | TF-IDF (char 3–5) + logistic regression | Spelling / OOV robustness |
| E3 | BiLSTM + trainable embeddings | Course RNN baseline |
| E4 | Fine-tuned `castorini/afriberta_base` | Main Transformer model |
| E5 | AfriBERTa + class weights | Imbalance experiment |

Results table: `results/experiments.csv` (updated when you train).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

## Training

```bash
python -m src.prepare_data
python -m src.train_baseline
python -m src.train_lstm --epochs 8
python -m src.train_transformer --epochs 3
python -m src.train_transformer --epochs 3 --class-weights   # E5
```

Or run the full pipeline:

```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

Metrics and confusion matrices are saved under `results/`. The best Transformer checkpoint is saved to `models/afriberta-best/` and registered in `models/production_model.json` for the app.

**GPU:** Transformer fine-tuning is much faster on CUDA or Google Colab; CPU works for baselines and short AfriBERTa runs.

## Evaluation & error analysis

```bash
python -m src.error_analysis
```

Outputs: `results/error_analysis.md` and `results/error_analysis_samples.json`.

## Local web app

```bash
streamlit run app/streamlit_app.py
```

Flow: **UI** → `src/predict.py` → tokenizer/model → softmax → label + probabilities.

## Deploy (Hugging Face Spaces)

1. Create a new **Streamlit** Space.
2. Push this repo (or copy `app/`, `src/`, `requirements.txt`, and your trained `models/afriberta-best/`).
3. Set `PYTHONPATH=.` in Space settings if needed.
4. For large models, upload the checkpoint to the Hugging Face Hub and set `production_model.json` to that repo id.

## Repository layout

```
├── app/streamlit_app.py
├── data/raw/              # downloaded TSVs (gitignored)
├── data/processed/        # CSV splits (gitignored)
├── models/                # trained artifacts (gitignored)
├── report/CAPSTONE_REPORT.md
├── results/
├── scripts/run_pipeline.sh
└── src/
    ├── prepare_data.py
    ├── train_baseline.py
    ├── train_lstm.py
    ├── train_transformer.py
    ├── predict.py
    └── error_analysis.py
```

## Citation

```bibtex
@inproceedings{muhammad2023afrisenti,
  title={AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages},
  author={Muhammad, Shamsuddeen Hassan and others},
  booktitle={Proceedings of EMNLP 2023},
  year={2023}
}
```

## License

Code: MIT (see `LICENSE`). AfriSenti data and `castorini/afriberta_base` follow their respective licenses on Hugging Face.
