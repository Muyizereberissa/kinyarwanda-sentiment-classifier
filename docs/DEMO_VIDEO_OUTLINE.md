# Demo video outline (7–10 minutes)

1. **Problem & language (1 min)** — Kinyarwanda Twitter sentiment; use case (monitoring public opinion).
2. **Dataset (1.5 min)** — AfriSenti `kin`; ~3.3k / 827 / ~1k splits; preprocessing; show `python -m src.prepare_data`.
3. **Baselines & models (2 min)** — E0–E2 in `train_baseline.py`; BiLSTM in `train_lstm.py`; AfriBERTa fine-tune in `train_transformer.py` (show hyperparameters).
4. **Results (2 min)** — `results/experiments.csv`, confusion matrix PNG, macro-F1 vs accuracy.
5. **Error analysis (1.5 min)** — Examples from `results/error_analysis.md` (success + failure).
6. **Live app (1.5 min)** — `streamlit run app/streamlit_app.py`; trace UI → `predict.py` → model.
7. **Limitations (30 s)** — Twitter domain, 2-epoch CPU run, not for high-stakes decisions.
