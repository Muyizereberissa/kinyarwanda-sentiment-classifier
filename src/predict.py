"""Unified inference for baseline, BiLSTM, and fine-tuned Transformer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, XLMRobertaTokenizer

from src.config import LABELS, MODELS_DIR
from src.preprocess import preprocess_text
from src.train_lstm import BiLSTMClassifier, Vocab


def _load_production_spec() -> Dict:
    # Set MODEL_ID on Hugging Face Spaces to a Hub repo such as
    # username/kinyarwanda-sentiment-afriberta
    hub_id = os.environ.get("MODEL_ID", "").strip()
    if hub_id:
        return {"type": "transformer", "path": hub_id}

    local_best = MODELS_DIR / "afriberta-best"
    spec_path = MODELS_DIR / "production_model.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        path = Path(spec.get("path", ""))
        if path.exists():
            return spec
    if local_best.exists():
        return {"type": "transformer", "path": str(local_best)}
    if (MODELS_DIR / "baseline_best.joblib").exists():
        return {"type": "baseline", "path": str(MODELS_DIR / "baseline_best.joblib")}
    raise FileNotFoundError(
        "No trained model found. Train locally, or set the MODEL_ID environment "
        "variable to a Hugging Face model repo that contains the fine-tuned checkpoint."
    )


class Predictor:
    def __init__(self, spec: Optional[Dict] = None):
        self.spec = spec or _load_production_spec()
        self.model_type = self.spec["type"]
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type == "baseline":
            self.pipe = joblib.load(self.spec["path"])
        elif self.model_type == "bilstm":
            ckpt = torch.load(self.spec["path"], map_location=self._device)
            vocab = Vocab([])
            vocab.itos = ckpt["vocab_itos"]
            vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
            self.vocab = vocab
            self.max_len = ckpt["max_len"]
            cfg = ckpt.get("config", {})
            self.model = BiLSTMClassifier(
                len(vocab.itos),
                cfg.get("embed_dim", 128),
                cfg.get("hidden_dim", 128),
                len(LABELS),
            )
            self.model.load_state_dict(ckpt["model_state"])
            self.model.to(self._device)
            self.model.eval()
        elif self.model_type == "transformer":
            path = self.spec["path"]
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(path)
            if getattr(self.tokenizer, "model_max_length", None) in (
                None,
                1e30,
                1000000000000000019889624719755765974,
            ):
                self.tokenizer.model_max_length = 512
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.to(self._device)
            self.model.eval()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict_one(self, text: str) -> Tuple[str, Dict[str, float]]:
        text = preprocess_text(text)
        if self.model_type == "baseline":
            label = self.pipe.predict([text])[0]
            if hasattr(self.pipe, "predict_proba"):
                proba = self.pipe.predict_proba([text])[0]
                scores = {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}
            else:
                scores = {label: 1.0}
            return label, scores

        if self.model_type == "bilstm":
            ids = torch.tensor(
                [self.vocab.encode(text, self.max_len)],
                dtype=torch.long,
                device=self._device,
            )
            with torch.no_grad():
                logits = self.model(ids)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            idx = int(proba.argmax())
            scores = {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}
            return LABELS[idx], scores

        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(proba.argmax())
        scores = {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}
        return LABELS[idx], scores

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        return [self.predict_one(t) for t in texts]


def predict(text: str) -> Tuple[str, Dict[str, float]]:
    return Predictor().predict_one(text)
