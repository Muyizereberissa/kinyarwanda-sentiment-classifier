"""Fine-tune AfriBERTa (or similar) on AfriSenti Kinyarwanda."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import DEFAULT_TRANSFORMER, LABEL2ID, LABELS, MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.data import load_splits
from src.metrics_utils import append_experiment_row, compute_metrics, save_confusion_matrix, save_metrics_json


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = [LABEL2ID[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_t = batch.pop("labels")
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(labels_t.cpu().tolist())
    y_pred = [LABELS[i] for i in preds]
    y_true = [LABELS[i] for i in labels]
    return compute_metrics(y_true, y_pred), y_true, y_pred


def train(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits()
    train_df = splits["train"]
    val_df = splits["validation"]
    test_df = splits["test"]

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    if getattr(tokenizer, "model_max_length", None) in (None, 1e30, 1000000000000000019889624719755765974):
        tokenizer.model_max_length = 512
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id=LABEL2ID,
    )
    model.to(device)

    train_ds = SentimentDataset(train_df["text"], train_df["label"], tokenizer, args.max_length)
    val_ds = SentimentDataset(val_df["text"], val_df["label"], tokenizer, args.max_length)
    test_ds = SentimentDataset(test_df["text"], test_df["label"], tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    if args.class_weights:
        weights = compute_class_weight(
            "balanced",
            classes=np.array(list(range(len(LABELS)))),
            y=train_df["label"].map(LABEL2ID).values,
        )
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    else:
        class_weights = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    out_dir = MODELS_DIR / "afriberta-best"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=loss.item())

        val_m, _, _ = evaluate(model, val_loader, device)
        print(f"Validation macro-F1: {val_m['macro_f1']:.4f}")
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    val_m, _, _ = evaluate(model, val_loader, device)
    test_m, test_true, test_pred = evaluate(model, test_loader, device)

    save_metrics_json(
        {
            "validation": val_m,
            "test": test_m,
            "model_name": args.model_name,
            "class_weights": args.class_weights,
            "hyperparameters": vars(args),
        },
        "afriberta",
    )
    eid = "E5" if args.class_weights else "E4"
    append_experiment_row(
        {
            "id": eid,
            "model": "AfriBERTa fine-tune",
            "key_change": f"lr={args.lr}, max_len={args.max_length}, class_weights={args.class_weights}",
            "macro_f1_val": val_m["macro_f1"],
            "macro_f1_test": test_m["macro_f1"],
        }
    )
    save_confusion_matrix(
        test_true,
        test_pred,
        RESULTS_DIR / f"confusion_matrix_{eid}_afriberta.png",
        title=f"{eid} AfriBERTa (test)",
    )
    (MODELS_DIR / "production_model.json").write_text(
        json.dumps({"type": "transformer", "path": str(out_dir)}, indent=2),
        encoding="utf-8",
    )
    print(f"Best model saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_TRANSFORMER)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--class-weights", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
