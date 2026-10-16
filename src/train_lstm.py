"""BiLSTM baseline with trainable embeddings (course RNN component)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import LABEL2ID, LABELS, MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.data import load_splits
from src.metrics_utils import append_experiment_row, compute_metrics, save_confusion_matrix, save_metrics_json


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class Vocab:
    def __init__(self, texts, min_freq: int = 2, max_size: int = 30_000):
        counts = Counter()
        for t in texts:
            counts.update(t.lower().split())
        items = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        self.itos = ["<pad>", "<unk>"] + items
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text: str, max_len: int):
        ids = [self.stoi.get(w, 1) for w in text.lower().split()[:max_len]]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids


class TweetDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocab, max_len: int):
        self.texts = texts
        self.labels = [LABEL2ID[l] for l in labels]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(self.vocab.encode(self.texts[idx], self.max_len), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h, _) = self.lstm(emb)
        # max pooling over sequence
        mask = (x != 0).unsqueeze(-1).float()
        summed = (out * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
        return self.fc(self.dropout(pooled))


def predict_model(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return [LABELS[i] for i in preds]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits()
    train_texts = splits["train"]["text"].tolist()
    train_labels = splits["train"]["label"].tolist()
    val_texts = splits["validation"]["text"].tolist()
    val_labels = splits["validation"]["label"].tolist()
    test_texts = splits["test"]["text"].tolist()
    test_labels = splits["test"]["label"].tolist()

    vocab = Vocab(train_texts)
    train_ds = TweetDataset(train_texts, train_labels, vocab, args.max_len)
    val_ds = TweetDataset(val_texts, val_labels, vocab, args.max_len)
    test_ds = TweetDataset(test_texts, test_labels, vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # class weights for imbalance
    counts = Counter(train_labels)
    weights = torch.tensor(
        [1.0 / counts[l] for l in LABELS],
        dtype=torch.float,
    )
    weights = weights / weights.sum() * len(LABELS)

    model = BiLSTMClassifier(
        len(vocab.itos),
        args.embed_dim,
        args.hidden_dim,
        len(LABELS),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    best_state = None
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_pred = predict_model(model, val_loader, device)
        val_m = compute_metrics(val_labels, val_pred)
        print(f"Epoch {epoch + 1}/{args.epochs} loss={loss:.4f} val macro-F1={val_m['macro_f1']:.4f}")
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    val_pred = predict_model(model, val_loader, device)
    test_pred = predict_model(model, test_loader, device)
    val_m = compute_metrics(val_labels, val_pred)
    test_m = compute_metrics(test_labels, test_pred)

    ckpt = {
        "model_state": model.state_dict(),
        "vocab_itos": vocab.itos,
        "max_len": args.max_len,
        "config": vars(args),
    }
    path = MODELS_DIR / "bilstm.pt"
    torch.save(ckpt, path)

    save_metrics_json({"validation": val_m, "test": test_m}, "bilstm")
    append_experiment_row(
        {
            "id": "E3",
            "model": "BiLSTM",
            "key_change": f"embed={args.embed_dim}, hidden={args.hidden_dim}, max_len={args.max_len}",
            "macro_f1_val": val_m["macro_f1"],
            "macro_f1_test": test_m["macro_f1"],
        }
    )
    save_confusion_matrix(
        test_labels,
        test_pred,
        RESULTS_DIR / "confusion_matrix_E3_bilstm.png",
        title="E3 BiLSTM (test)",
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
