"""Evaluate val loss for each seed's best checkpoint.

Loads each model, runs validation on the same 10% split (deterministic per seed),
and reports val loss for comparison.

Usage:
    python src/eval_valloss.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "raw" / "train.csv"

SEEDS_AND_MODELS = [
    (42,  PROJECT / "models" / "byt5-baseline-seed42"  / "best"),
    (123, PROJECT / "models" / "byt5-baseline-seed123" / "best"),
    (777, PROJECT / "models" / "byt5-baseline-seed777" / "best"),
    (280, PROJECT / "models" / "byt5-baseline-seed280" / "best"),
]


class SimpleDataset(Dataset):
    def __init__(self, transliterations, translations, tokenizer, max_len=512):
        self.transliterations = transliterations
        self.translations = translations
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prefix = "translate Akkadian to English: "

    def __len__(self):
        return len(self.transliterations)

    def __getitem__(self, idx):
        src = self.prefix + self.transliterations[idx]
        tgt = self.translations[idx]
        source = self.tokenizer(src, max_length=self.max_len, padding="max_length",
                                truncation=True, return_tensors="pt")
        target = self.tokenizer(tgt, max_length=self.max_len, padding="max_length",
                                truncation=True, return_tensors="pt")
        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels,
        }


def evaluate_model(model_path, seed, device, df):
    """Evaluate a single model checkpoint on its val split."""
    if not model_path.exists():
        return None

    # Same split logic as train_byt5.py
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    val_size = max(1, int(len(df) * 0.1))
    val_indices = indices[:val_size]
    val_df = df.iloc[val_indices].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = T5ForConditionalGeneration.from_pretrained(str(model_path))
    model = model.to(device)
    model.eval()

    val_dataset = SimpleDataset(
        val_df["transliteration"].astype(str).tolist(),
        val_df["translation"].astype(str).tolist(),
        tokenizer,
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            num_workers=2, pin_memory=True)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            n += 1

    avg_loss = total_loss / max(n, 1)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return avg_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} rows from {DATA}")

    print()
    print(f"{'Seed':>6}  {'Val Loss':>10}  {'Model Path'}")
    print("-" * 70)

    for seed, model_path in SEEDS_AND_MODELS:
        if not model_path.exists():
            print(f"{seed:>6}  {'N/A':>10}  {model_path}  (not found)")
            continue
        val_loss = evaluate_model(model_path, seed, device, df)
        print(f"{seed:>6}  {val_loss:>10.4f}  {model_path}")

    print()


if __name__ == "__main__":
    main()
