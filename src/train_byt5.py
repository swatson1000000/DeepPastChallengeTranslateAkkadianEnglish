"""
ByT5 fine-tuning for Akkadian→English translation.

Fine-tunes google/byt5-small on the Deep Past Challenge training data.

Training recipe (per top teams):
    epochs=10-15, max_length=512, batch=4, grad_accum=4, lr=5e-5, NO fp16

Usage:
    python src/train_byt5.py [--epochs 15] [--batch-size 4] [--lr 5e-5]
                             [--max-length 512] [--output-dir models/byt5-akkadian]

Runs with nohup:
    nohup python -u src/train_byt5.py --epochs 15 > log/train_byt5_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

import sys
import os
import re
import logging
import argparse
import math
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

from preprocess import clean_transliteration, clean_translation
from gpu_utils import setup_gpu, get_autocast_context, compile_model, log_gpu_info

logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────

class AkkadianDataset(Dataset):
    """Dataset for Akkadian→English translation pairs."""

    def __init__(
        self,
        transliterations: list[str],
        translations: list[str],
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 512,
        prefix: str = "translate Akkadian to English: ",
    ):
        self.transliterations = transliterations
        self.translations = translations
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.transliterations)

    def __getitem__(self, idx: int) -> dict:
        source = self.prefix + self.transliterations[idx]
        target = self.translations[idx]

        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze()
        # Replace padding token ids with -100 so they're ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


# ── Validation ───────────────────────────────────────────────────────────────

def validate(
    model: T5ForConditionalGeneration,
    val_loader: DataLoader,
    device: torch.device,
    autocast_ctx=None,
) -> float:
    """Compute average validation loss.

    Args:
        model: The T5 model.
        val_loader: Validation DataLoader.
        device: Torch device.
        autocast_ctx: Optional autocast context manager for BF16.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ── Training ─────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Main training function.

    Args:
        args: Command-line arguments.
    """
    # ── Setup (GB10-optimized) ────────────────────────────────────────────
    gpu_info = setup_gpu(bf16=args.bf16, compile_model_flag=args.compile)
    device = gpu_info["device"]
    autocast_ctx = get_autocast_context(bf16=args.bf16)

    # ── Load data ────────────────────────────────────────────────────────
    logger.info(f"Loading training data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} rows")

    # Filter suspect pairs if column exists and flag set
    if "is_suspect" in df.columns and not args.include_suspect:
        n_suspect = df["is_suspect"].sum()
        df = df[~df["is_suspect"]].reset_index(drop=True)
        logger.info(f"Removed {n_suspect} suspect pairs, {len(df)} remaining")

    # Log data source composition if column exists
    if "source" in df.columns:
        logger.info(f"Data composition: {df['source'].value_counts().to_dict()}")

    # Preprocess
    df["transliteration"] = df["transliteration"].apply(clean_transliteration)
    df["translation"] = df["translation"].apply(clean_translation)
    df = df[df["transliteration"].str.len() > 0].reset_index(drop=True)
    df = df[df["translation"].str.len() > 0].reset_index(drop=True)
    logger.info(f"After preprocessing: {len(df)} rows")

    # Train/val split (90/10)
    np.random.seed(args.seed)
    indices = np.random.permutation(len(df))
    val_size = max(1, int(len(df) * 0.1))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # ── Model & tokenizer ────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Enable gradient checkpointing to trade compute for memory
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: ENABLED (use_cache=False)")

    model = model.to(device)

    # Apply torch.compile for fused kernels (critical for GB10 bandwidth)
    model = compile_model(model, enable=args.compile)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {num_params:,} total, {trainable_params:,} trainable")

    # ── Datasets & DataLoaders ───────────────────────────────────────────
    train_dataset = AkkadianDataset(
        transliterations=train_df["transliteration"].tolist(),
        translations=train_df["translation"].tolist(),
        tokenizer=tokenizer,
        max_source_length=args.max_length,
        max_target_length=args.max_length,
    )

    val_dataset = AkkadianDataset(
        transliterations=val_df["transliteration"].tolist(),
        translations=val_df["translation"].tolist(),
        tokenizer=tokenizer,
        max_source_length=args.max_length,
        max_target_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ── Optimizer & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Training config:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {args.grad_accum}")
    logger.info(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    logger.info(f"  FP16: OFF (causes NaN with ByT5)")
    logger.info(f"  BF16: {'ON' if args.bf16 else 'OFF'} (safe — same exponent as FP32)")
    logger.info(f"  torch.compile: {'ON' if args.compile else 'OFF'}")

    # ── Training loop ────────────────────────────────────────────────────
    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = outputs.loss / args.grad_accum
            loss.backward()

            epoch_loss += outputs.loss.item()
            num_batches += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % (args.grad_accum * 10) == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(
                    f"  Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # Handle remaining gradients
        if len(train_loader) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── Validation ───────────────────────────────────────────────────
        val_loss = validate(model, val_loader, device, autocast_ctx=autocast_ctx)
        avg_train_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
            f"time={elapsed:.1f}s"
        )

        # ── Checkpointing ────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / "best"
            logger.info(f"  New best val_loss={val_loss:.4f} — saving to {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        # Save periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_path = output_dir / f"epoch_{epoch}"
            logger.info(f"  Saving checkpoint to {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    # ── Save final model ─────────────────────────────────────────────────
    final_path = output_dir / "final"
    logger.info(f"Saving final model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {output_dir / 'best'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ByT5 for Akkadian→English")
    parser.add_argument("--data", type=str, default="data/processed/train_aligned.csv",
                        help="Path to training CSV (aligned or raw)")
    parser.add_argument("--exclude-suspect", action="store_true", default=True,
                        help="Exclude rows flagged as suspect (if is_suspect column exists)")
    parser.add_argument("--include-suspect", action="store_true", default=False,
                        help="Include all rows even if flagged as suspect")
    parser.add_argument("--model-name", type=str, default="google/byt5-small",
                        help="Pretrained model name or path")
    parser.add_argument("--output-dir", type=str, default="models/byt5-akkadian",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length (source and target, covers 94%% of aligned data)")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing (saves VRAM, slight speed cost)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use BF16 mixed precision (safe for ByT5, unlike FP16)")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Use torch.compile for fused kernels (best on GB10)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)
