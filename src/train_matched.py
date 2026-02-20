"""
ByT5 fine-tuning — baseline-matched using HuggingFace Seq2SeqTrainer.

Matches the public DPC Starter (Takamichi Toda) baseline EXACTLY:
    - google/byt5-small with HF Seq2SeqTrainer
    - optim="adafactor" (HF sets relative_step=False, scale_parameter=False;
      uses fixed LR=1e-4 with linear decay scheduler)
    - DataCollatorForSeq2Seq for dynamic padding
    - label_smoothing_factor=0.2 (native Trainer)
    - per_device_train_batch_size=1, gradient_accumulation_steps=8
    - predict_with_generate=True with BLEU/chrF++ metrics
    - Bidirectional training (Akk→Eng + Eng→Akk)
    - fp16=False (prevents NaN), bf16=True on GB10

Key differences from our previous train_byt5.py:
    1. Dynamic padding via DataCollatorForSeq2Seq (not fixed max_length=512)
    2. Label smoothing handled natively by Trainer (not manual CrossEntropyLoss)
    3. Bidirectional training doubles training data
    4. HF Seq2SeqTrainer handles generation + best model selection properly

Usage:
    python src/train_matched.py --seed 42 --output-dir models/byt5-matched-seed42
    python src/train_matched.py --data data/processed/train_aligned.csv --doc-weight 0.3

Runs with nohup:
    nohup python -u src/train_matched.py --seed 42 --output-dir models/byt5-matched-seed42 \
        > log/train_matched_seed42_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

import sys
import os
import re
import logging
import argparse
import math
import time
import gc
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset, disable_progress_bars
disable_progress_bars()

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
)

import importlib
# Import the 'evaluate' package (not src/evaluate.py which shadows it)
import sys as _sys
_parent = str(Path(__file__).parent)
_sys_path_backup = _sys.path[:]
_sys.path = [p for p in _sys.path if p != _parent]
evaluate = importlib.import_module("evaluate")
_sys.path = _sys_path_backup

logger = logging.getLogger(__name__)

# ── Prefixes ──────────────────────────────────────────────────────────────
FWD_PREFIX = "translate Akkadian to English: "
BWD_PREFIX = "translate English to Akkadian: "


class EpochLoggingCallback(TrainerCallback):
    """Log a clean one-line summary at each evaluation (= each epoch)."""

    def __init__(self):
        self.epoch_start_time = None
        self.last_train_loss = None
        self.best_val_loss = float("inf")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch = int(state.epoch) if state.epoch else 0
        val_loss = metrics.get("eval_loss", float("nan"))
        bleu = metrics.get("eval_bleu", 0)
        chrf = metrics.get("eval_chrf", 0)
        geo = metrics.get("eval_geo_mean", 0)
        train_loss = self.last_train_loss if self.last_train_loss is not None else float("nan")
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        is_new_best = val_loss < self.best_val_loss
        if is_new_best:
            self.best_val_loss = val_loss

        best_marker = " ★ BEST" if is_new_best else ""
        logger.info(
            f"Epoch {epoch}/{args.num_train_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"BLEU={bleu:.2f} chrF++={chrf:.2f} GeoMean={geo:.2f} "
            f"time={elapsed:.0f}s{best_marker}"
        )


def load_and_prepare_data(args):
    """Load training data, apply filters, create train/val split.

    Args:
        args: Command-line arguments.

    Returns:
        Tuple of (train_dataset, val_dataset) as HuggingFace Datasets.
    """
    logger.info(f"Loading training data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} rows")

    # Filter suspect pairs if column exists
    if "is_suspect" in df.columns:
        n_suspect = df["is_suspect"].sum()
        df = df[~df["is_suspect"]].reset_index(drop=True)
        logger.info(f"Removed {n_suspect} suspect pairs, {len(df)} remaining")

    # Log data source composition
    if "source" in df.columns:
        logger.info(f"Data composition: {df['source'].value_counts().to_dict()}")

    # Downsample doc-level pairs if requested
    if args.doc_weight < 1.0 and "source" in df.columns:
        doc_mask = df["source"] == "doc"
        n_docs = doc_mask.sum()
        n_keep = max(1, int(n_docs * args.doc_weight))
        doc_indices = df[doc_mask].sample(n=n_keep, random_state=args.seed).index
        df = df[~doc_mask | df.index.isin(doc_indices)].reset_index(drop=True)
        logger.info(f"Doc downsampling ({args.doc_weight}): kept {n_keep}/{n_docs} doc pairs, {len(df)} total")

    # Ensure string types, remove empty
    df["transliteration"] = df["transliteration"].astype(str)
    df["translation"] = df["translation"].astype(str)
    df = df[df["transliteration"].str.len() > 0].reset_index(drop=True)
    df = df[df["translation"].str.len() > 0].reset_index(drop=True)
    logger.info(f"After filtering: {len(df)} rows")

    # Train/val split (90/10) — matches baseline's test_size=0.1, seed=42
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=args.seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Build bidirectional training data (exactly as baseline does it)
    # Forward: Akkadian → English
    fwd_sources = [FWD_PREFIX + t for t in train_df["transliteration"].tolist()]
    fwd_targets = train_df["translation"].tolist()

    if args.bidirectional:
        # Backward: English → Akkadian
        bwd_sources = [BWD_PREFIX + t for t in train_df["translation"].tolist()]
        bwd_targets = train_df["transliteration"].tolist()
        train_sources = fwd_sources + bwd_sources
        train_targets = fwd_targets + bwd_targets
        logger.info(f"Bidirectional training: {len(train_sources)} total pairs (2x)")
    else:
        train_sources = fwd_sources
        train_targets = fwd_targets

    # Shuffle combined training data
    combined = list(zip(train_sources, train_targets))
    np.random.seed(args.seed)
    np.random.shuffle(combined)
    train_sources, train_targets = zip(*combined)
    train_sources, train_targets = list(train_sources), list(train_targets)

    # Val is always forward direction only
    val_sources = [FWD_PREFIX + t for t in val_df["transliteration"].tolist()]
    val_targets = val_df["translation"].tolist()

    # Create HF datasets
    train_dataset = HFDataset.from_dict({
        "input_text": train_sources,
        "target_text": train_targets,
    })
    val_dataset = HFDataset.from_dict({
        "input_text": val_sources,
        "target_text": val_targets,
    })

    return train_dataset, val_dataset


def preprocess_function(examples, tokenizer, max_length):
    """Tokenize source and target texts.

    No padding here — DataCollatorForSeq2Seq handles dynamic padding per batch.

    Args:
        examples: Batch with 'input_text' and 'target_text' keys.
        tokenizer: ByT5 tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Tokenized inputs with 'input_ids', 'attention_mask', 'labels'.
    """
    inputs = [str(ex) for ex in examples["input_text"]]
    targets = [str(ex) for ex in examples["target_text"]]

    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    labels = tokenizer(targets, max_length=max_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train(args):
    """Main training function.

    Args:
        args: Command-line arguments.
    """
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ─────────────────────────────────────────────────────────────
    train_dataset, val_dataset = load_and_prepare_data(args)

    # ── Model & tokenizer ────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {num_params:,}")

    # ── Tokenize datasets ────────────────────────────────────────────────
    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val",
    )

    # ── Data collator (dynamic padding — matches baseline) ───────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        # Dynamic padding: pads to longest in batch, not max_length
    )

    # ── Evaluation metrics (matches baseline) ────────────────────────────
    metric_chrf = evaluate.load("chrf")
    metric_bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        # Handle logits (3D) vs token IDs (2D)
        if hasattr(preds, "ndim") and preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        preds = preds.astype(np.int64)
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        chrf = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels)["score"]
        bleu = metric_bleu.compute(
            predictions=decoded_preds,
            references=[[x] for x in decoded_labels],
        )["score"]
        geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0

        return {"chrf": chrf, "bleu": bleu, "geo_mean": geo_mean}

    # ── Training arguments (matches baseline EXACTLY) ─────────────────────
    output_dir = Path(args.output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        # Evaluation & saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Optimizer — "adafactor" in HF Trainer sets:
        #   relative_step=False, scale_parameter=False, lr=learning_rate
        # Use constant LR (no decay) so the model keeps learning throughout all epochs.
        # Linear decay to 0 (HF default) was leaving the last ~15 epochs nearly useless.
        optim="adafactor",
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,  # 5% warmup (~2 epochs at 40 total)
        label_smoothing_factor=args.label_smoothing,
        # Batch sizing
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Precision
        fp16=False,  # NEVER with ByT5
        bf16=args.bf16,
        # Training duration
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        # Generation during eval
        predict_with_generate=True,
        # Logging — only at epoch boundaries, no fractional-epoch spam
        logging_strategy="epoch",
        report_to="none",
        disable_tqdm=True,  # Clean log output
        # Memory optimization
        gradient_checkpointing=args.gradient_checkpointing,
        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
        # Dataloader
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    callbacks = [EpochLoggingCallback()]
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
        logger.info(f"Early stopping: patience={args.patience} epochs")

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ── Train ────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Grad accumulation: {args.grad_accum}")
    logger.info(f"  Effective batch: {args.batch_size * args.grad_accum}")
    logger.info(f"  Optimizer: adafactor (relative_step=False, constant LR + 5% warmup)")
    logger.info(f"  Learning rate (scheduler): {args.lr}")
    logger.info(f"  Label smoothing: {args.label_smoothing}")
    logger.info(f"  Bidirectional: {args.bidirectional}")
    logger.info(f"  BF16: {args.bf16}")
    logger.info(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Train samples: {len(tokenized_train)}")
    logger.info(f"  Val samples: {len(tokenized_val)}")
    logger.info(f"  Output: {args.output_dir}")

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    logger.info(f"Training complete in {elapsed/3600:.1f}h")

    # ── Save best model ──────────────────────────────────────────────────
    best_path = output_dir / "best"
    best_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    logger.info(f"Best model saved to {best_path}")

    # ── Final eval ───────────────────────────────────────────────────────
    eval_results = trainer.evaluate()
    logger.info(f"Final eval: loss={eval_results['eval_loss']:.4f} "
                f"BLEU={eval_results.get('eval_bleu', 0):.2f} "
                f"chrF++={eval_results.get('eval_chrf', 0):.2f} "
                f"GeoMean={eval_results.get('eval_geo_mean', 0):.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ByT5 (baseline-matched)")
    parser.add_argument("--data", type=str, default="data/raw/train.csv",
                        help="Path to training CSV")
    parser.add_argument("--model-name", type=str, default="google/byt5-small",
                        help="Pretrained model name")
    parser.add_argument("--output-dir", type=str, default="models/byt5-matched",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (for HF scheduler; Adafactor auto LR may override)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--label-smoothing", type=float, default=0.2,
                        help="Label smoothing factor")
    parser.add_argument("--doc-weight", type=float, default=1.0,
                        help="Fraction of doc-level pairs to keep (1.0=all)")
    parser.add_argument("--bidirectional", action="store_true", default=True,
                        help="Bidirectional training (Akk→Eng + Eng→Akk)")
    parser.add_argument("--no-bidirectional", action="store_true", default=False,
                        help="Disable bidirectional training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use BF16 mixed precision (safe for ByT5)")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0=disabled)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args()

    if args.no_bidirectional:
        args.bidirectional = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)
