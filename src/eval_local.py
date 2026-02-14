"""
Local evaluation: run inference on the training val split and compute competition metrics.

Reproduces the same 90/10 train/val split used by train_byt5.py (seed=42)
so we can measure BLEU/chrF++/GeoMean without waiting for Kaggle.

Usage:
    python src/eval_local.py --model models/byt5-akkadian-aligned/best
    python src/eval_local.py --model models/byt5-akkadian-aligned-v2/best --with-names
"""

import sys
import os
import logging
import argparse
import time
import math
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import sacrebleu
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from preprocess import clean_transliteration, clean_translation, postprocess_prediction
from gpu_utils import setup_gpu, get_autocast_context, compile_model

logger = logging.getLogger(__name__)

PREFIX = "translate Akkadian to English: "


def get_val_split(
    data_path: str = "data/processed/train_aligned.csv",
    seed: int = 42,
    exclude_suspect: bool = True,
    sentence_only: bool = False,
) -> pd.DataFrame:
    """Reproduce the exact val split from train_byt5.py.

    Args:
        data_path: Path to training CSV.
        seed: Random seed (must match train_byt5.py).
        exclude_suspect: Whether to exclude suspect pairs.
        sentence_only: If True, exclude doc-level pairs before splitting.

    Returns:
        DataFrame with val split rows (preprocessed transliteration + raw translation).
    """
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")

    if exclude_suspect and "is_suspect" in df.columns:
        n_before = len(df)
        df = df[~df["is_suspect"]].reset_index(drop=True)
        logger.info(f"Removed {n_before - len(df)} suspect pairs, {len(df)} remaining")

    # Filter to sentence-level only (matches --sentence-only training mode)
    if sentence_only and "source" in df.columns:
        n_before = len(df)
        df = df[df["source"] != "doc"].reset_index(drop=True)
        logger.info(f"Sentence-only mode: removed {n_before - len(df)} doc-level pairs, {len(df)} remaining")

    # Keep raw translation for reference scoring
    df["translation_raw"] = df["translation"].copy()

    # Preprocess (same as train_byt5.py)
    df["transliteration"] = df["transliteration"].apply(clean_transliteration)
    df["translation"] = df["translation"].apply(clean_translation)
    df = df[df["transliteration"].str.len() > 0].reset_index(drop=True)
    df = df[df["translation"].str.len() > 0].reset_index(drop=True)
    logger.info(f"After preprocessing: {len(df)} rows")

    # Same split logic as train_byt5.py
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    val_size = max(1, int(len(df) * 0.1))
    val_indices = indices[:val_size]

    val_df = df.iloc[val_indices].reset_index(drop=True)
    logger.info(f"Val split: {len(val_df)} samples")

    return val_df


def generate_translations(
    model: T5ForConditionalGeneration,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 8,
    num_beams: int = 8,
    max_new_tokens: int = 512,
    max_input_length: int = 512,
    length_penalty: float = 1.3,
    autocast_ctx=None,
) -> list[str]:
    """Generate translations using beam search.

    Args:
        model: Fine-tuned T5 model.
        tokenizer: Corresponding tokenizer.
        texts: List of preprocessed transliterations (with prefix).
        device: Torch device.
        batch_size: Inference batch size.
        num_beams: Number of beams for beam search.
        max_new_tokens: Maximum tokens to generate.
        max_input_length: Maximum input sequence length.
        length_penalty: Length penalty for beam search.
        autocast_ctx: Optional autocast context manager for BF16.

    Returns:
        List of generated translation strings.
    """
    if autocast_ctx is None:
        from gpu_utils import _nullcontext
        autocast_ctx = _nullcontext()

    model.eval()
    all_predictions = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            max_length=max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad(), autocast_ctx:
            outputs = model.generate(
                **inputs,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend(decoded)

    return all_predictions


def compute_metric(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute geometric mean of BLEU and chrF++ (competition metric).

    Args:
        predictions: List of predicted translations.
        references: List of reference translations.

    Returns:
        Dict with 'bleu', 'chrf', and 'score' (geometric mean).
    """
    refs = [references]

    bleu = sacrebleu.corpus_bleu(predictions, refs)
    chrf = sacrebleu.corpus_chrf(predictions, refs, word_order=2)

    bleu_score = bleu.score
    chrf_score = chrf.score

    if bleu_score > 0 and chrf_score > 0:
        geo_mean = math.sqrt(bleu_score * chrf_score)
    else:
        geo_mean = 0.0

    return {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "score": geo_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Local evaluation on val split")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--data", type=str, default="data/processed/train_aligned.csv",
                        help="Path to training CSV")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for val split (must match training)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Inference batch size (larger OK for local GPU)")
    parser.add_argument("--num-beams", type=int, default=8,
                        help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--length-penalty", type=float, default=1.3,
                        help="Beam search length penalty")
    parser.add_argument("--with-names", action="store_true",
                        help="Apply name normalization post-processing")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional: save predictions CSV for inspection")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use BF16 mixed precision (safe for ByT5)")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Use torch.compile for fused kernels (best on GB10)")
    parser.add_argument("--sentence-only", action="store_true", default=False,
                        help="Evaluate only on sentence-level data (exclude doc-level)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # GB10-optimized GPU setup
    gpu_info = setup_gpu(bf16=args.bf16, compile_model_flag=args.compile)
    autocast_ctx = get_autocast_context(bf16=args.bf16)

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Val split ────────────────────────────────────────────────────────
    val_df = get_val_split(args.data, args.seed, sentence_only=args.sentence_only)

    # ── Load model ───────────────────────────────────────────────────────
    logger.info(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model = model.to(device)
    model = compile_model(model, enable=args.compile)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Generate predictions ─────────────────────────────────────────────
    texts = [PREFIX + t for t in val_df["transliteration"].tolist()]

    start = time.time()
    predictions = generate_translations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        autocast_ctx=autocast_ctx,
    )
    elapsed = time.time() - start
    logger.info(f"Generated {len(predictions)} translations in {elapsed:.1f}s "
                f"({elapsed / max(len(texts), 1):.2f}s/sample)")

    # ── Post-process ─────────────────────────────────────────────────────
    predictions = [postprocess_prediction(p) for p in predictions]

    if args.with_names:
        from names import NameNormalizer
        nn = NameNormalizer()
        transliterations = val_df["transliteration"].tolist()
        predictions = [nn.normalize_names(t, p) for t, p in zip(transliterations, predictions)]
        logger.info("Name normalization applied")

    # ── Score ────────────────────────────────────────────────────────────
    references = val_df["translation"].tolist()

    results = compute_metric(predictions, references)
    logger.info("=" * 60)
    logger.info(f"  ALL DATA (n={len(predictions)})")
    logger.info(f"  BLEU:     {results['bleu']:.2f}")
    logger.info(f"  chrF++:   {results['chrf']:.2f}")
    logger.info(f"  GeoMean:  {results['score']:.2f}")
    logger.info("=" * 60)

    # ── Score by source (sentence vs doc) ────────────────────────────────
    if "source" in val_df.columns:
        sources = val_df["source"].values
        for src in sorted(val_df["source"].unique()):
            mask = sources == src
            src_preds = [p for p, m in zip(predictions, mask) if m]
            src_refs = [r for r, m in zip(references, mask) if m]
            if src_preds:
                src_results = compute_metric(src_preds, src_refs)
                logger.info(f"  {src} (n={len(src_preds)}): "
                            f"BLEU={src_results['bleu']:.2f} "
                            f"chrF++={src_results['chrf']:.2f} "
                            f"GeoMean={src_results['score']:.2f}")

        # Sentence-only score (most representative of Kaggle test)
        sent_mask = sources != "doc"
        sent_preds = [p for p, m in zip(predictions, sent_mask) if m]
        sent_refs = [r for r, m in zip(references, sent_mask) if m]
        if sent_preds:
            sent_results = compute_metric(sent_preds, sent_refs)
            logger.info("-" * 60)
            logger.info(f"  SENTENCE-ONLY (n={len(sent_preds)}) — best proxy for Kaggle LB")
            logger.info(f"  BLEU:     {sent_results['bleu']:.2f}")
            logger.info(f"  chrF++:   {sent_results['chrf']:.2f}")
            logger.info(f"  GeoMean:  {sent_results['score']:.2f}")
        logger.info("=" * 60)

    # ── Show samples ─────────────────────────────────────────────────────
    for i in range(min(10, len(predictions))):
        logger.info(f"\n--- Sample {i} ---")
        logger.info(f"  Input:  {val_df.iloc[i]['transliteration'][:120]}...")
        logger.info(f"  Pred:   {predictions[i][:120]}...")
        logger.info(f"  Ref:    {references[i][:120]}...")

    # ── Save predictions for inspection ──────────────────────────────────
    if args.output:
        out_df = pd.DataFrame({
            "transliteration": val_df["transliteration"],
            "prediction": predictions,
            "reference": references,
        })
        if "source" in val_df.columns:
            out_df["source"] = val_df["source"].values
        out_df.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
