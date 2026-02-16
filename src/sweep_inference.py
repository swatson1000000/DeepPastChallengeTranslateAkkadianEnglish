"""
Inference hyperparameter sweep for ByT5 Akkadian→English translation.

Tests combinations of num_beams, length_penalty, repetition_penalty,
and no_repeat_ngram_size on local val split to find optimal settings.

Uses the 3-model ensemble (or a single best model) and evaluates with
the competition metric (geometric mean of BLEU and chrF++).

Usage:
    nohup python -u src/sweep_inference.py > log/sweep_inference_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import logging
import math
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sacrebleu
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from gpu_utils import setup_gpu, get_autocast_context

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
PREFIX = "translate Akkadian to English: "


def get_val_split_raw(seed: int = 42) -> pd.DataFrame:
    """Get val split matching baseline training (raw text, no preprocessing)."""
    df = pd.read_csv(PROJECT / "data" / "raw" / "train.csv")
    df["transliteration"] = df["transliteration"].astype(str)
    df["translation"] = df["translation"].astype(str)

    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    val_size = max(1, int(len(df) * 0.1))
    val_indices = indices[:val_size]

    val_df = df.iloc[val_indices].reset_index(drop=True)
    return val_df


def generate(model, tokenizer, texts, device, autocast_ctx,
             batch_size=8, num_beams=4, max_new_tokens=512,
             length_penalty=1.0, repetition_penalty=1.0,
             no_repeat_ngram_size=0) -> list[str]:
    """Generate translations with given hyperparameters."""
    model.eval()
    all_preds = []

    gen_kwargs = dict(
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        early_stopping=True,
    )
    if length_penalty != 1.0:
        gen_kwargs["length_penalty"] = length_penalty
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, max_length=512, padding=True,
                           truncation=True, return_tensors="pt").to(device)
        with torch.no_grad(), autocast_ctx:
            outputs = model.generate(**inputs, **gen_kwargs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_preds.extend(decoded)

    return all_preds


def score(predictions, references):
    """Compute BLEU, chrF++, and GeoMean."""
    refs = [references]
    bleu = sacrebleu.corpus_bleu(predictions, refs).score
    chrf = sacrebleu.corpus_chrf(predictions, refs, word_order=2).score
    geo = math.sqrt(bleu * chrf) if bleu > 0 and chrf > 0 else 0.0
    return bleu, chrf, geo


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_gpu(bf16=True, compile_model_flag=False)
    autocast_ctx = get_autocast_context(bf16=True)

    # Use the ensemble model (best current model)
    model_path = PROJECT / "models" / "byt5-ensemble"
    if not model_path.exists():
        # Fallback to best single seed
        model_path = PROJECT / "models" / "byt5-baseline-seed777" / "best"

    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = T5ForConditionalGeneration.from_pretrained(str(model_path))
    model = model.to(device)
    logger.info("Model loaded")

    # Get val split (seed 42 matches first seed's split)
    val_df = get_val_split_raw(seed=42)
    logger.info(f"Val split: {len(val_df)} samples")

    texts = [PREFIX + t for t in val_df["transliteration"].tolist()]
    references = val_df["translation"].tolist()

    # ── Hyperparameter grid ──────────────────────────────────────────────
    beams_list         = [4, 6, 8, 12]
    length_pen_list    = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    rep_pen_list       = [1.0, 1.2, 1.5]
    ngram_list         = [0, 3, 4]

    # Phase 1: Sweep beams × length_penalty (most impactful, fixed rep=1.0, ngram=0)
    logger.info("=" * 70)
    logger.info("PHASE 1: beams × length_penalty sweep")
    logger.info("=" * 70)

    results = []

    for beams, lp in itertools.product(beams_list, length_pen_list):
        t0 = time.time()
        preds = generate(model, tokenizer, texts, device, autocast_ctx,
                         batch_size=8, num_beams=beams, length_penalty=lp)
        elapsed = time.time() - t0
        bleu, chrf, geo = score(preds, references)
        results.append({
            "beams": beams, "length_penalty": lp,
            "rep_penalty": 1.0, "no_repeat_ngram": 0,
            "bleu": bleu, "chrf": chrf, "geomean": geo, "time": elapsed,
        })
        logger.info(f"beams={beams:>2} lp={lp:.1f} | "
                     f"BLEU={bleu:5.2f} chrF++={chrf:5.2f} GeoMean={geo:5.2f} | "
                     f"{elapsed:.1f}s")

    # Find best from phase 1
    best = max(results, key=lambda x: x["geomean"])
    logger.info(f"\nBest Phase 1: beams={best['beams']} lp={best['length_penalty']:.1f} "
                f"→ GeoMean={best['geomean']:.2f}")

    best_beams = best["beams"]
    best_lp = best["length_penalty"]

    # Phase 2: Sweep repetition_penalty × no_repeat_ngram with best beams/lp
    logger.info("\n" + "=" * 70)
    logger.info(f"PHASE 2: rep_penalty × no_repeat_ngram (beams={best_beams}, lp={best_lp})")
    logger.info("=" * 70)

    for rp, ng in itertools.product(rep_pen_list, ngram_list):
        if rp == 1.0 and ng == 0:
            # Already tested in phase 1
            continue
        t0 = time.time()
        preds = generate(model, tokenizer, texts, device, autocast_ctx,
                         batch_size=8, num_beams=best_beams,
                         length_penalty=best_lp, repetition_penalty=rp,
                         no_repeat_ngram_size=ng)
        elapsed = time.time() - t0
        bleu, chrf, geo = score(preds, references)
        results.append({
            "beams": best_beams, "length_penalty": best_lp,
            "rep_penalty": rp, "no_repeat_ngram": ng,
            "bleu": bleu, "chrf": chrf, "geomean": geo, "time": elapsed,
        })
        logger.info(f"rp={rp:.1f} ngram={ng} | "
                     f"BLEU={bleu:5.2f} chrF++={chrf:5.2f} GeoMean={geo:5.2f} | "
                     f"{elapsed:.1f}s")

    # ── Final summary ────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("geomean", ascending=False)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS — sorted by GeoMean")
    logger.info("=" * 70)
    for _, row in results_df.head(15).iterrows():
        logger.info(
            f"beams={int(row['beams']):>2} lp={row['length_penalty']:.1f} "
            f"rp={row['rep_penalty']:.1f} ng={int(row['no_repeat_ngram'])} | "
            f"BLEU={row['bleu']:5.2f} chrF++={row['chrf']:5.2f} "
            f"GeoMean={row['geomean']:5.2f} | {row['time']:.0f}s"
        )

    overall_best = results_df.iloc[0]
    logger.info("\n" + "=" * 70)
    logger.info(f"★ BEST: beams={int(overall_best['beams'])} "
                f"lp={overall_best['length_penalty']:.1f} "
                f"rp={overall_best['rep_penalty']:.1f} "
                f"ng={int(overall_best['no_repeat_ngram'])} "
                f"→ GeoMean={overall_best['geomean']:.2f} "
                f"(BLEU={overall_best['bleu']:.2f}, chrF++={overall_best['chrf']:.2f})")
    logger.info("=" * 70)

    # Save results
    out_path = PROJECT / "data" / "processed" / "inference_sweep_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
