"""
Weight-averaged ensemble for ByT5 models.

Merges multiple ByT5 checkpoints by averaging their parameters.
This produces a single model that can run within Kaggle's 9hr limit.

The top public baseline (34.9) uses exactly this technique:
    3 ByT5 checkpoints → weighted parameter averaging → single merged model

Usage:
    python src/ensemble.py \
        --models models/byt5-seed42/best models/byt5-seed123/best models/byt5-seed777/best \
        --weights 0.4 0.3 0.3 \
        --output models/byt5-ensemble/

    # Equal-weight averaging (default):
    python src/ensemble.py \
        --models models/byt5-seed42/best models/byt5-seed123/best \
        --output models/byt5-ensemble/
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from collections import OrderedDict

sys.stdout.reconfigure(line_buffering=True)

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

logger = logging.getLogger(__name__)


def average_checkpoints(
    model_paths: list[str],
    weights: list[float] | None = None,
) -> OrderedDict:
    """Average model state dicts with optional weighting.

    Args:
        model_paths: Paths to model checkpoints.
        weights: Optional weights for each model (must sum to 1.0).
                 If None, uses equal weighting.

    Returns:
        Averaged state dict.
    """
    n = len(model_paths)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        assert len(weights) == n, f"Got {len(weights)} weights for {n} models"
        total = sum(weights)
        weights = [w / total for w in weights]  # Normalize

    logger.info(f"Averaging {n} checkpoints with weights: {[f'{w:.3f}' for w in weights]}")

    avg_state = None

    for i, (path, weight) in enumerate(zip(model_paths, weights)):
        logger.info(f"  Loading model {i+1}/{n}: {path}")
        model = T5ForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32)
        state = model.state_dict()

        if avg_state is None:
            avg_state = OrderedDict()
            for key, param in state.items():
                avg_state[key] = param.clone() * weight
        else:
            for key, param in state.items():
                avg_state[key] += param * weight

        # Free memory
        del model, state
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info("Averaging complete.")
    return avg_state


def merge_and_save(
    model_paths: list[str],
    output_dir: str,
    weights: list[float] | None = None,
    base_model: str | None = None,
) -> None:
    """Merge models and save the averaged checkpoint.

    Args:
        model_paths: Paths to model checkpoints.
        output_dir: Where to save the merged model.
        weights: Optional per-model weights.
        base_model: Model to use for architecture/config (defaults to first model).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Average the parameters
    avg_state = average_checkpoints(model_paths, weights)

    # Load base model architecture
    base = base_model or model_paths[0]
    logger.info(f"Loading base architecture from: {base}")
    model = T5ForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float32)
    model.load_state_dict(avg_state)

    # Save
    logger.info(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)

    # Copy tokenizer from first model
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
    tokenizer.save_pretrained(output_path)

    # Write metadata
    meta_path = output_path / "ensemble_info.txt"
    with open(meta_path, "w") as f:
        f.write("Ensemble Model Metadata\n")
        f.write("=" * 50 + "\n")
        for i, (path, w) in enumerate(zip(model_paths, weights or [1.0/len(model_paths)]*len(model_paths))):
            f.write(f"  Model {i+1}: {path} (weight={w:.3f})\n")
        f.write(f"\nOutput: {output_path}\n")

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Merged model saved: {num_params:,} parameters")
    logger.info(f"Metadata: {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weight-average ByT5 checkpoints")
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="Paths to model checkpoints to average",
    )
    parser.add_argument(
        "--weights", type=float, nargs="*", default=None,
        help="Weights for each model (optional; must match --models count)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for merged model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args()
    merge_and_save(args.models, args.output, args.weights)
