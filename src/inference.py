"""
Inference for ByT5 Akkadian→English translation.

Loads a fine-tuned ByT5 model and generates translations for test data.
Supports beam search with configurable parameters.

Usage:
    python src/inference.py --model models/byt5-akkadian/best --input data/raw/test.csv --output submission.csv
"""

import sys
import logging
import argparse
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from preprocess import clean_transliteration, postprocess_prediction
from names import NameNormalizer
from gpu_utils import setup_gpu, get_autocast_context, compile_model

logger = logging.getLogger(__name__)

PREFIX = "translate Akkadian to English: "


def generate_translations(
    model: T5ForConditionalGeneration,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 4,
    num_beams: int = 8,
    max_new_tokens: int = 512,
    length_penalty: float = 1.3,
    autocast_ctx=None,
) -> list[str]:
    """Generate translations for a list of input texts.

    Args:
        model: Fine-tuned T5 model.
        tokenizer: Corresponding tokenizer.
        texts: List of preprocessed transliterations (with prefix).
        device: Torch device.
        batch_size: Inference batch size.
        num_beams: Number of beams for beam search.
        max_new_tokens: Maximum tokens to generate.
        length_penalty: Length penalty for beam search (>1 = longer, <1 = shorter).
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
            max_length=512,
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


def main():
    parser = argparse.ArgumentParser(description="ByT5 inference for Akkadian→English")
    parser.add_argument("--model", type=str, default="models/byt5-akkadian/best",
                        help="Path to fine-tuned model")
    parser.add_argument("--input", type=str, default="data/raw/test.csv",
                        help="Path to test CSV")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Inference batch size")
    parser.add_argument("--num-beams", type=int, default=8,
                        help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--length-penalty", type=float, default=1.3,
                        help="Beam search length penalty")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use BF16 mixed precision (safe for ByT5)")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Use torch.compile for fused kernels (best on GB10)")
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

    # Load model
    logger.info(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model = model.to(device)
    model = compile_model(model, enable=args.compile)
    logger.info("Model loaded")

    # Load name normalizer
    name_normalizer = NameNormalizer()

    # Load test data
    logger.info(f"Loading test data from {args.input}")
    test_df = pd.read_csv(args.input)
    logger.info(f"Test samples: {len(test_df)}")

    # Preprocess transliterations
    test_df["transliteration_clean"] = test_df["transliteration"].apply(clean_transliteration)

    # Add prefix
    texts = [PREFIX + t for t in test_df["transliteration_clean"].tolist()]

    # Generate
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
    logger.info(f"Generated {len(predictions)} translations in {elapsed:.1f}s")

    # Post-process
    predictions = [postprocess_prediction(p) for p in predictions]

    # Name normalization using transliteration context
    transliterations = test_df["transliteration_clean"].tolist()
    predictions = [
        name_normalizer.normalize_names(t, p)
        for t, p in zip(transliterations, predictions)
    ]
    logger.info("Name normalization applied")

    # Write submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "translation": predictions,
    })
    submission.to_csv(args.output, index=False)
    logger.info(f"Submission saved to {args.output}")

    # Show samples
    for i in range(min(5, len(predictions))):
        logger.info(f"Sample {i}:")
        logger.info(f"  Input:  {test_df.iloc[i]['transliteration'][:100]}...")
        logger.info(f"  Output: {predictions[i][:100]}...")


if __name__ == "__main__":
    main()
