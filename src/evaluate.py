"""
Evaluation metric for the Deep Past Challenge.

Computes the geometric mean of BLEU and chrF++ scores using sacrebleu,
matching the official competition metric:
https://www.kaggle.com/code/metric/dpi-bleu-chrf

Usage:
    python src/evaluate.py --predictions predictions.csv --references data/raw/train.csv
"""

import math
import logging
import argparse

import pandas as pd
import sacrebleu

logger = logging.getLogger(__name__)


def compute_metric(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute geometric mean of BLEU and chrF++ (micro-averaged).

    Args:
        predictions: List of predicted translations.
        references: List of reference translations.

    Returns:
        Dict with 'bleu', 'chrf', and 'score' (geometric mean).
    """
    assert len(predictions) == len(references), (
        f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
    )

    # sacrebleu expects references as list of lists (one list per reference set)
    refs = [references]

    bleu = sacrebleu.corpus_bleu(predictions, refs)
    chrf = sacrebleu.corpus_chrf(predictions, refs, word_order=2)  # chrF++ = word_order=2

    bleu_score = bleu.score
    chrf_score = chrf.score

    # Geometric mean
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
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--predictions", type=str, required=True, help="CSV with 'translation' column")
    parser.add_argument("--references", type=str, required=True, help="CSV with 'translation' column")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    preds_df = pd.read_csv(args.predictions)
    refs_df = pd.read_csv(args.references)

    predictions = preds_df["translation"].fillna("").tolist()
    references = refs_df["translation"].fillna("").tolist()

    results = compute_metric(predictions, references)

    logger.info(f"BLEU:     {results['bleu']:.2f}")
    logger.info(f"chrF++:   {results['chrf']:.2f}")
    logger.info(f"GeoMean:  {results['score']:.2f}")


if __name__ == "__main__":
    main()
