"""
Preprocessing for Akkadian transliterations and English translations.

Follows the official Dataset Instructions from the Deep Past Challenge:
https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/overview/dataset-instructions

Usage:
    python src/preprocess.py --input data/raw/train.csv --output data/processed/train_clean.csv
"""

import re
import logging
import argparse
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Unicode substitution tables ──────────────────────────────────────────────

# Subscript digits → regular digits
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")

# Accent vowels → numbered forms (official table)
ACCENT_MAP = {
    "á": "a2", "à": "a3",
    "é": "e2", "è": "e3",
    "í": "i2", "ì": "i3",
    "ú": "u2", "ù": "u3",
}

# H normalization — training has Ḫ/ḫ, test has only H/h
H_MAP = {"Ḫ": "H", "ḫ": "h"}


def normalize_h(text: str) -> str:
    """Replace Ḫ→H and ḫ→h. Only one type of H in Akkadian."""
    for src, dst in H_MAP.items():
        text = text.replace(src, dst)
    return text


def normalize_accents(text: str) -> str:
    """Convert accented vowels to numbered forms: á→a2, à→a3, etc."""
    for src, dst in ACCENT_MAP.items():
        text = text.replace(src, dst)
    return text


def normalize_subscripts(text: str) -> str:
    """Convert subscript digits to regular digits: ₀→0, ₁→1, etc."""
    return text.translate(SUBSCRIPT_MAP)


def normalize_gaps(text: str) -> str:
    """Standardize broken text markers to <gap> and <big_gap>.

    Per official instructions:
      [x]    → <gap>    (single broken sign)
      …      → <big_gap> (large break)
      [… …]  → <big_gap> (large break in brackets)
    """
    # Must process [… …] and [x] before stripping brackets
    text = re.sub(r"\[\s*…\s*…?\s*\]", " <big_gap> ", text)
    text = re.sub(r"\[x+\]", " <gap> ", text)
    text = text.replace("…", " <big_gap> ")
    # Catch remaining xx+ patterns (unreadable signs)
    text = re.sub(r"\bx{2,}\b", " <gap> ", text)
    return text


def remove_scribal_notations(text: str) -> str:
    """Remove modern scribal notations per official list.

    Removes: ! ? / : (as word dividers)
    Keeps text inside < > (scribal insertions).
    Removes << >> entirely (erroneous signs).
    Removes half-brackets ˹ ˺ and square brackets [ ] (keep text inside).
    """
    # Double angle brackets → remove entirely (erroneous signs)
    text = re.sub(r"<<.*?>>", "", text)
    # Single angle brackets → keep text inside
    text = re.sub(r"<([^<>]*)>", r"\1", text)
    # Half brackets → remove (keep text)
    text = text.replace("˹", "").replace("˺", "")
    # Square brackets → remove brackets, keep text (but not [x] gaps — already handled)
    text = re.sub(r"\[([^\]]*)\]", r"\1", text)
    # Remove scribal punctuation: ! ? / : .
    # Be careful with : and . — only remove when used as word dividers
    text = re.sub(r"(?<!\d)[!?]", "", text)  # ! and ? (not after digits)
    text = re.sub(r"(?<!\d)/(?!\d)", "", text)  # / (not in fractions)
    # Note: : and . as word dividers — remove isolated ones
    # Keep . in decimal numbers like 0.33333
    text = re.sub(r"(?<!\d):(?!\d)", " ", text)
    return text


def clean_transliteration(text: str) -> str:
    """Full preprocessing pipeline for Akkadian transliterations.

    Args:
        text: Raw transliteration string.

    Returns:
        Cleaned transliteration string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Normalize gaps BEFORE removing brackets
    text = normalize_gaps(text)
    # 2. Remove scribal notations (brackets, !, ?, /)
    text = remove_scribal_notations(text)
    # 3. H normalization (Ḫ→H, ḫ→h)
    text = normalize_h(text)
    # 4. Accent vowels → numbered forms
    text = normalize_accents(text)
    # 5. Subscript digits → regular
    text = normalize_subscripts(text)
    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_translation(text: str) -> str:
    """Full preprocessing pipeline for English translations.

    Args:
        text: Raw translation string.

    Returns:
        Cleaned translation string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Normalize gaps
    text = normalize_gaps(text)
    # 2. Remove scribal notations
    text = remove_scribal_notations(text)
    # 3. H normalization
    text = normalize_h(text)
    # 4. Subscript digits → regular
    text = normalize_subscripts(text)
    # 5. Remove parenthetical scribal annotations: (fem), (plur), etc.
    text = re.sub(r"\((fem|plur|pl|sing|masc|m|f)\)", "", text, flags=re.IGNORECASE)
    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def postprocess_prediction(text: str) -> str:
    """Post-process model output before writing to submission.csv.

    Args:
        text: Raw model prediction string.

    Returns:
        Post-processed prediction.
    """
    if not isinstance(text, str):
        return ""

    # H normalization (model might output Ḫ/ḫ from training data patterns)
    text = normalize_h(text)
    # Subscript → regular
    text = normalize_subscripts(text)
    # Remove scribal annotations that leaked through
    text = remove_scribal_notations(text)
    # Remove repeated words: "the the" → "the"
    text = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame, has_translation: bool = True) -> pd.DataFrame:
    """Preprocess a full DataFrame of transliteration/translation pairs.

    Args:
        df: DataFrame with 'transliteration' column (and optionally 'translation').
        has_translation: Whether the DataFrame has a 'translation' column.

    Returns:
        Preprocessed DataFrame.
    """
    logger.info(f"Preprocessing {len(df)} rows...")

    df = df.copy()
    df["transliteration"] = df["transliteration"].apply(clean_transliteration)

    if has_translation and "translation" in df.columns:
        df["translation"] = df["translation"].apply(clean_translation)
        # Drop rows with empty translations
        before = len(df)
        df = df[df["translation"].str.len() > 0].reset_index(drop=True)
        if before - len(df) > 0:
            logger.info(f"Dropped {before - len(df)} rows with empty translations")

    # Drop rows with empty transliterations
    before = len(df)
    df = df[df["transliteration"].str.len() > 0].reset_index(drop=True)
    if before - len(df) > 0:
        logger.info(f"Dropped {before - len(df)} rows with empty transliterations")

    logger.info(f"Preprocessed: {len(df)} rows remaining")
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess Akkadian data")
    parser.add_argument("--input", type=str, default="data/raw/train.csv")
    parser.add_argument("--output", type=str, default="data/processed/train_clean.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from {args.input}")

    has_translation = "translation" in df.columns
    df = preprocess_dataframe(df, has_translation=has_translation)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Saved to {args.output}")

    # Show a sample
    for i in range(min(3, len(df))):
        logger.info(f"Sample {i}: {df.iloc[i]['transliteration'][:100]}...")
        if has_translation:
            logger.info(f"  → {df.iloc[i]['translation'][:100]}...")


if __name__ == "__main__":
    main()
