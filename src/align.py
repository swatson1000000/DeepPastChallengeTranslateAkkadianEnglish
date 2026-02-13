"""
Sentence alignment for Akkadian translation training data.

Uses Sentences_Oare_FirstWord_LinNum.csv to split document-level training pairs
into sentence-level aligned pairs. Also augments with sentence-level data from
published_texts.csv that aren't in train.csv.

Critical context: training data is document-level, but test data is sentence-level.
Aligning training data to sentence level is the single biggest data quality fix.

Usage:
    python src/align.py [--output data/processed/train_aligned.csv] [--report]

Output columns:
    oare_id, transliteration, translation, source, sent_idx
    source = "doc" (unsplit original), "sent_train" (split from train), "sent_pub" (from published_texts)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import re
import logging
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/raw")
TRAIN_PATH = DATA_DIR / "train.csv"
SENTENCES_PATH = DATA_DIR / "Sentences_Oare_FirstWord_LinNum.csv"
PUBLISHED_PATH = DATA_DIR / "published_texts.csv"
DEFAULT_OUTPUT = Path("data/processed/train_aligned.csv")
REPORT_PATH = Path("data/processed/alignment_report.txt")


# ── Fuzzy word matching ──────────────────────────────────────────────────────

def normalize_for_match(word: str) -> str:
    """Normalize a word for fuzzy matching: lower, strip determinatives, strip digits.

    Args:
        word: Akkadian transliteration word.

    Returns:
        Normalized form for comparison.
    """
    w = word.lower()
    # Remove determinative brackets
    w = re.sub(r'\{[^}]*\}', '', w)
    # Remove parentheses around signs
    w = re.sub(r'[()]', '', w)
    return w.strip()


def find_word_position(words: list[str], target: str, start: int = 0) -> int:
    """Find the position of target word in words list, starting from start.

    Tries exact match first, then case-insensitive, then normalized fuzzy match.

    Args:
        words: List of transliteration words.
        target: The first_word_spelling to search for.
        start: Index to start searching from.

    Returns:
        Word index if found, -1 otherwise.
    """
    if pd.isna(target) or not str(target).strip():
        return -1

    target = str(target).strip()

    # Pass 1: exact match
    for j in range(start, len(words)):
        if words[j] == target:
            return j

    # Pass 2: case-insensitive match
    target_lower = target.lower()
    for j in range(start, len(words)):
        if words[j].lower() == target_lower:
            return j

    # Pass 3: normalized fuzzy match
    target_norm = normalize_for_match(target)
    for j in range(start, len(words)):
        if normalize_for_match(words[j]) == target_norm:
            return j

    # Pass 4: target is a substring of a word (e.g. compound signs)
    for j in range(start, len(words)):
        if target_lower in words[j].lower() or words[j].lower() in target_lower:
            if len(target) >= 3 and len(words[j]) >= 3:  # avoid trivial matches
                return j

    return -1


def split_transliteration(
    transliteration: str,
    sentence_boundaries: pd.DataFrame,
) -> list[dict]:
    """Split a document transliteration into sentence segments using boundary data.

    Args:
        transliteration: Full document transliteration (space-separated words).
        sentence_boundaries: DataFrame of sentences for this text, sorted by
            sentence_obj_in_text. Must have columns: first_word_spelling, translation.

    Returns:
        List of dicts with keys: transliteration, translation, sent_idx, matched.
        If splitting fails, returns empty list.
    """
    if pd.isna(transliteration) or not str(transliteration).strip():
        return []

    words = str(transliteration).strip().split()
    if not words:
        return []

    boundaries = sentence_boundaries.sort_values("sentence_obj_in_text").reset_index(drop=True)

    # Find word positions for each sentence boundary
    positions = []
    prev_pos = 0
    all_found = True

    for _, row in boundaries.iterrows():
        fw = row.get("first_word_spelling", "")
        pos = find_word_position(words, fw, prev_pos)
        if pos >= 0:
            positions.append(pos)
            prev_pos = pos + 1
        else:
            all_found = False
            positions.append(-1)

    # If less than half of boundaries found, give up
    found_count = sum(1 for p in positions if p >= 0)
    if found_count < len(positions) * 0.5:
        return []

    # Fill in missing positions by interpolation
    # For missing boundaries between two found ones, estimate position
    for i in range(len(positions)):
        if positions[i] == -1:
            # Find nearest found positions before and after
            before = next((positions[j] for j in range(i - 1, -1, -1) if positions[j] >= 0), 0)
            after = next((positions[j] for j in range(i + 1, len(positions)) if positions[j] >= 0), len(words))
            # Count how many gaps between before_idx and after_idx
            gaps = sum(1 for j in range(i, len(positions)) if positions[j] == -1)
            # Simple linear interpolation
            step = (after - before) // (gaps + 1)
            positions[i] = before + step

    # Build segments
    segments = []
    for i in range(len(positions)):
        start = positions[i]
        end = positions[i + 1] if i + 1 < len(positions) else len(words)

        if start >= end:
            continue

        seg_translit = " ".join(words[start:end])
        seg_translation = boundaries.iloc[i].get("translation", "")

        if pd.isna(seg_translation) or not str(seg_translation).strip():
            continue

        segments.append({
            "transliteration": seg_translit,
            "translation": str(seg_translation).strip(),
            "sent_idx": i,
            "matched": positions[i] != -1,
        })

    return segments


def flag_bad_pairs(df: pd.DataFrame) -> pd.Series:
    """Flag suspicious training pairs: long transliteration → short translation.

    Conservative: only flags truly problematic pairs. Scholarly conventions like
    "..." for broken tablet sections are NOT flagged — they're valid training data.

    Args:
        df: DataFrame with transliteration and translation columns.

    Returns:
        Boolean Series, True for suspicious pairs.
    """
    trans_len = df["transliteration"].str.len().fillna(0)
    transl_len = df["translation"].str.len().fillna(0)

    # Flag if transliteration is 8x+ longer than translation (by char count)
    ratio_flag = (trans_len > 200) & (trans_len > transl_len * 8)

    # Flag if translation is very short (< 15 chars) but transliteration is long (> 300 chars)
    short_flag = (transl_len < 15) & (trans_len > 300)

    # Flag if translation is MOSTLY dots/ellipsis (>50% non-word chars)
    # Don't flag normal scholarly "..." — only flag truly empty translations
    def is_mostly_empty(text):
        if pd.isna(text):
            return True
        text = str(text).strip()
        if not text:
            return True
        # Remove dots, ellipses, brackets, spaces, dashes
        content = re.sub(r'[.…\[\]\s\-–—(){}]', '', text)
        return len(content) < 5 and len(text) > 10

    mostly_empty_flag = df["translation"].apply(is_mostly_empty)

    return ratio_flag | short_flag | mostly_empty_flag


def build_aligned_dataset(
    train_path: Path = TRAIN_PATH,
    sentences_path: Path = SENTENCES_PATH,
    published_path: Path = PUBLISHED_PATH,
    include_published: bool = True,
    flag_bad: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build sentence-aligned training dataset.

    Strategy:
    1. For train texts WITH sentence boundaries → split into sentence pairs
    2. For train texts WITHOUT boundaries → keep as document-level
    3. Optionally: for published_texts NOT in train but WITH sentence boundaries
       → create new sentence-level pairs (data augmentation)

    Args:
        train_path: Path to train.csv.
        sentences_path: Path to Sentences_Oare_FirstWord_LinNum.csv.
        published_path: Path to published_texts.csv.
        include_published: Whether to include sentence pairs from published_texts.
        flag_bad: Whether to flag and remove suspicious pairs.

    Returns:
        Tuple of (aligned DataFrame, statistics dict).
    """
    logger.info("Loading data files...")
    train = pd.read_csv(train_path)
    sentences = pd.read_csv(sentences_path)
    published = pd.read_csv(published_path)

    train_ids = set(train["oare_id"])
    stats = {
        "train_total": len(train),
        "sentences_total": len(sentences),
        "train_with_boundaries": 0,
        "train_without_boundaries": 0,
        "sent_pairs_from_train": 0,
        "sent_pairs_from_published": 0,
        "doc_pairs_kept": 0,
        "bad_pairs_flagged": 0,
        "split_failures": 0,
    }

    all_rows = []

    # ── Step 1: Split train texts that have sentence boundaries ──────────
    texts_with_boundaries = set(sentences["text_uuid"]) & train_ids
    texts_without_boundaries = train_ids - texts_with_boundaries
    stats["train_with_boundaries"] = len(texts_with_boundaries)
    stats["train_without_boundaries"] = len(texts_without_boundaries)

    logger.info(f"Train texts with sentence boundaries: {len(texts_with_boundaries)}")
    logger.info(f"Train texts without boundaries: {len(texts_without_boundaries)}")

    for text_id in texts_with_boundaries:
        train_row = train[train["oare_id"] == text_id].iloc[0]
        text_sents = sentences[sentences["text_uuid"] == text_id]

        segments = split_transliteration(
            train_row["transliteration"],
            text_sents,
        )

        if segments:
            for seg in segments:
                all_rows.append({
                    "oare_id": text_id,
                    "transliteration": seg["transliteration"],
                    "translation": seg["translation"],
                    "source": "sent_train",
                    "sent_idx": seg["sent_idx"],
                })
            stats["sent_pairs_from_train"] += len(segments)
        else:
            # Splitting failed — keep as document-level
            stats["split_failures"] += 1
            all_rows.append({
                "oare_id": text_id,
                "transliteration": train_row["transliteration"],
                "translation": train_row["translation"],
                "source": "doc",
                "sent_idx": -1,
            })
            stats["doc_pairs_kept"] += 1

    # ── Step 2: Keep unsplit train texts as document-level ────────────────
    for text_id in texts_without_boundaries:
        train_row = train[train["oare_id"] == text_id].iloc[0]
        if pd.isna(train_row["transliteration"]) or pd.isna(train_row["translation"]):
            continue
        all_rows.append({
            "oare_id": text_id,
            "transliteration": train_row["transliteration"],
            "translation": train_row["translation"],
            "source": "doc",
            "sent_idx": -1,
        })
        stats["doc_pairs_kept"] += 1

    # ── Step 3: Augment with published_texts sentence pairs ──────────────
    if include_published:
        pub_ids_with_sents = set(sentences["text_uuid"]) - train_ids
        pub_with_translit = published[
            published["oare_id"].isin(pub_ids_with_sents) &
            published["transliteration"].notna()
        ]

        logger.info(f"Published texts with sentence boundaries (not in train): {len(pub_with_translit)}")

        for _, pub_row in pub_with_translit.iterrows():
            text_id = pub_row["oare_id"]
            text_sents = sentences[sentences["text_uuid"] == text_id]

            segments = split_transliteration(
                pub_row["transliteration"],
                text_sents,
            )

            if segments:
                for seg in segments:
                    all_rows.append({
                        "oare_id": text_id,
                        "transliteration": seg["transliteration"],
                        "translation": seg["translation"],
                        "source": "sent_pub",
                        "sent_idx": seg["sent_idx"],
                    })
                stats["sent_pairs_from_published"] += len(segments)

    # ── Build final DataFrame ────────────────────────────────────────────
    result = pd.DataFrame(all_rows)

    # Remove rows with empty transliteration or translation
    before = len(result)
    result = result[
        result["transliteration"].notna() & (result["transliteration"].str.strip() != "") &
        result["translation"].notna() & (result["translation"].str.strip() != "")
    ].reset_index(drop=True)
    logger.info(f"Removed {before - len(result)} empty pairs")

    # ── Flag bad pairs ───────────────────────────────────────────────────
    if flag_bad and len(result) > 0:
        bad_mask = flag_bad_pairs(result)
        stats["bad_pairs_flagged"] = int(bad_mask.sum())
        logger.info(f"Flagged {bad_mask.sum()} suspicious pairs")
        # Add flag column but don't remove — let training script decide
        result["is_suspect"] = bad_mask

    # ── Deduplicate ──────────────────────────────────────────────────────
    before = len(result)
    result = result.drop_duplicates(
        subset=["transliteration", "translation"], keep="first"
    ).reset_index(drop=True)
    logger.info(f"Removed {before - len(result)} duplicate pairs")

    stats["final_total"] = len(result)
    stats["final_doc"] = int((result["source"] == "doc").sum())
    stats["final_sent_train"] = int((result["source"] == "sent_train").sum())
    stats["final_sent_pub"] = int((result["source"] == "sent_pub").sum())

    return result, stats


def write_report(stats: dict, output_path: Path = REPORT_PATH) -> None:
    """Write alignment statistics report.

    Args:
        stats: Statistics dict from build_aligned_dataset.
        output_path: Path for the report file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 60,
        "SENTENCE ALIGNMENT REPORT",
        "=" * 60,
        "",
        f"Train documents:              {stats['train_total']:>6}",
        f"Sentence boundaries available: {stats['sentences_total']:>6}",
        "",
        "── Splitting Results ──",
        f"Train texts with boundaries:   {stats['train_with_boundaries']:>6}",
        f"Train texts without:           {stats['train_without_boundaries']:>6}",
        f"Split failures:                {stats['split_failures']:>6}",
        "",
        "── Output Composition ──",
        f"Sentence pairs from train:     {stats['sent_pairs_from_train']:>6}",
        f"Sentence pairs from published: {stats['sent_pairs_from_published']:>6}",
        f"Document-level pairs kept:     {stats['doc_pairs_kept']:>6}",
        "",
        f"Bad pairs flagged:             {stats['bad_pairs_flagged']:>6}",
        "",
        f"TOTAL training pairs:          {stats['final_total']:>6}",
        f"  - doc-level:                 {stats['final_doc']:>6}",
        f"  - sent from train:           {stats['final_sent_train']:>6}",
        f"  - sent from published:       {stats['final_sent_pub']:>6}",
        "",
        f"Data expansion: {stats['train_total']} → {stats['final_total']} "
        f"({stats['final_total'] / stats['train_total']:.1f}x)",
        "=" * 60,
    ]
    report = "\n".join(lines)
    output_path.write_text(report)
    logger.info(f"Report written to {output_path}")
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description="Build sentence-aligned training data for Akkadian translation"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output CSV path (default: data/processed/train_aligned.csv)"
    )
    parser.add_argument(
        "--no-published", action="store_true",
        help="Skip augmentation from published_texts.csv"
    )
    parser.add_argument(
        "--no-flag-bad", action="store_true",
        help="Skip flagging suspicious pairs"
    )
    parser.add_argument(
        "--report", action="store_true", default=True,
        help="Write alignment report (default: True)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result, stats = build_aligned_dataset(
        include_published=not args.no_published,
        flag_bad=not args.no_flag_bad,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    logger.info(f"Aligned data written to {args.output} ({len(result)} rows)")

    if args.report:
        write_report(stats)


if __name__ == "__main__":
    main()
