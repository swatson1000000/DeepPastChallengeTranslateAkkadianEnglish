"""
Name normalization for Akkadian→English translation post-processing.

Uses the onomasticon (6,335 names) and OA_Lexicon_eBL.csv (13K+ PNs, 334 GNs)
to correct proper noun spellings in model output.

Strategy:
1. Build spelling→canonical name lookup from onomasticon + lexicon
2. Given a transliteration, identify name tokens by matching against the lookup
3. In model output, fuzzy-match name-like tokens against canonical forms
4. Replace with canonical spelling where confident

Usage:
    from names import NameNormalizer
    nn = NameNormalizer()
    corrected = nn.normalize_names(transliteration, model_output)
"""

import re
import logging
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/raw")
ONOMASTICON_PATH = DATA_DIR / "onomasticon" / "onomasticon.csv"
LEXICON_PATH = DATA_DIR / "OA_Lexicon_eBL.csv"

# ── Accent/subscript normalization for matching ─────────────────────────────

ACCENT_MAP = {
    "á": "a", "à": "a", "é": "e", "è": "e",
    "í": "i", "ì": "i", "ú": "u", "ù": "u",
}

SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")

H_MAP = {"Ḫ": "H", "ḫ": "h"}


def _normalize_spelling(text: str) -> str:
    """Normalize a spelling for fuzzy matching: strip accents, subscripts, H, lowercase."""
    t = text
    for src, dst in H_MAP.items():
        t = t.replace(src, dst)
    for src, dst in ACCENT_MAP.items():
        t = t.replace(src, dst)
    t = t.translate(SUBSCRIPT_MAP)
    t = re.sub(r'[{}()\[\]]', '', t)
    return t.lower().strip()


def _normalize_english_name(name: str) -> str:
    """Normalize an English name for matching: strip diacritics, lowercase."""
    t = name
    for src, dst in H_MAP.items():
        t = t.replace(src, dst)
    for src, dst in ACCENT_MAP.items():
        t = t.replace(src, dst)
    # Common Assyriological diacritics in English names
    t = t.replace('ā', 'a').replace('ē', 'e').replace('ī', 'i').replace('ū', 'u')
    t = t.replace('ṣ', 's').replace('ṭ', 't').replace('š', 'sh').replace('ṯ', 'th')
    return t.lower().strip()


class NameNormalizer:
    """Proper noun normalizer using onomasticon + lexicon data.

    Builds lookup tables from:
    - onomasticon.csv: spelling → canonical Name (6,335 entries)
    - OA_Lexicon_eBL.csv: form → norm for PN/GN types (13,758 entries)

    Attributes:
        spelling_to_names: dict mapping Akkadian spellings to canonical English names.
        canonical_names: set of all known canonical name forms.
        name_to_normalized: dict mapping canonical names to their lowercase match form.
    """

    def __init__(
        self,
        onomasticon_path: Path = ONOMASTICON_PATH,
        lexicon_path: Path = LEXICON_PATH,
    ):
        self.spelling_to_names: dict[str, list[str]] = {}
        self.canonical_names: set[str] = set()
        self.name_to_normalized: dict[str, str] = {}
        self._norm_to_canonical: dict[str, str] = {}  # normalized → best canonical

        self._load_onomasticon(onomasticon_path)
        self._load_lexicon(lexicon_path)
        self._build_reverse_index()

        logger.info(
            f"NameNormalizer: {len(self.spelling_to_names)} spelling entries, "
            f"{len(self.canonical_names)} canonical names"
        )

    def _load_onomasticon(self, path: Path) -> None:
        """Load onomasticon.csv and build spelling→name mappings."""
        if not path.exists():
            logger.warning(f"Onomasticon not found at {path}")
            return

        df = pd.read_csv(path)
        loaded = 0

        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            if not name or name == "nan":
                continue

            # Skip duplicates
            if str(row.get("Duplicate", "")).strip().lower() == "true":
                continue

            self.canonical_names.add(name)

            spellings_str = str(row.get("Spellings_semicolon_separated", ""))
            if spellings_str == "nan":
                continue

            for sp in spellings_str.split(";"):
                sp = sp.strip()
                if sp:
                    norm_sp = _normalize_spelling(sp)
                    if norm_sp not in self.spelling_to_names:
                        self.spelling_to_names[norm_sp] = []
                    if name not in self.spelling_to_names[norm_sp]:
                        self.spelling_to_names[norm_sp].append(name)
                    loaded += 1

            # Also index aliases
            aliases_str = str(row.get("Aliases", ""))
            if aliases_str != "nan":
                for alias in aliases_str.split(";"):
                    alias = alias.strip()
                    if alias:
                        self.canonical_names.add(alias)

        logger.info(f"Onomasticon: {loaded} spelling→name entries from {len(self.canonical_names)} names")

    def _load_lexicon(self, path: Path) -> None:
        """Load OA_Lexicon_eBL.csv for PN and GN entries."""
        if not path.exists():
            logger.warning(f"Lexicon not found at {path}")
            return

        df = pd.read_csv(path)
        pn_gn = df[df["type"].isin(["PN", "GN"])]
        loaded = 0

        for _, row in pn_gn.iterrows():
            form = str(row.get("form", "")).strip()
            norm = str(row.get("norm", "")).strip()

            if form == "nan" or norm == "nan" or not form or not norm:
                continue

            self.canonical_names.add(norm)
            norm_form = _normalize_spelling(form)

            if norm_form not in self.spelling_to_names:
                self.spelling_to_names[norm_form] = []
            if norm not in self.spelling_to_names[norm_form]:
                self.spelling_to_names[norm_form].append(norm)
            loaded += 1

        logger.info(f"Lexicon: {loaded} PN/GN form→norm entries added")

    def _build_reverse_index(self) -> None:
        """Build normalized→canonical reverse index for English name matching."""
        for name in self.canonical_names:
            norm = _normalize_english_name(name)
            self.name_to_normalized[name] = norm
            # Keep the version with most diacritics (typically the "best" scholarly form)
            if norm not in self._norm_to_canonical or len(name) >= len(self._norm_to_canonical[norm]):
                self._norm_to_canonical[norm] = name

    def lookup_transliteration(self, translit_word: str) -> list[str]:
        """Look up canonical names for a transliteration word.

        Args:
            translit_word: A single word from an Akkadian transliteration.

        Returns:
            List of canonical name forms, empty if not a known name.
        """
        norm = _normalize_spelling(translit_word)
        return self.spelling_to_names.get(norm, [])

    def find_names_in_transliteration(self, transliteration: str) -> dict[int, list[str]]:
        """Identify name tokens in a transliteration.

        Args:
            transliteration: Full transliteration string.

        Returns:
            Dict mapping word index → list of canonical names.
        """
        words = transliteration.split()
        name_positions = {}

        for i, word in enumerate(words):
            canonical = self.lookup_transliteration(word)
            if canonical:
                name_positions[i] = canonical

        return name_positions

    def _best_canonical_match(self, token: str, candidates: list[str]) -> str | None:
        """Find best canonical name match for a model output token.

        Uses edit-distance/similarity matching.

        Args:
            token: A name-like token from model output.
            candidates: List of canonical names to match against.

        Returns:
            Best matching canonical name, or None if no good match.
        """
        if not candidates:
            return None

        token_norm = _normalize_english_name(token)
        best_match = None
        best_score = 0.0

        for cand in candidates:
            cand_norm = _normalize_english_name(cand)
            score = SequenceMatcher(None, token_norm, cand_norm).ratio()
            if score > best_score:
                best_score = score
                best_match = cand

        # Require high similarity (>0.7) to avoid false corrections
        if best_score >= 0.7:
            return best_match
        return None

    def normalize_names(
        self,
        transliteration: str,
        prediction: str,
        min_similarity: float = 0.6,
    ) -> str:
        """Normalize proper nouns in a model prediction using transliteration context.

        Strategy:
        1. Find name tokens in the transliteration
        2. For each name-like word in the prediction, check if it could correspond
           to a known name from the transliteration
        3. Replace with canonical form if confident match

        Args:
            transliteration: The input Akkadian transliteration.
            prediction: The model's English translation output.
            min_similarity: Minimum similarity threshold for replacement.

        Returns:
            Prediction with normalized proper nouns.
        """
        if not transliteration or not prediction:
            return prediction

        # Step 1: Find all canonical names expected from this transliteration
        expected_names = set()
        for names in self.find_names_in_transliteration(transliteration).values():
            expected_names.update(names)

        if not expected_names:
            return prediction

        # Step 2: Find name-like tokens in prediction (capitalized words)
        # Pattern: capitalized words, possibly with diacritics and hyphens
        name_pattern = re.compile(
            r'\b([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿāēīūṣṭšḫ]*(?:-[A-ZÀ-ÖØ-Þa-zà-öø-ÿāēīūṣṭšḫ]+)*)\b'
        )

        # Common words that look like names but aren't
        skip_words = {
            "The", "This", "That", "These", "Those", "His", "Her", "Its",
            "He", "She", "They", "We", "You", "My", "Your", "Our", "Their",
            "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
            "Nine", "Ten", "Silver", "Gold", "Copper", "Bronze", "Iron",
            "Seal", "Seals", "Tablet", "Month", "Year", "Day", "Mina",
            "Minas", "Shekel", "Shekels", "Talent", "Son", "Daughter",
            "House", "City", "God", "King", "Witnesses", "Witness",
            "From", "Before", "After", "Until", "Into", "With", "About",
            "Concerning", "Regarding", "According", "Says", "Said",
            "Reckoned", "Total", "If", "When", "Then", "Broken", "Gap",
            "Period", "Eponymate", "Eponymy",
        }

        result = prediction
        replacements = []

        for match in name_pattern.finditer(prediction):
            token = match.group(1)
            if token in skip_words:
                continue
            if len(token) < 3:
                continue

            # Check if this token closely matches any expected canonical name
            token_norm = _normalize_english_name(token)

            best_canonical = None
            best_score = 0.0

            for cand in expected_names:
                cand_norm = _normalize_english_name(cand)
                score = SequenceMatcher(None, token_norm, cand_norm).ratio()
                if score > best_score:
                    best_score = score
                    best_canonical = cand

            # Only replace if: good match AND the canonical form is different
            if best_canonical and best_score >= min_similarity and best_canonical != token:
                replacements.append((match.start(), match.end(), token, best_canonical, best_score))

        # Apply replacements in reverse order to preserve positions
        for start, end, old, new, score in sorted(replacements, key=lambda x: x[0], reverse=True):
            result = result[:start] + new + result[end:]
            logger.debug(f"  Name normalized: '{old}' → '{new}' (score={score:.2f})")

        return result

    def normalize_names_global(self, prediction: str) -> str:
        """Normalize names in prediction without transliteration context.

        Falls back to matching against all known canonical names.
        More conservative — requires higher similarity threshold.

        Args:
            prediction: The model's English translation output.

        Returns:
            Prediction with normalized proper nouns.
        """
        if not prediction:
            return prediction

        name_pattern = re.compile(
            r'\b([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿāēīūṣṭšḫ]*(?:-[A-ZÀ-ÖØ-Þa-zà-öø-ÿāēīūṣṭšḫ]+)*)\b'
        )

        skip_words = {
            "The", "This", "That", "These", "Those", "His", "Her", "Its",
            "He", "She", "They", "We", "You", "My", "Your", "Our", "Their",
            "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
            "Nine", "Ten", "Silver", "Gold", "Copper", "Bronze", "Iron",
            "Seal", "Seals", "Tablet", "Month", "Year", "Day", "Mina",
            "Minas", "Shekel", "Shekels", "Talent", "Son", "Daughter",
            "House", "City", "God", "King", "Witnesses", "Witness",
            "From", "Before", "After", "Until", "Into", "With", "About",
            "Concerning", "Regarding", "According", "Says", "Said",
            "Reckoned", "Total", "If", "When", "Then", "Broken", "Gap",
            "Period", "Eponymate", "Eponymy",
        }

        result = prediction
        replacements = []

        for match in name_pattern.finditer(prediction):
            token = match.group(1)
            if token in skip_words or len(token) < 4:
                continue

            token_norm = _normalize_english_name(token)

            # Look up in reverse index
            if token_norm in self._norm_to_canonical:
                canonical = self._norm_to_canonical[token_norm]
                if canonical != token:
                    replacements.append((match.start(), match.end(), token, canonical))

        for start, end, old, new in sorted(replacements, key=lambda x: x[0], reverse=True):
            result = result[:start] + new + result[end:]

        return result


def build_name_lookup(
    onomasticon_path: Path = ONOMASTICON_PATH,
    lexicon_path: Path = LEXICON_PATH,
) -> NameNormalizer:
    """Build and return a NameNormalizer instance.

    Convenience function for use in notebooks/scripts.

    Args:
        onomasticon_path: Path to onomasticon.csv.
        lexicon_path: Path to OA_Lexicon_eBL.csv.

    Returns:
        Configured NameNormalizer.
    """
    return NameNormalizer(onomasticon_path, lexicon_path)
