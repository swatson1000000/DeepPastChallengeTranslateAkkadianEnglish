"""
Preprocessing module for Akkadian text normalization and cleaning.

This module handles:
- Removal of scribal notations
- Unicode normalization
- Determinative handling
- Gap/break standardization
- Character encoding fixes
"""

import re
from typing import Optional, Dict, List
from enum import Enum


class DeterminativeHandling(Enum):
    """How to handle Akkadian determinatives."""
    KEEP = "keep"  # Keep determinatives as-is
    NORMALIZE = "normalize"  # Normalize determinatives
    REMOVE = "remove"  # Remove determinatives


class AkkadianPreprocessor:
    """
    Comprehensive preprocessor for Old Assyrian/Akkadian text.
    
    Handles:
    - Scribal notations (!, ?, /, :, .)
    - Unicode character normalization (Ḫ→H, accents, subscripts)
    - Determinatives in curly brackets
    - Gap and break standardization
    - Structural element preservation
    """
    
    # Common Unicode normalization mappings
    UNICODE_NORMALIZATION = {
        'Ḫ': 'H', 'ḫ': 'h',  # H with cedilla
        'š': 'sz', 'Š': 'SZ',  # s with caron
        'ṣ': 's,', 'Ṣ': 'S,',  # s with dot below
        'ṭ': 't,', 'Ṭ': 'T,',  # t with dot below
    }
    
    # Determinative definitions
    DETERMINATIVES = {
        '{d}': 'dingir (god)',
        '{mul}': 'star',
        '{ki}': 'earth/location',
        '{lu₂}': 'person',
        '{e₂}': 'building',
        '{uru}': 'settlement',
        '{kur}': 'land/territory',
        '{mi}': 'feminine',
        '{m}': 'masculine',
        '{geš}': 'wood',
        '{tug₂}': 'textile',
        '{dub}': 'tablet',
        '{id₂}': 'river',
        '{mušen}': 'bird',
        '{na₄}': 'stone',
        '{kuš}': 'skin',
        '{u₂}': 'plant',
    }
    
    def __init__(
        self,
        remove_scribal_marks: bool = True,
        normalize_unicode: bool = True,
        handle_determinatives: DeterminativeHandling = DeterminativeHandling.NORMALIZE,
        normalize_gaps: bool = True,
        normalize_subscripts: bool = True,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            remove_scribal_marks: Remove !, ?, /, :, . marks
            normalize_unicode: Normalize special characters
            handle_determinatives: How to handle determinatives
            normalize_gaps: Standardize gap markers
            normalize_subscripts: Convert subscripts to standard notation
        """
        self.remove_scribal_marks = remove_scribal_marks
        self.normalize_unicode = normalize_unicode
        self.handle_determinatives = handle_determinatives
        self.normalize_gaps = normalize_gaps
        self.normalize_subscripts = normalize_subscripts
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw Akkadian text
            
        Returns:
            Cleaned text
        """
        text = self._normalize_unicode(text)
        text = self._remove_scribal_marks(text)
        text = self._handle_determinatives(text)
        text = self._normalize_gaps(text)
        text = self._normalize_subscripts(text)
        text = self._clean_whitespace(text)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize special Unicode characters."""
        if not self.normalize_unicode:
            return text
        
        for source, target in self.UNICODE_NORMALIZATION.items():
            text = text.replace(source, target)
        
        # Normalize accented characters
        # á→a, é→e, í→i, ú→u, etc.
        accented_chars = {
            'á': 'a', 'à': 'a', 'ä': 'a',
            'é': 'e', 'è': 'e', 'ë': 'e',
            'í': 'i', 'ì': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'ö': 'o',
            'ú': 'u', 'ù': 'u', 'ü': 'u',
        }
        for accented, base in accented_chars.items():
            text = text.replace(accented, base)
        
        return text
    
    def _remove_scribal_marks(self, text: str) -> str:
        """Remove scribal notations: !, ?, /, :, . (word dividers)."""
        if not self.remove_scribal_marks:
            return text
        
        # Remove certain reading mark
        text = text.replace('!', '')
        
        # Remove questionable reading mark
        text = text.replace('?', '')
        
        # Remove line divider
        text = text.replace('/', '')
        
        # Remove word dividers (: or .)
        # But preserve structure - replace with space
        text = re.sub(r'[:.] ', ' ', text)
        text = re.sub(r'[:.]$', '', text)
        
        return text
    
    def _handle_determinatives(self, text: str) -> str:
        """Handle Akkadian determinatives in curly brackets."""
        if self.handle_determinatives == DeterminativeHandling.KEEP:
            return text
        
        elif self.handle_determinatives == DeterminativeHandling.REMOVE:
            # Remove determinatives: a-lim{ki} → a-lim
            text = re.sub(r'\{[^}]+\}', '', text)
        
        elif self.handle_determinatives == DeterminativeHandling.NORMALIZE:
            # Keep but normalize: {ki} remains as is
            # This is default - preserve for reference but clean up
            pass
        
        return text
    
    def _normalize_gaps(self, text: str) -> str:
        """Standardize gap markers."""
        if not self.normalize_gaps:
            return text
        
        # [x] → <gap>
        text = re.sub(r'\[x\]', '<gap>', text)
        
        # […] or [… …] → <big_gap>
        text = re.sub(r'\[\.\.\.\]|\[\.\.\.\s*\.\.\.\]', '<big_gap>', text)
        
        # Remove square brackets from within text (broken signs)
        # [KÙ.BABBAR] → KÙ.BABBAR
        text = re.sub(r'\[([^\]]+)\]', r'\1', text)
        
        return text
    
    def _normalize_subscripts(self, text: str) -> str:
        """Convert subscripts to standard notation."""
        if not self.normalize_subscripts:
            return text
        
        # Convert subscript numbers to superscript notation
        # a₂ → a2, i₃ → i3, etc.
        subscript_map = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        }
        for subscript, num in subscript_map.items():
            text = text.replace(subscript, num)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up extra whitespace."""
        # Remove trailing/leading whitespace
        text = text.strip()
        
        # Normalize multiple spaces to single space
        text = re.sub(r' +', ' ', text)
        
        return text


class EnglishPreprocessor:
    """
    Preprocessor for English translations.
    
    Handles:
    - Scribal notations
    - Formatting cleanup
    - Case normalization (optional)
    """
    
    def __init__(
        self,
        remove_scribal_marks: bool = True,
        lowercase: bool = False,
    ):
        """
        Initialize the English preprocessor.
        
        Args:
            remove_scribal_marks: Remove scribal notations
            lowercase: Convert to lowercase
        """
        self.remove_scribal_marks = remove_scribal_marks
        self.lowercase = lowercase
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess English translation.
        
        Args:
            text: English translation
            
        Returns:
            Cleaned text
        """
        text = self._remove_scribal_marks(text)
        text = self._clean_brackets(text)
        text = self._clean_whitespace(text)
        
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def _remove_scribal_marks(self, text: str) -> str:
        """Remove scribal notations from English text."""
        if not self.remove_scribal_marks:
            return text
        
        # Remove exclamation marks (certain reading)
        text = text.replace('!', '')
        
        # Remove question marks (questionable reading)
        # Keep in English as they might be meaningful
        
        # Remove comments in parentheses: (comment) → [preserved for now]
        # Actually, let's keep parenthetical content for now
        
        return text
    
    def _clean_brackets(self, text: str) -> str:
        """Handle various bracket types."""
        # Remove partially broken sign markers ˹ ˺
        text = text.replace('˹', '')
        text = text.replace('˺', '')
        
        # Keep square brackets but remove extras
        # [text] → text (for broken signs)
        text = re.sub(r'\[([^\]]+)\]', r'\1', text)
        
        # Keep curly brackets for determinatives
        # {text} → {text}
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean whitespace."""
        text = text.strip()
        text = re.sub(r' +', ' ', text)
        return text


def preprocess_parallel_corpus(
    akkadian_file: str,
    english_file: str,
    output_akkadian: str,
    output_english: str,
) -> None:
    """
    Preprocess a parallel corpus.
    
    Args:
        akkadian_file: Path to raw Akkadian text file
        english_file: Path to raw English translation file
        output_akkadian: Path to save processed Akkadian
        output_english: Path to save processed English
    """
    ak_processor = AkkadianPreprocessor()
    en_processor = EnglishPreprocessor()
    
    with open(akkadian_file, 'r', encoding='utf-8') as f:
        akkadian_lines = f.readlines()
    
    with open(english_file, 'r', encoding='utf-8') as f:
        english_lines = f.readlines()
    
    assert len(akkadian_lines) == len(english_lines), \
        "Akkadian and English files must have same number of lines"
    
    processed_akkadian = []
    processed_english = []
    
    for ak_line, en_line in zip(akkadian_lines, english_lines):
        # Skip empty lines
        if not ak_line.strip() or not en_line.strip():
            continue
        
        # Process both texts
        ak_processed = ak_processor.preprocess(ak_line.strip())
        en_processed = en_processor.preprocess(en_line.strip())
        
        # Skip if either is empty after processing
        if ak_processed and en_processed:
            processed_akkadian.append(ak_processed)
            processed_english.append(en_processed)
    
    # Write output files
    with open(output_akkadian, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_akkadian) + '\n')
    
    with open(output_english, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_english) + '\n')
    
    print(f"Processed {len(processed_akkadian)} lines")
    print(f"Akkadian: {output_akkadian}")
    print(f"English: {output_english}")


if __name__ == '__main__':
    # Example usage
    ak_processor = AkkadianPreprocessor()
    
    test_akkadian = "na-pí-da-tum₂{mi} Á-šur-tá-ab-ni-a {mi} la iš-pur-tum"
    print(f"Original: {test_akkadian}")
    print(f"Processed: {ak_processor.preprocess(test_akkadian)}")
