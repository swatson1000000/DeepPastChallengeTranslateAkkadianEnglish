"""
Vocabulary and lexicon building module for Akkadian-English translation.

Handles:
- Extraction of proper nouns from training data
- Building BPE tokenizer for Akkadian text
- Creating character-level vocabulary
- Saving lexicons to data/lexicons/ directory
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.normalizers import Lowercase, Sequence
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VocabularyBuilder:
    """
    Build and manage vocabularies for Akkadian-English translation.
    
    Features:
    - Extract proper nouns (capitalized words, Sumerian logograms)
    - Build character-level vocabulary
    - Create BPE tokenizer
    - Save and load lexicons
    """
    
    def __init__(self, output_dir: str = "data/lexicons/"):
        """
        Initialize vocabulary builder.
        
        Args:
            output_dir: Directory to save vocabulary files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.akkadian_vocab: Set[str] = set()
        self.english_vocab: Set[str] = set()
        self.proper_nouns: Set[str] = set()
        self.sumerian_logograms: Set[str] = set()
        
        self.char_vocab_akkadian: Dict[str, int] = {}
        self.char_vocab_english: Dict[str, int] = {}
        
        logger.info(f"Initialized VocabularyBuilder with output_dir: {self.output_dir}")
    
    def extract_proper_nouns(self, texts: List[str]) -> Dict[str, int]:
        """
        Extract proper nouns from Akkadian texts.
        
        Proper nouns are identified by:
        - Capital first letter followed by lowercase (e.g., Kanesh)
        - ALL CAPS (Sumerian logograms, e.g., KÙ.BABBAR)
        
        Args:
            texts: List of Akkadian transliterated texts
            
        Returns:
            Dictionary mapping proper nouns to frequency
        """
        proper_noun_freq = Counter()
        
        for text in texts:
            # Extract words with capital first letter (proper nouns)
            # Pattern: Capital letter followed by lowercase letters/numbers/hyphens
            proper_matches = re.findall(r'\b[A-Z][a-zñ0-9-]*\b', text)
            for match in proper_matches:
                # Filter out likely false positives (single character)
                if len(match) > 1:
                    proper_noun_freq[match] += 1
                    self.proper_nouns.add(match)
            
            # Extract ALL CAPS (Sumerian logograms)
            # Pattern: ALL CAPS with dots, subscripts, etc.
            caps_matches = re.findall(r'\b[A-Z][A-Z0-9₀-₉.]*(?:\{[^}]*\})?\b', text)
            for match in caps_matches:
                if len(match) > 1 and match.isupper():
                    proper_noun_freq[match] += 1
                    self.sumerian_logograms.add(match)
        
        return dict(proper_noun_freq)
    
    def build_char_vocabulary(self, texts: List[str], language: str = 'akkadian') -> Dict[str, int]:
        """
        Build character-level vocabulary from texts.
        
        Args:
            texts: List of texts
            language: 'akkadian' or 'english'
            
        Returns:
            Dictionary mapping characters to indices
        """
        char_freq = Counter()
        
        for text in texts:
            for char in text:
                char_freq[char] += 1
        
        # Create vocabulary with special tokens first
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3,
        }
        
        # Add characters sorted by frequency
        idx = 4
        for char, _ in char_freq.most_common():
            vocab[char] = idx
            idx += 1
        
        if language == 'akkadian':
            self.char_vocab_akkadian = vocab
            logger.info(f"Built Akkadian character vocabulary: {len(vocab)} tokens")
        else:
            self.char_vocab_english = vocab
            logger.info(f"Built English character vocabulary: {len(vocab)} tokens")
        
        return vocab
    
    def build_word_vocabulary(self, texts: List[str], language: str = 'akkadian', min_freq: int = 2) -> Dict[str, int]:
        """
        Build word-level vocabulary from texts.
        
        Args:
            texts: List of texts
            language: 'akkadian' or 'english'
            min_freq: Minimum frequency threshold
            
        Returns:
            Dictionary mapping words to indices
        """
        word_freq = Counter()
        
        for text in texts:
            words = text.split()
            for word in words:
                if word:  # Skip empty strings
                    word_freq[word] += 1
        
        # Filter by minimum frequency
        filtered_freq = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
        
        # Create vocabulary with special tokens
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3,
        }
        
        # Add words sorted by frequency
        idx = 4
        for word, _ in sorted(filtered_freq.items(), key=lambda x: x[1], reverse=True):
            vocab[word] = idx
            idx += 1
        
        logger.info(f"Built {language} word vocabulary: {len(vocab)} tokens (min_freq={min_freq})")
        
        if language == 'akkadian':
            self.akkadian_vocab = set(vocab.keys())
        else:
            self.english_vocab = set(vocab.keys())
        
        return vocab
    
    def build_bpe_tokenizer(self, texts: List[str], vocab_size: int = 5000) -> Optional[Tokenizer]:
        """
        Build BPE tokenizer for Akkadian text.
        
        Args:
            texts: List of texts for tokenizer training
            vocab_size: Target vocabulary size
            
        Returns:
            Tokenizer object, or None if tokenizers library not available
        """
        if not TOKENIZERS_AVAILABLE:
            logger.warning("tokenizers library not available. Skipping BPE tokenizer.")
            return None
        
        try:
            # Initialize tokenizer with BPE model
            tokenizer = Tokenizer(BPE(unk_token='<unk>'))
            
            # Set up normalizer and pre-tokenizer
            tokenizer.normalizer = Sequence([
                Lowercase()
            ])
            tokenizer.pre_tokenizer = Whitespace()
            
            # Train the tokenizer
            # For simplicity, we'll use word-level splitting
            logger.info(f"Building BPE tokenizer with vocab_size={vocab_size}...")
            
            # Save texts to temporary file for training
            temp_file = self.output_dir / "temp_train.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + '\n')
            
            # Train tokenizer on corpus
            from tokenizers.trainers import BpeTrainer
            trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=['<pad>', '<unk>', '<sos>', '<eos>'])
            tokenizer.train(files=[str(temp_file)], trainer=trainer)
            
            # Clean up temp file
            temp_file.unlink()
            
            logger.info(f"Built BPE tokenizer with {vocab_size} tokens")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Error building BPE tokenizer: {str(e)}")
            return None
    
    def save_lexicons(self):
        """Save all lexicons to data/lexicons/ directory."""
        try:
            # Save proper nouns
            proper_nouns_file = self.output_dir / "proper_nouns.json"
            with open(proper_nouns_file, 'w', encoding='utf-8') as f:
                json.dump(sorted(list(self.proper_nouns)), f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.proper_nouns)} proper nouns to {proper_nouns_file}")
            
            # Save Sumerian logograms
            logograms_file = self.output_dir / "sumerian_logograms.json"
            with open(logograms_file, 'w', encoding='utf-8') as f:
                json.dump(sorted(list(self.sumerian_logograms)), f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.sumerian_logograms)} Sumerian logograms to {logograms_file}")
            
            # Save character vocabularies
            if self.char_vocab_akkadian:
                char_vocab_file = self.output_dir / "char_vocab_akkadian.json"
                with open(char_vocab_file, 'w', encoding='utf-8') as f:
                    json.dump(self.char_vocab_akkadian, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved Akkadian character vocabulary to {char_vocab_file}")
            
            if self.char_vocab_english:
                char_vocab_file = self.output_dir / "char_vocab_english.json"
                with open(char_vocab_file, 'w', encoding='utf-8') as f:
                    json.dump(self.char_vocab_english, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved English character vocabulary to {char_vocab_file}")
            
        except Exception as e:
            logger.error(f"Error saving lexicons: {str(e)}")
            raise
    
    def load_lexicons(self):
        """Load lexicons from data/lexicons/ directory."""
        try:
            proper_nouns_file = self.output_dir / "proper_nouns.json"
            if proper_nouns_file.exists():
                with open(proper_nouns_file, 'r', encoding='utf-8') as f:
                    self.proper_nouns = set(json.load(f))
            
            logograms_file = self.output_dir / "sumerian_logograms.json"
            if logograms_file.exists():
                with open(logograms_file, 'r', encoding='utf-8') as f:
                    self.sumerian_logograms = set(json.load(f))
            
            char_vocab_file = self.output_dir / "char_vocab_akkadian.json"
            if char_vocab_file.exists():
                with open(char_vocab_file, 'r', encoding='utf-8') as f:
                    self.char_vocab_akkadian = json.load(f)
            
            char_vocab_file = self.output_dir / "char_vocab_english.json"
            if char_vocab_file.exists():
                with open(char_vocab_file, 'r', encoding='utf-8') as f:
                    self.char_vocab_english = json.load(f)
            
            logger.info("Loaded lexicons from disk")
            
        except Exception as e:
            logger.error(f"Error loading lexicons: {str(e)}")


def main():
    """Main function to build vocabularies."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from data_loader import DataLoader
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load training data
        logger.info("Loading training data...")
        loader = DataLoader(data_dir="data/raw/")
        train_sources = loader.get_train_sources()
        train_targets = loader.get_train_targets()
        
        logger.info(f"Loaded {len(train_sources)} training samples")
        
        # Build vocabulary
        builder = VocabularyBuilder(output_dir="data/lexicons/")
        
        # Extract proper nouns
        logger.info("Extracting proper nouns from Akkadian texts...")
        proper_nouns_freq = builder.extract_proper_nouns(train_sources)
        logger.info(f"Found {len(proper_nouns_freq)} unique proper nouns (frequency > 0)")
        logger.info(f"Top proper nouns: {sorted(proper_nouns_freq.items(), key=lambda x: x[1], reverse=True)[:10]}")
        
        # Build character vocabularies
        logger.info("Building character vocabularies...")
        char_vocab_ak = builder.build_char_vocabulary(train_sources, language='akkadian')
        char_vocab_en = builder.build_char_vocabulary(train_targets, language='english')
        logger.info(f"Akkadian vocabulary: {len(char_vocab_ak)} characters")
        logger.info(f"English vocabulary: {len(char_vocab_en)} characters")
        
        # Build word vocabularies
        logger.info("Building word vocabularies...")
        word_vocab_ak = builder.build_word_vocabulary(train_sources, language='akkadian', min_freq=2)
        word_vocab_en = builder.build_word_vocabulary(train_targets, language='english', min_freq=2)
        logger.info(f"Akkadian vocabulary: {len(word_vocab_ak)} words")
        logger.info(f"English vocabulary: {len(word_vocab_en)} words")
        
        # Build BPE tokenizer
        logger.info("Building BPE tokenizer for Akkadian...")
        bpe_tokenizer = builder.build_bpe_tokenizer(train_sources, vocab_size=5000)
        if bpe_tokenizer:
            logger.info("Successfully built BPE tokenizer")
        
        # Save lexicons
        logger.info("Saving lexicons...")
        builder.save_lexicons()
        
        logger.info("Vocabulary building complete!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
