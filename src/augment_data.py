#!/usr/bin/env python3
"""
Data augmentation for Akkadian-English translation.
Implements back-translation and synthetic data generation using lexicons.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAugmentor:
    """Augment training data using lexicon-based back-translation and paraphrasing."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.lexicon = self.load_lexicon()
        self.training_data = None
        
    def load_lexicon(self):
        """Load OA_Lexicon_eBL.csv and build translation mappings."""
        # Project root is parent of src directory
        lexicon_path = self.project_root / "data" / "raw" / "OA_Lexicon_eBL.csv"
        
        if not lexicon_path.exists():
            logger.warning(f"Lexicon not found at {lexicon_path}")
            return {}
        
        logger.info(f"Loading lexicon from {lexicon_path}")
        lex_df = pd.read_csv(lexicon_path)
        
        # Build Akkadian → English mappings
        mappings = defaultdict(set)
        
        for _, row in lex_df.iterrows():
            akkadian = str(row['form']).strip()
            english = str(row['norm']).strip()
            
            if akkadian and english and akkadian != 'nan' and english != 'nan':
                # Clean up the English form
                english = english.replace('_', ' ').lower()
                mappings[akkadian].add(english)
        
        logger.info(f"✓ Loaded {len(mappings)} lexicon entries")
        return mappings
    
    def load_training_data(self):
        """Load training data."""
        train_path = self.project_root / "data" / "processed" / "train_clean.csv"
        self.training_data = pd.read_csv(train_path)
        logger.info(f"✓ Loaded {len(self.training_data)} training samples")
        return self.training_data
    
    def paraphrase_translation(self, translation: str) -> list:
        """
        Generate paraphrases of English translations.
        Creates multiple variations using phrase substitution and rewording.
        """
        paraphrases = [translation]
        
        # Strategy 1: Common phrase variations
        variations = {
            'of silver': ['of silver', 'in silver', 'from silver'],
            'minas of': ['minas of', 'mina(s) of', 'minas'],
            'seal of': ['seal of', 'the seal of', 'seal from'],
            'said:': ['stated:', 'said:', 'declared:', 'spoke:'],
            'he said': ['he said', 'saying', 'stating', 'he states'],
            'gave to': ['gave to', 'presented to', 'transferred to'],
            'the king': ['the king', 'king', 'his majesty'],
            'and': ['and', ',', 'plus'],
            'was': ['was', 'is', 'being'],
            'made': ['made', 'created', 'produced', 'crafted'],
        }
        
        current = translation
        for original, replacements in variations.items():
            if original in current:
                for replacement in replacements:
                    if replacement != original:
                        variant = current.replace(original, replacement, 1)
                        if variant != translation and len(variant) > 5:
                            paraphrases.append(variant)
        
        # Strategy 2: Punctuation and formatting variations
        punct_variants = [
            translation.replace(':', ''),
            translation.replace(',', ';'),
            translation.upper(),
            translation.title(),
        ]
        paraphrases.extend([p for p in punct_variants if p and p != translation])
        
        # Keep unique paraphrases, return up to 8
        return list(set(paraphrases))[:8]
    
    def backtranslate_from_lexicon(self, translation: str) -> str:
        """
        Back-translate English to Akkadian using lexicon.
        Then re-translate back to English for pseudo-paraphrase.
        """
        english_words = translation.lower().split()
        akkadian_tokens = []
        
        # Try to find Akkadian equivalents
        for word in english_words:
            found = False
            # Look for exact matches
            for akk, eng_set in self.lexicon.items():
                if word in eng_set:
                    akkadian_tokens.append(akk)
                    found = True
                    break
            
            if not found:
                # Keep English word if not found
                akkadian_tokens.append(word)
        
        # Re-translate back (simplified - just return modified version)
        if len(akkadian_tokens) > 3:
            # Rearrange to create variation
            indices = list(range(len(akkadian_tokens)))
            random.shuffle(indices[:5])  # Shuffle first few tokens
            shuffled = [akkadian_tokens[i] for i in indices]
            return ' '.join(shuffled[:len(english_words)])  # Match original length
        
        return translation
    
    def generate_synthetic_pairs(self, num_synthetic: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic training pairs using multiple strategies.
        """
        if self.training_data is None:
            self.load_training_data()
        
        logger.info(f"\nGenerating {num_synthetic} synthetic training pairs...")
        
        synthetic_pairs = []
        
        # Strategy 1: Paraphrase existing translations (largest portion)
        logger.info("  - Strategy 1: Paraphrasing translations")
        paraphrase_target = int(num_synthetic * 0.5)  # 50% from paraphrasing
        samples_needed = paraphrase_target
        
        for idx, row in self.training_data.iterrows():
            paraphrases = self.paraphrase_translation(row['translation'])
            for para in paraphrases[1:]:  # Skip original
                synthetic_pairs.append({
                    'transliteration': row['transliteration'],
                    'translation': para,
                    'augmentation_type': 'paraphrase'
                })
                samples_needed -= 1
                if samples_needed <= 0:
                    break
            if samples_needed <= 0:
                break
        
        # Strategy 2: Create variations by modifying translations
        logger.info("  - Strategy 2: Creating translation variations")
        variation_target = int(num_synthetic * 0.3)  # 30% from variations
        remaining = variation_target
        
        for idx, row in self.training_data.sample(min(remaining, len(self.training_data)), random_state=42).iterrows():
            translation = row['translation']
            
            # Multiple variation types
            variations = [
                translation.replace('(...)', '[details omitted]'),
                translation.replace('...', '[continues]'),
                translation.replace('  ', ' '),  # Normalize spaces
                translation + ' (continued)',
                translation.replace('[', '(').replace(']', ')'),
            ]
            
            for variation in variations:
                if variation != translation and len(variation) > 5:
                    synthetic_pairs.append({
                        'transliteration': row['transliteration'],
                        'translation': variation,
                        'augmentation_type': 'variation'
                    })
                    remaining -= 1
                    if remaining <= 0:
                        break
            if remaining <= 0:
                break
        
        # Strategy 3: Segment combination (creating new pairs from existing segments)
        logger.info("  - Strategy 3: Combining text segments")
        segment_target = int(num_synthetic * 0.2)  # 20% from segment combination
        remaining = segment_target
        
        for _ in range(min(remaining, len(self.training_data))):
            row1 = self.training_data.sample(1).iloc[0]
            row2 = self.training_data.sample(1).iloc[0]
            
            # Combine segments
            combined_trans = f"{row1['transliteration'][:40]} / {row2['transliteration'][:40]}"
            combined_trans = combined_trans.rstrip('/')
            combined_eng = f"{row1['translation']}; {row2['translation']}"
            
            if len(combined_trans) > 5 and len(combined_eng) > 10:
                synthetic_pairs.append({
                    'transliteration': combined_trans,
                    'translation': combined_eng,
                    'augmentation_type': 'segment_combination'
                })
                remaining -= 1
        
        logger.info(f"✓ Generated {len(synthetic_pairs)} synthetic pairs")
        logger.info(f"  - Paraphrasing: {sum(1 for p in synthetic_pairs if p['augmentation_type'] == 'paraphrase')}")
        logger.info(f"  - Variations: {sum(1 for p in synthetic_pairs if p['augmentation_type'] == 'variation')}")
        logger.info(f"  - Segment Combination: {sum(1 for p in synthetic_pairs if p['augmentation_type'] == 'segment_combination')}")
        return pd.DataFrame(synthetic_pairs)
    
    def augment_training_data(self, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Augment training data by combining original and synthetic pairs.
        
        Args:
            multiplier: Target size = original_size * multiplier
        
        Returns:
            Augmented dataset
        """
        if self.training_data is None:
            self.load_training_data()
        
        original_size = len(self.training_data)
        target_size = int(original_size * multiplier)
        num_synthetic = target_size - original_size
        
        logger.info(f"\n{'='*80}")
        logger.info(f"DATA AUGMENTATION")
        logger.info(f"{'='*80}")
        logger.info(f"Original dataset: {original_size} samples")
        logger.info(f"Target dataset: {target_size} samples")
        logger.info(f"Synthetic pairs to generate: {num_synthetic}")
        
        # Generate synthetic pairs
        synthetic_df = self.generate_synthetic_pairs(num_synthetic)
        
        # Combine original and synthetic
        augmented_df = pd.concat([
            self.training_data,
            synthetic_df
        ], ignore_index=True)
        
        # Shuffle
        augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"AUGMENTATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Final dataset size: {len(augmented_df)} samples")
        logger.info(f"Original: {original_size}, Synthetic: {len(synthetic_df)}")
        logger.info(f"Augmentation factor: {len(augmented_df) / original_size:.2f}x")
        
        return augmented_df
    
    def save_augmented_data(self, augmented_df: pd.DataFrame, output_path: str = None):
        """Save augmented data to CSV."""
        if output_path is None:
            output_path = self.project_root / "data/processed/train_augmented.csv"
        else:
            output_path = Path(output_path)
        
        # Keep only original columns
        output_df = augmented_df[['transliteration', 'translation']].copy()
        
        output_df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved augmented data to {output_path}")
        logger.info(f"  Rows: {len(output_df)}")
        
        return output_path

def main():
    """Run data augmentation."""
    project_root = Path(__file__).parent.parent  # src/augment_data.py -> project root
    
    augmentor = DataAugmentor(project_root)
    
    # Augment training data (6.25x expansion)
    augmented_df = augmentor.augment_training_data(multiplier=6.25)
    
    # Save augmented data
    output_path = augmentor.save_augmented_data(augmented_df)
    
    # Show samples
    logger.info(f"\nSample augmented pairs:")
    logger.info("─" * 80)
    for idx in range(min(5, len(augmented_df))):
        row = augmented_df.iloc[idx]
        logger.info(f"\nPair {idx+1}:")
        logger.info(f"  Akkadian: {row['transliteration'][:80]}...")
        logger.info(f"  English:  {row['translation'][:80]}...")

if __name__ == "__main__":
    main()
