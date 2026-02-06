#!/usr/bin/env python3
"""
Aggressive data augmentation for Akkadian-English translation.
Creates large volumes of training data from original 1,562 samples.

Augmentation strategies:
  - Token-level perturbations (insertion, deletion, swap, dropout)
  - Phrase paraphrasing (English)
  - Repetition with noise
  - Morphological variations
  - Punctuation/formatting variations

Target: Generate 50,000+ samples from 1,562 original
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
import random
import string

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AggressiveAugmentor:
    """Generate massive amounts of augmented training data."""
    
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        self.data = None
        self.augmentation_factor = 30  # 30x multiplication
    
    def load_data(self):
        """Load original training data."""
        path = self.project_root / "data" / "processed" / "train_clean.csv"
        self.data = pd.read_csv(path)
        logger.info(f"✓ Loaded {len(self.data)} original samples")
        return self.data
    
    def perturb_akkadian(self, text, strategy='mild'):
        """Apply token-level perturbations to Akkadian text."""
        tokens = text.split()
        
        if strategy == 'mild':
            # Minimal perturbation
            if len(tokens) > 2 and random.random() < 0.3:
                # Swap two random tokens
                i, j = random.sample(range(len(tokens)), 2)
                tokens[i], tokens[j] = tokens[j], tokens[i]
        
        elif strategy == 'moderate':
            # Token dropout (remove some tokens)
            if random.random() < 0.4:
                keep_tokens = [t for t in tokens if random.random() > 0.15]
                if keep_tokens:
                    tokens = keep_tokens
            
            # Token swap
            if len(tokens) > 2 and random.random() < 0.4:
                i, j = random.sample(range(len(tokens)), 2)
                tokens[i], tokens[j] = tokens[j], tokens[i]
        
        elif strategy == 'aggressive':
            # Aggressive perturbations
            # Dropout
            tokens = [t for t in tokens if random.random() > 0.2]
            
            # Swaps
            if len(tokens) > 2 and random.random() < 0.6:
                for _ in range(min(2, len(tokens) // 3)):
                    i, j = random.sample(range(len(tokens)), 2)
                    tokens[i], tokens[j] = tokens[j], tokens[i]
            
            # Duplication
            if random.random() < 0.3 and tokens:
                idx = random.randint(0, len(tokens) - 1)
                tokens.insert(idx, tokens[idx])
        
        return ' '.join(tokens) if tokens else text
    
    def paraphrase_english(self, text, method='synonym'):
        """Generate paraphrases of English translation."""
        # Common phrase synonyms and variations
        substitutions = {
            'he said': ['he declared', 'he stated', 'he spoke', 'he remarked'],
            'she said': ['she declared', 'she stated', 'she spoke'],
            'the king': ['king', 'his majesty', 'the ruler', 'royalty'],
            'the queen': ['queen', 'her majesty', 'the ruler', 'nobility'],
            'gave': ['presented', 'granted', 'bestowed', 'transferred'],
            'received': ['obtained', 'was given', 'accepted', 'took'],
            'silver': ['shekels of silver', 'silver coins', 'precious silver'],
            'minas of': ['minas of', 'mina(s) of'],
            'seal': ['royal seal', 'official seal', 'seal mark'],
            'said:': ['declared:', 'stated:', 'spoke:', 'exclaimed:'],
            'witnesses': ['witnesses', 'observers', 'bystanders'],
            'wrote': ['inscribed', 'recorded', 'documented', 'penned'],
            ' and ': [' and ', ', ', ' plus ',  ' with '],
            'was': ['became', 'is', 'proved'],
            'made': ['created', 'produced', 'fashioned', 'crafted'],
        }
        
        text_lower = text.lower()
        variants = [text]
        
        for original, replacements in substitutions.items():
            if original in text_lower:
                for replacement in replacements:
                    variant = text.replace(original, replacement, 1)
                    if variant != text and len(variant) > 5:
                        variants.append(variant)
        
        # Case variations
        variants.extend([text.upper(), text.title()])
        
        # Punctuation variations
        variants.extend([
            text.replace(':', ''),
            text.replace(',', ';'),
        ])
        
        return list(set(variants))
    
    def generate_augmented_batch(self, row, variations_per_row=30):
        """Generate multiple augmented versions of a single row."""
        original_akk = row['transliteration']
        original_eng = row['translation']
        
        augmented = []
        
        # Strategy 1: Straight paraphrasing (5 per row)
        paraphrases = self.paraphrase_english(original_eng, method='synonym')
        for para in paraphrases[:5]:
            augmented.append({
                'transliteration': original_akk,
                'translation': para,
                'augmentation': 'paraphrase'
            })
        
        # Strategy 2: Mild Akkadian + English paraphrase (8 per row)
        for i in range(8):
            perturbed_akk = self.perturb_akkadian(original_akk, strategy='mild')
            para = random.choice(self.paraphrase_english(original_eng))
            augmented.append({
                'transliteration': perturbed_akk,
                'translation': para,
                'augmentation': 'mild_perturbation'
            })
        
        # Strategy 3: Moderate Akkadian + English variation (10 per row)
        for i in range(10):
            perturbed_akk = self.perturb_akkadian(original_akk, strategy='moderate')
            para = random.choice(self.paraphrase_english(original_eng))
            augmented.append({
                'transliteration': perturbed_akk,
                'translation': para,
                'augmentation': 'moderate_perturbation'
            })
        
        # Strategy 4: Aggressive perturbation (7 per row)
        for i in range(7):
            perturbed_akk = self.perturb_akkadian(original_akk, strategy='aggressive')
            para = random.choice(self.paraphrase_english(original_eng))
            augmented.append({
                'transliteration': perturbed_akk,
                'translation': para,
                'augmentation': 'aggressive_perturbation'
            })
        
        # Strategy 5: Direct duplicates (normalize variations)
        # Ensure we have some clean copies with different paraphrases (5 more)
        for i in range(5):
            para = random.choice(self.paraphrase_english(original_eng))
            augmented.append({
                'transliteration': original_akk,
                'translation': para,
                'augmentation': 'clean_paraphrase'
            })
        
        return augmented
    
    def augment_all(self):
        """Generate augmented dataset for all rows."""
        if self.data is None:
            self.load_data()
        
        augmented_rows = []
        
        logger.info(f"\nGenerating {self.augmentation_factor}x augmented data...")
        logger.info(f"  Target: {len(self.data) * self.augmentation_factor:,} samples")
        
        for idx, row in self.data.iterrows():
            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed {idx + 1}/{len(self.data)}")
            
            batch = self.generate_augmented_batch(row, variations_per_row=self.augmentation_factor)
            augmented_rows.extend(batch)
        
        augmented_df = pd.DataFrame(augmented_rows)
        
        logger.info(f"\n✓ Generated {len(augmented_df):,} augmented samples")
        logger.info(f"  Augmentation breakdown:")
        logger.info(augmented_df['augmentation'].value_counts().to_string())
        
        return augmented_df
    
    def save_augmented(self, df, output_filename='train_augmented_aggressive_30x.csv'):
        """Save augmented data."""
        output_path = self.project_root / "data" / "processed" / output_filename
        
        # Select only the translation columns
        df_save = df[['transliteration', 'translation']].copy()
        df_save.to_csv(output_path, index=False)
        
        logger.info(f"\n✓ Saved to {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        return output_path


def main():
    """Generate aggressive augmented training data."""
    augmentor = AggressiveAugmentor()
    
    # Load original data
    augmentor.load_data()
    
    # Generate augmented data (30x multiplier)
    augmented = augmentor.augment_all()
    
    # Save
    output_path = augmentor.save_augmented(augmented, 'train_augmented_aggressive_30x.csv')
    
    logger.info("\n" + "="*80)
    logger.info("Augmentation complete!")
    logger.info("="*80)
    logger.info(f"\nOriginal samples: {len(augmentor.data):,}")
    logger.info(f"Augmented samples: {len(augmented):,}")
    logger.info(f"Multiplication factor: {len(augmented) / len(augmentor.data):.1f}x")
    logger.info(f"Output: {output_path}")
    logger.info("\nTo use in training:")
    logger.info("  python train.py --data-path data/processed/train_augmented_aggressive_30x.csv")


if __name__ == '__main__':
    main()
