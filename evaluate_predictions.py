#!/usr/bin/env python3
"""
Evaluate predictions quality and identify improvement areas.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_predictions():
    """Analyze prediction quality."""
    
    project_root = Path(__file__).parent
    pred_path = project_root / "predictions.csv"
    
    if not pred_path.exists():
        logger.error(f"Predictions file not found: {pred_path}")
        return
    
    logger.info("="*80)
    logger.info("PREDICTIONS QUALITY ANALYSIS")
    logger.info("="*80)
    
    # Load predictions
    pred_df = pd.read_csv(pred_path)
    logger.info(f"\n✓ Loaded {len(pred_df)} predictions\n")
    
    # Analyze each prediction
    for idx, row in pred_df.iterrows():
        text = row['translation']
        logger.info(f"ID {row['id']}:")
        logger.info(f"  Length: {len(text)} chars, {len(text.split())} words")
        
        # Issues detection
        issues = []
        
        # Check for repetition
        words = text.split()
        if len(words) > 0:
            word_freq = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1
            repeated = [w for w, c in word_freq.items() if c > 3]
            if repeated:
                issues.append(f"High repetition: {repeated[:3]}")
        
        # Check for punctuation issues
        if text.count('(') != text.count(')'):
            issues.append("Unbalanced parentheses")
        if text.count('"') % 2 != 0:
            issues.append("Unbalanced quotes")
        
        # Check for ellipsis and placeholders
        if "..." in text:
            issues.append("Contains ellipsis (...)")
        if "of ... " in text or " ... " in text:
            issues.append("Has incomplete/placeholder segments")
        
        # Check grammar indicators
        if text.startswith("the"):
            issues.append("Unusual start with 'the'")
        if " the the " in text:
            issues.append("Article repetition ('the the')")
        
        if issues:
            for issue in issues:
                logger.info(f"  ⚠ {issue}")
        else:
            logger.info(f"  ✓ No obvious issues")
        
        logger.info(f"  Text: {text[:120]}...")
        logger.info("")
    
    # Summary analysis
    logger.info("="*80)
    logger.info("KEY PROBLEMS IDENTIFIED:")
    logger.info("="*80)
    logger.info("""
1. WORD REPETITION
   - Words like "the", "of", "and" repeated excessively
   - Decoder stuck in repetition loops
   - Issue: Greedy decoding with no diversity penalty

2. INCOMPLETE TRANSLATIONS
   - Text trails off with "..." and placeholders
   - Decoder not properly decoding full sequences
   - Issue: Decoding stopping early or length penalties

3. WORD ORDER ISSUES
   - Unnatural English sentence structure
   - Missing connectors, poor flow
   - Issue: Model not learning proper English syntax

4. GRAMMAR ERRORS
   - Missing articles, prepositions
   - Verb-subject-object ordering incorrect
   - Issue: Low-resource language pair, 1,561 training samples

5. SPECIAL CHARACTER HANDLING
   - Akkadian proper nouns sometimes appear (Ali-ahum, Aššur-muttabbil)
   - Mixed code-switching between Akkadian and English
   - Issue: Tokenizer treating proper nouns as separate tokens

6. NUMERICAL VALUES
   - Numbers like "4.3333", "8", "114" appear in output
   - These are from training data but might be noise
   - Issue: Model learning to output numbers without context
    """)
    
    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENTS TO IMPLEMENT:")
    logger.info("="*80)
    logger.info("""
IMMEDIATE (DECODING LEVEL):
1. ✓ BEAM SEARCH - Replace greedy decoding with beam_width=5
   - Explores multiple hypotheses
   - Reduces repetition tendency
   
2. ✓ LENGTH PENALTY - Avoid short outputs
   - Penalize early EOS token selection
   - Force decoder to generate full sequences
   
3. ✓ NO REPEAT N-GRAMS - Prevent same n-grams from appearing twice
   - Directly addresses repetition issue
   - Common in seq2seq decoding
   
4. ✓ TEMPERATURE SAMPLING - Use temperature > 1.0 for diversity
   - Softens probability distribution
   - Reduces mode collapse on repeated words

MEDIUM TERM (TRAINING):
5. LONGER TRAINING - Currently stopped at epoch 100
   - Validation loss was still ~6.5 (high)
   - Could train to 200+ epochs with patience
   - Better feature learning would reduce repetition
   
6. BETTER TOKENIZATION - Move to subword tokenization
   - BPE (Byte-Pair Encoding) instead of word-level
   - Better handles OOV words and morphology
   - Proper nouns get broken into components
   
7. COPY MECHANISM - Allow decoder to copy from source
   - Akkadian proper nouns should copy through
   - Numbers should copy unchanged
   - Implemented as pointer-generator network

8. VOCABULARY EXPANSION - Include OOV handling
   - Current: 10,015 target vocabulary
   - Should map rare words → similar frequent words
   - Or use <UNK> replacement strategy

LONG TERM (DATA & MODEL):
9. DATA AUGMENTATION - Generate synthetic training data
   - Back-translation: English → Akkadian → English
   - Paraphrasing of existing translations
   - Domain-specific augmentation
   
10. LARGER MODEL - Increase model capacity
    - Current: 3-layer LSTM (768 hidden)
    - Try: 4-6 layers, 1024 hidden units
    - Transformer architecture (better for translation)
    
11. EXTERNAL KNOWLEDGE - Leverage lexicons
    - OA_Lexicon_eBL.csv has ~3,500 word pairs
    - Initialize embeddings with lexicon
    - Constraint decoder to respect lexicon mappings
    """)

if __name__ == "__main__":
    analyze_predictions()
