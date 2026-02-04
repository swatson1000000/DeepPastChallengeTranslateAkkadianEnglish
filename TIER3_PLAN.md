# TIER 3 IMPLEMENTATION PLAN

**Date**: February 3, 2026  
**Status**: Planning Phase  
**Priority**: Advanced improvements on top of TIER 1 & TIER 2

## Overview

TIER 3 advances the Akkadian-English translation system with sophisticated techniques that build on TIER 1 and TIER 2 foundations.

### TIER 3 Components

**Priority 1 (Recommended):**
1. **Beam Search Decoding** - Better translation quality
2. **Subword Tokenization (SentencePiece)** - Handle morphology
3. **Back-translation Data Augmentation** - Generate synthetic training data

**Priority 2 (Nice to have):**
4. **Multi-task Learning** - Bidirectional translation
5. **Ensemble Methods** - Multiple models
6. **Transformer Architecture** - If time permits

---

## Priority 1: Beam Search Decoding

### Current State (TIER 1-2)
- Greedy decoding (pick highest probability token at each step)
- Fast but often suboptimal
- No exploration of alternatives

### TIER 3 Implementation
- **Beam Search width: 5-10** - Explore multiple paths
- **Length normalization** - Prevent short sequences
- **Coverage penalty** - Prevent repetition (with TIER 2)
- **Language model reranking** - Score beams with LM

### Expected Impact
- **+3-8 BLEU improvement** over greedy
- Better quality translations
- Slightly slower inference (manageable)

### File
- `src/beam_search.py` - Beam search implementation
- Updated `inference.py` to use beam search

---

## Priority 1: Subword Tokenization (SentencePiece)

### Current State (TIER 1-2)
- Word-level tokenization
- Large vocabulary: 11k+ tokens
- Poor handling of morphology

### TIER 3 Implementation
- **SentencePiece BPE** - Learns subwords from data
- **Vocabulary: 4k-8k tokens** - More compact
- **Morphological awareness** - Breaks words into parts
- **Unknown word handling** - Always valid tokens

### Expected Impact
- **20% vocab reduction**
- **Better OOV handling**
- **+5-10 BLEU improvement**

### Files
- `src/subword_tokenizer.py` - SentencePiece wrapper
- `src/train_tokenizers.py` - Learn BPE models

---

## Priority 1: Back-translation Data Augmentation

### Current State (TIER 1)
- Manual augmentation: 1.71x expansion
- Pattern-based paraphrasing
- Limited coverage

### TIER 3 Implementation
- **English → Akkadian synthesis** - Using TIER 2 model
- **Create English-only corpus** - Monolingual data
- **Back-translate to create pairs** - Synthetic training data
- **Quality filtering** - Keep high-confidence pairs

### Expected Impact
- **+30-50% more training data**
- **Better vocabulary coverage**
- **+2-5 BLEU improvement**

### Files
- `src/back_translate.py` - Back-translation pipeline
- `scripts/generate_synthetic_data.py` - Data generation

---

## Priority 2: Multi-task Learning

### Concept
Train two directions simultaneously:
- Akkadian → English (main task)
- English → Akkadian (auxiliary task)

### Benefits
- Shared encoder learns better representations
- Leverage English monolingual data
- More robust to data scarcity

### Expected Impact: **+3-8 BLEU**

### Files
- `src/models/bidirectional_seq2seq.py` - Bi-directional model
- `configs/model_seq2seq_multitask.yaml` - Configuration

---

## Priority 2: Ensemble Methods

### Concept
Train 3-5 models with different random seeds, average predictions

### Approach
1. Train TIER 2 model × 5 with different seeds
2. At inference: run all 5 models
3. Average probabilities
4. Pick best token

### Expected Impact: **+2-4 BLEU**, +robustness

### Files
- `src/ensemble_inference.py` - Ensemble prediction
- `scripts/train_ensemble.py` - Train all models

---

## TIER 3 Architecture

```
TIER 1 (Foundation)
├─ Data Augmentation (1.71x)
├─ Model Capacity Reduction
└─ Lexicon Integration

TIER 2 (Advanced)
├─ Copy Mechanism
├─ Lexicon Constraints
└─ Coverage Tracking

TIER 3 (Sophisticated)
├─ Beam Search            [Priority 1]
├─ Subword Tokenization   [Priority 1]
├─ Back-translation       [Priority 1]
├─ Multi-task Learning    [Priority 2]
├─ Ensemble Methods       [Priority 2]
└─ Transformer            [Priority 3]
```

---

## Implementation Priority

### Phase 1: Beam Search (Easy, High Impact)
- Time: 1-2 hours
- Impact: +3-8 BLEU
- Complexity: Medium

### Phase 2: Subword Tokenization (Medium Effort)
- Time: 2-3 hours
- Impact: +5-10 BLEU
- Complexity: Medium-High

### Phase 3: Back-translation (Medium Effort)
- Time: 2-4 hours
- Impact: +2-5 BLEU
- Complexity: Medium

### Phase 4: Multi-task Learning (Harder)
- Time: 4-6 hours
- Impact: +3-8 BLEU
- Complexity: High

### Phase 5: Ensemble (Time-consuming but Easy)
- Time: 6-12 hours (training) + 2-3 hours (implementation)
- Impact: +2-4 BLEU
- Complexity: Medium

### Phase 6: Transformer (Most Work)
- Time: 8-16 hours
- Impact: +5-10 BLEU
- Complexity: Very High

---

## Expected Cumulative Improvements

| Baseline | TIER 1 | TIER 2 | TIER 3 Full | 
|----------|--------|--------|------------|
| BLEU: 5-8 | BLEU: 12-15 | BLEU: 18-23 | BLEU: 25-35 |
| | +50-100% | +50-100% | +25-50% |

---

## Commands (To Be Implemented)

### Train TIER 3 with Beam Search + Subword
```bash
python train.py --model tier3 --epochs 350 \
  --use-beam-search --beam-width 8 \
  --use-subword --vocab-size 6000
```

### Generate TIER 3 Predictions
```bash
python inference.py --model tier3 \
  --use-copy --use-beam-search --beam-width 8 \
  --subword-model models/spm.model
```

### Generate Back-translation Data
```bash
python scripts/generate_synthetic_data.py \
  --model tier2 --output data/synthetic_pairs.csv
```

### Train Ensemble
```bash
python scripts/train_ensemble.py --num-models 5 --model tier2
```

### Generate Ensemble Predictions
```bash
python inference.py --ensemble \
  --checkpoints checkpoints/tier2_*.pt \
  --output predictions_ensemble.csv
```

---

## Files to Create

### Core Components
- `src/beam_search.py` - Beam search decoder
- `src/subword_tokenizer.py` - SentencePiece wrapper
- `src/back_translate.py` - Back-translation pipeline
- `src/ensemble_inference.py` - Ensemble prediction
- `src/models/bidirectional_seq2seq.py` - Multi-task model

### Configuration
- `configs/model_seq2seq_tier3.yaml` - TIER 3 settings
- `configs/model_seq2seq_multitask.yaml` - Multi-task config
- `configs/model_seq2seq_transformer.yaml` - Transformer config

### Scripts
- `scripts/generate_synthetic_data.py` - Data generation
- `scripts/train_ensemble.py` - Train multiple models
- `scripts/train_subword_tokenizers.py` - Learn BPE
- `verify_tier3.py` - Verification script

### Documentation
- `TIER3_IMPLEMENTATION.md` - Full details
- `TIER3_QUICK_REFERENCE.md` - Quick guide
- Updated `train.py` for TIER 3
- Updated `inference.py` for TIER 3

---

## Next Steps

1. **Implement Beam Search** (Easy win)
   - Update `inference.py` with beam search decoder
   - Create config option `--use-beam-search`

2. **Implement Subword Tokenization**
   - Add SentencePiece model training
   - Update tokenizers

3. **Implement Back-translation**
   - Create back-translation pipeline
   - Generate synthetic training data

4. **Consider Multi-task Learning**
   - If resources and time allow

5. **Consider Ensemble**
   - If individual models perform well

---

## Expected Timeline

- **Phase 1-3 (Beam + Subword + Back-translation)**: 6-9 hours
- **Phase 4 (Multi-task)**: 4-6 hours additional
- **Phase 5 (Ensemble)**: 8-12 hours (mostly training)
- **Phase 6 (Transformer)**: 8-16 hours

---

## Success Metrics

- **TIER 3 Target BLEU**: 25-35
- **vs TIER 2**: +5-20 BLEU improvement
- **vs Baseline**: +20-27 BLEU improvement
- **Reduced repetition**: <5% repeated noun phrases
- **Better morphology**: +30% OOV handling improvement
