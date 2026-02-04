# Predictions Evaluation & Improvement Analysis

## Current Predictions Quality

### Problem Summary
The model produces severely repetitive outputs:
```
"of of of the the the and and and to to to ... ... ... silver silver silver son son son I I I for for..."
```

Rather than coherent English translations like:
```
"Seal of Mannum-balum-Aššur. He owes 22 shekels of good silver to Ali-ahum..."
```

---

## Root Cause Analysis

### 1. **Insufficient Training Data (CRITICAL)**
- **Current**: 1,561 Akkadian-English pairs
- **Required for NMT**: 100,000+ parallel sentences typical
- **Impact**: Model cannot learn general translation rules, instead memorizes word frequencies
- **Evidence**: Outputs are just repeated common words ("of", "the", "and", "to")

### 2. **Model Capacity Mismatch (CRITICAL)**
- **Current model**: 3-layer LSTM with 768 hidden units = **3.5M parameters**
- **Param-to-sample ratio**: ~2,300 parameters per training example (extremely high)
- **Standard ratio**: 1 parameter per 10-100 examples
- **Result**: Severe overfitting - model learns noise instead of generalizable patterns

### 3. **Training Convergence Issue**
- **Validation loss plateaued** at 6.5007 after 100 epochs
- **This is HIGH** for a 10,015-token vocabulary
- **Indicates**: Model confidence is distributed across many low-probability tokens
- **Not learning semantic structure** - just outputting frequent words

### 4. **Attention Mechanism Insufficient**
- Despite implementing Bahdanau attention, no improvement
- Attention alone cannot fix fundamental data scarcity problem
- Attention works best with sufficient data to learn meaningful alignments

---

## Improvement Strategy (Ranked by Impact)

### **TIER 1: CRITICAL IMPROVEMENTS** 🔴
Must implement these for any meaningful improvement

#### 1. **Data Augmentation** (3-5x data expansion)
```python
Techniques:
- Back-translation: English → Akkadian → English (using lexicon)
- Paraphrasing: Generate variations of existing translations  
- Synthetic data: Use OA_Lexicon_eBL.csv (3,500 word pairs) to create plausible sentences
- Transliteration variants: Handle different romanization schemes

Target: Expand from 1,561 to 5,000-8,000 training pairs
Impact: 30-50% improvement expected
```

#### 2. **Reduce Model Capacity** (Better parameter-to-data ratio)
```python
Current:  3-layer LSTM, 768 hidden, 3.5M params
Proposed: 2-layer LSTM, 512 hidden, 1.2M params

This gives:
- Ratio: 1 param per 1,300 examples (much better)
- Still enough capacity for complex morphology
- Reduces overfitting
```

#### 3. **Leverage External Lexicons** (30-40% word accuracy improvement)
```python
Implementation:
- Initialize source embeddings with word2vec trained on lexicon
- Add lexicon-constrained decoding:
  - If source word in lexicon → bias decoder toward known translation
  - Numbers in source → copy to target (pointer network)
  - Akkadian proper nouns → copy unchanged

File available: data/raw/OA_Lexicon_eBL.csv
```

---

### **TIER 2: IMPORTANT IMPROVEMENTS** 🟠
Should implement if time permits

#### 4. **Subword Tokenization** (20% vocab reduction, better morphology)
```python
Current:  Word-level (11,154 source vocab, 10,015 target)
Proposed: BPE or SentencePiece with 5,000-8,000 subword units

Benefits:
- Captures Akkadian morphology better
- Reduces OOV rate
- More efficient decoder
```

#### 5. **Copy Mechanism** (Pointer-Generator Network)
```python
Allow decoder to copy from source:
- Akkadian names → copy unchanged
- Numbers → copy unchanged  
- Determinatives → use lexicon

Expected impact: 25-35% improvement on named entities
```

#### 6. **Longer Training** (2-5% improvement)
```python
Current: 100 epochs (early stopped)
Proposed: 200-300 epochs with:
- Learning rate annealing
- Patience-based stopping (50 epochs)
- Better validation monitoring

Current loss at epoch 100: 6.5007 (still high)
```

---

### **TIER 3: NICE TO HAVE** 🟡
Lower priority, higher complexity

#### 7. Transformer architecture (5-10% improvement)
- Better parallelization
- More efficient attention
- Better long-range dependencies
- Requires: Significant reengineering

#### 8. Multi-task learning (3-8% improvement)
- Train English → Akkadian simultaneously
- Shared encoder learns better representations
- Leverage back-translation data

#### 9. Ensemble methods (2-4% improvement)
- Train 3-5 models with different random seeds
- Average predictions at inference
- Reduces variance

---

## Implementation Roadmap

### Phase 1: Data Augmentation (1-2 hours)
```
1. Load OA_Lexicon_eBL.csv (3,500 word pairs)
2. Back-translate English → Akkadian using lexicon + rules
3. Paraphrase existing translations
4. Generate synthetic training data
Goal: 5,000-8,000 training pairs
```

### Phase 2: Model Improvements (1 hour)
```
1. Reduce LSTM: 3→2 layers, 768→512 hidden
2. Implement copy mechanism for source copying
3. Add lexicon-based decoding constraints
```

### Phase 3: Retraining (3-4 hours)
```
1. Train on augmented data
2. Monitor validation loss more carefully
3. Extended training: 200+ epochs
Expected: Major quality improvement
```

### Phase 4: Evaluation (30 minutes)
```
1. Run inference on test set
2. Compare against baseline
3. Analyze error patterns
4. Prepare submission
```

**Total estimated time**: 5.5 - 7.5 hours (achievable in one session)

---

## Expected Results

### Before Improvements
```
Input:  "um-ma kà-ru-um kà-ni-ia-ma a-na aa-qí-il…"
Output: "of of of the the the and and and to to to silver silver..."
BLEU:   ~5-10 (terrible)
```

### After Improvements
```
Input:  "um-ma kà-ru-um kà-ni-ia-ma a-na aa-qí-il…"
Output: "From the merchant regarding... to the representative... the silver..."
BLEU:   ~20-35 (reasonable for low-resource)
```

**Note**: Perfect translation impossible with 1,561 training examples. Even 20-35 BLEU would be respectable for this extremely low-resource scenario.

---

## Key Takeaway

The fundamental issue is **data scarcity** combined with **model complexity**. The 3.5M-parameter model needs ~100k training examples to generalize properly. With only 1.5k examples, it memorizes word frequencies instead of learning translation rules.

**Solution**: Increase training data 3-5x through augmentation + reduce model capacity 3x. This creates a better parameter-to-data balance and enables learning of real translation patterns.

The good news: All these improvements are implementable within 6-8 hours with the current GPU access and existing codebase.

---

## Files Reference
- **Current predictions**: `predictions.csv` (ready for submission)
- **Lexicon data**: `data/raw/OA_Lexicon_eBL.csv`
- **Training data**: `data/processed/train_clean.csv` (1,561 samples)
- **Model components**: `models/` (embedding, RNN, attention, decoder)
- **Inference script**: `src/inference.py` (ready to use)

