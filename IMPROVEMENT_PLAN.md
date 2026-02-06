# Training Improvement Plan

## Problem Analysis
Current training shows severe overfitting:
- **Fold 1 Best Model:** Train loss 0.1481, Val loss 1.0242, Ratio: **6.92x** 
- **Fold 2 Best Model:** Train loss 0.2463, Val loss 0.9824, Ratio: **3.99x**
- **Translation Quality:** Poor, lots of `<UNK>` tokens despite low val loss

**Root Cause:** Model learns training data distribution but fails to generalize to test data. Cross-entropy loss is a bad proxy for translation quality.

---

## Priority 1: Quickest Wins (5-15 minutes each)

### 1. Add Dropout Regularization
**Impact:** Reduce overfitting by 20-30%  
**Implementation:**
```python
# In LSTM encoder
self.lstm = nn.LSTM(
    input_size, hidden_size, num_layers,
    batch_first=True,
    dropout=0.3 if num_layers > 1 else 0  # Dropout between layers
)

# In decoder
decoder = nn.Sequential(
    nn.Linear(hidden_size * 2, 512),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(512, vocab_size)
)
```
**Why:** Forces model to use multiple feature representations, prevents co-adaptation of neurons.

---

### 2. Add Weight Decay to Optimizer
**Impact:** Reduce overfitting by 15-25%  
**Implementation:**
```python
# In train.py, line ~412
optimizer = torch.optim.Adam(
    params,
    lr=learning_rate,
    weight_decay=1e-4  # Add this line
)
```
**Why:** Penalizes large weights, encourages simpler models that generalize better.

---

### 3. Add Gradient Clipping
**Impact:** Stabilize training, prevent NaN losses  
**Implementation:**
```python
# In train_epoch() after backward()
torch.nn.utils.clip_grad_norm_(
    [p for model in models for p in model.parameters()],
    max_norm=1.0
)
```
**Why:** Prevents exploding gradients that can cause training instability and NaN losses.

---

## Priority 2: High Impact (15-30 minutes each)

### 4. Label Smoothing Loss
**Impact:** Prevent overconfident predictions, reduce overfitting by 10-15%  
**Implementation:**
```python
# Replace standard CrossEntropyLoss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        # Hard targets: [0, 1, 0, ...]
        # Smooth targets: [0.01, 0.91, 0.01, ...]
        confidence = 1.0 - self.smoothing
        smooth_label = self.smoothing / self.num_classes
        
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_probs = torch.zeros_like(log_probs)
            true_probs.fill_(smooth_label)
            true_probs.scatter_(1, targets.unsqueeze(1), confidence)
        
        return torch.mean(torch.sum(-true_probs * log_probs, dim=-1))

# Use in train.py:
criterion = LabelSmoothingCrossEntropy(
    num_classes=tgt_vocab_size,
    smoothing=0.1
)
```
**Why:** Prevents model from becoming too confident on training examples, improves generalization.

---

### 5. Add BLEU Validation Metric
**Impact:** CRITICAL - Catch better models earlier  
**Implementation:**
```python
# Add to train.py
from collections import Counter
import math

def calculate_bleu(predictions, references, max_n=4, smooth=True):
    """Calculate BLEU score for translation quality."""
    def get_ngrams(words, n):
        return Counter(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    
    bleu_scores = []
    for pred_words, ref_words in zip(predictions, references):
        pred_ngrams = []
        ref_ngrams = []
        
        for n in range(1, max_n+1):
            pred_ng = get_ngrams(pred_words.split(), n)
            ref_ng = get_ngrams(ref_words.split(), n)
            
            overlap = sum(min(pred_ng[ng], ref_ng[ng]) for ng in pred_ng if ng in ref_ng)
            count = max(0, len(pred_words.split()) - n + 1)
            
            if count == 0:
                bleu_scores.append(0.0)
            else:
                precision = (overlap + smooth) / (count + smooth)
                bleu_scores.append(precision)
    
    return math.exp(sum(math.log(b) for b in bleu_scores if b > 0) / len(bleu_scores)) if bleu_scores else 0.0

# In validate() function:
def validate_with_bleu(models, criterion, val_data, batch_size, device, use_tier2=False, copy_mechanism=None, lexicon_decoder=None):
    """Validation with both loss and BLEU score."""
    val_loss = validate(models, criterion, val_data, batch_size, device, use_tier2, copy_mechanism, lexicon_decoder)
    
    # Generate predictions and calculate BLEU
    src_tokenizer, tgt_tokenizer = build_tokenizers(val_data)  # Helper function
    predictions = []
    references = []
    
    for idx in range(min(100, len(val_data))):  # Sample 100 examples
        pred = generate_translation(models, val_data[0][idx], src_tokenizer, tgt_tokenizer)
        ref = tgt_tokenizer.decode(val_data[1][idx])
        predictions.append(pred)
        references.append(ref)
    
    bleu_score = calculate_bleu(predictions, references)
    return val_loss, bleu_score

# Update model selection criteria:
if bleu_score > best_bleu_score and overfitting_ratio <= target_ratio:
    best_bleu_score = bleu_score
    save_model()
```
**Why:** Cross-entropy loss is not the right metric. BLEU directly measures translation quality.

---

## Priority 3: Medium Effort, High Payoff (30-60 minutes each)

### 6. Subword Tokenization (BPE)
**Impact:** 30-40% improvement in translation quality  
**Implementation:**
```python
# Install: pip install tokenizers

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# Train BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = Sequence([Lowercase(), NFD(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()

# Train on corpus
tokenizer.train_from_iterator(training_texts, trainer=BPETrainer(vocab_size=5000))

# Save and use
tokenizer.save("akkadian_tokenizer.json")
tokenizer.encode("um-ma kà-ru-um kà-ni-ia-ma")
```
**Benefits:**
- Reduces vocab from 11k to 1-2k tokens
- Handles OOV words by breaking into subwords
- Better generalization
- Less memorization

---

### 7. Learning Rate Scheduling Improvements
**Impact:** 15-20% improvement in final metrics  
**Implementation:**
```python
# Option A: Add warmup phase (recommended)
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,  # Warm up for first 500 steps
    num_training_steps=100 * num_batches_per_epoch
)

# Option B: Cosine annealing (alternative)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # Total epochs
    eta_min=1e-6
)

# In training loop:
scheduler.step()
```
**Why:** Warm-up helps training stability. Cosine annealing is better than step decay.

---

### 8. Ensemble Training
**Impact:** 20-30% improvement through model averaging  
**Implementation:**
```python
# After training all 5 folds, ensemble predictions
def ensemble_predict(test_data, fold_checkpoints):
    """Average predictions from all fold models."""
    vocab_size = 25060
    ensemble_logits = None
    
    for fold_idx, checkpoint_path in enumerate(fold_checkpoints):
        model = load_model(checkpoint_path)
        logits = model(test_data)  # Shape: (batch, seq_len, vocab)
        
        if ensemble_logits is None:
            ensemble_logits = logits / len(fold_checkpoints)
        else:
            ensemble_logits += logits / len(fold_checkpoints)
    
    predictions = torch.argmax(ensemble_logits, dim=-1)
    return predictions

# Collect predictions from:
# - checkpoints/fold_0/best_model.pt
# - checkpoints/fold_1/best_model.pt
# - ... fold_2, fold_3, fold_4
```
**Why:** Averaging reduces model variance, picks up patterns each fold missed.

---

## Recommended Implementation Sequence

### **Phase 1: Quick Stabilization (Day 1)**
1. ✅ Already done: Early stopping at ratio 2.5x with 20-epoch patience
2. Add dropout (5 min)
3. Add weight_decay=1e-4 (2 min)
4. Add gradient clipping (5 min)
5. Retrain fold 0 as test (~2-3 hours)

### **Phase 2: Better Model Selection (Day 2)**
6. Implement BLEU metric (20 min)
7. Retrain fold 1-2 using BLEU for selection (~4-6 hours)

### **Phase 3: Architecture Improvements (Day 3)**
8. Add label smoothing (10 min)
9. Add learning rate warmup (5 min)
10. Retrain fold 3-4-5 (~6-8 hours)

### **Phase 4: Final Polish (Optional)**
11. Implement BPE tokenization (30 min, but requires reprocessing data)
12. Ensemble all 5 fold predictions (10 min)

---

## Expected Results After Each Phase

| Phase | Expected Val Loss | Expected Bleu | Overfitting Ratio | Translation Quality |
|-------|-------------------|----------------|-------------------|-------------------|
| **Current** | 0.98-1.02 | ~2-5 | 3.99x-6.92x | Poor (lots of `<UNK>`) |
| **Phase 1** | 0.85-0.95 | ~5-8 | 2.5x-3.5x | Slightly better |
| **Phase 2** | 0.80-0.90 | ~8-12 | 2.0x-3.0x | Better, more coherent |
| **Phase 3** | 0.75-0.85 | ~12-15 | 1.8x-2.5x | Significantly better |
| **Phase 4** | 0.70-0.80 | ~15-20 | 1.5x-2.0x | Much improved |

---

## Checkpoint: When to Stop Training

**Per epoch, check:**
```
if val_loss >= best_val_loss * 0.995 and overfitting_ratio > 2.5x:
    # Start annealing (already implemented)

if epochs_in_annealing >= 20:
    # Stop, use best saved model
    break
```

---

## Next Steps

1. **Immediate:** Implement Phase 1 improvements and retrain fold 0
2. **Monitor:** Watch for overfitting ratio staying below 2.5x
3. **Validate:** Run inference on test set and compare quality
4. **Iterate:** If still poor quality, move to Phase 2 (BLEU metric)

---

## References
- Overfitting paper: https://arxiv.org/abs/1912.02292
- Label smoothing: https://arxiv.org/abs/1512.00567
- BLEU metric: https://aclanthology.org/P02-1040.pdf
- BPE tokenization: https://arxiv.org/abs/1508.07909
