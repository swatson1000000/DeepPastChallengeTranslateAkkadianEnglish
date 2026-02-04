# Deep Past Challenge - Machine Translation Design Plan
## Akkadian to English Translation

**Competition**: Deep Past Initiative - Machine Translation Challenge  
**Objective**: Build neural machine translation models to convert transliterated Old Assyrian (Akkadian) cuneiform text into English  
**Timeline**: December 16, 2025 - March 23, 2026  
**Dataset**: ~8,000 cuneiform texts with translations  
**Evaluation**: Geometric Mean of BLEU and chrF++ scores

---

## 1. Problem Analysis

### Challenge Characteristics
- **Low-resource language**: Only ~8,000 training examples (very small for NMT)
- **Morphologically complex**: Single Akkadian word often requires multiple English words
- **Heavily formatted text**: Numerous transliteration marks, determinatives, scribal notations
- **Specialized domain**: Ancient commercial/legal documents with unique vocabulary
- **Expert-level task**: Only ~12 scholars worldwide specialize in this language
- **Proper nouns**: Capitalization encodes semantic meaning (names, places, Sumerian logograms)

### Data Format Challenges
1. **Transliteration marks**: Superscripts, subscripts, special characters (š, ṭ, ḫ, etc.)
2. **Determinatives**: Curly brackets marking noun classifiers (e.g., {d} for god, {ki} for earth)
3. **Gaps/breaks**: <gap> for small breaks, <big_gap> for large breaks
4. **Scribal notations**: Line numbers with ', '', comments in (), insertions < >, etc.
5. **Character encoding issues**: Mix of Unicode standards, subscripts, accents
6. **Inconsistent capitalization**: Proper nouns, ALL CAPS logograms, determinatives

---

## 2. Project Phases

### Phase 1: Data Preparation & EDA (Week 1-2)
**Goal**: Clean, parse, and understand the training data

**Tasks**:
- Download train/test data from Kaggle API
- Exploratory Data Analysis (EDA):
  - Analyze text length distribution (source & target)
  - Vocabulary statistics (coverage, frequency)
  - Special character frequency
  - Proper noun analysis
  - Gap/break patterns
- Implement comprehensive text preprocessing pipeline:
  - Remove scribal notations (!, ?, /, :, . line dividers)
  - Normalize Unicode characters (Ḫ→H, accents, subscripts)
  - Handle determinatives (keep but tokenize separately or normalize)
  - Standardize gap markers
  - Remove/preserve structural elements
- Build parallel dataset in normalized format
- Create train/validation/test splits
- Build vocabulary and create lexicon file for proper nouns

**Deliverables**:
- `data_exploration.ipynb` - EDA notebook
- `preprocessing.py` - Preprocessing pipeline
- `processed_train.txt`, `processed_valid.txt` - Cleaned parallel texts
- `vocabulary.json`, `proper_nouns.json` - Lexicon files

---

### Phase 2: Baseline Models (Week 2-3)
**Goal**: Establish performance baselines with standard architectures

**Models to implement**:

1. **Rule-based heuristics**
   - Lookup table for known translations
   - Pattern matching for common structures
   
2. **Statistical Machine Translation (SMT)**
   - BLiSS or Moses baseline (if time permits)
   - Provides interpretability
   
3. **Sequence-to-Sequence with Attention**
   - Encoder-Decoder (GRU/LSTM)
   - Bahdanau attention
   - 2-3 layers, moderate embedding size (256-512)
   - Use pre-computed GloVe/FastText for initialization if possible

4. **Transformer-based (T5/mBART)**
   - Fine-tune mBART-50 (multilingual pre-trained)
   - Fine-tune mT5 (multilingual T5)
   - Few-shot prompting with GPT-2/DistilGPT-2 (if no commercial API available)

**Deliverables**:
- `baseline_models.py` - Implementation of all baselines
- `baseline_results.txt` - BLEU/chrF++ scores
- Training logs and hyperparameter settings

---

### Phase 3: Specialized Data Augmentation (Week 3-4)
**Goal**: Increase effective training data for low-resource scenario

**Techniques**:
1. **Back-translation**
   - English → Akkadian → English (using baseline models)
   - Iterative refinement
   
2. **Paraphrasing**
   - Paraphrase English translations while keeping meaning
   - Use T5/GPT-2 for paraphrase generation
   
3. **Multi-lingual transfer**
   - Incorporate other ancient languages (if available)
   - Use related modern Semitic languages (Hebrew, Arabic)
   
4. **Morphological awareness**
   - Create character-level representations
   - Segment words into morphemes when possible
   
5. **Synthetic data from patterns**
   - Extract and apply morphological patterns
   - Template-based generation

**Deliverables**:
- `data_augmentation.py` - Augmentation pipeline
- `augmented_train.txt` - Combined training data
- Analysis of augmentation quality

---

### Phase 4: Advanced Model Architecture (Week 4-5)
**Goal**: Optimize architecture for this specific task

**Approaches**:

1. **Specialized Tokenization**
   - Character-level + subword tokenization (SentencePiece)
   - Special handling for determinatives and gaps
   - Byte-pair encoding (BPE) with custom symbols

2. **Morphology-Aware Encoders**
   - Morphological analysis of Akkadian words
   - Separate encoding for morphemes
   - Hierarchical attention over morphemes

3. **Multi-task Learning**
   - Main task: Akkadian → English translation
   - Auxiliary tasks:
     - Proper noun recognition
     - Determinative classification
     - Gap/break prediction
     - Source word alignment

4. **Domain-Specific Embeddings**
   - Train embeddings on ancient text corpus
   - Incorporate semantic knowledge from lexicons
   - Use corpus-specific vocabulary

5. **Pointer Networks**
   - Copy mechanism for rare proper nouns
   - Attention over lexicon

**Deliverables**:
- `advanced_models.py` - Specialized architectures
- Ablation studies
- Training curves and convergence analysis

---

### Phase 5: Hyperparameter Optimization (Week 5)
**Goal**: Fine-tune best models for maximum performance

**Methods**:
- Grid search over learning rates, batch sizes, layers
- Learning rate scheduling (warmup, decay)
- Regularization (dropout, weight decay)
- Beam search tuning (beam width, length penalty)
- Ensembling strategies

**Deliverables**:
- `hyperparameter_search.py`
- Best configuration file

---

### Phase 6: Inference & Ensemble (Week 6)
**Goal**: Generate final predictions with ensemble

**Steps**:
1. Load best checkpoints
2. Ensemble multiple models:
   - Different architectures
   - Different random seeds
   - Different hyperparameter sets
3. Voting/averaging strategies
4. Post-processing:
   - Proper noun recovery
   - Inconsistency correction
   - Format validation

**Deliverables**:
- `inference.py` - Inference pipeline
- `submission.csv` - Final predictions
- `ensemble_weights.json` - Ensemble configuration

---

### Phase 7: Analysis & Documentation (Ongoing)
**Goal**: Understand model behavior and document findings

**Activities**:
- Error analysis:
  - Failure cases (long sentences, rare words, specific constructions)
  - BLEU vs chrF++ trade-offs
  - Proper noun handling
  - Determinative accuracy
  
- Attention visualization
- Saliency maps
- Ablation studies
- Comparative analysis

**Deliverables**:
- `error_analysis.ipynb`
- `model_analysis.ipynb`
- README with findings

---

## 3. Technical Stack

### Core Libraries
```
torch >= 2.0
transformers >= 4.30
datasets >= 2.0
sacrebleu  # For evaluation metrics
numpy
pandas
scikit-learn
```

### Optional
```
wandb  # Experiment tracking
tensorboard
pytorch-lightning  # Training framework
opennmt-py  # Statistical MT baseline
```

### Development Tools
```
jupyter
pytest
black
flake8
```

---

## 4. Project Structure

```
DeepPastChallengeTranslateAkkadianEnglish/
├── data/
│   ├── raw/
│   │   ├── train.txt
│   │   └── test.txt
│   ├── processed/
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   └── test.txt
│   ├── augmented/
│   │   └── augmented_train.txt
│   └── lexicons/
│       ├── vocabulary.json
│       └── proper_nouns.json
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── tokenization.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_seq2seq.py
│   │   ├── transformer_models.py
│   │   └── morphology_aware.py
│   ├── training.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── data_augmentation.py
│   └── utils.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_advanced_models.ipynb
│   └── 06_final_inference.ipynb
│
├── configs/
│   ├── preprocessing.yaml
│   ├── model_seq2seq.yaml
│   ├── model_transformer.yaml
│   └── training.yaml
│
├── checkpoints/
│   └── best_models/
│
├── results/
│   ├── baseline_results.txt
│   ├── training_logs/
│   └── error_analysis/
│
├── DESIGN_PLAN.md (this file)
├── README.md
├── requirements.txt
├── setup.py
└── Makefile
```

---

## 5. Key Strategies for Low-Resource Scenario

### 1. Transfer Learning
- Start with pre-trained models (mBART, mT5, mBERT)
- Fine-tune on task with careful regularization

### 2. Data Efficiency
- Back-translation for data augmentation
- Knowledge distillation from larger models
- Few-shot learning techniques

### 3. Morphological Awareness
- Character-level processing for inflections
- Sub-word tokenization (BPE)
- Morpheme-aware attention

### 4. Domain Specialization
- Custom vocabulary from lexicons
- Copy mechanism for proper nouns
- Multi-task learning with related tasks

### 5. Ensemble Approach
- Multiple diverse models
- Weighted averaging
- Voting on rare tokens

---

## 6. Evaluation Metrics

### Primary Metric
**Geometric Mean of BLEU and chrF++** (macro-averaged across corpus)

### Secondary Metrics
- BLEU-4 (word n-gram overlap)
- chrF++ (character n-gram overlap + word boundaries)
- METEOR (semantics + morphology)
- ROUGE (recall-oriented)

### Custom Metrics
- Proper noun accuracy (exact match)
- Determinative accuracy (binary classification)
- Gap handling correctness
- OOV (Out-of-Vocabulary) token handling rate

---

## 7. Timeline & Milestones

| Week | Phase | Milestones |
|------|-------|-----------|
| 1-2 | Data Prep | EDA complete, preprocessing pipeline done, clean data ready |
| 2-3 | Baselines | Seq2Seq working, mBART/mT5 fine-tuned, baseline BLEU ~15-20 |
| 3-4 | Augmentation | Augmentation pipeline ready, 2x+ training data |
| 4-5 | Advanced | Specialized models implemented, improvements to ~22-25 BLEU |
| 5 | Optimization | Hyperparameter tuning, convergence reached |
| 6 | Ensemble & Inference | Multiple models ensembled, final submission ready |
| 7 | Analysis | Error analysis, documentation complete |

---

## 8. Success Criteria

### Baseline Success
- BLEU score > 15 (better than random)
- Proper nouns handled with > 70% accuracy
- System can process all test instances

### Target Success
- BLEU/chrF++ average > 25
- Proper nouns > 85% accuracy
- Competitive with top submissions
- Top 5% on leaderboard

### Stretch Goals
- BLEU/chrF++ > 30
- Top 1% on leaderboard
- Interpretable error patterns
- Production-ready inference pipeline

---

## 9. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Small dataset (8K) | Model overfitting | Early stopping, regularization, augmentation |
| Unique formatting | Parsing errors | Comprehensive preprocessing tests, manual verification |
| Limited compute | Slow training | Distilled models, efficient architectures, caching |
| Low baseline | Difficult to improve | Transfer learning, ensemble, multi-task learning |
| Domain shift | OOV tokens | Morphological analysis, copy mechanism, lexicon |
| Time constraints | Incomplete work | Prioritize (data→baselines→ensemble), parallel work |

---

## 10. Training Optimization Updates (February 3, 2026)

### Quick-Win Improvements Implemented

After analyzing initial training and inference logs, the following improvements were implemented to address model collapse and poor inference quality:

#### 1. **Extended Early Stopping Patience (10 → 25 epochs)**
   - **Issue**: Model was stopping too early at epoch 120, missing potential recovery
   - **Solution**: Increased early stopping patience from 10 to 25 epochs
   - **Impact**: Allows model to continue training through plateaus, potentially reaching better convergence
   - **Implementation**: Updated `src/models/train.py` line 161
   ```yaml
   early_stopping_patience: 25  # was 10
   ```

#### 2. **Tokenizer Output Debugging**
   - **Issue**: Model outputs repetitive nonsense values (18.5, "Nanaya.") suggesting token corruption
   - **Solution**: Added logging to display first 20 target vocabulary tokens during training startup
   - **Impact**: Can now identify if numeric values like `18.5` are actual vocabulary tokens vs. data corruption
   - **Implementation**: Added validation in `src/models/train.py` lines 60-75
   ```python
   first_target_tokens = list(tgt_tokenizer.idx2word.items())[:20]
   logger.info("First 20 target tokens: {}".format(first_target_tokens))
   ```

#### 3. **Data Preprocessing Verification**
   - **Issue**: Inference showing suspicious numeric tokens (18.5), indicating data corruption
   - **Solution**: Added comprehensive data validation checks:
     - Min/max transliteration and translation lengths
     - Detection of suspicious numeric patterns in translations
     - Sample display for manual inspection
   - **Impact**: Can identify corrupted data entries that cause model collapse
   - **Implementation**: Added validation in `src/models/train.py` lines 77-97
   ```python
   # Check for suspicious numeric values
   has_numeric = df['translation'].str.contains(r'\d{2}\.\d', regex=True).sum()
   if has_numeric > 0:
       logger.warning(f"  ⚠ Found {has_numeric} rows with decimal numbers in translations!")
   ```

#### 4. **Beam Search Implementation for Inference**
   - **Issue**: Greedy decoding produces repetitive predictions; model lacks exploration of alternatives
   - **Solution**: Implemented beam search decoding with configurable beam width
   - **Impact**: Reduces repetition, explores multiple hypothesis sequences, improves translation quality
   - **Implementation**: Added `beam_search_decode()` method in `src/inference.py` lines 130-180
   ```python
   def beam_search_decode(
       self,
       encoder_outputs, hidden, cell,
       max_length=100,
       beam_width=3,
       vocab_size=3000
   ) -> str:
       # Tracks top beam_width sequences by score
       # Detects EOS token for early stopping
       # Returns best completed sequence
   ```

#### 5. **GRU Baseline Option for Debugging**
   - **Issue**: LSTM complexity makes it hard to isolate issues; model has 3 layers with bidirectionality
   - **Solution**: Added support for simpler GRU encoder as alternative
   - **Impact**: Reduces parameters, easier to debug, faster training for validation tests
   - **Implementation**: Added GRU support in `src/models/train.py` lines 119-140
   ```yaml
   use_gru: false  # Set to true to use GRU instead of LSTM
   ```
   - Usage: `config['training']['use_gru'] = True` to enable

### Expected Outcomes

| Improvement | Expected Benefit | Validation Method |
|-------------|-----------------|-------------------|
| Extended patience | Avoid premature stopping, better convergence | Check if training continues past epoch 120 |
| Tokenizer debugging | Identify data corruption root cause | Review first 20 vocab tokens in logs |
| Data validation | Catch corrupted entries early | Count `has_numeric` warnings during training |
| Beam search | Better predictions, less repetition | Compare inference outputs vs greedy |
| GRU baseline | Faster debugging iteration | Train GRU model in < half the time |

### Configuration for Quick Wins

To enable all improvements, ensure `configs/model_seq2seq.yaml` contains:
```yaml
training:
  epochs: 200
  early_stopping_patience: 25  # Increased from 10
  use_gru: false  # Set true for debugging
  batch_size: 128
  learning_rate: 0.0005
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
```

### Next Steps for Model Improvement

1. **Run training with debugging enabled** - Observe vocab tokens and data issues
2. **Investigate any `18.5` or numeric patterns** - Likely data preprocessing bug
3. **Compare GRU vs LSTM performance** - Simpler model may work better
4. **Apply beam search during inference** - Test with beam_width=3-5
5. **Analyze failure modes** - Use attention visualization to understand model predictions

### Files Modified

- `src/models/train.py` - Early stopping, debugging, GRU option
- `src/inference.py` - Beam search implementation
- `configs/model_seq2seq.yaml` - Training configuration

---

## 11. Advanced Training Techniques (February 3, 2026 - Continued)

### Intelligent Learning Rate Annealing & Checkpoint Management

Building on the quick-win improvements, implemented sophisticated training strategies to escape plateaus and preserve best models:

#### 1. **Plateau-Aware Learning Rate Reduction (ReduceLROnPlateau)**
   - **Issue**: Model plateau at epoch 113 without mechanism to fine-tune with smaller steps
   - **Solution**: Added `ReduceLROnPlateau` scheduler that monitors validation loss
   - **Mechanism**:
     - Waits 5 epochs for validation improvement
     - If no improvement, reduces LR by factor of 0.5 (e.g., 0.0005 → 0.00025)
     - Continues until min_lr of 1e-6 is reached
   - **Implementation**: `src/models/train.py` lines 187-195
   ```python
   plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer,
       mode='min',
       factor=0.5,
       patience=5,
       verbose=True,
       min_lr=1e-6
   )
   ```

#### 2. **Best Model Checkpointing**
   - **Issue**: No mechanism to recover from bad training decisions or exploration
   - **Solution**: Automatic checkpoint saving whenever validation loss improves
   - **Saves**:
     - Embedding weights
     - RNN (LSTM/GRU) weights
     - Attention layer weights (NEW)
     - Decoder weights
     - Optimizer state (for resuming training)
   - **Location**: `checkpoints/embedding_best.pt`, `rnn_best.pt`, `attention_best.pt`, `decoder_best.pt`

#### 3. **Aggressive Annealing After Plateau**
   - **Issue**: Standard plateau scheduler might not be aggressive enough after 5-10 epochs of stagnation
   - **Solution**: Trigger aggressive annealing when no improvement for 5 epochs
   - **Steps**:
     1. Restore best model from checkpoint
     2. Apply much more aggressive LR reduction (0.1x vs 0.5x)
     3. Reset early stopping counter (give 25 more epochs)
     4. Log all actions for transparency
   - **Impact**: Model gets second chance with dramatically smaller steps
   - **Implementation**: `src/models/train.py` lines 308-330
   ```python
   if epochs_since_improvement >= 5 and not aggressive_annealing_triggered:
       # Restore best model
       # Apply new_lr = current_lr * 0.1
       # Reset patience_counter = 0
       aggressive_annealing_triggered = True
   ```

#### 4. **Bahdanau Attention Mechanism**
   - **Critical Issue Found**: Training script was NOT using the attention layer at all!
     - Had attention defined in `seq2seq.py` but unused
     - Was using simple `Embedding → LSTM → Linear Decoder` without attention
   - **Solution**: Integrated Bahdanau attention into training pipeline
   - **Architecture**:
     ```
     Input → Embedding → LSTM → [Attention Layer] → Concatenated Decoder → Output
                                 (computes context     (context + hidden)
                                  from encoder outputs)
     ```
   - **Benefits**:
     - Model learns which source tokens to focus on
     - Prevents "garbage output" problem (18.5, repetitive tokens)
     - Helps with long sequences
     - Grounding predictions in source tokens
   - **Implementation**: `src/models/train.py` lines 24-48 (AttentionLayer class)
   ```python
   class AttentionLayer(nn.Module):
       def forward(self, query, keys):
           # query: (batch_size, hidden_dim) - last encoder output
           # keys: (batch_size, seq_len, hidden_dim) - all encoder outputs
           # Returns: context vector + attention weights
   ```

### Training Flow with New Features

```
Epoch 100: Val Loss = 5.95 → BEST MODEL SAVED ✓
Epochs 101-105: Val Loss increases, no improvement
            ↓ STANDARD PLATEAU SCHEDULER
            LR: 0.0005 → 0.00025 (0.5x reduction)
            Continue training

Epochs 106-110: Still no improvement
            ↓ AGGRESSIVE ANNEALING TRIGGERED
            1. Restore best model checkpoint (epoch 100)
            2. Apply aggressive LR reduction: 0.00025 → 0.000025 (0.1x)
            3. Reset early stopping counter (25 more epochs allowed)
            4. Try again with much finer gradient steps

Epochs 111-135: With aggressive annealing
            IF improvement: Reset counters, continue
            IF no improvement: Early stop at patience=25
```

### Configuration for Advanced Training

```yaml
# configs/model_seq2seq.yaml
training:
  epochs: 200
  early_stopping_patience: 25
  
  # Plateau-aware scheduler
  plateau_scheduler: true
  plateau_patience: 5           # Reduce LR after 5 epochs with no improvement
  plateau_factor: 0.5           # Reduce LR by 50%
  plateau_min_lr: 1.0e-6        # Floor for learning rate
  
  # Aggressive annealing
  aggressive_annealing: true
  aggressive_annealing_threshold: 5    # Trigger after 5 epochs with no improvement
  aggressive_annealing_factor: 0.1     # Reduce by 90% (10x more aggressive)
  
  # Attention mechanism
  use_attention: true           # NEW - Bahdanau attention enabled
  attention_hidden_dim: 768     # Size of attention computation
  
  use_gru: false               # false=LSTM, true=GRU for debugging
```

### Expected Impact of Changes

| Feature | Problem Solved | Expected Outcome |
|---------|----------------|------------------|
| Plateau scheduler (0.5x) | Normal plateau after 5 epochs | Fine-tune with reduced LR |
| Aggressive annealing (0.1x) | Deep plateau after 10+ epochs | Escape local minima with much smaller steps |
| Model checkpointing | Can't recover from bad epochs | Always restore best state, never degrade |
| Early stop counter reset | Runs out of patience too quickly | Get fresh 25 epochs after annealing |
| **Attention mechanism** | **Garbage outputs & repetition** | **Model focuses on source tokens** |

### Critical Fix: Attention Was Missing

The most critical discovery: The model had no attention mechanism in the training loop, which explains:
- ✗ Repetitive garbage outputs (18.5, "Nanaya")
- ✗ Poor semantic grounding
- ✗ Weak handling of long sequences
- ✗ Low inference quality despite decent training loss

Now with attention:
- ✓ Encoder outputs weighted by relevance
- ✓ Model learns alignment between source/target
- ✓ Better gradient flow for long sequences
- ✓ Interpretable attention weights for analysis

### Files Modified (Phase 2)

- `src/models/train.py` - Attention layer, plateau scheduler, checkpoint management, aggressive annealing
- `configs/model_seq2seq.yaml` - Training configuration with all new parameters

### Verification

✓ All 17 Python scripts compile successfully without syntax errors
✓ Attention layer properly integrated into forward pass
✓ Checkpoint saving/loading for all model components
✓ Gradient clipping updated for all 4 components (embedding, RNN, attention, decoder)

---

## 12. Overfitting-Based Early Stopping & Adaptive Learning Rate Annealing (February 4, 2026)

### Problem with Previous Approach
The earlier training strategy used **fixed patience counters** for early stopping, which had limitations:
- Stopped too early if validation loss increased even slightly (patience=20)
- Rigid time-based annealing schedule (reduce LR every 10 epochs) regardless of actual progress
- Didn't account for the fundamental train/val loss trade-off in low-resource scenarios
- Couldn't distinguish between normal fluctuations and actual overfitting

### Solution: Intelligent Overfitting Detection + Data-Driven Annealing

#### 1. **Overfitting Ratio-Based Early Stopping**
   - **Metric**: Track `train_loss / val_loss` ratio each epoch
   - **Stopping Criterion**: Stop only when ratio exceeds **2.0x** (severe overfitting)
   - **Safety Thresholds**:
     - Minimum 20 epochs before stopping (allow proper learning phase)
     - Maximum 15 epochs plateau (time-based safety net to save computation)
   - **Rationale**: A 2.0x ratio means train loss is half the validation loss—clear sign of severe overfitting
   - **Implementation**: `src/train.py` lines 495-545
   ```python
   overfitting_ratio = val_loss / train_loss if train_loss > 0 else float('inf')
   
   if epoch >= min_epochs and overfitting_ratio > overfitting_threshold:
       logger.info(f"SEVERE OVERFITTING: {overfitting_ratio:.2f}x ratio")
       break
   ```

#### 2. **ReduceLROnPlateau with Adaptive Reduction**
   - **Data-Driven**: Only reduces learning rate when validation loss plateaus (5 epochs)
   - **Gentle First Reduction**: Factor of 0.5 (reduce by 50%)
   - **Smart Cooldown**: 2-epoch cooldown between reductions to avoid oscillation
   - **Floor Protection**: Never reduce below 1e-7 to maintain gradient flow
   - **Threshold**: Minimum improvement of 0.01% required to reset plateau counter
   - **Implementation**: `src/train.py` lines 517-529
   ```python
   lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer,
       mode='min',
       factor=0.5,           # Reduce by 50% per reduction
       patience=5,           # After 5 epochs with no improvement
       threshold=0.0001,     # Minimum improvement required
       threshold_mode='rel',
       cooldown=2,           # Wait 2 epochs before next reduction
       min_lr=1e-7          # Floor
   )
   ```

#### 3. **Logging & Monitoring**
   - **Per-Epoch Output**: Shows train loss, val loss, and **overfitting ratio** 
   - **LR Changes**: Logs each learning rate reduction with plateau count and reduction number
   - **Early Stopping Reason**: Clear message indicating why training stopped (overfitting vs plateau)
   - **Final Summary**: Reports final overfitting ratio and total LR reductions applied

#### 4. **Expected Training Behavior**
   ```
   Epoch 1-20:  Train loss decreases, val loss decreases
               (model learning, ratio = 1.1-1.3x)
   
   Epoch 21-40: Train loss continues down, val loss plateaus
               (some overfitting, ratio = 1.5-1.8x)
   
   Epoch 41:    Val loss plateaus for 5 epochs
               → ReduceLROnPlateau triggers
               → Learning rate: 0.0005 → 0.00025 (0.5x)
   
   Epoch 42-50: Finer gradient steps help escape plateau
               (train improves further, val recovers)
   
   Epoch 51-60: Ratio climbs to 2.0x (train_loss << val_loss)
               → SEVERE OVERFITTING DETECTED
               → Training stops to preserve best checkpoint
   ```

### Comparison: Old vs New Approach

| Aspect | Old (Fixed Patience) | New (Overfitting-Based) |
|--------|-------------------|----------------------|
| **Early Stopping** | Patience=20 epochs | Stop when ratio > 2.0x |
| **LR Reduction** | Every 10 epochs (fixed) | Every 5 plateau epochs (adaptive) |
| **Rationale** | Time-based | Loss-based |
| **Flexibility** | Rigid | Adaptive to actual learning |
| **Computation** | May stop too early | Optimal stopping point |
| **Overfitting Control** | Weak (just stop) | Strong (only stop at 2.0x) |
| **Resource Efficiency** | Good | Better |

### Configuration

```yaml
# New parameters in src/train.py
training:
  overfitting_threshold: 2.0        # Stop at train/val ratio > 2.0x
  min_epochs: 20                    # Don't stop before epoch 20
  max_patience_on_plateau: 15       # Max 15 epochs plateau before stopping
  
  # ReduceLROnPlateau scheduler
  lr_scheduler:
    factor: 0.5                     # Reduce by 50% per step
    patience: 5                     # After 5 epochs no improvement
    threshold: 0.0001               # Minimum improvement threshold (0.01%)
    threshold_mode: 'rel'           # Relative improvement
    cooldown: 2                     # Wait 2 epochs between reductions
    min_lr: 1.0e-7                  # Learning rate floor
```

### Advantages Over Previous Approach

1. **Scientifically Sound**: Based on actual overfitting metric (train/val ratio), not arbitrary patience
2. **Adaptive**: Learning rate only drops when needed, not on fixed schedule
3. **Prevents Premature Stopping**: Won't stop just because val loss fluctuates
4. **Prevents Severe Overfitting**: Hard stop at 2.0x ratio guarantees reasonable generalization
5. **Efficient**: Saves computation by stopping at true convergence point
6. **Interpretable**: Logging shows exactly why each LR reduction and stop occurred

### Files Modified

- `src/train.py`:
  - Lines 497-531: Overfitting ratio calculation and LR scheduler setup
  - Lines 533-565: Training loop with overfitting detection
  - Lines 573-580: Final summary with overfitting ratio reporting

### Verification

✓ Overfitting ratio calculated and logged each epoch
✓ ReduceLROnPlateau scheduler properly integrated with Adam optimizer
✓ Early stopping triggers only when ratio exceeds 2.0x after min 20 epochs
✓ Plateau timeout (15 epochs) provides safety net for edge cases
✓ All logging messages clear and informative

---

## 13. References & Resources

### Competition
- [Deep Past Initiative](https://www.deeppast.org/)
- [Challenge on Kaggle](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)

### Related Work
- mBART-50: Multilingual BART (Facebook)
- mT5: Multilingual T5 (Google)
- Morphological NMT papers
- Low-resource translation techniques

### Tools & Frameworks
- Hugging Face Transformers
- Fairseq
- SacreBLEU
- SentencePiece

---

## 14. Next Steps

1. **Immediate**: Set up project structure and download data
2. **Week 1**: Complete EDA and preprocessing pipeline
3. **Week 2**: Implement and train baseline models
4. **Week 3**: Data augmentation and model improvements
5. **Week 4+**: Advanced optimization and ensemble

---

**Document Version**: 4.0  
**Last Updated**: February 4, 2026  
**Status**: Advanced Training with Overfitting-Based Early Stopping Implemented
