# Deep Past Challenge - Executive Summary

## Project Overview

**Goal**: Build a neural machine translation system to translate Old Assyrian (Akkadian) cuneiform tablets from 2000 BC into English.

**Competition**: Kaggle Deep Past Initiative - Machine Translation Challenge  
**Prize Pool**: $50,000 USD  
**Timeline**: December 16, 2025 - March 23, 2026 (10 weeks)  
**Evaluation**: Geometric mean of BLEU and chrF++ scores

---

## The Challenge

### What We're Solving
- **4,000-year-old business records**: Cuneiform tablets from ancient Assyrian merchants
- **~23,000 tablets exist**: Only ~11,500 translated; 11,500+ remain unread
- **Expert scarcity**: Only ~12 scholars worldwide can read Old Assyrian
- **Your impact**: Successfully translate just the training set would unlock 10,000+ tablets

### Why It's Hard (Unique Challenges)

1. **Extreme Low-Resource Scenario**
   - Only ~8,000 training examples (tiny for deep learning)
   - Standard NMT fails with this little data
   - Solution: Transfer learning + aggressive augmentation

2. **Morphologically Complex Language**
   - Single Akkadian word = multiple English words
   - Akkadian is 4,000 years old with ancient grammar rules
   - Solution: Morpheme-aware tokenization, character-level models

3. **Heavily Formatted/Noisy Text**
   - Ancient scribal marks, line numbers, comments
   - Special characters (š, ṭ, ḫ) with Unicode encoding issues
   - Determinatives (noun classifiers) in curly brackets
   - Gaps and breaks marking damaged parts of tablets
   - Solution: Comprehensive preprocessing pipeline

4. **Domain-Specific Vocabulary**
   - Ancient merchant/legal vocabulary
   - Proper nouns critical (names, place names)
   - No modern dictionaries available
   - Solution: Lexicon building, copy mechanisms

---

## Our Solution Architecture

### Three-Phase Approach

#### Phase 1: Data Foundation (Week 1-2)
```
Raw Data → Preprocessing → Clean Corpus → Vocabulary Building
```
- Implement robust preprocessing pipeline
- Handle all 20+ formatting challenges
- Build proper noun lexicon
- Create train/validation/test splits with data augmentation

#### Phase 2: Baseline Models (Week 2-3)
```
Clean Data → Seq2Seq + mBART/mT5 → Evaluation
```
- Seq2Seq with Attention: Quick baseline (~12-15 BLEU)
- mBART-50 (pre-trained): Strong baseline (~18-22 BLEU)
- mT5 (pre-trained): Alternative baseline
- Performance target: Beat random by 3-5x

#### Phase 3: Advanced Optimization (Week 3-6)
```
Models → Data Augmentation + Architecture Improvements + Ensemble → Final Submission
```
- Back-translation for data augmentation (2x training data)
- Morphology-aware encoders
- Multi-task learning (translation + proper noun recognition)
- Copy mechanism for rare words
- Ensemble multiple diverse models
- Performance target: Top 10% on leaderboard

### Expected Performance Progression

```
Baseline Seq2Seq:        BLEU ~15   → Geometric Mean ~21
Pre-trained mBART:       BLEU ~20   → Geometric Mean ~28
+ Augmentation:          BLEU ~23   → Geometric Mean ~31
+ Advanced Models:       BLEU ~25   → Geometric Mean ~33
+ Ensemble (Final):      BLEU ~27   → Geometric Mean ~35
```

---

## Project Structure

```
DeepPastChallengeTranslateAkkadianEnglish/
├── DESIGN_PLAN.md              ← Comprehensive 11-section design
├── PROJECT_KICKOFF.md          ← Setup instructions & quick reference
├── README.md                   ← Project overview & usage
│
├── data/
│   ├── raw/                    ← Download competition data here
│   ├── processed/              ← Cleaned training/validation data
│   ├── augmented/              ← Augmented data (back-translation, etc)
│   └── lexicons/               ← Vocabulary, proper nouns, mappings
│
├── src/
│   ├── preprocessing.py        ← Text cleaning & formatting (✓ DONE)
│   ├── tokenization.py         ← BPE, SentencePiece (TODO)
│   ├── models/
│   │   ├── baseline_seq2seq.py (TODO)
│   │   ├── transformer_models.py (TODO)
│   │   └── morphology_aware.py (TODO)
│   ├── training.py             ← Training loop (TODO)
│   ├── evaluation.py           ← BLEU/chrF++ metrics (TODO)
│   └── inference.py            ← Prediction pipeline (TODO)
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Data exploration (TODO)
│   ├── 02_preprocessing.ipynb   ← Preprocessing demo (TODO)
│   ├── 03_baseline_training.ipynb (TODO)
│   ├── 04_model_evaluation.ipynb (TODO)
│   └── 05_final_inference.ipynb (TODO)
│
├── configs/
│   ├── model_seq2seq.yaml      ← Baseline config (✓ DONE)
│   └── model_mbart50.yaml      ← Pre-trained config (✓ DONE)
│
├── checkpoints/                ← Saved model weights
├── results/                    ← Training logs, analysis
└── requirements.txt            ← Python dependencies (✓ DONE)
```

---

## Key Design Decisions

### 1. Preprocessing Strategy
- **Remove**: All scribal marks (!, ?, /, :, .)
- **Normalize**: Unicode characters (Ḫ→H, š→sz)
- **Standardize**: Gap markers (<gap>, <big_gap>)
- **Keep**: Determinatives in curly brackets (domain-specific info)
- **Result**: Clean, consistent corpus without information loss

### 2. Tokenization Approach
- Primary: **SentencePiece BPE** (handles subwords & special chars)
- Alternative: **Character + word hybrid** (morpheme awareness)
- Vocabulary size: 10,000 tokens
- Special tokens for: `<gap>`, `<big_gap>`, determinatives

### 3. Model Selection

| Model | Reason | Expected BLEU |
|-------|--------|---------------|
| **Seq2Seq + Attention** | Interpretable baseline | 12-15 |
| **mBART-50** | 50 languages, strong transfer | 18-22 |
| **mT5** | Multilingual T5, good alternative | 18-22 |
| **Morphology-Aware** | Custom for Akkadian complexity | 20-24 |
| **Ensemble** | Combines strengths | 25-28 |

### 4. Data Augmentation Strategy
- **Back-translation**: English→Akkadian→English (iterative)
- **Paraphrasing**: Rephrase English while keeping meaning
- **Synthetic data**: Generate from morphological patterns
- **Multi-lingual transfer**: Use related ancient languages
- **Target**: 2x training data with high quality

### 5. Evaluation Approach
- **Primary metric**: Geometric Mean of BLEU × chrF++
- **BLEU**: Word n-gram overlap (measures surface similarity)
- **chrF++**: Character n-grams (robust to morphology)
- **Secondary**: Proper noun accuracy, determinative accuracy

---

## Key Milestones & Timeline

| Week | Phase | Deliverables | Target Metrics |
|------|-------|--------------|-----------------|
| 1-2 | Data Prep | Clean corpus, EDA, vocabulary | N/A |
| 2-3 | Baselines | Seq2Seq, mBART, evaluation | BLEU ~15-20 |
| 3-4 | Augmentation | Back-translation, 2x data | BLEU +3-5 |
| 4-5 | Advanced | Morphology models, multi-task | BLEU +2-4 |
| 5 | Optimization | Hyperparameter tuning | BLEU +1-2 |
| 6 | Ensemble | Final predictions | BLEU +1-3 |
| 7 | Analysis | Error analysis, docs | Documentation |

---

## Technology Stack

### Core Libraries
```
PyTorch 2.0+          # Deep learning framework
Transformers 4.40+    # Pre-trained models (mBART, mT5)
Datasets 2.14+        # Efficient data loading
SacreBLEU 2.3+        # Official evaluation metrics
NumPy, Pandas, SciPy  # Data processing
```

### Models to Use
```
facebook/mbart-large-50      # Multilingual pre-trained
google/mt5-base              # Multilingual T5
Custom Seq2Seq               # Baseline architecture
```

### Development Tools
```
Jupyter Notebooks    # Exploration & analysis
Weights & Biases     # Experiment tracking (optional)
TensorBoard          # Training visualization
```

---

## Success Criteria

### Minimum (Top 50%)
- ✓ BLEU > 15 (better than random)
- ✓ System handles all 8,000 examples
- ✓ Proper nouns recognized

### Target (Top 10%)
- ✓ BLEU > 22
- ✓ Geometric mean > 30
- ✓ Proper nouns > 85% accuracy
- ✓ Clean error analysis

### Stretch (Top 1%)
- ✓ BLEU > 27
- ✓ Geometric mean > 35
- ✓ Production-ready code
- ✓ Comprehensive documentation

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Small dataset** | Overfitting | Transfer learning, regularization, augmentation |
| **Text complexity** | Preprocessing errors | Comprehensive test suite, manual verification |
| **Slow training** | Limited iterations | Efficient architectures, distilled models, caching |
| **Low baseline** | Hard to improve | Ensemble multiple approaches |
| **Rare words** | OOV issues | Morphological analysis, copy mechanism |
| **Time pressure** | Incomplete work | Weekly milestones, parallel implementation |

---

## Getting Started

### 1. Setup (30 minutes)
```bash
cd DeepPastChallengeTranslateAkkadianEnglish
pip install -r requirements.txt
```

### 2. Download Data (10 minutes)
```bash
# Configure Kaggle API, then:
kaggle competitions download -c deep-past-initiative-machine-translation -p data/raw/
```

### 3. First Steps
- [ ] Read [DESIGN_PLAN.md](DESIGN_PLAN.md) (detailed design)
- [ ] Read [PROJECT_KICKOFF.md](PROJECT_KICKOFF.md) (practical setup)
- [ ] Run preprocessing test: `python src/preprocessing.py`
- [ ] Run EDA notebook: `notebooks/01_eda.ipynb`

---

## Expected Outcomes

### By End of Project
1. **Trained Models**
   - ✓ Baseline Seq2Seq
   - ✓ Fine-tuned mBART-50
   - ✓ Advanced morphology-aware model
   - ✓ Ensemble of top models

2. **Submission Files**
   - ✓ `submission.csv` with 8,000 predictions
   - ✓ BLEU/chrF++ scores
   - ✓ Proper noun analysis

3. **Documentation**
   - ✓ Design document (DESIGN_PLAN.md)
   - ✓ Setup guide (PROJECT_KICKOFF.md)
   - ✓ Error analysis
   - ✓ Architecture explanation

### Impact
- Unlock thousands of ancient clay tablets
- Advance low-resource NMT research
- Contribute to AI for endangered languages
- Compete for $50,000 prize

---

## Next Steps

**Immediate Actions** (Today):
1. ✓ Review project structure
2. ✓ Read DESIGN_PLAN.md thoroughly
3. [ ] Set up Python environment
4. [ ] Configure Kaggle API
5. [ ] Download competition data
6. [ ] Run first EDA notebook

**This Week**:
- Complete Phase 1 (Data Preparation)
- Implement preprocessing pipeline
- Create vocabulary and lexicons
- Baseline EDA complete

**Next Week**:
- Train baseline Seq2Seq model
- Fine-tune mBART-50
- Achieve competitive BLEU score
- Begin data augmentation

---

## References

### Competition Resources
- [Deep Past Initiative](https://www.deeppast.org/)
- [Kaggle Competition](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/)
- [Official Discussion Forum](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion)

### Machine Translation
- "Attention Is All You Need" - Vaswani et al. (2017)
- "Massively Multilingual BART" - Liu et al. (2021)
- Low-resource NMT techniques

### Related Kaggle Competitions
- WMT Machine Translation Challenges
- Other Kaggle NLP competitions for reference

---

## Document Information

**Project**: Deep Past Challenge - Akkadian to English Translation  
**Status**: Project Initialized - Ready for Phase 1  
**Created**: February 2, 2026  
**Version**: 1.0  
**Last Updated**: February 2, 2026  

**Next Review**: After Phase 1 completion (1 week)

---

## Contact & Questions

Refer to documentation in order:
1. [DESIGN_PLAN.md](DESIGN_PLAN.md) - Detailed technical design
2. [PROJECT_KICKOFF.md](PROJECT_KICKOFF.md) - Practical setup guide
3. [README.md](README.md) - Project overview
4. Kaggle Competition Discussions

**Good luck! Let's bring Bronze Age voices back to life! 🏛️**
