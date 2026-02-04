# Deep Past Challenge - Project Setup & Design Summary

## Quick Start Checklist

- [x] Reviewed Kaggle competition page
- [x] Created design plan document
- [x] Set up project structure
- [x] Created preprocessing module
- [x] Created configuration templates
- [ ] Download competition data (requires Kaggle API)
- [ ] Run exploratory data analysis
- [ ] Train baseline models
- [ ] Implement data augmentation
- [ ] Build advanced models
- [ ] Optimize and ensemble
- [ ] Generate final submission

## Project Setup Instructions

### 1. Environment Setup

```bash
# Navigate to project directory
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Kaggle API Setup

```bash
# Configure Kaggle credentials
# Place kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
# Copy your kaggle.json to ~/.kaggle/

# Download competition data
kaggle competitions download -c deep-past-initiative-machine-translation -p data/raw/

# Extract data
cd data/raw/
unzip -q '*.zip'
cd ../..
```

### 3. Verify Installation

```bash
# Test Python environment
python -c "import torch; import transformers; print('✓ Dependencies installed')"

# Test preprocessing module
python src/preprocessing.py
```

## Key Implementation Notes

### Data Format Challenges (from competition overview)

**Critical preprocessing requirements:**

1. **Scribal Notations to Remove**
   - `!` - certain reading mark
   - `?` - questionable reading mark
   - `/` - line divider
   - `:` or `.` - word dividers (but keep as spaces)
   - `< >` - scribal insertions (keep text, remove brackets)
   - `˹ ˺` - partially broken signs
   - `[ ]` - square brackets (remove from transliteration)

2. **Character Normalization**
   - `Ḫ/ḫ` → `H/h` (special H with cedilla)
   - `š` → `sz` (s with caron)
   - `ṣ/Ṣ` → `s,/S,` (s with dot below)
   - `ṭ/Ṭ` → `t,/T,` (t with dot below)
   - Subscripts (₀-₉) → superscripts (0-9)

3. **Gap Standardization**
   - `[x]` → `<gap>` (small break)
   - `[…]` or `[… …]` → `<big_gap>` (large break)

4. **Determinatives (Akkadian classifiers)**
   - Keep in curly brackets: `{d}`, `{ki}`, `{lu₂}`, etc.
   - Or remove depending on strategy
   - **17 types defined** - see DESIGN_PLAN.md for full list

### Preprocessing Module

The `src/preprocessing.py` module provides:

```python
from src.preprocessing import AkkadianPreprocessor

# Create preprocessor
processor = AkkadianPreprocessor(
    remove_scribal_marks=True,
    normalize_unicode=True,
    handle_determinatives='normalize',
    normalize_gaps=True,
    normalize_subscripts=True
)

# Process text
cleaned = processor.preprocess(raw_text)
```

### Model Configurations

Two configuration templates provided:

1. **Seq2Seq with Attention** (`configs/model_seq2seq.yaml`)
   - 2-layer LSTM encoder/decoder
   - Bahdanau attention
   - Learning rate: 5e-4
   - Target: 12-15 BLEU

2. **mBART-50** (`configs/model_mbart50.yaml`)
   - Pre-trained multilingual model
   - Learning rate: 5e-5
   - Target: 18-22 BLEU

## Next Steps

### Immediate (This Week)
1. Download and extract competition data
2. Explore data with EDA notebook
3. Implement full preprocessing pipeline
4. Create vocabulary and lexicon files

### Week 1-2
1. Implement baseline Seq2Seq model
2. Fine-tune mBART-50
3. Achieve baseline BLEU ~15

### Week 2-3
1. Implement data augmentation (back-translation)
2. Add morphology-aware components
3. Achieve improved BLEU ~20

### Week 3+
1. Optimize and ensemble models
2. Error analysis and refinement
3. Final submission preparation

## File Organization

```
├── data/
│   ├── raw/               # Download data here
│   ├── processed/         # Cleaned parallel corpus
│   ├── augmented/         # Augmented training data
│   └── lexicons/          # Vocabulary and proper nouns
├── src/
│   ├── preprocessing.py   # Data cleaning pipeline ✓
│   ├── tokenization.py    # Tokenization strategies (TODO)
│   ├── models/            # Model implementations (TODO)
│   ├── training.py        # Training loop (TODO)
│   ├── evaluation.py      # Metrics (TODO)
│   └── inference.py       # Prediction pipeline (TODO)
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory analysis (TODO)
│   ├── 02_preprocessing.ipynb
│   └── ...
├── configs/
│   ├── model_seq2seq.yaml ✓
│   └── model_mbart50.yaml ✓
├── DESIGN_PLAN.md         # Detailed design ✓
└── README.md              # Project overview ✓
```

## Key Performance Targets

| Milestone | BLEU | chrF++ | Geometric Mean | Timeline |
|-----------|------|--------|-----------------|----------|
| Baseline (Seq2Seq) | 12-15 | 25-30 | ~18-21 | Week 2 |
| mBART Fine-tune | 18-22 | 35-40 | ~26-29 | Week 2-3 |
| With Augmentation | 20-23 | 40-43 | ~29-32 | Week 3-4 |
| Advanced Models | 22-25 | 42-45 | ~31-34 | Week 4-5 |
| Ensemble (Final) | 24-28 | 45-48 | ~33-36 | Week 5-6 |

## Evaluation Metrics Explained

### BLEU (BiLingual Evaluation Understudy)
- Measures n-gram overlap between prediction and reference
- Range: 0-1 (higher is better)
- Good for surface-level word overlap
- Can miss semantic meaning

### chrF++ (Character F-score with plus)
- Character-level F-score metric
- More robust to morphological variations
- Good for morphologically complex languages like Akkadian
- Range: 0-1 (higher is better)

### Geometric Mean
- **Competition metric**: sqrt(BLEU × chrF++)
- Balances both metrics
- Prevents overfitting to one metric

## Important Considerations

### Low-Resource Challenge
- Only ~8,000 training examples
- Need **aggressive data augmentation**
- Back-translation, paraphrasing, synthetic data

### Morphological Complexity
- Akkadian words map to multiple English words
- **Character-level tokenization** essential
- Sub-word BPE/SentencePiece recommended

### Formatting Complexity
- Heavy preprocessing required
- Custom tokenization handling
- Determinative special handling

### Computational Resources
- GPU strongly recommended (NVIDIA with CUDA 11.8+)
- 16GB+ RAM needed for transformer models
- Consider gradient accumulation for smaller GPUs

## References & Resources

### Competition
- [Deep Past Initiative](https://www.deeppast.org/)
- [Kaggle Competition](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/)

### Models to Use
- **mBART-50**: facebook/mbart-large-50
- **mT5**: google/mt5-base
- **DistilBERT**: distilbert-base-multilingual-cased

### Evaluation Libraries
- **SacreBLEU**: Official BLEU/chrF++ implementation
- **Hugging Face Datasets**: Easy data handling
- **Transformers**: Pre-trained models

### Key Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Massively Multilingual Bart" (Liu et al., 2021)
- "BLEU: A Method for Automatic Evaluation" (Papineni et al., 2002)

## Known Issues & Solutions

| Issue | Solution |
|-------|----------|
| Kaggle API 403 error | Check API credentials, ensure kaggle.json in ~/.kaggle/ |
| CUDA out of memory | Reduce batch size, use gradient accumulation |
| Slow preprocessing | Use multiprocessing, vectorize operations |
| Low BLEU scores | Check preprocessing, ensure data quality, try augmentation |
| Overfitting | Add regularization, use dropout, early stopping |

## Contact & Questions

For questions about this project setup, refer to:
- [DESIGN_PLAN.md](DESIGN_PLAN.md) - Comprehensive design document
- [README.md](README.md) - Project overview
- Competition discussion forums on Kaggle

---

**Status**: Project initialized and ready for Phase 1 (Data Preparation)  
**Last Updated**: February 2, 2026  
**Next Action**: Download data and run EDA
