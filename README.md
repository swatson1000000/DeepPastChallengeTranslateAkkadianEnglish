# Deep Past Challenge - Akkadian to English Machine Translation

Bringing Bronze Age voices back to life through neural machine translation of Old Assyrian cuneiform tablets.

## Challenge Overview

This is a competitive machine translation project for the **Kaggle Deep Past Initiative Challenge**. The objective is to build neural machine translation models that convert transliterated Old Assyrian (Akkadian) cuneiform text into English.

### Key Facts
- **Dataset**: ~8,000 cuneiform texts from ancient Assyrian merchants
- **Timeline**: December 16, 2025 - March 23, 2026
- **Prize Pool**: $50,000 USD (1st: $15,000, 2nd: $10,000, 3rd: $8,000, 4th: $7,000, 5th-6th: $5,000 each)
- **Evaluation**: Geometric Mean of BLEU and chrF++ scores
- **Challenges**: Low-resource language, morphologically complex, heavily formatted texts

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (if CUDA available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Quick Start

```bash
# 1. Download competition data (requires Kaggle API credentials)
kaggle competitions download -c deep-past-initiative-machine-translation -p data/raw/

# 2. Run EDA
jupyter notebook notebooks/01_eda.ipynb

# 3. Preprocess data
python src/preprocessing.py

# 4. Train baseline model
python src/training.py --config configs/model_seq2seq.yaml
```

## Project Structure

See [DESIGN_PLAN.md](DESIGN_PLAN.md) for detailed project structure and implementation plan.

Quick overview:
```
├── data/              # Training and test data (download required)
├── src/               # Source code for models, training, inference
├── notebooks/         # Jupyter notebooks for exploration and analysis
├── configs/           # Configuration files (YAML)
├── checkpoints/       # Saved model checkpoints
├── results/           # Training logs and results
├── DESIGN_PLAN.md     # Detailed project design and timeline
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Key Challenges

### 1. Low-Resource Language
- Only ~8,000 training examples (very small for neural models)
- Standard deep learning architectures often fail
- **Strategy**: Transfer learning from pre-trained models (mBART, mT5)

### 2. Morphological Complexity
- Single Akkadian word often maps to multiple English words
- Complex inflectional and derivational morphology
- **Strategy**: Character-level and sub-word tokenization, morpheme-aware architectures

### 3. Complex Text Formatting
- Special characters: š, ṭ, ḫ, subscripts, superscripts
- Determinatives in curly brackets: {d}, {ki}, {lu₂}
- Scribal notations: gaps, line numbers, insertions
- **Strategy**: Comprehensive preprocessing and character normalization

### 4. Domain-Specific Vocabulary
- Ancient commercial and legal domain
- Proper nouns (place names, merchant names) critical for meaning
- **Strategy**: Lexicon-based processing, copy mechanism, multi-task learning

## Implementation Phases

### Phase 1: Data Preparation (Week 1-2)
- [ ] Download and explore raw data
- [ ] Analyze formatting and special characters
- [ ] Build preprocessing pipeline
- [ ] Create clean parallel corpus
- [ ] Build vocabulary and lexicons

### Phase 2: Baseline Models (Week 2-3)
- [ ] Implement Seq2Seq with attention
- [ ] Fine-tune mBART-50
- [ ] Fine-tune mT5
- [ ] Establish baseline BLEU/chrF++ scores

### Phase 3: Data Augmentation (Week 3-4)
- [ ] Back-translation
- [ ] Paraphrasing
- [ ] Synthetic data generation
- [ ] Multi-lingual transfer (if data available)

### Phase 4: Advanced Models (Week 4-5)
- [ ] Specialized tokenization (SentencePiece)
- [ ] Morphology-aware encoders
- [ ] Multi-task learning
- [ ] Copy mechanism for rare words

### Phase 5: Optimization (Week 5)
- [ ] Hyperparameter search
- [ ] Learning rate scheduling
- [ ] Beam search tuning
- [ ] Model ensembling

### Phase 6: Inference & Submission (Week 6)
- [ ] Generate predictions on test set
- [ ] Post-processing and validation
- [ ] Create submission file
- [ ] Final leaderboard submission

### Phase 7: Analysis (Ongoing)
- [ ] Error analysis
- [ ] Attention visualization
- [ ] Ablation studies
- [ ] Documentation

## Models to Implement

### Baseline Models
1. **Sequence-to-Sequence with Attention**
   - 2-3 layer GRU/LSTM encoder-decoder
   - Bahdanau/Luong attention
   - Baseline BLEU target: ~12-15

2. **Transformer (mBART/mT5)**
   - Fine-tune pre-trained multilingual models
   - Leverages knowledge from 50+ languages
   - Target BLEU: ~18-22

### Advanced Models
3. **Morphology-Aware Encoder**
   - Character-level + morpheme-level encoding
   - Hierarchical attention
   - Target BLEU: ~20-25

4. **Multi-Task Learning**
   - Main: Translation
   - Auxiliary: Proper noun recognition, determinative classification
   - Target improvement: +2-3 BLEU

5. **Ensemble**
   - Combine multiple diverse models
   - Weighted voting or averaging
   - Target BLEU: ~24-28

## Evaluation

### Metrics
- **BLEU**: Geometric mean score (macro-averaged)
- **chrF++**: Character n-gram score (macro-averaged)
- **Geometric Mean**: sqrt(BLEU × chrF++)

### Performance Targets
| Model | Target BLEU | Target chrF++ |
|-------|------------|---------------|
| Baseline Seq2Seq | 12-15 | 25-30 |
| mBART-50 | 18-22 | 35-40 |
| Advanced + Augmentation | 22-25 | 42-45 |
| Ensemble | 24-28 | 45-48 |

## Results Summary

Results will be updated as models are trained:

| Model | BLEU | chrF++ | Geometric Mean | Notes |
|-------|------|--------|-----------------|-------|
| Baseline Seq2Seq | TBD | TBD | TBD | Initial baseline |
| mBART-50 | TBD | TBD | TBD | Pre-trained multilingual |
| Advanced | TBD | TBD | TBD | Morphology-aware |
| Ensemble | TBD | TBD | TBD | Final submission |

## Configuration Examples

### Model Configuration (configs/model_seq2seq.yaml)
```yaml
model:
  encoder:
    type: lstm
    num_layers: 2
    hidden_size: 512
    embedding_size: 256
    dropout: 0.3
  decoder:
    type: lstm
    num_layers: 2
    hidden_size: 512
    embedding_size: 256
    dropout: 0.3
  attention: bahdanau

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  early_stopping_patience: 5
  validation_split: 0.1

inference:
  beam_width: 5
  length_penalty: 0.6
```

## Usage Examples

### Preprocess Data
```python
from src.preprocessing import AkkadianPreprocessor

preprocessor = AkkadianPreprocessor(
    remove_scribal_marks=True,
    normalize_unicode=True,
    handle_determinatives='normalize'
)

clean_text = preprocessor.preprocess(raw_akkadian_text)
```

### Train Model
```python
from src.training import Trainer
from src.models.transformer_models import MBARTTranslator

model = MBARTTranslator('facebook/mbart-large-50')
trainer = Trainer(model, config_path='configs/model_transformer.yaml')
trainer.train('data/processed/train.txt', 'data/processed/valid.txt')
```

### Generate Predictions
```python
from src.inference import Inference

inferencer = Inference(checkpoint_path='checkpoints/best_models/model.pt')
predictions = inferencer.predict('data/processed/test.txt')
```

## References

### Competition
- [Deep Past Initiative](https://www.deeppast.org/)
- [Kaggle Competition](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)

### Models & Papers
- mBART: Massively Multilingual Bart (Facebook AI)
- mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer (Google)
- Attention is All You Need (Vaswani et al., 2017)
- Neural Machine Translation by Attention (Bahdanau et al., 2015)

### Evaluation
- [SacreBLEU](https://github.com/mjpost/sacrebleu/) - Official BLEU/chrF++ implementation
- BLEU: Papineni et al. (2002)
- chrF: Popović (2015)

## Contributing

Contributions are welcome! Please:
1. Follow PEP 8 style guide
2. Add tests for new features
3. Update documentation
4. Submit pull requests to main branch

## License

This project is for educational and research purposes. See LICENSE file for details.

## Authors & Contact

**Project Lead**: [Your Name]  
**University/Organization**: [Institution]  
**Email**: [email@example.com]

## Acknowledgments

- Deep Past Initiative for organizing this competition
- Kaggle for hosting and providing platform
- All competitors and community contributors
- Museum experts who provided the cuneiform translations
