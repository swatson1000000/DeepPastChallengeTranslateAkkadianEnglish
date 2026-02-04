# Consolidated Training & Inference Scripts

## Overview

The project now uses **two main scripts** instead of multiple scattered training and inference files:

- **`train.py`** - Unified training for all model variants
- **`inference.py`** - Unified inference for all model variants

All previous training/inference scripts have been archived in `archive/` for reference.

## Quick Start

### Training

```bash
# Train baseline Seq2Seq model
python train.py --model baseline --epochs 100

# Train TIER 1 improved model
python train.py --model improved --epochs 200

# Train TIER 2 model with copy mechanism and lexicon constraints
python train.py --model tier2 --epochs 300 --use-copy --use-lexicon
```

### Inference

```bash
# Generate predictions with improved model
python inference.py --model improved --output predictions.csv

# Use TIER 2 features for inference
python inference.py --model tier2 --use-copy --output tier2_predictions.csv

# Process only first 100 samples
python inference.py --model improved --max-samples 100
```

## Configuration

Model training uses YAML configs:
- `configs/model_seq2seq.yaml` - Baseline settings
- `configs/model_seq2seq_improved.yaml` - TIER 1 improvements
- `configs/model_seq2seq_tier2.yaml` - TIER 2 improvements

All configs can be overridden via command-line arguments.

## Model Variants

### Baseline
- Standard Seq2Seq with LSTM encoder
- Bahdanau attention
- Basic greedy decoding

### Improved (TIER 1)
- Optimized architecture (2 layers, 512 hidden)
- Reduced embedding dimension (384)
- Temperature scaling for diversity
- Better data augmentation

### TIER 2
- Copy mechanism for proper nouns and numbers
- Lexicon-constrained decoding
- Coverage tracking to prevent repetition
- Extended training (300 epochs)

## Features

### Training Features
- Automatic GPU detection and optimization
- Early stopping with patience
- Learning rate scheduling (cosine annealing + plateau reduction)
- Gradient clipping
- Data validation and preprocessing
- Checkpoint saving

### Inference Features
- Batch and single-sample inference
- Attention visualization support
- Copy mechanism integration
- Token repetition prevention
- Configurable temperature for diversity

## Checkpoints

Models are saved to `checkpoints/` with variant-specific names:
- `checkpoints/baseline_best.pt`
- `checkpoints/improved_best.pt`
- `checkpoints/tier2_best.pt`

Each checkpoint includes:
- Embedding layer
- RNN weights
- Attention mechanism
- Decoder
- Copy mechanism (TIER 2 only)
- Lexicon constraints (TIER 2 only)

## Output

Predictions are saved as CSV with format:
```
id,transliteration,translation
0,akkadian_text,english_translation
1,akkadian_text,english_translation
...
```

## Archived Files

Old scripts are preserved in:
- `archive/old_training_scripts/` - Previous train variants
- `archive/old_inference_scripts/` - Previous inference variants
- `archive/old_utilities/` - Helper scripts and configs

## Implementation Details

### Tokenizers
- Custom word-level tokenizer built from training data
- Vocabulary sizes: ~4,700+ tokens per language
- Special tokens: PAD, UNK, SOS, EOS

### Model Architecture
- Variable LSTM layers and hidden dimensions per config
- Attention: Bahdanau mechanism
- Decoder: Linear layer to vocabulary

### TIER 2 Components
- **CopyMechanism**: Pointer-generator network with coverage tracking
- **LexiconConstrainedDecoder**: Token masking for valid vocabulary
- **Lexicon Loading**: 35K+ Akkadian-English word pairs

## Performance Notes

- TIER 1 improvements reduce redundant predictions
- TIER 2 adds copy mechanism for better handling of proper nouns
- Temperature scaling helps maintain diversity
- Lexicon constraints prevent gibberish generation

## Command Line Reference

### train.py
```
--model {baseline, improved, tier2}
--epochs N                    # Override epochs from config
--batch-size N               # Override batch size
--use-copy                   # Enable copy mechanism
--use-lexicon               # Enable lexicon constraints
--data-path PATH            # Training data location
--config PATH               # Custom config file
```

### inference.py
```
--model {baseline, improved, tier2}
--checkpoint PATH           # Custom checkpoint path
--test-data PATH            # Test data location
--output PATH               # Output predictions file
--use-copy                  # Enable copy mechanism
--max-samples N             # Limit number of samples
```

## Troubleshooting

If inference fails to load a checkpoint:
1. Verify checkpoint exists: `ls -la checkpoints/`
2. Check model variant matches checkpoint: `--model tier2` requires `tier2_best.pt`
3. Ensure training data is available for tokenizer building
4. Check GPU memory: TIER 2 models require more memory

If training diverges:
1. Reduce learning rate in config
2. Increase batch size for stability
3. Enable gradient clipping (enabled by default)
4. Reduce model complexity (baseline config)
