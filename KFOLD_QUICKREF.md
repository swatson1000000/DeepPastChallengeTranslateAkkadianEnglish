# K-Fold Training - Quick Reference

## Overview
K-fold cross-validation is now integrated into `train.py`. Use it to train multiple independent models on different data splits, which helps break through overfitting plateaus.

## Basic Usage

### Regular 80/20 Training (Default)
```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
rm -f log/train_*.log
nohup python train.py --model tier3 --epochs 100 > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### K-Fold Training (5 Folds)
```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
rm -f log/train_*.log
nohup python train.py --model tier3 --folds 5 --epochs 100 \
  --data data/processed/train_clean.csv > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Command Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model variant | improved |
| `--folds` | Number of k-folds (None = 80/20 split) | None |
| `--epochs` | Max epochs per fold | From config |
| `--batch-size` | Batch size | 64 |
| `--data-path` | Training data path | data/processed/train_augmented.csv |
| `--use-copy` | Enable copy mechanism | False |
| `--use-lexicon` | Enable lexicon constraints | False |

## Examples

### 3-Fold Training (Faster)
```bash
nohup python train.py --folds 3 --epochs 100 --model tier3 \
  > log/train_3fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 10-Fold Training (More Thorough)
```bash
nohup python train.py --folds 10 --epochs 100 --model tier3 \
  > log/train_10fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### With Augmented Data
```bash
nohup python train.py --folds 5 --epochs 100 --model tier3 \
  --data data/processed/train_augmented_paraphrase_only.csv \
  > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Monitoring Training

### Real-time logs
```bash
tail -f log/train_kfold_*.log
```

### Check fold progress
```bash
grep "FOLD" log/train_kfold_*.log
grep "complete: Best val loss" log/train_kfold_*.log
```

### Process status
```bash
ps aux | grep "python.*train" | grep -v grep
```

## Output Structure

Each k-fold training creates:
```
checkpoints/
├── fold_0/
│   └── best_model.pt
├── fold_1/
│   └── best_model.pt
├── fold_2/
│   └── best_model.pt
├── fold_3/
│   └── best_model.pt
└── fold_4/
    └── best_model.pt
```

## Summary Report

At the end, the log shows:
```
================================================================================
K-FOLD CROSS-VALIDATION COMPLETE
================================================================================

Fold Results:
  Fold 1: Val Loss = 4.8532
  Fold 2: Val Loss = 4.6789
  Fold 3: Val Loss = 4.7234
  Fold 4: Val Loss = 4.5891
  Fold 5: Val Loss = 4.9012

Average Val Loss: 4.7492 ± 0.1219
Best Fold: 4 (Loss: 4.5891)
```

## Key Differences: K-Fold vs Regular Training

| Aspect | Regular (80/20) | K-Fold (5-fold) |
|--------|-----------------|-----------------|
| Training samples per split | 1,248 | 1,248 |
| Validation samples | 313 | 313 |
| Total models trained | 1 | 5 |
| Training time | ~1.5 hours | ~7-8 hours |
| Checkpoint count | 1 | 5 |
| Overfitting resistance | Lower | Much higher |
| Generalization | Good | Excellent |

## Expected Benefits

### Performance
- **Val Loss**: Should reach 4.5-4.8 (vs. 5.0+ in single model)
- **Ensemble**: Averaging k models typically reduces overfitting 15-30%
- **Stability**: Less sensitive to random seed variations

### Training Characteristics
- **Annealing triggers**: Will detect overfitting in each fold independently
- **Checkpoint size**: 5x larger (5 model sets)
- **Gentle approach**: Same overfitting detection (1.5x) and annealing in each fold

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
nohup python train.py --folds 5 --batch-size 32 --epochs 100 > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Wrong data producing out-of-vocab errors
Use clean data instead:
```bash
nohup python train.py --folds 5 --data data/processed/train_clean.csv > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Check if training is still running
```bash
ps aux | grep "python.*train" | grep -v grep
```

### Stop training
```bash
pkill -f "python.*train"
```

## Next Steps

After k-fold training completes:

1. **Review best fold**: Check which fold had lowest validation loss
2. **Use best model**: Copy best fold checkpoint for inference
3. **Ensemble inference**: Combine predictions from all 5 folds
4. **Evaluate**: Compare metrics with single-model baseline

## Integration Note

Previously separate scripts (`train_kfold.py`, `inference_ensemble.py`) have been consolidated into the main `train.py`. This provides:
- Single point of maintenance
- Consistent configuration across modes
- Unified logging and monitoring
- Easier version control

