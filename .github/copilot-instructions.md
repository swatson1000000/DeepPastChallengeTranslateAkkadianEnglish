# Project Guidelines

## Overview

Kaggle **code competition** for translating Old Assyrian (Akkadian) transliterations to English. Competition: `deep-past-initiative-machine-translation`. Very small corpus (1,562 training pairs) — data augmentation is critical.

## Environment

- **Conda env**: `phi4` — activate before any Python command
- **GPU**: NVIDIA with 75GB VRAM locally; Kaggle submission limited to 16GB GPU, ≤9 hours
- **Install**: `conda activate phi4 && pip install -r requirements.txt`

## Code Style

- Python 3, `snake_case` functions/variables, `PascalCase` classes
- Type hints on function signatures; `logging.getLogger(__name__)` for all logging
- Docstrings with `Args:` / `Returns:` sections — see `src/train.py` as reference
- `sys.stdout.reconfigure(line_buffering=True)` for unbuffered output in long-running scripts

## Architecture

**Primary model**: Custom LSTM Seq2Seq with Bahdanau attention, defined inline in `src/train.py` and `src/inference.py`.

Key components (all in `src/train.py`):
- `SimpleTokenizer` — word-level, special tokens: `<PAD>`=0, `<UNK>`=1, `<SOS>`=2, `<EOS>`=3
- `AttentionLayer` — Bahdanau attention
- `CopyMechanism` — pointer-generator (TIER 2)
- `LexiconConstrainedDecoder` — constrains output with lexicon entries (TIER 2)
- `LabelSmoothingCrossEntropy` — label smoothing loss

**Note**: Model classes are duplicated in `src/train.py` and `src/inference.py`. Keep both in sync when modifying architecture.

## Data Pipeline

```
data/raw/train.csv → src/full_preprocessing.py → data/processed/train_clean.csv
                   → src/augment_data.py       → data/processed/train_augmented_*.csv
                   → src/train.py (loads CSV, builds vocab on the fly, trains)
                   → src/inference.py           → predictions.csv → submission.csv
```

- **Raw columns**: `oare_id`, `transliteration`, `translation` (train); `id`, `transliteration` (test)
- **Submission format**: CSV with columns `id`, `translation` — file must be named `submission.csv`
- **Metric**: Geometric mean of BLEU and chrF++ (micro-averaged). Implemented in `src/evaluation.py`.

## Build and Run

```bash
# Training (default: 75 epochs, 3-fold CV)
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
nohup python src/train.py --model improved --epochs 75 > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Inference
nohup python src/inference.py --model improved --output predictions.csv > log/inference_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Kaggle notebook submission
cd jupyter && kaggle kernels push -p .
```

All scripts MUST run via `nohup` with timestamped logs in `log/`. See `CLAUDE.md` for full execution policy.

## Config Files

YAML configs in `configs/` control model hyperparameters. Train script reads them via `--config`:
- `model_seq2seq.yaml` — baseline (3-layer bidir LSTM, hidden=768)
- `model_seq2seq_tier3.yaml` — best config (4-layer, hidden=1024, beam=8)
- `model_mbart50.yaml` — HuggingFace mBART-50 fine-tuning (secondary approach)

CLI args (`--epochs`, `--batch-size`, `--folds`) override config values.

## Kaggle Submission

The notebook `jupyter/akkadian-english-seq2seq.ipynb` is the submission artifact:
- Reads data from `/kaggle/input/deep-past-initiative-machine-translation/`
- Writes `submission.csv` to `/kaggle/working/`
- GPU enabled, internet disabled, ≤9 hours runtime
- `jupyter/kernel-metadata.json` configures the Kaggle kernel push

## Key Conventions

- **Checkpoints**: Saved to `checkpoints/` — component-level (`attention_best.pt`, `rnn_best.pt`) and epoch-level (`seq2seq_epoch_N.pt`)
- **K-fold**: 3-fold CV by default (`--folds 3`). Fold checkpoints in `checkpoints/fold_N/`
- **Augmentation**: Up to 30x via lexicon back-translation & paraphrasing. Critical due to tiny dataset
- **Preprocessing**: Unicode normalization (Ḫ→H), determinative handling, scribal notation removal — see `src/preprocessing.py`
- **No test suite**: `pytest` is in requirements but no tests exist yet
