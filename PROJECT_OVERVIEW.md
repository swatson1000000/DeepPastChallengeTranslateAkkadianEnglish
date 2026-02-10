# Project Overview

## Competition

**Deep Past Challenge — Translate Akkadian to English** is a Kaggle featured code competition ($50K prize pool) asking competitors to build neural machine translation models that convert transliterated Old Assyrian (Akkadian) cuneiform into English. Old Assyrian is a Bronze Age dialect of Akkadian — the oldest documented Semitic language — used on ~23,000 clay tablets recording merchant trade between Mesopotamia and Anatolia (~4,000 years ago). Only half have been translated, and fewer than a dozen scholars in the world can read the rest.

- **Competition slug:** `deep-past-initiative-machine-translation`
- **Metric:** Geometric mean of BLEU and chrF++ (micro-averaged across the corpus)
- **Submission:** Code competition — notebook must run end-to-end on Kaggle (GPU ≤16GB, ≤9 hours, no internet)
- **Deadline:** March 23, 2026

---

## Dataset

### Training Data (`data/raw/train.csv`)

1,562 parallel sentence pairs. Extremely small for NMT — data augmentation is critical.

| Column | Description |
|--------|-------------|
| `oare_id` | Unique identifier from the Old Assyrian Research Environment |
| `transliteration` | Akkadian text in Latin script transliteration (hyphen-separated syllables, determinatives in `{}`, Sumerian logograms in ALL CAPS) |
| `translation` | English translation |

**Example:**
```
transliteration: a-na A-šur-i-mi-tí qí-bi-ma um-ma Ku-ku-a-num-ma
translation:     Say to Aššur-imitī, thus (says) Kukuanum:
```

### Test Data (`data/raw/test.csv`)

| Column | Description |
|--------|-------------|
| `id` | Integer ID for each test sample |
| `transliteration` | Akkadian transliteration to translate |

### Submission Format

CSV with header: `id,translation` — file must be named `submission.csv`.

### Lexicon (`data/raw/OA_Lexicon_eBL.csv`)

~35K+ entries mapping Akkadian forms to English normalizations. Used for data augmentation (lexicon-based back-translation) and lexicon-constrained decoding.

### Supplementary Data

- `Sentences_Oare_FirstWord_LinNum.csv` — OARE sentence data with line numbers
- `eBL_Dictionary.csv` — electronic Babylonian Literature dictionary
- `bibliography.csv`, `publications.csv`, `published_texts.csv`, `resources.csv` — metadata

---

## Architecture

### Primary Model: LSTM Seq2Seq with Bahdanau Attention

A custom encoder-decoder architecture using PyTorch, defined inline in `src/train.py` (and duplicated in `src/inference.py` for loading).

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENCODER                                  │
│  Input tokens → Embedding → LSTM (2-4 layers, bidirectional)   │
│  Output: encoder_outputs (all hidden states), final (h, c)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    encoder_outputs
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ATTENTION (Bahdanau)                         │
│  query = decoder hidden state                                   │
│  keys = encoder_outputs                                         │
│  → attention weights → context vector                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                      context vector
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DECODER                                  │
│  Previous token → Embedding → LSTM                              │
│  Concat(hidden_state, context) → Linear → vocab logits          │
│                                                                 │
│  Optional TIER 2 additions:                                     │
│  ├── CopyMechanism: pointer-generator to copy source tokens     │
│  └── LexiconConstrainedDecoder: mask invalid tokens             │
└─────────────────────────────────────────────────────────────────┘
```

### Model Components (all in `src/train.py`)

#### SimpleTokenizer
Word-level tokenizer that splits on whitespace. Special tokens:
- `<PAD>` = 0 (padding)
- `<UNK>` = 1 (unknown/out-of-vocabulary)
- `<SOS>` = 2 (start of sequence)
- `<EOS>` = 3 (end of sequence)

Vocabulary is built on-the-fly from the training data (`build_vocab(texts, min_freq=1)`). Sequences are encoded to fixed length (`max_len=180`) with padding.

#### AttentionLayer (Bahdanau Additive Attention)
Computes attention weights between the decoder's current hidden state (query) and all encoder outputs (keys):
1. Projects query and keys through separate linear layers
2. Combines with `tanh` activation
3. Produces scalar score per encoder timestep via learned vector `v`
4. Softmax → attention weights → weighted sum of encoder outputs = context vector

#### LabelSmoothingCrossEntropy
Standard cross-entropy loss modified to prevent overconfident predictions:
- `smoothing=0.1`: distributes 10% of probability mass uniformly across the vocabulary
- `ignore_index=0`: ignores `<PAD>` tokens in loss computation
- Improves generalization, especially critical for this tiny dataset

#### CopyMechanism (TIER 2)
Pointer-generator network that allows the decoder to copy tokens directly from the source sequence — essential for proper nouns (`Aššur-imitī`, `Kukuanum`) and numbers that appear verbatim in translations:
- Learns a `copy_gate` (sigmoid) to decide between generating from vocabulary vs. copying from source
- Uses attention weights to build a copy distribution over source tokens
- Final distribution = `gate × vocab_dist + (1-gate) × copy_dist`

#### LexiconConstrainedDecoder (TIER 2)
Constrains the decoder's output distribution to only allow tokens that appear in the lexicon:
- Builds a binary mask of valid token IDs from the lexicon file
- Applies mask to logits before softmax, effectively blocking impossible tokens
- Special tokens (`<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`) are always allowed

### Secondary Model: mBART-50

HuggingFace `facebook/mbart-large-50` fine-tuning approach. Config at `configs/model_mbart50.yaml`. Uses pretrained multilingual model with transfer learning. Not the primary submission approach due to GPU memory constraints on Kaggle (16GB).

---

## Tiered Improvement System

The project was developed incrementally in tiers of increasing complexity:

### TIER 1 — Baseline Improvements
- Dropout (embedding: 0.1, LSTM: 0.3, decoder: 0.1)
- Weight decay (1e-4)
- Gradient clipping (max_norm=1.0)
- Label smoothing (0.1)
- Early stopping with overfitting detection
- Learning rate annealing on overfitting
- Model size reduction (3.5M → 1.2M parameters)

### TIER 2 — Advanced Decoding
- Copy mechanism (pointer-generator) for proper noun copying
- Lexicon-constrained decoding
- Coverage penalty to prevent repetition
- Copy probability threshold filtering

### TIER 3 — Full Pipeline
- Beam search decoding (width=8, length penalty=0.6)
- Larger model (4-layer LSTM, hidden=1024, embedding=512)
- Cosine LR schedule with warm restarts
- Back-translation data augmentation
- Aggressive augmentation (30x multiplication)
- Subword tokenization (BPE — optional via `tokenizers` library)

---

## Source Files — Detailed Reference

### Core Pipeline

#### `src/train.py` (1,027 lines) — Training Pipeline
The central training script. Contains all model class definitions inline (no shared module). Handles:
- Config loading from YAML files
- Vocabulary building from training data
- K-fold cross-validation (default: 3 folds)
- Mixed precision training (fp16) with automatic batch size adjustment
- Gradient accumulation (effective batch size = batch_size × accumulation_steps)
- Early stopping with overfitting detection (monitors val/train loss ratio)
- LR annealing: halves LR when overfitting detected, further reduces by 0.7 every 5 annealing epochs
- Component-level checkpoint saving (separate `.pt` files for embedding, RNN, attention, decoder)
- GPU memory monitoring and automatic cache clearing

**K-fold behavior:** Creates a fresh model for each fold, trains independently, saves the best fold's checkpoint as `{model_name}_best.pt`. Each fold saves intermediate checkpoints in `checkpoints/fold_N/`.

**Overfitting detection:** If `val_loss / train_loss > 1.15` after epoch 15, enters "annealing mode" — reduces LR and gives the model up to 8 more epochs to recover before early stopping.

#### `src/inference.py` (725 lines) — Inference Pipeline
Mirrors `train.py`'s model class definitions (must be kept in sync). The `Seq2SeqInference` class:
1. Loads component-level checkpoint files
2. Rebuilds tokenizer vocabulary from training data
3. Reads test CSV, encodes each transliteration
4. Decodes using greedy search or beam search
5. Writes `predictions.csv` with `id,translation` columns

Supports all the same model variants and features as training (copy mechanism, lexicon constraints, beam search).

#### `src/beam_search.py` (269 lines) — Beam Search Decoder
Two implementations:
- `BeamSearchDecoder`: Full-featured with coverage penalty, length normalization (`length_penalty=0.6`), and early stopping. Maintains beam of `(log_prob, tokens, hidden_state, cell_state)` tuples.
- `SimpleBeamSearch`: Lightweight alternative for simpler use cases.
- `beam_search_decode()`: Standalone function operating on logit histories.

### Data Processing

#### `src/preprocessing.py` (356 lines) — Text Preprocessing
Two preprocessor classes following the competition's Dataset Instructions:

**`AkkadianPreprocessor`** — normalizes Akkadian transliterations:
- **Unicode normalization:** `Ḫ` → `H`, `ḫ` → `h`, `š` → `sz`, `ṣ` → `s,`, `ṭ` → `t,` (test data uses ASCII only)
- **Scribal mark removal:** `!` (certain reading), `?` (uncertain), `/` (line divider), `:` and `.` (word dividers)
- **Determinative handling:** `{d}` (deity), `{ki}` (place), `{m}` (masculine name), etc. — can KEEP, NORMALIZE, or REMOVE
- **Gap standardization:** `[x]` → `<gap>`, `[… …]` → `<big_gap>`
- **Bracket removal:** `[KÙ.BABBAR]` → `KÙ.BABBAR` (removes scholarly brackets)
- **Subscript normalization:** `₀-₉` → `0-9`

**`EnglishPreprocessor`** — normalizes English translations:
- Removes scribal notations from translations
- Cleans bracket artifacts
- Normalizes whitespace

#### `src/full_preprocessing.py` (298 lines) — End-to-End Pipeline
Orchestrates the complete preprocessing workflow:
1. Loads raw `train.csv`
2. Instantiates both preprocessors
3. Applies preprocessing to all rows
4. Filters out empty/invalid pairs
5. Saves `data/processed/train_clean.csv`
6. Generates `data/processed/preprocessing_report.txt` with statistics

#### `src/data_loader.py` (326 lines) — Data Loading
`DataLoader` class wrapping pandas CSV reading. `ParallelPair` dataclass holds `(source, target, sample_id)` tuples. Provides methods for batched iteration, statistics, and train/test separation.

### Data Augmentation

#### `src/augment_data.py` (305 lines) — Standard Augmentation
`DataAugmentor` class implementing:
- **Lexicon back-translation:** Uses `OA_Lexicon_eBL.csv` to find Akkadian words in translations, substitute with alternative English glosses
- **Paraphrasing:** Rule-based substitutions for common phrases ("of silver" → "in silver", "said:" → "stated:", etc.)
- **Synthetic pair generation:** Combines segments from different training examples

Typical multiplier: 5x (1,562 → ~7,800 samples).

#### `src/aggressive_augment.py` (252 lines) — Aggressive Augmentation
`AggressiveAugmentor` class targeting 30x multiplication (~47K samples):
- **Token perturbation:** Random insertion, deletion, swap, dropout of Akkadian tokens
- **Morphological variation:** Suffix/prefix modifications
- **Punctuation/formatting variations:** Bracket style changes, whitespace modifications
- **Repetition with noise:** Duplicate samples with small random changes

#### `src/back_translate.py` (259 lines) — Model-Based Back-Translation
`BackTranslationGenerator` class that uses a trained model to generate synthetic Akkadian from English text:
1. Takes English text as input
2. Generates synthetic Akkadian transliteration using the reverse direction
3. Computes a confidence score for each synthetic pair
4. Filters out low-confidence pairs (threshold-based)
5. Saves high-quality synthetic pairs for training

### Evaluation

#### `src/evaluation.py` (359 lines) — Competition Metrics
`TranslationMetrics` class implementing the competition's scoring:
- **BLEU:** Uses `sacrebleu` library if available, falls back to inline implementation (up to 4-gram)
- **chrF++:** Character-level F-score with word n-grams (via `sacrebleu`)
- **Geometric mean:** `sqrt(BLEU × chrF++)` — the actual competition metric, micro-averaged
- `EvaluationReport` class generates formatted reports with per-sample and corpus-level scores

#### `src/evaluate_predictions.py` (174 lines) — Prediction Quality Analysis
Standalone script that analyzes `predictions.csv` for common issues:
- Token repetition detection
- Unbalanced parentheses/quotes
- Excessive ellipsis usage
- Article repetition ("the the", "a a")
- Empty or very short translations

### Vocabulary & Lexicon

#### `src/vocabulary.py` (359 lines) — Vocabulary Builder
`VocabularyBuilder` class that creates various vocabulary artifacts:
- **Proper noun extraction:** Identifies capitalized words appearing in both source and target
- **Character-level vocabularies:** Separate for Akkadian and English → `data/lexicons/char_vocab_*.json`
- **Word-level vocabularies:** With minimum frequency filtering
- **BPE tokenizer:** Uses HuggingFace `tokenizers` library (optional; provides subword segmentation)
- **Sumerian logogram list:** Extracts ALL CAPS words from Akkadian text → `data/lexicons/sumerian_logograms.json`

### Utilities

#### `src/gpu_optimizations.py` (101 lines) — GPU Config Reference
Static configuration reference for the 75GB GPU setup. Contains recommended batch sizes, hidden dimensions, and memory optimization settings. Not directly imported by other modules — serves as a reference document.

#### `src/eda.py` (343 lines) — Exploratory Data Analysis
`AkkadianEDA` class that generates a comprehensive text report analyzing:
- Text length distributions (source vs. target)
- Word count statistics
- Special character frequencies
- Determinative usage patterns
- Proper noun counts
- Gap/break frequencies
- Abbreviation patterns in translations

Output: `data/processed/eda_report.txt`

#### `src/tier2_improvements.py` (234 lines) — TIER 2 Module
Standalone definitions of `CopyMechanism`, `LexiconConstrainedDecoder`, and `TIER2Decoder`. These classes are also duplicated inline in `train.py` and `inference.py`. This file exists as a clean reference implementation.

#### `src/QUICK_START.py` (172 lines) — Usage Guide
Documentation-only script containing a formatted ASCII usage guide. Running it prints training examples, model comparisons, common workflows, and troubleshooting tips.

---

## Configuration System

YAML configs in `configs/` control model hyperparameters. The train script selects a config based on `--model` or loads a custom one via `--config`. CLI args (`--epochs`, `--batch-size`, `--folds`) override config values.

### `configs/model_seq2seq.yaml` — Baseline
```
Encoder: 3-layer bidirectional LSTM, hidden=768, embedding=384
Decoder: 3-layer LSTM, hidden=768, embedding=384
Training: batch=64, lr=0.0005, 200 epochs, gradient_accumulation=2, mixed precision
Inference: beam_width=5, length_penalty=0.6
```

### `configs/model_seq2seq_improved.yaml` — TIER 1
```
Encoder: 2-layer bidirectional LSTM, hidden=512, embedding=384
Decoder: 1-layer LSTM, hidden=512, embedding=384
Training: batch=64, lr=0.0005, 50 epochs, gradient_accumulation=8 (effective batch=512)
Scheduler: Cosine annealing with warm restarts (T_0=50, T_mult=2)
Model size: ~1.2M params (65% reduction from baseline)
```

### `configs/model_seq2seq_tier2.yaml` — TIER 2
```
Encoder: 2-layer unidirectional LSTM, hidden=512, embedding=384
Copy mechanism: enabled, coverage_penalty=0.1
Lexicon constraints: enabled
Training: batch=64, 300 epochs, gradient_accumulation=2, patience=50
Decoding: greedy default, beam_width=5, temperature=0.8
```

### `configs/model_seq2seq_tier3.yaml` — TIER 3 (Best)
```
Encoder: 4-layer unidirectional LSTM, hidden=1024, embedding=512
Beam search: width=8, length_penalty=0.6, coverage_penalty=0.1, min_length=5
Training: batch=64, lr=0.001, 400 epochs, weight_decay=1e-4, patience=8
Regularization: label_smoothing=0.1, embedding_dropout=0.1, variational dropout
Augmentation: synthetic + back-translation (confidence threshold=0.8)
```

### `configs/model_mbart50.yaml` — mBART-50
```
Pretrained: facebook/mbart-large-50
Training: batch=96, lr=2e-5, 25 epochs, fp16, gradient checkpointing
Inference: num_beams=8
```

---

## Checkpoint System

### Component-Level Checkpoints (`checkpoints/`)

During training, model components are saved separately:
- `embedding_best.pt` — source embedding weights
- `rnn_best.pt` — encoder LSTM weights
- `decoder_best.pt` — decoder LSTM weights (if separate from encoder)
- `attention_best.pt` — attention layer weights
- `optimizer_best.pt` — optimizer state (for training resumption)

### Epoch Checkpoints
Periodic full-model snapshots: `seq2seq_epoch_10.pt`, `seq2seq_epoch_20.pt`, etc.

### K-Fold Checkpoints
Each fold saves its best model to `checkpoints/fold_N/best_model.pt`. After all folds complete, the best fold's checkpoint is copied to the root as `{model}_best.pt`.

### Inference Loading
`src/inference.py` detects checkpoint format automatically:
- If component-level files exist (`embedding_best.pt`, etc.), loads each separately
- If a single checkpoint file exists, loads all components from it
- Rebuilds tokenizer vocabulary from training data (vocabulary is not saved in checkpoints)

---

## Data Flow Diagram

```
data/raw/train.csv (1,562 pairs)
    │
    ▼
src/full_preprocessing.py
    │  Unicode normalization, scribal mark removal,
    │  determinative handling, gap standardization
    ▼
data/processed/train_clean.csv
    │
    ├──▶ src/augment_data.py ──────▶ data/processed/train_augmented.csv (~5x)
    │
    └──▶ src/aggressive_augment.py ▶ data/processed/train_augmented_aggressive_30x.csv (~30x)
            │
            ▼
    src/train.py
        │  Builds SimpleTokenizer vocabulary on-the-fly
        │  3-fold cross-validation
        │  Mixed precision, gradient accumulation
        │  Early stopping with overfitting detection
        ▼
    checkpoints/fold_N/best_model.pt
    checkpoints/{model}_best.pt
        │
        ▼
    src/inference.py
        │  Loads checkpoints + rebuilds tokenizer
        │  Greedy or beam search decoding
        ▼
    predictions.csv (id, translation)
        │
        ▼
    submission.csv (copied/renamed for Kaggle)
```

---

## Kaggle Submission

### Notebook (`jupyter/akkadian-english-seq2seq.ipynb`)

Self-contained notebook that trains and generates predictions entirely on Kaggle hardware. Contains its own copies of all model classes, tokenizer, augmentation logic, and beam search — independent of the `src/` scripts.

**Key specs:**
- Reads from `/kaggle/input/deep-past-initiative-machine-translation/`
- Writes `submission.csv` to `/kaggle/working/`
- EMBEDDING_DIM=256, HIDDEN_DIM=512, NUM_LAYERS=2
- 50 epochs, 3-fold CV, beam_width=5
- 6.25x augmentation (smaller than local due to time constraints)
- GPU budget: 16GB, <9 hours

### Kernel Metadata (`jupyter/kernel-metadata.json`)

Configures the Kaggle API push:
```json
{
  "id": "stevewatson999/akkadian-english-seq2seq",
  "enable_gpu": true,
  "enable_internet": false,
  "competition_sources": ["deep-past-initiative-machine-translation"]
}
```

### Submission Workflow
```bash
cd jupyter && kaggle kernels push -p .
kaggle kernels status stevewatson999/akkadian-english-seq2seq-with-attention
# Then go to Kaggle → kernel output → click Submit on submission.csv
```

---

## Key Challenges

1. **Tiny corpus (1,562 pairs):** Standard NMT needs millions of pairs. Heavy augmentation is required but risks injecting noise.
2. **Morphological complexity:** Akkadian words encode what takes multiple English words ("a-ḫi-a-am" → "each other"). Word-level tokenization loses morphological information.
3. **Proper nouns:** Names like `Aššur-imitī` must be preserved exactly. The copy mechanism addresses this.
4. **Scribal notations:** Training and test data contain scholarly annotations (`!`, `?`, `[...]`, `{d}`) that must be normalized consistently.
5. **Unicode inconsistency:** Training data uses `Ḫ ḫ` but test data uses plain `H h`. Preprocessing normalizes this.
6. **Kaggle GPU constraint:** Local training uses 75GB GPU; Kaggle submission limited to 16GB. The notebook uses a smaller model to fit.

---

## Directory Structure

```
├── .github/
│   └── copilot-instructions.md    # AI coding agent instructions
├── archive/                        # Archived old scripts
│   ├── old_inference_scripts/
│   ├── old_training_scripts/
│   └── old_utilities/
├── checkpoints/                    # Training checkpoints (.pt files)
│   ├── fold_0/, fold_1/, fold_2/  # K-fold checkpoints
│   └── *.pt                       # Component and epoch checkpoints
├── configs/                        # YAML model configurations
├── data/
│   ├── raw/                       # Original Kaggle data + lexicon
│   ├── processed/                 # Preprocessed and augmented CSVs
│   ├── lexicons/                  # Extracted vocabularies (JSON)
│   └── augmented/                 # (unused)
├── jupyter/
│   ├── akkadian-english-seq2seq.ipynb   # Kaggle submission notebook
│   └── kernel-metadata.json             # Kaggle API config
├── log/                            # Timestamped execution logs
├── models/                         # Final saved models
├── results/                        # Evaluation outputs
├── src/                            # All source code
│   ├── train.py                   # Training pipeline (1,027 lines)
│   ├── inference.py               # Inference pipeline (725 lines)
│   ├── preprocessing.py           # Text preprocessing (356 lines)
│   ├── full_preprocessing.py      # End-to-end preprocessing (298 lines)
│   ├── data_loader.py             # Data loading (326 lines)
│   ├── augment_data.py            # Standard augmentation (305 lines)
│   ├── aggressive_augment.py      # 30x augmentation (252 lines)
│   ├── beam_search.py             # Beam search decoder (269 lines)
│   ├── evaluation.py              # Metrics: BLEU, chrF++ (359 lines)
│   ├── vocabulary.py              # Vocab/lexicon builder (359 lines)
│   ├── back_translate.py          # Back-translation augmentation (259 lines)
│   ├── tier2_improvements.py      # Copy + lexicon modules (234 lines)
│   ├── evaluate_predictions.py    # Prediction quality analysis (174 lines)
│   ├── eda.py                     # Exploratory data analysis (343 lines)
│   ├── gpu_optimizations.py       # GPU config reference (101 lines)
│   └── QUICK_START.py             # Usage guide (172 lines)
├── CLAUDE.md                       # Execution policy
└── EXECUTION.md                    # Execution commands reference
```
