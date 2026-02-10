# Execution Guide

All commands assume you are in the project root directory with the `phi4` conda environment active.

```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
```

---

## 1. Data Preprocessing

Converts raw training data into cleaned, normalized text suitable for model training.

```bash
nohup python src/full_preprocessing.py > log/preprocess_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Input:** `data/raw/train.csv`
**Output:** `data/processed/train_clean.csv`
**What it does:** Unicode normalization (Ḫ→H), determinative handling, scribal notation removal, gap standardization.

---

## 2. Data Augmentation

Expands the 1,562-sample training set. Choose one:

```bash
# Standard augmentation (~5x)
nohup python src/augment_data.py > log/augment_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Aggressive augmentation (~30x, ~47K samples)
nohup python src/aggressive_augment.py > log/augment_aggressive_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Input:** `data/processed/train_clean.csv` + `data/raw/OA_Lexicon_eBL.csv`
**Output:** `data/processed/train_augmented.csv` (or `train_augmented_aggressive_30x.csv`)
**Strategies:** Lexicon back-translation, paraphrasing, token perturbation, morphological variations.

---

## 3. Training

Trains the Seq2Seq model with 3-fold cross-validation.

```bash
# Default: improved model, 75 epochs, 3-fold CV
nohup python src/train.py --model improved --epochs 75 > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# With TIER 2 features (copy mechanism + lexicon constraints)
nohup python src/train.py --model tier2 --use-copy --use-lexicon --epochs 75 > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# With custom config
nohup python src/train.py --config configs/model_seq2seq_tier3.yaml --epochs 75 > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Specify augmented data file
nohup python src/train.py --model improved --data-path data/processed/train_augmented_aggressive_30x.csv > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**CLI arguments:**
| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `improved` | Model variant: `baseline`, `improved`, `tier2`, `tier3` |
| `--epochs` | 75 | Number of training epochs |
| `--batch-size` | from config | Batch size (auto-adjusted for GPU memory) |
| `--folds` | 3 | Number of K-fold CV splits |
| `--use-copy` | off | Enable pointer-generator copy mechanism |
| `--use-lexicon` | off | Enable lexicon-constrained decoding |
| `--data-path` | `data/processed/train_augmented.csv` | Training data CSV |
| `--config` | auto | YAML config file path |

**Output:** Checkpoints in `checkpoints/` (component-level and epoch-level `.pt` files).

---

## 4. Inference

Generates translations for the test set.

```bash
# Standard inference
nohup python src/inference.py --model improved --output predictions.csv > log/inference_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# With beam search
nohup python src/inference.py --model improved --use-beam-search --beam-width 5 --output predictions.csv > log/inference_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**CLI arguments:**
| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `improved` | Model variant to load |
| `--checkpoint` | auto | Override checkpoint path |
| `--test-data` | `data/raw/test.csv` | Test data CSV |
| `--output` | `predictions.csv` | Output predictions file |
| `--use-beam-search` | off | Enable beam search decoding |
| `--beam-width` | 5 | Beam search width |
| `--use-copy` | off | Enable copy mechanism |
| `--use-lexicon` | off | Enable lexicon constraints |
| `--max-samples` | all | Limit number of test samples |

**Output:** `predictions.csv` with columns `id`, `translation`.

---

## 5. Evaluation

Score predictions against reference translations.

```bash
nohup python src/evaluation.py > log/eval_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup python src/evaluate_predictions.py > log/eval_pred_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Metrics:** BLEU, chrF++, geometric mean (competition scoring metric).

---

## 6. Kaggle Submission

The notebook at `jupyter/akkadian-english-seq2seq.ipynb` is the submission artifact. It trains and runs inference end-to-end on Kaggle hardware.

```bash
# Push notebook to Kaggle
cd jupyter && kaggle kernels push -p .

# Check kernel status
kaggle kernels status stevewatson999/akkadian-english-seq2seq-with-attention
```

After the kernel finishes running, go to the kernel's Output tab on Kaggle and click **Submit** on `submission.csv`.

**Kaggle constraints:** GPU ≤16GB, ≤9 hours runtime, no internet access.

---

## 7. Monitoring

```bash
# Watch training logs live
tail -f log/train_*.log

# Check running Python processes
ps aux | grep python

# Check GPU utilization
nvidia-smi

# List background jobs
jobs -l
```

---

## 8. Cleanup Before Re-training

Always clean old logs before starting a new training run to avoid confusion.

```bash
rm -f log/train_*.log
```

---

## 9. Full Pipeline (End-to-End)

```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish

# Step 1: Preprocess
nohup python src/full_preprocessing.py > log/preprocess_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Wait for completion, then:

# Step 2: Augment
nohup python src/augment_data.py > log/augment_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Wait for completion, then:

# Step 3: Train
rm -f log/train_*.log
nohup python src/train.py --model improved --epochs 75 > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Wait for completion, then:

# Step 4: Inference
nohup python src/inference.py --model improved --use-beam-search --output predictions.csv > log/inference_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Step 5: Evaluate
nohup python src/evaluation.py > log/eval_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
