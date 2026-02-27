# Project Guidelines

## Overview

Kaggle **code competition** translating Old Assyrian (Akkadian) transliterations→English.
Competition: `deep-past-initiative-machine-translation`. Tiny corpus (1,561 document-level training pairs).
Current score: 2.7 (LSTM baseline). Target: 36+ (silver medal). Deadline: March 23, 2026.

**Critical constraint**: Training data is document-level, test data is sentence-level.

## Environment

- **Conda env**: `phi4` — activate before any Python command
- **GPU**: NVIDIA GB10 ~80GB VRAM locally; Kaggle submission limited to 16GB GPU, ≤9 hours
- **Install**: `conda activate phi4 && pip install -r requirements.txt`

## Code Style

- Python 3, `snake_case` functions/variables, `PascalCase` classes
- Type hints on function signatures; `logging.getLogger(__name__)` for all logging
- Docstrings with `Args:` / `Returns:` sections — see `src/train_byt5.py` as reference
- `sys.stdout.reconfigure(line_buffering=True)` at top of long-running scripts

## Architecture

**Primary model**: `google/byt5-small` (300M params) fine-tuned via HuggingFace Transformers.
Byte-level tokenizer — no vocab mismatch issues with Akkadian Unicode.

| File | Purpose |
|------|---------|
| `src/train_byt5.py` | Fine-tuning: AdamW, linear warmup, gradient checkpointing, BF16 support |
| `src/inference.py` | Beam search generation (beams=8, length_penalty=1.3) |
| `src/preprocess.py` | Transliteration/translation preprocessing + postprocessing |
| `src/evaluate.py` | Geometric mean of BLEU & chrF++ via sacrebleu |
| `src/gpu_utils.py` | GB10 GPU optimizations: SDPA, BF16 autocast, torch.compile |

Preprocessing is inlined in `jupyter/akkadian-byt5-submission.ipynb` for Kaggle (which cannot import `src/`). **Keep the notebook preprocessing in sync with `src/preprocess.py`.**

## Data Pipeline

```
data/raw/train.csv ──→ src/preprocess.py ──→ data/processed/train_clean.csv
                   ──→ src/train_byt5.py (preprocesses inline, fine-tunes ByT5)
                   ──→ models/byt5-akkadian/best/ (saved checkpoint)
                   ──→ src/inference.py ──→ submission.csv
```

- **Train columns**: `oare_id`, `transliteration`, `translation` (1,561 document-level pairs)
- **Test columns**: `id`, `text_id`, `line_start`, `line_end`, `transliteration` (sentence-level; dummy in download)
- **Submission**: CSV with `id`, `translation` — named `submission.csv`
- **Metric**: `sqrt(BLEU * chrF++)` micro-averaged. See `src/evaluate.py`.
- **Lexicon**: `data/raw/OA_Lexicon_eBL.csv` (39K entries with word type: word/PN/GN)
- **Alignment aid**: `data/raw/Sentences_Oare_FirstWord_LinNum.csv` (9,782 sentence boundaries)

## Build and Run

```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish

# Training (defaults: 25 epochs, batch=32, max_length=512, gradient checkpointing)
nohup python -u src/train_byt5.py --output-dir models/byt5-akkadian > log/train_byt5_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Training with GB10 optimizations (BF16 + fused kernels)
nohup python -u src/train_byt5.py --bf16 --compile --output-dir models/byt5-akkadian > log/train_byt5_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Inference
nohup python -u src/inference.py --model models/byt5-akkadian/best > log/inference_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Kaggle submission
cd jupyter && kaggle kernels push -p .
```

All scripts MUST run via `nohup` with timestamped logs in `log/`. See `CLAUDE.md` for full execution policy.

## Kaggle Submission

Notebook `jupyter/akkadian-byt5-submission.ipynb` is the submission artifact:
- Loads model from `/kaggle/input/byt5-akkadian-finetuned/best`
- `jupyter/kernel-metadata.json` configures kernel push (GPU, no internet)
- `dataset_sources` in metadata must reference the uploaded model dataset
- Writes `submission.csv` to `/kaggle/working/`

## Key Conventions

- **Checkpoints**: `models/byt5-akkadian/best/` (best val loss), `models/byt5-akkadian/epoch_N/`, `models/byt5-akkadian/final/`
- **No fp16**: Causes NaN errors with ByT5 — FP16 exponent range too small
- **No bf16 for ByT5**: Empirically costs ~8 points on Kaggle vs FP32. BF16's reduced mantissa (8 bits vs 23) degrades byte-level generation quality. Always train and infer in **FP32**.
- **torch.compile is safe for inference**: Use `--compile` for ~1.3–1.5x speedup on GB10 during inference. **Not compatible with HF Seq2SeqTrainer** — compiled model can't be pickled by DataLoader worker processes.
- **GB10 optimizations**: `--bf16 --compile` enables BF16 autocast + torch.compile fused kernels. See `src/gpu_utils.py`
- **Preprocessing**: H normalization (Ḫ→H), gap markers (`<gap>`/`<big_gap>`), accent→numbered forms, scribal notation removal. See `src/preprocess.py`
- **Determinatives**: Keep `{d}`, `{ki}`, `{m}` etc. as-is — they encode semantic meaning
- **DO NOT use EvaCun/ORACC data**: Neo-Assyrian (wrong era by 1,000 years)
- **Published test split**: 34% public / 66% private — expect LB shake
- **Plan**: `plan.md` tracks 5-phase roadmap with timeline and score targets
