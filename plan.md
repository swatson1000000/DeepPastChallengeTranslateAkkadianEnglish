# Deep Past Challenge — ByT5 Improvement Plan

**Competition**: [Deep Past Initiative: Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)
**Deadline**: March 23, 2026 (Entry/Merger: March 16)
**Metric**: `sqrt(BLEU × chrF++)` micro-averaged (sentence-level)
**Prizes**: $50,000 total

---

## Current State

| Item | Value |
|------|-------|
| Kaggle LB score | **26.0** (3-seed ensemble: seeds 42/123/777) |
| Public baseline (Toda DPC Starter) | **34.9** (pre-rescore) — same architecture |
| Gap to close | ~9 points |
| Target | 36+ (silver medal zone) |
| Architecture | `google/byt5-small` (300M params), HF `Seq2SeqTrainer` |
| Training config | batch=1, grad_accum=8, Adafactor lr=1e-4, label_smoothing=0.2, bidirectional, FP32, 50 epochs |
| Submission notebook | `jupyter/akkadian-byt5-submission.ipynb` — version 40 pushed |
| Training script | `src/train_matched.py` |
| Ensemble script | `scripts/train_matched_ensemble.sh` |

### Seed 777 Val Metrics (the only properly-trained seed, with generation_max_length=512)

| Epoch | BLEU | chrF++ | GeoMean | Saved? |
|-------|------|--------|---------|--------|
| 38 | 30.66 | 53.97 | 40.68 | ✅ BEST (val_loss) |
| 39 | 31.54 | 54.17 | 41.34 | ❌ |
| 42 | 32.46 | 54.59 | 42.09 | ✅ BEST (val_loss) |
| 43 | 32.23 | 55.34 | 42.23 | ❌ |
| 44 | 32.87 | 55.93 | 42.87 | ❌ |
| 49 | 33.25 | 55.83 | 43.08 | ❌ |
| **50** | **33.26** | **56.22** | **43.24** | ❌ |

Seeds 42 and 123 had `generation_max_length` unset → ByT5 default of 20 bytes → BLEU=0.00, GeoMean=0.00 throughout all 50 epochs. Their checkpoints were selected purely on val_loss with no GeoMean signal.

---

## Ranked Improvements

---

### #1 — Retrain seeds 42 & 123 (`generation_max_length` was broken) — **+3–6 pts estimated**

**Evidence**: Seeds 42/123 ran 50 epochs each with `generation_max_length` unset. ByT5's `config.max_length=20` caused all val generations to be truncated to 20 bytes, making BLEU=0.000000 and GeoMean≈0.003 throughout. Seed 777 with the fix achieved GeoMean=43.24 at epoch 50. The current ensemble averages a properly-trained seed 777 with two broken seeds — the broken seeds are almost certainly dragging down ensemble quality.

**Fix**: Already in `train_matched.py` since tag `ByT5_27.2`. Just retrain.

**Action**:
```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
rm -f log/train_*.log
nohup bash scripts/train_matched_ensemble.sh \
    > log/train_matched_ensemble_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Note: seed 777 does NOT need retraining — its checkpoint is valid. The ensemble script retrains all 3 seeds. Consider modifying the script to skip seed 777 and reuse `models/byt5-matched-seed777/best`.

---

### #2 — Save checkpoint on GeoMean, not val_loss — **+0.5–1.5 pts estimated**

**Evidence**: From seed 777 logs, the checkpoint saved at epoch 42 (GeoMean=42.09) missed epochs 43–50 which had GeoMean 42.09–43.24. The val_loss and GeoMean peak at consistently different epochs. Switching to GeoMean selection means each seed's best checkpoint directly optimises the competition metric.

**Fix in `src/train_matched.py`**:
```python
# Change in Seq2SeqTrainingArguments:
metric_for_best_model="eval_geo_mean",
greater_is_better=True,
```
`compute_metrics` already returns `"geo_mean"` → HF exposes it as `"eval_geo_mean"`. No other code changes needed.

Also update `EpochLoggingCallback` to track `best_geo_mean` instead of `best_val_loss` for the `★ BEST` marker.

**Must be applied before retraining seeds 42/123** (fix #1).

---

### #3 — Fix epoch time format in `EpochLoggingCallback` — **cosmetic, 0 pts**

**Requirement** (CLAUDE.md): log format must be `time=Xm XXs`, e.g. `8m22s`.
**Current**: `time={elapsed:.0f}s` (raw seconds, e.g. `time=598s`).

**Fix in `EpochLoggingCallback.on_evaluate`**:
```python
mins, secs = divmod(int(elapsed), 60)
time_str = f"{mins}m{secs:02d}s"
# then use time_str in the log line
```

Apply together with fix #2.

---

> **Testing items #1, #2, #3** — Implemented and training launched **2026-02-27**. Seeds 42 and 123 retraining with `generation_max_length=512` (fix #1), `metric_for_best_model="eval_geo_mean"` (fix #2), and `time=Xm XXs` log format (fix #3). Seed 777 reused from existing best checkpoint. Ensemble will be rebuilt on completion.

---

### #4 — Train longer (70–100 epochs) — **+0.5–2 pts estimated**

**Evidence**: Seed 777 val GeoMean was still rising at epoch 50 (+1.15 from ep 42→50 with essentially flat val_loss 2.020–2.026). No sign of overfitting on GeoMean. Training longer costs only time; the GeoMean checkpoint saves the best seen so far so there is no downside.

**Action**: In `scripts/train_matched_ensemble.sh`, change `EPOCHS=50` → `EPOCHS=100`.
Also consider running seed 777 for another 50 epochs starting from its current best.

---

### #5 — Unidirectional training ablation — **+0 to +5 pts, high uncertainty**

**Hypothesis** (from plan analysis and community discussion): Toda's public 34.9 baseline almost certainly trains only Akkadian→English (1,562 pairs). We train bidirectionally (3,122 pairs) — half of all gradient updates teach English→Akkadian, a direction never used at inference. This could cost 5+ points by splitting model capacity.

**Action**: Train one seed with `--no-bidirectional` and compare val GeoMean against the equivalent bidirectional seed. If unidirectional scores ≥1 point higher on val, drop bidirectional from the full ensemble retrain.

```bash
nohup python -u src/train_matched.py \
    --seed 42 \
    --no-bidirectional \
    --output-dir models/byt5-unidirectional-seed42 \
    > log/train_unidirectional_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Do not retrain the full ensemble on this change until the ablation confirms it helps.**

---

### #6 — Inference parameter tuning (num_beams sweep) — **+0–1 pt estimated**

**Current**: `num_beams=4`, `length_penalty=1.0` (set in v40 push).
**Hypothesis**: `num_beams=8` with an optimised `length_penalty` may improve GeoMean; Toda's baseline may use higher beams.

**Action**: After retraining, run a local val sweep over `num_beams ∈ {4, 6, 8}` and `length_penalty ∈ {1.0, 1.3, 1.5}`. Takes ~30 min. Only update the notebook if GeoMean improves.

Note: with `no_repeat_ngram_size=10` and `repetition_penalty=2.0` already in place (v40), repetition loops should be suppressed — length_penalty can safely be raised again if needed.

---

### #7 — MBR decoding — **+0–1 pt estimated, 3–4× slower**

Generate 15–20 diverse candidates per input (temperature=0.7 sampling + beam search), score each against all others with chrF++, select the consensus winner. Reduces hallucination and improves robustness. Within Kaggle's 9hr limit for ~4,000 test sentences.

**Status**: Untested. Only one competitor (Hikari_30) was publicly evaluating this. Implement only after other improvements are exhausted.

---

### #8 — Larger model: byt5-base (580M params) — **+0–3 pts, high risk**

Higher capacity could improve translation quality but risks overfitting on the tiny 1,561-document dataset. Only viable with stronger regularization (`label_smoothing=0.3+`, `weight_decay=0.1+`, possibly dropout). Kaggle inference time must still complete within 9hrs.

**Status**: Do not attempt until the 3-seed byt5-small ensemble is near its ceiling.

---

### #9 — Extract training data from publications.csv — **+1–4 pts, very high effort**

~880 OCR'd scholarly PDFs explicitly provided by the competition for this purpose. Top competitors (MPWARE, Jack) are exploiting this. OCR quality is poor; translations are in English/French/German/Turkish. Requires significant preprocessing and alignment work.

**Status**: Multi-week effort. Only worth attempting if leaderboard score stalls below 33 after all other fixes are applied.

---

### #10 — Onomasticon name normalization — **~0 pts**

Community-verified: Prayag Patel (286th) tested on Kaggle LB and got exactly +0.0. The model already outputs English names correctly. Already implemented in `src/names.py` and inlined in the submission notebook. No further work needed.

---

## Immediate Action Sequence

```
1. Apply fix #2 (GeoMean checkpoint) + fix #3 (time format) to src/train_matched.py
2. Retrain seeds 42 & 123 (fix #1) — ~2h each
3. Rebuild ensemble from seeds 42/123 (retrained) + 777 (existing best)
4. Upload ensemble + push notebook → submit
5. In parallel: run unidirectional ablation (fix #5) to test hypothesis
6. If unidirectional wins: plan a full unidirectional ensemble for the next run
7. If score < 30: investigate longer training (fix #4) and beam sweep (fix #6)
```

---

## Key Constraints & Warnings

- **NEVER use FP16** — NaN errors with ByT5
- **NEVER use BF16** — BF16's 8-bit mantissa costs ~8 Kaggle points vs FP32; always train and infer in FP32
- **DO NOT use EvaCun/ORACC data** — Neo-Assyrian (wrong era by ~1,000 years)
- **DO NOT preprocess transliterations** — creates train/test mismatch; raw text is mandatory
- **Training = document-level, test = sentence-level** — must handle this granularity mismatch
- **Public LB = 34% of test data only** — expect significant private LB shake (~60% probability)
- **`torch.compile` is safe for inference** but incompatible with HF Seq2SeqTrainer (DataLoader pickling)
- **Determinatives `{d}`, `{ki}`, `{m}` etc.** — keep as-is, they encode semantic meaning
- **`save_total_limit=2`** in training args — only the 2 most recent checkpoints are kept; with GeoMean selection this is fine since best is always tracked separately by HF Trainer

---

## File Reference

| File | Purpose |
|------|---------|
| `src/train_matched.py` | HF Seq2SeqTrainer fine-tuning (primary training script) |
| `src/ensemble.py` | Weighted parameter averaging for model merging |
| `src/inference.py` | Beam search generation with configurable params |
| `src/evaluate.py` | Geometric mean of BLEU & chrF++ via sacrebleu |
| `src/preprocess.py` | Transliteration preprocessing (NOT used in current training — raw text only) |
| `src/names.py` | Onomasticon-based name normalization (post-processing) |
| `scripts/train_matched_ensemble.sh` | 3-seed training pipeline + ensemble merge |
| `scripts/ensemble_and_push.sh` | Watcher: merges when seed 777 done, uploads, pushes notebook |
| `jupyter/akkadian-byt5-submission.ipynb` | Kaggle submission notebook |
| `jupyter/kernel-metadata.json` | Kernel config (GPU, no internet, dataset sources) |
| `data/raw/train.csv` | 1,561 document-level training pairs |
| `data/raw/test.csv` | Sentence-level test data (~4,000 sentences from ~400 docs) |
| `models/byt5-matched-seed777/best/` | Valid trained checkpoint (gen_max_len fix applied) |
| `models/byt5-matched-ensemble/` | Current submitted ensemble (seeds 42+123 broken) |

---

## Score Tracker

| Model | Config | Kaggle LB |
|-------|--------|-----------|
| v2 (aligned-v2) | preprocessed, AdamW | 13.9 |
| v3 (sent-only) | preprocessed, sentence-only | 13.9 |
| HF ensemble (3x batch=128) | large batch, wrong LR | 18.2 |
| baseline single seed 42 | raw, bidi, Adafactor, 30ep | 21.3 |
| **matched-ensemble 3x (seeds 42/123/777)** | **gen_max_len bug in 42+123, LP=2.0** | **26.0** |
| matched-ensemble + rep_pen | same + repetition_penalty=1.3 | pending (v19) |
| matched-ensemble + inference fixes | LP=1.0, no_repeat=10, rep_pen=2.0 | pending (v40) |
| *public baseline (Toda DPC Starter)* | *3-seed ensemble, same architecture* | *34.9 (pre-rescore)* |
