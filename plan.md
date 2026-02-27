# Deep Past Challenge — ByT5 Competition Plan

**Competition**: [Deep Past Initiative: Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)  
**Goal**: Maximize geometric mean of BLEU and chrF++ on Akkadian→English translation.  
**Deadline**: March 23, 2026 (Entry/Merger deadline: March 16)  
**Constraints**: Kaggle code competition — GPU notebook ≤9hrs, internet OFF, `submission.csv`  
**Prizes**: $50,000 total ($15K / $10K / $8K / $7K / $5K / $5K)  
**Current score**: ~23 GeoMean on sentence-level val (v2 model, Feb 14)  
**Target**: 36+ (silver medal zone)  
**Participants**: 1,876 teams / 8,619 entrants

---

## Competition Data Overview

**CRITICAL: Training is DOCUMENT-level, test is SENTENCE-level.**  
The model must learn from full-document pairs but predict individual sentences.

| File | Records | Description |
|------|---------|-------------|
| `train.csv` | ~1,500 | Document-level transliteration↔translation pairs (`oare_id`, `transliteration`, `translation`) |
| `test.csv` | ~4,000 sentences from ~400 docs | Sentence-level (`id`, `text_id`, `line_start`, `line_end`, `transliteration`). Dummy data in download — replaced at scoring. |
| `published_texts.csv` | ~8,000 | Transliterations only (no translations), with OARE/CDLI IDs and metadata |
| `publications.csv` | ~880 PDFs | OCR'd scholarly publications — may contain translations in English/French/German/Turkish |
| `OA_Lexicon_eBL.csv` | ~39,000 | Word-level lexicon: form, norm, lexeme, type (word/PN/GN), eBL links |
| `eBL_Dictionary.csv` | — | Complete Akkadian dictionary from eBL |
| `Sentences_Oare_FirstWord_LinNum.csv` | 9,782 | Sentence-level alignment aid: first word + line numbers for `train.csv` docs |
| `bibliography.csv` | — | Bibliographic data for `publications.csv` |
| `resources.csv` | — | List of additional resources |

**Submission format**: CSV with `id,translation` — one sentence per row, file named `submission.csv`

---

## Phase 1: Baseline ByT5 (Score target: ~34) ✅ CODE COMPLETE — NEEDS TRAINING & TESTING

### 1.1 Download & upload pretrained ByT5-small
- Download `google/byt5-small` weights locally
- Upload as Kaggle Dataset for offline use in submission notebook
- Verify loading works without internet

### 1.2 Minimal fine-tuning pipeline
- Load `train.csv` (~1,500 document-level pairs)
- Prefix input: `"translate Akkadian to English: " + transliteration`
- Fine-tune ByT5 with basic settings:
  - **Epochs**: 10–15
  - **Max length**: 512 (NOT 256 — each step from 385→475→512 gives +0.2–0.3)
  - **Batch size**: 4
  - **Gradient accumulation**: 4 (effective batch = 16)
  - **Learning rate**: 5e-5
  - **FP32 ONLY**: Never use FP16 (NaN errors) or BF16 (insufficient mantissa precision for ByT5 byte-level tokenizer — causes ~8 point Kaggle score drop vs FP32)
  - **Optimizer**: AdamW with weight decay 0.01
- Save checkpoint, upload to Kaggle as Dataset

### 1.3 Inference notebook
- Load fine-tuned model from `/kaggle/input/`
- Beam search: `num_beams=8, max_new_tokens=512, length_penalty=1.3, early_stopping=True`
- Write `submission.csv`

**Expected score**: ~33–34 with zero preprocessing effort

---

## Phase 2: Preprocessing & Formatting (Score target: ~35–36)

The #2 ranked player says: *"If your formatting is off, especially on translations, nothing else you do will matter since your model isn't matching the desired output format."*

### 2.1 Transliteration preprocessing (input)
Apply to both training transliterations AND test transliterations.

**Per official Dataset Instructions:**

```
# Gap handling (official recommendation: standardize to exactly two markers)
[x]         → <gap>       (single broken sign)
…           → <big_gap>   (large break)
[… …]       → <big_gap>   (large break in brackets)
xx+         → <gap>       (unreadable sign)

# H normalization — training has Ḫ/ḫ but test has ONLY H/h
Ḫ → H
ḫ → h

# Accent characters → numbered forms (official table)
á → a2, à → a3, é → e2, è → e3
í → i2, ì → i3, ú → u2, ù → u3

# Subscript numbers → regular
₀₁₂₃₄₅₆₇₈₉ → 0123456789
ₓ → x

# Special characters (official table)
š ↔ sz (U+0161), ṣ (U+1E63), ṭ (U+1E6D), ʾ (U+02BE)

# Remove modern scribal notations (official list)
!  (certain reading)      → remove
?  (questionable reading)  → remove
/  (line divider)          → remove
:  (word divider)          → remove
.  (word divider)          → remove (careful: keep decimal points in numbers)
˹ ˺ (half brackets)        → remove from transliteration
[ ] (square brackets)      → remove brackets, keep text inside

# Scribal insertions (official)
< >  → remove brackets, keep text inside
<< >> → remove entirely (erroneous signs)

# Determinatives — KEEP as-is in curly brackets (they carry semantic meaning)
# {d} = deity, {ki} = place, {m}/{mi} = masc/fem name, {lu₂} = profession, etc.
# e.g. A-mur-{d}UTU, a-lim{ki}
```

**Capitalization rules** (preserve — they encode meaning):
- First letter capitalized → proper noun (person/place name)
- ALL CAPS → Sumerian logogram (e.g. KÙ.BABBAR = silver)

### 2.2 Translation preprocessing (output/training targets)
Apply to training translations:

```
# Same gap normalization as transliteration
# Same H normalization
# Remove scribal annotations: (fem), (plur), (pl), (sing), (?), (!)
# Fix fractions: 0.5 → ½, 0.25 → ¼, 0.75 → ¾
# Remove forbidden characters: !?()"—–<>⌈⌋⌊[]+ʾ/;
# Remove word repetitions: "the the" → "the"
```

### 2.3 Post-processing (inference output)
Apply to model predictions before writing submission.csv:

```python
def postprocess(text):
    # H normalization
    text = text.replace('ḫ', 'h').replace('Ḫ', 'H')
    # Subscript → regular numbers
    # Normalize gaps in output
    # Remove scribal annotations
    # Fix fractions
    # Remove forbidden characters (protect <gap>/<big_gap> markers)
    # Remove word repetitions
    # Collapse whitespace
    return text
```

**Expected score**: +1–2 points from preprocessing alone

---

## Phase 3: Sentence Alignment (Score target: ~36–37)

### 3.1 The problem
~50% of `train.csv` pairs are misaligned — the transliteration covers more or fewer lines than the translation. This is the single biggest data quality issue.

### 3.2 Use Sentences_Oare_FirstWord_LinNum.csv
- This file has 9,782 sentence-level alignments
- Match against `train.csv` using `first_word` and line numbers
- Split document-level pairs into properly aligned sentence pairs
- One competitor reported +3 points from fixing alignment alone

### 3.3 Manual review pass
- Jack (#2): *"I really started to gain in score when I manually reviewed each and every document in train.csv"*
- Use LLM to flag suspicious pairs (very long transliteration → very short translation)
- Remove or fix the ~163 known incomplete translations (pattern: long Akkadian → "1 talent … …")

### 3.4 Extract additional training data from publications.csv
The competition provides ~880 OCR'd scholarly PDFs and explicitly suggests this workflow:
1. **Locate each text and its translation** — match document IDs/aliases/museum numbers between `published_texts.csv` and `publications.csv`
2. **Convert all translations to English** — source may be in French, German, or Turkish
3. **Create sentence-level alignments** — split transliteration + translation into sentence pairs

This is the officially-suggested path to more training data.

### 3.5 published_texts.csv for self-training
- ~8,000 transliterations available (no translations)
- All confirmed Old Assyrian — safe to use (unlike EvaCun/ORACC)
- Can use for self-training: translate with current model, filter high-confidence pairs
- Has metadata: `genre_label`, `description`, `note`, `interlinear_commentary`

**Expected score**: +1–3 points from clean aligned data

---

## Phase 4: Ensemble & Advanced Techniques (Score target: 37+)

### 4.1 Weight-averaged ensemble
- Train 3 ByT5 checkpoints with different:
  - Random seeds
  - Training data splits/orderings
  - Slight hyperparameter variations
- Merge via weighted parameter averaging (not inference-time ensemble)
- Single merged model runs within 9hr time limit

### 4.2 Bucket batching
- Group similar-length inputs together to minimize padding waste
- Reduces padding by 20–40%
- Reported +0.1 improvement

### 4.3 MBR decoding (Minimum Bayes Risk)
- Generate 15 diverse candidates (temperature=0.7) + 5 beam search candidates
- Score each against all others using chrF++
- Select consensus winner
- Reduces hallucination, improves robustness
- ~3–4x slower but may improve private LB

### 4.4 Lexicon-guided name handling
- Use `OA_Lexicon_eBL.csv` (39K entries) — has `type` field: `PN` (person), `GN` (geography)
- Use `eBL_Dictionary.csv` — complete Akkadian dictionary from electronic Babylonian Library
- Cross-reference proper nouns against lexicon `form`→`norm`→`lexeme` mappings
- Post-process: ensure proper nouns match normalized publication forms

### 4.5 Onomasticon-based name normalization (NEW — from discussion #668485)
- **Source**: `data/raw/onomasticon/onomasticon.csv` (6,335 names, 5,973 spelling→name mappings)
  - Downloaded from `deeppast/old-assyrian-grammars-and-other-resources`
  - 984 names have multiple spelling variants
  - 90 names have aliases (alternative canonical forms)
- **Combined with OA_Lexicon_eBL.csv**: 13,424 PN form→norm + 334 GN form→norm
- **Coverage**: 2,212 lexicon PN forms found in training transliterations
- **Strategy**: Build `src/names.py` with:
  1. Spelling→canonical name lookup from onomasticon + lexicon
  2. Given transliteration input, identify name tokens (capitalized / preceded by `{m}`, `{d}`, etc.)
  3. In model output, fuzzy-match name-like tokens against canonical forms
  4. Replace model's name rendering with canonical spelling
- **Also in dataset**:
  - `secondary_sources.csv` (1,346 OCR'd scholarly texts) — potential future data augmentation
  - `eSAD/` — 23 Akkadian dictionary PDFs by letter
  - Kouwenberg 2019 OA grammar PDF — reference material
- **Expected score**: +1–2 points from correct name rendering
- **Community report**: Prayag Patel (286th) tested onomasticon name replacement and saw 0.0 gain on public LB — "Model already outputs English names correctly". However, this was tested on the public 3-model ensemble baseline, not on a custom-trained model. Our model may benefit differently since it's trained on aligned data where names may be less memorized.

### 4.6 eBL_Dictionary.csv analysis (NEW — Feb 2026)

**eBL Dictionary**: 19,215 entries, 15,344 with definitions, 10,780 with extractable English glosses.

**Cross-reference with OA Lexicon**:
- OA Lexicon has 25,574 word entries across 1,751 unique lexemes
- 807 eBL dictionary entries match OA lexemes → gives us Akkadian→English glossary
- 11,034 OA word forms can be linked to English translations via lexeme→eBL chain

**Key trade vocabulary mappings verified against training data**:

| Logogram | Occ. | Akkadian lexeme | English |
|----------|------|-----------------|---------|
| KÙ.BABBAR | 3391 | kaspu | silver |
| DUMU | 1932 | mer'u | son |
| IGI | 1128 | maḫar / šēbu | before / witness |
| URUDU | 578 | warī'um | copper |
| AN.NA | 490 | annuku | tin |
| DINGIR | 345 | ilu | god |
| SIG₅ | 325 | damiqtu | fine quality |
| DU₁₀ | 247 | ṭābu | good |
| DAM | 211 | aššutu | wife |
| ITU.KAM | 200 | warḫu | month |
| GAL | 138 | rabiu | chief/great |
| DIRI | 88 | watāru | excess |
| ILLAT | 61 | ellutu | caravan |
| DUG | 55 | karputu | vessel/pot |
| LUGAL | 28 | šarru | king |
| DUB | 14 | ṭuppu | tablet |

**Top English words in training translations**: silver (3643), shekels (1621), minas (1537), son (1966), seal (790), textiles (681), tablet (672), copper (618).

**Assessment**: The eBL dictionary is most useful as a **training data augmentation signal** (lexicon-guided tags) or for **post-processing verification**, not direct word-by-word translation. The model learns these mappings through context during fine-tuning.

**Potential use**: Add Sumerian logogram→English hints as prefix tags during training:
```
translate Akkadian to English [KÙ.BABBAR=silver] [DUMU=son]: ...transliteration...
```
This idea from the "Helper signals from lexicon" discussion thread. Could help with rare logograms the model sees infrequently.

### 4.7 Competition discussion insights (Feb 2026 survey)

**Leaderboard state** (as of Feb 13, 2026):
- Public plateau at 35.1 — hundreds of competitors using same 3-model ensemble (Assia Benkedia's weighted parameter average)
- Top tier (36.5–38.1): Likely custom training on private extracted data
- To break 35.5+: need custom training data, novel techniques, or different ensemble components
- hongan (#23): "Single model reaching 34.5 with just formatting/preprocessing. Formatting will get you to ~36-37."
- Jack (#4): "95% is preprocessing/formatting" — but means CAREFUL MANUAL REVIEW of every training document
- MPWARE (#17): Extracting translations from Larsen PDFs (difficult — OCR quality bad)
- FML (372nd): Published manually extracted translations from Larsen/AKT volumes
- Tomorin (485th): Sentence alignment → +3 points (28.4 → 31.4), bidirectional training → +1 (→ 32.4), ensemble → +0.6 (→ 33.0)

**Key validated findings**:
1. **Sentence alignment = biggest single gain** (+3 points per Tomorin). We already have this ✅
2. **Formatting/preprocessing gets you to ~36-37** per hongan (#23). We have good preprocessing ✅
3. **Onomasticon replacement = 0.0 gain** on public LB per Prayag (tested on Assia ensemble). May differ for our model.
4. **MBR decoding**: Only Hikari_30 testing publicly. 3-4x slower, could improve private LB robustness.
5. **Private LB shake**: ~60% probability (Aaron Bornstein). Top models are ~40% pattern matchers. Conservative models may rank higher on private LB.
6. **Top models hallucinate**: Generate generic trade sentences from corpus templates. Reducing hallucination (MBR, better data) matters for private LB.
7. **Published data**: FML sharing manually extracted translations from AKT volumes — potential additional training data.

**Failed approaches to avoid**:
- AKK-300m model (outputs only `<big_gap>`, wrong dialect)
- BetterTransformer (deprecated in Kaggle environment)
- Submission blending of the same ensemble (0.0 gain)
- Hyperparameter tuning via Optuna (minimal impact, ~0.4 max)

### 4.8 Ensemble strategy (CONCRETE PLAN)

**Approach**: Weighted parameter averaging of 3 ByT5-small checkpoints trained with different seeds.

**Why this works**: The public 34.9 baseline is literally this technique (3 models merged). Each model learns slightly different representations due to random initialization of the training loop (data shuffling, dropout). Averaging smooths out noise and reduces variance.

**Training plan** (after v2 completes):

| Model | Seed | Start from | LR | Epochs | Output dir |
|-------|------|------------|-----|--------|-----------|
| v2 (current) | 42 | v1 best | 3e-5 | 25 | models/byt5-akkadian-aligned-v2/ |
| v3 | 123 | google/byt5-small | 5e-5 | 50 | models/byt5-akkadian-seed123/ |
| v4 | 777 | google/byt5-small | 5e-5 | 50 | models/byt5-akkadian-seed777/ |

**Commands** (run sequentially, each ~12 hrs):
```bash
# v3 — seed 123
nohup python -u src/train_byt5.py \
    --seed 123 --epochs 50 --lr 5e-5 --batch-size 32 \
    --output-dir models/byt5-akkadian-seed123 \
    > log/train_byt5_seed123_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# v4 — seed 777
nohup python -u src/train_byt5.py \
    --seed 777 --epochs 50 --lr 5e-5 --batch-size 32 \
    --output-dir models/byt5-akkadian-seed777 \
    > log/train_byt5_seed777_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Merge all 3
python src/ensemble.py \
    --models models/byt5-akkadian-aligned-v2/best \
            models/byt5-akkadian-seed123/best \
            models/byt5-akkadian-seed777/best \
    --weights 0.4 0.3 0.3 \
    --output models/byt5-ensemble/
```

**Timeline**:
- v2 finishes: ~Feb 14 morning → eval + upload
- v3 starts: Feb 14 → finishes ~Feb 15
- v4 starts: Feb 15 → finishes ~Feb 16
- Ensemble merge + eval: Feb 16
- Upload + submit: Feb 16-17
- Time for iteration: Feb 17-March 23

**Weight rationale**: v2 gets 0.4 weight because it continued from v1 (more total training). v3/v4 get 0.3 each. Can tune weights based on individual val scores.

---

## Phase 5: Kaggle Submission Pipeline

### 5.1 Local training workflow
```bash
# Train on local GPU (128GB VRAM)
conda activate phi4
python src/train_byt5.py --epochs 15 --max-length 512

# Upload checkpoint to Kaggle
kaggle datasets create -p models/byt5-akkadian/
```

### 5.2 Submission notebook structure
```
jupyter/
  akkadian-byt5-submission.ipynb   # The submission notebook
  kernel-metadata.json             # Kaggle kernel config (GPU, no internet)
```

Notebook flow:
1. Load model from `/kaggle/input/byt5-akkadian-finetuned/`
2. Load & preprocess test.csv
3. Bucket-batched inference with beam search
4. Post-process outputs
5. **Name normalization** (inline `NameNormalizer` from `src/names.py`)
6. Write `submission.csv` to `/kaggle/working/`

### 5.4 Notebook update checklist (after retraining)
When deploying a new model + name normalization to Kaggle:
1. **Inline `NameNormalizer`** — copy class from `src/names.py` into a notebook cell (notebook can't import `src/`)
2. **Bundle `onomasticon.csv`** into the model dataset upload (`stevewatson999/byt5-akkadian-finetuned`), or upload as a separate dataset and add to `dataset_sources` in `kernel-metadata.json`
3. **`OA_Lexicon_eBL.csv`** — already available on Kaggle at `/kaggle/input/deep-past-initiative-machine-translation/` (no upload needed)
4. **Add normalization step** in notebook after translation generation: call `normalize_names(transliteration, prediction)` on each output
5. **Keep inline preprocessing in sync** with `src/preprocess.py`

### 5.3 Push to Kaggle
```bash
cd jupyter && kaggle kernels push -p .
```

---

## Key Warnings

- **DO NOT use EvaCun/ORACC data** — Neo-Assyrian (911–539 BCE), 1,000 years wrong for Old Assyrian
- **DO NOT use fp16** — causes NaN errors with ByT5
- **DO NOT use bf16** — BF16 has only 8 bits of mantissa vs 23 in FP32. For ByT5's byte-level tokenizer (256+ vocab), this reduced precision degrades generation quality and costs ~8 points on Kaggle. Always train and infer in FP32.
- **Public LB uses only 34% of test data** — private LB may shake significantly
- **Conservative, semantically-accurate models** may outperform pattern-matchers on private LB
- **The metric rewards n-gram overlap, not semantic accuracy** — optimize for BLEU/chrF++ directly
- **Training = document-level, test = sentence-level** — must handle this mismatch
- **Test `line_start`/`line_end` are strings** (e.g. `1`, `1'`, `1''`) — apostrophes mark broken line numbering
- **Determinatives `{d}`, `{ki}` etc. are the ONLY use of `{}` in transliteration** — keep them
- **AICC_translation links in published_texts.csv are "very poor quality"** per competition organizers

---

## File Structure

```
DeepPastChallengeTranslateAkkadianEnglish/
├── plan.md                  # This file
├── requirements.txt         # Python dependencies
├── .gitignore
├── data/
│   ├── raw/                 # Competition data (train.csv, test.csv, lexicon)
│   └── processed/           # Cleaned & aligned training data
├── src/
│   ├── preprocess.py        # Transliteration & translation preprocessing
│   ├── align.py             # Sentence alignment using FirstWord/LinNum
│   ├── extract_publications.py  # Extract translations from publications.csv
│   ├── train_byt5.py        # ByT5 fine-tuning script
│   ├── inference.py         # Inference with beam search + post-processing
│   ├── evaluate.py          # Geometric mean of BLEU & chrF++ (sacrebleu)
│   └── ensemble.py          # Weight averaging for model merging
├── models/                  # Fine-tuned checkpoints (gitignored)
├── jupyter/
│   ├── akkadian-byt5-submission.ipynb
│   └── kernel-metadata.json
└── log/                     # Training logs
```

---

## Timeline

| Week | Phase | Milestone |
|------|-------|-----------|
| 1 (Feb 11–17) | Phase 1 | ByT5 baseline fine-tuned, first submission ~34 |
| 2 (Feb 18–24) | Phase 2 | Preprocessing pipeline, score ~35 |
| 3 (Feb 25–Mar 3) | Phase 3 | Sentence alignment, score ~36 |
| 4 (Mar 4–10) | Phase 4 | Ensemble + advanced techniques, score 37+ |
| 5 (Mar 11–23) | Phase 5 | Final tuning, robustness testing, final submission |

---

## v2 Evaluation Results & Scoring Gap Analysis (Feb 14, 2026)

### v2 Training Summary
- **Model**: `byt5-akkadian-aligned-v2/best` (fine-tuned from v1, 25 epochs on 9,336 aligned rows)
- **Final val_loss**: 0.5320 (improved every epoch except epoch 22)
- **Training time**: 10.7 hours (25 epochs × ~25 min/epoch)

### Local Eval Scores

| Eval subset | BLEU | chrF++ | GeoMean | Notes |
|-------------|------|--------|---------|-------|
| All (n=933) | 6.98 | 25.30 | **13.29** | Dragged down by doc-level pairs |
| Sentence-only (n=798) | 14.74 | 35.78 | **22.97** | Best proxy for Kaggle test |
| sent_pub only (n=692) | 16.27 | 37.22 | **24.61** | Highest-quality aligned data |
| doc only (n=135) | 0.84 | 14.94 | **3.53** | Model can't generate full documents |

**Key insight**: Kaggle test is sentence-level, so real performance is ~23–25, not 13.29.

### Root Causes of Gap (23 → 34+ target)

1. **Under-generation** — Predictions average 75% of reference length; model generates short summaries instead of full translations. Caused by training on mixed doc/sentence data where docs have long refs the model learned to truncate.
2. **Name/number errors** — Only 22.8% of numbers match exactly, only 26.5% of names fully correct. 699 names missed, 463 hallucinated.
3. **Paraphrasing** — Model gets the gist but uses different phrasings, killing BLEU (e.g., "your tablet" vs. "your word", "1/2 shekels per month" vs. "at the rate of the colony").
4. **Doc-level contamination** — 135/933 val pairs are doc-level with GeoMean=3.53, dragging overall score. These don't represent the test format.
5. **Eval reporting** — Overall GeoMean=13.29 was misleading; sentence-only score of ~23 is the real benchmark.

### 5 Fixes for v3 Training

#### Fix 1: Sentence-only training mode (`--sentence-only`)
- **Problem**: Doc-level pairs (14% of data) teach the model to generate short summaries
- **Solution**: Added `--sentence-only` flag to `train_byt5.py` that filters out `source=="doc"` before training
- **Also added**: `--doc-weight` flag for partial downsampling (e.g., `--doc-weight 0.3` keeps 30% of docs)
- **Impact**: Model learns correct output length for sentence-level test data
- **Files changed**: `src/train_byt5.py`

#### Fix 2: Eval reports scores by source type
- **Problem**: Overall GeoMean=13.29 was misleading — doc pairs crushed the score
- **Solution**: `eval_local.py` now reports BLEU/chrF++/GeoMean broken down by source (sent_pub, sent_train, doc) plus a "SENTENCE-ONLY" aggregate as best Kaggle LB proxy
- **Files changed**: `src/eval_local.py`

#### Fix 3: Sentence-only eval split (`--sentence-only`)
- **Problem**: Val split included doc-level pairs not representative of test
- **Solution**: Added `--sentence-only` flag to `eval_local.py` that excludes doc-level pairs from the val split entirely (matches training filtering)
- **Files changed**: `src/eval_local.py`

#### Fix 4: max_new_tokens verified sufficient
- **Problem**: Suspected output truncation at 512 tokens
- **Analysis**: Sentence-level translations: mean=79 bytes, p99=299 bytes, only 0.1% exceed 512 bytes. max_new_tokens=512 is adequate.
- **Root cause**: Under-generation is a model behavior issue from mixed doc/sentence training, not a token limit. Sentence-only training (Fix 1) addresses this directly.
- **No code change needed**

#### Fix 5: Name normalization in eval pipeline
- **Problem**: Eval didn't apply name normalization, so predictions had uncorrected name spellings
- **Analysis**: `NameNormalizer` has 12,348 spelling→name mappings but limited impact on current errors (fuzzy match threshold too strict for most mismatches)
- **Status**: Already integrated in `eval_local.py` via `--with-names` flag and in `inference.py`
- **Note**: Name normalizer cannot fix semantic errors (e.g., "Malahum" vs. "the boatman" — that's a translation, not a spelling variant)

### v3 Training Plan

```bash
# v3: Sentence-only, seed 123, starting from v2 best
nohup python -u src/train_byt5.py \
    --model-name models/byt5-akkadian-aligned-v2/best \
    --output-dir models/byt5-akkadian-v3-sentonly \
    --sentence-only \
    --seed 123 \
    --epochs 25 \
    --lr 3e-5 \
    --batch-size 32 \
    --max-length 512 \
    > log/train_byt5_v3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Expected improvements from sentence-only**:
- Better output length calibration (no doc truncation behavior)
- Cleaner val signal (no doc pairs inflating loss)
- Faster training (~8,180 sentence pairs instead of 9,336 mixed)
- Estimated score: 28–32 GeoMean (sentence-level eval)

---

## ★ BEST CURRENT — Baseline-Matched Training (2026-02-14 03:37 UTC)

**Rationale**: v2 and v3 both scored **13.9** on Kaggle LB. Root cause analysis revealed critical mismatches with the proven public baseline (Takamichi Toda's DPC Starter, scoring ~30+ single model / ~34.9 ensemble). This config replicates that baseline exactly.

**Key changes from v2/v3**:
1. **No preprocessing** — raw transliterations (preprocessing created train/test mismatch)
2. **Adafactor optimizer** (not AdamW) with fixed LR 1e-4
3. **Label smoothing 0.2** (manual CrossEntropyLoss, since T5 config doesn't support it)
4. **Bidirectional training** — English→Akkadian reverse pairs (2x data: 2810 train, ~312 val)
5. **Fresh google/byt5-small** — no inherited overfitting from v1→v2→v3

```bash
# BEST CURRENT — baseline-matched training (started 2026-02-14 03:37)
nohup python -u src/train_byt5.py \
    --data data/raw/train.csv \
    --model-name google/byt5-small \
    --output-dir models/byt5-baseline \
    --optimizer adafactor \
    --lr 1e-4 \
    --label-smoothing 0.2 \
    --bidirectional \
    --no-preprocess \
    --epochs 30 \
    --batch-size 1 \
    --grad-accum 8 \
    --max-length 512 \
    --seed 42 \
    > log/train_byt5_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Training config summary**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `google/byt5-small` (300M params) | Fresh pretrained, no prior fine-tuning |
| Data | `data/raw/train.csv` (1561 rows) | Raw text, no preprocessing |
| Optimizer | Adafactor (fixed LR) | Matches public baseline |
| Learning rate | 1e-4 | 3x higher than v2/v3's 3e-5 |
| Label smoothing | 0.2 | Via manual CrossEntropyLoss |
| Bidirectional | Yes | Akk→Eng + Eng→Akk (2810 train pairs) |
| Batch size | 1 (effective 8 via grad_accum=8) | Matches baseline exactly |
| Epochs | 30 | |
| Max length | 512 | |
| Warmup steps | 1053 (10% of 10,530 total) | |
| Precision | FP32 | BF16 degrades ByT5 quality — always use FP32 |
| Gradient checkpointing | Yes | |
| Preprocessing | OFF | Raw transliterations |
| Seed | 42 | |
| Output | `models/byt5-baseline/` | |

**Inference config** (notebook updated):

| Parameter | Value |
|-----------|-------|
| Beams | 4 (not 8) |
| Length penalty | 1.0 (neutral) |
| Batch size | 8 |
| Max input length | 512 |
| Max new tokens | 512 |
| Test preprocessing | None (raw text) |
| Post-processing | Strip + 'broken text' fallback only |

**Expected score**: 25–32 single model (vs. 13.9 for v2/v3)

---

## ★ NEXT STEPS — Revised Strategy (Feb 19, 2026)

### Current Best: 23.8 Kaggle LB (3x ensemble + LENGTH_PENALTY=2.0)
- Model: `models/byt5-ensemble` (weighted avg of seeds 42/123/777, trained 40ep each)
- Config: batch=1, grad_accum=8, Adafactor lr=1e-4, label_smoothing=0.2, bidi, raw text
- Inference: beams=4, length_penalty=2.0
- Public baseline (Toda's DPC Starter): **34.9** with same architecture → **11-point gap to close**

### Lessons Learned
- **HF Trainer experiment (batch=128) FAILED** — scored 18.2 (regression from 23.8). Large batches destroy generalization on tiny datasets. With 2,800 bidirectional pairs, batch=128 gives only 22 gradient updates/epoch vs ~350 at batch=8. Stick to batch=1, grad_accum=8.
- **Preprocessing creates train/test mismatch** — v2/v3 scored 13.9 on Kaggle despite 22.97 local GeoMean. Raw text is mandatory.
- **LENGTH_PENALTY=2.0 >> 1.0** — inference sweep confirmed +3.7 GeoMean on val.
- **no_repeat_ngram_size destroys byte-level output** — 3 bytes < 1 word, too aggressive.

### Public Baseline Analysis (Feb 19, 2026)

Downloaded and analyzed Takamichi Toda's DPC Starter notebook (34.9 on Kaggle LB).
Same architecture as ours (ByT5-small, 3-seed ensemble, weight-averaged), but critical training differences:

| Parameter | Our train_byt5.py | Public Baseline (Toda) | Impact |
|-----------|-------------------|------------------------|--------|
| Optimizer | Adafactor, fixed LR, manual warmup/decay | HF `optim="adafactor"` (relative_step=False, scale_parameter=False, linear decay scheduler) | **Major** — HF Trainer manages LR schedule properly |
| Padding | Fixed `max_length=512` for all samples | `DataCollatorForSeq2Seq` (dynamic, pads to longest in batch) | **Moderate** — eliminates wasted compute on padding tokens |
| Label smoothing | Manual CrossEntropyLoss | `label_smoothing_factor=0.2` native in Trainer | **Minor** — cleaner implementation, may behave slightly differently |
| Eval metrics | Only val loss | `predict_with_generate=True` → BLEU/chrF++ per epoch | **Diagnostic** — better model selection signal |
| Best model | Save lowest val loss | `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"` | **Minor** — same criterion but HF handles it natively |
| Sentence aligner | N/A | `simple_sentence_aligner()` — produces **0 splits** (1,561→1,561) | **None** — baseline trains on same 1,561 doc-level pairs |

**Key insight**: The gap is in training mechanics, NOT data. Toda's baseline trains on the exact same 1,561 doc-level pairs and scores 34.9. Our custom training loop has subtle bugs in LR scheduling, loss computation, or optimization that cost us ~11 points.

### Step 1: HF Seq2SeqTrainer baseline-matched training (HIGHEST ROI)

New training script `src/train_matched.py` uses HF Seq2SeqTrainer to match Toda's baseline exactly.
Ensemble script: `scripts/train_matched_ensemble.sh` (3 seeds → weight-averaged merge).

**Training config** (`src/train_matched.py`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Framework | HF `Seq2SeqTrainer` | Replaces custom training loop |
| Model | `google/byt5-small` (300M params) | Fresh pretrained |
| Data | `data/raw/train.csv` (1,561 rows) | Raw text, no preprocessing |
| Optimizer | `optim="adafactor"` | HF sets relative_step=False, scale_parameter=False |
| LR | 1e-4 with linear decay | HF default scheduler |
| Label smoothing | 0.2 | Native `label_smoothing_factor` in Trainer |
| Batch size | 1 (effective 8 via grad_accum=8) | Matches baseline exactly |
| Padding | `DataCollatorForSeq2Seq` | Dynamic padding per batch |
| Epochs | 20 | |
| Bidirectional | Yes | Akk→Eng + Eng→Akk doubles data |
| Max length | 512 | |
| Precision | FP32 | BF16 degrades ByT5 quality — always use FP32 |
| Gradient checkpointing | Yes | |
| Eval | `predict_with_generate=True` | BLEU/chrF++/GeoMean per epoch |
| Model selection | `load_best_model_at_end=True` | Best val loss |
| Weight decay | 0.01 | |
| Seeds | 42, 123, 777 | 3-seed ensemble |

```bash
# Launch full pipeline (3 seeds + ensemble merge)
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
rm -f log/train_*.log
nohup bash scripts/train_matched_ensemble.sh \
    > log/train_matched_ensemble_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Expected**: ~30-34 single model, ~34-35 ensemble (matching public baseline)

### Step 2: Submit 4-model ensemble (quick win)
`models/byt5-ensemble-4x` (seeds 42/123/777/280, equal weight) already exists but was never submitted.
Upload and submit to see if 4 models > 3 models.

### Step 3: Train longer (50-60 epochs)
Val loss was still dropping at 40 epochs for all seeds. More gradient updates on tiny data = more learning.
Train best config for 60 epochs and check convergence.

### Step 4: MBR decoding (inference improvement)
Generate 15 diverse candidates (temperature=0.7) + 5 beam search candidates per input.
Score each against all others using chrF++, select consensus winner.
Reduces hallucination. 3-4x slower but within 9hr Kaggle limit.

### Step 5: Larger model (byt5-base, 580M params)
`google/byt5-base` has ~2x parameters. More capacity may help if regularized well.
Risk: may overfit on small data. Test with strong regularization (label_smoothing=0.3, dropout).

### Score Tracker

| Model | Config | Val Loss | Local GeoMean | Kaggle LB |
|-------|--------|----------|---------------|-----------|
| v2 (aligned-v2) | preprocessed, from v1 | 0.532 | 22.97 (sent) | 13.9 |
| v3 (sent-only) | preprocessed, sent-only | — | — | 13.9 |
| HF ensemble (3x) | batch=128, lr=3e-4, 30ep | 1.487 | 16.57 (val) | **18.2** |
| baseline (single) | raw text, bidi, Adafactor, 30ep | 0.642 | 5.11 (doc) | **21.3** |
| baseline-seed42 | same + 40ep | 0.612 | — | — |
| baseline-seed123 | same + seed 123, 40ep | 0.651 | — | — |
| baseline-seed777 | same + seed 777, 40ep | 0.596 | — | — |
| **ensemble (3x)** | weighted avg (42/123/777) + LP=2.0 | — | 23.5 (val) | **23.8** |
| ensemble-4x | 4 seeds (42/123/777/280) | — | — | not submitted |
| *public baseline* | *Toda DPC Starter, 3-seed ensemble* | — | — | *34.9* |
| matched-ensemble (3x) | HF Trainer, constant LR, 50ep | 2.027 | — | **26.0** |
| matched-ensemble + rep_pen | same + repetition_penalty=1.3 | — | — | pending (v19) |
| *public baseline* | *Toda DPC Starter, 3-seed ensemble* | — | — | *34.9 (pre-rescore)* |

---

## Monday Feb 23 Action Plan

### Context
- Final dataset update (3rd revision) expected Monday from Adam Anderson.
- After that update, Kaggle will rescore **all** existing submissions — leaderboard will shift.
- Our `matched-ensemble` scored **26.0** against the new labels. The public baseline's 34.9 is pre-rescore against old labels — not a fair comparison yet.
- One confirmed deviation from Toda's baseline: we used `lr_scheduler_type="constant_with_warmup"` for 50 epochs; the baseline uses HF default **linear decay** for ~20 epochs. This is the primary untested fix.

### Steps

**1. Download the final `train.csv`** (after confirming Adam has posted the update):
```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
kaggle competitions download -c deep-past-initiative-machine-translation -f train.csv -p data/raw/ --force
```

**2. Check the rescored leaderboard** — note updated scores for all submissions, especially the public baseline's new score, before spending time training.

**3. Fix `train_matched.py`** — switch to linear LR decay to match Toda exactly:
In `training_args`, change:
- `lr_scheduler_type="constant_with_warmup"` → `"linear"`
- `num_train_epochs=50` → `20` (linear decay makes late epochs near-zero LR anyway)

**4. Update `scripts/train_matched_ensemble.sh`** to pass `--epochs 20`.

**5. Retrain 3-seed ensemble** on the final `train.csv`:
```bash
rm -f log/train_*.log
nohup bash scripts/train_matched_ensemble.sh > log/train_matched_ensemble_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
Expected runtime: ~2h/seed × 3 seeds = ~6h total.

**6. Submit** and compare against the rescored leaderboard.

**7. Investigate why val metrics are stuck (matched-ensemble seed 42 analysis)**

The completed seed 42 run (50 epochs, FP32, constant LR) showed:
- **BLEU ≈ 0.0** throughout all 50 epochs — model never generates correct n-gram sequences on the val set
- **chrF++ stuck ~5.1–5.2** from epoch 20 onward, barely moving despite val_loss still improving (2.05 → 2.04)
- Training loss descending steadily (1.89 at epoch 50) with a widening train/val gap → mild overfitting
- Val metrics here are **document-level**; Kaggle test is sentence-level — inherent mismatch

Actions:
- Sample a few val predictions from seed 42's best checkpoint and inspect them manually
- Check if the model is collapsing to a degenerate output (e.g. repeated tokens, empty strings)
- Confirm whether doc-level chrF++ ~5 is consistent with our ~26 Kaggle sentence-level score (it may just be the mismatched granularity)
- If BLEU truly never fires, consider whether `max_new_tokens` during val eval is too short to generate full translations

**8. Fix phrase repetition loops in beam search output**

Inspecting val predictions revealed outputs like:
> "our messenger, our messenger, our messenger and the stations..."
> "as for this day, as for this day, as for this day anyone else..."

The notebook already uses `repetition_penalty=1.3` but it doesn't work for ByT5 — byte-level tokenization means "our messenger" spans ~13 unique bytes, so the penalty barely registers until 10+ full phrase repetitions.

**Option A — `no_repeat_ngram_size=12`** in `model.generate()` in the notebook:
- Blocks any 12-byte sequence from repeating, roughly a 2-word no-repeat constraint
- Previously removed (commit `6e63056`) at an unknown value because it hurt score — try higher value (12–16 bytes)
- Risk: may block legitimate repeated phrases (e.g. proper nouns, formulaic OA language)

**Option B — Post-processing deduplication** (safer):
- After generation, detect repeated n-word phrases (e.g. trigrams repeated 2+ consecutive times) and collapse them
- Only fires when loops actually occur; no impact on clean outputs
- Add to both `src/inference.py` and the notebook

Recommended: implement Option B as a safety net first, then test Option A on a held-out val sample to see if it improves GeoMean before adding it to the submission notebook.

### Data quality issues to monitor (from MPWARE's Feb 21 audit)
These may still be present in Monday's update — check after downloading:
- `fem.` / `pl.` / `sing.` still in some translation rows
- Inconsistent fractions: some `0.3333` as float, some as `⅓` — **leave as-is, don't convert**
- Bare `x` tokens in translations that should be `<gap>`
- Possessives inconsistent (`Anna's` vs `Annas`)
- Roman numerals (e.g. "IVth") — no guidance yet; leave as-is

---

## Known Issues / Improvements for Next Training Run

### Fix: Save checkpoint on GeoMean, not val_loss
**Status**: Not yet applied — will be in next retraining run.

Currently `Seq2SeqTrainingArguments` uses:
```python
metric_for_best_model="eval_loss",
greater_is_better=False,
```

This saves on val_loss, which is only a proxy. The competition metric is `GeoMean = sqrt(BLEU × chrF++)`. In seed 777, epoch 39 had GeoMean 41.34 but was not saved because its val_loss (2.026) was slightly worse than epoch 38 (2.023, GeoMean 40.68).

**Fix** — change to:
```python
metric_for_best_model="eval_geo_mean",
greater_is_better=True,
```

### Fix: BLEU was zero due to generation_max_length not set
**Status**: Fixed Feb 26, 2026 — applied in seed 777 only.

ByT5's `config.max_length = 20` bytes. Without `generation_max_length` in `Seq2SeqTrainingArguments`, HF Trainer generated only 20 bytes per val example during eval. With references averaging 511 bytes, sacrebleu's brevity penalty → 0 → BLEU ≈ 1e-6 throughout all epochs. Seeds 42 and 123 ran 50 epochs each with completely meaningless BLEU/GeoMean metrics (though val_loss-based checkpointing was unaffected).

**Fix**: `generation_max_length=512` (covers p50=392 bytes and keeps eval fast ~220s vs ~870s at 2048).

---

## After Feb 23 Ensemble — Next Ablations

After submitting the linear-LR matched ensemble, investigate these hypotheses if the score remains below ~30. Each is a candidate explanation for the ~8.5 point gap vs. Toda's 34.9.

### A. Bidirectional training may be hurting us (HIGH PRIORITY)
We train on Akk→Eng + Eng→Akk (2,810 pairs). Toda almost certainly trains only Akk→Eng (1,562 pairs). Half the model's capacity goes to a direction never tested at inference. This alone could easily cost 5+ points.

**Experiment**: Train one seed unidirectional (`--no-bidirectional`) and compare val GeoMean against the equivalent bidirectional seed. If unidirectional is higher, drop bidirectional from the ensemble.

### B. Inference parameters may be hurting us
We just added `no_repeat_ngram_size=16` and `repetition_penalty=1.3` — these are untested on the actual Kaggle test set and could hurt as easily as help. Toda's inference is probably vanilla beam search with no penalties.

**Experiment**: Run inference on local val set with and without these parameters. Compare GeoMean. If neutral or negative, revert to vanilla in the notebook before submitting.

### C. Instruction prefix mismatch
We prepend `"translate Akkadian to English: "` to all inputs during training and inference. If Toda uses no prefix or a different prefix, inference behavior would diverge from what it was trained on.

**Check**: Inspect Toda's public notebook (if available on Kaggle discussion) for exact prefix string used.

### D. num_beams too low
We use `num_beams=4`. Toda may use 8 with a different `length_penalty`. Higher beam count generally improves translation quality at the cost of inference speed.

**Experiment**: Test `num_beams=8` with `length_penalty=1.3` vs current `num_beams=4` on local val. If GeoMean improves, update the notebook. Note: Kaggle 9hr limit — ensure inference still completes in time.

### E. Data quality audit on final train.csv
After downloading the final `train.csv` (3rd revision from Adam Anderson), check for these known issues from MPWARE's Feb 21 audit:

- `fem.` / `pl.` / `sing.` still in some translation rows — strip these annotations
- Bare `x` tokens in translations that should be `<gap>` — normalize
- Possessives inconsistent (`Anna's` vs `Annas`) — leave as-is unless a clear rule emerges
- Roman numerals (e.g. "IVth") — leave as-is, no guidance yet
- Inconsistent fractions: some `0.3333` as float, some as `⅓` — **leave as-is, don't convert**

**Script**: Run `python src/preprocess.py --audit data/raw/train.csv` (or equivalent) to flag rows matching these patterns and decide whether to clean or leave. Any cleaning should be applied consistently to both training targets and inference post-processing.
