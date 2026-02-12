# Deep Past Challenge — ByT5 Competition Plan

**Competition**: [Deep Past Initiative: Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)  
**Goal**: Maximize geometric mean of BLEU and chrF++ on Akkadian→English translation.  
**Deadline**: March 23, 2026 (Entry/Merger deadline: March 16)  
**Constraints**: Kaggle code competition — GPU notebook ≤9hrs, internet OFF, `submission.csv`  
**Prizes**: $50,000 total ($15K / $10K / $8K / $7K / $5K / $5K)  
**Current score**: 2.7 (LSTM baseline)  
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
  - **FP16**: OFF (critical — prevents NaN errors per top teams)
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
5. Write `submission.csv` to `/kaggle/working/`

### 5.3 Push to Kaggle
```bash
cd jupyter && kaggle kernels push -p .
```

---

## Key Warnings

- **DO NOT use EvaCun/ORACC data** — Neo-Assyrian (911–539 BCE), 1,000 years wrong for Old Assyrian
- **DO NOT use fp16** — causes NaN errors with ByT5
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
