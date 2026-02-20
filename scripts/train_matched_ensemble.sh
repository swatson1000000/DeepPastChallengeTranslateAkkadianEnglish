#!/bin/bash
# ── Baseline-Matched Ensemble Training Pipeline ──────────────────────
# Matches the public DPC Starter (Takamichi Toda) baseline exactly:
#   - HF Seq2SeqTrainer with optim="adafactor" (relative_step=False, fixed LR + linear decay)
#   - DataCollatorForSeq2Seq for dynamic padding
#   - label_smoothing=0.2, batch=1, grad_accum=8
#   - Bidirectional, BF16, gradient checkpointing
#
# Trains 3 seeds sequentially, then merges into ensemble.
#
# Usage:
#   conda activate phi4
#   cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
#   rm -f log/train_*.log
#   nohup bash scripts/train_matched_ensemble.sh > log/train_matched_ensemble_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

# ── Activate conda environment ───────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate phi4

cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish

DATA="data/raw/train.csv"
EPOCHS=50
BATCH=1
GRAD_ACCUM=8
MAX_LEN=512
LABEL_SMOOTH=0.2
LR=1e-4
SEEDS=(42 123 777)
NUM_SEEDS=${#SEEDS[@]}

echo ""
echo "============================================================"
echo "  Baseline-Matched Ensemble Training Pipeline"
echo "  Data: ${DATA} (1,561 rows — exact baseline match)"
echo "  Config: HF Seq2SeqTrainer, Adafactor (relative_step=False, linear decay)"
echo "  Batch: ${BATCH} x ${GRAD_ACCUM} = $((BATCH * GRAD_ACCUM)) effective"
echo "  Epochs: ${EPOCHS}, label_smoothing=${LABEL_SMOOTH}"
echo "  BF16: ON, gradient_checkpointing: ON"
echo "  Seeds: ${SEEDS[*]}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ── Train each seed ──────────────────────────────────────────────────
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    IDX=$((i + 1))
    OUTDIR="models/byt5-matched-seed${SEED}"

    echo "============================================"
    echo "[${IDX}/${NUM_SEEDS}] Training seed ${SEED} — $(date)"
    echo "============================================"

    python -u src/train_matched.py \
        --data "${DATA}" \
        --model-name google/byt5-small \
        --output-dir "${OUTDIR}" \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH} \
        --grad-accum ${GRAD_ACCUM} \
        --max-length ${MAX_LEN} \
        --label-smoothing ${LABEL_SMOOTH} \
        --lr ${LR} \
        --bidirectional \
        --seed ${SEED} \
        --bf16

    echo "[${IDX}/${NUM_SEEDS}] Seed ${SEED} complete — $(date)"
    echo ""
done

# ── Ensemble merge ────────────────────────────────────────────────────
echo "============================================"
echo "Merging seeds into ensemble — $(date)"
echo "============================================"

python -u src/ensemble.py \
    --models models/byt5-matched-seed42/best \
            models/byt5-matched-seed123/best \
            models/byt5-matched-seed777/best \
    --weights 0.34 0.33 0.33 \
    --output models/byt5-matched-ensemble

echo ""
echo "============================================"
echo "  Ensemble complete — $(date)"
echo "  Ensemble: models/byt5-matched-ensemble"
echo "============================================"

# ── Upload to Kaggle ──────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Uploading model to Kaggle — $(date)"
echo "============================================"

# Create dataset-metadata.json in ensemble dir
cat > models/byt5-matched-ensemble/dataset-metadata.json <<EOF
{
  "title": "byt5-akkadian-finetuned",
  "id": "stevewatson999/byt5-akkadian-finetuned",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Copy onomasticon for name normalization in notebook
cp -f data/raw/onomasticon/onomasticon.csv models/byt5-matched-ensemble/ 2>/dev/null || true

# Upload (create new version of existing dataset)
kaggle datasets version -p models/byt5-matched-ensemble -m "Matched-baseline ensemble (3 seeds, train.csv, batch=1, 50ep, constant LR)" --dir-mode zip

echo "Upload complete — $(date)"

# Wait for Kaggle to finish processing the new dataset version before pushing the
# notebook — otherwise the notebook may run against the previous dataset version.
echo "Waiting 120s for Kaggle dataset version to be processed..."
sleep 120

# ── Push submission notebook ──────────────────────────────────────────
echo ""
echo "============================================"
echo "Pushing submission notebook to Kaggle — $(date)"
echo "============================================"

cd jupyter && kaggle kernels push -p .
cd ..

echo ""
echo "============================================"
echo "  Full pipeline complete — $(date)"
echo "  Ensemble: models/byt5-matched-ensemble"
echo "  Kaggle dataset: stevewatson999/byt5-akkadian-finetuned"
echo "  Notebook pushed: stevewatson999/akkadian-byt5-submission"
echo "============================================"
