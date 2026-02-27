#!/bin/bash
# ── Ensemble merge + Kaggle upload ────────────────────────────────────
# Waits for seed 777 training to finish, then:
#   1. Merges seeds 42, 123, 777 into an ensemble
#   2. Uploads ensemble to Kaggle dataset
#   3. Pushes submission notebook to Kaggle
#
# Usage (run after relaunching seed 777):
#   nohup bash scripts/ensemble_and_push.sh > log/ensemble_and_push_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

eval "$(conda shell.bash hook)"
conda activate phi4

cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish

SEED777_BEST="models/byt5-matched-seed777/best"
POLL_INTERVAL=60  # seconds between checks

echo "============================================================"
echo "  Ensemble-and-Push Watcher"
echo "  Waiting for: ${SEED777_BEST}"
echo "  Started: $(date)"
echo "============================================================"

# ── Poll until seed 777 best model exists ────────────────────────────
while [ ! -d "${SEED777_BEST}" ]; do
    echo "$(date): seed 777 not done yet, checking again in ${POLL_INTERVAL}s..."
    sleep ${POLL_INTERVAL}
done

# Extra guard: wait until the tokenizer_config.json is present (model fully saved)
while [ ! -f "${SEED777_BEST}/tokenizer_config.json" ]; do
    echo "$(date): model dir exists but save incomplete, waiting ${POLL_INTERVAL}s..."
    sleep ${POLL_INTERVAL}
done

echo ""
echo "$(date): Seed 777 complete. Proceeding with ensemble merge."
echo ""

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

echo "Ensemble merge complete — $(date)"
echo ""

# ── Upload to Kaggle ──────────────────────────────────────────────────
echo "============================================"
echo "Uploading model to Kaggle — $(date)"
echo "============================================"

cat > models/byt5-matched-ensemble/dataset-metadata.json <<EOF
{
  "title": "byt5-akkadian-finetuned",
  "id": "stevewatson999/byt5-akkadian-finetuned",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

cp -f data/raw/onomasticon/onomasticon.csv models/byt5-matched-ensemble/ 2>/dev/null || true

kaggle datasets version \
    -p models/byt5-matched-ensemble \
    -m "Matched-baseline ensemble (3 seeds, train.csv, batch=1, 50ep, constant LR, gen_max_len=512)" \
    --dir-mode zip

echo "Upload complete — $(date)"

echo "Waiting 120s for Kaggle to process dataset version..."
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
echo "============================================"
