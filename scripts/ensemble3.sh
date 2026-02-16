#!/bin/bash
# Create 3-model ensemble and submit to Kaggle
# Usage: nohup bash scripts/ensemble3.sh > log/ensemble3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -e

PROJECT_DIR="/home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish"
cd "$PROJECT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate phi4

echo ""
echo "============================================"
echo "Creating 3-model ensemble (seeds 42/123/777)"
echo "============================================"

python -u src/ensemble.py \
    --models \
        models/byt5-baseline-seed42/best \
        models/byt5-baseline-seed123/best \
        models/byt5-baseline-seed777/best \
    --weights 0.34 0.33 0.33 \
    --output models/byt5-ensemble/

echo ""
echo "============================================"
echo "Stage for Kaggle upload"
echo "============================================"

rm -rf /tmp/kaggle-upload
mkdir -p /tmp/kaggle-upload

cp models/byt5-ensemble/* /tmp/kaggle-upload/
cp data/raw/onomasticon/onomasticon.csv /tmp/kaggle-upload/

cat > /tmp/kaggle-upload/dataset-metadata.json << 'EOF'
{
  "title": "ByT5 Akkadian Finetuned",
  "id": "stevewatson999/byt5-akkadian-finetuned",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo "Files staged at /tmp/kaggle-upload/:"
ls -lh /tmp/kaggle-upload/

echo ""
echo "============================================"
echo "Uploading dataset to Kaggle"
echo "============================================"

kaggle datasets version -p /tmp/kaggle-upload/ -m "3-model ensemble (seeds 42/123/777, 40ep each)" --dir-mode zip

echo ""
echo "============================================"
echo "Pushing notebook to Kaggle"
echo "============================================"

cd "$PROJECT_DIR/jupyter"
kaggle kernels push -p .

echo ""
echo "============================================"
echo "Done! Dataset uploaded and notebook submitted."
echo "============================================"
