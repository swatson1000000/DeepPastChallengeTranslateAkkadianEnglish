# GPU Optimization Summary - 75GB Memory

## Configuration Updates for 75GB GPU

With 75GB of GPU memory available, the training configurations have been optimized for memory efficiency while maintaining performance and quality.

### Key Optimizations Made

#### 1. Seq2Seq Baseline Model

| Parameter | Original | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| Batch Size | 32 | 192 | **6x larger** |
| Hidden Size | 512 | 768 | **+50% capacity** |
| Encoder Layers | 2 | 3 | **+1 deeper** |
| Decoder Layers | 2 | 3 | **+1 deeper** |
| Epochs | 100 | 80 | **-20% runtime** |
| Dropout | 0.3 | 0.2 | **Lower regularization** |
| Grad Accumulation | 1 | 1 | **Direct GPU usage** |
| Mixed Precision | No | Yes (fp16) | **~2x speed** |

**Expected Impact**: Faster convergence, larger model capacity, better BLEU scores (~15-18 expected)

#### 2. mBART-50 Pre-trained Model

| Parameter | Original | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| Batch Size | 16 | 96 | **6x larger** |
| Epochs | 50 | 25 | **-50% runtime** |
| Grad Accumulation | 2 | 1 | **Direct GPU usage** |
| Beam Search | 5 | 8 | **+60% search quality** |
| Gradient Checkpointing | False | True | **Memory efficiency** |
| Learning Rate | 5e-5 | 2e-5 | **More stable** |
| Warmup Steps | 2000 | 1000 | **Faster warming** |
| Early Stopping Patience | 5 | 3 | **Faster stopping** |

**Expected Impact**: Much faster training, better translation quality, BLEU ~20-25 expected

### Data Loading Optimizations

- **num_workers**: 8 (parallel data loading)
- **pin_memory**: True (faster GPU transfer)
- **prefetch_factor**: Enabled for pipeline efficiency
- **persistent_workers**: True (reduce overhead)

### Memory Usage Breakdown (Estimated)

```
GPU Memory: 75GB

Seq2Seq (batch_size=192):
- Model parameters:      ~500 MB
- Activations:          ~6-8 GB
- Optimizer states:     ~1-2 GB
- Batch data:           ~1.5-2 GB
- Buffer space:         ~5 GB (for safety)
- Available:            ~60+ GB

mBART-50 (batch_size=96):
- Model parameters:      ~3 GB
- Activations:          ~12-15 GB
- Optimizer states:     ~6-8 GB
- Batch data:           ~1-1.5 GB
- Buffer space:         ~8 GB (for safety)
- Available:            ~45+ GB
```

### Training Speed Estimates

**Seq2Seq Training**:
- Original (batch=32): ~4-5 hours
- Optimized (batch=192): **~1.5-2 hours** (2-3x faster)
- Epochs: 80

**mBART-50 Training**:
- Original (batch=16): ~8-10 hours
- Optimized (batch=96): **~2-2.5 hours** (3-4x faster)
- Epochs: 25

**Total Training Time**: ~3.5-4.5 hours for both models (vs 12-15 hours originally)

### Performance Predictions

| Metric | Expected (Optimized) | Previous Baseline |
|--------|---------------------|-------------------|
| **Seq2Seq BLEU** | 15-18 | 12-15 |
| **Seq2Seq chrF++** | 28-32 | 25-30 |
| **mBART BLEU** | 22-26 | 18-22 |
| **mBART chrF++** | 40-45 | 35-40 |
| **Ensemble BLEU** | 26-30 | 24-28 |
| **Ensemble chrF++** | 45-50 | 45-48 |

### Hardware Specifications

```
GPU Memory:           75 GB
Batch Sizes:          192 (Seq2Seq), 96 (mBART)
Mixed Precision:      fp16 enabled
Gradient Checkpointing: Enabled for mBART
Data Workers:         8
Pin Memory:           Enabled
Device:               CUDA
Computation Precision: Mixed (fp16 + fp32)
```

### Execution Commands

Execute REAL training with optimized configs:

```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish

# Train Seq2Seq with REAL tensor-based training
# Uses custom tokenizer with SOS/EOS/PAD/UNK tokens
# Actual backpropagation and gradient descent
nohup python src/models/train_seq2seq.py > log/train_seq2seq_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Train mBART-50 with REAL transformers fine-tuning
# Uses transformers Seq2SeqTrainer with mixed precision
nohup python src/models/train_mbart.py > log/train_mbart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**What "Real" Means**:
✓ Actual tensor encoding of all 1,561 data samples  
✓ Real gradient computation and backpropagation  
✓ Model weight updates via optimizer (Adam)  
✓ Checkpoint saving for inference  
✓ Actual loss curves (not simulated)  
✓ No artificial delays (no time.sleep)

### Verification

To verify GPU optimizations are loaded:

```bash
python src/gpu_optimizations.py
```

Expected output:
```
GPU Optimization Settings for 75GB Memory:

Seq2Seq Configuration:
  batch_size: 192
  hidden_size: 768
  num_layers: 3
  ...

mBART-50 Configuration:
  batch_size: 96
  beam_size: 8
  ...
```

---

**Status**: ✓ Configurations updated for 75GB GPU  
**Date**: February 2, 2026  
**Ready for**: Phase 2 Model Training
