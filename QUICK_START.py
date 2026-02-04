#!/usr/bin/env python3
"""
QUICK START GUIDE FOR CONSOLIDATED SCRIPTS

Instead of multiple scattered scripts, use:
  python train.py --model {baseline|improved|tier2}
  python inference.py --model {baseline|improved|tier2}

See CONSOLIDATED_SCRIPTS.md for full documentation.
"""

USAGE_EXAMPLES = """
╔════════════════════════════════════════════════════════════════════════════╗
║                      AKKADIAN TRANSLATION PIPELINE                        ║
║                      Consolidated Scripts Guide                           ║
╚════════════════════════════════════════════════════════════════════════════╝

TRAINING EXAMPLES
═════════════════════════════════════════════════════════════════════════════

1. Train baseline Seq2Seq model
   python train.py --model baseline --epochs 100

2. Train TIER 1 improved model (recommended)
   python train.py --model improved --epochs 200

3. Train TIER 2 model with advanced features
   python train.py --model tier2 --epochs 300 --use-copy --use-lexicon

4. Custom training options
   python train.py --model improved \\
     --batch-size 64 \\
     --epochs 250 \\
     --data-path data/processed/train_augmented.csv

5. Use custom config file
   python train.py --config configs/model_seq2seq_improved.yaml


INFERENCE EXAMPLES
═════════════════════════════════════════════════════════════════════════════

1. Generate predictions with improved model
   python inference.py --model improved

2. Use TIER 2 features for inference
   python inference.py --model tier2 --use-copy

3. Custom output location
   python inference.py --model improved --output my_predictions.csv

4. Process limited samples (for testing)
   python inference.py --model improved --max-samples 100

5. Specify custom test data
   python inference.py --model improved --test-data data/raw/test.csv


MODEL COMPARISON
═════════════════════════════════════════════════════════════════════════════

BASELINE
  - Standard Seq2Seq architecture
  - LSTM encoder with attention
  - Greedy decoding
  - ~100 epochs recommended
  - Best for: Quick baselines

IMPROVED (TIER 1)
  - Optimized architecture (2 layers, 512 hidden)
  - Better attention and decoding
  - Temperature scaling for diversity
  - ~200 epochs recommended
  - Best for: Production use

TIER 2
  - Copy mechanism for proper nouns
  - Lexicon-constrained decoding
  - Coverage tracking
  - ~300 epochs recommended
  - Best for: Maximum accuracy


COMMON WORKFLOWS
═════════════════════════════════════════════════════════════════════════════

Quick Test
──────────
1. python train.py --model baseline --epochs 10
2. python inference.py --model baseline --max-samples 10

Production Training
──────────────────
1. python train.py --model improved --epochs 200
2. python inference.py --model improved --output final_predictions.csv

Advanced Training
─────────────────
1. python train.py --model tier2 --use-copy --use-lexicon --epochs 300
2. python inference.py --model tier2 --use-copy


CHECKPOINTS
═════════════════════════════════════════════════════════════════════════════

After training, checkpoints are saved to:
  checkpoints/baseline_best.pt
  checkpoints/improved_best.pt
  checkpoints/tier2_best.pt

To use a specific checkpoint:
  python inference.py --model improved --checkpoint checkpoints/improved_best.pt


FILE STRUCTURE
═════════════════════════════════════════════════════════════════════════════

train.py ..................... Main training script (all variants)
inference.py ................. Main inference script (all variants)
CONSOLIDATED_SCRIPTS.md ....... Full documentation
configs/
  model_seq2seq.yaml ......... Baseline config
  model_seq2seq_improved.yaml  TIER 1 config
  model_seq2seq_tier2.yaml ... TIER 2 config
checkpoints/ ................. Saved models
archive/ ..................... Old scripts (for reference)


TROUBLESHOOTING
═════════════════════════════════════════════════════════════════════════════

Q: "ModuleNotFoundError: No module named 'src'"
A: Run scripts from the project root directory

Q: "CUDA out of memory"
A: Use baseline model or reduce batch size:
   python train.py --model baseline --batch-size 32

Q: "Test data not found"
A: Ensure test.csv exists at data/raw/test.csv

Q: "Checkpoint not found"
A: Run training first to generate checkpoint:
   python train.py --model improved


PERFORMANCE TIPS
═════════════════════════════════════════════════════════════════════════════

- TIER 1 (improved) is the best balance of accuracy and speed
- Increase epochs for better accuracy (diminishing returns after 200)
- Use --use-copy in TIER 2 for proper noun handling
- Adjust temperature for diversity vs coherence tradeoff
- Monitor val_loss for early stopping


NEXT STEPS
═════════════════════════════════════════════════════════════════════════════

1. Start with: python train.py --model improved --epochs 10
2. Check output in log files
3. Run inference: python inference.py --model improved
4. Evaluate predictions: python evaluate_predictions.py
5. Tweak config and retrain as needed

═════════════════════════════════════════════════════════════════════════════
For detailed documentation, see: CONSOLIDATED_SCRIPTS.md
"""

if __name__ == '__main__':
    print(USAGE_EXAMPLES)
