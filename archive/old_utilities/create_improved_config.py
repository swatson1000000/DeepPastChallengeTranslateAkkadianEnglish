#!/usr/bin/env python3
"""
Create an improved training configuration with reduced model capacity.
TIER 1 Improvement: Reduce parameter-to-data ratio for better learning.
"""

import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def create_improved_config():
    """
    Create improved training config with:
    - 2-layer LSTM (reduced from 3)
    - 512 hidden units (reduced from 768)
    - Uses augmented training data
    """
    
    config = {
        'model': {
            'name': 'seq2seq_reduced_capacity_v2',
            'encoder': {
                'type': 'lstm',
                'embedding_size': 384,  # Keep same
                'hidden_size': 512,     # REDUCED from 768 (33% reduction)
                'num_layers': 2,        # REDUCED from 3 (33% reduction)
                'dropout': 0.2,
            },
            'attention': {
                'type': 'bahdanau',
                'hidden_size': 512,
            },
            'decoder': {
                'hidden_size': 512,
            }
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'epochs': 250,  # Extended training with reduced capacity
            'early_stopping_patience': 30,  # Increased patience
            'use_gru': False,
            'gradient_accumulation_steps': 1,
            'use_augmented_data': True,  # USE AUGMENTED DATA
        },
        'data': {
            'source_lang': 'akkadian',
            'target_lang': 'english',
            'train_file': 'data/processed/train_augmented.csv',  # AUGMENTED DATA
            'test_file': 'data/raw/test.csv',
            'vocab_size_source': None,  # Auto-determined
            'vocab_size_target': None,  # Auto-determined
            'max_length': 256,
        }
    }
    
    return config

def save_config(config, config_name='model_seq2seq_improved.yaml'):
    """Save configuration to file."""
    project_root = Path(__file__).parent.parent.parent
    config_dir = project_root / 'configs'
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / config_name
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"✓ Saved improved config to {config_path}")
    return config_path

def main():
    """Generate and save improved configuration."""
    logger.info("="*80)
    logger.info("TIER 1 IMPROVEMENT #2: REDUCE MODEL CAPACITY")
    logger.info("="*80)
    
    config = create_improved_config()
    
    logger.info("\nNew Configuration (Reduced Capacity):")
    logger.info("-" * 80)
    logger.info(f"Encoder LSTM:")
    logger.info(f"  - Hidden size: {config['model']['encoder']['hidden_size']} (was 768)")
    logger.info(f"  - Num layers: {config['model']['encoder']['num_layers']} (was 3)")
    logger.info(f"  - Embedding: {config['model']['encoder']['embedding_size']}")
    logger.info(f"\nTraining:")
    logger.info(f"  - Max epochs: {config['training']['epochs']}")
    logger.info(f"  - Early stopping patience: {config['training']['early_stopping_patience']}")
    logger.info(f"  - Use augmented data: {config['training']['use_augmented_data']}")
    logger.info(f"  - Batch size: {config['training']['batch_size']}")
    logger.info(f"\nData:")
    logger.info(f"  - Train file: {config['data']['train_file']} (augmented)")
    
    # Calculate parameter reduction
    old_params = 384 * 768 * 3 * 4 + 768 * 768 * 3 * 4  # Rough estimate
    new_params = 384 * 512 * 2 * 4 + 512 * 512 * 2 * 4  # Rough estimate
    reduction = (1 - new_params / old_params) * 100
    
    logger.info(f"\nParameter Reduction:")
    logger.info(f"  - Estimated reduction: {reduction:.1f}%")
    logger.info(f"  - Better param-to-data ratio for learning")
    
    # Save config
    config_path = save_config(config)
    
    logger.info(f"\n{'='*80}")
    logger.info("NEXT STEP: Run data augmentation, then retrain with this config")
    logger.info("="*80)

if __name__ == "__main__":
    main()
