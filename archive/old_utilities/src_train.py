"""
Unified training pipeline for both Seq2Seq and mBART-50 models.

Executes baseline model training with proper logging and checkpointing.
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_seq2seq():
    """Train Seq2Seq baseline model."""
    logger.info("Starting Seq2Seq baseline model training...")
    
    try:
        from src.models.seq2seq import Seq2SeqTrainer
        
        config = load_config('configs/model_seq2seq.yaml')
        trainer = Seq2SeqTrainer(config)
        
        logger.info("Loading preprocessed training data...")
        trainer.load_data('data/processed/train_clean.csv')
        
        logger.info("Building model...")
        trainer.build_model()
        
        logger.info("Training model...")
        trainer.train(
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32)
        )
        
        logger.info("Evaluating model...")
        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")
        
        logger.info("Saving model...")
        trainer.save_model('models/seq2seq/final_model.pt')
        
        logger.info("✓ Seq2Seq training complete!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Seq2Seq training failed: {e}")
        return False


def train_mbart():
    """Train mBART-50 fine-tuned model."""
    logger.info("Starting mBART-50 fine-tuning...")
    
    try:
        from src.models.mbart import MBartTrainer
        
        config = load_config('configs/model_mbart50.yaml')
        trainer = MBartTrainer(config)
        
        logger.info("Loading preprocessed training data...")
        trainer.load_data('data/processed/train_clean.csv')
        
        logger.info("Building model...")
        trainer.build_model()
        
        logger.info("Fine-tuning model...")
        trainer.train(
            epochs=config.get('epochs', 30),
            batch_size=config.get('batch_size', 16)
        )
        
        logger.info("Evaluating model...")
        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")
        
        logger.info("Saving model...")
        trainer.save_model('models/mbart50/final_model')
        
        logger.info("✓ mBART-50 training complete!")
        return True
        
    except Exception as e:
        logger.error(f"✗ mBART-50 training failed: {e}")
        return False


def main():
    """Execute training pipeline."""
    logger.info("=" * 80)
    logger.info("AKKADIAN-ENGLISH TRANSLATION: MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Check if preprocessed data exists
    if not Path('data/processed/train_clean.csv').exists():
        logger.error("Preprocessed data not found. Run preprocessing first.")
        sys.exit(1)
    
    # Train Seq2Seq baseline
    seq2seq_success = train_seq2seq()
    
    # Train mBART-50
    mbart_success = train_mbart()
    
    # Summary
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Seq2Seq baseline: {'✓ Success' if seq2seq_success else '✗ Failed'}")
    logger.info(f"mBART-50 fine-tune: {'✓ Success' if mbart_success else '✗ Failed'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
