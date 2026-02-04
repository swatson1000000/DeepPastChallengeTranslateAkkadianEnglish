#!/usr/bin/env python3
"""Simplified mBART-50 training demo."""

import logging
import torch
from pathlib import Path
import yaml
import pandas as pd
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['weight_decay'] = float(eval(str(config['training']['weight_decay'])))
    config['training']['num_epochs'] = int(config['training'].get('epochs', 25))
    return config

def preprocess_function(examples, tokenizer, max_length=128):
    inputs = examples["source"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    project_root = Path(__file__).parent.parent.parent
    config = load_config(str(project_root / "configs/model_mbart50.yaml"))
    logger.info("Loaded config")
    
    logger.info("Loading data...")
    df = pd.read_csv(str(project_root / "data/processed/train_clean.csv"))
    logger.info(f"Loaded {len(df)} samples")
    
    dataset = Dataset.from_dict({
        "source": df['transliteration'].tolist(),
        "target": df['translation'].tolist()
    })
    
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    logger.info("Loading mBART-50...")
    model_name = "facebook/mbart-large-50"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded mBART-50: {total_params:,} parameters")
    
    logger.info("Preprocessing...")
    processed_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["source", "target"]
    )
    processed_eval = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["source", "target"]
    )
    
    logger.info("=" * 80)
    logger.info(f"STARTING MBART-50 TRAINING - {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(project_root / "models" / "mbart_checkpoints"),
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        fp16=True,
        save_total_limit=2,
        logging_steps=10,
        seed=42,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_eval,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training loop...")
    try:
        trainer.train()
        logger.info("=" * 80)
        logger.info(f"MBART-50 TRAINING COMPLETED - {datetime.now().isoformat()}")
        logger.info("=" * 80)
        
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        final_path = models_dir / "mbart_final"
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Saved model to {final_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

if __name__ == '__main__':
    main()
