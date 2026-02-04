"""
mBART-50 fine-tuning script for Akkadian-English translation.

Fine-tunes facebook/mbart-large-50 pretrained model on Akkadian data.

Features:
- Mixed precision training (fp16)
- Gradient accumulation
- Early stopping
- Model checkpointing

Configuration loaded from configs/model_mbart50.yaml
"""

import logging
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml

try:
    from transformers import (
        MBartForConditionalGeneration,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset, DatasetDict
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MBart50FineTuner:
    """
    Fine-tuner for mBART-50 model on Akkadian-English translation.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'cuda',
    ):
        """
        Initialize fine-tuner.
        
        Args:
            config: Configuration dictionary from YAML
            device: Device to use
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers datasets")
        
        self.config = config
        self.device = device
        
        self.model_name = config['model']['pretrained_model']
        
        # Load pretrained model and tokenizer
        logger.info(f"Loading pretrained model: {self.model_name}")
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set language codes (mBART-50 uses specific codes)
        # Akkadian: 'ak_AF' (closest available)
        # English: 'en_XX'
        self.source_lang = "ak_AF"  # Akkadian
        self.target_lang = "en_XX"  # English
        
        logger.info(f"Model loaded with {self.model.num_parameters()} parameters")
        logger.info(f"Source language: {self.source_lang}, Target language: {self.target_lang}")
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Preprocess batch for model.
        
        Args:
            examples: Batch of examples with 'transliteration' and 'translation' keys
            
        Returns:
            Processed examples with input_ids, attention_mask, labels
        """
        max_input_length = self.config['model']['max_length']
        max_target_length = self.config['model']['max_length']
        
        inputs = examples.get('transliteration', examples.get('source', []))
        targets = examples.get('translation', examples.get('target', []))
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
        
        # Set labels (set padding tokens to -100 so they're ignored in loss)
        model_inputs['labels'] = labels['input_ids']
        model_inputs['labels'][model_inputs['labels'] == self.tokenizer.pad_token_id] = -100
        
        return model_inputs
    
    def create_dataset(self, df) -> Dataset:
        """
        Create HuggingFace Dataset from DataFrame.
        
        Args:
            df: DataFrame with 'transliteration' and 'translation' columns
            
        Returns:
            Dataset object
        """
        dataset = Dataset.from_dict({
            'transliteration': df['transliteration'].tolist(),
            'translation': df['translation'].tolist(),
        })
        
        # Preprocess
        dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=32,
            remove_columns=['transliteration', 'translation'],
        )
        
        return dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "models/mbart50/",
    ) -> None:
        """
        Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Directory to save model
        """
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['training']['epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            warmup_steps=self.config['training']['scheduler']['warmup_steps'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            logging_steps=100,
            save_steps=1000,
            eval_steps=500 if eval_dataset else None,
            save_total_limit=3,
            fp16=self.config['training']['mixed_precision'] == 'fp16',
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model='eval_loss' if eval_dataset else None,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def generate(
        self,
        texts: list,
        max_length: int = 100,
        num_beams: int = 5,
        early_stopping: bool = True,
    ) -> list:
        """
        Generate translations.
        
        Args:
            texts: List of source texts
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            early_stopping: Whether to use early stopping
            
        Returns:
            List of generated translations
        """
        self.model.eval()
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
            )
        
        # Decode
        translations = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True,
        )
        
        return translations
    
    def save_model(self, path: str) -> None:
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Saved model to {path}")


def load_config(config_path: str = "configs/model_mbart50.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Example usage."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not TRANSFORMERS_AVAILABLE:
        logger.error("transformers library not available. Install with: pip install transformers datasets")
        return
    
    try:
        # Load config
        config = load_config("configs/model_mbart50.yaml")
        
        # Create fine-tuner
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        finetuner = MBart50FineTuner(config, device=device)
        
        logger.info(f"mBART-50 fine-tuner initialized on {device}")
        logger.info(f"Model ready for fine-tuning on Akkadian-English data")
        
        # Example usage would load data and call finetuner.train()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)


if __name__ == '__main__':
    main()
