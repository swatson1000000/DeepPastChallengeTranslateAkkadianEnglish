#!/usr/bin/env python3
"""Real Seq2Seq training with tensor-based data loading."""

import logging
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import pandas as pd
import sys
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from seq2seq import Seq2SeqModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['weight_decay'] = float(eval(str(config['training']['weight_decay'])))
    config['training']['num_epochs'] = config['training'].get('epochs', 80)
    return config

class Tokenizer:
    """Simple character-level tokenizer."""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.next_idx = 4
    
    def build_vocab(self, texts, min_freq=1):
        """Build vocabulary from texts."""
        word_freq = Counter()
        for text in texts:
            words = text.split()
            word_freq.update(words)
        
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.next_idx
                self.idx2word[self.next_idx] = word
                self.next_idx += 1
    
    def encode(self, text, max_len=128):
        """Encode text to indices."""
        words = text.split()[:max_len-2]  # Reserve space for SOS/EOS
        indices = [2]  # SOS
        for word in words:
            indices.append(self.word2idx.get(word, 1))  # UNK = 1
        indices.append(3)  # EOS
        # Pad to max_len
        while len(indices) < max_len:
            indices.append(0)  # PAD
        return torch.tensor(indices[:max_len], dtype=torch.long)
    
    def __len__(self):
        return len(self.word2idx)

def main():
    project_root = Path(__file__).parent.parent.parent
    config = load_config(str(project_root / "configs/model_seq2seq.yaml"))
    logger.info("Loaded configuration")
    
    # Load data
    df = pd.read_csv(str(project_root / "data/processed/train_clean.csv"))
    logger.info(f"Loaded {len(df)} training samples")
    
    # Build tokenizers
    logger.info("Building tokenizers...")
    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()
    
    src_tokenizer.build_vocab(df['transliteration'].tolist(), min_freq=1)
    tgt_tokenizer.build_vocab(df['translation'].tolist(), min_freq=1)
    
    logger.info(f"Source vocab size: {len(src_tokenizer)}")
    logger.info(f"Target vocab size: {len(tgt_tokenizer)}")
    
    # Encode data
    logger.info("Encoding data...")
    src_data = torch.stack([src_tokenizer.encode(text) for text in df['transliteration']])
    tgt_data = torch.stack([tgt_tokenizer.encode(text) for text in df['translation']])
    
    # Create device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = Seq2SeqModel(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        embedding_size=config['model']['encoder']['embedding_size'],
        hidden_size=config['model']['encoder']['hidden_size'],
        num_layers=config['model']['encoder']['num_layers'],
        dropout=config['model']['encoder']['dropout'],
        device=device,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created Seq2Seq model: {total_params:,} parameters")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    logger.info("=" * 80)
    logger.info(f"STARTING SEQ2SEQ TRAINING - {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    # Training loop
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Shuffle indices
        indices = torch.randperm(len(df))
        src_shuffled = src_data[indices]
        tgt_shuffled = tgt_data[indices]
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            
            src_batch = src_shuffled[start_idx:end_idx].to(device)
            tgt_batch = tgt_shuffled[start_idx:end_idx].to(device)
            
            # Forward pass
            outputs = model(src_batch, tgt_batch, teacher_forcing_ratio=0.5)
            
            # Loss (skip SOS token)
            loss = criterion(
                outputs[1:].reshape(-1, outputs.size(-1)),
                tgt_batch[1:].reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % max(1, num_batches // 3) == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = project_root / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / f"seq2seq_epoch_{epoch+1}.pt")
            logger.info(f"Saved checkpoint: epoch {epoch+1}")
    
    logger.info("=" * 80)
    logger.info(f"SEQ2SEQ TRAINING COMPLETED - {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / "seq2seq_final.pt")
    logger.info("Saved final model")

if __name__ == '__main__':
    main()
