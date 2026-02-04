#!/usr/bin/env python3
"""Inference script for trained Seq2Seq model."""

import logging
import torch
import torch.nn as nn
import pandas as pd
import yaml
from pathlib import Path
from collections import Counter
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTokenizer:
    """Simple word-level tokenizer."""
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
    
    def encode(self, text, max_len=256):
        """Encode text to tensor."""
        words = text.split()
        indices = [self.word2idx.get(w, 1) for w in words[:max_len]]
        indices = [2] + indices + [3]  # Add SOS and EOS
        
        # Pad to max_len
        while len(indices) < max_len:
            indices.append(0)
        
        return torch.tensor(indices[:max_len], dtype=torch.long)
    
    def decode(self, indices):
        """Decode tensor to text."""
        words = []
        for idx in indices:
            if idx == 0:  # PAD
                continue
            if idx == 3:  # EOS
                break
            if idx == 2:  # SOS
                continue
            word = self.idx2word.get(idx, '<UNK>')
            words.append(word)
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)

def main():
    logger.info("="*80)
    logger.info(f"INFERENCE - {datetime.now().isoformat()}")
    logger.info("="*80)
    
    project_root = Path(__file__).parent.parent.parent
    
    # Load config
    logger.info("\nLoading configuration...")
    with open(project_root / "configs/model_seq2seq.yaml") as f:
        config = yaml.safe_load(f)
    logger.info(f"✓ Config loaded: {config['model']['name']}")
    
    # Check GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("⚠ GPU not available, using CPU")
    
    # Load data to build tokenizers
    logger.info("\nLoading data...")
    df = pd.read_csv(project_root / "data/processed/train_clean.csv")
    logger.info(f"✓ Loaded {len(df)} samples")
    
    # Build tokenizers (same as training)
    logger.info("\nBuilding tokenizers...")
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()
    
    src_tokenizer.build_vocab(df['transliteration'].values)
    tgt_tokenizer.build_vocab(df['translation'].values)
    
    logger.info(f"✓ Source vocab: {len(src_tokenizer)} tokens")
    logger.info(f"✓ Target vocab: {len(tgt_tokenizer)} tokens")
    
    # Create model
    logger.info("\nCreating model...")
    embedding_dim = config['model']['encoder']['embedding_size']
    hidden_dim = config['model']['encoder']['hidden_size']
    num_layers = config['model']['encoder']['num_layers']
    
    embedding = nn.Embedding(len(src_tokenizer), embedding_dim).to(device)
    lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2).to(device)
    decoder = nn.Linear(hidden_dim, len(tgt_tokenizer)).to(device)
    
    total_params = sum(p.numel() for model in [embedding, lstm, decoder] for p in model.parameters())
    logger.info(f"✓ Model created with {total_params:,} parameters")
    
    # Set to eval mode
    embedding.eval()
    lstm.eval()
    decoder.eval()
    
    # Test samples - select random ones from dataset
    logger.info("\n" + "="*80)
    logger.info("INFERENCE ON TEST SAMPLES")
    logger.info("="*80)
    
    # Get random test samples
    test_indices = [0, 50, 100, 200, 500, 1000, 1500]
    
    with torch.no_grad():
        for idx in test_indices:
            if idx >= len(df):
                continue
            
            src_text = df['transliteration'].iloc[idx]
            tgt_text = df['translation'].iloc[idx]
            
            # Encode source
            src_encoded = src_tokenizer.encode(src_text).unsqueeze(0).to(device)
            
            # Forward pass
            embedded = embedding(src_encoded)
            outputs, _ = lstm(embedded)
            logits = decoder(outputs)
            
            # Get predictions (greedy decoding)
            predictions = torch.argmax(logits, dim=-1).squeeze(0)
            pred_text = tgt_tokenizer.decode(predictions.cpu().numpy())
            
            logger.info(f"\n--- Sample {idx} ---")
            logger.info(f"Input (Akkadian):  {src_text[:70]}...")
            logger.info(f"Target (English):  {tgt_text[:70]}...")
            logger.info(f"Predicted Output:  {pred_text[:70]}...")
    
    logger.info("\n" + "="*80)
    logger.info(f"INFERENCE COMPLETE - {datetime.now().isoformat()}")
    logger.info("="*80)
    logger.info("\n✓ Inference finished successfully!")

if __name__ == "__main__":
    main()
