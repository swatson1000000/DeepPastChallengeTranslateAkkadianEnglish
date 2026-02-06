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


class AttentionLayer(nn.Module):
    """Bahdanau attention mechanism."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
    
    def forward(self, query, keys):
        if query.dim() == 1:
            query = query.unsqueeze(0)
        if keys.dim() == 2:
            keys = keys.unsqueeze(0)
        query_proj = self.query_proj(query).unsqueeze(1)
        key_proj = self.key_proj(keys)
        scores = torch.tanh(query_proj + key_proj)
        scores = self.v(scores).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * keys).sum(dim=1)
        return context, weights


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
    
    def encode(self, text, max_len=180):
        """Encode text to tensor."""
        words = text.split()
        indices = [self.word2idx.get(w, 1) for w in words[:max_len-2]]
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
    config_path = project_root / "configs/model_seq2seq_tier3.yaml"
    if not config_path.exists():
        config_path = project_root / "configs/model_seq2seq.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"✓ Config loaded from {config_path.name}")
    
    # Check GPU
    GPU_MEMORY_LIMIT_GB = 80
    if torch.cuda.is_available():
        device = torch.device("cuda")
        total_gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        mem_fraction = min(GPU_MEMORY_LIMIT_GB / total_gpu_mem_gb, 0.95)
        torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)
        logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory cap: {GPU_MEMORY_LIMIT_GB} GB ({mem_fraction:.0%} of {total_gpu_mem_gb:.0f} GB)")
    else:
        device = torch.device("cpu")
        logger.warning("⚠ GPU not available, using CPU")
    
    # Load data to build tokenizers
    logger.info("\nLoading data...")
    train_path = project_root / "data/processed/train_augmented.csv"
    if not train_path.exists():
        train_path = project_root / "data/processed/train_clean.csv"
    df = pd.read_csv(train_path)
    logger.info(f"✓ Loaded {len(df)} samples")
    
    # Build tokenizers (same as training)
    logger.info("\nBuilding tokenizers...")
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()
    
    src_tokenizer.build_vocab(df['transliteration'].values)
    tgt_tokenizer.build_vocab(df['translation'].values)
    
    logger.info(f"✓ Source vocab: {len(src_tokenizer)} tokens")
    logger.info(f"✓ Target vocab: {len(tgt_tokenizer)} tokens")
    
    # Create model — read architecture from config
    logger.info("\nCreating model...")
    encoder_cfg = config.get('encoder', config.get('model', {}).get('encoder', {}))
    embedding_dim = encoder_cfg.get('embedding_dim', encoder_cfg.get('embedding_size', 512))
    hidden_dim = encoder_cfg.get('hidden_size', 1024)
    num_layers = encoder_cfg.get('num_layers', 4)
    dropout_rate = encoder_cfg.get('dropout', 0.3)
    
    embedding = nn.Embedding(len(src_tokenizer), embedding_dim).to(device)
    tgt_embedding = nn.Embedding(len(tgt_tokenizer), embedding_dim).to(device)
    encoder_rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                  dropout=dropout_rate if num_layers > 1 else 0).to(device)
    decoder_rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                  dropout=dropout_rate if num_layers > 1 else 0).to(device)
    attention = AttentionLayer(hidden_dim).to(device)
    decoder = nn.Linear(hidden_dim * 2, len(tgt_tokenizer)).to(device)
    
    total_params = sum(p.numel() for model in [embedding, tgt_embedding, encoder_rnn, decoder_rnn, attention, decoder] for p in model.parameters())
    logger.info(f"✓ Model created with {total_params:,} parameters")
    logger.info(f"  - Embedding: {embedding_dim} dim")
    logger.info(f"  - Encoder LSTM: {num_layers} layers x {hidden_dim} hidden")
    logger.info(f"  - Decoder LSTM: {num_layers} layers x {hidden_dim} hidden")
    logger.info(f"  - Decoder linear: {hidden_dim * 2} -> {len(tgt_tokenizer)}")
    
    # Load checkpoint if available
    checkpoint_path = project_root / "checkpoints/tier3_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = project_root / "checkpoints/improved_best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        embedding.load_state_dict(checkpoint['src_embedding'])
        tgt_embedding.load_state_dict(checkpoint['tgt_embedding'])
        encoder_rnn.load_state_dict(checkpoint['rnn'])
        if 'decoder_rnn' in checkpoint:
            decoder_rnn.load_state_dict(checkpoint['decoder_rnn'])
            logger.info(f"✓ Loaded separate decoder RNN from checkpoint")
        else:
            # Older checkpoint with shared RNN — copy encoder weights to decoder
            decoder_rnn.load_state_dict(checkpoint['rnn'])
            logger.info(f"  (Using shared encoder/decoder RNN — older checkpoint)")
        attention.load_state_dict(checkpoint['attention'])
        decoder.load_state_dict(checkpoint['decoder'])
        logger.info(f"✓ Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning("⚠ No checkpoint found, using random weights")
    
    # Set to eval mode
    embedding.eval()
    tgt_embedding.eval()
    encoder_rnn.eval()
    decoder_rnn.eval()
    attention.eval()
    decoder.eval()
    
    # Test samples - select random ones from dataset
    logger.info("\n" + "="*80)
    logger.info("INFERENCE ON TEST SAMPLES")
    logger.info("="*80)
    
    # Get random test samples
    test_indices = [0, 50, 100, 200, 500, 1000, 1500]
    max_len = 180
    
    with torch.no_grad():
        for idx in test_indices:
            if idx >= len(df):
                continue
            
            src_text = df['transliteration'].iloc[idx]
            tgt_text = df['translation'].iloc[idx]
            
            # Encode source
            src_encoded = src_tokenizer.encode(src_text, max_len).unsqueeze(0).to(device)
            
            # Forward pass with attention
            embedded = embedding(src_encoded)
            rnn_out, (hidden, cell) = encoder_rnn(embedded)
            
            # Decode step by step with decoder RNN
            predicted_tokens = []
            input_token = torch.tensor([2], device=device)  # SOS
            for step in range(max_len):
                prev_embedded = tgt_embedding(input_token)
                _, (hidden, cell) = decoder_rnn(prev_embedded.unsqueeze(1), (hidden, cell))
                hidden_vec = hidden[-1]
                context, _ = attention(hidden_vec, rnn_out.squeeze(0))
                decoder_input = torch.cat([hidden_vec, context], dim=-1)
                logits = decoder(decoder_input)
                next_token = logits.argmax(-1).item()
                if next_token == 3:  # EOS
                    break
                predicted_tokens.append(next_token)
                input_token = torch.tensor([next_token], device=device)
            
            pred_text = tgt_tokenizer.decode(predicted_tokens)
            
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
