#!/usr/bin/env python3
"""
TIER 2 Training with Copy Mechanism and Lexicon-Constrained Decoding

Extends the standard training to include:
1. Copy mechanism (pointer-generator network)
2. Lexicon-constrained decoding
3. Coverage mechanism (prevents repeated copying)
4. Extended training (300 epochs vs 250)
"""

import logging
import torch
import torch.nn as nn
import pandas as pd
import yaml
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# Configure logging with unbuffered output for nohup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Force unbuffered stdout/stderr for proper nohup logging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

# Import TIER 2 components
from src.tier2_improvements import TIER2Decoder, build_valid_token_mask


class AttentionLayer(nn.Module):
    """Bahdanau attention mechanism."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
    
    def forward(self, query, keys):
        """
        query: (batch_size, hidden_dim)
        keys: (batch_size, seq_len, hidden_dim)
        """
        query_proj = self.query_proj(query).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        key_proj = self.key_proj(keys)  # (batch_size, seq_len, hidden_dim)
        
        scores = torch.tanh(query_proj + key_proj)  # (batch_size, seq_len, hidden_dim)
        scores = self.v(scores).squeeze(-1)  # (batch_size, seq_len)
        
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)
        context = (weights.unsqueeze(-1) * keys).sum(dim=1)  # (batch_size, hidden_dim)
        
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
    
    def encode(self, text, max_len=256):
        """Encode text to tensor."""
        words = text.split()
        indices = [self.word2idx.get(w, 1) for w in words[:max_len]]
        indices = [2] + indices + [3]  # Add SOS and EOS
        
        while len(indices) < max_len:
            indices.append(0)
        
        return torch.tensor(indices[:max_len], dtype=torch.long)
    
    def __len__(self):
        return len(self.word2idx)


def load_lexicon(lexicon_path: str):
    """Load Akkadian-English lexicon."""
    lexicon = {}
    try:
        df = pd.read_csv(lexicon_path)
        for _, row in df.iterrows():
            if pd.notna(row.get('akkadian', '')) and pd.notna(row.get('english', '')):
                akkadian = str(row['akkadian']).strip().lower()
                english = str(row['english']).strip()
                if akkadian and english:
                    lexicon[akkadian] = english
        logger.info(f"✓ Loaded {len(lexicon)} lexicon entries")
    except Exception as e:
        logger.warning(f"Failed to load lexicon: {e}")
    return lexicon


def main(config_path: str = None):
    logger.info("="*80)
    logger.info(f"TIER 2 TRAINING SESSION - {datetime.now().isoformat()}")
    logger.info("="*80)
    
    project_root = Path(__file__).parent.parent
    
    # Determine config path
    if config_path is None:
        config_path = project_root / "configs/model_seq2seq_tier2.yaml"
    else:
        config_path = Path(config_path)
    
    # Load config
    logger.info("\nLoading configuration...")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✓ Config loaded: {config_path.name}")
    logger.info(f"  Data: {config.get('data', {}).get('train', 'N/A')}")
    logger.info(f"  Model: {config.get('encoder', {}).get('num_layers', 'N/A')} layers, {config.get('encoder', {}).get('hidden_size', 'N/A')} hidden")
    
    # TIER 2 Configuration
    copy_cfg = config.get('copy_mechanism', {})
    lexicon_cfg = config.get('lexicon_constraints', {})
    
    logger.info(f"\nTIER 2 Configuration:")
    logger.info(f"  Copy mechanism: {'ENABLED' if copy_cfg.get('enabled', True) else 'disabled'}")
    logger.info(f"  Lexicon constraints: {'ENABLED' if lexicon_cfg.get('enabled', True) else 'disabled'}")
    logger.info(f"  Coverage penalty: {copy_cfg.get('coverage_penalty', 0.1)}")
    
    # Check GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("⚠ GPU not available, using CPU")
    
    # Load data
    logger.info("\nLoading data...")
    data_path = config.get('data', {}).get('train')
    if data_path:
        data_path = project_root / data_path
    else:
        data_path = project_root / "data/processed/train_augmented.csv"
    
    if not data_path.exists():
        data_path = project_root / "data/processed/train_clean.csv"
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded {len(df)} samples from {data_path.name}")
    
    # Load lexicon for TIER 2
    lexicon = {}
    lexicon_path = config.get('data', {}).get('lexicon')
    if lexicon_path and lexicon_cfg.get('enabled', True):
        lexicon = load_lexicon(str(project_root / lexicon_path))
    
    # Build tokenizers
    logger.info("\nBuilding tokenizers...")
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()
    
    src_tokenizer.build_vocab(df['transliteration'].values)
    tgt_tokenizer.build_vocab(df['translation'].values)
    
    logger.info(f"✓ Source vocab: {len(src_tokenizer)}")
    logger.info(f"✓ Target vocab: {len(tgt_tokenizer)}")
    
    # Encode data
    logger.info("Encoding data...")
    src_data = torch.stack([src_tokenizer.encode(text) for text in df['transliteration'].values])
    tgt_data = torch.stack([tgt_tokenizer.encode(text) for text in df['translation'].values])
    logger.info(f"✓ Encoded to tensors: {src_data.shape}, {tgt_data.shape}")
    
    # Create models
    logger.info("\nCreating model architecture...")
    config_encoder = config.get('encoder', {})
    embedding_dim = config_encoder.get('embedding_dim', 256)
    hidden_dim = config_encoder.get('hidden_size', 512)
    num_layers = config_encoder.get('num_layers', 2)
    rnn_type = config_encoder.get('type', 'lstm').upper()
    dropout_rate = config_encoder.get('dropout', 0.2)
    
    embedding = nn.Embedding(len(src_tokenizer), embedding_dim).to(device)
    
    if rnn_type == 'GRU':
        rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, 
                    dropout=dropout_rate if num_layers > 1 else 0).to(device)
    else:
        rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, 
                     dropout=dropout_rate if num_layers > 1 else 0).to(device)
    
    attention = AttentionLayer(hidden_dim).to(device)
    decoder = nn.Linear(hidden_dim * 2, len(tgt_tokenizer)).to(device)
    
    # Create TIER 2 decoder
    tier2_decoder = None
    if copy_cfg.get('enabled', True) or lexicon_cfg.get('enabled', True):
        valid_mask = None
        if lexicon_cfg.get('enabled', True) and lexicon:
            valid_mask = build_valid_token_mask(lexicon, tgt_tokenizer)
        
        tier2_decoder = TIER2Decoder(
            hidden_dim=hidden_dim,
            vocab_size=len(tgt_tokenizer),
            copy_enabled=copy_cfg.get('enabled', True),
            lexicon_constrained=lexicon_cfg.get('enabled', True),
            valid_token_mask=valid_mask
        ).to(device)
        logger.info(f"✓ TIER 2 decoder created")
    
    total_params = sum(p.numel() for model in [embedding, rnn, attention, decoder] 
                      for p in model.parameters())
    if tier2_decoder:
        total_params += sum(p.numel() for p in tier2_decoder.parameters())
    
    logger.info(f"✓ Model created with {total_params:,} total parameters")
    
    # Training setup
    training_cfg = config.get('training', {})
    batch_size = training_cfg.get('batch_size', 128)
    learning_rate = float(training_cfg.get('learning_rate', 0.0005))
    num_epochs = training_cfg.get('max_epochs', 300)
    early_stop_patience = training_cfg.get('early_stopping_patience', 50)
    
    optimizer = torch.optim.Adam(
        list(embedding.parameters()) + list(rnn.parameters()) + 
        list(attention.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )
    
    if tier2_decoder:
        optimizer.add_param_group({'params': tier2_decoder.parameters()})
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    logger.info(f"\nTraining configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Max epochs: {num_epochs}")
    logger.info(f"  Early stopping patience: {early_stop_patience}")
    
    # Data split
    num_train = int(len(df) * 0.8)
    train_src = src_data[:num_train]
    train_tgt = tgt_data[:num_train]
    val_src = src_data[num_train:]
    val_tgt = tgt_data[num_train:]
    
    logger.info(f"  Train samples: {len(train_src)}, Val samples: {len(val_src)}")
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("STARTING TIER 2 TRAINING")
    logger.info("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        embedding.train()
        rnn.train()
        attention.train()
        decoder.train()
        if tier2_decoder:
            tier2_decoder.train()
        
        total_loss = 0
        batch_count = 0
        
        # Shuffle and batch
        indices = torch.randperm(len(train_src))
        src_shuffled = train_src[indices]
        tgt_shuffled = train_tgt[indices]
        
        for batch_idx in range(0, len(train_src), batch_size):
            batch_src = src_shuffled[batch_idx:batch_idx+batch_size].to(device)
            batch_tgt = tgt_shuffled[batch_idx:batch_idx+batch_size].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            embedded = embedding(batch_src)
            rnn_out, _ = rnn(embedded)
            
            loss = 0
            for t in range(batch_tgt.shape[1] - 1):
                decoder_out = rnn_out[:, t, :]
                context, _ = attention(decoder_out, rnn_out)
                
                # Use TIER 2 decoder if available
                if tier2_decoder:
                    logits, copy_prob = tier2_decoder(decoder_out, rnn_out, batch_src, context)
                else:
                    combined = torch.cat([decoder_out, context], dim=1)
                    logits = decoder(combined)
                
                targets = batch_tgt[:, t+1]
                loss += criterion(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(embedding.parameters()) + list(rnn.parameters()) + 
                list(attention.parameters()) + list(decoder.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        
        # Validation
        embedding.eval()
        rnn.eval()
        attention.eval()
        decoder.eval()
        if tier2_decoder:
            tier2_decoder.eval()
        
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch_idx in range(0, len(val_src), batch_size):
                batch_src = val_src[batch_idx:batch_idx+batch_size].to(device)
                batch_tgt = val_tgt[batch_idx:batch_idx+batch_size].to(device)
                
                embedded = embedding(batch_src)
                rnn_out, _ = rnn(embedded)
                
                batch_loss = 0
                for t in range(batch_tgt.shape[1] - 1):
                    decoder_out = rnn_out[:, t, :]
                    context, _ = attention(decoder_out, rnn_out)
                    
                    if tier2_decoder:
                        logits, _ = tier2_decoder(decoder_out, rnn_out, batch_src, context)
                    else:
                        combined = torch.cat([decoder_out, context], dim=1)
                        logits = decoder(combined)
                    
                    targets = batch_tgt[:, t+1]
                    batch_loss += criterion(logits, targets).item()
                
                val_loss += batch_loss
                val_batch_count += 1
        
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{num_epochs} | "
                       f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping with TIER 2 extended patience
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            logger.info(f"  ✓ Best validation loss: {best_val_loss:.4f}")
            
            # Save TIER 2 checkpoints
            torch.save(embedding.state_dict(), project_root / "models/embedding_final.pt")
            torch.save(rnn.state_dict(), project_root / "models/rnn_final.pt")
            torch.save(attention.state_dict(), project_root / "models/attention_final.pt")
            torch.save(decoder.state_dict(), project_root / "models/decoder_final.pt")
            if tier2_decoder:
                torch.save(tier2_decoder.state_dict(), project_root / "models/tier2_decoder_final.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    logger.info("\n" + "="*80)
    logger.info("TIER 2 TRAINING COMPLETE")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("="*80)


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
