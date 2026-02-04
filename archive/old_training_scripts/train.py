#!/usr/bin/env python3
"""Real training with actual data and model configs - 80 epochs Seq2Seq."""

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
        # Project query and keys
        query_proj = self.query_proj(query).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        key_proj = self.key_proj(keys)  # (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        scores = torch.tanh(query_proj + key_proj)  # (batch_size, seq_len, hidden_dim)
        scores = self.v(scores).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply softmax
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)
        
        # Compute context vector
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
        
        # Pad to max_len
        while len(indices) < max_len:
            indices.append(0)
        
        return torch.tensor(indices[:max_len], dtype=torch.long)
    
    def __len__(self):
        return len(self.word2idx)

def main(config_path: str = None):
    logger.info("="*80)
    logger.info(f"REAL TRAINING SESSION - {datetime.now().isoformat()}")
    logger.info("="*80)
    
    project_root = Path(__file__).parent.parent.parent
    
    # Determine config path
    if config_path is None:
        config_path = project_root / "configs/model_seq2seq_improved.yaml"
        if not config_path.exists():
            config_path = project_root / "configs/model_seq2seq.yaml"
    else:
        config_path = Path(config_path)
    
    # Load config
    logger.info("\nLoading configuration...")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✓ Config loaded: {config_path.name}")
    logger.info(f"  Data: {config.get('data', {}).get('train', 'N/A')}")
    logger.info(f"  Model: {config.get('encoder', {}).get('num_layers', 'N/A')} layers, {config.get('encoder', {}).get('hidden_size', 'N/A')} hidden")
    
    # Check GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("⚠ GPU not available, using CPU")
    
    # Load and preprocess data
    logger.info("\nLoading data...")
    # Use config-specified data path, fallback to train_clean.csv
    data_path = config.get('data', {}).get('train')
    if data_path:
        data_path = project_root / data_path
    else:
        data_path = project_root / "data/processed/train_clean.csv"
    
    if not data_path.exists():
        # Fallback to augmented if original specified but not found
        data_path = project_root / "data/processed/train_augmented.csv"
        if not data_path.exists():
            data_path = project_root / "data/processed/train_clean.csv"
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded {len(df)} samples from {data_path.name}")
    
    # DATA VALIDATION: Check for corruption or numeric tokens
    logger.info("\n✓ Data validation:")
    logger.info(f"  - Min transliteration length: {df['transliteration'].str.len().min()}")
    logger.info(f"  - Max transliteration length: {df['transliteration'].str.len().max()}")
    logger.info(f"  - Min translation length: {df['translation'].str.len().min()}")
    logger.info(f"  - Max translation length: {df['translation'].str.len().max()}")
    
    # Check for suspicious numeric values
    has_numeric = df['translation'].str.contains(r'\d{2}\.\d', regex=True).sum()
    if has_numeric > 0:
        logger.warning(f"  ⚠ Found {has_numeric} rows with decimal numbers in translations!")
        logger.info("  Sample corrupted translations:")
        for idx, row in df[df['translation'].str.contains(r'\d{2}\.\d', regex=True)].head(3).iterrows():
            logger.info(f"    - ID {idx}: {row['translation'][:100]}")
    
    # Check for proper content
    logger.info("\n  Sample transliteration: {}".format(df['transliteration'].iloc[0][:80]))
    logger.info("  Sample translation: {}".format(df['translation'].iloc[0][:80]))
    
    # Build tokenizers
    logger.info("\nBuilding tokenizers...")
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()
    
    src_tokenizer.build_vocab(df['transliteration'].values)
    tgt_tokenizer.build_vocab(df['translation'].values)
    
    # DEBUG: Log first 20 tokens from target vocabulary
    logger.info("\n✓ Source vocab: {} tokens".format(len(src_tokenizer)))
    logger.info("✓ Target vocab: {} tokens".format(len(tgt_tokenizer)))
    first_target_tokens = list(tgt_tokenizer.idx2word.items())[:20]
    logger.info("First 20 target tokens: {}".format(first_target_tokens))
    
    
    # Encode data
    logger.info("\nEncoding data to tensors...")
    max_len = 256
    src_data = torch.stack([src_tokenizer.encode(text, max_len) for text in df['transliteration']])
    tgt_data = torch.stack([tgt_tokenizer.encode(text, max_len) for text in df['translation']])
    logger.info(f"✓ Data shape: {src_data.shape}, {tgt_data.shape}")
    
    # Create model components
    logger.info("\nCreating model...")
    # Use new config structure if available, fallback to old
    encoder_cfg = config.get('encoder', config.get('model', {}).get('encoder', {}))
    embedding_dim = encoder_cfg.get('embedding_dim', config.get('model', {}).get('encoder', {}).get('embedding_size', 384))
    hidden_dim = encoder_cfg.get('hidden_size', config.get('model', {}).get('encoder', {}).get('hidden_size', 768))
    num_layers = encoder_cfg.get('num_layers', config.get('model', {}).get('encoder', {}).get('num_layers', 3))
    dropout_rate = encoder_cfg.get('dropout', 0.3)
    
    # Use GRU for simpler baseline if specified
    use_gru = config.get('training', {}).get('use_gru', False)
    rnn_type = "GRU" if use_gru else "LSTM"
    logger.info(f"✓ Using {rnn_type} encoder")
    logger.info(f"  - Layers: {num_layers}, Hidden: {hidden_dim}, Embedding: {embedding_dim}, Dropout: {dropout_rate}")
    
    embedding = nn.Embedding(len(src_tokenizer), embedding_dim).to(device)
    
    if use_gru:
        rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0).to(device)
    else:
        rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0).to(device)
    
    # Add attention mechanism
    attention = AttentionLayer(hidden_dim).to(device)
    decoder = nn.Linear(hidden_dim * 2, len(tgt_tokenizer)).to(device)  # *2 for [rnn_output, context]
    
    # Properly count all parameters
    models = [embedding, rnn, attention, decoder]
    total_params = sum(p.numel() for model in models for p in model.parameters())
    logger.info(f"✓ Model created with {total_params:,} total parameters")
    logger.info(f"  - Embedding: {len(src_tokenizer)} vocab x {embedding_dim} dim")
    logger.info(f"  - {rnn_type}: {num_layers} layers x {hidden_dim} hidden")
    logger.info(f"  - Attention: Bahdanau mechanism")
    logger.info(f"  - Decoder: {hidden_dim * 2} -> {len(tgt_tokenizer)} vocab (with attention)")
    
    # Training setup
    batch_size = config['training']['batch_size']
    # Get training config with fallbacks
    training_cfg = config.get('training', config.get('training', {}))
    learning_rate = float(training_cfg.get('learning_rate', config.get('training', {}).get('learning_rate', 0.0005)))
    weight_decay = float(training_cfg.get('weight_decay', config.get('training', {}).get('weight_decay', '1e-5')))
    num_epochs = training_cfg.get('max_epochs', training_cfg.get('epochs', 100))
    batch_size = training_cfg.get('batch_size', config.get('training', {}).get('batch_size', 128))
    early_stop_patience = training_cfg.get('early_stopping_patience', config.get('training', {}).get('early_stopping_patience', 25))
    
    optimizer = torch.optim.Adam(
        list(embedding.parameters()) + list(rnn.parameters()) + list(attention.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    grad_accum_steps = training_cfg.get('gradient_accumulation_steps', config.get('training', {}).get('gradient_accumulation_steps', 1))
    
    logger.info(f"\nTraining config:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Early stopping patience: {early_stop_patience} epochs")
    
    # Early stopping setup
    best_val_loss_early_stop = float('inf')
    patience = early_stop_patience
    patience_counter = 0
    
    # Split data into train/validation (80/20 split)
    num_train = int(len(df) * 0.8)
    train_indices = list(range(num_train))
    val_indices = list(range(num_train, len(df)))
    
    train_src = src_data[:num_train]
    train_tgt = tgt_data[:num_train]
    val_src = src_data[num_train:]
    val_tgt = tgt_data[num_train:]
    
    num_train_batches = (len(train_indices) + batch_size - 1) // batch_size
    num_val_batches = (len(val_indices) + batch_size - 1) // batch_size
    
    logger.info(f"  Train samples: {len(train_indices)}, batches: {num_train_batches}")
    logger.info(f"  Val samples: {len(val_indices)}, batches: {num_val_batches}")
    
    # Model checkpointing
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pt"
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    aggressive_annealing_triggered = False
    
    # Learning rate scheduler - cosine annealing (after we know num_train_batches)
    total_steps = num_epochs * num_train_batches
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Plateau-aware scheduler: Reduce LR if validation loss plateaus
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,  # Reduce LR after 5 epochs with no improvement
        min_lr=1e-6
    )
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("STARTING REAL TRAINING")
    logger.info("="*80)
    
    for epoch in range(num_epochs):
        embedding.train()
        rnn.train()
        attention.train()
        decoder.train()
        
        total_loss = 0
        batch_count = 0
        
        # Shuffle training data
        indices = torch.randperm(len(train_src))
        src_shuffled = train_src[indices]
        tgt_shuffled = train_tgt[indices]
        
        # Training batches with gradient accumulation
        for batch_idx in range(num_train_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_src))
            
            src_batch = src_shuffled[start_idx:end_idx].to(device)
            tgt_batch = tgt_shuffled[start_idx:end_idx].to(device)
            
            # Forward pass
            embedded = embedding(src_batch)
            outputs, _ = rnn(embedded)
            
            # Apply attention
            last_hidden = outputs[:, -1, :]  # Use last output as query
            context, attn_weights = attention(last_hidden, outputs)
            
            # Concatenate context with last output
            decoder_input = torch.cat([last_hidden, context], dim=1)
            predictions = decoder(decoder_input).unsqueeze(1).expand(-1, outputs.size(1), -1)
            
            # Loss (scaled by accumulation steps)
            loss = criterion(predictions.reshape(-1, len(tgt_tokenizer)), tgt_batch.reshape(-1))
            loss = loss / grad_accum_steps
            
            # Backward
            loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(embedding.parameters()) + list(rnn.parameters()) + list(attention.parameters()) + list(decoder.parameters()),
                    config['training'].get('max_grad_norm', 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            batch_count += 1
        
        # Compute training loss
        train_loss = total_loss / batch_count
        
        # Validation phase
        embedding.eval()
        rnn.eval()
        attention.eval()
        decoder.eval()
        
        val_loss_total = 0
        val_batch_count = 0
        
        with torch.no_grad():
            indices = torch.randperm(len(val_src))
            src_val_shuffled = val_src[indices]
            tgt_val_shuffled = val_tgt[indices]
            
            for batch_idx in range(num_val_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(val_src))
                
                src_batch = src_val_shuffled[start_idx:end_idx].to(device)
                tgt_batch = tgt_val_shuffled[start_idx:end_idx].to(device)
                
                embedded = embedding(src_batch)
                outputs, _ = rnn(embedded)
                
                # Apply attention
                last_hidden = outputs[:, -1, :]
                context, attn_weights = attention(last_hidden, outputs)
                
                # Concatenate context with last output
                decoder_input = torch.cat([last_hidden, context], dim=1)
                predictions = decoder(decoder_input).unsqueeze(1).expand(-1, outputs.size(1), -1)
                
                loss = criterion(predictions.reshape(-1, len(tgt_tokenizer)), tgt_batch.reshape(-1))
                val_loss_total += loss.item()
                val_batch_count += 1
        
        val_loss = val_loss_total / val_batch_count
        
        # Log results
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # ========== MODEL CHECKPOINTING ==========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            
            # Save best model checkpoint
            torch.save(embedding.state_dict(), checkpoint_dir / "embedding_best.pt")
            torch.save(rnn.state_dict(), checkpoint_dir / "rnn_best.pt")
            torch.save(attention.state_dict(), checkpoint_dir / "attention_best.pt")
            torch.save(decoder.state_dict(), checkpoint_dir / "decoder_best.pt")
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_best.pt")
            
            logger.info(f"  ✓ Model checkpoint saved! Val Loss: {val_loss:.4f}")
            aggressive_annealing_triggered = False  # Reset flag
        else:
            epochs_since_improvement += 1
        
        # Plateau-aware learning rate scheduling
        plateau_scheduler.step(val_loss)
        
        # ========== AGGRESSIVE ANNEALING IF PLATEAU ==========
        # If still no improvement after 5 epochs at current LR, restore best model and anneal more aggressively
        if epochs_since_improvement >= 5 and not aggressive_annealing_triggered:
            logger.info(f"  ⚠ No improvement for {epochs_since_improvement} epochs. Triggering aggressive annealing!")
            logger.info(f"    Restoring best model from checkpoint...")
            
            # Restore best model
            embedding.load_state_dict(torch.load(checkpoint_dir / "embedding_best.pt"))
            rnn.load_state_dict(torch.load(checkpoint_dir / "rnn_best.pt"))
            decoder.load_state_dict(torch.load(checkpoint_dir / "decoder_best.pt"))
            optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer_best.pt"))
            
            # Apply more aggressive learning rate reduction
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = current_lr * 0.1  # Much more aggressive (0.1x instead of 0.5x)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            logger.info(f"    Learning rate reduced from {current_lr:.2e} to {new_lr:.2e}")
            logger.info(f"    Resetting early stopping counter to give aggressive annealing a fair chance")
            
            aggressive_annealing_triggered = True
            epochs_since_improvement = 0  # Reset counter
            patience_counter = 0  # IMPORTANT: Reset early stopping counter too!
        
        # Early stopping check (but not before epoch 100)
        if val_loss < best_val_loss_early_stop:
            best_val_loss_early_stop = val_loss
            patience_counter = 0
            logger.info(f"  ✓ Validation loss improved to {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter % 5 == 0 or patience_counter == 1:
                logger.info(f"  ⚠ No improvement for {patience_counter}/{patience} epochs (best: {best_val_loss_early_stop:.4f})")
            
            if patience_counter >= patience and epoch >= 99:  # Allow at least 100 epochs (epoch is 0-indexed)
                logger.info(f"\n{'='*80}")
                logger.info(f"EARLY STOPPING TRIGGERED")
                logger.info(f"No validation loss improvement for {patience} epochs")
                logger.info(f"Best validation loss: {best_val_loss_early_stop:.4f} at epoch {epoch + 1 - patience}")
                logger.info(f"{'='*80}\n")
                break
        
        # Check gradients periodically
        if (epoch + 1) % 10 == 0:
            grads = []
            for model in [embedding, rnn, decoder]:
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.abs().mean().item())
            if grads:
                logger.info(f"  Gradient check: avg={sum(grads)/len(grads):.6f}")
    
    logger.info("\n" + "="*80)
    logger.info(f"TRAINING COMPLETE - {datetime.now().isoformat()}")
    logger.info("="*80)
    logger.info("\n✓ Real training finished successfully!")
    logger.info(f"  - {num_epochs} epochs completed")
    logger.info(f"  - {len(train_indices)} training samples")
    logger.info(f"  - {len(val_indices)} validation samples")
    logger.info(f"  - Final train loss: {train_loss:.4f}")
    logger.info(f"  - Final val loss: {val_loss:.4f}")
    logger.info(f"  - Best val loss: {best_val_loss:.4f}")
    
    # Load best model and save as final
    logger.info("\n✓ Loading best model checkpoint...")
    embedding.load_state_dict(torch.load(checkpoint_dir / "embedding_best.pt"))
    rnn.load_state_dict(torch.load(checkpoint_dir / "rnn_best.pt"))
    attention.load_state_dict(torch.load(checkpoint_dir / "attention_best.pt"))
    decoder.load_state_dict(torch.load(checkpoint_dir / "decoder_best.pt"))
    
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embedding.state_dict(), models_dir / "embedding_final.pt")
    torch.save(rnn.state_dict(), models_dir / "rnn_final.pt")
    torch.save(attention.state_dict(), models_dir / "attention_final.pt")
    torch.save(decoder.state_dict(), models_dir / "decoder_final.pt")
    logger.info(f"✓ Best model saved to {models_dir}")
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
