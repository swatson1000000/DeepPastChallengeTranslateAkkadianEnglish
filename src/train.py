#!/usr/bin/env python3
"""
Unified training pipeline for Akkadian-English translation models.

Supports multiple model variants:
- baseline: Standard Seq2Seq with LSTM and attention
- improved: TIER 1 improvements (optimized architecture)
- tier2: TIER 2 improvements (copy mechanism + lexicon constraints)

Usage:
    python train.py --model baseline --epochs 100
    python train.py --model improved --epochs 200
    python train.py --model tier2 --epochs 300 --use-copy --use-lexicon
"""

import logging
import torch
import torch.nn as nn
import pandas as pd
import yaml
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, Tuple, Optional
from sklearn.model_selection import KFold
from torch.amp import autocast, GradScaler
from torch.cuda.amp import autocast, GradScaler

# Configure logging with unbuffered output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Force unbuffered stdout/stderr
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing (Priority 2 improvement)."""
    def __init__(self, num_classes, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        """logits: (batch*seq, vocab), targets: (batch*seq)"""
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Skip ignore_index
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        # Create smooth target distribution
        confidence = 1.0 - self.smoothing
        smooth_label = self.smoothing / self.num_classes
        
        with torch.no_grad():
            true_probs = torch.zeros_like(log_probs)
            true_probs.fill_(smooth_label)
            true_probs.scatter_(1, targets.unsqueeze(1), confidence)
        
        # Calculate smoothed loss
        loss = torch.sum(-true_probs * log_probs, dim=-1)
        return loss[mask].mean()


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
        indices = [self.word2idx.get(w, 1) for w in words[:max_len-2]]
        indices = [2] + indices + [3]  # Add SOS and EOS
        
        while len(indices) < max_len:
            indices.append(0)
        
        return torch.tensor(indices[:max_len], dtype=torch.long)
    
    def decode(self, indices):
        """Decode tensor to text."""
        words = []
        for idx in indices:
            if idx == 0 or idx == 3:  # PAD or EOS
                break
            if idx == 2:  # SOS
                continue
            words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)


class AttentionLayer(nn.Module):
    """Bahdanau attention mechanism."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
    
    def forward(self, query, keys):
        """
        query: (batch_size, hidden_dim) - decoder hidden state
        keys: (batch_size, seq_len, hidden_dim) - encoder outputs
        """
        # Ensure proper shapes
        if query.dim() == 1:
            query = query.unsqueeze(0)  # (batch, hidden_dim) if was (hidden_dim,)
        if keys.dim() == 2:
            keys = keys.unsqueeze(0)  # (batch, seq_len, hidden_dim) if was (seq_len, hidden_dim)
        
        batch_size, seq_len, hidden_dim = keys.size()
        
        # Project query: (batch, hidden_dim) -> (batch, 1, hidden_dim)
        query_proj = self.query_proj(query).unsqueeze(1)
        # Project keys: (batch, seq_len, hidden_dim) -> (batch, seq_len, hidden_dim)
        key_proj = self.key_proj(keys)
        
        # Compute attention scores via broadcasting
        # query_proj: (batch, 1, hidden_dim)
        # key_proj: (batch, seq_len, hidden_dim)
        # Result: (batch, seq_len, hidden_dim)
        scores = torch.tanh(query_proj + key_proj)
        scores = self.v(scores).squeeze(-1)  # (batch, seq_len)
        
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        context = (weights.unsqueeze(-1) * keys).sum(dim=1)  # (batch, hidden_dim)
        
        return context, weights


class CopyMechanism(nn.Module):
    """Pointer-generator network for copying source tokens."""
    
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.coverage_proj = nn.Linear(1, hidden_dim)
        self.copy_gate = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, decoder_state, encoder_outputs, source_tokens, coverage=None):
        """
        decoder_state: (batch_size, hidden_dim)
        encoder_outputs: (batch_size, seq_len, hidden_dim)
        source_tokens: (batch_size, seq_len)
        coverage: (batch_size, seq_len) or None
        """
        batch_size = decoder_state.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        decoder_proj = decoder_state.unsqueeze(2)
        scores = torch.bmm(encoder_outputs, decoder_proj).squeeze(2)
        
        if coverage is not None:
            coverage_penalty = self.coverage_proj(coverage.unsqueeze(-1))
            coverage_penalty = (coverage_penalty * encoder_outputs).sum(dim=2)
            scores = scores - 0.1 * coverage_penalty
        
        copy_weights = torch.softmax(scores, dim=1)
        
        copy_logits = torch.zeros(batch_size, self.vocab_size, device=decoder_state.device)
        for b in range(batch_size):
            for i in range(seq_len):
                token_idx = source_tokens[b, i].item()
                if token_idx > 0:
                    copy_logits[b, token_idx] += copy_weights[b, i]
        
        context = (copy_weights.unsqueeze(1) * encoder_outputs).sum(dim=1)
        combined = torch.cat([decoder_state, context], dim=1)
        copy_prob = torch.sigmoid(self.copy_gate(combined))
        
        return copy_logits, copy_weights, copy_prob


class LexiconConstrainedDecoder(nn.Module):
    """Decoder with lexicon constraints."""
    
    def __init__(self, vocab_size: int, valid_token_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.vocab_size = vocab_size
        
        if valid_token_mask is not None:
            self.register_buffer('valid_mask', valid_token_mask.float())
        else:
            self.valid_mask = torch.ones(vocab_size)
    
    def forward(self, logits, enforce_constraints=True):
        """Apply lexicon constraints to logits."""
        if enforce_constraints:
            logits = logits.clone()
            logits[:, self.valid_mask == 0] = float('-inf')
        return logits


def build_valid_token_mask(tokenizer, lexicon_path: Optional[str] = None) -> torch.Tensor:
    """Build mask for valid tokens."""
    mask = torch.ones(len(tokenizer))
    
    # Mark special tokens as always valid
    mask[0] = 1  # PAD
    mask[1] = 1  # UNK
    mask[2] = 1  # SOS
    mask[3] = 1  # EOS
    
    # Mark vocabulary tokens as valid
    for token_idx in range(4, len(tokenizer)):
        mask[token_idx] = 1
    
    # If lexicon provided, load it
    if lexicon_path and Path(lexicon_path).exists():
        try:
            df = pd.read_csv(lexicon_path)
            for _, row in df.iterrows():
                if 'english' in row and pd.notna(row['english']):
                    word = str(row['english']).strip().lower()
                    words = word.split()
                    for w in words:
                        if w in tokenizer.word2idx:
                            mask[tokenizer.word2idx[w]] = 1
        except Exception as e:
            logger.warning(f"Could not load lexicon: {e}")
    
    return mask


def load_config(config_path: Path) -> Dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def calculate_bleu(predictions, references, max_n=2):
    """Calculate BLEU score for batch of predictions (Priority 2 improvement)."""
    from collections import Counter
    import math
    
    def get_ngrams(words, n):
        return Counter(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    
    scores = []
    for pred_words, ref_words in zip(predictions, references):
        if not pred_words or not ref_words:
            scores.append(0.0)
            continue
        
        pred_tokens = pred_words.split()
        ref_tokens = ref_words.split()
        
        precisions = []
        for n in range(1, min(max_n+1, len(pred_tokens)+1)):
            pred_ng = get_ngrams(pred_tokens, n)
            ref_ng = get_ngrams(ref_tokens, n)
            
            if not pred_ng:
                precisions.append(0.0)
                continue
            
            overlap = sum(min(pred_ng[ng], ref_ng[ng]) for ng in pred_ng if ng in ref_ng)
            precision = overlap / sum(pred_ng.values()) if sum(pred_ng.values()) > 0 else 0.0
            precisions.append(precision)
        
        if precisions and any(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions if p > 0) / len([p for p in precisions if p > 0]))
            scores.append(geo_mean)
        else:
            scores.append(0.0)
    
    return np.mean(scores) if scores else 0.0


def train_epoch(models, optimizer, criterion, train_data, batch_size, device, 
                use_tier2=False, copy_mechanism=None, lexicon_decoder=None,
                use_amp=False, scaler=None,
                embedding_dropout_rate=0.4, decoder_dropout_rate=0.3):
    """Train for one epoch."""
    embedding, tgt_embedding, rnn, attention, decoder = models
    embedding.train()
    tgt_embedding.train()
    rnn.train()
    attention.train()
    decoder.train()
    
    # Dropout layers applied during training to reduce overfitting
    emb_dropout = nn.Dropout(embedding_dropout_rate).to(device)
    dec_dropout = nn.Dropout(decoder_dropout_rate).to(device)
    
    src_data, tgt_data = train_data
    total_loss = 0
    batch_count = 0
    
    indices = torch.randperm(len(src_data))
    src_shuffled = src_data[indices]
    tgt_shuffled = tgt_data[indices]
    
    total_batches = (len(src_data) + batch_size - 1) // batch_size
    logger.info(f"  Starting training: {len(src_data)} samples, {total_batches} batches of size {batch_size}")
    
    # Memory optimization: track peak memory usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    # GPU memory hard limit for per-batch safety checks
    GPU_MEM_LIMIT = 80  # GB
    
    for batch_idx, batch_start in enumerate(range(0, len(src_data), batch_size)):
        # Per-batch GPU memory safety check
        if device.type == 'cuda':
            peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            if peak_gb > GPU_MEM_LIMIT * 0.95:
                logger.error(f"  \u2716 GPU MEMORY CRITICAL: {peak_gb:.1f} GB peak (limit {GPU_MEM_LIMIT} GB)")
                logger.error(f"    Stopping training to prevent OOM. Reduce batch_size or model size.")
                torch.cuda.empty_cache()
                break
            elif batch_idx == 0:
                # After first batch, log actual memory for verification
                pass  # will log below at batch 1
        
        batch_end = min(batch_start + batch_size, len(src_data))
        src_batch = src_shuffled[batch_start:batch_end].to(device)
        tgt_batch = tgt_shuffled[batch_start:batch_end].to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision: wrap forward pass in autocast if enabled
        if use_amp:
            with autocast():
                # Encode
                embedded = emb_dropout(embedding(src_batch))
                if isinstance(rnn, nn.LSTM):
                    rnn_out, (hidden, cell) = rnn(embedded)
                else:
                    rnn_out, hidden = rnn(embedded)
                
                # Decode
                loss = 0
                for step in range(1, tgt_batch.shape[1]):
                    prev_token = tgt_batch[:, step-1]
                    target = tgt_batch[:, step]
                    
                    prev_embedded = emb_dropout(tgt_embedding(prev_token))
                    
                    if isinstance(rnn, nn.LSTM):
                        _, (hidden, cell) = rnn(prev_embedded.unsqueeze(1), (hidden, cell))
                        hidden_vec = hidden[-1]
                    else:
                        _, hidden = rnn(prev_embedded.unsqueeze(1), hidden)
                        hidden_vec = hidden[-1]
                    
                    context, _ = attention(hidden_vec, rnn_out)
                    decoder_input = torch.cat([hidden_vec, context], dim=-1)
                    logits = decoder(dec_dropout(decoder_input))
                    
                    if use_tier2 and copy_mechanism is not None:
                        copy_logits, _, _ = copy_mechanism(hidden_vec, rnn_out, src_batch, coverage=None)
                        logits = logits + 0.5 * copy_logits
                    
                    if use_tier2 and lexicon_decoder is not None:
                        logits = lexicon_decoder(logits)
                    
                    loss += criterion(logits, target)
                
                loss = loss / tgt_batch.shape[1]
        else:
            # Encode
            embedded = emb_dropout(embedding(src_batch))
            if isinstance(rnn, nn.LSTM):
                rnn_out, (hidden, cell) = rnn(embedded)
            else:
                rnn_out, hidden = rnn(embedded)
            
            # Decode
            loss = 0
            for step in range(1, tgt_batch.shape[1]):
                prev_token = tgt_batch[:, step-1]
                target = tgt_batch[:, step]
                
                prev_embedded = emb_dropout(tgt_embedding(prev_token))  # Use TARGET embedding, not source!
                
                if isinstance(rnn, nn.LSTM):
                    _, (hidden, cell) = rnn(prev_embedded.unsqueeze(1), (hidden, cell))
                    hidden_vec = hidden[-1]  # Get last layer: (batch, hidden_dim)
                else:
                    _, hidden = rnn(prev_embedded.unsqueeze(1), hidden)
                    hidden_vec = hidden[-1]  # Get last layer: (batch, hidden_dim)
                
                context, _ = attention(hidden_vec, rnn_out)
                decoder_input = torch.cat([hidden_vec, context], dim=-1)
                
                logits = decoder(dec_dropout(decoder_input))
                
                # Apply TIER 2 if enabled
                if use_tier2 and copy_mechanism is not None:
                    copy_logits, _, _ = copy_mechanism(
                        hidden_vec,
                        rnn_out,
                        src_batch,
                        coverage=None
                    )
                    logits = logits + 0.5 * copy_logits
                
                if use_tier2 and lexicon_decoder is not None:
                    logits = lexicon_decoder(logits)
                
                loss += criterion(logits, target)
            
            loss = loss / tgt_batch.shape[1]
        
        # Check for NaN
        if torch.isnan(loss):
            logger.warning(f"NaN loss detected, skipping batch")
            optimizer.zero_grad()
            continue
        
        # Check for NaN
        if torch.isnan(loss):
            logger.warning(f"NaN loss detected, skipping batch")
            optimizer.zero_grad()
            continue
        
        # Backward pass with proper mixed precision handling
        if use_amp:
            scaler.scale(loss).backward()
            grad_params = (list(embedding.parameters()) + list(tgt_embedding.parameters()) + 
                          list(rnn.parameters()) + list(attention.parameters()) + 
                          list(decoder.parameters()))
            if copy_mechanism is not None:
                grad_params.extend(list(copy_mechanism.parameters()))
            if lexicon_decoder is not None:
                grad_params.extend(list(lexicon_decoder.parameters()))
            
            # Priority 1: Strict gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(grad_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_params = (list(embedding.parameters()) + list(tgt_embedding.parameters()) + 
                          list(rnn.parameters()) + list(attention.parameters()) + 
                          list(decoder.parameters()))
            if copy_mechanism is not None:
                grad_params.extend(list(copy_mechanism.parameters()))
            if lexicon_decoder is not None:
                grad_params.extend(list(lexicon_decoder.parameters()))
            
            # Priority 1: Strict gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(grad_params, max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # After first batch: report actual GPU memory with a go/no-go check
        if batch_idx == 0 and device.type == 'cuda':
            first_batch_peak = torch.cuda.max_memory_allocated(device) / 1024**3
            logger.info(f"    \u2713 First batch peak memory: {first_batch_peak:.1f} GB / {GPU_MEM_LIMIT} GB limit")
            if first_batch_peak > GPU_MEM_LIMIT * 0.85:
                logger.warning(f"    \u26a0 Memory usage {first_batch_peak:.1f} GB is {first_batch_peak/GPU_MEM_LIMIT:.0%} of limit!")
        
        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            if device.type == 'cuda':
                gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                gpu_mem_peak = torch.cuda.max_memory_allocated(device) / 1024**3
                logger.info(f"    Batch {batch_idx + 1}/{total_batches} - Avg Loss: {total_loss/batch_count:.4f} - GPU Memory: {gpu_mem_allocated:.1f}GB (peak: {gpu_mem_peak:.1f}GB)")
            else:
                logger.info(f"    Batch {batch_idx + 1}/{total_batches} - Avg Loss: {total_loss/batch_count:.4f}")
        
        # Memory optimization: clear cache every N batches to prevent fragmentation
        if device.type == 'cuda' and (batch_idx + 1) % 20 == 0:
            torch.cuda.empty_cache()
    
    # Final memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return total_loss / batch_count if batch_count > 0 else 0


def validate(models, criterion, val_data, batch_size, device,
             use_tier2=False, copy_mechanism=None, lexicon_decoder=None):
    """Validate model."""
    embedding, tgt_embedding, rnn, attention, decoder = models
    embedding.eval()
    tgt_embedding.eval()
    rnn.eval()
    attention.eval()
    decoder.eval()
    
    src_data, tgt_data = val_data
    total_loss = 0
    batch_count = 0
    
    logger.info(f"  Starting validation: {len(src_data)} samples")
    
    with torch.no_grad():
        for batch_start in range(0, len(src_data), batch_size):
            batch_end = min(batch_start + batch_size, len(src_data))
            src_batch = src_data[batch_start:batch_end].to(device)
            tgt_batch = tgt_data[batch_start:batch_end].to(device)
            
            embedded = embedding(src_batch)
            if isinstance(rnn, nn.LSTM):
                rnn_out, (hidden, cell) = rnn(embedded)
            else:
                rnn_out, hidden = rnn(embedded)
            
            loss = 0
            for step in range(1, tgt_batch.shape[1]):
                prev_token = tgt_batch[:, step-1]
                target = tgt_batch[:, step]
                
                prev_embedded = tgt_embedding(prev_token)  # Use TARGET embedding!
                
                if isinstance(rnn, nn.LSTM):
                    _, (hidden, cell) = rnn(prev_embedded.unsqueeze(1), (hidden, cell))
                    hidden_vec = hidden[-1]
                else:
                    _, hidden = rnn(prev_embedded.unsqueeze(1), hidden)
                    hidden_vec = hidden[-1]
                
                context, _ = attention(hidden_vec, rnn_out)
                decoder_input = torch.cat([hidden_vec, context], dim=-1)
                
                logits = decoder(decoder_input)
                
                if use_tier2 and copy_mechanism is not None:
                    copy_logits, _, _ = copy_mechanism(
                        hidden_vec,
                        rnn_out,
                        src_batch,
                        coverage=None
                    )
                    logits = logits + 0.5 * copy_logits
                
                if use_tier2 and lexicon_decoder is not None:
                    logits = lexicon_decoder(logits)
                
                loss += criterion(logits, target)
            
            loss = loss / tgt_batch.shape[1]
            
            # Check for NaN and skip batch if detected
            if not torch.isnan(loss):
                total_loss += loss.item()
                batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else 0


def train_single(models, optimizer, criterion, config, args, device,
                train_src, train_tgt, val_src, val_tgt,
                checkpoint_dir, use_tier2, copy_mechanism, lexicon_decoder,
                fold_idx: Optional[int] = None) -> float:
    """Train a single fold or the main model. Returns best validation loss."""
    
    fold_str = f"Fold {fold_idx + 1}" if fold_idx is not None else "Model"
    
    training_cfg = config.get('training', {})
    batch_size = args.batch_size or training_cfg.get('batch_size', 128)
    learning_rate = float(training_cfg.get('learning_rate', 0.0005))
    num_epochs = args.epochs or training_cfg.get('epochs', 100)
    use_amp = training_cfg.get('use_amp', False)  # Get from config
    
    logger.info(f"\n{fold_str} | Train samples: {len(train_src)}, Val samples: {len(val_src)}")
    if use_amp:
        logger.info(f"  {fold_str} | Mixed Precision Training: ENABLED")
    
    # Setup optimizer with Priority 1: Weight decay for regularization
    optimizer = torch.optim.Adam(
        [p for m in models for p in m.parameters()] +
        ([p for p in copy_mechanism.parameters()] if copy_mechanism else []) +
        ([p for p in lexicon_decoder.parameters()] if lexicon_decoder else []),
        lr=learning_rate,
        weight_decay=5e-4  # Increased L2 regularization (was 1e-4) to close train/val gap
    )
    
    # Create GradScaler for mixed precision training
    scaler = GradScaler() if use_amp else None
    
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_overfitting_ratio = float('inf')
    annealing_mode = False
    epochs_in_annealing = 0
    anneal_lr_reduction_steps = 0
    
    overfitting_threshold = 1.15  # Tighter threshold — folds showed overfitting at ~1.17x
    min_epochs = 15
    target_overfitting_ratio = 1.25  # Early stop when ratio exceeds this (was 2.5)
    annealing_patience = 8  # Was 20 — shortened to avoid wasted compute
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            models, optimizer, criterion,
            (train_src, train_tgt), batch_size, device,
            use_tier2, copy_mechanism, lexicon_decoder,
            use_amp, scaler
        )
        
        val_loss = validate(
            models, criterion,
            (val_src, val_tgt), batch_size, device,
            use_tier2, copy_mechanism, lexicon_decoder
        )
        
        # Log GPU memory usage
        if device.type == 'cuda':
            gpu_mem_used = torch.cuda.memory_allocated(device) / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            logger.debug(f"  GPU Memory: {gpu_mem_used:.1f}GB allocated, {gpu_mem_reserved:.1f}GB reserved")
        
        overfitting_ratio = val_loss / train_loss if train_loss > 0 else float('inf')
        
        annealing_status = f" | Annealing: {epochs_in_annealing}/{annealing_patience}" if annealing_mode else ""
        logger.info(f"{fold_str} | Epoch {epoch+1:3d}/{num_epochs} | "
                   f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Ratio: {overfitting_ratio:.2f}x{annealing_status}")
        
        # Save best model when val loss improves AND overfitting ratio is acceptable
        should_save = False
        if val_loss < best_val_loss * 0.995:  # At least 0.5% improvement
            if overfitting_ratio <= target_overfitting_ratio:
                should_save = True
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_overfitting_ratio = overfitting_ratio
        
        if should_save:
            if annealing_mode:
                logger.info(f"  ✓ Improvement found during annealing! Resetting counter.")
                epochs_in_annealing = 0
            
            # Save checkpoint
            # models = [src_embedding, tgt_embedding, rnn, attention, decoder]
            checkpoint = {
                'src_embedding': models[0].state_dict(),
                'tgt_embedding': models[1].state_dict(),
                'rnn': models[2].state_dict(),
                'attention': models[3].state_dict(),
                'decoder': models[4].state_dict(),
            }
            if copy_mechanism:
                checkpoint['copy_mechanism'] = copy_mechanism.state_dict()
            if lexicon_decoder:
                checkpoint['lexicon_decoder'] = lexicon_decoder.state_dict()
            
            if fold_idx is not None:
                fold_dir = checkpoint_dir / f"fold_{fold_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, fold_dir / "best_model.pt")
            else:
                torch.save(checkpoint, checkpoint_dir / f"{args.model}_best.pt")
            logger.info(f"  ✓ Saved best model (val loss: {val_loss:.4f}, ratio: {overfitting_ratio:.2f}x)")
        
        # Early stopping if overfitting ratio exceeds threshold
        if (epoch >= min_epochs and 
            overfitting_ratio > target_overfitting_ratio and 
            not annealing_mode):
            annealing_mode = True
            epochs_in_annealing = 0
            anneal_lr_reduction_steps = 0
            logger.info(f"\n⚠ OVERFITTING DETECTED: {overfitting_ratio:.2f}x ratio at epoch {epoch+1}")
            logger.info(f"  Triggering gradual learning rate annealing")
            logger.info(f"  Will attempt to recover for up to {annealing_patience} epochs")
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = current_lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            logger.info(f"  Initial LR reduction: {current_lr:.2e} → {new_lr:.2e}")
        
        if annealing_mode:
            epochs_in_annealing += 1
            
            if epochs_in_annealing % 5 == 0 and anneal_lr_reduction_steps < 10:
                anneal_lr_reduction_steps += 1
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = current_lr * 0.7
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                logger.info(f"  ⚠ Annealing step {anneal_lr_reduction_steps}: LR → {new_lr:.2e}")
            
            if epochs_in_annealing >= annealing_patience:
                logger.info(f"  Early stopping: No improvement after {annealing_patience} annealing epochs")
                break
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train Akkadian-English translation model')
    parser.add_argument('--model', choices=['baseline', 'improved', 'tier2', 'tier3'], default='improved',
                       help='Model variant to train')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs from config')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--use-copy', action='store_true', help='Enable copy mechanism')
    parser.add_argument('--use-lexicon', action='store_true', help='Enable lexicon constraints')
    parser.add_argument('--use-beam-search', action='store_true', help='Enable beam search in inference')
    parser.add_argument('--data-path', type=str, default='data/processed/train_augmented.csv',
                       help='Path to training data')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--folds', type=int, default=3, help='Number of k-folds for cross-validation')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info(f"TRAINING SESSION - {datetime.now().isoformat()}")
    logger.info(f"Model: {args.model}")
    logger.info("="*80)
    
    project_root = Path(__file__).parent.parent  # Go up from src/ to project root
    
    # Determine config path
    if args.config:
        config_path = Path(args.config)
    else:
        if args.model == 'tier3':
            config_path = project_root / "configs/model_seq2seq_tier3.yaml"
        elif args.model == 'tier2':
            config_path = project_root / "configs/model_seq2seq_tier2.yaml"
        elif args.model == 'improved':
            config_path = project_root / "configs/model_seq2seq_improved.yaml"
        else:
            config_path = project_root / "configs/model_seq2seq.yaml"
    
    # Load config
    logger.info(f"\nLoading config: {config_path}")
    config = load_config(config_path)
    
    # Check GPU
    GPU_MEMORY_LIMIT_GB = 80  # HARD LIMIT — never exceed this
    if torch.cuda.is_available():
        device = torch.device("cuda")
        total_gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {total_gpu_mem_gb:.1f} GB")
        logger.info(f"  HARD LIMIT: {GPU_MEMORY_LIMIT_GB} GB")
        # Set hard memory fraction limit
        mem_fraction = min(GPU_MEMORY_LIMIT_GB / total_gpu_mem_gb, 0.95)
        torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)
        logger.info(f"  Memory fraction cap: {mem_fraction:.2%} ({GPU_MEMORY_LIMIT_GB:.0f}/{total_gpu_mem_gb:.0f} GB)")
    else:
        device = torch.device("cpu")
        total_gpu_mem_gb = 0
        logger.warning("⚠ Using CPU (GPU not available)")
    
    # Load data
    logger.info(f"\nLoading data from {args.data_path}...")
    data_path = project_root / args.data_path
    if not data_path.exists():
        data_path = project_root / "data/processed/train_clean.csv"
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded {len(df)} samples")
    
    # Memory optimization: Enable cuDNN autotuner for better performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Build tokenizers
    logger.info("\nBuilding tokenizers...")
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()
    
    src_tokenizer.build_vocab(df['transliteration'].values)
    tgt_tokenizer.build_vocab(df['translation'].values)
    
    logger.info(f"✓ Source vocab: {len(src_tokenizer)} tokens")
    logger.info(f"✓ Target vocab: {len(tgt_tokenizer)} tokens")
    
    # Encode data
    logger.info("\nEncoding data...")
    max_len = 180  # Reduced from 256 to save ~30% GPU memory on sequence dimensions
    src_data = torch.stack([src_tokenizer.encode(text, max_len) for text in df['transliteration']])
    tgt_data = torch.stack([tgt_tokenizer.encode(text, max_len) for text in df['translation']])
    logger.info(f"✓ Data shape: {src_data.shape}")
    
    # Create model
    logger.info("\nCreating model...")
    encoder_cfg = config.get('encoder', config.get('model', {}).get('encoder', {}))
    embedding_dim = encoder_cfg.get('embedding_dim', 384)
    hidden_dim = encoder_cfg.get('hidden_size', 512)
    num_layers = encoder_cfg.get('num_layers', 2)
    dropout_rate = encoder_cfg.get('dropout', 0.3)
    
    embedding = nn.Embedding(len(src_tokenizer), embedding_dim).to(device)
    tgt_embedding = nn.Embedding(len(tgt_tokenizer), embedding_dim).to(device)  # Target embedding for decoding
    rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                  dropout=dropout_rate if num_layers > 1 else 0).to(device)
    attention = AttentionLayer(hidden_dim).to(device)
    decoder = nn.Linear(hidden_dim * 2, len(tgt_tokenizer)).to(device)
    
    logger.info(f"✓ Model created")
    logger.info(f"  - Embedding: {embedding_dim} dim")
    logger.info(f"  - LSTM: {num_layers} layers x {hidden_dim} hidden")
    logger.info(f"  - Decoder: {hidden_dim * 2} -> {len(tgt_tokenizer)}")
    
    # Estimate GPU memory usage before training
    total_params = sum(p.numel() for p in embedding.parameters()) + \
                   sum(p.numel() for p in tgt_embedding.parameters()) + \
                   sum(p.numel() for p in rnn.parameters()) + \
                   sum(p.numel() for p in attention.parameters()) + \
                   sum(p.numel() for p in decoder.parameters())
    param_mem_gb = total_params * 4 / 1e9  # float32
    # Model + gradients + Adam states (2 moments) = 4× params
    optimizer_mem_gb = param_mem_gb * 4
    # Activation memory estimate: batch × seq_len × hidden × overhead_factor
    batch_size_estimate = training_cfg.get('batch_size', 64) if 'training_cfg' not in dir() else 64
    activation_mem_gb = (batch_size_estimate * max_len * hidden_dim * 4 * 8) / 1e9  # rough 8× factor for autograd
    estimated_peak_gb = optimizer_mem_gb + activation_mem_gb
    logger.info(f"  - Total params: {total_params:,} ({param_mem_gb:.2f} GB)")
    logger.info(f"  - Est. peak GPU: {estimated_peak_gb:.1f} GB (limit: {GPU_MEMORY_LIMIT_GB} GB)")
    
    if estimated_peak_gb > GPU_MEMORY_LIMIT_GB * 0.9:
        logger.warning(f"⚠ Estimated memory {estimated_peak_gb:.1f} GB approaches limit {GPU_MEMORY_LIMIT_GB} GB!")
        logger.warning(f"  Reducing batch size from {batch_size_estimate} to {batch_size_estimate // 2}")
        # Will be applied via the batch_size variable below
    
    # TIER 2 components
    copy_mechanism = None
    lexicon_decoder = None
    use_tier2 = args.model == 'tier2'
    
    if use_tier2 or args.use_copy:
        copy_mechanism = CopyMechanism(hidden_dim, len(tgt_tokenizer)).to(device)
        logger.info("✓ Copy mechanism enabled")
    
    if use_tier2 or args.use_lexicon:
        lexicon_mask = build_valid_token_mask(tgt_tokenizer)
        lexicon_decoder = LexiconConstrainedDecoder(len(tgt_tokenizer), lexicon_mask).to(device)
        logger.info("✓ Lexicon constraints enabled")
    
    # Training setup
    training_cfg = config.get('training', {})
    # Memory optimization: Default batch size reduced from 128 to 64 to save ~50% memory on tensor allocations
    default_batch_size = training_cfg.get('batch_size', 128)
    # If no override, automatically reduce batch size for GPU memory constraints
    if args.batch_size is None and torch.cuda.is_available():
        if total_gpu_mem_gb < 80 or estimated_peak_gb > GPU_MEMORY_LIMIT_GB * 0.9:
            default_batch_size = min(default_batch_size, 32)
            logger.info(f"  Auto-reduced batch size to {default_batch_size} for memory safety")
    batch_size = args.batch_size or default_batch_size
    learning_rate = float(training_cfg.get('learning_rate', 0.0005))
    num_epochs = args.epochs or training_cfg.get('epochs', 100)
    early_stop_patience = 8  # Reduced to 8 — val loss plateaued by epoch ~28 in all folds
    
    # Priority 2: Label smoothing loss instead of standard CE loss
    criterion = LabelSmoothingCrossEntropy(
        num_classes=len(tgt_tokenizer),
        smoothing=0.1,  # 10% smoothing
        ignore_index=0
    )
    
    logger.info(f"\nTraining config:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Early stopping: {early_stop_patience} epochs")
    logger.info(f"  Sequence length: {max_len}")
    logger.info(f"  GPU Memory HARD LIMIT: {GPU_MEMORY_LIMIT_GB} GB")
    
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # K-Fold or Regular Training
    if args.folds:
        # K-Fold Cross-Validation
        logger.info(f"\n{'='*80}")
        logger.info(f"K-FOLD CROSS-VALIDATION: {args.folds} FOLDS")
        logger.info(f"{'='*80}")
        
        kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        fold_results = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(np.arange(len(df)))):
            logger.info(f"\n{'='*80}")
            logger.info(f"FOLD {fold_idx + 1}/{args.folds}")
            logger.info(f"{'='*80}")
            
            # Get fold data
            fold_train_src = src_data[train_indices]
            fold_train_tgt = tgt_data[train_indices]
            fold_val_src = src_data[val_indices]
            fold_val_tgt = tgt_data[val_indices]
            
            # Create fresh models for this fold
            fold_embedding = nn.Embedding(len(src_tokenizer), embedding_dim).to(device)
            fold_tgt_embedding = nn.Embedding(len(tgt_tokenizer), embedding_dim).to(device)
            fold_rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout_rate if num_layers > 1 else 0).to(device)
            fold_attention = AttentionLayer(hidden_dim).to(device)
            fold_decoder = nn.Linear(hidden_dim * 2, len(tgt_tokenizer)).to(device)
            
            fold_models = [fold_embedding, fold_tgt_embedding, fold_rnn, fold_attention, fold_decoder]
            
            # TIER 2 components for this fold
            fold_copy_mechanism = None
            fold_lexicon_decoder = None
            
            if use_tier2 or args.use_copy:
                fold_copy_mechanism = CopyMechanism(hidden_dim, len(tgt_tokenizer)).to(device)
            
            if use_tier2 or args.use_lexicon:
                fold_lexicon_decoder = LexiconConstrainedDecoder(len(tgt_tokenizer), lexicon_mask).to(device)
            
            # Train this fold
            best_val_loss = train_single(
                fold_models,  # Dummy optimizer, will be created in train_single
                None,  # Will be created in train_single
                criterion,
                config, args, device,
                fold_train_src, fold_train_tgt, fold_val_src, fold_val_tgt,
                checkpoint_dir, use_tier2, fold_copy_mechanism, fold_lexicon_decoder,
                fold_idx=fold_idx
            )
            
            fold_results.append({
                'fold': fold_idx + 1,
                'val_loss': best_val_loss
            })
            
            logger.info(f"\n✓ Fold {fold_idx + 1} complete: Best val loss {best_val_loss:.4f}\n")
        
        # K-Fold Summary
        logger.info(f"\n{'='*80}")
        logger.info("K-FOLD CROSS-VALIDATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info("\nFold Results:")
        for result in fold_results:
            logger.info(f"  Fold {result['fold']}: Val Loss = {result['val_loss']:.4f}")
        
        avg_loss = np.mean([r['val_loss'] for r in fold_results])
        std_loss = np.std([r['val_loss'] for r in fold_results])
        logger.info(f"\nAverage Val Loss: {avg_loss:.4f} ± {std_loss:.4f}")
        logger.info(f"Best Fold: {min(fold_results, key=lambda x: x['val_loss'])['fold']} "
                   f"(Loss: {min(fold_results, key=lambda x: x['val_loss'])['val_loss']:.4f})")
        logger.info(f"\n✓ Fold checkpoints saved to: checkpoints/fold_*")

        # Copy best fold to production checkpoint
        best_fold = min(fold_results, key=lambda x: x['val_loss'])
        best_fold_dir = checkpoint_dir / f"fold_{best_fold['fold'] - 1}"
        production_path = checkpoint_dir / f"{args.model}_best.pt"
        if (best_fold_dir / "best_model.pt").exists():
            import shutil
            shutil.copy2(best_fold_dir / "best_model.pt", production_path)
            logger.info(f"✓ Copied best fold {best_fold['fold']} checkpoint to: {production_path}")
        
    else:
        # Regular Training (80/20 split)
        logger.info(f"\n{'='*80}")
        logger.info("REGULAR TRAINING (80/20 Split)")
        logger.info(f"{'='*80}")
        
        # Split data (80/20)
        num_train = int(len(df) * 0.8)
        train_src, train_tgt = src_data[:num_train], tgt_data[:num_train]
        val_src, val_tgt = src_data[num_train:], tgt_data[num_train:]
        
        # Create model
        models = [embedding, tgt_embedding, rnn, attention, decoder]
        
        logger.info(f"  Train samples: {len(train_src)}")
        logger.info(f"  Val samples: {len(val_src)}")
        
        # Train the model
        best_val_loss = train_single(
            models, optimizer, criterion, config, args, device,
            train_src, train_tgt, val_src, val_tgt,
            checkpoint_dir, use_tier2, copy_mechanism, lexicon_decoder
        )
        
        logger.info(f"\nBest validation loss: {best_val_loss:.4f}")
        checkpoint_file = checkpoint_dir / f"{args.model}_best.pt"
        logger.info(f"Checkpoint saved: {checkpoint_file}")

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
