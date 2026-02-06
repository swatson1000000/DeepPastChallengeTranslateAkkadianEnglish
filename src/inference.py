#!/usr/bin/env python3
"""
Unified inference pipeline for Akkadian-English translation models.

Model trained with optimizations for speed:
  - Sequence length: 128 tokens (reduced from 256)
  - Batch size: 512 (doubled from 256)
  - Mixed precision: fp16 for 2-3x training speedup
  - K-folds: 3 folds (reduced from 5 for faster ensemble)
  - Max epochs: 50 with early stopping at overfitting ratio 2.5x

Priority improvements maintained:
  Priority 1: Dropout (0.3), Weight Decay (1e-4), Gradient Clipping (max_norm=1.0)
  Priority 2: Label Smoothing (0.1), Early stopping at ratio≤2.5x

Supports multiple model variants:
- baseline: Standard Seq2Seq with LSTM and attention
- improved: TIER 1 improvements (optimized architecture)
- tier2: TIER 2 improvements (copy mechanism + lexicon constraints)

Usage:
    python inference.py --model improved --output predictions.csv
    python inference.py --model tier2 --use-copy --use-lexicon
"""

import logging
import torch
import torch.nn as nn
import pandas as pd
import yaml
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Optional
import csv
import heapq
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


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
        indices = [2] + indices + [3]
        
        while len(indices) < max_len:
            indices.append(0)
        
        return torch.tensor(indices[:max_len], dtype=torch.long)
    
    def decode(self, indices):
        """Decode tensor to text."""
        words = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx == 0 or idx == 3:
                break
            if idx == 2:
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
        query: (batch_size, hidden_dim) or (hidden_dim,)
        keys: (batch_size, seq_len, hidden_dim) or (seq_len, hidden_dim)
        """
        query_was_1d = query.dim() == 1
        keys_was_2d = keys.dim() == 2
        
        if query_was_1d:
            query = query.unsqueeze(0)
        if keys_was_2d:
            keys = keys.unsqueeze(0)
        
        query_proj = self.query_proj(query).unsqueeze(1)
        key_proj = self.key_proj(keys)
        
        scores = torch.tanh(query_proj + key_proj)
        scores = self.v(scores).squeeze(-1)
        
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * keys).sum(dim=1)
        
        if query_was_1d:
            context = context.squeeze(0)
            weights = weights.squeeze(0)
        
        return context, weights


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
        decoder_state: (batch_size, hidden_dim) or (hidden_dim,)
        encoder_outputs: (batch_size, seq_len, hidden_dim) or (seq_len, hidden_dim)
        source_tokens: (batch_size, seq_len) or (seq_len,)
        """
        if decoder_state.dim() == 1:
            decoder_state = decoder_state.unsqueeze(0)
        if encoder_outputs.dim() == 2:
            encoder_outputs = encoder_outputs.unsqueeze(0)
        if source_tokens.dim() == 1:
            source_tokens = source_tokens.unsqueeze(0)
        
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
        
        context = (copy_weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)
        combined = torch.cat([decoder_state, context], dim=1)
        copy_prob = torch.sigmoid(self.copy_gate(combined))
        
        return copy_logits, copy_weights, copy_prob


class Seq2SeqInference:
    """Unified inference for Seq2Seq models."""
    
    def __init__(self, model_path: str, device: str = None, model_variant: str = 'improved'):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_variant = model_variant
        self.max_len = 180  # Must match training max_len
        
        logger.info(f"Initializing {model_variant} inference")
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        
        # Enforce 80GB GPU memory cap
        if torch.cuda.is_available():
            GPU_MEMORY_LIMIT_GB = 80
            total_gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            mem_fraction = min(GPU_MEMORY_LIMIT_GB / total_gpu_mem_gb, 0.95)
            torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)
            logger.info(f"GPU memory cap: {GPU_MEMORY_LIMIT_GB} GB ({mem_fraction:.0%} of {total_gpu_mem_gb:.0f} GB)")
        
        # Load models
        self.embedding = None
        self.rnn = None
        self.attention = None
        self.decoder = None
        self.copy_mechanism = None
        self.lexicon_decoder = None
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        
        self.load_models()
        self.load_tokenizers()
    
    def load_models(self):
        """Load model components from checkpoint."""
        logger.info(f"Loading models from {self.model_path}...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Source Embedding
        src_vocab_size = checkpoint['src_embedding']['weight'].shape[0]
        src_embed_dim = checkpoint['src_embedding']['weight'].shape[1]
        self.src_vocab_size = src_vocab_size  # Store for index clamping
        self.embedding = nn.Embedding(src_vocab_size, src_embed_dim)
        self.embedding.load_state_dict(checkpoint['src_embedding'])
        self.embedding = self.embedding.to(self.device).eval()
        logger.info(f"✓ Source embedding loaded (vocab={src_vocab_size}, dim={src_embed_dim})")
        
        # Target Embedding
        tgt_embedding_state = checkpoint['tgt_embedding']
        tgt_vocab_size = tgt_embedding_state['weight'].shape[0]
        self.tgt_vocab_size = tgt_vocab_size  # Store for index clamping
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            tgt_embedding_state['weight'].shape[1]
        )
        self.tgt_embedding.load_state_dict(tgt_embedding_state)
        self.tgt_embedding = self.tgt_embedding.to(self.device).eval()
        logger.info(f"✓ Target embedding loaded (vocab={tgt_vocab_size})")
        
        # LSTM Encoder
        lstm_state = checkpoint['rnn']
        
        # Detect LSTM parameters
        is_lstm = 'weight_ih_l0' in lstm_state
        if not is_lstm:
            raise ValueError("Expected LSTM state dict but not found")
        
        num_layers = max([int(k.split('_l')[-1]) for k in lstm_state.keys() if '_l' in k]) + 1
        hidden_size = lstm_state['weight_hh_l0'].shape[0] // 4  # LSTM hidden size (weight_hh is 4*hidden)
        input_size = lstm_state['weight_ih_l0'].shape[1]
        self.hidden_size = hidden_size  # Store for copy mechanism
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=0.4 if num_layers > 1 else 0)
        self.rnn.load_state_dict(lstm_state)
        self.rnn = self.rnn.to(self.device).eval()
        logger.info(f"✓ LSTM encoder ({num_layers} layers, {hidden_size} hidden) loaded")
        
        # Attention Mechanism
        self.attention = AttentionLayer(hidden_size)
        self.attention.load_state_dict(checkpoint['attention'])
        self.attention = self.attention.to(self.device).eval()
        logger.info("✓ Attention mechanism loaded")
        
        # Output Decoder (may not exist in checkpoint, create with dummy weights)
        if 'decoder' in checkpoint:
            decoder_state = checkpoint['decoder']
            self.decoder = nn.Linear(decoder_state['weight'].shape[1], decoder_state['weight'].shape[0])
            self.decoder.load_state_dict(decoder_state)
        else:
            # Create dummy decoder: hidden_dim * 2 → tgt_vocab_size
            tgt_vocab_size = tgt_embedding_state['weight'].shape[0]
            self.decoder = nn.Linear(hidden_size * 2, tgt_vocab_size)
            logger.info(f"⚠️ Output decoder not in checkpoint, created dummy: {hidden_size * 2} → {tgt_vocab_size}")
        
        self.decoder = self.decoder.to(self.device).eval()
        logger.info(f"✓ Output decoder ready")
        
        # Copy mechanism (TIER 2)
        if 'copy_mechanism' in checkpoint:
            vocab_size = checkpoint['decoder']['weight'].shape[0]
            self.copy_mechanism = CopyMechanism(self.hidden_size, vocab_size)
            self.copy_mechanism.load_state_dict(checkpoint['copy_mechanism'])
            self.copy_mechanism = self.copy_mechanism.to(self.device).eval()
            logger.info("✓ Copy mechanism loaded")
        
        # Lexicon decoder (TIER 2)
        if 'lexicon_decoder' in checkpoint:
            vocab_size = checkpoint['decoder']['weight'].shape[0]
            self.lexicon_decoder = LexiconConstrainedDecoder(vocab_size, None)
            self.lexicon_decoder.load_state_dict(checkpoint['lexicon_decoder'], strict=False)
            self.lexicon_decoder = self.lexicon_decoder.to(self.device).eval()
            logger.info("✓ Lexicon decoder loaded")
    
    def load_tokenizers(self):
        """Build tokenizers from training data."""
        logger.info("Building tokenizers from training data...")
        
        # Navigate to project root from checkpoint path
        # Handles both checkpoints/fold_X/best_model.pt and checkpoints/tier3_best.pt
        model_path_obj = Path(self.model_path)
        # Walk up until we find src/ or configs/ directory to identify project root
        candidate = model_path_obj.parent
        for _ in range(5):
            if (candidate / "src").exists() or (candidate / "configs").exists():
                break
            candidate = candidate.parent
        project_root = candidate
        
        train_path = project_root / "data/processed/train_augmented.csv"
        if not train_path.exists():
            train_path = project_root / "data/processed/train_clean.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        df = pd.read_csv(train_path)
        logger.info(f"Loaded {len(df)} training samples")
        
        self.src_tokenizer = SimpleTokenizer()
        self.tgt_tokenizer = SimpleTokenizer()
        
        self.src_tokenizer.build_vocab(df['transliteration'].values)
        self.tgt_tokenizer.build_vocab(df['translation'].values)
        
        logger.info(f"✓ Source vocab: {len(self.src_tokenizer)} tokens")
        logger.info(f"✓ Target vocab: {len(self.tgt_tokenizer)} tokens")
    
    def greedy_decode(self, encoder_outputs, hidden_state, cell_state=None, src_tokens=None,
                     max_len=180, temperature=0.8, use_copy=False):
        """Greedy decoding with optional copy mechanism."""
        decoded_tokens = [2]  # SOS
        attentions = []
        
        # encoder_outputs shape: (batch, seq_len, hidden) or (seq_len, hidden)
        # Ensure batch dimension
        if encoder_outputs.dim() == 2:
            encoder_outputs = encoder_outputs.unsqueeze(0)  # Add batch dim
        
        batch_size = encoder_outputs.shape[0]
        coverage = torch.zeros(batch_size, encoder_outputs.shape[1], device=self.device)
        
        # Ensure src_tokens for copy mechanism
        if use_copy and self.copy_mechanism and src_tokens is None:
            raise ValueError("src_tokens required for copy mechanism")
        
        with torch.no_grad():
            for step in range(max_len - 1):
                current_token = decoded_tokens[-1]
                
                if current_token == 3:  # EOS
                    break
                
                # Clamp token index to valid range [0, tgt_vocab_size)
                current_token = max(0, min(current_token, self.tgt_vocab_size - 1))
                
                current_embedded = self.tgt_embedding(torch.tensor([current_token], device=self.device))
                
                if isinstance(self.rnn, nn.LSTM):
                    _, (hidden_state, cell_state) = self.rnn(
                        current_embedded.unsqueeze(1), (hidden_state, cell_state)
                    )
                    hidden_vec = hidden_state[-1, :, :]  # Get last layer, all batch
                else:
                    _, hidden_state = self.rnn(current_embedded.unsqueeze(1), hidden_state)
                    hidden_vec = hidden_state[-1, :, :]
                
                # Use first batch element if needed
                if hidden_vec.dim() > 1 and batch_size == 1:
                    hidden_vec = hidden_vec[0]
                
                context, attn_weights = self.attention(hidden_vec, encoder_outputs[0])
                
                try:
                    attentions.append(attn_weights.detach().cpu().numpy())
                except:
                    pass  # Skip if attention computation fails
                
                decoder_input = torch.cat([hidden_vec, context], dim=-1)
                logits = self.decoder(decoder_input)
                
                # Add copy mechanism if enabled
                if use_copy and self.copy_mechanism and src_tokens is not None:
                    copy_logits, copy_weights, copy_prob = self.copy_mechanism(
                        hidden_vec.unsqueeze(0),  # Add batch dimension
                        encoder_outputs.unsqueeze(0),  # Add batch dimension
                        src_tokens.unsqueeze(0),  # Add batch dimension
                        coverage=coverage
                    )
                    coverage = coverage + copy_weights  # Update coverage
                    logits = logits + 0.5 * copy_logits[0]  # Remove batch dimension
                
                logits = logits / temperature
                
                # Prevent repetition
                for prev_token in decoded_tokens[-5:]:  # Check last 5 tokens
                    if prev_token not in [0, 1, 2, 3]:
                        logits[prev_token] -= 10.0
                
                next_token = logits.argmax(-1).item()
                decoded_tokens.append(next_token)
        
        return decoded_tokens, attentions
    
    def beam_search_decode(self, encoder_outputs, hidden_state, cell_state=None, src_tokens=None,
                          use_copy: bool = False, beam_width: int = 5, 
                          max_len: int = 256, temperature: float = 1.0,
                          length_penalty: float = 0.0, coverage_penalty: float = 0.0):
        """
        Beam search decoding with optional copy mechanism.
        
        Args:
            encoder_outputs: Encoder output sequence (seq_len, hidden_dim)
            hidden_state: Initial hidden state (num_layers, hidden_dim)
            cell_state: Initial cell state for LSTM (num_layers, hidden_dim)
            src_tokens: Source token indices for copy mechanism (seq_len,)
            use_copy: Whether to use copy mechanism
            beam_width: Number of beams
            max_len: Maximum sequence length
            temperature: Temperature for logits
            length_penalty: Length normalization factor (0=none, 1=linear, >1=favor longer)
            coverage_penalty: Penalty for repeated attention (0=none, >0=penalize)
        
        Returns:
            best_tokens: Best sequence found (list of token indices)
            attentions: Attention weights per step
        """
        
        # Beam state: list of (log_prob, tokens, hidden, cell, coverage, attention_sum)
        beam = [(0.0, [2], hidden_state, cell_state, 
                torch.zeros(1, encoder_outputs.shape[0], device=self.device), [])]
        
        completed = []  # (log_prob, tokens)
        
        with torch.no_grad():
            for step in range(max_len - 1):
                candidates = []  # (log_prob, tokens, hidden, cell, coverage, attention_sum)
                
                for log_prob, tokens, hid, cell, coverage, attn_sum in beam:
                    if tokens[-1] == 3:  # EOS token
                        completed.append((log_prob, tokens, attn_sum))
                        continue
                    
                    current_token = tokens[-1]
                    # Clamp token index to valid range
                    current_token = max(0, min(current_token, self.tgt_vocab_size - 1))
                    current_embedded = self.tgt_embedding(torch.tensor([current_token], device=self.device))
                    
                    # RNN forward step
                    if isinstance(self.rnn, nn.LSTM):
                        _, (new_hidden, new_cell) = self.rnn(
                            current_embedded.unsqueeze(1), (hid, cell)
                        )
                    else:
                        _, new_hidden = self.rnn(current_embedded.unsqueeze(1), hid)
                        new_cell = cell
                    
                    hidden_vec = new_hidden[-1, 0] if new_hidden.dim() == 3 else new_hidden[-1]
                    context, attn_weights = self.attention(hidden_vec, encoder_outputs)
                    
                    decoder_input = torch.cat([hidden_vec, context], dim=-1)
                    logits = self.decoder(decoder_input)
                    
                    # Add copy mechanism if enabled
                    if use_copy and self.copy_mechanism and src_tokens is not None:
                        copy_logits, copy_weights, copy_prob = self.copy_mechanism(
                            hidden_vec.unsqueeze(0),
                            encoder_outputs.unsqueeze(0),
                            src_tokens.unsqueeze(0),
                            coverage=coverage
                        )
                        logits = logits + 0.5 * copy_logits[0]
                    
                    logits = logits / temperature
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Repetition penalty
                    for prev_token in tokens[-5:]:
                        if prev_token not in [0, 1, 2, 3]:
                            log_probs[prev_token] -= 10.0
                    
                    # Coverage penalty
                    if coverage_penalty > 0:
                        coverage_loss = coverage_penalty * torch.sum(torch.min(coverage, torch.ones_like(coverage)))
                        log_probs = log_probs - coverage_loss / encoder_outputs.shape[0]
                    
                    # Get top-k candidates (limit to vocab size)
                    k = min(beam_width, log_probs.shape[-1])
                    top_log_probs, top_indices = torch.topk(log_probs, k)
                    
                    for next_log_prob, next_token in zip(top_log_probs, top_indices):
                        new_log_prob = log_prob + next_log_prob.item()
                        
                        # Length normalization
                        if length_penalty > 0:
                            new_log_prob = new_log_prob / (len(tokens) + 1) ** length_penalty
                        
                        new_tokens = tokens + [next_token.item()]
                        new_coverage = coverage + attn_weights if use_copy else coverage
                        new_attn = attn_sum + [attn_weights.cpu().numpy()]
                        
                        candidates.append((new_log_prob, new_tokens, new_hidden, new_cell, new_coverage, new_attn))
                
                # Keep top-k candidates
                candidates.sort(reverse=True, key=lambda x: x[0])
                beam = candidates[:beam_width]
                
                if not beam or (completed and len(completed) >= beam_width):
                    break
        
        # Combine completed and active sequences
        all_sequences = completed + [(log_prob, tokens, attn) for log_prob, tokens, _, _, _, attn in beam]
        all_sequences.sort(reverse=True, key=lambda x: x[0])
        
        if all_sequences:
            best_log_prob, best_tokens, best_attn = all_sequences[0]
            return best_tokens, best_attn
        else:
            return [2, 3], []  # Fallback: SOS + EOS

    
    def generate_predictions(self, test_data_path: str, output_path: str = 'predictions.csv',
                           use_copy: bool = False, use_lexicon: bool = False, use_beam_search: bool = False,
                           beam_width: int = 5, max_samples: Optional[int] = None):
        """
        Generate predictions on test set.
        
        Args:
            test_data_path: Path to test CSV
            output_path: Where to save predictions
            use_copy: Whether to use copy mechanism
            use_lexicon: Whether to use lexicon constraints
            use_beam_search: Whether to use beam search (vs greedy)
            beam_width: Beam width for beam search
            max_samples: Max samples to process
        """
        logger.info(f"\nGenerating predictions from {test_data_path}...")
        
        df = pd.read_csv(test_data_path)
        if max_samples:
            df = df.head(max_samples)
        
        logger.info(f"Processing {len(df)} samples...")
        logger.info(f"Decoding strategy: {'Beam Search (width=' + str(beam_width) + ')' if use_beam_search else 'Greedy'}")
        
        predictions = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"  Progress: {idx}/{len(df)}")
            
            akkadian = str(row['transliteration'])
            
            # Encode
            src_tensor = self.src_tokenizer.encode(akkadian)
            src_tensor = src_tensor.unsqueeze(0).to(self.device)
            
            # Clamp all source indices to valid embedding range
            src_tensor = torch.clamp(src_tensor, 0, self.src_vocab_size - 1)
            
            # Encode sequence
            with torch.no_grad():
                embedded = self.embedding(src_tensor)
                if isinstance(self.rnn, nn.LSTM):
                    encoder_outputs, (hidden, cell) = self.rnn(embedded)
                else:
                    encoder_outputs, hidden = self.rnn(embedded)
                    cell = None
            
            # Decode with selected strategy
            if use_beam_search:
                decoded, _ = self.beam_search_decode(
                    encoder_outputs, hidden, cell, 
                    src_tokens=src_tensor, 
                    use_copy=use_copy,
                    beam_width=beam_width,
                    length_penalty=0.0,
                    coverage_penalty=0.0
                )
            else:
                if isinstance(self.rnn, nn.LSTM):
                    decoded, _ = self.greedy_decode(encoder_outputs, hidden, cell, src_tokens=src_tensor, use_copy=use_copy)
                else:
                    decoded, _ = self.greedy_decode(encoder_outputs, hidden, None, src_tokens=src_tensor, use_copy=use_copy)
            
            translation = self.tgt_tokenizer.decode(decoded)
            
            predictions.append({
                'id': row.get('id', idx),
                'transliteration': akkadian,
                'translation': translation
            })
        
        # Save predictions
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'transliteration', 'translation'])
            writer.writeheader()
            writer.writerows(predictions)
        
        logger.info(f"✓ Predictions saved to {output_path}")
        logger.info(f"Sample predictions:")
        for pred in predictions[:3]:
            logger.info(f"  {pred['transliteration'][:50]} -> {pred['translation'][:50]}")


def main():
    parser = argparse.ArgumentParser(description='Generate predictions with Akkadian-English model')
    parser.add_argument('--model', choices=['baseline', 'improved', 'tier2', 'tier3'], default='improved',
                       help='Model variant to use')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--test-data', type=str, default='data/raw/test.csv',
                       help='Test data path')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output file path')
    parser.add_argument('--use-copy', action='store_true', help='Enable copy mechanism')
    parser.add_argument('--use-lexicon', action='store_true', help='Enable lexicon constraints')
    parser.add_argument('--use-beam-search', action='store_true', help='Use beam search instead of greedy')
    parser.add_argument('--beam-width', type=int, default=5, help='Beam width for beam search')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to process')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info(f"INFERENCE SESSION - {datetime.now().isoformat()}")
    logger.info(f"Model: {args.model}")
    logger.info("="*80)
    
    project_root = Path(__file__).parent.parent  # Go up from src/ to project root
    
    # Determine checkpoint path - handle both single model and fold-based ensembles
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Check for fold-based checkpoints first (3 folds from optimized training)
        fold_dir = project_root / "checkpoints"
        fold_checkpoints = sorted([f for f in fold_dir.glob('fold_*/best_model.pt')])[:3]
        
        if fold_checkpoints:
            logger.info(f"Found {len(fold_checkpoints)} fold checkpoints for ensemble")
            checkpoint_path = fold_checkpoints[0]  # Use first fold for single inference
        else:
            checkpoint_path = f"checkpoints/{args.model}_best.pt"
    
    checkpoint_path = project_root / checkpoint_path
    
    try:
        # Initialize inference
        inference = Seq2SeqInference(str(checkpoint_path), model_variant=args.model)
        
        # Generate predictions
        test_data = project_root / args.test_data
        output_file = project_root / args.output
        
        inference.generate_predictions(
            str(test_data),
            str(output_file),
            use_copy=args.use_copy,
            use_lexicon=args.use_lexicon,
            use_beam_search=args.use_beam_search,
            beam_width=args.beam_width,
            max_samples=args.max_samples
        )
        
        logger.info("\n" + "="*80)
        logger.info("✓ INFERENCE COMPLETE")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        raise


if __name__ == '__main__':
    main()
