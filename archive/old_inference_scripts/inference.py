#!/usr/bin/env python3
"""
Complete inference pipeline with full neural network decoding.
Loads trained Seq2Seq model and generates English translations.
"""

import logging
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import csv
import sys
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        # Store original shapes to restore later
        query_was_1d = query.dim() == 1
        keys_was_2d = keys.dim() == 2
        
        # Ensure batch dimension
        if query_was_1d:
            query = query.unsqueeze(0)  # (1, hidden_dim)
        if keys_was_2d:
            keys = keys.unsqueeze(0)  # (1, seq_len, hidden_dim)
            
        # Project query and keys
        query_proj = self.query_proj(query).unsqueeze(1)  # (batch, 1, hidden_dim)
        key_proj = self.key_proj(keys)  # (batch, seq_len, hidden_dim)
        
        # Compute attention scores
        scores = torch.tanh(query_proj + key_proj)  # (batch, seq_len, hidden_dim)
        scores = self.v(scores).squeeze(-1)  # (batch, seq_len)
        
        # Apply softmax
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Compute context vector
        context = (weights.unsqueeze(-1) * keys).sum(dim=1)  # (batch, hidden_dim)
        
        # Remove batch dim if input was unbatched
        if query_was_1d:
            context = context.squeeze(0)  # (hidden_dim,)
            weights = weights.squeeze(0)  # (seq_len,)
        
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
        indices = [self.word2idx.get(w, 1) for w in words[:max_len-2]]  # -2 for SOS/EOS
        indices = [2] + indices + [3]  # Add SOS and EOS
        
        # Pad to max_len
        while len(indices) < max_len:
            indices.append(0)
        
        return torch.tensor(indices[:max_len], dtype=torch.long)
    
    def decode(self, indices):
        """Decode tensor to text."""
        words = []
        for idx in indices:
            idx = int(idx)
            if idx == 3:  # EOS
                break
            if idx >= 4:  # Skip special tokens except content
                word = self.idx2word.get(idx, '<UNK>')
                if word != '<UNK>' and word != '<SOS>' and word != '<PAD>':
                    words.append(word)
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)


class Seq2SeqInference:
    def __init__(self, model_dir: str = "models/", device: str = None):
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")
        logger.info(f"Initializing Seq2Seq inference (device={self.device})")
        
        # Initialize models
        self.embedding = None
        self.rnn = None
        self.attention = None
        self.decoder = None
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        
        self.load_models()
        self.load_tokenizers()
    
    def load_models(self):
        """Load all model components from saved files."""
        logger.info(f"Loading model components from {self.model_dir}")
        
        try:
            # Load state dicts
            embedding_state = torch.load(
                self.model_dir / "embedding_final.pt",
                map_location=self.device
            )
            rnn_state = torch.load(
                self.model_dir / "rnn_final.pt",
                map_location=self.device
            )
            attention_state = torch.load(
                self.model_dir / "attention_final.pt",
                map_location=self.device
            )
            decoder_state = torch.load(
                self.model_dir / "decoder_final.pt",
                map_location=self.device
            )
            
            logger.info("✓ Loaded all model state dicts")
            
            # Extract dimensions
            src_vocab_size = embedding_state['weight'].shape[0]
            embedding_dim = embedding_state['weight'].shape[1]
            output_vocab_size = decoder_state['weight'].shape[0]
            hidden_dim = decoder_state['weight'].shape[1] // 2  # Divided by 2 because concatenated
            
            logger.info(f"Model dimensions:")
            logger.info(f"  - Source vocab: {src_vocab_size}")
            logger.info(f"  - Embedding dim: {embedding_dim}")
            logger.info(f"  - Hidden dim: {hidden_dim}")
            logger.info(f"  - Output vocab: {output_vocab_size}")
            
            # Detect number of LSTM layers from state dict keys
            rnn_keys = list(rnn_state.keys())
            num_layers = max([int(k.split('_l')[1][0]) for k in rnn_keys if '_l' in k]) + 1
            logger.info(f"  - Detected LSTM layers: {num_layers}")
            
            # Create and load model layers
            self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
            self.embedding.load_state_dict(embedding_state)
            self.embedding = self.embedding.to(self.device)
            self.embedding.eval()
            logger.info("✓ Embedding loaded")
            
            # Create RNN (LSTM) with detected number of layers
            dropout = 0.2 if num_layers > 1 else 0
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.rnn.load_state_dict(rnn_state)
            self.rnn = self.rnn.to(self.device)
            self.rnn.eval()
            logger.info(f"✓ RNN loaded ({num_layers} layers LSTM)")
            
            # Create attention
            self.attention = AttentionLayer(hidden_dim)
            self.attention.load_state_dict(attention_state)
            self.attention = self.attention.to(self.device)
            self.attention.eval()
            logger.info("✓ Attention mechanism loaded")
            
            # Create decoder
            self.decoder = nn.Linear(hidden_dim * 2, output_vocab_size)
            self.decoder.load_state_dict(decoder_state)
            self.decoder = self.decoder.to(self.device)
            self.decoder.eval()
            logger.info("✓ Decoder loaded")
            
            logger.info("✓ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def load_tokenizers(self):
        """Rebuild tokenizers from training data."""
        logger.info("Building tokenizers from training data...")
        
        try:
            # Get project root - src/inference_complete.py -> src -> project_root
            project_root = Path(__file__).parent.parent
            
            # Try augmented data first, fallback to clean data
            train_path = project_root / "data/processed/train_augmented.csv"
            if not train_path.exists():
                train_path = project_root / "data/processed/train_clean.csv"
            
            if not train_path.exists():
                logger.error(f"Training data not found at {train_path}")
                raise FileNotFoundError(f"Training data not found: {train_path}")
            
            # Load training data
            df = pd.read_csv(train_path)
            logger.info(f"Loaded {len(df)} training samples from {train_path.name}")
            
            # Build tokenizers
            self.src_tokenizer = SimpleTokenizer()
            self.tgt_tokenizer = SimpleTokenizer()
            
            self.src_tokenizer.build_vocab(df['transliteration'].values)
            self.tgt_tokenizer.build_vocab(df['translation'].values)
            
            logger.info(f"✓ Source vocab: {len(self.src_tokenizer)} tokens")
            logger.info(f"✓ Target vocab: {len(self.tgt_tokenizer)} tokens")
            
        except Exception as e:
            logger.error(f"Error loading tokenizers: {e}")
            raise
    
    def encode_sequence(self, text, max_len=256):
        """Encode Akkadian text to tensor."""
        return self.src_tokenizer.encode(text, max_len)
    
    def greedy_decode(self, encoder_outputs, hidden_state, max_len=256, temperature=0.8):
        """
        Greedy decoding with attention, no-repeat tokens, and diversity.
        
        Args:
            encoder_outputs: (seq_len, hidden_dim) - all encoder outputs
            hidden_state: (hidden_dim,) - final encoder hidden state
            max_len: maximum output length
            temperature: softmax temperature for diversity (0.8 reduces prob peaks)
        
        Returns:
            decoded_tokens: list of token indices
            attentions: list of attention weights
        """
        decoded_tokens = [2]  # Start with SOS token
        attentions = []
        current_hidden = hidden_state
        
        # Track tokens to avoid excessive repetition
        token_counts = {}
        REPEAT_LIMIT = 2  # Allow tokens to appear up to 2 times
        
        with torch.no_grad():
            for step in range(max_len - 1):
                # Get current token
                current_token_idx = decoded_tokens[-1]
                
                # If EOS, stop
                if current_token_idx == 3:
                    break
                
                # Apply attention
                context, attn_weights = self.attention(current_hidden, encoder_outputs)
                attentions.append(attn_weights.cpu().numpy())
                
                # Concatenate hidden state with context
                decoder_input = torch.cat([current_hidden, context], dim=-1)
                
                # Get logits
                logits = self.decoder(decoder_input)
                
                # Apply temperature for diversity
                logits = logits / temperature
                
                # Penalize overly repeated tokens
                for token_id, count in token_counts.items():
                    if count > REPEAT_LIMIT:
                        logits[token_id] -= (count - REPEAT_LIMIT) * 2.0  # Soft penalty
                
                # Get next token via greedy selection
                next_token = logits.argmax(dim=0).item()
                
                decoded_tokens.append(next_token)
                token_counts[next_token] = token_counts.get(next_token, 0) + 1
                current_hidden = context
        
        return decoded_tokens, attentions
    
    def beam_search_decode(self, encoder_outputs, hidden_state, beam_width=3, max_len=256, length_penalty=0.6):
        """
        Beam search decoding with attention and length normalization.
        
        Args:
            encoder_outputs: (seq_len, hidden_dim)
            hidden_state: (hidden_dim,)
            beam_width: number of beams to maintain
            max_len: maximum output length
            length_penalty: penalty factor for beam normalization (0.6 works well)
        
        Returns:
            best_sequence: list of token indices
        """
        # Initialize beams: (sequence, log_score, hidden_state, token_counts)
        beams = [([2], 0.0, hidden_state, {})]  # SOS token
        finished_sequences = []  # (sequence, score)
        
        REPEAT_LIMIT = 2
        
        with torch.no_grad():
            for step in range(max_len - 1):
                candidates = []
                
                for sequence, score, current_hidden, token_counts in beams:
                    # Skip if sequence ended
                    if sequence[-1] == 3:  # EOS
                        finished_sequences.append((sequence, score))
                        continue
                    
                    # Apply attention
                    context, _ = self.attention(current_hidden, encoder_outputs)
                    
                    # Decoder forward pass
                    decoder_input = torch.cat([current_hidden, context], dim=-1)
                    logits = self.decoder(decoder_input)
                    
                    # Penalize overly repeated tokens
                    for token_id, count in token_counts.items():
                        if count > REPEAT_LIMIT:
                            logits[token_id] -= (count - REPEAT_LIMIT) * 1.5
                    
                    # Get log probabilities
                    log_probs = torch.log_softmax(logits, dim=0)
                    top_k_probs, top_k_indices = torch.topk(log_probs, min(beam_width * 2, logits.shape[0]))
                    
                    for prob, token_idx in zip(top_k_probs, top_k_indices):
                        token_idx_int = token_idx.item()
                        new_sequence = sequence + [token_idx_int]
                        new_score = score + prob.item()
                        new_token_counts = token_counts.copy()
                        new_token_counts[token_idx_int] = new_token_counts.get(token_idx_int, 0) + 1
                        candidates.append((new_sequence, new_score, context, new_token_counts))
                
                # Separate finished and ongoing sequences
                ongoing = [c for c in candidates if c[0][-1] != 3]
                finished = [c for c in candidates if c[0][-1] == 3]
                finished_sequences.extend([(c[0], c[1]) for c in finished])
                
                # Sort by score and keep top beam_width
                ongoing.sort(key=lambda x: x[1], reverse=True)
                beams = ongoing[:beam_width]
                
                # Early stopping if all beams ended
                if not beams:
                    break
        
        # Combine finished and ongoing sequences
        all_sequences = finished_sequences + [(b[0], b[1]) for b in beams]
        
        # Apply length normalization
        normalized_scores = []
        for seq, score in all_sequences:
            normalized_score = score / ((len(seq)) ** length_penalty) if len(seq) > 0 else score
            normalized_scores.append((seq, normalized_score))
        
        # Return best sequence
        best_seq, _ = max(normalized_scores, key=lambda x: x[1])
        return best_seq
    
    def infer(self, text, use_beam_search=False, beam_width=3):
        """
        Full inference pipeline: tokenize → encode → forward pass → decode.
        
        Args:
            text: Akkadian transliterated text
            use_beam_search: whether to use beam search or greedy decoding
            beam_width: beam width for beam search
        
        Returns:
            translation: English translation string
        """
        try:
            # Encode input
            src_tensor = self.encode_sequence(text)
            src_tensor = src_tensor.unsqueeze(0).to(self.device)  # (1, seq_len)
            
            with torch.no_grad():
                # Embedding
                embedded = self.embedding(src_tensor)  # (1, seq_len, embedding_dim)
                
                # Encoder (LSTM)
                encoder_outputs, (hidden, cell) = self.rnn(embedded)  # (1, seq_len, hidden_dim)
                
                # Get final hidden state (from last layer, last timestep)
                encoder_outputs = encoder_outputs.squeeze(0)  # (seq_len, hidden_dim)
                hidden = hidden[-1].squeeze(0)  # (hidden_dim,) - last layer
                
                # Decode
                if use_beam_search:
                    decoded_tokens = self.beam_search_decode(encoder_outputs, hidden, beam_width)
                else:
                    decoded_tokens, _ = self.greedy_decode(encoder_outputs, hidden)
                
                # Convert tokens to text
                translation = self.tgt_tokenizer.decode(decoded_tokens)
                
                return translation
        
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return f"[Error: {str(e)[:50]}...]"
    
    def infer_batch(self, texts, use_beam_search=False):
        """
        Infer on multiple texts.
        
        Args:
            texts: list of Akkadian texts
            use_beam_search: whether to use beam search
        
        Returns:
            translations: list of English translations
        """
        translations = []
        for i, text in enumerate(texts):
            if (i + 1) % max(1, len(texts) // 10) == 0:
                logger.info(f"  Processed {i+1}/{len(texts)} samples")
            
            translation = self.infer(text, use_beam_search=use_beam_search)
            translations.append(translation)
        
        return translations


def main():
    """Run inference on test data and generate predictions."""
    
    logger.info("="*80)
    logger.info("SEQ2SEQ INFERENCE - COMPLETE PIPELINE")
    logger.info("="*80)
    
    # Setup paths - src/inference_complete.py -> src -> project_root
    project_root = Path(__file__).parent.parent
    test_path = project_root / "data/raw/test.csv"
    output_path = project_root / "predictions.csv"
    
    # Check test data
    if not test_path.exists():
        logger.error(f"Test data not found at {test_path}")
        sys.exit(1)
    
    # Load test data
    logger.info(f"\nLoading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    logger.info(f"✓ Loaded {len(test_df)} test samples")
    
    # Initialize inference pipeline
    logger.info("\nInitializing inference pipeline...")
    pipeline = Seq2SeqInference(model_dir=str(project_root / "models"))
    
    # Run inference
    logger.info(f"\nRunning inference on {len(test_df)} samples...")
    logger.info("(Using greedy decoding with temperature=0.8 and repetition penalty)")
    
    translations = pipeline.infer_batch(
        test_df['transliteration'].values,
        use_beam_search=False
    )
    
    logger.info(f"✓ Inference complete")
    
    # Save predictions
    logger.info(f"\nSaving predictions to {output_path}")
    output_df = pd.DataFrame({
        'id': test_df['id'],
        'translation': translations
    })
    output_df.to_csv(output_path, index=False)
    logger.info(f"✓ Predictions saved: {len(output_df)} rows")
    
    # Show sample predictions
    logger.info("\nSample predictions:")
    for i in range(min(3, len(output_df))):
        logger.info(f"  ID {output_df.iloc[i]['id']}: {output_df.iloc[i]['translation'][:100]}")
    
    logger.info("\n" + "="*80)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
