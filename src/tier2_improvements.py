#!/usr/bin/env python3
"""
TIER 2 Improvements - Copy Mechanism and Lexicon-Constrained Decoding
Implements pointer-generator network for copying source tokens
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class CopyMechanism(nn.Module):
    """
    Pointer-Generator Network for copying source tokens.
    Allows decoder to copy directly from source input for:
    - Proper nouns (Akkadian names)
    - Numbers
    - Determinatives
    """
    
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Coverage mechanism to avoid copying same token repeatedly
        self.coverage_proj = nn.Linear(1, hidden_dim)
        
        # Copy probability generation
        self.copy_gate = nn.Linear(hidden_dim * 2, 1)  # [context, decoder_state]
        
    def forward(self, decoder_state, encoder_outputs, source_tokens, coverage=None):
        """
        decoder_state: (batch_size, hidden_dim)
        encoder_outputs: (batch_size, seq_len, hidden_dim)
        source_tokens: (batch_size, seq_len) - source token indices
        coverage: (batch_size, seq_len) - previous coverage weights
        
        Returns:
        - copy_logits: (batch_size, vocab_size) - logits for copying
        - copy_weights: (batch_size, seq_len) - attention weights over source
        - copy_prob: (batch_size, 1) - probability of copying vs generating
        """
        batch_size = decoder_state.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        # Compute attention over source sequence
        # Simple dot-product attention
        decoder_proj = decoder_state.unsqueeze(2)  # (batch_size, hidden_dim, 1)
        scores = torch.bmm(encoder_outputs, decoder_proj).squeeze(2)  # (batch_size, seq_len)
        
        # Add coverage penalty if available (discourages repeated copying)
        if coverage is not None:
            coverage_penalty = self.coverage_proj(coverage.unsqueeze(-1))  # (batch_size, seq_len, hidden_dim)
            coverage_penalty = (coverage_penalty * encoder_outputs).sum(dim=2)  # (batch_size, seq_len)
            scores = scores - 0.1 * coverage_penalty
        
        # Attention weights
        copy_weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)
        
        # Create copy logits: scatter attention weights to vocab indices
        copy_logits = torch.zeros(batch_size, self.vocab_size, device=decoder_state.device)
        for b in range(batch_size):
            for i in range(seq_len):
                token_idx = source_tokens[b, i].item()
                if token_idx > 0:  # Skip padding tokens
                    copy_logits[b, token_idx] += copy_weights[b, i]
        
        # Generate copy probability (how likely to copy vs generate)
        context = (copy_weights.unsqueeze(1) * encoder_outputs).sum(dim=1)  # (batch_size, hidden_dim)
        combined = torch.cat([decoder_state, context], dim=1)  # (batch_size, hidden_dim*2)
        copy_prob = torch.sigmoid(self.copy_gate(combined))  # (batch_size, 1)
        
        return copy_logits, copy_weights, copy_prob


class LexiconConstrainedDecoder(nn.Module):
    """
    Constraints decoder output to valid words from lexicon.
    Prevents generation of gibberish and ensures domain correctness.
    """
    
    def __init__(self, vocab_size: int, valid_token_mask: torch.Tensor = None):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Mask for valid tokens (lexicon + common words)
        if valid_token_mask is not None:
            self.register_buffer('valid_mask', valid_token_mask.float())
        else:
            self.valid_mask = torch.ones(vocab_size)
    
    def forward(self, logits, enforce_constraints=True):
        """
        Apply lexicon constraints to decoder logits.
        
        logits: (batch_size, vocab_size)
        enforce_constraints: whether to apply constraints
        
        Returns:
        - constrained_logits: (batch_size, vocab_size) with -inf for invalid tokens
        """
        if not enforce_constraints or self.valid_mask is None:
            return logits
        
        # Set invalid tokens to very negative value (will have near-zero probability)
        invalid_mask = (self.valid_mask == 0).unsqueeze(0)  # (1, vocab_size)
        constrained = logits.clone()
        constrained[invalid_mask] = float('-inf')
        
        return constrained
    
    def set_valid_tokens(self, token_ids: torch.Tensor):
        """
        Set which tokens are valid (from lexicon).
        
        token_ids: list of valid token indices
        """
        mask = torch.zeros(self.vocab_size)
        mask[token_ids] = 1.0
        self.register_buffer('valid_mask', mask)


class TIER2Decoder(nn.Module):
    """
    Enhanced decoder with copy mechanism and lexicon constraints.
    Combines:
    1. Standard generation path (generate new words)
    2. Copy mechanism (copy from source)
    3. Lexicon constraints (only valid words)
    """
    
    def __init__(self, hidden_dim: int, vocab_size: int, copy_enabled=True, 
                 lexicon_constrained=True, valid_token_mask=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.copy_enabled = copy_enabled
        self.lexicon_constrained = lexicon_constrained
        
        # Standard generation path
        self.generate = nn.Linear(hidden_dim * 2, vocab_size)
        
        # Copy mechanism
        if copy_enabled:
            self.copy_mechanism = CopyMechanism(hidden_dim, vocab_size)
        
        # Lexicon constraints
        if lexicon_constrained:
            self.lexicon_decoder = LexiconConstrainedDecoder(vocab_size, valid_token_mask)
    
    def forward(self, decoder_state, encoder_outputs, source_tokens, context=None,
                coverage=None, enforce_lexicon=True):
        """
        decoder_state: (batch_size, hidden_dim)
        encoder_outputs: (batch_size, seq_len, hidden_dim)
        source_tokens: (batch_size, seq_len) - source token indices
        context: (batch_size, hidden_dim) - attention context
        coverage: (batch_size, seq_len) - previous coverage weights
        enforce_lexicon: whether to apply lexicon constraints
        
        Returns:
        - logits: (batch_size, vocab_size) - output logits
        - copy_prob: float or None - probability of copying
        """
        
        # Combine decoder state with attention context
        if context is None:
            combined = decoder_state
        else:
            combined = torch.cat([decoder_state, context], dim=1)
        
        # Standard generation path
        gen_logits = self.generate(combined) if context is not None else self.generate(
            torch.cat([decoder_state, decoder_state], dim=1)
        )
        
        copy_prob = None
        
        # Add copy mechanism if enabled
        if self.copy_enabled:
            copy_logits, copy_weights, copy_prob = self.copy_mechanism(
                decoder_state, encoder_outputs, source_tokens, coverage
            )
            
            # Combine generation and copy logits
            # Higher copy_prob = rely more on copy mechanism
            # Lower copy_prob = rely more on generation
            combined_logits = (1 - copy_prob) * gen_logits + copy_prob * copy_logits
        else:
            combined_logits = gen_logits
        
        # Apply lexicon constraints if enabled
        if self.lexicon_constrained and enforce_lexicon:
            combined_logits = self.lexicon_decoder(combined_logits, enforce_constraints=True)
        
        return combined_logits, copy_prob


def build_valid_token_mask(lexicon: dict, tokenizer, special_tokens=None):
    """
    Build a mask of valid tokens from lexicon + common words.
    
    lexicon: dict mapping Akkadian → English
    tokenizer: vocabulary mapper
    special_tokens: list of special token indices to always allow
    
    Returns:
    - mask: torch.Tensor of shape (vocab_size,) with 1.0 for valid, 0.0 for invalid
    """
    vocab_size = len(tokenizer)
    mask = torch.zeros(vocab_size)
    
    # Always allow special tokens
    special_indices = [0, 1, 2, 3]  # PAD, UNK, SOS, EOS
    if special_tokens:
        special_indices.extend(special_tokens)
    
    for idx in special_indices:
        if idx < vocab_size:
            mask[idx] = 1.0
    
    # Allow all words from lexicon English side
    for english_word in lexicon.values():
        words = english_word.split()
        for word in words:
            if word in tokenizer.word2idx:
                idx = tokenizer.word2idx[word]
                mask[idx] = 1.0
    
    return mask
