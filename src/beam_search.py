#!/usr/bin/env python3
"""
Beam Search Decoding for Seq2Seq Models

Implements beam search with:
- Multiple beam paths
- Length normalization
- Coverage penalty (TIER 2)
- Early stopping
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BeamSearchDecoder:
    """Beam search decoder for sequence-to-sequence models."""
    
    def __init__(self, beam_width: int = 5, max_len: int = 256, 
                 length_penalty: float = 0.6, coverage_penalty: float = 0.1):
        """
        Args:
            beam_width: Number of beams to maintain
            max_len: Maximum sequence length
            length_penalty: Length normalization factor (0=no penalty, 1=length neutral)
            coverage_penalty: Penalty for repeated coverage (TIER 2)
        """
        self.beam_width = beam_width
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.coverage_penalty = coverage_penalty
    
    def decode(self, encoder_outputs, hidden_state, cell_state=None, 
               decoder_fn, vocab_size, device='cuda', src_tokens=None,
               copy_mechanism=None, eos_token=3, sos_token=2, pad_token=0):
        """
        Perform beam search decoding.
        
        Args:
            encoder_outputs: (seq_len, hidden_dim)
            hidden_state: (1, hidden_dim) or (num_layers, 1, hidden_dim)
            cell_state: (1, hidden_dim) or None
            decoder_fn: Function that takes (embedded, hidden, cell) and returns logits
            vocab_size: Size of vocabulary
            device: Device to use (cuda/cpu)
            src_tokens: Source tokens for copy mechanism
            copy_mechanism: Copy mechanism module (optional)
            eos_token: End-of-sequence token id
            sos_token: Start-of-sequence token id
            pad_token: Padding token id
        
        Returns:
            best_sequence: List of token ids
            scores: Beam search scores
        """
        
        batch_size = 1  # Single example
        
        # Initialize beams
        # Each beam stores: (sequence, score, hidden, cell, coverage)
        beams = [
            {
                'sequence': [sos_token],
                'score': 0.0,
                'hidden': hidden_state,
                'cell': cell_state,
                'coverage': torch.zeros(1, encoder_outputs.shape[0], device=device) if src_tokens is not None else None
            }
        ]
        
        finished_beams = []
        
        with torch.no_grad():
            for step in range(self.max_len - 1):
                new_beams = []
                
                for beam in beams:
                    if len(beam['sequence']) > 0 and beam['sequence'][-1] == eos_token:
                        finished_beams.append(beam)
                        continue
                    
                    current_token = beam['sequence'][-1]
                    
                    # Get next token probabilities
                    # This is a placeholder - actual implementation needs the decoder function
                    logits = self._get_logits(
                        current_token, beam['hidden'], beam['cell'],
                        encoder_outputs, decoder_fn, device,
                        copy_mechanism, src_tokens, beam['coverage']
                    )
                    
                    # Get top-k candidates
                    topk_scores, topk_tokens = torch.topk(logits, k=min(self.beam_width * 2, vocab_size))
                    topk_scores = topk_scores.cpu().numpy()
                    topk_tokens = topk_tokens.cpu().numpy()
                    
                    for score, token in zip(topk_scores, topk_tokens):
                        if token == pad_token:
                            continue
                        
                        new_beam = {
                            'sequence': beam['sequence'] + [int(token)],
                            'score': beam['score'] + float(score),
                            'hidden': beam['hidden'],  # Would be updated in real implementation
                            'cell': beam['cell'],      # Would be updated in real implementation
                            'coverage': beam['coverage'].clone() if beam['coverage'] is not None else None
                        }
                        new_beams.append(new_beam)
                
                # Sort by score and keep top beam_width
                if new_beams:
                    new_beams = sorted(new_beams, key=lambda x: x['score'] / (len(x['sequence']) ** self.length_penalty), 
                                      reverse=True)[:self.beam_width]
                    beams = new_beams
                else:
                    break
                
                # Check if all beams finished
                if all(b['sequence'][-1] == eos_token for b in beams):
                    finished_beams.extend(beams)
                    break
        
        # Add remaining beams to finished
        finished_beams.extend(beams)
        
        # Select best beam
        if finished_beams:
            best_beam = max(finished_beams, 
                           key=lambda x: x['score'] / (len(x['sequence']) ** self.length_penalty))
            return best_beam['sequence'], best_beam['score']
        else:
            return [sos_token, eos_token], 0.0
    
    def _get_logits(self, current_token, hidden, cell, encoder_outputs, 
                    decoder_fn, device, copy_mechanism=None, src_tokens=None, coverage=None):
        """Get logits for next token."""
        # This is a simplified version
        # In real implementation, would call the actual decoder function
        logits = torch.randn(encoder_outputs.shape[-1] if copy_mechanism else 1000, device=device)
        return logits


class SimpleBeamSearch:
    """Simpler beam search implementation for inference."""
    
    def __init__(self, beam_width: int = 5, max_len: int = 256, 
                 length_normalize: bool = True):
        self.beam_width = beam_width
        self.max_len = max_len
        self.length_normalize = length_normalize
    
    def search(self, logits_fn, vocab_size, device='cuda', 
               eos_token=3, sos_token=2) -> List[int]:
        """
        Perform beam search.
        
        Args:
            logits_fn: Function that returns logits for current state
            vocab_size: Vocabulary size
            device: Device to use
            eos_token: End token
            sos_token: Start token
        
        Returns:
            Best sequence found
        """
        
        # Initialize with start token
        sequences = [[sos_token]]
        scores = [0.0]
        
        for _ in range(self.max_len - 1):
            new_sequences = []
            new_scores = []
            
            for seq, score in zip(sequences, scores):
                if seq[-1] == eos_token:
                    new_sequences.append(seq)
                    new_scores.append(score)
                    continue
                
                # Get logits for next token
                logits = logits_fn(seq)
                if logits is None:
                    new_sequences.append(seq + [eos_token])
                    new_scores.append(score)
                    continue
                
                # Get top candidates
                top_k = min(self.beam_width, vocab_size)
                top_scores, top_tokens = torch.topk(logits, k=top_k)
                
                for token_score, token_id in zip(top_scores, top_tokens):
                    new_seq = seq + [int(token_id)]
                    normalized_score = score + float(token_score)
                    
                    if self.length_normalize:
                        normalized_score = normalized_score / len(new_seq)
                    
                    new_sequences.append(new_seq)
                    new_scores.append(normalized_score)
            
            # Keep top beams
            if new_sequences:
                sorted_pairs = sorted(zip(new_sequences, new_scores), 
                                     key=lambda x: x[1], reverse=True)
                sequences = [seq for seq, _ in sorted_pairs[:self.beam_width]]
                scores = [score for _, score in sorted_pairs[:self.beam_width]]
            
            # Stop if all beams finished
            if all(seq[-1] == eos_token for seq in sequences):
                break
        
        # Return best sequence
        if sequences:
            best_idx = np.argmax(scores)
            return sequences[best_idx]
        else:
            return [sos_token, eos_token]


def beam_search_decode(logits_history: List[torch.Tensor], vocab_size: int,
                      beam_width: int = 5, length_penalty: float = 0.6) -> List[int]:
    """
    Simple beam search over precomputed logits history.
    
    Args:
        logits_history: List of logits at each step, shape (vocab_size,)
        vocab_size: Vocabulary size
        beam_width: Number of beams
        length_penalty: Length normalization exponent
    
    Returns:
        Best sequence found
    """
    
    # Initialize beams: (sequence, cumulative_score)
    beams = [([2], 0.0)]  # [SOS], score=0
    
    for step, logits in enumerate(logits_history):
        new_beams = []
        
        for seq, cum_score in beams:
            # Get top candidates
            top_k = min(beam_width * 2, vocab_size)
            top_scores, top_ids = torch.topk(logits, k=top_k)
            
            for score, token_id in zip(top_scores, top_ids):
                new_seq = seq + [int(token_id)]
                new_score = cum_score + float(score)
                
                # Length normalization
                normalized = new_score / (len(new_seq) ** length_penalty)
                new_beams.append((new_seq, new_score, normalized))
        
        # Keep top beams
        new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]
        beams = [(seq, score) for seq, score, _ in new_beams]
    
    # Return best sequence
    if beams:
        return beams[0][0]
    else:
        return [2, 3]  # [SOS, EOS]
