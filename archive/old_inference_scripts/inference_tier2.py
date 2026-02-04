#!/usr/bin/env python3
"""
TIER 2 Enhanced Inference with Copy Mechanism and Lexicon-Constrained Decoding
"""

import torch
import torch.nn as nn
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import sys

# Configure logging with unbuffered output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from src.tier2_improvements import TIER2Decoder, build_valid_token_mask


class TIER2Seq2SeqInference:
    """
    Enhanced Seq2Seq inference with TIER 2 improvements:
    1. Copy mechanism for proper nouns/numbers
    2. Lexicon-constrained decoding
    3. Better beam search with copy probabilities
    """
    
    def __init__(self, model_dir: str = "models/", device: str = None,
                 lexicon_path: str = None, use_copy=True, use_lexicon_constraints=True):
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_copy = use_copy
        self.use_lexicon_constraints = use_lexicon_constraints
        
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")
        
        logger.info(f"Initializing TIER 2 Seq2Seq inference (device={self.device})")
        logger.info(f"Copy mechanism: {'ENABLED' if use_copy else 'DISABLED'}")
        logger.info(f"Lexicon constraints: {'ENABLED' if use_lexicon_constraints else 'DISABLED'}")
        
        # Load lexicon if provided
        self.lexicon = {}
        if lexicon_path and use_lexicon_constraints:
            self.load_lexicon(lexicon_path)
        
        # Initialize models
        self.embedding = None
        self.rnn = None
        self.attention = None
        self.decoder = None  # Standard decoder
        self.tier2_decoder = None  # Enhanced decoder with TIER 2
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        
        self.load_models()
        self.load_tokenizers()
        self.initialize_tier2_decoder()
    
    def load_lexicon(self, lexicon_path: str):
        """Load Akkadian-English lexicon"""
        try:
            df = pd.read_csv(lexicon_path)
            for _, row in df.iterrows():
                if pd.notna(row.get('akkadian', '')) and pd.notna(row.get('english', '')):
                    akkadian = str(row['akkadian']).strip().lower()
                    english = str(row['english']).strip()
                    if akkadian and english:
                        self.lexicon[akkadian] = english
            logger.info(f"✓ Loaded {len(self.lexicon)} lexicon entries")
        except Exception as e:
            logger.warning(f"Failed to load lexicon: {e}")
    
    def load_models(self):
        """Load all model components"""
        logger.info(f"Loading model components from {self.model_dir}")
        
        try:
            # Load state dicts
            embedding_state = torch.load(self.model_dir / "embedding_final.pt", map_location=self.device)
            rnn_state = torch.load(self.model_dir / "rnn_final.pt", map_location=self.device)
            attention_state = torch.load(self.model_dir / "attention_final.pt", map_location=self.device)
            decoder_state = torch.load(self.model_dir / "decoder_final.pt", map_location=self.device)
            
            logger.info("✓ Loaded all model state dicts")
            
            # Extract dimensions
            src_vocab_size = embedding_state['weight'].shape[0]
            embedding_dim = embedding_state['weight'].shape[1]
            output_vocab_size = decoder_state['weight'].shape[0]
            hidden_dim = decoder_state['weight'].shape[1] // 2
            
            logger.info(f"Model dimensions:")
            logger.info(f"  - Source vocab: {src_vocab_size}")
            logger.info(f"  - Embedding dim: {embedding_dim}")
            logger.info(f"  - Hidden dim: {hidden_dim}")
            logger.info(f"  - Output vocab: {output_vocab_size}")
            
            # Detect LSTM layers
            rnn_keys = list(rnn_state.keys())
            num_layers = max([int(k.split('_l')[1][0]) for k in rnn_keys if '_l' in k]) + 1
            logger.info(f"  - Detected LSTM layers: {num_layers}")
            
            # Load models
            self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
            self.embedding.load_state_dict(embedding_state)
            self.embedding = self.embedding.to(self.device).eval()
            logger.info("✓ Embedding loaded")
            
            dropout = 0.2 if num_layers > 1 else 0
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.rnn.load_state_dict(rnn_state)
            self.rnn = self.rnn.to(self.device).eval()
            logger.info(f"✓ RNN loaded ({num_layers} layers LSTM)")
            
            # Attention
            from src.inference import AttentionLayer
            self.attention = AttentionLayer(hidden_dim)
            self.attention.load_state_dict(attention_state)
            self.attention = self.attention.to(self.device).eval()
            logger.info("✓ Attention mechanism loaded")
            
            # Standard decoder
            self.decoder = nn.Linear(hidden_dim * 2, output_vocab_size)
            self.decoder.load_state_dict(decoder_state)
            self.decoder = self.decoder.to(self.device).eval()
            logger.info("✓ Standard decoder loaded")
            
            self.hidden_dim = hidden_dim
            self.output_vocab_size = output_vocab_size
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def initialize_tier2_decoder(self):
        """Initialize TIER 2 enhanced decoder"""
        logger.info("\nInitializing TIER 2 decoder...")
        
        # Build valid token mask if using lexicon constraints
        valid_mask = None
        if self.use_lexicon_constraints and self.lexicon:
            valid_mask = build_valid_token_mask(
                self.lexicon,
                self.tgt_tokenizer if self.tgt_tokenizer else None
            )
            logger.info(f"✓ Built valid token mask: {valid_mask.sum().item():.0f}/{self.output_vocab_size} tokens")
        
        # Create TIER2 decoder
        self.tier2_decoder = TIER2Decoder(
            hidden_dim=self.hidden_dim,
            vocab_size=self.output_vocab_size,
            copy_enabled=self.use_copy,
            lexicon_constrained=self.use_lexicon_constraints,
            valid_token_mask=valid_mask
        ).to(self.device).eval()
        
        logger.info(f"✓ TIER 2 decoder initialized")
        logger.info(f"  - Copy mechanism: {'ENABLED' if self.use_copy else 'disabled'}")
        logger.info(f"  - Lexicon constraints: {'ENABLED' if self.use_lexicon_constraints else 'disabled'}")
    
    def load_tokenizers(self):
        """Rebuild tokenizers from training data"""
        logger.info("Building tokenizers from training data...")
        
        try:
            project_root = Path(__file__).parent.parent
            
            # Try augmented data first
            train_path = project_root / "data/processed/train_augmented.csv"
            if not train_path.exists():
                train_path = project_root / "data/processed/train_clean.csv"
            
            df = pd.read_csv(train_path)
            logger.info(f"Loaded {len(df)} training samples from {train_path.name}")
            
            # Import tokenizer
            from src.inference import SimpleTokenizer
            
            self.src_tokenizer = SimpleTokenizer()
            self.tgt_tokenizer = SimpleTokenizer()
            
            self.src_tokenizer.build_vocab(df['transliteration'].values)
            self.tgt_tokenizer.build_vocab(df['translation'].values)
            
            logger.info(f"✓ Source vocab: {len(self.src_tokenizer)} tokens")
            logger.info(f"✓ Target vocab: {len(self.tgt_tokenizer)} tokens")
            
        except Exception as e:
            logger.error(f"Error loading tokenizers: {e}")
            raise
    
    def decode_greedy_with_copy(self, src_tokens, max_len=256, temperature=0.8,
                               copy_threshold=0.5):
        """
        Greedy decoding with copy mechanism and lexicon constraints.
        
        src_tokens: (seq_len,) - source tokens
        copy_threshold: if copy_prob > threshold, prefer copying
        """
        src_tokens = src_tokens.unsqueeze(0).to(self.device)  # (1, seq_len)
        batch_size = 1
        
        # Encode source
        with torch.no_grad():
            embedded = self.embedding(src_tokens)
            rnn_out, _ = self.rnn(embedded)
            
            # Decode
            sos_token = torch.tensor([2]).unsqueeze(0).to(self.device)  # SOS
            decoder_input = self.embedding(sos_token)
            
            predictions = []
            coverage = torch.zeros(1, src_tokens.shape[1], device=self.device)
            
            for step in range(max_len):
                # One step of RNN
                decoder_out, _ = self.rnn(decoder_input.unsqueeze(0))
                decoder_state = decoder_out[:, -1, :]
                
                # Attention
                context, attn_weights = self.attention(decoder_state, rnn_out)
                coverage = coverage + attn_weights
                
                # Get next token with TIER 2 decoder
                logits, copy_prob = self.tier2_decoder(
                    decoder_state, rnn_out, src_tokens, context=context,
                    coverage=coverage, enforce_lexicon=self.use_lexicon_constraints
                )
                
                # Apply temperature
                logits = logits / temperature
                
                # Get next token
                next_token = torch.argmax(logits, dim=1)
                predictions.append(next_token.item())
                
                # Stop if EOS
                if next_token.item() == 3:  # EOS
                    break
                
                # Prepare next input
                decoder_input = self.embedding(next_token.unsqueeze(0))
        
        return predictions
    
    def generate_predictions(self, test_csv: str, output_csv: str = "predictions.csv"):
        """Generate predictions for test set with TIER 2 improvements"""
        logger.info(f"\nLoading test data from {test_csv}")
        
        try:
            df = pd.read_csv(test_csv)
            logger.info(f"✓ Loaded {len(df)} test samples")
            
            results = []
            logger.info(f"\nRunning TIER 2 inference with copy mechanism and lexicon constraints...")
            
            for idx, row in df.iterrows():
                akkadian = row['transliteration']
                src_tokens = self.src_tokenizer.encode(akkadian, max_len=256)
                
                # Generate with TIER 2 improvements
                pred_tokens = self.decode_greedy_with_copy(src_tokens)
                translation = self.tgt_tokenizer.decode(pred_tokens)
                
                results.append({
                    'id': row.get('id', idx),
                    'translation': translation
                })
                
                logger.info(f"  Processed {idx+1}/{len(df)} samples")
            
            # Save predictions
            out_df = pd.DataFrame(results)
            out_df.to_csv(output_csv, index=False)
            logger.info(f"\n✓ Predictions saved: {len(results)} rows to {output_csv}")
            
            # Show samples
            logger.info(f"\nSample predictions (with TIER 2 improvements):")
            for i, row in out_df.head(3).iterrows():
                logger.info(f"  ID {row['id']}: {row['translation'][:100]}...")
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise


def main():
    logger.info("="*80)
    logger.info("TIER 2 ENHANCED SEQ2SEQ INFERENCE")
    logger.info("="*80)
    
    project_root = Path(__file__).parent.parent
    
    # Initialize TIER 2 inference
    inference = TIER2Seq2SeqInference(
        model_dir=str(project_root / "models"),
        lexicon_path=str(project_root / "data/raw/OA_Lexicon_eBL.csv"),
        use_copy=True,
        use_lexicon_constraints=True
    )
    
    # Generate predictions
    test_path = project_root / "data/raw/test.csv"
    output_path = project_root / "predictions_tier2.csv"
    
    inference.generate_predictions(str(test_path), str(output_path))
    
    logger.info("\n" + "="*80)
    logger.info("TIER 2 INFERENCE COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
