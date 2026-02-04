"""
LSTM Seq2Seq model with Bahdanau attention for Akkadian-English translation.

Architecture:
- Encoder: Bidirectional LSTM
- Decoder: LSTM with attention mechanism
- Attention: Bahdanau (additive) attention
- Loss: CrossEntropyLoss with label smoothing

Configuration loaded from configs/model_seq2seq.yaml
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import yaml
import json

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.
    
    Computes attention weights over encoder outputs based on decoder hidden state.
    """
    
    def __init__(self, hidden_size: int):
        """
        Initialize attention.
        
        Args:
            hidden_size: Size of attention matrices
        """
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        query: torch.Tensor,  # (batch_size, hidden_size)
        keys: torch.Tensor,   # (seq_len, batch_size, hidden_size)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention.
        
        Args:
            query: Decoder hidden state (batch_size, hidden_size)
            keys: Encoder outputs (seq_len, batch_size, hidden_size)
            
        Returns:
            Tuple of (context, attention_weights)
        """
        seq_len = keys.size(0)
        batch_size = keys.size(1)
        
        # Expand query to match sequence length
        # (batch_size, hidden_size) -> (seq_len, batch_size, hidden_size)
        query_expanded = query.unsqueeze(0).expand(seq_len, -1, -1)
        
        # Concatenate query and keys
        # (seq_len, batch_size, hidden_size*2)
        combined = torch.cat([query_expanded, keys], dim=2)
        
        # Apply attention layers
        # (seq_len, batch_size, hidden_size)
        attn_hidden = torch.tanh(self.attn(combined))
        
        # Compute attention scores
        # (seq_len, batch_size, 1)
        scores = self.v(attn_hidden)
        
        # Remove last dimension and transpose for softmax
        # (batch_size, seq_len)
        scores = scores.squeeze(2).transpose(0, 1)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=1)
        
        # Compute context: weighted sum of keys
        # (batch_size, seq_len) x (seq_len, batch_size, hidden_size)
        # -> (batch_size, hidden_size)
        context = torch.matmul(attention_weights, keys.transpose(0, 1))
        
        return context, attention_weights


class Encoder(nn.Module):
    """
    Bidirectional LSTM encoder.
    
    Encodes source sequences into context vectors.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize encoder.
        
        Args:
            vocab_size: Size of input vocabulary
            embedding_size: Size of embeddings
            hidden_size: Size of hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(
        self,
        src: torch.Tensor,  # (seq_len, batch_size)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode source sequence.
        
        Args:
            src: Source sequence (seq_len, batch_size)
            
        Returns:
            Tuple of (outputs, (hidden, cell))
                outputs: (seq_len, batch_size, hidden_size*2)
                hidden: (num_layers*2, batch_size, hidden_size)
                cell: (num_layers*2, batch_size, hidden_size)
        """
        # Embed source
        embedded = self.embedding(src)  # (seq_len, batch_size, embedding_size)
        
        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    """
    LSTM decoder with attention.
    
    Decodes context vectors into target sequences.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_hidden_size: int = 512,
    ):
        """
        Initialize decoder.
        
        Args:
            vocab_size: Size of output vocabulary
            embedding_size: Size of embeddings
            hidden_size: Size of hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            attention_hidden_size: Size of attention mechanism
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention = Attention(attention_hidden_size)
        
        self.lstm = nn.LSTM(
            embedding_size + attention_hidden_size,
            hidden_size,
            num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.fc_out = nn.Linear(hidden_size + attention_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
    
    def forward(
        self,
        tgt: torch.Tensor,           # (batch_size,) - single timestep
        hidden: torch.Tensor,         # (num_layers, batch_size, hidden_size)
        cell: torch.Tensor,           # (num_layers, batch_size, hidden_size)
        encoder_outputs: torch.Tensor,# (seq_len, batch_size, hidden_size*2)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode single timestep.
        
        Args:
            tgt: Target token (batch_size,)
            hidden: Decoder hidden state
            cell: Decoder cell state
            encoder_outputs: Encoder outputs for attention
            
        Returns:
            Tuple of (output, hidden, cell)
        """
        # Embed target
        embedded = self.embedding(tgt).unsqueeze(0)  # (1, batch_size, embedding_size)
        embedded = self.dropout(embedded)
        
        # Apply attention
        context, _ = self.attention(hidden[-1], encoder_outputs)
        context = context.unsqueeze(0)  # (1, batch_size, hidden_size)
        
        # Concatenate with embedded target
        rnn_input = torch.cat([embedded, context], dim=2)  # (1, batch_size, embed+hidden)
        
        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        # Generate output
        prediction = self.fc_out(torch.cat([outputs, context], dim=2))
        prediction = prediction.squeeze(0)  # (batch_size, vocab_size)
        
        return prediction, hidden, cell


class Seq2SeqModel(nn.Module):
    """
    Complete Seq2Seq model with attention.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: str = 'cuda',
    ):
        """
        Initialize Seq2Seq model.
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            embedding_size: Embedding dimension
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embedding_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embedding_size, hidden_size, num_layers, dropout, hidden_size * 2)
        self.device = device
    
    def forward(
        self,
        src: torch.Tensor,     # (seq_len, batch_size)
        tgt: torch.Tensor,     # (seq_len, batch_size)
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            src: Source sequence
            tgt: Target sequence
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Decoder outputs (seq_len, batch_size, vocab_size)
        """
        batch_size = src.size(1)
        tgt_len = tgt.size(0)
        tgt_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size, device=self.device)
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # First decoder input is <sos> token (assumed to be index 2)
        decoder_input = tgt[0]
        
        for t in range(1, tgt_len):
            # Decode
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[t] = output
            
            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt[t]
            else:
                decoder_input = output.argmax(1)
        
        return outputs


class Seq2SeqTrainer:
    """
    Trainer for Seq2Seq model.
    """
    
    def __init__(
        self,
        model: Seq2SeqModel,
        config: Dict[str, Any],
        device: str = 'cuda',
    ):
        """
        Initialize trainer.
        
        Args:
            model: Seq2Seq model
            config: Configuration dictionary from YAML
            device: Device to use
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )
        
        # Setup loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding token
        
        logger.info(f"Initialized Seq2Seq trainer on device: {device}")
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Epoch number
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Forward pass
            outputs = self.model(src, tgt, teacher_forcing_ratio=0.5)
            
            # Calculate loss
            loss = self.criterion(
                outputs[1:].view(-1, outputs.shape[-1]),
                tgt[1:].view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_model(self, path: str) -> None:
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model to {path}")


def load_config(config_path: str = "configs/model_seq2seq.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Example usage."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config("configs/model_seq2seq.yaml")
    
    # Create model
    model = Seq2SeqModel(
        src_vocab_size=5000,
        tgt_vocab_size=3000,
        embedding_size=config['model']['encoder']['embedding_size'],
        hidden_size=config['model']['encoder']['hidden_size'],
        num_layers=config['model']['encoder']['num_layers'],
        dropout=config['model']['encoder']['dropout'],
        device=config['model']['device'],
    )
    
    logger.info(f"Created Seq2Seq model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Print model architecture
    logger.info(f"\nModel architecture:\n{model}")


if __name__ == '__main__':
    main()
