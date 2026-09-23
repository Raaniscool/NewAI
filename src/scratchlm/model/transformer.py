"""
ScratchLM - Transformer-based language model.

This module implements the complete decoder-only transformer model
for language modeling from scratch.
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig
from .layers import TransformerBlock, LayerNormalization


class PositionalEmbedding(nn.Module):
    """
    Positional embedding for transformer models.
    
    Uses learned positional embeddings (not sinusoidal).
    
    Args:
        max_length: Maximum sequence length
        d_model: Embedding dimension
    """
    
    def __init__(self, max_length: int, d_model: int):
        super().__init__()
        
        self.max_length = max_length
        self.d_model = d_model
        
        # Position embeddings
        self.embeddings = nn.Parameter(torch.zeros(max_length, d_model))
        
        # Initialize
        nn.init.normal_(self.embeddings, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional embeddings added
        """
        batch_size, seq_len, _ = x.shape
        
        # Get position embeddings for the first seq_len positions
        pos_emb = self.embeddings[:seq_len, :]  # (seq_len, d_model)
        
        # Add to input
        return x + pos_emb.unsqueeze(0)  # (1, seq_len, d_model) -> broadcast


class ScratchLM(nn.Module):
    """
    ScratchLM - A decoder-only transformer language model.
    
    This is the main model class that implements a complete transformer
    for causal language modeling. The task is to predict the next token
    given the previous tokens.
    
    Architecture:
    1. Token embeddings (input embeddings)
    2. Positional embeddings
    3. Stack of transformer blocks
    4. Layer normalization
    5. Output projection (to vocabulary size)
    
    Args:
        config: Transformer configuration
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )
        
        # Positional embeddings
        self.position_embeddings = PositionalEmbedding(
            max_length=config.context_length,
            d_model=config.d_model,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.n_layers)
        ])
        
        # Final layer normalization
        self.final_ln = LayerNormalization(
            d_model=config.d_model,
            eps=config.layer_norm_eps,
        )
        
        # Output projection
        # If tying embeddings, use the token embeddings
        if config.tie_embeddings:
            self.output_proj = None
        else:
            self.output_proj = nn.Linear(
                in_features=config.d_model,
                out_features=config.vocab_size,
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Token embeddings
        nn.init.normal_(self.token_embeddings.weight, std=self.config.init_std)
        
        # Output projection (if not tied)
        if self.output_proj is not None:
            nn.init.normal_(self.output_proj.weight, std=self.config.init_std)
            if self.output_proj.bias is not None:
                nn.init.constant_(self.output_proj.bias, self.config.init_bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing token IDs
            attention_mask: Optional tensor of shape (batch_size, seq_len)
                          where 1 = attend, 0 = don't attend
                          
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.token_embeddings(input_ids)
        
        # Add positional embeddings
        x = self.position_embeddings(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer normalization
        x = self.final_ln(x)
        
        # Output projection
        if self.output_proj is not None:
            logits = self.output_proj(x)
        else:
            # Tied embeddings: use token embeddings as output
            # (vocab_size, d_model) @ (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
            logits = torch.matmul(x, self.token_embeddings.weight.transpose(0, 1))
        
        return logits
    
    def get_next_token_logits(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get logits for the next token only.
        
        This is useful for autoregressive generation where we only
        need the logits for the next position.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing token IDs
            
        Returns:
            Logits tensor of shape (batch_size, vocab_size) for the next token
        """
        # Forward pass
        logits = self.forward(input_ids)
        
        # Get logits for the last position only
        # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size)
        return logits[:, -1, :]
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs of shape (batch_size, seq_len)
            max_length: Maximum length to generate (including input)
            temperature: Temperature for sampling (1.0 = no change)
            top_k: Number of top tokens to consider (None for all)
            top_p: Probability mass to consider (None for all)
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Token ID for padding
            
        Returns:
            Generated token IDs of shape (batch_size, max_length)
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize output with input
        output = input_ids.clone()
        
        # Generate tokens one at a time
        for step in range(seq_len, max_length):
            # Get next token logits
            next_logits = self.get_next_token_logits(output)
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                # Get top-k logits
                top_logits, _ = torch.topk(next_logits, top_k, dim=-1)
                # Create mask
                mask = next_logits >= top_logits[:, -1:]
                next_logits = next_logits.masked_fill(~mask, float('-inf'))
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                # Sort logits
                sorted_logits, sorted_indices = torch.sort(next_logits, dim=-1, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                
                # Compute cumulative probabilities
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Find the smallest set of top tokens with cumulative probability >= top_p
                mask = cum_probs >= top_p
                
                # Get the cutoff index
                cutoff = mask.int().argmax(dim=-1, keepdim=True)
                
                # Create mask for original logits
                mask = torch.zeros_like(next_logits, dtype=torch.bool)
                for i in range(batch_size):
                    mask[i, sorted_indices[i, :cutoff[i]]] = True
                
                next_logits = next_logits.masked_fill(~mask, float('-inf'))
            
            # Sample or greedy
            if do_sample:
                # Convert to probabilities
                probs = F.softmax(next_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            else:
                # Greedy: take argmax
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (batch_size, 1)
            
            # Append to output
            output = torch.cat([output, next_token], dim=1)
            
            # Stop if all sequences have generated pad token
            # (This is a simple stopping criterion)
            if (next_token == pad_token_id).all():
                break
        
        return output
    
    def get_num_params(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """Save the model to a file."""
        torch.save({
            'config': self.config.to_dict(),
            'state_dict': self.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'ScratchLM':
        """Load a model from a file."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create config from dict
        config_dict = checkpoint['config']
        config = TransformerConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])
        
        return model
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: int = 0,
        loss: float = 0.0,
    ):
        """
        Save a training checkpoint.
        
        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer state to save
            step: Training step
            loss: Current loss
        """
        checkpoint = {
            'config': self.config.to_dict(),
            'model_state_dict': self.state_dict(),
            'step': step,
            'loss': loss,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple['ScratchLM', int, float]:
        """
        Load a training checkpoint.
        
        Args:
            path: Path to checkpoint file
            optimizer: Optional optimizer to load state into
            
        Returns:
            Tuple of (model, step, loss)
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create model
        config_dict = checkpoint['config']
        config = TransformerConfig.from_dict(config_dict)
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        step = checkpoint.get('step', 0)
        loss = checkpoint.get('loss', 0.0)
        
        return model, step, loss


if __name__ == "__main__":
    # Test the model
    print("Testing ScratchLM model...")
    
    from .config import get_config_preset
    
    # Get a small config
    model_config, _, _ = get_config_preset("tiny")
    
    print(f"\nModel configuration:")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  n_heads: {model_config.n_heads}")
    print(f"  vocab_size: {model_config.vocab_size}")
    print(f"  context_length: {model_config.context_length}")
    print(f"  Estimated params: {model_config.n_params:,}")
    
    # Create model
    print("\nCreating model...")
    model = ScratchLM(model_config)
    
    # Count parameters
    num_params = model.get_num_params()
    print(f"Actual parameters: {num_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Enable eval mode for deterministic output
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Output shape: {logits.shape}")
    
    # Test next token prediction
    print("\nTesting next token prediction...")
    next_logits = model.get_next_token_logits(input_ids)
    print(f"Next token logits shape: {next_logits.shape}")
    
    # Test generation
    print("\nTesting generation...")
    # Start with a simple prompt
    prompt = torch.tensor([[1, 2, 3]])  # (batch_size=1, seq_len=3)
    generated = model.generate(
        prompt,
        max_length=20,
        temperature=0.7,
        do_sample=True,
    )
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    print("\nAll tests passed!")
