"""
Layer implementations for ScratchLM transformer model.

This module contains the building blocks of the transformer architecture:
- Multi-head attention
- Feed-forward networks
- Transformer blocks
- Layer normalization
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormalization(nn.Module):
    """
    Layer normalization as described in "Layer Normalization" (Ba et al., 2016).
    
    Normalizes the activations of a layer for each given example.
    
    Args:
        d_model: Dimension of the input features
        eps: Small constant for numerical stability
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Normalized tensor of the same shape
        """
        # Calculate mean and variance along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as described in "Attention Is All You Need" (Vaswani et al., 2017).
    
    Computes scaled dot-product attention with multiple attention heads.
    
    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        dropout: Dropout rate
        use_causal_mask: Whether to use causal masking (for decoder)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_causal_mask = use_causal_mask
        
        # Linear transformations for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional mask tensor of shape (batch_size, seq_len)
                          where 1 = attend, 0 = don't attend
                          
        Returns:
            Tuple of (output, attention_weights)
            output: Tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Optional tensor of shape (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k = self.k_proj(x)  # (batch_size, seq_len, d_model)
        v = self.v_proj(x)  # (batch_size, seq_len, d_model)
        
        # Reshape to separate heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # q: (batch_size, n_heads, seq_len, head_dim)
        # k: (batch_size, n_heads, seq_len, head_dim)
        # scores: (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len) where 1 = attend, 0 = don't attend
            # We need to expand it to (batch_size, 1, 1, seq_len) for key masking
            # This masks out padding tokens in the keys
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            mask = mask.expand(-1, -1, seq_len, -1)  # (batch_size, 1, seq_len, seq_len)
            
            # For causal masking (decoder), also mask future tokens
            if self.use_causal_mask:
                # Create causal mask: upper triangular with diagonal=1
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                    diagonal=1,  # Mask tokens after current position
                )
                # Expand causal mask: (1, 1, seq_len, seq_len)
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
                
                # Combine masks: attend where BOTH are True
                # mask: (batch_size, 1, seq_len, seq_len) - padding mask
                # causal_mask: (1, 1, seq_len, seq_len) - future mask
                combined_mask = mask.bool() & causal_mask.bool()
                scores = scores.masked_fill(~combined_mask, float('-inf'))
            else:
                # Just apply padding mask
                scores = scores.masked_fill(~mask.bool(), float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # attn_weights: (batch_size, n_heads, seq_len, seq_len)
        # v: (batch_size, n_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)  # (batch_size, n_heads, seq_len, head_dim)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()  # (batch_size, seq_len, n_heads, head_dim)
        output = output.view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)
        
        # Project output
        output = self.out_proj(output)
        
        return output, attn_weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network as described in "Attention Is All You Need".
    
    Applies a two-layer feed-forward network with a ReLU activation.
    
    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the feed-forward network (usually 4 * d_model)
        dropout: Dropout rate
        activation: Activation function to use
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu" or activation == "swish":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of the same shape
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block (decoder block).
    
    Consists of:
    1. Causal multi-head attention
    2. Layer normalization
    3. Feed-forward network
    4. Residual connections
    
    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        d_ff: Dimension of the feed-forward network
        dropout: Dropout rate
        attention_dropout: Dropout rate for attention weights
        activation: Activation function for FFN
        layer_norm_eps: Epsilon for layer normalization
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Self-attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout,
            use_causal_mask=True,
        )
        
        # Layer normalization
        self.attn_ln = LayerNormalization(d_model, layer_norm_eps)
        self.ffn_ln = LayerNormalization(d_model, layer_norm_eps)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of the same shape
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, attention_mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output
        x = self.attn_ln(x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = x + ffn_output
        x = self.ffn_ln(x)
        
        return x


if __name__ == "__main__":
    # Test layers
    print("Testing transformer layers...")
    
    # Test parameters
    d_model = 256
    n_heads = 8
    d_ff = 1024
    batch_size = 4
    seq_len = 32
    
    # Test LayerNormalization
    print("\n=== LayerNormalization ===")
    ln = LayerNormalization(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    output = ln(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
    
    # Test MultiHeadAttention
    print("\n=== MultiHeadAttention ===")
    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_weights = mha(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test FeedForwardNetwork
    print("\n=== FeedForwardNetwork ===")
    ffn = FeedForwardNetwork(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    output = ffn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test TransformerBlock
    print("\n=== TransformerBlock ===")
    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
