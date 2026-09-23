"""
Model configuration for ScratchLM.

Defines the architecture and hyperparameters for the transformer model.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class TransformerConfig:
    """
    Configuration for a decoder-only transformer model.
    
    This defines all the hyperparameters needed to instantiate the model.
    """
    
    # Model identification
    version: str = "v0.1"
    name: str = "ScratchLM"
    description: str = "Decoder-only transformer language model"
    
    # Architecture parameters
    d_model: int = 256              # Embedding dimension
    n_layers: int = 4              # Number of transformer layers
    n_heads: int = 8              # Number of attention heads
    d_ff: int = 1024              # Feed-forward dimension
    
    # Sequence parameters
    context_length: int = 128     # Maximum sequence length (in tokens)
    vocab_size: int = 8192        # Vocabulary size
    
    # Dropout
    dropout: float = 0.1           # Dropout rate
    attention_dropout: float = 0.1
    
    # Activation
    activation: str = "gelu"      # Activation function (gelu, relu, silu)
    
    # Initialization
    init_std: float = 0.02        # Standard deviation for weight initialization
    init_bias: float = 0.0        # Bias initialization value
    
    # Layer normalization
    layer_norm_eps: float = 1e-5  # Layer norm epsilon
    
    # Positional embeddings
    use_rotary: bool = False      # Use rotary positional embeddings
    
    # Output
    tie_embeddings: bool = True   # Tie input and output embeddings
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        return self.d_model // self.n_heads
    
    @property
    def n_params(self) -> int:
        """
        Approximate number of parameters in the model.
        
        This is an estimate and may not be exact due to:
        - Bias terms
        - Layer normalization parameters
        - Positional embeddings
        """
        # Embeddings: d_model * vocab_size
        embed_params = self.d_model * self.vocab_size
        
        # Per-layer parameters
        per_layer = 0
        
        # Attention
        # Q, K, V projections: 3 * d_model * d_model
        attn_proj = 3 * self.d_model * self.d_model
        # Output projection: d_model * d_model
        attn_out = self.d_model * self.d_model
        # Layer norm: 2 * d_model (gamma and beta)
        attn_ln = 2 * self.d_model
        per_layer += attn_proj + attn_out + attn_ln
        
        # Feed-forward
        # First projection: d_model * d_ff
        ff1 = self.d_model * self.d_ff
        # Second projection: d_ff * d_model
        ff2 = self.d_ff * self.d_model
        # Layer norm: 2 * d_model
        ff_ln = 2 * self.d_model
        per_layer += ff1 + ff2 + ff_ln
        
        # Total per layer
        per_layer_total = per_layer
        
        # All layers
        layer_params = per_layer_total * self.n_layers
        
        # Final layer norm
        final_ln = 2 * self.d_model
        
        # Total
        total = embed_params + layer_params + final_ln
        
        # If not tying embeddings, add output layer
        if not self.tie_embeddings:
            total += self.d_model * self.vocab_size
        
        return total
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'version': self.version,
            'name': self.name,
            'description': self.description,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'context_length': self.context_length,
            'vocab_size': self.vocab_size,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'activation': self.activation,
            'init_std': self.init_std,
            'init_bias': self.init_bias,
            'layer_norm_eps': self.layer_norm_eps,
            'use_rotary': self.use_rotary,
            'tie_embeddings': self.tie_embeddings,
            # Computed properties
            'head_dim': self.head_dim,
            'n_params_estimate': self.n_params,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TransformerConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: Path):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Path) -> 'TransformerConfig':
        """Load configuration from YAML file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)


@dataclass
class TrainingConfig:
    """
    Configuration for training the model.
    """
    
    # Training identification
    experiment_id: str = "exp001"
    model_version: str = "v0.1"
    description: str = "Initial training run"
    
    # Data parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, linear, constant
    warmup_steps: int = 100
    max_steps: Optional[int] = None
    num_epochs: int = 10
    
    # Gradient clipping
    clip_grad_norm: Optional[float] = 1.0
    
    # Regularization
    dropout: float = 0.1
    
    # Checkpointing
    save_checkpoint_every: int = 500  # Steps
    save_best: bool = True
    metric_to_watch: str = "val_loss"
    
    # Validation
    validate_every: int = 100  # Steps
    
    # Logging
    log_every: int = 10  # Steps
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.001
    
    # Mixed precision (CPU doesn't support this well, but we include it for future)
    use_amp: bool = False
    
    # Device
    device: str = "cpu"  # cpu, cuda, mps
    
    # Random seed
    seed: int = 42
    deterministic: bool = True
    
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size accounting for gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'model_version': self.model_version,
            'description': self.description,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'effective_batch_size': self.effective_batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'betas': self.betas,
            'eps': self.eps,
            'use_scheduler': self.use_scheduler,
            'scheduler_type': self.scheduler_type,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            'num_epochs': self.num_epochs,
            'clip_grad_norm': self.clip_grad_norm,
            'dropout': self.dropout,
            'save_checkpoint_every': self.save_checkpoint_every,
            'save_best': self.save_best,
            'metric_to_watch': self.metric_to_watch,
            'validate_every': self.validate_every,
            'log_every': self.log_every,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'use_amp': self.use_amp,
            'device': self.device,
            'seed': self.seed,
            'deterministic': self.deterministic,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: Path):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Path) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)


@dataclass
class TokenizerConfig:
    """
    Configuration for the tokenizer.
    """
    
    # Tokenizer identification
    version: str = "bpe_8k"
    name: str = "BPE Tokenizer"
    description: str = "Byte Pair Encoding tokenizer for English"
    
    # Tokenizer type
    tokenizer_type: str = "bpe"  # bpe, wordpiece, character
    
    # Vocabulary
    vocab_size: int = 8192
    
    # Special tokens
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<s>"      # Beginning of sequence
    eos_token: str = "</s>"     # End of sequence
    
    # BPE-specific parameters
    min_frequency: int = 2       # Minimum frequency for a pair to be merged
    special_tokens: List[str] = field(default_factory=lambda: [
        "<pad>", "<unk>", "<s>", "</s>"
    ])
    
    # Training parameters
    num_merges: Optional[int] = None  # Number of merges (vocab_size - base_vocab)
    
    @property
    def base_vocab_size(self) -> int:
        """Base vocabulary size (unique characters + special tokens)."""
        return len(self.special_tokens) + 256  # ASCII characters
    
    @property
    def num_train_merges(self) -> int:
        """Number of merges to perform during training."""
        if self.num_merges is not None:
            return self.num_merges
        return self.vocab_size - self.base_vocab_size
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'version': self.version,
            'name': self.name,
            'description': self.description,
            'tokenizer_type': self.tokenizer_type,
            'vocab_size': self.vocab_size,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens,
            'num_merges': self.num_merges,
            'base_vocab_size': self.base_vocab_size,
            'num_train_merges': self.num_train_merges,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TokenizerConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: Path):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Path) -> 'TokenizerConfig':
        """Load configuration from YAML file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)


# Pre-defined configurations for easy use

def get_config_preset(name: str):
    """
    Get a pre-defined configuration preset.
    
    Args:
        name: Preset name (e.g., "tiny", "small", "medium")
        
    Returns:
        Tuple of (model_config, training_config, tokenizer_config)
    """
    presets = {
        "tiny": {
            "model": TransformerConfig(
                version="v0.1",
                d_model=128,
                n_layers=2,
                n_heads=4,
                d_ff=512,
                context_length=128,
                vocab_size=4096,
                dropout=0.1,
            ),
            "training": TrainingConfig(
                experiment_id="tiny_001",
                batch_size=8,
                learning_rate=3e-4,
                num_epochs=20,
            ),
            "tokenizer": TokenizerConfig(
                version="bpe_4k",
                vocab_size=4096,
            ),
        },
        "small": {
            "model": TransformerConfig(
                version="v0.1",
                d_model=256,
                n_layers=4,
                n_heads=8,
                d_ff=1024,
                context_length=128,
                vocab_size=8192,
                dropout=0.1,
            ),
            "training": TrainingConfig(
                experiment_id="small_001",
                batch_size=4,
                learning_rate=1e-4,
                num_epochs=20,
            ),
            "tokenizer": TokenizerConfig(
                version="bpe_8k",
                vocab_size=8192,
            ),
        },
        "medium": {
            "model": TransformerConfig(
                version="v0.1",
                d_model=512,
                n_layers=6,
                n_heads=8,
                d_ff=2048,
                context_length=256,
                vocab_size=16384,
                dropout=0.1,
            ),
            "training": TrainingConfig(
                experiment_id="medium_001",
                batch_size=2,
                learning_rate=5e-5,
                num_epochs=10,
            ),
            "tokenizer": TokenizerConfig(
                version="bpe_16k",
                vocab_size=16384,
            ),
        },
    }
    
    if name not in presets:
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
    
    return presets[name]["model"], presets[name]["training"], presets[name]["tokenizer"]


if __name__ == "__main__":
    # Test configurations
    print("=== Tiny Configuration ===")
    model, train, tok = get_config_preset("tiny")
    print(f"Model params: {model.n_params:,}")
    print(f"Training: {train.to_dict()}")
    print(f"Tokenizer: {tok.to_dict()}")
    
    print("\n=== Small Configuration ===")
    model, train, tok = get_config_preset("small")
    print(f"Model params: {model.n_params:,}")
    print(f"Training: {train.to_dict()}")
    print(f"Tokenizer: {tok.to_dict()}")
