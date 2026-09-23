"""
Tokenizer utilities for ScratchLM.

Provides functions for loading, saving, and managing tokenizers.
"""

import json
from pathlib import Path
from typing import Optional, Union

from .base import BaseTokenizer
from .bpe import BPETokenizer


# Tokenizer type registry
_TOKENIZER_TYPES = {
    "bpe": BPETokenizer,
}


def register_tokenizer_type(name: str, tokenizer_class):
    """
    Register a new tokenizer type.
    
    Args:
        name: Name of the tokenizer type
        tokenizer_class: Tokenizer class
    """
    _TOKENIZER_TYPES[name] = tokenizer_class


def get_tokenizer_type(name: str):
    """
    Get a tokenizer class by type name.
    
    Args:
        name: Tokenizer type name
        
    Returns:
        Tokenizer class
    """
    if name not in _TOKENIZER_TYPES:
        raise ValueError(f"Unknown tokenizer type: {name}. Available: {list(_TOKENIZER_TYPES.keys())}")
    return _TOKENIZER_TYPES[name]


def save_tokenizer(tokenizer: BaseTokenizer, directory: Union[str, Path]):
    """
    Save a tokenizer to a directory.
    
    Args:
        tokenizer: Tokenizer instance to save
        directory: Directory to save to
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        "tokenizer_type": tokenizer.__class__.__name__.lower(),
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": tokenizer.special_tokens,
    }
    
    config_path = directory / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save vocabulary
    vocab_path = directory / "vocab.json"
    vocab = tokenizer.get_vocab()
    id_to_token = tokenizer.get_id_to_token()
    
    vocab_data = {
        "token_to_id": vocab,
        "id_to_token": id_to_token,
    }
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    # Call tokenizer-specific save if available
    if hasattr(tokenizer, 'save'):
        tokenizer.save(directory)


def load_tokenizer(directory: Union[str, Path]) -> BaseTokenizer:
    """
    Load a tokenizer from a directory.
    
    Args:
        directory: Directory containing tokenizer files
        
    Returns:
        Loaded tokenizer instance
    """
    directory = Path(directory)
    
    # Load config
    config_path = directory / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tokenizer_type = config.get("tokenizer_type", "bpe")
    tokenizer_class = get_tokenizer_type(tokenizer_type)
    
    # Load tokenizer
    if hasattr(tokenizer_class, 'load'):
        return tokenizer_class.load(directory)
    else:
        # Fallback: create empty tokenizer and load vocab
        tokenizer = tokenizer_class()
        
        vocab_path = directory / "vocab.json"
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        tokenizer._vocab = vocab_data.get("token_to_id", {})
        tokenizer._id_to_token = vocab_data.get("id_to_token", {})
        
        return tokenizer


def load_tokenizer_from_config(config: dict, directory: Optional[Union[str, Path]] = None) -> BaseTokenizer:
    """
    Load a tokenizer from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        directory: Optional directory to load tokenizer files from
        
    Returns:
        Tokenizer instance
    """
    tokenizer_type = config.get("tokenizer_type", "bpe")
    tokenizer_class = get_tokenizer_type(tokenizer_type)
    
    if directory is not None:
        return load_tokenizer(directory)
    
    # Create tokenizer from config
    if tokenizer_type == "bpe":
        return BPETokenizer(
            vocab_size=config.get("vocab_size", 8192),
            pad_token=config.get("pad_token", "<pad>"),
            unk_token=config.get("unk_token", "<unk>"),
            bos_token=config.get("bos_token", "<s>"),
            eos_token=config.get("eos_token", "</s>"),
            min_frequency=config.get("min_frequency", 2),
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


if __name__ == "__main__":
    # Test tokenizer utilities
    from pathlib import Path
    import tempfile
    
    print("Testing tokenizer utilities...")
    
    # Create a simple tokenizer
    tokenizer = BPETokenizer(vocab_size=100)
    
    # Save to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_tokenizer"
        save_tokenizer(tokenizer, save_path)
        print(f"Saved tokenizer to {save_path}")
        
        # Load it back
        loaded = load_tokenizer(save_path)
        print(f"Loaded tokenizer. Vocab size: {loaded.vocab_size}")
        
        # Verify they're the same
        assert loaded.vocab_size == tokenizer.vocab_size
        print("Tokenizer save/load test passed!")
