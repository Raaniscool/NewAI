"""
Base tokenizer interface for ScratchLM.

Defines the common interface that all tokenizers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers.
    
    This defines the interface that all tokenizers must implement
    to ensure consistency across the codebase.
    """
    
    def __init__(self):
        """Initialize the tokenizer."""
        self._vocab: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._special_tokens: Dict[str, int] = {}
        self._token_to_id: Dict[str, int] = {}
    
    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self._vocab)
    
    @property
    def special_tokens(self) -> Dict[str, int]:
        """Dictionary of special tokens and their IDs."""
        return self._special_tokens
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """ID of the padding token, or None if not defined."""
        return self._special_tokens.get("<pad>")
    
    @property
    def unk_token_id(self) -> Optional[int]:
        """ID of the unknown token, or None if not defined."""
        return self._special_tokens.get("<unk>")
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """ID of the beginning-of-sequence token, or None if not defined."""
        return self._special_tokens.get("<s>")
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """ID of the end-of-sequence token, or None if not defined."""
        return self._special_tokens.get("</s>")
    
    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add special tokens (BOS, EOS)
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens (strings).
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token strings
        """
        pass
    
    @abstractmethod
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert token strings to token IDs.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to token strings.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of token strings
        """
        pass
    
    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of text strings
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of lists of token IDs
        """
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def batch_decode(self, token_ids_list: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token ID lists.
        
        Args:
            token_ids_list: List of lists of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded text strings
        """
        return [self.decode(ids, skip_special_tokens) for ids in token_ids_list]
    
    def pad(
        self,
        token_ids_list: List[List[int]],
        max_length: Optional[int] = None,
        padding: str = "right",
        pad_token_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Pad a batch of token ID lists to the same length.
        
        Args:
            token_ids_list: List of lists of token IDs
            max_length: Maximum length to pad to. If None, uses the longest sequence.
            padding: "left" or "right" - which side to pad
            pad_token_id: Token ID to use for padding. If None, uses the tokenizer's pad token.
            
        Returns:
            NumPy array of shape (batch_size, max_length) with padded sequences
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
            if pad_token_id is None:
                raise ValueError("No padding token defined and none provided")
        
        # Find max length
        if max_length is None:
            max_length = max(len(ids) for ids in token_ids_list)
        
        # Create padded array
        batch_size = len(token_ids_list)
        padded = np.full((batch_size, max_length), pad_token_id, dtype=np.int64)
        
        # Fill with actual tokens
        for i, ids in enumerate(token_ids_list):
            length = min(len(ids), max_length)
            if padding == "right":
                padded[i, :length] = ids[:length]
            else:  # left
                padded[i, -length:] = ids[-length:]
        
        return padded
    
    def create_attention_mask(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Create an attention mask for padded sequences.
        
        Args:
            token_ids: Padded token IDs array of shape (batch_size, seq_length)
            
        Returns:
            Attention mask of shape (batch_size, seq_length) where 1 = attend, 0 = don't attend
        """
        if self.pad_token_id is None:
            raise ValueError("No padding token defined")
        
        return (token_ids != self.pad_token_id).astype(np.int64)
    
    def __call__(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Alias for encode."""
        return self.encode(text, add_special_tokens)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the full vocabulary dictionary."""
        return self._vocab.copy()
    
    def get_id_to_token(self) -> Dict[int, str]:
        """Get the ID to token mapping."""
        return self._id_to_token.copy()
    
    def token_to_id(self, token: str) -> int:
        """Get the ID for a single token."""
        return self._vocab.get(token, self.unk_token_id)
    
    def id_to_token(self, token_id: int) -> str:
        """Get the token for a single ID."""
        return self._id_to_token.get(token_id, self._id_to_token.get(self.unk_token_id, "<unk>"))
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self._vocab
