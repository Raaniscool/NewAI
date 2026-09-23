"""
Byte Pair Encoding (BPE) Tokenizer for ScratchLM.

Implements BPE tokenization from scratch, including training.
"""

import re
import json
import heapq
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import numpy as np

from .base import BaseTokenizer
from scratchlm.utils.paths import paths


class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding (BPE) Tokenizer.
    
    This implements BPE tokenization as described in the original paper:
    "Neural Machine Translation of Rare Word Problem Using Subword Units"
    by Rico Sennrich, Barry Haddow, and Alexandra Birch (2016).
    
    The tokenizer works by:
    1. Starting with a base vocabulary of individual characters
    2. Iteratively merging the most frequent pair of tokens
    3. Building up a vocabulary of subword units
    
    Example:
        >>> tokenizer = BPETokenizer()
        >>> tokenizer.train(["hello world", "hello there"], vocab_size=20)
        >>> tokenizer.encode("hello world")
        [token_ids...]
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        min_frequency: int = 2,
    ):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (including special tokens)
            pad_token: Padding token string
            unk_token: Unknown token string
            bos_token: Beginning of sequence token string
            eos_token: End of sequence token string
            min_frequency: Minimum frequency for a pair to be considered for merging
        """
        super().__init__()
        
        # Configuration
        self._target_vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Special tokens
        self._special_tokens = {
            pad_token: 0,
            unk_token: 1,
            bos_token: 2,
            eos_token: 3,
        }
        
        # Base vocabulary starts with special tokens
        self._vocab = self._special_tokens.copy()
        self._id_to_token = {v: k for k, v in self._special_tokens.items()}
        
        # Merges: maps (left, right) token IDs to new token ID
        self._merges: Dict[Tuple[int, int], int] = {}
        
        # Reverse merges: maps token ID to (left, right) token IDs
        self._reverse_merges: Dict[int, Tuple[int, int]] = {}
        
        # Token frequency counts (for training)
        self._token_counts: Counter = Counter()
        
        # Pair frequency counts (for training)
        self._pair_counts: Counter = Counter()
        
        # Training texts (stored for count updates)
        self._training_texts: List[str] = []
        
        # Base vocabulary (characters) - will be built during training
        self._base_vocab: Dict[str, int] = {}
        
        # Next available token ID
        self._next_id = len(self._special_tokens)
        
        # Pattern for splitting text into base tokens (characters)
        self._pattern = re.compile(r"''|'|[^\w' ]|")
    
    def train(
        self,
        texts: List[str],
        num_merges: Optional[int] = None,
        show_progress: bool = True,
    ):
        """
        Train the BPE tokenizer on a list of texts.
        
        This performs the BPE algorithm:
        1. Split all texts into base tokens (characters)
        2. Count token frequencies
        3. Iteratively merge the most frequent pair
        4. Update counts and repeat until vocab_size is reached
        
        Args:
            texts: List of text strings to train on
            num_merges: Number of merge operations to perform.
                       If None, will merge until vocab_size is reached.
            show_progress: Whether to print progress during training
        """
        # Store training texts for count updates
        self._training_texts = texts
        
        if num_merges is None:
            # Calculate how many merges we need
            # Start with special tokens + unique characters
            # Then merge until we reach vocab_size
            num_merges = self._target_vocab_size - len(self._special_tokens)
        
        # Ensure we have at least some merges
        if num_merges < 0:
            num_merges = 0
        
        # Step 1: Build initial vocabulary from characters
        self._build_base_vocab(texts)
        
        # Note: _build_base_vocab already initializes counts, so we don't call _initialize_counts
        
        # Step 3: Perform merges
        for i in range(num_merges):
            if len(self._vocab) >= self._target_vocab_size:
                break
            
            # Find the most frequent pair
            most_frequent_pair, pair_count = self._find_most_frequent_pair()
            
            if most_frequent_pair is None or pair_count < self.min_frequency:
                if show_progress:
                    print(f"No more frequent pairs (min_frequency={self.min_frequency})")
                break
            
            # Merge the pair
            self._merge_pair(most_frequent_pair)
            
            # Rebuild counts after merge
            self._rebuild_counts()
            
            if show_progress and (i + 1) % 100 == 0:
                print(f"Merge {i + 1}/{num_merges} | Vocab size: {len(self._vocab)} | "
                      f"Merged: {most_frequent_pair} -> {self._id_to_token[self._next_id - 1]}")
        
        if show_progress:
            print(f"\nTraining complete!")
            print(f"Final vocabulary size: {len(self._vocab)}")
            print(f"Number of merges performed: {len(self._merges)}")
    
    def _build_base_vocab(self, texts: List[str]):
        """Build the base vocabulary from characters in the texts."""
        # Get all unique characters
        chars = set()
        for text in texts:
            # Split into characters using our pattern
            base_tokens = self._split_into_base_tokens(text)
            chars.update(base_tokens)
        
        # Add characters to vocabulary
        for char in sorted(chars):
            if char not in self._vocab:
                self._vocab[char] = self._next_id
                self._id_to_token[self._next_id] = char
                self._next_id += 1
        
        # Store base vocabulary
        self._base_vocab = {k: v for k, v in self._vocab.items() 
                           if k not in self._special_tokens}
        
        # Initialize counts
        self._token_counts = Counter()
        self._pair_counts = Counter()
        
        # Count initial token frequencies
        for text in texts:
            base_tokens = self._split_into_base_tokens(text)
            token_ids = [self._vocab.get(token, self.unk_token_id) for token in base_tokens]
            self._token_counts.update(token_ids)
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                self._pair_counts[pair] += 1
    
    def _split_into_base_tokens(self, text: str) -> List[str]:
        """
        Split text into base tokens (characters).
        
        This uses a simple regex pattern to split on word boundaries
        and punctuation.
        """
        # Use regex to find all tokens
        # This pattern matches: apostrophes (as separate tokens), 
        # individual non-word/non-apostrophe/non-space characters,
        # and word characters
        tokens = []
        for match in self._pattern.finditer(text):
            token = match.group()
            if token:
                tokens.append(token)
        
        # Also add any remaining characters
        # This handles cases where the pattern doesn't match everything
        remaining = self._pattern.sub('', text)
        if remaining:
            tokens.extend(list(remaining))
        
        # Filter out empty strings and whitespace
        tokens = [t for t in tokens if t.strip() or t in ' \t\n']
        
        return tokens
    
    def _initialize_counts(self, texts: List[str]):
        """Initialize token and pair frequency counts."""
        self._token_counts = Counter()
        self._pair_counts = Counter()
        
        for text in texts:
            # Split into base tokens
            base_tokens = self._split_into_base_tokens(text)
            
            # Convert to token IDs
            token_ids = []
            for token in base_tokens:
                if token in self._vocab:
                    token_ids.append(self._vocab[token])
                else:
                    # Unknown token - use UNK
                    token_ids.append(self.unk_token_id)
            
            # Update token counts
            self._token_counts.update(token_ids)
            
            # Update pair counts
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                self._pair_counts[pair] += 1
    
    def _find_most_frequent_pair(self) -> Tuple[Optional[Tuple[int, int]], int]:
        """
        Find the most frequent pair of consecutive token IDs.
        
        Returns:
            Tuple of (pair, count) or (None, 0) if no pairs found
        """
        if not self._pair_counts:
            return None, 0
        
        # Get the most common pair
        most_common = self._pair_counts.most_common(1)
        if not most_common:
            return None, 0
        
        pair, count = most_common[0]
        return pair, count
    
    def _merge_pair(self, pair: Tuple[int, int]):
        """
        Merge a pair of token IDs into a new token.
        
        Args:
            pair: Tuple of (left_token_id, right_token_id) to merge
        """
        left_id, right_id = pair
        
        # Create new token ID
        new_id = self._next_id
        self._next_id += 1
        
        # Create new token string
        left_token = self._id_to_token[left_id]
        right_token = self._id_to_token[right_id]
        new_token = left_token + right_token
        
        # Add to vocabulary
        self._vocab[new_token] = new_id
        self._id_to_token[new_id] = new_token
        
        # Record the merge
        self._merges[pair] = new_id
        self._reverse_merges[new_id] = pair
        
        # Update counts: replace all occurrences of the pair with the new token
        self._update_counts_after_merge(left_id, right_id, new_id)
    
    def _rebuild_counts(self):
        """
        Rebuild token and pair counts from the training texts.
        
        This is called after each merge to update counts with the new vocabulary.
        """
        self._token_counts = Counter()
        self._pair_counts = Counter()
        
        # Re-tokenize all training texts with current vocabulary and count
        for text in self._training_texts:
            # Tokenize using current vocab (which includes merges so far)
            token_ids = self.encode(text, add_special_tokens=False)
            
            # Update token counts
            self._token_counts.update(token_ids)
            
            # Update pair counts
            for j in range(len(token_ids) - 1):
                pair = (token_ids[j], token_ids[j + 1])
                self._pair_counts[pair] += 1
    
    def _update_counts_after_merge(
        self,
        left_id: int,
        right_id: int,
        new_id: int,
    ):
        """
        Update counts after a merge by rebuilding from training texts.
        
        This is a simplified approach - we rebuild all counts from scratch.
        """
        # Just call rebuild_counts
        self._rebuild_counts()
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text into token IDs using BPE.
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS and EOS tokens
            
        Returns:
            List of token IDs
        """
        # Split into base tokens
        base_tokens = self._split_into_base_tokens(text)
        
        # Convert to token IDs
        token_ids = []
        for token in base_tokens:
            token_id = self._vocab.get(token, self.unk_token_id)
            token_ids.append(token_id)
        
        # Apply BPE merges
        token_ids = self._apply_merges(token_ids)
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        return token_ids
    
    def _apply_merges(self, token_ids: List[int]) -> List[int]:
        """
        Apply BPE merges to a sequence of token IDs.
        
        This greedily applies the merge operations in the order they were learned.
        
        Args:
            token_ids: List of token IDs (base tokens)
            
        Returns:
            List of token IDs after applying merges
        """
        # Start with the original token IDs
        tokens = token_ids.copy()
        
        # Apply merges in order (from most frequent to least)
        # We need to check for each merge if it can be applied
        # and apply it greedily
        
        # Create a list of merges sorted by when they were added
        # (earlier merges are more frequent)
        merges_list = []
        for (left, right), new_id in self._merges.items():
            merges_list.append(((left, right), new_id))
        
        # Sort by merge order (earlier merges first)
        # Since we add merges in order, we can just use the order in _merges
        # But dict doesn't preserve order in Python < 3.7, so we'll use a list
        
        # For efficiency, we'll use a different approach:
        # Build a map of which merges are possible at each position
        
        changed = True
        while changed:
            changed = False
            
            # Check for each merge
            for (left, right), new_id in merges_list:
                i = 0
                new_tokens = []
                
                while i < len(tokens):
                    # Check if current and next tokens match the merge
                    if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                        new_tokens.append(new_id)
                        i += 2
                        changed = True
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                
                tokens = new_tokens
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in self._special_tokens.values():
                continue
            token = self._id_to_token.get(token_id, self.unk_token)
            tokens.append(token)
        
        return "".join(tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into token strings (not IDs).
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token strings
        """
        token_ids = self.encode(text, add_special_tokens=False)
        return [self._id_to_token.get(id, self.unk_token) for id in token_ids]
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert token strings to token IDs.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token IDs
        """
        token_ids = []
        for token in tokens:
            # Try to find the token in vocabulary
            if token in self._vocab:
                token_ids.append(self._vocab[token])
            else:
                # Try to split into sub-tokens
                # This is a simplified approach
                # A proper implementation would use the merge rules in reverse
                sub_tokens = []
                remaining = token
                
                # Try to find the longest matching token
                found = False
                for length in range(len(remaining), 0, -1):
                    for start in range(len(remaining) - length + 1):
                        substring = remaining[start:start + length]
                        if substring in self._vocab:
                            sub_tokens.append(self._vocab[substring])
                            remaining = remaining[:start] + remaining[start + length:]
                            found = True
                            break
                    if found:
                        break
                
                if not remaining:
                    token_ids.extend(sub_tokens)
                else:
                    # Fall back to UNK
                    token_ids.append(self.unk_token_id)
        
        return token_ids
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to token strings.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of token strings
        """
        return [self._id_to_token.get(id, self.unk_token) for id in ids]
    
    def save(self, directory: Path):
        """
        Save the tokenizer to a directory.
        
        Args:
            directory: Directory to save tokenizer files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        vocab_path = directory / "vocab.json"
        vocab_data = {
            "vocab": self._vocab,
            "id_to_token": self._id_to_token,
            "special_tokens": self._special_tokens,
        }
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        # Save merges
        merges_path = directory / "merges.txt"
        with open(merges_path, 'w') as f:
            for (left, right), new_id in sorted(self._merges.items(), key=lambda x: x[1]):
                left_token = self._id_to_token[left]
                right_token = self._id_to_token[right]
                new_token = self._id_to_token[new_id]
                f.write(f"{left_token} {right_token} -> {new_token} ({left} {right} -> {new_id})\n")
        
        # Save config
        config_path = directory / "config.json"
        config = {
            "vocab_size": self._target_vocab_size,
            "min_frequency": self.min_frequency,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "tokenizer_type": "bpe",
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, directory: Path) -> 'BPETokenizer':
        """
        Load a tokenizer from a directory.
        
        Args:
            directory: Directory containing tokenizer files
            
        Returns:
            Loaded BPETokenizer instance
        """
        directory = Path(directory)
        
        # Load config
        config_path = directory / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create tokenizer
        tokenizer = cls(
            vocab_size=config.get("vocab_size", 8192),
            min_frequency=config.get("min_frequency", 2),
            pad_token=config.get("pad_token", "<pad>"),
            unk_token=config.get("unk_token", "<unk>"),
            bos_token=config.get("bos_token", "<s>"),
            eos_token=config.get("eos_token", "</s>"),
        )
        
        # Load vocabulary
        vocab_path = directory / "vocab.json"
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        tokenizer._vocab = vocab_data["vocab"]
        tokenizer._id_to_token = vocab_data["id_to_token"]
        tokenizer._special_tokens = vocab_data["special_tokens"]
        
        # Update next_id
        tokenizer._next_id = max(tokenizer._vocab.values()) + 1
        
        # Load merges
        merges_path = directory / "merges.txt"
        if merges_path.exists():
            with open(merges_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse merge line: "a b -> ab (id1 id2 -> id3)"
                    # This is a simplified parse
                    parts = line.split("->")
                    if len(parts) < 2:
                        continue
                    
                    left_right = parts[0].strip().split()
                    new_token_part = parts[1].strip().split()[0]
                    
                    if len(left_right) == 2:
                        left_token, right_token = left_right
                        new_token = new_token_part
                        
                        # Find IDs
                        left_id = tokenizer._vocab.get(left_token)
                        right_id = tokenizer._vocab.get(right_token)
                        new_id = tokenizer._vocab.get(new_token)
                        
                        if left_id is not None and right_id is not None and new_id is not None:
                            tokenizer._merges[(left_id, right_id)] = new_id
                            tokenizer._reverse_merges[new_id] = (left_id, right_id)
        
        return tokenizer
    
    def get_merge_rules(self) -> List[Tuple[str, str, str]]:
        """
        Get all merge rules as (left, right, new) token strings.
        
        Returns:
            List of merge rules
        """
        rules = []
        for (left_id, right_id), new_id in sorted(self._merges.items(), key=lambda x: x[1]):
            left = self._id_to_token[left_id]
            right = self._id_to_token[right_id]
            new = self._id_to_token[new_id]
            rules.append((left, right, new))
        return rules


def train_bpe_tokenizer(
    texts: List[str],
    vocab_size: int = 8192,
    min_frequency: int = 2,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
    bos_token: str = "<s>",
    eos_token: str = "</s>",
    show_progress: bool = True,
) -> BPETokenizer:
    """
    Convenience function to train a BPE tokenizer.
    
    Args:
        texts: List of text strings to train on
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a pair to be merged
        pad_token: Padding token string
        unk_token: Unknown token string
        bos_token: Beginning of sequence token string
        eos_token: End of sequence token string
        show_progress: Whether to show training progress
        
    Returns:
        Trained BPETokenizer instance
    """
    tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        pad_token=pad_token,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
    )
    
    tokenizer.train(texts, show_progress=show_progress)
    
    return tokenizer


if __name__ == "__main__":
    # Test the BPE tokenizer
    print("Testing BPE Tokenizer...")
    
    # Simple test texts
    texts = [
        "hello world",
        "hello there",
        "the quick brown fox jumps over the lazy dog",
        "hello world hello world",
        "this is a test of the tokenizer",
    ]
    
    # Train tokenizer
    print("\nTraining tokenizer...")
    tokenizer = train_bpe_tokenizer(
        texts,
        vocab_size=50,  # Small for testing
        min_frequency=1,
        show_progress=True,
    )
    
    # Test encoding
    print("\nTesting encoding...")
    test_text = "hello world"
    token_ids = tokenizer.encode(test_text)
    print(f"Text: {test_text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {tokenizer.decode(token_ids)}")
    
    # Test tokenization
    print("\nTokenization:")
    tokens = tokenizer.tokenize(test_text)
    print(f"Tokens: {tokens}")
    
    # Show vocabulary
    print(f"\nVocabulary size: {len(tokenizer._vocab)}")
    print("\nFirst 20 vocabulary items:")
    for token, id in list(tokenizer._vocab.items())[:20]:
        print(f"  {id}: {token}")
    
    # Show merges
    print(f"\nNumber of merges: {len(tokenizer._merges)}")
    print("\nFirst 10 merges:")
    for i, (pair, new_id) in enumerate(list(tokenizer._merges.items())[:10]):
        left, right = pair
        left_tok = tokenizer._id_to_token[left]
        right_tok = tokenizer._id_to_token[right]
        new_tok = tokenizer._id_to_token[new_id]
        print(f"  {left_tok} + {right_tok} -> {new_tok}")
