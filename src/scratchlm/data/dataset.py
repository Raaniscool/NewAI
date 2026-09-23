"""
Dataset classes for ScratchLM.

Provides PyTorch Dataset classes for text data.
"""

import random
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from scratchlm.tokenizer import BaseTokenizer
from scratchlm.utils import paths


class TextDataset(Dataset):
    """
    Dataset of raw text strings.
    
    This dataset stores text strings and provides them for processing.
    Useful for initial data loading before tokenization.
    """
    
    def __init__(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ):
        """
        Initialize the text dataset.
        
        Args:
            texts: List of text strings
            max_length: Optional maximum length for filtering
        """
        self.texts = texts
        self.max_length = max_length
        
        # Filter texts by length if specified
        if max_length is not None:
            self.texts = [t for t in self.texts if len(t) <= max_length]
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]
    
    def get_batch(self, indices: List[int]) -> List[str]:
        """Get a batch of texts by indices."""
        return [self.texts[i] for i in indices]


class TokenizedDataset(Dataset):
    """
    Dataset of tokenized text for language modeling.
    
    This dataset stores tokenized text and provides batches for training.
    Each sample is a sequence of token IDs with optional attention masks.
    """
    
    def __init__(
        self,
        token_ids_list: List[List[int]],
        context_length: int,
        tokenizer: Optional[BaseTokenizer] = None,
        pad_token_id: Optional[int] = None,
    ):
        """
        Initialize the tokenized dataset.
        
        Args:
            token_ids_list: List of token ID sequences
            context_length: Maximum sequence length (for truncation/padding)
            tokenizer: Optional tokenizer (for padding info)
            pad_token_id: Token ID for padding. If None, uses tokenizer's pad token.
        """
        self.token_ids_list = token_ids_list
        self.context_length = context_length
        self.tokenizer = tokenizer
        
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        elif tokenizer is not None:
            self.pad_token_id = tokenizer.pad_token_id
        else:
            raise ValueError("Either tokenizer or pad_token_id must be provided")
        
        # Pre-process all sequences to the same length
        self._processed_sequences = []
        for token_ids in token_ids_list:
            processed = self._process_sequence(token_ids)
            self._processed_sequences.append(processed)
    
    def _process_sequence(self, token_ids: List[int]) -> np.ndarray:
        """
        Process a token sequence to the target length.
        
        Truncates if too long, pads if too short.
        """
        # Truncate if necessary
        if len(token_ids) > self.context_length:
            token_ids = token_ids[:self.context_length]
        
        # Pad if necessary
        if len(token_ids) < self.context_length:
            padding = [self.pad_token_id] * (self.context_length - len(token_ids))
            token_ids = token_ids + padding
        
        return np.array(token_ids, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self._processed_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a sample for language modeling.
        
        Returns:
            Tuple of (input_ids, labels) where labels is input_ids shifted by 1
        """
        sequence = self._processed_sequences[idx]
        
        # For language modeling, labels are input shifted by 1
        # Input: [x1, x2, x3, ..., xn]
        # Labels: [x2, x3, ..., xn, pad]
        input_ids = sequence[:-1]
        labels = sequence[1:]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).astype(np.int64)
        
        return input_ids, labels, attention_mask
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a batch of samples.
        
        Returns:
            Tuple of (input_ids, labels, attention_mask) arrays
        """
        input_ids = []
        labels = []
        attention_masks = []
        
        for idx in indices:
            ipt, lbl, mask = self[idx]
            input_ids.append(ipt)
            labels.append(lbl)
            attention_masks.append(mask)
        
        return (
            np.stack(input_ids),
            np.stack(labels),
            np.stack(attention_masks),
        )


class SlidingWindowDataset(Dataset):
    """
    Dataset that creates sliding window samples from long texts.
    
    This is useful for training on long documents without splitting them
    into separate sequences. Each window becomes a separate training sample.
    """
    
    def __init__(
        self,
        token_ids_list: List[List[int]],
        context_length: int,
        stride: int = 1,
        pad_token_id: int = 0,
    ):
        """
        Initialize the sliding window dataset.
        
        Args:
            token_ids_list: List of token ID sequences (each can be long)
            context_length: Length of each window
            stride: Stride between consecutive windows
            pad_token_id: Token ID for padding
        """
        self.context_length = context_length
        self.stride = stride
        self.pad_token_id = pad_token_id
        
        # Create all windows
        self.windows = []
        for token_ids in token_ids_list:
            windows = self._create_windows(token_ids)
            self.windows.extend(windows)
    
    def _create_windows(self, token_ids: List[int]) -> List[np.ndarray]:
        """Create sliding windows from a token sequence."""
        windows = []
        
        # Skip sequences shorter than context_length
        if len(token_ids) < self.context_length:
            # Pad and add as single window
            padded = token_ids + [self.pad_token_id] * (self.context_length - len(token_ids))
            windows.append(np.array(padded, dtype=np.int64))
            return windows
        
        # Create windows with stride
        for start in range(0, len(token_ids) - self.context_length + 1, self.stride):
            end = start + self.context_length
            window = token_ids[start:end]
            windows.append(np.array(window, dtype=np.int64))
        
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a window for language modeling.
        
        Returns:
            Tuple of (input_ids, labels, attention_mask)
        """
        window = self.windows[idx]
        
        # Input is all tokens except last
        input_ids = window[:-1]
        
        # Labels are all tokens except first (shifted by 1)
        labels = window[1:]
        
        # Attention mask (all 1s since we don't pad windows)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        
        return input_ids, labels, attention_mask


class RandomBatchSampler:
    """
    Sampler that yields random batches from a dataset.
    
    This is useful for training when you want to sample batches
    without replacement.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize the batch sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices each epoch
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)
        
        self.current_idx = 0
    
    def __iter__(self):
        """Iterate through batches."""
        while True:
            batch = self.next_batch()
            if batch is None:
                break
            yield batch
    
    def next_batch(self) -> Optional[List[int]]:
        """
        Get the next batch of indices.
        
        Returns:
            List of indices for the next batch, or None if done
        """
        if self.current_idx >= len(self.indices):
            return None
        
        end_idx = self.current_idx + self.batch_size
        if end_idx > len(self.indices):
            if self.drop_last:
                return None
            end_idx = len(self.indices)
        
        batch = self.indices[self.current_idx:end_idx]
        self.current_idx = end_idx
        
        return batch
    
    def reset(self):
        """Reset the sampler for a new epoch."""
        self.current_idx = 0
        if self.shuffle:
            random.shuffle(self.indices)


if __name__ == "__main__":
    # Test datasets
    print("Testing dataset classes...")
    
    # Create sample data
    texts = [
        "Hello world",
        "This is a test",
        "The quick brown fox",
        "jumps over the lazy dog",
    ]
    
    # Test TextDataset
    print("\n=== TextDataset ===")
    text_dataset = TextDataset(texts)
    print(f"Length: {len(text_dataset)}")
    print(f"Sample 0: {text_dataset[0]}")
    
    # Test TokenizedDataset (without actual tokenizer)
    print("\n=== TokenizedDataset ===")
    # Simulate tokenized data
    token_ids_list = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11, 12],
        [13, 14],
    ]
    
    tokenized_dataset = TokenizedDataset(
        token_ids_list,
        context_length=5,
        pad_token_id=0,
    )
    
    print(f"Length: {len(tokenized_dataset)}")
    input_ids, labels, mask = tokenized_dataset[0]
    print(f"Sample 0 input_ids: {input_ids}")
    print(f"Sample 0 labels: {labels}")
    print(f"Sample 0 mask: {mask}")
    
    # Test SlidingWindowDataset
    print("\n=== SlidingWindowDataset ===")
    long_sequence = list(range(20))  # [0, 1, 2, ..., 19]
    sliding_dataset = SlidingWindowDataset(
        [long_sequence],
        context_length=5,
        stride=2,
    )
    
    print(f"Length: {len(sliding_dataset)}")
    print(f"Sample 0: {sliding_dataset[0]}")
    print(f"Sample 1: {sliding_dataset[1]}")
    print(f"Sample 2: {sliding_dataset[2]}")
    
    # Test RandomBatchSampler
    print("\n=== RandomBatchSampler ===")
    sampler = RandomBatchSampler(text_dataset, batch_size=2, shuffle=False)
    batch = sampler.next_batch()
    print(f"Batch 1: {batch}")
    batch = sampler.next_batch()
    print(f"Batch 2: {batch}")
    sampler.reset()
    batch = sampler.next_batch()
    print(f"Batch 1 (after reset): {batch}")
