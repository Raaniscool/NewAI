"""
Data splitting utilities for ScratchLM.

Provides functions for creating train/validation/test splits.
"""

import random
import json
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import numpy as np

from scratchlm.utils import paths, set_seed
from scratchlm.utils.seed import get_seed


def create_splits(
    texts: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/validation/test splits from a list of texts.
    
    Args:
        texts: List of text strings
        train_ratio: Ratio of data for training (0-1)
        val_ratio: Ratio of data for validation (0-1)
        test_ratio: Ratio of data for testing (0-1)
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_texts, val_texts, test_texts)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Ratios must sum to 1, got {total_ratio}")
    
    # Set seed if provided
    if seed is not None:
        set_seed(seed)
    else:
        seed = get_seed()
    
    # Shuffle if requested
    if shuffle:
        texts = texts.copy()
        random.shuffle(texts)
    
    # Calculate split indices
    n = len(texts)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Create splits
    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]
    
    return train_texts, val_texts, test_texts


def create_splits_by_document(
    documents: List[List[str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create splits by keeping documents together.
    
    This ensures that all text from the same document goes to the same split,
    which is important for avoiding data leakage.
    
    Args:
        documents: List of documents, where each document is a list of text strings
        train_ratio: Ratio of documents for training
        val_ratio: Ratio of documents for validation
        test_ratio: Ratio of documents for testing
        shuffle: Whether to shuffle documents before splitting
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_texts, val_texts, test_texts) - flattened lists
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Ratios must sum to 1, got {total_ratio}")
    
    # Set seed
    if seed is not None:
        set_seed(seed)
    
    # Shuffle documents
    if shuffle:
        documents = documents.copy()
        random.shuffle(documents)
    
    # Calculate split indices
    n = len(documents)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split documents
    train_docs = documents[:train_end]
    val_docs = documents[train_end:val_end]
    test_docs = documents[val_end:]
    
    # Flatten
    train_texts = [text for doc in train_docs for text in doc]
    val_texts = [text for doc in val_docs for text in doc]
    test_texts = [text for doc in test_docs for text in doc]
    
    return train_texts, val_texts, test_texts


def save_splits(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    directory: Union[str, Path],
    name: str = "dataset",
) -> Path:
    """
    Save splits to a directory.
    
    Args:
        train_texts: Training texts
        val_texts: Validation texts
        test_texts: Test texts
        directory: Directory to save to
        name: Name for the dataset
        
    Returns:
        Path to the saved directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    # Save each split
    splits = {
        'train': train_texts,
        'val': val_texts,
        'test': test_texts,
    }
    
    for split_name, texts in splits.items():
        split_file = directory / f"{name}_{split_name}.txt"
        with open(split_file, 'w') as f:
            for text in texts:
                f.write(text + '\n')
    
    # Save metadata
    metadata = {
        'name': name,
        'split_sizes': {
            'train': len(train_texts),
            'val': len(val_texts),
            'test': len(test_texts),
        },
        'total': len(train_texts) + len(val_texts) + len(test_texts),
    }
    
    metadata_file = directory / f"{name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return directory


def load_splits(
    directory: Union[str, Path],
    name: str = "dataset",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load splits from a directory.
    
    Args:
        directory: Directory containing split files
        name: Name of the dataset
        
    Returns:
        Tuple of (train_texts, val_texts, test_texts)
    """
    directory = Path(directory)
    
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_file = directory / f"{name}_{split_name}.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
            splits[split_name] = texts
        else:
            splits[split_name] = []
    
    return splits['train'], splits['val'], splits['test']


def load_split_metadata(
    directory: Union[str, Path],
    name: str = "dataset",
) -> dict:
    """
    Load split metadata.
    
    Args:
        directory: Directory containing split files
        name: Name of the dataset
        
    Returns:
        Metadata dictionary
    """
    directory = Path(directory)
    metadata_file = directory / f"{name}_metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    return {}


def stratified_split(
    texts: List[str],
    labels: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """
    Create stratified splits based on labels.
    
    This ensures that each split has approximately the same distribution of labels.
    
    Args:
        texts: List of text strings
        labels: List of label values (same length as texts)
        train_ratio: Ratio for training
        val_ratio: Ratio for validation
        test_ratio: Ratio for testing
        shuffle: Whether to shuffle within each label group
        seed: Random seed
        
    Returns:
        Tuple of (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
    """
    # Validate
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    
    # Group by label
    label_groups = {}
    for text, label in zip(texts, labels):
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(text)
    
    # Create splits for each label
    train_texts = []
    val_texts = []
    test_texts = []
    train_labels = []
    val_labels = []
    test_labels = []
    
    for label, label_texts in label_groups.items():
        # Split this label group
        t, v, te = create_splits(
            label_texts,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            shuffle=shuffle,
            seed=seed,
        )
        
        train_texts.extend(t)
        val_texts.extend(v)
        test_texts.extend(te)
        train_labels.extend([label] * len(t))
        val_labels.extend([label] * len(v))
        test_labels.extend([label] * len(te))
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


if __name__ == "__main__":
    # Test splitting functions
    print("Testing data splitting...")
    
    # Create sample data
    texts = [f"Text {i}" for i in range(100)]
    
    # Test create_splits
    print("\n=== create_splits ===")
    train, val, test = create_splits(texts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Test create_splits_by_document
    print("\n=== create_splits_by_document ===")
    documents = [[f"Doc {i}_Line {j}" for j in range(5)] for i in range(20)]
    train, val, test = create_splits_by_document(documents, seed=42)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Test save/load splits
    print("\n=== save/load splits ===")
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_splits(train, val, test, tmpdir, "test_dataset")
        print(f"Saved splits to {tmpdir}")
        
        loaded_train, loaded_val, loaded_test = load_splits(tmpdir, "test_dataset")
        print(f"Loaded - Train: {len(loaded_train)}, Val: {len(loaded_val)}, Test: {len(loaded_test)}")
        
        metadata = load_split_metadata(tmpdir, "test_dataset")
        print(f"Metadata: {metadata}")
