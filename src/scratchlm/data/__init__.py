"""
Data pipeline for ScratchLM.

Handles data loading, preprocessing, splitting, and dataset creation.
"""

from .preprocessing import clean_text, normalize_text, filter_text, deduplicate_texts, is_english_text
from .dataset import TextDataset, TokenizedDataset, SlidingWindowDataset
from .splits import create_splits, load_splits, save_splits, load_split_metadata
from .loader import load_text_file, load_text_directory, save_processed_text, iterate_text_files, get_file_stats

__all__ = [
    "clean_text",
    "normalize_text",
    "filter_text",
    "TextDataset",
    "TokenizedDataset",
    "create_splits",
    "load_splits",
    "load_text_file",
    "load_text_directory",
    "save_processed_text",
]
