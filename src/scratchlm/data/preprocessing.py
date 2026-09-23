"""
Text preprocessing for ScratchLM.

Provides functions for cleaning, normalizing, and filtering text data.
"""

import re
import unicodedata
from typing import List, Optional, Callable
from dataclasses import dataclass


@dataclass
class TextCleaner:
    """
    Configurable text cleaner for preprocessing.
    
    This class provides a flexible way to clean and normalize text
    with configurable options.
    """
    
    # Normalization options
    lowercase: bool = False
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    
    # Cleaning options
    remove_urls: bool = True
    remove_emails: bool = True
    remove_html: bool = True
    remove_excessive_whitespace: bool = True
    remove_control_chars: bool = True
    
    # Filtering options
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_chars: Optional[str] = None  # Regex pattern for allowed characters
    
    # Token-level options
    fix_apostrophes: bool = True
    fix_quotes: bool = True
    fix_dashes: bool = True
    
    def __call__(self, text: str) -> Optional[str]:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text, or None if text should be filtered out
        """
        # Apply all cleaning steps
        text = self._clean_text(text)
        
        # Check if text passes filters
        if not self._passes_filters(text):
            return None
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Apply all cleaning steps to text."""
        # Normalize unicode
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Remove control characters
        if self.remove_control_chars:
            text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Fix apostrophes
        if self.fix_apostrophes:
            text = re.sub(r"[`']", "'", text)
        
        # Fix quotes
        if self.fix_quotes:
            text = re.sub(r'"', '"', text)
        
        # Fix dashes
        if self.fix_dashes:
            text = re.sub(r'[-–—]', '-', text)
        
        # Remove excessive whitespace
        if self.remove_excessive_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _passes_filters(self, text: str) -> bool:
        """Check if text passes all filters."""
        # Length filters
        if self.min_length is not None and len(text) < self.min_length:
            return False
        if self.max_length is not None and len(text) > self.max_length:
            return False
        
        # Allowed characters filter
        if self.allowed_chars is not None:
            if not re.fullmatch(self.allowed_chars, text):
                return False
        
        return True


# Default cleaner for English text
DEFAULT_ENGLISH_CLEANER = TextCleaner(
    lowercase=False,
    normalize_unicode=True,
    normalize_whitespace=True,
    remove_urls=True,
    remove_emails=True,
    remove_html=True,
    remove_excessive_whitespace=True,
    remove_control_chars=True,
    fix_apostrophes=True,
    fix_quotes=True,
    fix_dashes=True,
    min_length=10,
    max_length=10000,
)


# Aggressive cleaner for very clean text
AGGRESSIVE_CLEANER = TextCleaner(
    lowercase=True,
    normalize_unicode=True,
    normalize_whitespace=True,
    remove_urls=True,
    remove_emails=True,
    remove_html=True,
    remove_excessive_whitespace=True,
    remove_control_chars=True,
    fix_apostrophes=True,
    fix_quotes=True,
    fix_dashes=True,
    min_length=20,
    max_length=5000,
    allowed_chars=None,  # Can be set to a regex pattern
)


def clean_text(
    text: str,
    cleaner: Optional[TextCleaner] = None,
) -> Optional[str]:
    """
    Clean text using the default cleaner.
    
    Args:
        text: Input text
        cleaner: Optional custom cleaner. If None, uses DEFAULT_ENGLISH_CLEANER.
        
    Returns:
        Cleaned text, or None if text should be filtered out
    """
    if cleaner is None:
        cleaner = DEFAULT_ENGLISH_CLEANER
    
    return cleaner(text)


def normalize_text(
    text: str,
    lowercase: bool = False,
    normalize_unicode: bool = True,
    normalize_whitespace: bool = True,
) -> str:
    """
    Normalize text with basic normalization.
    
    Args:
        text: Input text
        lowercase: Whether to lowercase
        normalize_unicode: Whether to normalize unicode
        normalize_whitespace: Whether to normalize whitespace
        
    Returns:
        Normalized text
    """
    if normalize_unicode:
        text = unicodedata.normalize('NFKC', text)
    
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    
    if lowercase:
        text = text.lower()
    
    return text


def filter_text(
    text: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    contains_alpha: bool = True,
    contains_word: bool = True,
    min_word_length: int = 3,
) -> bool:
    """
    Filter text based on various criteria.
    
    Args:
        text: Input text
        min_length: Minimum length
        max_length: Maximum length
        contains_alpha: Whether text must contain alphabetic characters
        contains_word: Whether text must contain words of min_word_length
        min_word_length: Minimum word length to consider
        
    Returns:
        True if text passes all filters, False otherwise
    """
    # Length filters
    if min_length is not None and len(text) < min_length:
        return False
    if max_length is not None and len(text) > max_length:
        return False
    
    # Alphabetic characters
    if contains_alpha and not re.search(r'[a-zA-Z]', text):
        return False
    
    # Word length
    if contains_word:
        words = re.findall(r'\b\w+\b', text)
        if not any(len(word) >= min_word_length for word in words):
            return False
    
    return True


def is_english_text(
    text: str,
    min_english_ratio: float = 0.8,
    common_english_words: Optional[List[str]] = None,
) -> bool:
    """
    Check if text appears to be English.
    
    This is a simple heuristic check based on common English words.
    
    Args:
        text: Input text
        min_english_ratio: Minimum ratio of English words to total words
        common_english_words: List of common English words to check for
        
    Returns:
        True if text appears to be English, False otherwise
    """
    if common_english_words is None:
        common_english_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which',
            'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
            'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good',
            'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
            'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well',
            'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give',
            'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had',
        ]
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    if not words:
        return False
    
    # Count English words
    english_count = sum(1 for word in words if word in common_english_words)
    
    # Check ratio
    ratio = english_count / len(words)
    
    return ratio >= min_english_ratio


def deduplicate_texts(
    texts: List[str],
    similarity_threshold: float = 0.9,
) -> List[str]:
    """
    Remove duplicate and near-duplicate texts.
    
    Args:
        texts: List of text strings
        similarity_threshold: Jaccard similarity threshold for near-duplicates
        
    Returns:
        List of unique texts
    """
    unique_texts = []
    seen_hashes = set()
    
    for text in texts:
        # Simple hash for exact duplicates
        text_hash = hash(text)
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_texts.append(text)
        # Note: For near-duplicates, we'd need a more sophisticated approach
        # like TF-IDF cosine similarity or minhash. This is a simplified version.
    
    return unique_texts


if __name__ == "__main__":
    # Test preprocessing
    test_texts = [
        "Hello, World! This is a test.",
        "Visit https://example.com for more info.",
        "Contact me at test@example.com.",
        "<p>This is <b>HTML</b> content.</p>",
        "The quick brown fox jumps over the lazy dog.",
        "  Extra   whitespace   here.  ",
        "It's a nice day, isn't it?",
        "Some \u2013 weird \u2014 dashes.",
    ]
    
    print("Testing text preprocessing...")
    print("=" * 60)
    
    for text in test_texts:
        cleaned = clean_text(text)
        print(f"Original: {repr(text)}")
        print(f"Cleaned:  {repr(cleaned)}")
        print()
    
    # Test English detection
    english_text = "The quick brown fox jumps over the lazy dog."
    non_english_text = "Bonjour le monde comment ca va?"
    
    print("=" * 60)
    print("Testing English detection...")
    print(f"English text: {is_english_text(english_text)}")
    print(f"Non-English text: {is_english_text(non_english_text)}")
