"""
Deduplication utilities for ScratchLM corpus engineering.

Provides exact hash-based deduplication and MinHash/Jaccard n-gram 
near-duplicate detection to prevent duplicate passages in training data.
"""

import hashlib
import re
from typing import List, Set, Tuple, Dict, Any, Optional
from dataclasses import dataclass


def get_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character or word n-grams from text."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < n:
        return set(words)
    return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""
    unique_documents: List[Any]
    exact_duplicates_removed: int
    near_duplicates_removed: int
    total_original: int
    total_retained: int
    duplicate_rate_pct: float


class CorpusDeduplicator:
    """
    Deduplicator for corpus documents using both exact hashing and
    n-gram Jaccard similarity for near-duplicate detection.
    """

    def __init__(self, near_dup_threshold: float = 0.8, ngram_size: int = 3):
        self.near_dup_threshold = near_dup_threshold
        self.ngram_size = ngram_size

    def deduplicate(self, documents: List[Any], text_key: str = "text") -> Tuple[List[Any], DeduplicationResult]:
        """
        Deduplicate a list of document objects or dictionaries.
        
        Args:
            documents: List of document objects (or dicts)
            text_key: Attribute or key containing text
            
        Returns:
            Tuple of (deduplicated_documents, DeduplicationResult)
        """
        total_original = len(documents)
        seen_exact_hashes: Set[str] = set()
        exact_unique_docs: List[Any] = []
        exact_dups = 0

        # Pass 1: Exact deduplication via normalized SHA-256
        for doc in documents:
            text = getattr(doc, text_key, None) or doc.get(text_key, "")
            norm_text = re.sub(r'\s+', ' ', text.strip().lower())
            doc_hash = hashlib.sha256(norm_text.encode('utf-8')).hexdigest()

            if doc_hash in seen_exact_hashes:
                exact_dups += 1
            else:
                seen_exact_hashes.add(doc_hash)
                exact_unique_docs.append(doc)

        # Pass 2: Near-duplicate detection using Jaccard n-gram comparison
        near_unique_docs: List[Any] = []
        doc_ngrams: List[Set[str]] = []
        near_dups = 0

        for doc in exact_unique_docs:
            text = getattr(doc, text_key, None) or doc.get(text_key, "")
            ngrams = get_ngrams(text, n=self.ngram_size)

            is_near_dup = False
            for existing_ngrams in doc_ngrams:
                sim = jaccard_similarity(ngrams, existing_ngrams)
                if sim >= self.near_dup_threshold:
                    is_near_dup = True
                    break

            if is_near_dup:
                near_dups += 1
            else:
                near_unique_docs.append(doc)
                doc_ngrams.append(ngrams)

        total_retained = len(near_unique_docs)
        total_removed = exact_dups + near_dups
        dup_rate = (total_removed / total_original * 100.0) if total_original > 0 else 0.0

        stats = DeduplicationResult(
            unique_documents=near_unique_docs,
            exact_duplicates_removed=exact_dups,
            near_duplicates_removed=near_dups,
            total_original=total_original,
            total_retained=total_retained,
            duplicate_rate_pct=round(dup_rate, 2),
        )

        return near_unique_docs, stats
