"""
Evaluation metrics for ScratchLM.

Implements various metrics for evaluating language model performance.
"""

import math
from typing import List, Optional, Callable
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class MetricResult:
    """Result from a metric calculation."""
    name: str
    value: float
    description: str = ""
    
    def __str__(self):
        return f"{self.name}: {self.value:.4f}"


class BaseMetric:
    """Base class for all metrics."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    def compute(self, *args, **kwargs) -> MetricResult:
        """Compute the metric."""
        raise NotImplementedError
    
    def reset(self):
        """Reset the metric state."""
        pass


class PerplexityMetric(BaseMetric):
    """
    Perplexity metric.
    
    Perplexity is the exponential of the average loss.
    Lower perplexity indicates better performance.
    """
    
    def __init__(self):
        super().__init__(
            name="perplexity",
            description="Exponential of average loss. Lower is better.",
        )
    
    def compute(self, loss: float) -> MetricResult:
        """
        Compute perplexity from loss.
        
        Args:
            loss: Average cross-entropy loss
            
        Returns:
            Perplexity value
        """
        perplexity = math.exp(loss)
        return MetricResult(
            name=self.name,
            value=perplexity,
            description=self.description,
        )


class AccuracyMetric(BaseMetric):
    """
    Accuracy metric.
    
    Measures the fraction of correctly predicted tokens.
    """
    
    def __init__(self, ignore_index: Optional[int] = None):
        super().__init__(
            name="accuracy",
            description="Fraction of correctly predicted tokens.",
        )
        self.ignore_index = ignore_index
        self.correct = 0
        self.total = 0
    
    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> MetricResult:
        """
        Compute accuracy.
        
        Args:
            logits: Model predictions of shape (batch_size, seq_len, vocab_size)
            targets: Ground truth of shape (batch_size, seq_len)
            
        Returns:
            Accuracy value
        """
        # Get predictions
        preds = torch.argmax(logits, dim=-1)
        
        # Create mask for non-ignored tokens
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            correct = (preds == targets) & mask
            total = mask.sum()
        else:
            correct = (preds == targets)
            total = targets.numel()
        
        # Convert to Python scalars
        correct_count = correct.sum().item()
        total_count = total.item() if isinstance(total, torch.Tensor) else total
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        return MetricResult(
            name=self.name,
            value=accuracy,
            description=self.description,
        )
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update metric state with a batch."""
        preds = torch.argmax(logits, dim=-1)
        
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            self.correct += ((preds == targets) & mask).sum().item()
            self.total += mask.sum().item()
        else:
            self.correct += (preds == targets).sum().item()
            self.total += targets.numel()
    
    def get_value(self) -> float:
        """Get current accuracy value."""
        return self.correct / self.total if self.total > 0 else 0.0
    
    def reset(self):
        self.correct = 0
        self.total = 0


class LossMetric(BaseMetric):
    """
    Loss metric.
    
    Tracks the average loss over batches.
    """
    
    def __init__(self, ignore_index: Optional[int] = None):
        super().__init__(
            name="loss",
            description="Average cross-entropy loss.",
        )
        self.ignore_index = ignore_index
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def compute(self, loss: float, num_tokens: int) -> MetricResult:
        """
        Compute average loss.
        
        Args:
            loss: Total loss
            num_tokens: Number of tokens
            
        Returns:
            Average loss
        """
        avg_loss = loss / num_tokens if num_tokens > 0 else 0.0
        return MetricResult(
            name=self.name,
            value=avg_loss,
            description=self.description,
        )
    
    def update(self, loss: float, num_tokens: int):
        """Update metric state with a batch."""
        self.total_loss += loss
        self.total_tokens += num_tokens
    
    def get_value(self) -> float:
        """Get current average loss."""
        return self.total_loss / self.total_tokens if self.total_tokens > 0 else 0.0
    
    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0


class RepetitionMetric(BaseMetric):
    """
    Repetition metric.
    
    Measures the fraction of repeated n-grams in generated text.
    Higher values indicate more repetition (worse).
    """
    
    def __init__(self, n: int = 2):
        super().__init__(
            name=f"repetition_{n}gram",
            description=f"Fraction of repeated {n}-grams.",
        )
        self.n = n
    
    def compute(self, text: str) -> MetricResult:
        """
        Compute repetition score for a text.
        
        Args:
            text: Generated text
            
        Returns:
            Repetition score (0-1, higher is worse)
        """
        # Tokenize text into words
        words = text.split()
        
        if len(words) < self.n:
            return MetricResult(name=self.name, value=0.0, description=self.description)
        
        # Count n-grams
        ngrams = []
        for i in range(len(words) - self.n + 1):
            ngram = tuple(words[i:i + self.n])
            ngrams.append(ngram)
        
        # Count unique and total
        unique_ngrams = set(ngrams)
        total_ngrams = len(ngrams)
        
        if total_ngrams == 0:
            return MetricResult(name=self.name, value=0.0, description=self.description)
        
        # Repetition score: fraction of n-grams that appear more than once
        counts = {}
        for ngram in ngrams:
            counts[ngram] = counts.get(ngram, 0) + 1
        
        repeated = sum(1 for count in counts.values() if count > 1)
        repetition_score = repeated / total_ngrams
        
        return MetricResult(
            name=self.name,
            value=repetition_score,
            description=self.description,
        )


class CoherenceMetric(BaseMetric):
    """
    Coherence metric.
    
    Simple heuristic for text coherence based on sentence length
    and presence of sentence-ending punctuation.
    """
    
    def __init__(self):
        super().__init__(
            name="coherence",
            description="Simple coherence score based on sentence structure.",
        )
    
    def compute(self, text: str) -> MetricResult:
        """
        Compute coherence score.
        
        Args:
            text: Generated text
            
        Returns:
            Coherence score (0-1, higher is better)
        """
        # Check for sentence-ending punctuation
        sentences = [s for s in text.split('.') if s.strip()]
        sentences += [s for s in text.split('!') if s.strip()]
        sentences += [s for s in text.split('?') if s.strip()]
        
        # Remove empty
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return MetricResult(name=self.name, value=0.0, description=self.description)
        
        # Score based on average sentence length
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Normalize to 0-1 range (assuming good sentences are 5-20 words)
        score = min(1.0, max(0.0, (avg_length - 3) / 17))
        
        return MetricResult(
            name=self.name,
            value=score,
            description=self.description,
        )


class DiversityMetric(BaseMetric):
    """
    Diversity metric.
    
    Measures the vocabulary diversity in generated text.
    Higher values indicate more diverse vocabulary (better).
    """
    
    def __init__(self):
        super().__init__(
            name="diversity",
            description="Type-token ratio (vocabulary diversity).",
        )
    
    def compute(self, text: str) -> MetricResult:
        """
        Compute diversity score.
        
        Args:
            text: Generated text
            
        Returns:
            Type-token ratio (0-1, higher is better)
        """
        words = text.split()
        
        if not words:
            return MetricResult(name=self.name, value=0.0, description=self.description)
        
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        return MetricResult(
            name=self.name,
            value=ttr,
            description=self.description,
        )


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Test perplexity
    print("\n=== PerplexityMetric ===")
    ppl_metric = PerplexityMetric()
    result = ppl_metric.compute(loss=2.0)
    print(result)
    
    # Test accuracy
    print("\n=== AccuracyMetric ===")
    acc_metric = AccuracyMetric()
    logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])  # (batch=1, seq=2, vocab=2)
    targets = torch.tensor([[1, 0]])  # (batch=1, seq=2)
    result = acc_metric.compute(logits, targets)
    print(result)
    
    # Test repetition
    print("\n=== RepetitionMetric ===")
    rep_metric = RepetitionMetric(n=2)
    text = "hello world hello world hello hello"
    result = rep_metric.compute(text)
    print(result)
    
    # Test coherence
    print("\n=== CoherenceMetric ===")
    coh_metric = CoherenceMetric()
    text = "This is a sentence. This is another sentence! And another one?"
    result = coh_metric.compute(text)
    print(result)
    
    # Test diversity
    print("\n=== DiversityMetric ===")
    div_metric = DiversityMetric()
    text = "the the the the the"
    result = div_metric.compute(text)
    print(f"Low diversity: {result}")
    
    text = "apple banana cherry date elderberry"
    result = div_metric.compute(text)
    print(f"High diversity: {result}")
