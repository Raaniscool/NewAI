"""
Evaluation suite for ScratchLM.

Provides metrics and tasks for evaluating language model performance.
"""

from .metrics import (
    PerplexityMetric,
    AccuracyMetric,
    LossMetric,
    RepetitionMetric,
    CoherenceMetric,
)
from .suite import EvaluationSuite
from .tasks import (
    NextTokenPredictionTask,
    SentenceCompletionTask,
    GrammarTask,
    VocabularyTask,
    CoherenceTask,
)

__all__ = [
    "PerplexityMetric",
    "AccuracyMetric",
    "LossMetric",
    "RepetitionMetric",
    "CoherenceMetric",
    "EvaluationSuite",
    "NextTokenPredictionTask",
    "SentenceCompletionTask",
    "GrammarTask",
    "VocabularyTask",
    "CoherenceTask",
]
