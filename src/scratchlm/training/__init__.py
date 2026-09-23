"""
Training pipeline for ScratchLM.

Provides training loop, optimizer configurations, and checkpoint management.
"""

from .trainer import Trainer
from .loop import TrainingLoop
from .optimizer import create_optimizer, create_scheduler
from .checkpoint import CheckpointManager

__all__ = [
    "Trainer",
    "TrainingLoop",
    "create_optimizer",
    "create_scheduler",
    "CheckpointManager",
]
