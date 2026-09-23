"""
Utilities for ScratchLM.
"""

from .paths import ScratchLmPaths, paths, reset_paths
from .logging import setup_logging, get_logger
from .seed import set_seed, get_seed

__all__ = [
    "ScratchLmPaths",
    "paths",
    "reset_paths",
    "setup_logging",
    "get_logger",
    "set_seed",
    "get_seed",
]
