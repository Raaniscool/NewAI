"""
Path management for ScratchLM project.

Centralizes all path definitions to ensure consistency across the codebase.
"""

import os
from pathlib import Path
from typing import Optional


# Project root detection
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # NewAI/


class ScratchLmPaths:
    """Central path management for the ScratchLM project."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize path manager.
        
        Args:
            project_root: Root directory of the project. If None, auto-detected.
        """
        if project_root is None:
            self.root = PROJECT_ROOT
        else:
            self.root = Path(project_root)
        
        # Ensure root exists
        if not self.root.exists():
            raise FileNotFoundError(f"Project root not found: {self.root}")
    
    @property
    def root(self) -> Path:
        """Project root directory."""
        return self._root
    
    @root.setter
    def root(self, value: Path):
        self._root = value
        # Rebuild all paths
        self._build_paths()
    
    def _build_paths(self):
        """Build all path properties."""
        # Source directories
        self.src = self.root / "src"
        self.scratchlm = self.src / "scratchlm"
        self.scripts = self.src / "scripts"
        
        # Data directories
        self.data = self.root / "data"
        self.data_raw = self.data / "raw"
        self.data_processed = self.data / "processed"
        self.data_splits = self.data / "splits"
        self.data_metadata = self.data / "metadata"
        
        # Tokenizer directory
        self.tokenizers = self.root / "tokenizers"
        
        # Checkpoints directory
        self.checkpoints = self.root / "checkpoints"
        
        # Experiments directory
        self.experiments = self.root / "experiments"
        
        # Evaluations directory
        self.evaluations = self.root / "evaluations"
        
        # Reports directory
        self.reports = self.root / "reports"
        
        # Configs directory
        self.configs = self.root / "configs"
        self.configs_model = self.configs / "model"
        self.configs_training = self.configs / "training"
        self.configs_tokenizer = self.configs / "tokenizer"
        
        # Docs directory
        self.docs = self.root / "docs"
        
        # Tests directory
        self.tests = self.root / "tests"
    
    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.root,
            self.src,
            self.scripts,
            self.data,
            self.data_raw,
            self.data_processed,
            self.data_splits,
            self.data_metadata,
            self.tokenizers,
            self.checkpoints,
            self.experiments,
            self.evaluations,
            self.reports,
            self.configs,
            self.configs_model,
            self.configs_training,
            self.configs_tokenizer,
            self.docs,
            self.tests,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, version: str, epoch: Optional[int] = None, 
                           step: Optional[int] = None, loss: Optional[float] = None) -> Path:
        """
        Generate a checkpoint path.
        
        Args:
            version: Model version (e.g., "v0.1")
            epoch: Training epoch number
            step: Training step number
            loss: Current loss value
            
        Returns:
            Path to checkpoint directory or file
        """
        checkpoint_dir = self.checkpoints / version
        
        if epoch is None and step is None and loss is None:
            return checkpoint_dir
        
        # Format checkpoint filename
        parts = [f"ep{epoch}"] if epoch is not None else []
        parts += [f"step{step}"] if step is not None else []
        parts += [f"loss{loss:.4f}"] if loss is not None else []
        
        filename = f"model_{'_'.join(parts)}.pt" if parts else "model.pt"
        
        return checkpoint_dir / filename
    
    def get_tokenizer_path(self, name: str) -> Path:
        """Get path to a tokenizer directory."""
        return self.tokenizers / name
    
    def get_experiment_path(self, experiment_id: str) -> Path:
        """Get path to an experiment directory."""
        return self.experiments / experiment_id
    
    def get_evaluation_path(self, model_version: str) -> Path:
        """Get path to evaluation results for a model version."""
        return self.evaluations / model_version
    
    def get_report_path(self, name: str) -> Path:
        """Get path to a report file."""
        return self.reports / f"{name}.md"
    
    def get_data_path(self, category: str, filename: str) -> Path:
        """
        Get path to a data file.
        
        Args:
            category: One of 'raw', 'processed', 'splits', 'metadata'
            filename: The filename
            
        Returns:
            Full path to the data file
        """
        category_map = {
            'raw': self.data_raw,
            'processed': self.data_processed,
            'splits': self.data_splits,
            'metadata': self.data_metadata,
        }
        
        if category not in category_map:
            raise ValueError(f"Invalid category: {category}. Must be one of {list(category_map.keys())}")
        
        return category_map[category] / filename


# Global paths instance
paths = ScratchLmPaths()


def reset_paths(project_root: Optional[Path] = None):
    """Reset the global paths instance."""
    global paths
    paths = ScratchLmPaths(project_root)


if __name__ == "__main__":
    # Test path setup
    print("Project root:", paths.root)
    print("Source directory:", paths.scratchlm)
    print("Data directory:", paths.data)
    print("Checkpoints directory:", paths.checkpoints)
    
    # Create directories
    paths.ensure_directories()
    print("\nAll directories ensured!")
