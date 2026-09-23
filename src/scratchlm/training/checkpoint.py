"""
Checkpoint management for ScratchLM training.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch

from scratchlm.utils.paths import paths


class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.
    
    Handles:
    - Saving model, optimizer, and training state
    - Loading checkpoints
    - Managing checkpoint versions
    - Tracking best checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        model_version: str = "v0.1",
        experiment_id: str = "exp001",
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model_version: Version of the model
            experiment_id: ID of the current experiment
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_version = model_version
        self.experiment_id = experiment_id
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment directory
        self.experiment_dir = self.checkpoint_dir / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Best checkpoint tracking
        self.best_checkpoint_path = None
        self.best_metric_value = float('inf')  # Lower is better for loss
        self.metric_to_watch = "val_loss"
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: int = 0,
        epoch: int = 0,
        loss: float = 0.0,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        save_optimizer: bool = True,
    ) -> Path:
        """
        Save a training checkpoint.
        
        Args:
            model: The model to save
            optimizer: Optimizer to save (optional)
            step: Training step number
            epoch: Training epoch number
            loss: Training loss
            val_loss: Validation loss (optional)
            metrics: Additional metrics to save
            is_best: Whether this is the best checkpoint so far
            save_optimizer: Whether to save optimizer state
            
        Returns:
            Path to the saved checkpoint
        """
        # Create checkpoint name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"step{step}_ep{epoch}_loss{loss:.4f}"
        
        if val_loss is not None:
            checkpoint_name += f"_val{val_loss:.4f}"
        
        checkpoint_path = self.experiment_dir / f"{checkpoint_name}.pt"
        
        # Create checkpoint dictionary
        checkpoint = {
            'model_version': self.model_version,
            'experiment_id': self.experiment_id,
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'val_loss': val_loss,
            'metrics': metrics,
            'timestamp': timestamp,
            'model_state_dict': model.state_dict(),
        }
        
        # Save optimizer if requested
        if optimizer is not None and save_optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        metadata = {
            'model_version': self.model_version,
            'experiment_id': self.experiment_id,
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'val_loss': val_loss,
            'metrics': metrics,
            'timestamp': timestamp,
            'file': checkpoint_name + '.pt',
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Track best checkpoint
        if is_best:
            self.best_checkpoint_path = checkpoint_path
            self.best_metric_value = val_loss if val_loss is not None else loss
            
            # Save best checkpoint symlink
            best_path = self.experiment_dir / "best.pt"
            if best_path.exists():
                best_path.unlink()
            best_path.symlink_to(checkpoint_path.name)
        
        return checkpoint_path
    
    def save_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: int = 0,
        epoch: int = 0,
        loss: float = 0.0,
        val_loss: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        Save a checkpoint if it's the best so far.
        
        Args:
            model: The model to save
            optimizer: Optimizer to save
            step: Training step
            epoch: Training epoch
            loss: Training loss
            val_loss: Validation loss
            metrics: Additional metrics
            
        Returns:
            Path to the saved checkpoint (if saved), None otherwise
        """
        # Check if this is the best so far
        metric_value = metrics.get(self.metric_to_watch, val_loss) if metrics else val_loss
        
        if metric_value < self.best_metric_value:
            checkpoint_path = self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                epoch=epoch,
                loss=loss,
                val_loss=val_loss,
                metrics=metrics,
                is_best=True,
            )
            return checkpoint_path
        
        return None
    
    def load_checkpoint(
        self,
        path: Optional[Path] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            path: Path to checkpoint file. If None, loads the best checkpoint.
            model: Model to load state into. If None, returns model state dict.
            optimizer: Optimizer to load state into.
            
        Returns:
            Dictionary with checkpoint information
        """
        if path is None:
            # Try to load best checkpoint
            best_path = self.experiment_dir / "best.pt"
            if best_path.exists():
                # Resolve symlink
                path = best_path.resolve()
            else:
                # Find the latest checkpoint
                checkpoints = list(self.experiment_dir.glob("*.pt"))
                if checkpoints:
                    path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                else:
                    raise FileNotFoundError("No checkpoints found")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load model state
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def load_latest_checkpoint(
        self,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Load the latest checkpoint from the experiment directory.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            
        Returns:
            Dictionary with checkpoint information
        """
        # Find all checkpoints
        checkpoints = list(self.experiment_dir.glob("*.pt"))
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.experiment_dir}")
        
        # Get the latest checkpoint by modification time
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        return self.load_checkpoint(latest_checkpoint, model, optimizer)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all checkpoints in the experiment directory.
        
        Returns:
            List of checkpoint metadata dictionaries
        """
        checkpoints = []
        
        # Find all checkpoint files
        checkpoint_files = list(self.experiment_dir.glob("*.pt"))
        
        for checkpoint_file in checkpoint_files:
            # Try to load metadata
            metadata_file = checkpoint_file.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                # Extract info from filename
                metadata = {
                    'file': checkpoint_file.name,
                    'path': str(checkpoint_file),
                }
            
            checkpoints.append(metadata)
        
        # Sort by step
        checkpoints.sort(key=lambda x: x.get('step', 0))
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, max_checkpoints: int = 10):
        """
        Remove old checkpoints, keeping only the most recent.
        
        Args:
            max_checkpoints: Maximum number of checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= max_checkpoints:
            return
        
        # Sort by step and remove oldest
        checkpoints.sort(key=lambda x: x.get('step', 0))
        
        # Remove checkpoints that are not the best
        for checkpoint in checkpoints[:-max_checkpoints]:
            if checkpoint.get('file') != 'best.pt':
                checkpoint_path = self.experiment_dir / checkpoint['file']
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                
                # Remove metadata file
                metadata_path = checkpoint_path.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()


if __name__ == "__main__":
    # Test checkpoint manager
    print("Testing CheckpointManager...")
    
    import tempfile
    import torch.nn as nn
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create checkpoint manager
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model_version="v0.1",
            experiment_id="test_exp",
        )
        
        # Create a simple model
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Save a checkpoint
        print("\nSaving checkpoint...")
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=100,
            epoch=5,
            loss=0.5,
            val_loss=0.6,
            metrics={'accuracy': 0.9},
        )
        print(f"Saved to: {checkpoint_path}")
        
        # Save another checkpoint (better)
        print("\nSaving better checkpoint...")
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=200,
            epoch=10,
            loss=0.4,
            val_loss=0.45,
            metrics={'accuracy': 0.95},
            is_best=True,
        )
        print(f"Saved to: {checkpoint_path}")
        
        # List checkpoints
        print("\nListing checkpoints...")
        checkpoints = manager.list_checkpoints()
        for cp in checkpoints:
            print(f"  {cp.get('file')}: step={cp.get('step')}, val_loss={cp.get('val_loss')}")
        
        # Load latest checkpoint
        print("\nLoading latest checkpoint...")
        loaded = manager.load_latest_checkpoint()
        print(f"Loaded checkpoint: step={loaded.get('step')}, val_loss={loaded.get('val_loss')}")
