"""
Training loop for ScratchLM.

Implements the main training loop with support for:
- Training batches
- Validation
- Checkpointing
- Logging
- Early stopping
"""

import time
from typing import Optional, Dict, List, Tuple, Callable, Any
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from scratchlm.model import ScratchLM
from scratchlm.model.config import TrainingConfig
from scratchlm.data import TokenizedDataset, SlidingWindowDataset
from scratchlm.utils import paths, get_logger, set_seed
from scratchlm.utils.logging import TrainingLogger
from .checkpoint import CheckpointManager
from .optimizer import create_optimizer, create_scheduler, get_lr


@dataclass
class BatchResult:
    """Results from processing a single batch."""
    loss: float
    num_tokens: int
    time: float


@dataclass
class EpochResult:
    """Results from a training or validation epoch."""
    total_loss: float
    num_tokens: int
    num_batches: int
    time: float
    
    @property
    def avg_loss(self) -> float:
        return self.total_loss / self.num_batches if self.num_batches > 0 else 0.0
    
    @property
    def perplexity(self) -> float:
        return np.exp(self.avg_loss)


class TrainingLoop:
    """
    Main training loop for ScratchLM.
    
    Handles:
    - Training batches
    - Validation
    - Checkpointing
    - Logging
    - Early stopping
    """
    
    def __init__(
        self,
        model: ScratchLM,
        train_dataset: TokenizedDataset,
        config: TrainingConfig,
        val_dataset: Optional[TokenizedDataset] = None,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        epoch_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the training loop.
        
        Args:
            model: The model to train
            train_dataset: Training dataset
            config: Training configuration
            val_dataset: Optional validation dataset
            checkpoint_dir: Directory for checkpoints
            device: Device to train on (cpu, cuda, mps)
            step_callback: Called on batch/step completion with progress details
            epoch_callback: Called on epoch completion with summary metrics
            cancel_callback: Function returning True if cancellation is requested
            log_callback: Called with human-readable log strings
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        self.step_callback = step_callback
        self.epoch_callback = epoch_callback
        self.cancel_callback = cancel_callback
        self.log_callback = log_callback
        self.was_cancelled = False
        
        # Set device
        if device is None:
            device = config.device
        self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up checkpointing
        if checkpoint_dir is None:
            checkpoint_dir = paths.checkpoints / config.model_version / config.experiment_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            model_version=config.model_version,
            experiment_id=config.experiment_id,
        )
        
        # Set up logging
        self.logger = get_logger(f"scratchlm.training.{config.experiment_id}")
        self.training_logger = TrainingLogger(
            experiment_dir=self.checkpoint_dir,
            name=f"{config.model_version}_{config.experiment_id}",
        )
        
        # Set up optimizer
        self.optimizer = create_optimizer(self.model, config)
        
        # Set up scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            config,
            num_training_steps=len(train_dataset) * config.num_epochs // config.batch_size,
        )
        
        # Set random seed
        set_seed(config.seed, deterministic=config.deterministic)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # CPU training, no multiprocessing
            pin_memory=False,
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
        else:
            self.val_loader = None
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_token_id)
    
    def _emit_log(self, message: str):
        self.logger.info(message)
        if self.log_callback is not None:
            try:
                self.log_callback(message)
            except Exception:
                pass

    def train(self):
        """Run the full training loop."""
        self._emit_log(f"Starting training for {self.config.model_version} ({self.config.experiment_id})")
        self._emit_log(f"Device: {self.device}")
        self._emit_log(f"Train samples: {len(self.train_dataset)}")
        if self.val_dataset:
            self._emit_log(f"Val samples: {len(self.val_dataset)}")
        
        last_checkpoint_path = None
        
        # Training loop
        for self.epoch in range(self.config.num_epochs):
            if self.cancel_callback and self.cancel_callback():
                self._emit_log("Training stop requested before epoch start.")
                self.was_cancelled = True
                break
                
            self._emit_log(f"\nEpoch {self.epoch + 1}/{self.config.num_epochs}")
            
            # Train epoch
            train_result = self.train_epoch()
            
            if self.was_cancelled:
                self._emit_log(f"Training stopped early by user during Epoch {self.epoch + 1}.")
                break

            # Log training results
            self._emit_log(
                f"Train | Loss: {train_result.avg_loss:.4f} | "
                f"Perplexity: {train_result.perplexity:.2f} | "
                f"Tokens/sec: {train_result.num_tokens / max(0.001, train_result.time):.2f}"
            )
            
            # Validation
            val_result = None
            if self.val_loader is not None:
                val_result = self.validate()
                self._emit_log(
                    f"Val   | Loss: {val_result.avg_loss:.4f} | "
                    f"Perplexity: {val_result.perplexity:.2f}"
                )
            
            metrics = {
                'epoch': self.epoch + 1,
                'total_epochs': self.config.num_epochs,
                'step': self.step,
                'train_loss': train_result.avg_loss,
                'train_perplexity': train_result.perplexity,
                'lr': get_lr(self.optimizer),
            }
            if val_result is not None:
                metrics['val_loss'] = val_result.avg_loss
                metrics['val_perplexity'] = val_result.perplexity

            self.training_logger.log_metrics(
                metrics,
                step=self.step,
                epoch=self.epoch + 1,
            )

            # Checkpoint
            is_best = False
            if val_result is not None:
                if val_result.avg_loss < self.best_val_loss:
                    self.best_val_loss = val_result.avg_loss
                    is_best = True
            else:
                if train_result.avg_loss < self.best_val_loss:
                    self.best_val_loss = train_result.avg_loss
                    is_best = True
            
            chk_path = self.save_checkpoint(
                loss=train_result.avg_loss,
                val_loss=val_result.avg_loss if val_result else None,
                metrics=metrics,
                is_best=is_best,
            )
            last_checkpoint_path = chk_path

            if self.epoch_callback is not None:
                try:
                    cb_data = dict(metrics)
                    cb_data['checkpoint_path'] = str(chk_path)
                    cb_data['is_best'] = is_best
                    self.epoch_callback(cb_data)
                except Exception:
                    pass

            # Early stopping check
            if val_result is not None and self.config.early_stopping_patience is not None:
                if is_best:
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config.early_stopping_patience:
                    self._emit_log(
                        f"Early stopping triggered after {self.patience_counter} epochs "
                        f"without improvement"
                    )
                    break

        # Save final checkpoint
        final_chk_path = self.save_checkpoint(
            loss=train_result.avg_loss if 'train_result' in locals() else 0.0,
            val_loss=val_result.avg_loss if ('val_result' in locals() and val_result) else None,
            metrics={'train_loss': train_result.avg_loss if 'train_result' in locals() else 0.0, 'epoch': self.epoch + 1},
            is_final=True,
        )
        
        status_msg = "Training cancelled safely." if self.was_cancelled else "Training complete!"
        self._emit_log(f"\n{status_msg}")
        return train_result if 'train_result' in locals() else EpochResult(0.0, 0, 0, 0.0)

    def train_epoch(self) -> EpochResult:
        """
        Run one training epoch.
        
        Returns:
            EpochResult with training statistics
        """
        self.model.train()
        
        total_loss = 0.0
        num_tokens = 0
        num_batches = 0
        start_time = time.time()
        
        total_batches = len(self.train_loader)
        
        for batch_idx, (input_ids, labels, attention_mask) in enumerate(self.train_loader):
            if self.cancel_callback and self.cancel_callback():
                self._emit_log(f"Cancellation requested at batch {batch_idx + 1}/{total_batches}. Stopping epoch cleanly...")
                self.was_cancelled = True
                break

            # Move to device
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            with torch.set_grad_enabled(True):
                logits = self.model(input_ids, attention_mask)
                
                # Reshape for loss calculation
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                # Calculate loss
                loss = self.loss_fn(logits_flat, labels_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm,
                )
            
            # Update weights
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Update statistics
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_tokens = (labels != self.train_dataset.pad_token_id).sum().item()
            num_tokens += batch_tokens
            num_batches += 1
            self.step += 1
            
            elapsed = time.time() - start_time
            
            if self.step_callback is not None:
                try:
                    self.step_callback({
                        'step': self.step,
                        'batch_idx': batch_idx + 1,
                        'total_batches': total_batches,
                        'epoch': self.epoch + 1,
                        'total_epochs': self.config.num_epochs,
                        'batch_loss': batch_loss,
                        'avg_loss': total_loss / num_batches,
                        'lr': get_lr(self.optimizer),
                        'elapsed_seconds': elapsed,
                        'tokens_per_sec': num_tokens / max(0.001, elapsed),
                    })
                except Exception:
                    pass

            # Log progress
            if self.config.log_every > 0 and (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_loss / num_batches
                self._emit_log(
                    f"Step {self.step} | Batch {batch_idx + 1}/{total_batches} | "
                    f"Loss: {avg_loss:.4f} | LR: {get_lr(self.optimizer):.6f}"
                )
        
        epoch_time = time.time() - start_time
        
        return EpochResult(
            total_loss=total_loss,
            num_tokens=num_tokens,
            num_batches=num_batches,
            time=epoch_time,
        )
    
    def validate(self) -> EpochResult:
        """
        Run validation on the validation set.
        
        Returns:
            EpochResult with validation statistics
        """
        if self.val_loader is None:
            raise ValueError("No validation dataset configured")
        
        self.model.eval()
        
        total_loss = 0.0
        num_tokens = 0
        num_batches = 0
        start_time = time.time()
        
        with torch.no_grad():
            for input_ids, labels, attention_mask in self.val_loader:
                # Move to device
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss = self.loss_fn(logits_flat, labels_flat)
                
                # Update statistics
                batch_loss = loss.item()
                total_loss += batch_loss
                batch_tokens = (labels != self.val_dataset.pad_token_id).sum().item() if self.val_dataset else labels.numel()
                num_tokens += batch_tokens
                num_batches += 1
        
        epoch_time = time.time() - start_time
        
        return EpochResult(
            total_loss=total_loss,
            num_tokens=num_tokens,
            num_batches=num_batches,
            time=epoch_time,
        )
    
    def save_checkpoint(
        self,
        loss: float,
        metrics: Dict[str, float],
        val_loss: Optional[float] = None,
        is_best: bool = False,
        is_final: bool = False,
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            loss: Current loss value
            metrics: Dictionary of metrics
            val_loss: Validation loss (optional)
            is_best: Whether this is the best checkpoint so far
            is_final: Whether this is the final checkpoint
        """
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            epoch=self.epoch + 1,
            loss=loss,
            val_loss=val_loss,
            metrics=metrics,
            is_best=is_best,
            save_optimizer=True,
        )
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_final:
            # Save final checkpoint with special name
            final_path = self.checkpoint_dir / f"{self.config.model_version}_final.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.model.config.to_dict(),
                'step': self.step,
                'epoch': self.epoch + 1,
                'loss': loss,
                'val_loss': val_loss,
            }, final_path)
            self.logger.info(f"Final checkpoint saved: {final_path}")

        return checkpoint_path


from pathlib import Path  # Added import


class TrainingState:
    """Tracks the state of training."""
    
    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_losses = []
        self.validation_losses = []


if __name__ == "__main__":
    print("Testing TrainingLoop...")
    
    # This is a minimal test - a full test would require actual data
    from scratchlm.model.config import get_config_preset
    from scratchlm.model import ScratchLM
    from scratchlm.data import TokenizedDataset
    
    # Get config
    model_config, train_config, _ = get_config_preset("tiny")
    
    # Create model
    model = ScratchLM(model_config)
    
    # Create dummy dataset
    dummy_data = [[1, 2, 3, 4, 5] for _ in range(100)]
    dataset = TokenizedDataset(
        token_ids_list=dummy_data,
        context_length=model_config.context_length,
        pad_token_id=0,
    )
    
    # Create training loop
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        loop = TrainingLoop(
            model=model,
            train_dataset=dataset,
            config=train_config,
            checkpoint_dir=tmpdir,
        )
        
        print("Training loop created successfully!")
        print(f"Model parameters: {model.get_num_params():,}")
        print(f"Train dataset size: {len(dataset)}")
