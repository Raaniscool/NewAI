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
from typing import Optional, Dict, List, Tuple, Callable
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
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
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
    
    def train(self):
        """Run the full training loop."""
        self.logger.info(f"Starting training for {self.config.model_version} ({self.config.experiment_id})")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Train samples: {len(self.train_dataset)}")
        if self.val_dataset:
            self.logger.info(f"Val samples: {len(self.val_dataset)}")
        
        # Training loop
        for self.epoch in range(self.config.num_epochs):
            self.logger.info(f"\nEpoch {self.epoch + 1}/{self.config.num_epochs}")
            
            # Train epoch
            train_result = self.train_epoch()
            
            # Log training results
            self.logger.info(
                f"Train | Loss: {train_result.avg_loss:.4f} | "
                f"Perplexity: {train_result.perplexity:.2f} | "
                f"Tokens/sec: {train_result.num_tokens / train_result.time:.2f}"
            )
            
            # Validation
            if self.val_loader is not None and self.config.validate_every > 0:
                if (self.epoch + 1) % self.config.validate_every == 0:
                    val_result = self.validate()
                    
                    self.logger.info(
                        f"Val | Loss: {val_result.avg_loss:.4f} | "
                        f"Perplexity: {val_result.perplexity:.2f}"
                    )
                    
                    # Log metrics
                    metrics = {
                        'train_loss': train_result.avg_loss,
                        'val_loss': val_result.avg_loss,
                        'train_perplexity': train_result.perplexity,
                        'val_perplexity': val_result.perplexity,
                        'lr': get_lr(self.optimizer),
                    }
                    
                    self.training_logger.log_metrics(
                        metrics,
                        step=self.step,
                        epoch=self.epoch + 1,
                    )
                    
                    # Checkpoint
                    if self.config.save_checkpoint_every > 0:
                        if (self.step // self.config.batch_size) % self.config.save_checkpoint_every == 0:
                            self.save_checkpoint(val_result.avg_loss, metrics)
                    
                    # Early stopping
                    if self.config.early_stopping_patience is not None:
                        if val_result.avg_loss < self.best_val_loss:
                            self.best_val_loss = val_result.avg_loss
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1
                            
                        if self.patience_counter >= self.config.early_stopping_patience:
                            self.logger.info(
                                f"Early stopping triggered after {self.patience_counter} epochs "
                                f"without improvement"
                            )
                            break
                
                # Save best checkpoint
                if self.config.save_best:
                    self.checkpoint_manager.save_best_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        step=self.step,
                        epoch=self.epoch + 1,
                        loss=train_result.avg_loss,
                        val_loss=val_result.avg_loss if self.val_loader else None,
                        metrics={'train_loss': train_result.avg_loss},
                    )
            else:
                # Log training metrics only
                metrics = {
                    'train_loss': train_result.avg_loss,
                    'train_perplexity': train_result.perplexity,
                    'lr': get_lr(self.optimizer),
                }
                
                self.training_logger.log_metrics(
                    metrics,
                    step=self.step,
                    epoch=self.epoch + 1,
                )
                
                # Checkpoint
                if self.config.save_checkpoint_every > 0:
                    if (self.step // self.config.batch_size) % self.config.save_checkpoint_every == 0:
                        self.save_checkpoint(train_result.avg_loss, metrics)
        
        # Save final checkpoint
        self.save_checkpoint(
            loss=train_result.avg_loss,
            metrics={'train_loss': train_result.avg_loss, 'epoch': self.epoch + 1},
            is_final=True,
        )
        
        self.logger.info("Training complete!")
        return train_result
    
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
        
        for batch_idx, (input_ids, labels, attention_mask) in enumerate(self.train_loader):
            # Move to device
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            with torch.set_grad_enabled(True):
                logits = self.model(input_ids, attention_mask)
                
                # Reshape for loss calculation
                # logits: (batch_size, seq_len, vocab_size)
                # labels: (batch_size, seq_len)
                # We need: (batch_size * seq_len, vocab_size) and (batch_size * seq_len)
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
            
            # Log progress
            if self.config.log_every > 0 and (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_loss / num_batches
                self.logger.info(
                    f"Step {self.step} | Batch {batch_idx + 1}/{len(self.train_loader)} | "
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
        is_final: bool = False,
    ):
        """
        Save a checkpoint.
        
        Args:
            loss: Current loss value
            metrics: Dictionary of metrics
            is_final: Whether this is the final checkpoint
        """
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            epoch=self.epoch + 1,
            loss=loss,
            metrics=metrics,
            save_optimizer=True,
        )
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_final:
            # Save final checkpoint with special name
            final_path = self.checkpoint_dir / f"{self.config.model_version}_final.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.to_dict(),
                'step': self.step,
                'epoch': self.epoch + 1,
                'loss': loss,
            }, final_path)
            self.logger.info(f"Final checkpoint saved: {final_path}")


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
