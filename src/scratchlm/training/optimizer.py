"""
Optimizer and scheduler configurations for ScratchLM.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LinearLR

from scratchlm.model.config import TrainingConfig


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: The model to optimize
        config: Training configuration
        
    Returns:
        Configured optimizer
    """
    # Get all parameters
    params = model.parameters()
    
    # Create optimizer based on configuration
    # For now, we use AdamW as it's the most common for transformer training
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=tuple(config.betas),
        eps=config.eps,
    )
    
    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
    num_training_steps: Optional[int] = None,
) -> Optional[LambdaLR]:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        config: Training configuration
        num_training_steps: Total number of training steps
        
    Returns:
        Learning rate scheduler, or None if not using a scheduler
    """
    if not config.use_scheduler:
        return None
    
    if config.scheduler_type == "cosine":
        # Cosine annealing with warmup
        if num_training_steps is None:
            raise ValueError("num_training_steps must be provided for cosine scheduler")
        
        def lr_lambda(current_step):
            # Linear warmup
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            
            # Cosine annealing
            progress = float(current_step - config.warmup_steps) / float(
                max(1, num_training_steps - config.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif config.scheduler_type == "linear":
        # Linear decay with warmup
        if num_training_steps is None:
            raise ValueError("num_training_steps must be provided for linear scheduler")
        
        def lr_lambda(current_step):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            
            progress = float(current_step - config.warmup_steps) / float(
                max(1, num_training_steps - config.warmup_steps)
            )
            return max(0.0, 1.0 - progress)
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif config.scheduler_type == "constant":
        # Constant learning rate (with optional warmup)
        def lr_lambda(current_step):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            return 1.0
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
    
    return scheduler


import math  # Added import


def get_lr(optimizer: Optimizer) -> float:
    """Get the current learning rate from an optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def set_lr(optimizer: Optimizer, lr: float):
    """Set the learning rate for an optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    # Test optimizer creation
    print("Testing optimizer creation...")
    
    import torch.nn as nn
    
    # Create a simple model
    model = nn.Linear(10, 10)
    
    # Create config
    from scratchlm.model.config import TrainingConfig
    config = TrainingConfig(
        learning_rate=1e-4,
        weight_decay=0.01,
        betas=[0.9, 0.999],
        eps=1e-8,
        use_scheduler=True,
        scheduler_type="cosine",
        warmup_steps=100,
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    print(f"Optimizer: {optimizer}")
    print(f"Learning rate: {get_lr(optimizer)}")
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config, num_training_steps=1000)
    print(f"Scheduler: {scheduler}")
    
    # Test learning rate
    print("\nLearning rate schedule (first 10 steps):")
    for i in range(10):
        lr = get_lr(optimizer)
        print(f"  Step {i}: {lr:.6f}")
        optimizer.step()
        if scheduler:
            scheduler.step()
