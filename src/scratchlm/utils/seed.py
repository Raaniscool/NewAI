"""
Random seed management for reproducibility.

Ensures that all random operations are reproducible across runs.
"""

import random
import numpy as np
import torch


# Global seed value
_global_seed: int = 42


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for all relevant libraries.
    
    This ensures reproducibility across runs by setting the same seed
    for Python's random, NumPy, and PyTorch.
    
    Args:
        seed: The seed value to use
        deterministic: If True, also sets PyTorch to use deterministic
                     algorithms where available. This may slow down training
                     but ensures exact reproducibility.
    """
    global _global_seed
    _global_seed = seed
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Deterministic CUDA operations
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # For multi-processing (DataLoader workers)
    # Note: This needs to be set before DataLoader initialization
    # and also in worker init functions
    torch.use_deterministic_algorithms(deterministic)


def get_seed() -> int:
    """Get the current global seed value."""
    return _global_seed


def set_global_seed(seed: int):
    """
    Set the global seed value without affecting library seeds.
    
    Use this if you want to track the seed but have already set
    library seeds separately.
    """
    global _global_seed
    _global_seed = seed


def get_state() -> dict:
    """
    Get the current random state for all libraries.
    
    Returns:
        Dictionary containing the random state for each library.
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
    }
    
    # PyTorch state
    if torch.cuda.is_available():
        state['torch'] = torch.get_rng_state()
        state['torch_cuda'] = torch.cuda.get_rng_state()
    else:
        state['torch'] = torch.get_rng_state()
    
    return state


def set_state(state: dict):
    """
    Set the random state for all libraries.
    
    Args:
        state: Dictionary containing random states (from get_state())
    """
    if 'python' in state:
        random.setstate(state['python'])
    
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['torch_cuda'])


if __name__ == "__main__":
    # Test seed setting
    set_seed(12345)
    print(f"Global seed: {get_seed()}")
    
    # Test random generation
    import random
    print(f"Random int: {random.randint(0, 100)}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
