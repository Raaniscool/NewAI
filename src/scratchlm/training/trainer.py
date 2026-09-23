"""
Main trainer class for ScratchLM.

Provides a high-level interface for training language models.
"""

import time
from typing import Optional, Dict, List, Tuple, Callable, Union
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scratchlm.model import ScratchLM
from scratchlm.model.config import TransformerConfig, TrainingConfig, TokenizerConfig
from scratchlm.tokenizer import BaseTokenizer, BPETokenizer
from scratchlm.data import TextDataset, TokenizedDataset, SlidingWindowDataset
from scratchlm.utils import paths, get_logger, set_seed
from scratchlm.utils.logging import TrainingLogger
from .loop import TrainingLoop
from .checkpoint import CheckpointManager
from .optimizer import create_optimizer, create_scheduler


class Trainer:
    """
    High-level trainer for ScratchLM models.
    
    This class provides a simple interface for training a language model
    from scratch. It handles:
    - Data loading and preprocessing
    - Tokenizer training (optional)
    - Model initialization
    - Training loop
    - Checkpointing
    - Evaluation
    """
    
    def __init__(
        self,
        model_config: Optional[TransformerConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        tokenizer_config: Optional[TokenizerConfig] = None,
        model_version: str = "v0.1",
        experiment_id: str = "exp001",
        device: Optional[str] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model_config: Model configuration. If None, uses default.
            training_config: Training configuration. If None, uses default.
            tokenizer_config: Tokenizer configuration. If None, uses default.
            model_version: Version identifier for the model
            experiment_id: Identifier for this training run
            device: Device to train on (cpu, cuda, mps)
        """
        # Set up configurations
        if model_config is None or training_config is None or tokenizer_config is None:
            model_config, training_config, tokenizer_config = self._get_default_configs()
        
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer_config = tokenizer_config
        
        # Set version and experiment ID
        self.model_config.version = model_version
        self.training_config.model_version = model_version
        self.training_config.experiment_id = experiment_id
        
        # Set device
        if device is not None:
            self.training_config.device = device
        self.device = torch.device(self.training_config.device)
        
        # Set up paths
        self.checkpoint_dir = paths.checkpoints / model_version / experiment_id
        self.experiment_dir = paths.experiments / experiment_id
        
        # Set up logging
        self.logger = get_logger(f"scratchlm.trainer.{experiment_id}")
        
        # Initialize components
        self.model: Optional[ScratchLM] = None
        self.tokenizer: Optional[BaseTokenizer] = None
        self.train_dataset: Optional[TokenizedDataset] = None
        self.val_dataset: Optional[TokenizedDataset] = None
        self.test_dataset: Optional[TokenizedDataset] = None
        self.training_loop: Optional[TrainingLoop] = None
    
    def _get_default_configs(self) -> Tuple[TransformerConfig, TrainingConfig, TokenizerConfig]:
        """Get default configurations."""
        from scratchlm.model.config import get_config_preset
        return get_config_preset("tiny")
    
    def train_tokenizer(
        self,
        texts: List[str],
        save_path: Optional[Path] = None,
    ) -> BPETokenizer:
        """
        Train a BPE tokenizer on the given texts.
        
        Args:
            texts: List of text strings to train on
            save_path: Optional path to save the tokenizer
            
        Returns:
            Trained tokenizer
        """
        self.logger.info(f"Training tokenizer on {len(texts)} texts...")
        
        # Train tokenizer
        tokenizer = BPETokenizer(
            vocab_size=self.tokenizer_config.vocab_size,
            pad_token=self.tokenizer_config.pad_token,
            unk_token=self.tokenizer_config.unk_token,
            bos_token=self.tokenizer_config.bos_token,
            eos_token=self.tokenizer_config.eos_token,
            min_frequency=self.tokenizer_config.min_frequency,
        )
        
        tokenizer.train(texts, show_progress=True)
        
        # Set tokenizer
        self.tokenizer = tokenizer
        
        # Update model config with actual vocab size
        self.model_config.vocab_size = tokenizer.vocab_size
        
        # Save tokenizer
        if save_path is None:
            save_path = paths.tokenizers / f"{self.model_config.version}_bpe_{tokenizer.vocab_size}"
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(save_path)
        
        self.logger.info(f"Tokenizer trained and saved to {save_path}")
        self.logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
        
        return tokenizer
    
    def load_tokenizer(self, path: Union[str, Path]) -> BaseTokenizer:
        """
        Load a tokenizer from a directory.
        
        Args:
            path: Path to tokenizer directory
            
        Returns:
            Loaded tokenizer
        """
        from scratchlm.tokenizer.utils import load_tokenizer
        
        path = Path(path)
        tokenizer = load_tokenizer(path)
        
        self.tokenizer = tokenizer
        self.model_config.vocab_size = tokenizer.vocab_size
        
        self.logger.info(f"Tokenizer loaded from {path}")
        self.logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
        
        return tokenizer
    
    def create_model(self, random_init: bool = True) -> ScratchLM:
        """
        Create the language model.
        
        Args:
            random_init: Whether to initialize with random weights
            
        Returns:
            Created model
        """
        self.logger.info("Creating model...")
        self.logger.info(f"Configuration: {self.model_config.to_dict()}")
        
        # Create model
        model = ScratchLM(self.model_config)
        
        # Set model
        self.model = model
        
        # Move to device
        model.to(self.device)
        
        self.logger.info(f"Model created with {model.get_num_params():,} parameters")
        
        return model
    
    def load_model(self, checkpoint_path: Union[str, Path]) -> ScratchLM:
        """
        Load a model from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded model
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create model from config
        config_dict = checkpoint['config']
        config = TransformerConfig.from_dict(config_dict)
        
        model = ScratchLM(config)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Set model
        self.model = model
        self.model_config = config
        
        # Move to device
        model.to(self.device)
        
        self.logger.info(f"Model loaded from {checkpoint_path}")
        self.logger.info(f"Parameters: {model.get_num_params():,}")
        
        return model
    
    def prepare_data(
        self,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None,
        test_texts: Optional[List[str]] = None,
        context_length: Optional[int] = None,
    ):
        """
        Prepare datasets for training.
        
        Args:
            train_texts: Training texts
            val_texts: Optional validation texts
            test_texts: Optional test texts
            context_length: Optional context length (uses model config if None)
        """
        if context_length is None:
            context_length = self.model_config.context_length
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be trained or loaded first")
        
        self.logger.info("Preparing datasets...")
        
        # Tokenize texts
        def tokenize_texts(texts):
            return [self.tokenizer.encode(text, add_special_tokens=False) for text in texts]
        
        # Create datasets
        train_token_ids = tokenize_texts(train_texts)
        self.train_dataset = TokenizedDataset(
            token_ids_list=train_token_ids,
            context_length=context_length,
            tokenizer=self.tokenizer,
        )
        
        self.logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        
        if val_texts is not None:
            val_token_ids = tokenize_texts(val_texts)
            self.val_dataset = TokenizedDataset(
                token_ids_list=val_token_ids,
                context_length=context_length,
                tokenizer=self.tokenizer,
            )
            self.logger.info(f"Val dataset: {len(self.val_dataset)} samples")
        
        if test_texts is not None:
            test_token_ids = tokenize_texts(test_texts)
            self.test_dataset = TokenizedDataset(
                token_ids_list=test_token_ids,
                context_length=context_length,
                tokenizer=self.tokenizer,
            )
            self.logger.info(f"Test dataset: {len(self.test_dataset)} samples")
    
    def create_training_loop(self) -> TrainingLoop:
        """
        Create a training loop.
        
        Returns:
            TrainingLoop instance
        """
        if self.model is None:
            raise ValueError("Model must be created first")
        if self.train_dataset is None:
            raise ValueError("Training dataset must be prepared first")
        
        self.logger.info("Creating training loop...")
        
        # Create training loop
        self.training_loop = TrainingLoop(
            model=self.model,
            train_dataset=self.train_dataset,
            config=self.training_config,
            val_dataset=self.val_dataset,
            checkpoint_dir=self.checkpoint_dir,
            device=self.training_config.device,
        )
        
        return self.training_loop
    
    def train(
        self,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None,
        test_texts: Optional[List[str]] = None,
        train_tokenizer: bool = True,
    ):
        """
        Run the complete training pipeline.
        
        Args:
            train_texts: Training texts
            val_texts: Optional validation texts
            test_texts: Optional test texts
            train_tokenizer: Whether to train a new tokenizer
        """
        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        set_seed(self.training_config.seed, deterministic=self.training_config.deterministic)
        
        # Step 1: Train tokenizer
        if train_tokenizer:
            self.train_tokenizer(train_texts)
        
        # Step 2: Create model
        self.create_model()
        
        # Step 3: Prepare data
        self.prepare_data(train_texts, val_texts, test_texts)
        
        # Step 4: Create training loop
        self.create_training_loop()
        
        # Step 5: Train
        self.logger.info("Starting training...")
        start_time = time.time()
        
        result = self.training_loop.train()
        
        training_time = time.time() - start_time
        
        self.logger.info(f"Training complete in {training_time:.2f} seconds")
        self.logger.info(f"Final train loss: {result.avg_loss:.4f}")
        self.logger.info(f"Final perplexity: {result.perplexity:.2f}")
        
        return result
    
    def evaluate(
        self,
        dataset: Optional[TokenizedDataset] = None,
        split: str = "test",
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on. If None, uses the test dataset.
            split: Name of the split being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        if dataset is None:
            dataset = self.test_dataset
        
        if dataset is None:
            raise ValueError("No dataset provided and no test dataset configured")
        
        if self.model is None:
            raise ValueError("Model must be loaded or created first")
        
        self.logger.info(f"Evaluating on {split} dataset...")
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Evaluate
        self.model.eval()
        
        total_loss = 0.0
        num_tokens = 0
        num_batches = 0
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
        
        with torch.no_grad():
            for input_ids, labels, attention_mask in loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                loss = loss_fn(logits_flat, labels_flat)
                
                total_loss += loss.item() * labels.size(1)
                num_tokens += labels.size(1)
                num_batches += 1
        
        avg_loss = total_loss / num_tokens if num_tokens > 0 else 0
        perplexity = float(np.exp(avg_loss))
        
        metrics = {
            f'{split}_loss': avg_loss,
            f'{split}_perplexity': perplexity,
        }
        
        self.logger.info(f"{split} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        
        return metrics
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Starting text
            max_length: Maximum length to generate (including prompt)
            temperature: Temperature for sampling
            top_k: Number of top tokens to consider
            top_p: Probability mass to consider
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model must be loaded or created first")
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded or trained first")
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        
        return generated_text


if __name__ == "__main__":
    print("Testing Trainer...")
    
    # Create trainer
    trainer = Trainer(
        model_version="v0.1_test",
        experiment_id="test_exp",
    )
    
    print(f"Model config: {trainer.model_config.to_dict()}")
    print(f"Training config: {trainer.training_config.to_dict()}")
    print(f"Tokenizer config: {trainer.tokenizer_config.to_dict()}")
    
    # Create a simple dataset
    train_texts = [
        "Hello world",
        "This is a test",
        "The quick brown fox jumps over the lazy dog",
    ] * 100
    
    # Train tokenizer
    print("\nTraining tokenizer...")
    trainer.train_tokenizer(train_texts)
    
    # Create model
    print("\nCreating model...")
    trainer.create_model()
    
    # Prepare data
    print("\nPreparing data...")
    trainer.prepare_data(train_texts)
    
    # Create training loop
    print("\nCreating training loop...")
    trainer.create_training_loop()
    
    print("\nTrainer setup complete!")
