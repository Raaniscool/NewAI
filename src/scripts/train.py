#!/usr/bin/env python3
"""
Training script for ScratchLM.

This script trains a language model from scratch on English text data.
"""

import argparse
import json
import sys
from pathlib import Path

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scratchlm.utils import paths, set_seed
from scratchlm.model.config import TransformerConfig, TrainingConfig, TokenizerConfig, get_config_preset
from scratchlm.training import Trainer
from scratchlm.data import load_text_file, load_text_directory, clean_text, deduplicate_texts


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ScratchLM - A language model from scratch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        nargs='+',
        required=True,
        help="Paths to text files or directories containing training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        nargs='*',
        default=None,
        help="Paths to validation data (optional)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        nargs='*',
        default=None,
        help="Paths to test data (optional, will be held out)",
    )
    
    # Model arguments
    parser.add_argument(
        "--preset",
        type=str,
        choices=["tiny", "small", "medium"],
        default="tiny",
        help="Model preset to use",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v0.1",
        help="Version identifier for this model",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="exp001",
        help="Experiment identifier",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--train-tokenizer",
        action="store_true",
        default=True,
        help="Whether to train a new tokenizer",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to train on",
    )
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: checkpoints/{version}/{experiment})",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for training logs",
    )
    
    # Data processing arguments
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum text length to keep",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=10000,
        help="Maximum text length to keep",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=True,
        help="Whether to deduplicate texts",
    )
    
    return parser.parse_args()


def load_data(paths: List[str], min_length: int, max_length: int, deduplicate: bool) -> List[str]:
    """Load and preprocess text data from files or directories."""
    all_texts = []
    
    for path in paths:
        path = Path(path)
        
        if path.is_dir():
            # Load all text files from directory
            texts = load_text_directory(path)
        else:
            # Load single file
            texts = load_text_file(path)
        
        all_texts.extend(texts)
    
    # Clean and filter texts
    cleaned_texts = []
    for text in all_texts:
        cleaned = clean_text(text)
        if cleaned is not None and min_length <= len(cleaned) <= max_length:
            cleaned_texts.append(cleaned)
    
    # Deduplicate if requested
    if deduplicate:
        cleaned_texts = deduplicate_texts(cleaned_texts)
    
    return cleaned_texts


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configurations
    model_config, training_config, tokenizer_config = get_config_preset(args.preset)
    
    # Update configs from arguments
    model_config.version = args.model_version
    training_config.model_version = args.model_version
    training_config.experiment_id = args.experiment_id
    training_config.num_epochs = args.epochs
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate
    training_config.device = args.device
    training_config.seed = args.seed
    
    # Load data
    print(f"Loading training data from {args.data}...")
    train_texts = load_data(args.data, args.min_length, args.max_length, args.deduplicate)
    print(f"Loaded {len(train_texts)} training texts")
    
    # Load validation data
    val_texts = None
    if args.val_data:
        print(f"Loading validation data from {args.val_data}...")
        val_texts = load_data(args.val_data, args.min_length, args.max_length, args.deduplicate)
        print(f"Loaded {len(val_texts)} validation texts")
    
    # Load test data
    test_texts = None
    if args.test_data:
        print(f"Loading test data from {args.test_data}...")
        test_texts = load_data(args.test_data, args.min_length, args.max_length, args.deduplicate)
        print(f"Loaded {len(test_texts)} test texts")
    
    # Create trainer
    trainer = Trainer(
        model_config=model_config,
        training_config=training_config,
        tokenizer_config=tokenizer_config,
        model_version=args.model_version,
        experiment_id=args.experiment_id,
        device=args.device,
    )
    
    # Train
    print("\nStarting training...")
    print(f"Model: {args.preset}")
    print(f"Version: {args.model_version}")
    print(f"Experiment: {args.experiment_id}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Train texts: {len(train_texts)}")
    if val_texts:
        print(f"Val texts: {len(val_texts)}")
    if test_texts:
        print(f"Test texts: {len(test_texts)}")
    
    try:
        result = trainer.train(
            train_texts=train_texts,
            val_texts=val_texts,
            test_texts=test_texts,
            train_tokenizer=args.train_tokenizer,
        )
        
        print(f"\nTraining complete!")
        print(f"Final train loss: {result.avg_loss:.4f}")
        print(f"Final perplexity: {result.perplexity:.2f}")
        
        # Save configuration
        config_dir = paths.configs / "experiments" / args.experiment_id
        config_dir.mkdir(parents=True, exist_ok=True)
        
        model_config.save(config_dir / "model.yaml")
        training_config.save(config_dir / "training.yaml")
        tokenizer_config.save(config_dir / "tokenizer.yaml")
        
        print(f"Configurations saved to {config_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
