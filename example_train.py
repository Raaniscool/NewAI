#!/usr/bin/env python3
"""
Example: Train ScratchLM on sample English text.

This script demonstrates how to train a small language model from scratch
on English text data using ScratchLM.
"""

import sys
sys.path.insert(0, '/home/user/NewAI/src')

import torch

from scratchlm.utils import paths, set_seed
from scratchlm.model.config import get_config_preset
from scratchlm.training import Trainer
from scratchlm.data import load_text_file


def main():
    print("=" * 70)
    print("ScratchLM - Example Training")
    print("=" * 70)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load configurations
    print("\n1. Loading configurations...")
    model_config, train_config, tokenizer_config = get_config_preset("tiny")
    
    # For small datasets, reduce vocab size for better tokenizer training
    # The tiny preset has vocab_size=4096, but we'll use 256 for this example
    tokenizer_config.vocab_size = 256
    tokenizer_config.min_frequency = 1
    model_config.vocab_size = 256
    
    print(f"   Model preset: tiny")
    print(f"   Estimated parameters: {model_config.n_params:,}")
    print(f"   Context length: {model_config.context_length}")
    print(f"   Vocabulary size: {tokenizer_config.vocab_size}")
    print(f"   Min frequency: {tokenizer_config.min_frequency}")
    
    # Load training data
    print("\n2. Loading training data...")
    sample_file = paths.data_raw / "sample_english.txt"
    if not sample_file.exists():
        print(f"   Error: Sample file not found at {sample_file}")
        print("   Please ensure data/raw/sample_english.txt exists")
        return
    
    train_texts = load_text_file(sample_file)
    print(f"   Loaded {len(train_texts)} lines of text")
    print(f"   First line: {train_texts[0][:60]}...")
    
    # Create trainer
    print("\n3. Creating trainer...")
    trainer = Trainer(
        model_config=model_config,
        training_config=train_config,
        tokenizer_config=tokenizer_config,
        model_version="v0.1",
        experiment_id="example_run",
        device="cpu",
    )
    
    # Update training config for this example
    trainer.training_config.num_epochs = 5  # Short for example
    trainer.training_config.batch_size = 2  # Smaller batch for stability
    trainer.training_config.log_every = 5
    trainer.training_config.validate_every = 1000  # Disable validation for this small example
    trainer.training_config.learning_rate = 1e-4
    trainer.training_config.clip_grad_norm = 1.0
    trainer.training_config.warmup_steps = 10
    
    print(f"   Trainer created")
    print(f"   Epochs: {trainer.training_config.num_epochs}")
    print(f"   Batch size: {trainer.training_config.batch_size}")
    
    # Train the model
    print("\n4. Training model...")
    print("   This will take a few minutes on CPU...")
    print()
    
    try:
        result = trainer.train(
            train_texts=train_texts,
            val_texts=None,  # No validation for this example
            test_texts=None,
            train_tokenizer=True,  # Train a new tokenizer
        )
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"\nFinal training loss: {result.avg_loss:.4f}")
        print(f"Final perplexity: {result.perplexity:.2f}")
        print(f"\nCheckpoints saved to: {trainer.checkpoint_dir}")
        print(f"Tokenizer saved to: {paths.tokenizers}")
        
        # Generate a sample
        print("\n5. Testing generation...")
        prompt = "The quick brown fox"
        # Tokenize the prompt first
        input_ids = trainer.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids])
        
        generated_ids = trainer.model.generate(
            input_ids=input_ids,
            max_length=30,
            temperature=0.7,
            do_sample=True,
            pad_token_id=trainer.tokenizer.pad_token_id,
        )
        generated = trainer.tokenizer.decode(generated_ids[0].tolist())
        print(f"   Prompt: '{prompt}'")
        print(f"   Generated: '{generated}'")
        
        print("\n" + "=" * 70)
        print("Example complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Try with your own text data")
        print("  - Increase num_epochs for better results")
        print("  - Try different model presets (small, medium)")
        print("  - Add validation and test data")
        print("  - Experiment with hyperparameters")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
