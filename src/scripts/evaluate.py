#!/usr/bin/env python3
"""
Evaluation script for ScratchLM.

This script evaluates a trained model on various metrics and tasks.
"""

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scratchlm.utils import paths, set_seed
from scratchlm.model import ScratchLM
from scratchlm.tokenizer import BPETokenizer, load_tokenizer
from scratchlm.data import TokenizedDataset, load_text_file, load_text_directory
from scratchlm.evaluation import EvaluationSuite


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ScratchLM - A language model from scratch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer directory (optional, will try to find in checkpoint)",
    )
    
    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        nargs='*',
        default=None,
        help="Paths to text files or directories for evaluation",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples for task evaluation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum generation length for tasks",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to evaluate on",
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (optional)",
    )
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load checkpoint
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Create model from config
    config_dict = checkpoint['config']
    from scratchlm.model.config import TransformerConfig
    config = TransformerConfig.from_dict(config_dict)
    
    model = ScratchLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    
    print(f"Model loaded: {config.version}")
    print(f"Parameters: {model.get_num_params():,}")
    
    # Load tokenizer
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        # Try to find tokenizer in checkpoint directory
        checkpoint_dir = Path(args.checkpoint).parent
        possible_paths = [
            checkpoint_dir / "tokenizer",
            checkpoint_dir.parent / "tokenizer",
            paths.tokenizers / f"{config.version}_bpe_{config.vocab_size}",
        ]
        
        for path in possible_paths:
            if path.exists():
                tokenizer_path = path
                break
    
    if tokenizer_path is None:
        print("Warning: No tokenizer found. Creating dummy tokenizer.")
        tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    else:
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = load_tokenizer(tokenizer_path)
    
    # Load evaluation data
    eval_texts = None
    if args.data:
        print(f"Loading evaluation data from {args.data}...")
        from scratchlm.data import load_data
        eval_texts = load_data(args.data, min_length=10, max_length=10000, deduplicate=False)
        print(f"Loaded {len(eval_texts)} evaluation texts")
    
    # Create evaluation suite
    suite = EvaluationSuite(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    
    if eval_texts:
        # Create dataset
        tokenized_texts = [tokenizer.encode(text, add_special_tokens=False) for text in eval_texts]
        dataset = TokenizedDataset(
            token_ids_list=tokenized_texts,
            context_length=config.context_length,
            tokenizer=tokenizer,
        )
        
        # Evaluate on dataset
        metrics = suite.evaluate_dataset(
            dataset,
            split="eval",
            batch_size=args.batch_size,
        )
        
        print(f"\nDataset Evaluation:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
    
    # Evaluate on tasks
    print("\nTask Evaluation:")
    task_results, samples = suite.evaluate_tasks(num_samples=args.num_samples)
    
    for task_name, task_metrics in task_results.items():
        print(f"  {task_name}:")
        for metric_name, value in task_metrics.items():
            print(f"    {metric_name}: {value:.4f}")
    
    # Print some samples
    if samples:
        print(f"\nSamples:")
        for i, sample in enumerate(samples[:3]):
            print(f"  {i + 1}. {sample}")
    
    # Save results
    if args.output:
        result_path = Path(args.output)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'checkpoint': args.checkpoint,
            'model_version': config.version,
            'metrics': metrics,
            'task_results': task_results,
            'samples': samples,
        }
        
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {result_path}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    import torch
    main()
