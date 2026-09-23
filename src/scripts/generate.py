#!/usr/bin/env python3
"""
Text generation script for ScratchLM.

This script generates text from a trained model.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scratchlm.utils import paths, set_seed
from scratchlm.model import ScratchLM
from scratchlm.tokenizer import BPETokenizer, load_tokenizer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with ScratchLM",
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
        help="Path to tokenizer directory",
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Starting prompt for generation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum length to generate (including prompt)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (lower = more deterministic)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top tokens to consider (None for all)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Probability mass to consider (None for all)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        default=False,
        help="Use greedy decoding (no sampling)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to generate on",
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
    """Main generation function."""
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
    model.eval()
    
    print(f"Model loaded: {config.version}")
    print(f"Parameters: {model.get_num_params():,}")
    
    # Load tokenizer
    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer)
    else:
        # Try to find tokenizer
        checkpoint_dir = Path(args.checkpoint).parent
        possible_paths = [
            checkpoint_dir / "tokenizer",
            checkpoint_dir.parent / "tokenizer",
            paths.tokenizers / f"{config.version}_bpe_{config.vocab_size}",
        ]
        
        tokenizer_path = None
        for path in possible_paths:
            if path.exists():
                tokenizer_path = path
                break
    
    if tokenizer_path is None:
        print("Error: No tokenizer found. Please specify --tokenizer")
        sys.exit(1)
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Generate text
    print(f"\nGenerating text...")
    print(f"Prompt: '{args.prompt}'")
    print(f"Max length: {args.max_length}")
    print(f"Temperature: {args.temperature}")
    if args.top_k:
        print(f"Top-k: {args.top_k}")
    if args.top_p:
        print(f"Top-p: {args.top_p}")
    print(f"Greedy: {args.greedy}")
    print()
    
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"Sample {i + 1}:")
        
        try:
            generated = model.generate(
                prompt=args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=not args.greedy,
                tokenizer=tokenizer,
            )
            
            print(generated)
            print()
            
        except Exception as e:
            print(f"Error generating sample {i + 1}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
