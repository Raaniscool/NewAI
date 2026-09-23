#!/usr/bin/env python3
"""
Text generation script for ScratchLM CLI.

Uses the unified InferenceEngine shared with the GUI application.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scratchlm.utils import set_seed
from scratchlm.inference import InferenceEngine


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with ScratchLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
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
        help="Maximum tokens to generate",
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
        help="Number of top tokens to consider",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Probability mass to consider",
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
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to generate on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def main():
    """Main generation function using shared InferenceEngine."""
    args = parse_args()
    set_seed(args.seed)

    engine = InferenceEngine(device=args.device)
    
    print(f"Loading checkpoint: {args.checkpoint}...")
    info = engine.load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
    )
    
    print(f"Model loaded: {info.version} ({info.preset_name}) | Parameters: {info.num_params:,}")
    print(f"Tokenizer loaded: {info.tokenizer_path} (vocab size: {info.vocab_size:,})")
    
    print(f"\nGenerating text...")
    print(f"Prompt: '{args.prompt}'")
    print(f"Max length: {args.max_length} | Temp: {args.temperature} | Greedy: {args.greedy}\n")
    
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"--- Sample {i + 1} ---")
            
        result = engine.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            greedy=args.greedy,
        )
        
        print(f"Generated text:\n{result.full_text}\n")
        print(f"Stats: {result.tokens_generated} tokens in {result.elapsed_seconds}s ({result.tokens_per_second} tok/s)\n")


if __name__ == "__main__":
    main()
