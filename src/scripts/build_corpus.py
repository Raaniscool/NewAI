#!/usr/bin/env python3
"""
CLI Script to build the ScratchLM Phase 1 English Foundation Corpus (500k-1M+ Scale).

Ingests real public-domain human text (80%) and synthetic targeted English (20%),
applies quality filtering, deduplication, train/val splitting, enforces quality gates,
and produces a full analytics report.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scratchlm.data.corpus import CorpusPipeline
from scratchlm.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ScratchLM Phase 1 English Corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=1000000,
        help="Target total retained word count for corpus (default: 1,000,000)",
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=0.20,
        help="Target ratio of synthetic augmentation data (0.0 to 0.5)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pipeline reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    pipeline = CorpusPipeline(
        target_words=args.target_words,
        synthetic_ratio=args.synthetic_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    report = pipeline.build_corpus()

    print("Corpus pipeline execution finished successfully.")
    print("Report written to data/processed/corpus/CORPUS_REPORT.md")


if __name__ == "__main__":
    main()
