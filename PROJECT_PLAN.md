# ScratchLM - Build a Language Model from Scratch

## Project Overview

This project aims to build a language model completely from scratch, starting with random initialization and training only on English text (Phase 1). The model will eventually become a specialized Roblox/Luau coding assistant, but only after establishing a solid English foundation.

## Phase 1: English Foundation

### Current Status: NOT STARTED

### Objectives
- Train a small transformer-based language model from random initialization
- Use only English text data
- Build custom tokenizer from scratch
- Implement full training pipeline
- Establish evaluation framework

### Hardware Constraints (Current Environment)
- CPU: Intel Xeon @ 2.60GHz (2 cores)
- RAM: ~4GB available in container (user reports ~12GB on host)
- GPU: None (Intel integrated graphics only)
- OS: Linux (container) / Windows (host)

## Architecture Design for Phase 1

### Model Specifications (ScratchLM v0.1)

| Component | Specification | Rationale |
|-----------|--------------|-----------|
| Architecture | Decoder-only Transformer | Standard for causal LM |
| Parameters | ~100-500k | Trainable on CPU with 4-12GB RAM |
| Layers | 2-4 | Balance complexity vs. trainability |
| Embedding Dim | 128-256 | Small enough for CPU |
| Attention Heads | 4-8 | Parallelizable on CPU |
| Context Length | 128-256 tokens | Memory efficient |
| Vocab Size | 8192-16384 | Good coverage of English |

### Tokenizer (Custom BPE)
- Type: Byte Pair Encoding (BPE)
- Vocabulary: 8192 tokens
- Special tokens: `<pad>`, `<unk>`, `<sos>`, `<eos>`
- Trained on: English dataset

### Dataset (Initial)
- Source: Public domain English text
- Size: 1-10 MB initially
- Cleaning: Deduplication, filtering, encoding fix
- Split: 80% train, 10% validation, 10% test

### Training Configuration
- Batch size: 4-8 sequences
- Learning rate: 1e-3 to 1e-4
- Optimizer: AdamW
- Loss: Cross-entropy
- Gradient clipping: 1.0
- Epochs: Start with 10-20, monitor loss

### Estimated Requirements
- Memory per batch (context=128, batch=4, d_model=256): ~10-20MB
- Training time per epoch: Minutes to hours on CPU
- Total training time: Hours to days depending on dataset size

## Project Structure

```
NewAI/
├── README.md                    # Main project documentation
├── PROJECT_PLAN.md             # This file - development plan
├── CHANGELOG.md                # Version history and changes
├── 
├── docs/
│   ├── ARCHITECTURE.md         # Model architecture details
│   ├── TOKENIZER.md            # Tokenizer documentation
│   ├── DATASET.md              # Dataset pipeline documentation
│   ├── TRAINING.md             # Training process documentation
│   └── EVALUATION.md           # Evaluation methodology
│
├── configs/
│   ├── model/                  # Model configuration files
│   │   ├── v0.1.yaml           # ScratchLM v0.1 config
│   │   └── ...
│   ├── training/               # Training configuration files
│   │   ├── v0.1.yaml
│   │   └── ...
│   └── tokenizer/              # Tokenizer configuration files
│       └── bpe_8k.yaml
│
├── src/
│   ├── scratchlm/              # Main package
│   │   ├── __init__.py
│   │   ├── tokenizer/          # Tokenizer implementation
│   │   │   ├── __init__.py
│   │   │   ├── bpe.py          # BPE tokenizer
│   │   │   ├── base.py         # Base tokenizer interface
│   │   │   └── utils.py        # Tokenizer utilities
│   │   │
│   │   ├── model/              # Model implementation
│   │   │   ├── __init__.py
│   │   │   ├── config.py       # Model configuration
│   │   │   ├── transformer.py  # Transformer architecture
│   │   │   ├── layers.py       # Individual layers (attention, FFN)
│   │   │   └── utils.py        # Model utilities
│   │   │
│   │   ├── data/               # Data pipeline
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py      # Dataset classes
│   │   │   ├── preprocessing.py # Text cleaning/preprocessing
│   │   │   ├── loader.py       # Data loading utilities
│   │   │   └── splits.py       # Train/val/test splitting
│   │   │
│   │   ├── training/           # Training pipeline
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py      # Main trainer class
│   │   │   ├── loop.py         # Training loop
│   │   │   ├── optimizer.py    # Optimizer configurations
│   │   │   └── checkpoint.py   # Checkpoint saving/loading
│   │   │
│   │   ├── evaluation/         # Evaluation suite
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py      # Evaluation metrics
│   │   │   ├── suite.py        # Evaluation suite runner
│   │   │   └── tasks.py        # Specific evaluation tasks
│   │   │
│   │   └── utils/              # Utilities
│   │       ├── __init__.py
│   │       ├── logging.py      # Logging configuration
│   │       ├── paths.py        # Path management
│   │       └── seed.py         # Random seed management
│   │
│   └── scripts/                # Command-line scripts
│       ├── train.py            # Training script
│       ├── evaluate.py         # Evaluation script
│       ├── generate.py          # Text generation script
│       ├── preprocess_data.py  # Data preprocessing
│       ├── train_tokenizer.py  # Tokenizer training
│       └── analyze.py          # Model analysis
│
├── data/                       # Data directory
│   ├── raw/                    # Raw downloaded data
│   ├── processed/              # Processed/cleaned data
│   ├── splits/                 # Train/val/test splits
│   └── metadata/               # Dataset metadata
│
├── tokenizers/                 # Trained tokenizers
│   ├── bpe_8k/                 # 8k vocab BPE tokenizer
│   │   ├── vocab.json
│   │   ├── merges.txt
│   │   └── config.json
│   └── ...
│
├── checkpoints/                # Model checkpoints
│   ├── v0.1/                   # ScratchLM v0.1
│   │   ├── config.json
│   │   ├── model.pt            # PyTorch model
│   │   ├── optimizer.pt        # Optimizer state
│   │   ├── tokenizer/          # Tokenizer used
│   │   └── metrics.json       # Training metrics
│   └── ...
│
├── experiments/                # Experiment tracking
│   ├── v0.1/                   # Experiment v0.1
│   │   ├── config.yaml
│   │   ├── train_log.csv
│   │   ├── val_log.csv
│   │   └── report.md
│   └── ...
│
├── evaluations/                # Evaluation results
│   ├── v0.1/                   # Evaluation of v0.1
│   │   ├── results.json
│   │   ├── samples.txt         # Generated samples
│   │   └── failures.md         # Documented failures
│   └── ...
│
├── reports/                    # Analysis reports
│   ├── v0.1_training_report.md
│   ├── v0.1_evaluation_report.md
│   └── ...
│
└── tests/                      # Unit tests
    ├── test_tokenizer.py
    ├── test_model.py
    └── test_data.py
```

## Implementation Order

1. **Project Setup** (This phase)
   - Create directory structure
   - Set up Python package
   - Install required dependencies
   - Configure logging and paths

2. **Tokenizer**
   - Implement BPE tokenizer from scratch
   - Train on English dataset
   - Add serialization/deserialization
   - Test tokenization

3. **Dataset Pipeline**
   - Download/collect English text
   - Implement cleaning and preprocessing
   - Create train/val/test splits
   - Build PyTorch Dataset classes

4. **Model Architecture**
   - Implement transformer layers
   - Build full model
   - Add configuration system
   - Test forward pass

5. **Training Pipeline**
   - Implement training loop
   - Add checkpointing
   - Add logging
   - Test with small dataset

6. **Evaluation Suite**
   - Implement metrics
   - Create English evaluation tasks
   - Run baseline evaluation

7. **First Training Run**
   - Train ScratchLM v0.1
   - Evaluate results
   - Document findings

## Versioning

### Model Versions
- `v0.1`: Initial baseline (random init, small config)
- `v0.2`: Improved dataset
- `v0.3`: Improved tokenizer/training
- `v0.4`: Improved architecture

### Checkpoint Naming
- Format: `{version}_ep{epoch}_step{step}_loss{loss:.4f}`
- Example: `v0.1_ep5_step1000_loss2.3456.pt`

## Dependencies

### Required
- Python 3.10+
- PyTorch 2.0+
- NumPy
- PyYAML
- tqdm
- datasets (HuggingFace) - optional for some data loading

### Optional (for later phases)
- accelerate - for multi-GPU training
- bitsandbytes - for 8-bit optimizers
- peft - for parameter-efficient fine-tuning
- transformers - for comparison/baselines

## Success Criteria for Phase 1

The model should demonstrate:
1. Decreasing training loss (learning is happening)
2. Validation loss that tracks training loss
3. Ability to generate coherent English-like text
4. Basic understanding of English grammar and vocabulary
5. Measurable improvement across versions

## Transition to Phase 2

Phase 1 is considered complete when:
- Model achieves < 3.0 cross-entropy loss on validation set
- Model can generate 50+ token coherent continuations
- Model shows basic grammatical correctness
- Evaluation suite shows consistent improvement
- At least 3 model versions have been trained and evaluated

Only then will we introduce programming concepts.

---

*Last updated: 2026-09-22*
*Phase: Planning*
