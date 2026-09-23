# ScratchLM - Build a Language Model from Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

**ScratchLM** is a project to build, train, and understand a language model completely from scratch. Starting with random initialization and English-only training (Phase 1), this project aims to create a specialized Roblox/Luau coding assistant through incremental phases.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Raaniscool/NewAI.git
cd NewAI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Training Your First Model

```bash
# Train a tiny model on your own text data
python src/scripts/train.py \
    --data data/my_texts.txt \
    --preset tiny \
    --model-version v0.1 \
    --experiment-id my_first_model \
    --epochs 20 \
    --batch-size 4
```

### Generating Text

```bash
# Generate text from a trained model
python src/scripts/generate.py \
    --checkpoint checkpoints/v0.1/my_first_model/best.pt \
    --tokenizer tokenizers/v0.1_bpe_4096 \
    --prompt "The quick brown fox" \
    --max-length 50 \
    --temperature 0.7
```

### Evaluating a Model

```bash
# Evaluate model performance
python src/scripts/evaluate.py \
    --checkpoint checkpoints/v0.1/my_first_model/best.pt \
    --tokenizer tokenizers/v0.1_bpe_4096 \
    --data data/test_texts.txt \
    --num-samples 20
```

## 📚 Project Structure

```
NewAI/
├── README.md                    # This file
├── PROJECT_PLAN.md             # Detailed project plan
├── CHANGELOG.md                # Version history
├── setup.py                    # Python package setup
├── requirements.txt            # Dependencies
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
│   ├── training/               # Training configuration files
│   └── tokenizer/              # Tokenizer configuration files
│
├── src/
│   ├── scratchlm/              # Main package
│   │   ├── __init__.py
│   │   ├── tokenizer/          # Tokenizer implementations
│   │   ├── model/              # Model implementations
│   │   ├── data/               # Data pipeline
│   │   ├── training/           # Training pipeline
│   │   ├── evaluation/         # Evaluation suite
│   │   └── utils/              # Utilities
│   │
│   └── scripts/                # Command-line scripts
│       ├── train.py            # Training script
│       ├── evaluate.py         # Evaluation script
│       └── generate.py         # Text generation script
│
├── data/                       # Data directory
│   ├── raw/                    # Raw downloaded data
│   ├── processed/              # Processed/cleaned data
│   ├── splits/                 # Train/val/test splits
│   └── metadata/               # Dataset metadata
│
├── tokenizers/                 # Trained tokenizers
│   └── bpe_8k/                 # Example BPE tokenizer
│
├── checkpoints/                # Model checkpoints
│   └── v0.1/                   # Model version checkpoints
│
├── experiments/                # Experiment tracking
│   └── exp001/                 # Experiment results
│
├── evaluations/                # Evaluation results
│   └── v0.1/                   # Evaluation results
│
└── reports/                    # Analysis reports
    └── v0.1_training_report.md
```

## 🎯 Project Phases

### Phase 1: English Foundation ✅ **CURRENT**
- Train a small transformer-based language model from random initialization
- Use only English text data
- Build custom tokenizer from scratch
- Implement full training pipeline
- Establish evaluation framework

**Success Criteria:**
- Model achieves < 3.0 cross-entropy loss on validation set
- Model can generate 50+ token coherent continuations
- Model shows basic grammatical correctness
- At least 3 model versions have been trained and evaluated

### Phase 2: Programming Concepts
- Introduce programming-related English text
- Teach basic programming concepts in English
- Maintain English language quality

### Phase 3: Luau Language
- Introduce Luau syntax and semantics
- Train on Luau code examples
- Teach language-specific patterns

### Phase 4: Roblox Platform
- Introduce Roblox APIs and concepts
- Teach Studio-specific patterns
- Client/server architecture
- RemoteEvents, RemoteFunctions
- Security best practices

### Phase 5: Specialized Assistant
- Fine-tune for Roblox development
- Add personality and roleplay
- Implement RLHF for helpfulness
- Optimize for coding tasks

## 🏗️ Architecture

### Model (ScratchLM)
- **Architecture**: Decoder-only Transformer
- **Parameters**: ~100k-500k (tiny/small presets for CPU training)
- **Context Length**: 128-256 tokens
- **Vocabulary**: 4k-16k tokens (custom BPE tokenizer)
- **Layers**: 2-6 transformer blocks
- **Heads**: 4-8 attention heads

### Tokenizer
- **Type**: Byte Pair Encoding (BPE)
- **Vocabulary**: Custom-trained on English text
- **Special Tokens**: `<pad>`, `<unk>`, `<s>`, `</s>`

### Training
- **Objective**: Next token prediction (causal language modeling)
- **Loss**: Cross-entropy
- **Optimizer**: AdamW
- **Scheduler**: Cosine annealing with warmup
- **Batch Size**: 4-8 (CPU-limited)
- **Gradient Clipping**: 1.0

## 📊 Model Presets

| Preset | Parameters | Layers | Heads | d_model | d_ff | Context | Vocab Size |
|--------|-----------|--------|-------|---------|-------|---------|------------|
| tiny | ~100k | 2 | 4 | 128 | 512 | 128 | 4,096 |
| small | ~500k | 4 | 8 | 256 | 1,024 | 128 | 8,192 |
| medium | ~2M | 6 | 8 | 512 | 2,048 | 256 | 16,384 |

## 🎓 Learning Objectives

By working through this project, you will learn:

1. **Tokenization**
   - How text is converted to tokens
   - Byte Pair Encoding (BPE) algorithm
   - Vocabulary building
   - Special tokens and their roles

2. **Transformer Architecture**
   - Embeddings and positional encoding
   - Multi-head self-attention
   - Feed-forward networks
   - Layer normalization
   - Residual connections

3. **Training Process**
   - Next token prediction objective
   - Cross-entropy loss
   - Backpropagation
   - Optimizers and learning rates
   - Gradient clipping
   - Batch processing

4. **Data Pipeline**
   - Data loading and preprocessing
   - Text cleaning and filtering
   - Train/validation/test splits
   - Deduplication
   - Quality control

5. **Evaluation**
   - Perplexity calculation
   - Accuracy metrics
   - Coherence and diversity metrics
   - Task-based evaluation
   - Model comparison

6. **Experiment Tracking**
   - Checkpoint management
   - Training logging
   - Reproducibility
   - Version control

## 📈 Training Process

### 1. Prepare Data

```python
from scratchlm.data import load_text_file, clean_text, deduplicate_texts

# Load text files
texts = load_text_file("data/my_texts.txt")

# Clean and filter
cleaned_texts = [clean_text(t) for t in texts if clean_text(t) is not None]

# Deduplicate
dedup_texts = deduplicate_texts(cleaned_texts)
```

### 2. Train Tokenizer

```python
from scratchlm.tokenizer import BPETokenizer

# Train BPE tokenizer
tokenizer = BPETokenizer(vocab_size=8192)
tokenizer.train(dedup_texts, show_progress=True)

# Save tokenizer
tokenizer.save("tokenizers/my_tokenizer")
```

### 3. Create Model

```python
from scratchlm.model import ScratchLM
from scratchlm.model.config import get_config_preset

# Get configuration
model_config, training_config, tokenizer_config = get_config_preset("tiny")

# Create model
model = ScratchLM(model_config)
print(f"Parameters: {model.get_num_params():,}")
```

### 4. Train Model

```python
from scratchlm.training import Trainer

# Create trainer
trainer = Trainer(
    model_config=model_config,
    training_config=training_config,
    tokenizer_config=tokenizer_config,
)

# Train
trainer.train(
    train_texts=train_texts,
    val_texts=val_texts,
    train_tokenizer=False,  # Already trained
)
```

### 5. Evaluate Model

```python
from scratchlm.evaluation import EvaluationSuite

# Create evaluation suite
suite = EvaluationSuite(model=model, tokenizer=tokenizer)

# Evaluate
results = suite.run_full_evaluation(dataset=test_dataset)
print(f"Test loss: {results.metrics['test_loss']:.4f}")
print(f"Test perplexity: {results.metrics['test_perplexity']:.2f}")
```

### 6. Generate Text

```python
# Generate from a prompt
prompt = "The quick brown fox"
generated = model.generate(
    prompt=prompt,
    max_length=50,
    temperature=0.7,
    tokenizer=tokenizer,
)
print(generated)
```

## 🎯 Current Status

- ✅ Project structure created
- ✅ Configuration system implemented
- ✅ Tokenizer (BPE) implemented
- ✅ Model architecture implemented
- ✅ Training loop implemented
- ✅ Evaluation suite implemented
- ✅ Command-line scripts created
- ✅ Documentation started

## 📁 Data Sources for Phase 1

For English-only training, consider these public domain sources:

1. **Project Gutenberg** (https://www.gutenberg.org/)
   - Classic literature (Shakespeare, Austen, Dickens, etc.)
   - Public domain, high-quality English

2. **Wikipedia** (https://en.wikipedia.org/)
   - Use simple Wikipedia dump or API
   - Filter for English articles

3. **Public Domain Books**
   - Children's stories
   - Educational texts
   - Simple non-fiction

4. **News Articles** (public domain or with permission)
   - Simple news reporting
   - Educational content

5. **Conversations**
   - Movie scripts (public domain)
   - Play dialogues
   - Simple chat logs

## 🚨 Important Rules

1. **ENGLISH ONLY** - No Luau, Roblox, or coding data in Phase 1
2. **FROM SCRATCH** - Random initialization, no pretrained weights
3. **QUALITY FIRST** - Clean data, proper evaluation, honest results
4. **SMALL FIRST** - Start with tiny models that can train on CPU
5. **REPRODUCIBLE** - Set random seeds, save configurations
6. **DOCUMENTED** - Track experiments, record failures, measure progress

## 🤝 Contributing

This is a personal learning project, but contributions are welcome! Please:

1. Follow the phase progression (English first!)
2. Maintain code quality and documentation
3. Test changes before committing
4. Keep the project structure clean

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the "Let's build GPT" series by Andrej Karpathy
- Thanks to the PyTorch team for their excellent framework
- Grateful to the open-source AI community for their educational resources

---

*Built with ❤️ for learning and understanding AI*
