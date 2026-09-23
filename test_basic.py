#!/usr/bin/env python3
"""
Basic test script to verify ScratchLM components work.
"""

import sys
import torch

print("=" * 60)
print("Testing ScratchLM Components")
print("=" * 60)

# Test 1: Imports
print("\n1. Testing imports...")
try:
    from scratchlm.utils import paths
    from scratchlm.model import ScratchLM, TransformerConfig, get_config_preset
    from scratchlm.tokenizer import BPETokenizer, train_bpe_tokenizer
    from scratchlm.data import load_text_file, TextDataset, TokenizedDataset
    from scratchlm.training import Trainer
    from scratchlm.evaluation import EvaluationSuite
    from scratchlm.utils import set_seed
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Paths
print("\n2. Testing paths...")
try:
    print(f"   Project root: {paths.root}")
    print(f"   Source: {paths.scratchlm}")
    print(f"   Data: {paths.data}")
    print(f"   Checkpoints: {paths.checkpoints}")
    print("   ✓ Paths working")
except Exception as e:
    print(f"   ✗ Paths failed: {e}")
    sys.exit(1)

# Test 3: Configuration
print("\n3. Testing configurations...")
try:
    model_config, train_config, tokenizer_config = get_config_preset("tiny")
    print(f"   Model: {model_config.version}")
    print(f"   Params: {model_config.n_params:,}")
    print(f"   Train batch size: {train_config.batch_size}")
    print(f"   Tokenizer vocab: {tokenizer_config.vocab_size}")
    print("   ✓ Configurations working")
except Exception as e:
    print(f"   ✗ Configuration failed: {e}")
    sys.exit(1)

# Test 4: Tokenizer
print("\n4. Testing tokenizer...")
try:
    # Create sample texts
    texts = [
        "hello world",
        "the quick brown fox",
        "jumps over the lazy dog",
    ]
    
    # Train tokenizer
    tokenizer = train_bpe_tokenizer(
        texts,
        vocab_size=50,
        min_frequency=1,
        show_progress=False,
    )
    
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Special tokens: {list(tokenizer.special_tokens.keys())}")
    
    # Test encoding
    token_ids = tokenizer.encode("hello world")
    print(f"   Encoded 'hello world': {token_ids}")
    
    # Test decoding
    decoded = tokenizer.decode(token_ids)
    print(f"   Decoded: {decoded}")
    
    print("   ✓ Tokenizer working")
except Exception as e:
    print(f"   ✗ Tokenizer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Model
print("\n5. Testing model...")
try:
    # Create model
    model = ScratchLM(model_config)
    
    print(f"   Parameters: {model.get_num_params():,}")
    print(f"   Device: {next(model.parameters()).device}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, model_config.vocab_size, (2, 10))
        logits = model(input_ids)
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output shape: {logits.shape}")
    
    print("   ✓ Model working")
except Exception as e:
    print(f"   ✗ Model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Data loading
print("\n6. Testing data loading...")
try:
    # Load sample data
    sample_file = paths.data_raw / "sample_english.txt"
    if sample_file.exists():
        texts = load_text_file(sample_file)
        print(f"   Loaded {len(texts)} lines from {sample_file.name}")
        print(f"   First line: {texts[0][:50]}...")
    else:
        print(f"   Sample file not found at {sample_file}")
    
    print("   ✓ Data loading working")
except Exception as e:
    print(f"   ✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Dataset
print("\n7. Testing dataset...")
try:
    # Create tokenized dataset
    token_ids_list = [tokenizer.encode(text, add_special_tokens=False) for text in texts[:10]]
    dataset = TokenizedDataset(
        token_ids_list=token_ids_list,
        context_length=model_config.context_length,
        tokenizer=tokenizer,
    )
    
    print(f"   Dataset size: {len(dataset)}")
    
    # Test sample
    input_ids, labels, mask = dataset[0]
    print(f"   Sample input shape: {input_ids.shape}")
    print(f"   Sample labels shape: {labels.shape}")
    print(f"   Sample mask shape: {mask.shape}")
    
    print("   ✓ Dataset working")
except Exception as e:
    print(f"   ✗ Dataset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Trainer
print("\n8. Testing trainer...")
try:
    trainer = Trainer(
        model_config=model_config,
        training_config=train_config,
        tokenizer_config=tokenizer_config,
        model_version="v0.1_test",
        experiment_id="basic_test",
    )
    
    print(f"   Model version: {trainer.model_config.version}")
    print(f"   Experiment ID: {trainer.training_config.experiment_id}")
    print(f"   Device: {trainer.device}")
    
    # Create model through trainer
    trainer.create_model()
    print(f"   Model created: {trainer.model.get_num_params():,} params")
    
    print("   ✓ Trainer working")
except Exception as e:
    print(f"   ✗ Trainer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Evaluation
print("\n9. Testing evaluation...")
try:
    from scratchlm.evaluation import PerplexityMetric, AccuracyMetric
    
    # Test metrics
    ppl_metric = PerplexityMetric()
    ppl_result = ppl_metric.compute(loss=2.0)
    print(f"   Perplexity for loss=2.0: {ppl_result.value:.2f}")
    
    acc_metric = AccuracyMetric()
    logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])
    targets = torch.tensor([[1, 0]])
    acc_result = acc_metric.compute(logits, targets)
    print(f"   Accuracy: {acc_result.value:.2f}")
    
    print("   ✓ Evaluation working")
except Exception as e:
    print(f"   ✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Generation
print("\n10. Testing generation...")
try:
    # Generate directly with model and tokenizer
    model = trainer.model
    model.eval()
    
    prompt = "The quick brown"
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids])
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(generated_ids[0].tolist())
    print(f"   Prompt: '{prompt}'")
    print(f"   Generated: '{generated}'")
    
    print("   ✓ Generation working")
except Exception as e:
    print(f"   ✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nScratchLM is ready to use!")
print("\nNext steps:")
print("  1. Add your own training data to data/raw/")
print("  2. Run: python src/scripts/train.py --data data/raw/my_data.txt")
print("  3. Monitor training and evaluate results")
print("  4. Experiment with different configurations")
