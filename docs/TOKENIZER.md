# ScratchLM Tokenizer Documentation

This document describes the tokenizer implementation for ScratchLM.

## Overview

A **tokenizer** is the component that converts text into tokens (numerical IDs) that the model can process. It consists of two main operations:

1. **Tokenization**: Splitting text into subword units
2. **Numericalization**: Mapping tokens to unique integer IDs

## Why Build Our Own Tokenizer?

Building a custom tokenizer provides several benefits:

1. **Understanding**: Learn how tokenization actually works
2. **Control**: Full control over vocabulary and tokenization behavior
3. **Customization**: Can be tailored to specific needs (English, code, etc.)
4. **Independence**: No dependency on external pretrained tokenizers
5. **Reproducibility**: Tokenizer is versioned with the model

## Tokenizer Types

ScratchLM currently implements:

### Byte Pair Encoding (BPE)

**Status**: ✅ Implemented

BPE is a subword tokenization algorithm that:
1. Starts with a vocabulary of individual characters
2. Iteratively merges the most frequent pair of tokens
3. Builds up a vocabulary of subword units

**Advantages**:
- Handles out-of-vocabulary words well
- Good balance between word and character-level
- Used by GPT-2, RoBERTa, and others

**Disadvantages**:
- Training can be slow for large vocabularies
- Requires careful tuning of merge operations

## BPE Tokenizer Implementation

### Algorithm

```
1. Initialize vocabulary with all characters + special tokens
2. Count frequency of all token pairs in the corpus
3. Merge the most frequent pair into a new token
4. Update counts with the new merged tokens
5. Repeat until vocabulary size is reached
```

### Example

Training on: `["hello world", "hello there"]`

**Initial vocabulary** (characters + special tokens):
- Special: `<pad>=0`, `<unk>=1`, `<s>=2`, `</s>=3`
- Characters: `h, e, l, o,  , w, r, d, t, h, e, r, e` (with IDs 4+)

**After training**:
- Merges: `h + e → he`, `l + l → ll`, `o +  → o_`, etc.
- Final vocabulary includes: `hello`, `world`, `there`, etc.

### Tokenization Example

Text: `"hello world"`

With trained BPE tokenizer:
1. Split into base tokens: `[h, e, l, l, o,  , w, o, r, l, d]`
2. Apply merges: `[he, ll, o,  , w, o, rl, d]`
3. Further merges: `[hell, o,  , world]`
4. Final: `[hello,  , world]` (if these merges were learned)

Token IDs: `[ID_hello, ID_space, ID_world]`

## Tokenizer Configuration

```yaml
# tokenizer.yaml
tokenizer_type: bpe
vocab_size: 8192
min_frequency: 2
pad_token: "<pad>"
unk_token: "<unk>"
bos_token: "<s>"
eos_token: "</s>"
```

### Parameters

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|--------|
| `tokenizer_type` | Type of tokenizer | `bpe` | Algorithm to use |
| `vocab_size` | Target vocabulary size | 8192 | Number of tokens |
| `min_frequency` | Minimum pair frequency to merge | 2 | Controls merge threshold |
| `pad_token` | Padding token string | `<pad>` | Used for sequence padding |
| `unk_token` | Unknown token string | `<unk>` | Used for OOV tokens |
| `bos_token` | Beginning of sequence | `<s>` | Marks start of sequence |
| `eos_token` | End of sequence | `</s>` | Marks end of sequence |

## Special Tokens

### `<pad>` - Padding Token (ID: 0)
- **Purpose**: Fill sequences to a fixed length
- **Usage**: Added during batching to make all sequences same length
- **Masking**: Ignored in attention and loss calculation

### `<unk>` - Unknown Token (ID: 1)
- **Purpose**: Represent tokens not in vocabulary
- **Usage**: Returned when tokenizer encounters unseen text
- **Goal**: Minimize UNK tokens through good vocabulary

### `<s>` - Beginning of Sequence (ID: 2)
- **Purpose**: Mark the start of a sequence
- **Usage**: Added at the beginning of each input
- **Alternative names**: `<bos>`, `[CLS]` (in some models)

### `</s>` - End of Sequence (ID: 3)
- **Purpose**: Mark the end of a sequence
- **Usage**: Added at the end of each input
- **Alternative names**: `<eos>`, `[SEP]` (in some models)

## Training the Tokenizer

### Step-by-Step

```python
from scratchlm.tokenizer import BPETokenizer

# 1. Prepare training texts
train_texts = [
    "This is the first sentence.",
    "This is the second sentence.",
    "And here is a third one.",
]

# 2. Create tokenizer
tokenizer = BPETokenizer(
    vocab_size=100,  # Small for example
    min_frequency=1,
)

# 3. Train
tokenizer.train(train_texts, show_progress=True)

# 4. Save
tokenizer.save("tokenizers/my_tokenizer")
```

### Training Process

1. **Build Base Vocabulary**
   - Extract all unique characters from training texts
   - Assign IDs to each character
   - Add special tokens

2. **Initialize Counts**
   - Split all texts into base tokens (characters)
   - Count frequency of each token
   - Count frequency of each token pair

3. **Merge Loop**
   - Find most frequent pair
   - Merge pair into new token
   - Update counts
   - Repeat until vocab_size reached

4. **Finalize**
   - Build token-to-ID and ID-to-token mappings
   - Save merge rules

### Training Tips

1. **Vocabulary Size**
   - 4k-8k: Good for small models, CPU training
   - 16k-32k: Better for medium models
   - 50k+: For large models (requires GPU)

2. **Data Quality**
   - Clean text = better tokenizer
   - Remove boilerplate, headers, footers
   - Normalize whitespace and encoding

3. **Data Diversity**
   - Include various text types (stories, news, conversations)
   - Avoid too much repetition
   - Balance different domains

4. **Minimum Frequency**
   - Higher (5-10): More conservative, fewer merges
   - Lower (1-2): More aggressive, more merges
   - Default: 2 (good starting point)

## Using the Tokenizer

### Encoding (Text → Token IDs)

```python
# Encode text to token IDs
text = "Hello, world!"
token_ids = tokenizer.encode(text)
# [2, 45, 12, 89, 32, 100, 5, 3]  # <s>, Hello, ,,  world, !, </s>

# Without special tokens
token_ids = tokenizer.encode(text, add_special_tokens=False)
# [45, 12, 89, 32, 100, 5]
```

### Decoding (Token IDs → Text)

```python
# Decode token IDs to text
token_ids = [2, 45, 12, 89, 32, 100, 5, 3]
text = tokenizer.decode(token_ids)
# "Hello, world!"

# Skip special tokens
text = tokenizer.decode(token_ids, skip_special_tokens=True)
# "Hello, world!" (without <s> and </s>)
```

### Tokenization (Text → Tokens)

```python
# Get token strings (not IDs)
tokens = tokenizer.tokenize("Hello, world!")
# ['Hello', ',', ' ', 'world', '!']

# Convert tokens to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
# [45, 12, 89, 32, 100, 5]

# Convert IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(token_ids)
# ['Hello', ',', ' ', 'world', '!']
```

### Batch Processing

```python
# Encode batch of texts
texts = ["Hello", "World", "!"]
token_ids_list = tokenizer.batch_encode(texts)
# [[2, 45, 3], [2, 100, 3], [2, 5, 3]]

# Decode batch
decoded_texts = tokenizer.batch_decode(token_ids_list)
# ['Hello', 'World', '!']
```

### Padding

```python
# Pad sequences to same length
token_ids_list = [
    [1, 2, 3],      # Length 3
    [4, 5],         # Length 2
    [6, 7, 8, 9],   # Length 4
]

padded = tokenizer.pad(token_ids_list, max_length=5)
# array([
#   [1, 2, 3, 0, 0],
#   [4, 5, 0, 0, 0],
#   [6, 7, 8, 9, 0]
# ])

# Create attention mask (1 = attend, 0 = don't attend)
mask = tokenizer.create_attention_mask(padded)
# array([
#   [1, 1, 1, 0, 0],
#   [1, 1, 0, 0, 0],
#   [1, 1, 1, 1, 0]
# ])
```

## Tokenizer File Format

When saved, a tokenizer creates these files:

```
tokenizers/my_tokenizer/
├── config.json          # Tokenizer configuration
├── vocab.json           # Vocabulary mappings
└── merges.txt           # Merge rules (for BPE)
```

### config.json

```json
{
  "tokenizer_type": "bpe",
  "vocab_size": 100,
  "min_frequency": 2,
  "pad_token": "<pad>",
  "unk_token": "<unk>",
  "bos_token": "<s>",
  "eos_token": "</s>",
  "base_vocab_size": 260,
  "num_train_merges": 97
}
```

### vocab.json

```json
{
  "token_to_id": {
    "<pad>": 0,
    "<unk>": 1,
    "<s>": 2,
    "</s>": 3,
    "the": 4,
    "": 5,
    "hello": 6,
    ...
  },
  "id_to_token": {
    "0": "<pad>",
    "1": "<unk>",
    "2": "<s>",
    "3": "</s>",
    "4": "the",
    "5": " ",
    "6": "hello",
    ...
  }
}
```

### merges.txt

```
# Format: left right -> new (left_id right_id -> new_id)
h e -> he (10 15 -> 20)
l l -> ll (12 12 -> 21)
o  -> o_ (14 5 -> 22)
...
```

## Loading a Tokenizer

```python
from scratchlm.tokenizer import load_tokenizer

# Load from directory
tokenizer = load_tokenizer("tokenizers/my_tokenizer")

# Use with model
model_config.vocab_size = tokenizer.vocab_size
```

## Tokenizer Statistics

### Vocabulary Analysis

```python
# Get vocabulary
vocab = tokenizer.get_vocab()

# Vocabulary size
print(f"Vocab size: {len(vocab)}")

# Most common tokens
from collections import Counter
token_counts = Counter()
for text in texts:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_counts.update(token_ids)

print("Most common tokens:")
for token_id, count in token_counts.most_common(10):
    token = tokenizer.id_to_token(token_id)
    print(f"  {token}: {count}")
```

### Merge Analysis

```python
# Get merge rules
merges = tokenizer.get_merge_rules()

print("Sample merges:")
for left, right, new in merges[:10]:
    print(f"  {left} + {right} -> {new}")
```

## Common Issues and Solutions

### Issue: Too Many UNK Tokens

**Symptoms**: Many tokens are mapped to `<unk>`

**Solutions**:
1. Increase vocabulary size
2. Lower minimum frequency
3. Train on more diverse data
4. Check for data quality issues

### Issue: Poor Tokenization of Common Words

**Symptoms**: Common words are split into many tokens

**Solutions**:
1. Ensure common words appear frequently in training data
2. Increase vocabulary size
3. Check merge rules for expected merges

### Issue: Tokenizer is Slow

**Symptoms**: Tokenization takes a long time

**Solutions**:
1. Reduce vocabulary size
2. Use a more efficient implementation
3. Cache tokenized results
4. Use batch processing

### Issue: Inconsistent Tokenization

**Symptoms**: Same text tokenizes differently at different times

**Solutions**:
1. Check for randomness in tokenizer
2. Ensure deterministic merge application
3. Verify tokenizer is properly loaded

## Comparison with Other Tokenizers

| Tokenizer | Type | Pros | Cons | Used By |
|-----------|------|------|------|---------|
| BPE | Subword | Handles OOV well, good balance | Training can be slow | GPT-2, RoBERTa |
| WordPiece | Subword | Similar to BPE, slightly different | Less common | BERT |
| Unigram | Subword | Fast training | Requires EM algorithm | T5, ALBERT |
| Character | Character | Simple, no OOV | Very long sequences | Character LM |
| Byte | Byte | Universal, no OOV | Very long sequences | Canine |

## Future Enhancements

### 1. WordPiece Tokenizer

Similar to BPE but uses a different merging criterion.

### 2. SentencePiece

Google's tokenizer that supports multiple tokenization algorithms.

### 3. HuggingFace Tokenizers

Wrapper around HuggingFace's fast tokenizers (Rust-based).

### 4. Custom Token Patterns

- Word-level for some tokens
- Subword for others
- Hybrid approaches

### 5. Efficient Training

- Use a more efficient BPE implementation
- Support for large-scale training
- Incremental training

## References

1. Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare Word Problem Using Subword Units". arXiv:1508.07909
2. Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., ... & Dean, J. (2016). "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation". arXiv:1609.08144
3. Kudo, T., & Richardson, J. (2018). "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing". arXiv:1808.06226
4. HuggingFace. (2020). "Tokenizers: State of the Art Tokenizers". https://github.com/huggingface/tokenizers
