# ScratchLM Architecture Documentation

This document describes the architecture and design decisions behind ScratchLM.

## Overview

ScratchLM is a **decoder-only transformer** language model designed for **causal language modeling** - predicting the next token given the previous tokens. This architecture is the foundation of most modern language models including GPT, Llama, and others.

## Model Architecture

### High-Level Structure

```
Input Tokens → Token Embeddings → Positional Embeddings → Transformer Blocks → Layer Norm → Output Projection → Logits
```

### Components

#### 1. Token Embeddings

- **Purpose**: Convert token IDs to dense vector representations
- **Implementation**: `nn.Embedding` layer
- **Size**: `vocab_size × d_model`
- **Initialization**: Normal distribution with std=0.02

```python
self.token_embeddings = nn.Embedding(
    num_embeddings=config.vocab_size,
    embedding_dim=config.d_model,
)
```

#### 2. Positional Embeddings

- **Purpose**: Add positional information to token embeddings
- **Type**: Learned embeddings (not sinusoidal)
- **Size**: `context_length × d_model`
- **Added to**: Token embeddings

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, max_length: int, d_model: int):
        self.embeddings = nn.Parameter(torch.zeros(max_length, d_model))
    
    def forward(self, x):
        return x + self.embeddings[:x.size(1), :]
```

#### 3. Transformer Blocks

Each transformer block consists of:

```
Input → Layer Norm → Multi-Head Attention → Residual → Layer Norm → Feed-Forward → Residual → Output
```

##### Multi-Head Attention

- **Purpose**: Capture relationships between tokens
- **Heads**: Multiple parallel attention heads
- **Head Dimension**: `d_model / n_heads`
- **Scaling**: `1 / sqrt(head_dim)`
- **Masking**: Causal mask (prevents attending to future tokens)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d_model // n_heads)
```

**Attention Mechanism**:
1. Project input to Q, K, V matrices
2. Split into multiple heads
3. Compute attention scores: `Q @ K^T * scale`
4. Apply causal mask (upper triangular matrix with -inf)
5. Apply softmax to get attention weights
6. Apply dropout to attention weights
7. Compute output: `attention_weights @ V`
8. Concatenate heads and project back to d_model

##### Feed-Forward Network

- **Purpose**: Non-linear transformation of each position independently
- **Structure**: Two linear layers with activation in between
- **Dimensions**: `d_model → d_ff → d_model`
- **Activation**: GELU (default), ReLU, or SiLU

```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

##### Layer Normalization

- **Purpose**: Stabilize training by normalizing activations
- **Parameters**: Learnable gamma (scale) and beta (shift)
- **Applied**: Before attention and FFN, and at the end

```python
class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
```

#### 4. Output Projection

- **Purpose**: Convert final hidden states to vocabulary logits
- **Type**: Linear layer
- **Size**: `d_model × vocab_size`
- **Tied Embeddings**: Option to share weights with token embeddings

```python
if config.tie_embeddings:
    # Use token embeddings transposed
    logits = torch.matmul(x, self.token_embeddings.weight.transpose(0, 1))
else:
    logits = self.output_proj(x)
```

## Forward Pass

```python
def forward(self, input_ids, attention_mask=None):
    # 1. Token embeddings
    x = self.token_embeddings(input_ids)  # (batch, seq_len, d_model)
    
    # 2. Positional embeddings
    x = self.position_embeddings(x)
    
    # 3. Transformer blocks
    for block in self.blocks:
        x = block(x, attention_mask)
    
    # 4. Final layer norm
    x = self.final_ln(x)
    
    # 5. Output projection
    if self.output_proj:
        logits = self.output_proj(x)
    else:
        logits = torch.matmul(x, self.token_embeddings.weight.transpose(0, 1))
    
    return logits
```

## Training Objective

### Next Token Prediction

The model is trained to predict the next token given the previous tokens.

**Input**: Sequence of token IDs `[x₁, x₂, ..., xₙ]`
**Target**: Sequence of token IDs `[x₂, x₃, ..., xₙ₊₁]` (shifted by 1)

### Loss Function

**Cross-Entropy Loss** with softmax:

```python
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

# Reshape for loss calculation
logits_flat = logits.view(-1, vocab_size)  # (batch * seq_len, vocab_size)
targets_flat = targets.view(-1)  # (batch * seq_len)

loss = loss_fn(logits_flat, targets_flat)
```

### Why Cross-Entropy?

Cross-entropy is equivalent to negative log-likelihood, which is the standard loss for classification tasks. For language modeling:

- It measures how well the model predicts the actual next token
- Lower loss = better predictions
- Perplexity = exp(loss) (human-interpretable metric)

## Hyperparameters

### Model Hyperparameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|--------|
| `d_model` | Embedding dimension | 128-512 | Model capacity, memory usage |
| `n_layers` | Number of transformer blocks | 2-6 | Model depth, expressiveness |
| `n_heads` | Number of attention heads | 4-8 | Parallel attention, must divide d_model |
| `d_ff` | Feed-forward dimension | 4×d_model | FFN capacity |
| `context_length` | Maximum sequence length | 128-256 | Memory usage, long-range dependencies |
| `vocab_size` | Vocabulary size | 4k-16k | Token coverage, embedding size |
| `dropout` | Dropout rate | 0.1 | Regularization, prevents overfitting |

### Training Hyperparameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|--------|
| `batch_size` | Samples per batch | 4-8 (CPU) | Memory usage, gradient stability |
| `learning_rate` | Step size for optimization | 1e-4 to 3e-4 | Convergence speed, stability |
| `weight_decay` | L2 regularization | 0.01 | Prevents large weights |
| `warmup_steps` | Learning rate warmup | 100-1000 | Stable initial training |
| `num_epochs` | Training iterations | 10-100 | Total training time |
| `clip_grad_norm` | Gradient clipping | 1.0 | Prevents exploding gradients |

## Design Decisions

### Why Decoder-Only?

1. **Simplicity**: Fewer components than encoder-decoder
2. **Causal Modeling**: Naturally suited for next-token prediction
3. **Standard**: Most modern LMs use decoder-only (GPT, Llama, etc.)
4. **Efficiency**: No need for encoder-decoder attention

### Why Learned Positional Embeddings?

1. **Simplicity**: Easier to implement than sinusoidal
2. **Flexibility**: Can learn any positional pattern
3. **Standard**: Used in most modern implementations
4. **Performance**: Works well in practice

**Note**: Sinusoidal embeddings have the advantage of handling longer sequences than trained for, but learned embeddings are simpler and often perform better for the trained context length.

### Why Tied Embeddings?

1. **Parameter Efficiency**: Saves vocab_size × d_model parameters
2. **Theoretical**: Input and output share the same embedding space
3. **Standard**: Used in many implementations

**Trade-off**: May slightly reduce performance, but the parameter savings are significant for small models.

### Why GELU Activation?

1. **Smooth**: Differentiable everywhere (unlike ReLU)
2. **Non-linearity**: More expressive than linear
3. **Standard**: Used in BERT and other transformers
4. **Performance**: Works well in practice

**Alternatives**: ReLU (faster), SiLU/Swish (smooth like GELU)

## Model Sizes

### Tiny Model (CPU-friendly)

```
Config:
- d_model: 128
- n_layers: 2
- n_heads: 4
- d_ff: 512
- context_length: 128
- vocab_size: 4096

Parameters: ~100,000
Memory per batch (batch=4): ~10-20 MB
Training time per epoch: Minutes to hours on CPU
```

### Small Model

```
Config:
- d_model: 256
- n_layers: 4
- n_heads: 8
- d_ff: 1024
- context_length: 128
- vocab_size: 8192

Parameters: ~500,000
Memory per batch (batch=4): ~20-40 MB
Training time per epoch: Hours on CPU
```

### Medium Model (GPU recommended)

```
Config:
- d_model: 512
- n_layers: 6
- n_heads: 8
- d_ff: 2048
- context_length: 256
- vocab_size: 16384

Parameters: ~2,000,000
Memory per batch (batch=2): ~100-200 MB
Training time per epoch: Hours on GPU
```

## Mathematical Formulation

### Self-Attention

Given input `X ∈ ℝ^{n×d}` where `n` is sequence length and `d` is d_model:

```
Q = XW_q, K = XW_k, V = XW_v  # Linear projections

Attention(Q, K, V) = softmax(QK^T / √d_k)V  # Scaled dot-product attention
```

### Multi-Head Attention

```
head_i = Attention(XW_q^i, XW_k^i, XW_v^i) for i = 1...h

MultiHead(X) = Concat(head_1, ..., head_h)W_o
```

where `h` is number of heads, and `W_o` is output projection.

### Transformer Block

```
FFN(x) = W_2 σ(W_1 x + b_1) + b_2  # Feed-forward network

TransformerBlock(x) = x + FFN(LayerNorm(x + MultiHeadAttention(LayerNorm(x))))
```

where `σ` is the activation function (GELU).

### Full Model

```
ScratchLM(x) = Output(LayerNorm(TransformerBlock_n(...TransformerBlock_1(Embed(x) + PosEmbed)...)))
```

## Implementation Details

### Causal Masking

To prevent the model from attending to future tokens, we apply a causal mask:

```python
# Create upper triangular mask (1s on and above diagonal)
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

# Invert and convert to -inf for masking
causal_mask = causal_mask.bool()
scores = scores.masked_fill(~causal_mask, float('-inf'))
```

This ensures that position `i` can only attend to positions `≤ i`.

### Padding Mask

To ignore padding tokens in attention:

```python
# attention_mask: (batch_size, seq_len) with 1=real, 0=padding
mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
mask = mask.expand(-1, -1, seq_len, -1)  # (batch, 1, seq_len, seq_len)

scores = scores.masked_fill(~mask.bool(), float('-inf'))
```

### Gradient Clipping

To prevent exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This clips the norm of the gradient to prevent individual gradients from becoming too large.

### Mixed Precision (Future)

For faster training on GPUs:

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    logits = model(input_ids)
    loss = loss_fn(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Note**: Not currently used as we're targeting CPU training initially.

## Performance Considerations

### Memory Usage

The main memory consumers are:

1. **Model parameters**: `n_params × 4 bytes` (float32)
2. **Gradients**: Same as parameters
3. **Optimizer states**: AdamW uses 2× parameters
4. **Activations**: Intermediate values during forward pass

For a tiny model (~100k params):
- Parameters: ~400 KB
- Gradients: ~400 KB
- Optimizer: ~800 KB
- Total: ~1.6 MB (plus activations)

### Computation

The main computational costs are:

1. **Attention**: `O(n² × d_model)` per layer (n = sequence length)
2. **FFN**: `O(n × d_model × d_ff)` per layer

For short sequences (n=128), attention is manageable on CPU.
For long sequences (n=1024+), attention becomes expensive.

### CPU vs GPU

| Aspect | CPU | GPU |
|--------|-----|-----|
| Parallelism | Limited (few cores) | Massive (thousands of cores) |
| Memory Bandwidth | Low | High |
| Matrix Operations | Slow | Fast (optimized) |
| Training Speed | Slow (hours-days) | Fast (minutes-hours) |
| Memory | 4-12 GB typical | 8-24 GB typical |

**Recommendation**: Start with CPU for tiny/small models, then scale to GPU for larger models.

## Extensions

### Rotary Positional Embeddings

Alternative to learned embeddings that can handle longer sequences:

```python
# Rotary embeddings use complex numbers to encode position
# This is more advanced and not currently implemented
```

### ALiBi (Attention with Linear Biases)

Alternative to causal masking that doesn't require explicit masks:

```python
# Add bias based on distance: -0.0001 * (i - j)
# This penalizes attention to distant tokens
```

### Flash Attention

Memory-efficient attention implementation:

```python
# Uses tiling to reduce memory usage
# Not currently implemented (requires CUDA)
```

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need". arXiv:1706.03762
2. Ba, J. L., et al. (2016). "Layer Normalization". arXiv:1607.06450
3. Sennrich, R., et al. (2016). "Neural Machine Translation of Rare Word Problem Using Subword Units". arXiv:1508.07909
4. Karpathy, A. (2022). "Let's build GPT: from scratch, in code, spelled out". https://youtu.be/kCc8Fm9dUt0
