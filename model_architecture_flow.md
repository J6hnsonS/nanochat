# 🏗️ Exact Architecture Flow of nanochat d20 Model

## 📋 Quick Answer

Your proposed structure is **almost correct**, but here's the **precise** flow:

```
Input tokens (B, T)
  ↓
Token Embedding (wte)
  ↓
RMSNorm
  ↓
┌─────────────────────────────────┐
│  20× Transformer Blocks:        │
│                                 │
│  For each block:                │
│    x_input (residual saved)     │
│      ↓                           │
│    RMSNorm                       │
│      ↓                           │
│    Attention (Q,K,V + RoPE)     │
│      ↓                           │
│    x = x_input + attn_out       │  ← Residual Add
│      ↓                           │
│    x_input2 (residual saved)    │
│      ↓                           │
│    RMSNorm                       │
│      ↓                           │
│    MLP (Linear→ReLU²→Linear)    │
│      ↓                           │
│    x = x_input2 + mlp_out       │  ← Residual Add
│      ↓                           │
│  (repeat 20 times)              │
└─────────────────────────────────┘
  ↓
Final RMSNorm
  ↓
LM Head (Linear)
  ↓
Logit Softcapping (tanh)
  ↓
Output logits (B, T, vocab_size)
  ↓
Cross-entropy Loss
```

---

## 🔍 Detailed Code-Level Architecture

Let me trace through the **actual code** to show you EXACTLY what happens:

### 1️⃣ Token Embedding + Initial Norm

```python
# From gpt.py lines 256-257
x = self.transformer.wte(idx)        # (B, T) → (B, T, 1280)
x = norm(x)                           # RMSNorm right after embedding
```

**Shape:** `(batch, seq_len) → (batch, seq_len, 1280)`

---

### 2️⃣ Single Transformer Block (repeated 20×)

```python
# From gpt.py lines 132-135 (Block.forward)
def forward(self, x, cos_sin, kv_cache):
    x = x + self.attn(norm(x), cos_sin, kv_cache)  # Pre-norm + residual
    x = x + self.mlp(norm(x))                      # Pre-norm + residual
    return x
```

Let me break this down step-by-step:

#### Step A: Attention Sub-layer
```python
# Pseudo-code expansion:
x_input = x                    # Save for residual
x_normed = norm(x_input)       # RMSNorm (pre-norm)
attn_out = attention(x_normed) # Attention with RoPE inside
x = x_input + attn_out         # Residual connection
```

**Inside Attention:**
```python
# From gpt.py lines 66-110 (CausalSelfAttention.forward)
1. Project to Q, K, V
2. Reshape to multi-head format
3. Apply RoPE to Q and K
4. Apply QK normalization
5. Scaled dot-product attention
6. Concatenate heads
7. Output projection
```

#### Step B: MLP Sub-layer
```python
# Pseudo-code expansion:
x_input = x                    # Save for residual (new residual!)
x_normed = norm(x_input)       # RMSNorm (pre-norm)
mlp_out = mlp(x_normed)        # MLP with ReLU²
x = x_input + mlp_out          # Residual connection
```

**Inside MLP:**
```python
# From gpt.py lines 119-123
x = self.c_fc(x)          # Linear: 1280 → 5120
x = F.relu(x).square()    # ReLU² activation
x = self.c_proj(x)        # Linear: 5120 → 1280
```

---

### 3️⃣ Final Processing

```python
# From gpt.py lines 258-276
# After all 20 blocks:
for block in self.transformer.h:
    x = block(x, cos_sin, kv_cache)
    
x = norm(x)                           # Final RMSNorm

# Forward the lm_head (compute logits)
softcap = 15
if targets is not None:
    logits = self.lm_head(x)          # (B, T, 1280) → (B, T, 65536)
    logits = softcap * torch.tanh(logits / softcap)  # Logits softcapping
    logits = logits.float()           # Use fp32 for numerical stability
    loss = F.cross_entropy(...)       # Cross-entropy loss
    return loss
```

---

## 🎯 Your Original Structure vs Actual

### ❌ Your Version (close but not quite):
```
Embedding → 20×(
  Attn layer(RMSnorm → RoPE for Q,K → Multihead attn) →
  MLP layer(RMSnorm → MLP → ReLU)
) → LM head → softcap → Muon & Adam
```

### ✅ Actual Structure:
```
Embedding → RMSNorm → 20×(
  RMSNorm → Attn(RoPE inside) → Add Residual →
  RMSNorm → MLP(ReLU² inside) → Add Residual
) → RMSNorm → LM head → Softcap → Loss
```

### 🔑 Key Differences:

1. **RMSNorm BEFORE each sublayer** (pre-norm), not after
2. **Residual connections AFTER each sublayer** (x = x + sublayer(norm(x)))
3. **Initial RMSNorm** right after token embedding
4. **Final RMSNorm** before LM head
5. **RoPE is applied INSIDE attention** (not a separate layer)
6. **ReLU² is INSIDE the MLP** (not after)
7. **Softcapping is part of forward pass**, not a separate layer
8. **Muon/AdamW are optimizers** (not part of model structure)

---

## 📐 Visual ASCII Architecture

```
                    INPUT: Token IDs [B, T]
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │     Token Embedding (wte)               │
        │     [B, T] → [B, T, 1280]              │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │          RMSNorm (initial)              │
        └─────────────────────────────────────────┘
                              │
        ╔═════════════════════╧═════════════════════╗
        ║                                           ║
        ║    TRANSFORMER BLOCK 0                    ║
        ║                                           ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  x_saved ← x  (save residual) │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │       RMSNorm(x)              │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  Project Q, K, V              │     ║
        ║    │  Q: [B,T,1280] → [B,10,T,128]│     ║
        ║    │  K: [B,T,1280] → [B,10,T,128]│     ║
        ║    │  V: [B,T,1280] → [B,10,T,128]│     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  Apply RoPE to Q, K           │     ║
        ║    │  (rotary position encoding)   │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  QK Normalization             │     ║
        ║    │  Q ← norm(Q), K ← norm(K)     │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  Scaled Dot-Product Attention │     ║
        ║    │  softmax(QK^T/√128) × V      │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  Concatenate & Project        │     ║
        ║    │  [B,10,T,128] → [B,T,1280]   │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  x ← x_saved + attn_output    │     ║
        ║    │  (RESIDUAL CONNECTION)        │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║    ╌╌╌╌╌╌╌╌╌╌╌╌╌╌│╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌       ║
        ║                  │                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  x_saved ← x  (save residual) │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │       RMSNorm(x)              │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  Linear (expand 4×)           │     ║
        ║    │  [B,T,1280] → [B,T,5120]     │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  ReLU² Activation             │     ║
        ║    │  x ← relu(x).square()         │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  Linear (project back)        │     ║
        ║    │  [B,T,5120] → [B,T,1280]     │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ║                  ▼                        ║
        ║    ┌───────────────────────────────┐     ║
        ║    │  x ← x_saved + mlp_output     │     ║
        ║    │  (RESIDUAL CONNECTION)        │     ║
        ║    └───────────────────────────────┘     ║
        ║                  │                        ║
        ╚═══════════════════╧═══════════════════════╝
                              │
                              ▼
                    (Repeat 19 more times)
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │       Final RMSNorm                     │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         LM Head (Linear)                │
        │    [B,T,1280] → [B,T,65536]            │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Logit Softcapping                  │
        │   logits ← 15*tanh(logits/15)          │
        │   (bounds logits to [-15, 15])         │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Convert to Float32                 │
        │   (for numerical stability)             │
        └─────────────────────────────────────────┘
                              │
                              ▼
                  OUTPUT: Logits [B, T, 65536]
                              │
                              ▼
                    (If training: compute loss)
        ┌─────────────────────────────────────────┐
        │      Cross-Entropy Loss                 │
        │   F.cross_entropy(logits, targets)     │
        └─────────────────────────────────────────┘
```

---

## 🔬 Code Trace Through One Forward Pass

Let's trace a concrete example with batch_size=2, seq_len=4:

```python
# Input
idx = [[1, 2, 3, 4],
       [5, 6, 7, 8]]  # shape: (2, 4)

# Step 1: Token Embedding
x = transformer.wte(idx)
# shape: (2, 4, 1280)

# Step 2: Initial norm
x = norm(x)
# shape: (2, 4, 1280)

# Step 3-22: 20 Transformer blocks
for block_idx in range(20):
    # ---- Attention sub-layer ----
    x_residual = x                    # Save for residual
    x_normed = norm(x)                # Pre-norm
    
    # Inside attention:
    q = c_q(x_normed)                 # (2,4,1280) → (2,4,1280)
    k = c_k(x_normed)                 # (2,4,1280) → (2,4,1280)
    v = c_v(x_normed)                 # (2,4,1280) → (2,4,1280)
    
    q = q.view(2, 4, 10, 128)         # Reshape to heads
    k = k.view(2, 4, 10, 128)
    v = v.view(2, 4, 10, 128)
    
    q = apply_rotary_emb(q, cos, sin) # Apply RoPE
    k = apply_rotary_emb(k, cos, sin)
    
    q = norm(q)                       # QK norm
    k = norm(k)
    
    q = q.transpose(1, 2)             # (2,10,4,128)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = y.transpose(1, 2)             # (2,4,10,128)
    y = y.reshape(2, 4, 1280)         # Concatenate heads
    
    attn_out = c_proj(y)              # Output projection
    
    x = x_residual + attn_out         # RESIDUAL ADD
    
    # ---- MLP sub-layer ----
    x_residual = x                    # Save for residual
    x_normed = norm(x)                # Pre-norm
    
    # Inside MLP:
    h = c_fc(x_normed)                # (2,4,1280) → (2,4,5120)
    h = F.relu(h).square()            # ReLU² activation
    mlp_out = c_proj(h)               # (2,4,5120) → (2,4,1280)
    
    x = x_residual + mlp_out          # RESIDUAL ADD
    
# shape after 20 blocks: (2, 4, 1280)

# Step 23: Final norm
x = norm(x)
# shape: (2, 4, 1280)

# Step 24: LM head
logits = lm_head(x)
# shape: (2, 4, 65536)

# Step 25: Softcapping
softcap = 15
logits = softcap * torch.tanh(logits / softcap)
# shape: (2, 4, 65536), values in [-15, 15]

# Step 26: Convert to float32
logits = logits.float()

# Step 27: Compute loss (if training)
if targets is not None:
    loss = F.cross_entropy(logits.view(-1, 65536), targets.view(-1))
    return loss
else:
    return logits
```

---

## 🎯 Key Architectural Details

### 1. **Pre-Norm Architecture**
```python
# PRE-NORM (what nanochat uses):
x = x + sublayer(norm(x))

# vs POST-NORM (older, like original Transformer):
x = norm(x + sublayer(x))
```

**Why pre-norm?**
- Better gradient flow
- More stable training
- Can train deeper models

### 2. **Two Separate Residual Streams**
```python
# Block has TWO residual connections:
x = x + attention(norm(x))    # First residual
x = x + mlp(norm(x))          # Second residual (NOT from original input!)
```

Each sub-layer has its own residual, they don't share!

### 3. **RMSNorm Placement**
```
- After token embedding (stabilize magnitudes)
- Before EACH attention layer
- Before EACH MLP layer  
- After final transformer block (before logits)
```

Total: `1 + 20×2 + 1 = 42 RMSNorm calls` (but 0 parameters!)

### 4. **RoPE is Inside Attention**

RoPE is not a "layer" - it's an operation inside attention:
```python
q = self.c_q(x)               # Project to Q
q = apply_rotary_emb(q, ...)  # Apply RoPE IN-PLACE
# Q now has positional information!
```

### 5. **Softcapping Happens in Forward**

Not a separate layer, just a transformation:
```python
logits = lm_head(x)
logits = 15 * tanh(logits / 15)  # Bounds to [-15, 15]
```

---

## 🧪 Verify This Yourself

Run this to see the actual structure:

```python
import torch
from nanochat.gpt import GPT, GPTConfig

# Create d20 model
config = GPTConfig(
    sequence_len=2048,
    vocab_size=65536,
    n_layer=20,
    n_head=10,
    n_kv_head=10,
    n_embd=1280
)

model = GPT(config)

# Print architecture
print(model)

# Trace a forward pass
dummy_input = torch.randint(0, 65536, (2, 4))
with torch.no_grad():
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be (2, 4, 65536)
```

---

## 📝 Summary: Correct Structure

```
┌─────────────────────────────────────────────────────────┐
│ Input tokens [B, T]                                     │
│   ↓                                                     │
│ Token Embedding → [B, T, 1280]                         │
│   ↓                                                     │
│ RMSNorm                                                 │
│   ↓                                                     │
│ ┌─────────────────────────────────────────────────┐   │
│ │ FOR i = 0 to 19:                                │   │
│ │   x ← x + Attention(RMSNorm(x))  [with RoPE]    │   │
│ │   x ← x + MLP(RMSNorm(x))        [with ReLU²]   │   │
│ └─────────────────────────────────────────────────┘   │
│   ↓                                                     │
│ RMSNorm                                                 │
│   ↓                                                     │
│ LM Head → [B, T, 65536]                                │
│   ↓                                                     │
│ Logit Softcapping (15*tanh(·/15))                      │
│   ↓                                                     │
│ Float32 conversion                                      │
│   ↓                                                     │
│ Cross-Entropy Loss (if training)                        │
└─────────────────────────────────────────────────────────┘
```

**Optimizers (separate from architecture):**
- Muon: for all Linear layers in transformer blocks (70% of params)
- AdamW: for token embedding + LM head (30% of params)

---

Hope this clarifies the exact structure! The key insight is the **pre-norm + residual** pattern repeated 20 times. 🎯
