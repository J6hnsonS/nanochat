# 🎓 How to Count Transformer Parameters: Complete Guide

This guide teaches you how to calculate parameters in any transformer model, step by step.

## 🎯 Quick Answer for nanochat d20

```
Total: 560,988,160 parameters (561M)

Breakdown:
├── Token Embedding (wte):    83,886,080  (15.0%)
├── 20 Transformer Blocks:   393,216,000  (70.1%)
│   ├── Attention per block:   6,553,600  (33.3%)
│   └── MLP per block:        13,107,200  (66.7%)
└── LM Head (unembedding):    83,886,080  (15.0%)
```

---

## 📐 The Universal Formula

For **any** standard transformer with untied embeddings:

```
TOTAL = 2(V × D) + L × 12 × D²

Where:
  V = vocab_size
  D = d_model (hidden dimension)
  L = n_layers (depth)
```

**Why 12?**
- Attention: 4 matrices × D² = 4D²
- MLP: 2 matrices × 4D expansion = 8D²
- Total per block: 4D² + 8D² = 12D²

---

## 🔍 Step-by-Step Calculation

### Step 1: Identify the configuration

For d20:
```
vocab_size = 65,536
d_model = 1,280
n_layers = 20
```

### Step 2: Token Embedding

```python
# nn.Embedding(vocab_size, d_model)
token_embedding = vocab_size × d_model
                = 65,536 × 1,280
                = 83,886,080
```

### Step 3: ONE Attention Layer

Attention has **4 weight matrices**: Q, K, V, Output

```python
# For standard Multi-Head Attention (MHA):
# Each projection: d_model × d_model

Q_params = 1,280 × 1,280 = 1,638,400
K_params = 1,280 × 1,280 = 1,638,400
V_params = 1,280 × 1,280 = 1,638,400
O_params = 1,280 × 1,280 = 1,638,400
                           ─────────
Total:                     6,553,600

# OR use formula: 4 × d_model²
attention = 4 × 1,280² = 6,553,600 ✓
```

### Step 4: ONE MLP Layer

MLP has **2 weight matrices** with 4× expansion:

```python
# Expand: d_model → 4 × d_model
# Project: 4 × d_model → d_model

fc_params   = 1,280 × 5,120 = 6,553,600
proj_params = 5,120 × 1,280 = 6,553,600
                              ─────────
Total:                       13,107,200

# OR use formula: 8 × d_model²
mlp = 8 × 1,280² = 13,107,200 ✓
```

### Step 5: ONE Complete Block

```python
block = attention + mlp
      = 6,553,600 + 13,107,200
      = 19,660,800

# OR use formula: 12 × d_model²
block = 12 × 1,280² = 19,660,800 ✓
```

### Step 6: ALL Blocks

```python
all_blocks = n_layers × block
           = 20 × 19,660,800
           = 393,216,000
```

### Step 7: LM Head (Unembedding)

```python
# nn.Linear(d_model, vocab_size, bias=False)
lm_head = d_model × vocab_size
        = 1,280 × 65,536
        = 83,886,080
```

### Step 8: GRAND TOTAL

```python
total = token_embedding + all_blocks + lm_head
      = 83,886,080 + 393,216,000 + 83,886,080
      = 560,988,160
      ≈ 561M parameters ✓
```

---

## 🔑 Key Principles

### 1. Linear Layer Parameters

```python
# WITHOUT bias (nanochat, most modern models):
params = in_features × out_features

# WITH bias (older models like GPT-2):
params = in_features × out_features + out_features
```

### 2. What Has ZERO Parameters?

- ✅ **RMSNorm** (functional, no learnable weights)
- ✅ **LayerNorm** (if implemented without affine parameters)
- ✅ **RoPE** (precomputed, no learning)
- ✅ **Attention mechanism** (uses Q, K, V matrices, but mechanism itself has no params)
- ✅ **Activations** (ReLU, GELU, SiLU, etc.)
- ✅ **Residual connections** (just addition)
- ✅ **Dropout** (just masking)

### 3. Where Are The Parameters?

Only in:
- ✅ **Linear layers** (`nn.Linear`)
- ✅ **Embedding layers** (`nn.Embedding`)
- ✅ **LayerNorm** (if has learnable scale/shift)
- ✅ **Convolutions** (`nn.Conv1d`, etc.)

---

## 📊 Detailed Breakdown: Where Each Parameter Lives

### Token Embedding (wte)
```
Shape: [vocab_size, d_model]
Size:  [65,536, 1,280]
Params: 83,886,080

This is a lookup table: each token ID maps to a d_model vector
```

### Attention - Q Projection
```
Shape: [d_model, n_heads × head_dim]
Size:  [1,280, 10 × 128] = [1,280, 1,280]
Params: 1,638,400

Input [B,T,1280] → Output [B,T,1280] → Reshape to [B,T,10,128]
```

### Attention - K Projection
```
Same as Q: 1,638,400 params
```

### Attention - V Projection  
```
Same as Q: 1,638,400 params
```

### Attention - Output Projection
```
Shape: [n_heads × head_dim, d_model]
Size:  [10 × 128, 1,280] = [1,280, 1,280]
Params: 1,638,400

After attention: [B,T,10,128] → Concat → [B,T,1280] → Project → [B,T,1280]
```

### MLP - Expansion Layer (c_fc)
```
Shape: [d_model, 4 × d_model]
Size:  [1,280, 5,120]
Params: 6,553,600

Input [B,T,1280] → Output [B,T,5120]
```

### MLP - Projection Layer (c_proj)
```
Shape: [4 × d_model, d_model]
Size:  [5,120, 1,280]
Params: 6,553,600

Input [B,T,5120] → Output [B,T,1280]
```

### LM Head (unembedding)
```
Shape: [d_model, vocab_size]
Size:  [1,280, 65,536]
Params: 83,886,080

Final layer: [B,T,1280] → [B,T,65536] (logits for each token)
```

---

## 🎨 Visual Architecture with Parameter Counts

```
Input: Token IDs [B, T]
  ↓
┌─────────────────────────────────────┐
│ Token Embedding                     │
│ [vocab_size, d_model]              │
│ [65,536, 1,280]                    │
│ 83,886,080 params                  │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ RMSNorm                             │
│ 0 params (functional)              │
└─────────────────────────────────────┘
  ↓
┌═════════════════════════════════════┐
║ Transformer Block 1-20              ║
║ Each block: 19,660,800 params      ║
║                                     ║
║ ┌─────────────────────────────────┐ ║
║ │ RMSNorm (0 params)              │ ║
║ └─────────────────────────────────┘ ║
║   ↓                                 ║
║ ┌─────────────────────────────────┐ ║
║ │ Attention (6,553,600 params)    │ ║
║ │   • Q proj: 1,638,400           │ ║
║ │   • K proj: 1,638,400           │ ║
║ │   • V proj: 1,638,400           │ ║
║ │   • O proj: 1,638,400           │ ║
║ │   • RoPE: 0 params              │ ║
║ │   • QK norm: 0 params           │ ║
║ └─────────────────────────────────┘ ║
║   ↓                                 ║
║ [Residual Add] ← ← ← ← ← ← ← ← ←   ║
║   ↓                                 ║
║ ┌─────────────────────────────────┐ ║
║ │ RMSNorm (0 params)              │ ║
║ └─────────────────────────────────┘ ║
║   ↓                                 ║
║ ┌─────────────────────────────────┐ ║
║ │ MLP (13,107,200 params)         │ ║
║ │   • Expand: 6,553,600           │ ║
║ │   • ReLU²: 0 params             │ ║
║ │   • Project: 6,553,600          │ ║
║ └─────────────────────────────────┘ ║
║   ↓                                 ║
║ [Residual Add] ← ← ← ← ← ← ← ← ←   ║
║                                     ║
║ (Repeat 19 more times)             ║
║                                     ║
║ Total: 20 × 19,660,800             ║
║      = 393,216,000 params          ║
└═════════════════════════════════════┘
  ↓
┌─────────────────────────────────────┐
│ Final RMSNorm                       │
│ 0 params                           │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ LM Head                             │
│ [d_model, vocab_size]              │
│ [1,280, 65,536]                    │
│ 83,886,080 params                  │
└─────────────────────────────────────┘
  ↓
Output: Logits [B, T, vocab_size]

═══════════════════════════════════════
TOTAL: 560,988,160 params (561M)
═══════════════════════════════════════
```

---

## 🧪 Practice Problems

### Problem 1: Calculate d12 parameters
```
Given: vocab=65,536, d_model=768, n_layers=12
Answer: ?
```

<details>
<summary>Click for solution</summary>

```python
# Use formula: 2(V × D) + L × 12 × D²
total = 2 × (65536 × 768) + 12 × 12 × 768²
      = 100,663,296 + 84,934,656
      = 185,597,952
      ≈ 186M parameters ✓
```
</details>

### Problem 2: What if d20 used smaller vocab (32K)?
```
Given: vocab=32,000, d_model=1,280, n_layers=20
Answer: ?
```

<details>
<summary>Click for solution</summary>

```python
# Only embeddings change
old_embeddings = 2 × (65536 × 1280) = 167,772,160
new_embeddings = 2 × (32000 × 1280) = 81,920,000

# Blocks stay the same: 393,216,000

new_total = 81,920,000 + 393,216,000 = 475,136,000
≈ 475M parameters (85M less!)
```
</details>

### Problem 3: How many params saved with tied embeddings?
```
If d20 used tied embeddings instead of untied?
```

<details>
<summary>Click for solution</summary>

```python
# Untied: both token_emb AND lm_head
untied = 83,886,080 + 83,886,080 = 167,772,160

# Tied: only token_emb (shared with lm_head)
tied = 83,886,080

saved = 167,772,160 - 83,886,080 = 83,886,080
≈ 84M parameters saved (15% reduction!)

New total: 561M - 84M = 477M
```
</details>

---

## 📈 Comparison Table

| Model | d_model | Layers | Block | Total | Memory |
|-------|---------|--------|-------|-------|--------|
| d4 | 256 | 4 | 0.79M | 37M | 148 MB |
| d8 | 512 | 8 | 3.15M | 92M | 368 MB |
| d12 | 768 | 12 | 7.08M | 186M | 744 MB |
| d16 | 1024 | 16 | 12.58M | 336M | 1.34 GB |
| **d20** | **1280** | **20** | **19.66M** | **561M** | **2.24 GB** |
| d26 | 1664 | 26 | 33.22M | 1.08B | 4.32 GB |
| d32 | 2048 | 32 | 50.33M | 1.88B | 7.52 GB |

Memory = params × 4 bytes (FP32)

---

## 🚀 Tools Provided

I've created several tools for you:

1. **`parameter_calculation_guide.py`** - Step-by-step calculation with explanations
2. **`parameter_formula_guide.md`** - Mathematical formulas and examples
3. **`annotated_architecture.py`** - Code walkthrough with parameter counts
4. **`parameter_calculator.py`** - Interactive calculator for any config

### Usage:
```bash
# Run step-by-step guide
python3 parameter_calculation_guide.py

# Run annotated architecture
python3 annotated_architecture.py

# Run calculator (compare different configs)
python3 parameter_calculator.py

# Use as library
from parameter_calculator import calculate_params
params = calculate_params(vocab_size=65536, d_model=1280, n_layers=20)
print(f"Total: {params['total']:,}")
```

---

## 🎯 Summary

**To count transformer parameters:**

1. **Embeddings**: 2 × vocab_size × d_model (if untied)
2. **Each block**: 12 × d_model²
3. **All blocks**: n_layers × 12 × d_model²
4. **Total**: Add them up!

**Remember:**
- No parameters in: RMSNorm, RoPE, attention mechanism, activations
- Parameters only in: Linear layers, Embeddings
- MLP is 2× larger than attention (8D² vs 4D²)

**Quick formula:**
```
TOTAL = 2VD + 12LD²
```

You now know how to count parameters in any transformer! 🎉

---

## 📚 References

- Code: `nanochat/gpt.py` - The actual implementation
- Docs: `PARAMETER_GUIDE.md` - Official guide
- Model: `scripts/base_train.py` - Training configuration

---

**Next steps:**
- Try calculating parameters for different model sizes
- Modify the architecture (MQA, GQA, different MLP expansion)
- Understand memory requirements for training vs inference
- Experiment with the code!
