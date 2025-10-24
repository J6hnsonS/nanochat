# 📐 Transformer Parameter Calculation Formula Guide

## 🎯 The Master Formula

For **any** transformer model, here's the complete formula:

```
TOTAL = Token_Embedding + All_Blocks + LM_Head

Where:
  Token_Embedding = vocab_size × d_model
  All_Blocks = n_layers × (Attention_params + MLP_params)
  LM_Head = d_model × vocab_size  (if untied)
```

---

## 🔍 Breaking Down ONE Transformer Block

### Attention Component

```
Attention has 4 weight matrices:

┌──────────────────────────────────────────────────────────┐
│ 1. Q (Query):   d_model × (n_heads × head_dim)          │
│ 2. K (Key):     d_model × (n_kv_heads × head_dim)       │
│ 3. V (Value):   d_model × (n_kv_heads × head_dim)       │
│ 4. Output:      (n_heads × head_dim) × d_model          │
└──────────────────────────────────────────────────────────┘

For standard Multi-Head Attention (MHA):
  n_kv_heads = n_heads
  n_heads × head_dim = d_model

So each projection is: d_model × d_model

TOTAL = 4 × (d_model × d_model) = 4 × d_model²
```

### MLP Component

```
MLP has 2 weight matrices with expansion_factor (usually 4):

┌──────────────────────────────────────────────────────────┐
│ 1. Expand:   d_model × (expansion × d_model)            │
│ 2. Project:  (expansion × d_model) × d_model            │
└──────────────────────────────────────────────────────────┘

For expansion = 4:

TOTAL = 2 × (d_model × 4 × d_model) = 8 × d_model²
```

### Block Total

```
ONE BLOCK = Attention + MLP
          = 4 × d_model² + 8 × d_model²
          = 12 × d_model²
```

---

## 🧮 Calculation for d20 Model

### Given Configuration
```
vocab_size = 65,536
d_model = 1,280
n_layers = 20
n_heads = 10
head_dim = 128
```

### Step-by-Step Calculation

#### 1️⃣ Token Embedding
```
vocab_size × d_model
= 65,536 × 1,280
= 83,886,080 parameters
```

#### 2️⃣ ONE Attention Layer
```
Q: 1,280 × 1,280 = 1,638,400
K: 1,280 × 1,280 = 1,638,400
V: 1,280 × 1,280 = 1,638,400
O: 1,280 × 1,280 = 1,638,400
─────────────────────────────
Total:           6,553,600
```

Or use the formula:
```
4 × d_model²
= 4 × 1,280²
= 4 × 1,638,400
= 6,553,600 ✓
```

#### 3️⃣ ONE MLP Layer
```
Expand:  1,280 × 5,120 = 6,553,600
Project: 5,120 × 1,280 = 6,553,600
────────────────────────────────
Total:                13,107,200
```

Or use the formula:
```
8 × d_model²
= 8 × 1,280²
= 8 × 1,638,400
= 13,107,200 ✓
```

#### 4️⃣ ONE Complete Block
```
Attention + MLP
= 6,553,600 + 13,107,200
= 19,660,800 parameters
```

Or use the formula:
```
12 × d_model²
= 12 × 1,280²
= 12 × 1,638,400
= 19,660,800 ✓
```

#### 5️⃣ ALL Blocks
```
n_layers × block_params
= 20 × 19,660,800
= 393,216,000 parameters
```

#### 6️⃣ LM Head (Unembedding)
```
d_model × vocab_size
= 1,280 × 65,536
= 83,886,080 parameters
```

#### 7️⃣ GRAND TOTAL
```
Token_Embedding + All_Blocks + LM_Head
= 83,886,080 + 393,216,000 + 83,886,080
= 560,988,160 parameters
≈ 561M parameters
```

---

## 📊 Visual Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│                 NANOCHAT d20 MODEL                          │
│                 Total: 561M Parameters                      │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    ┌───▼───┐         ┌─────▼─────┐      ┌────▼────┐
    │  wte  │         │ 20 Blocks │      │ lm_head │
    │ 83.9M │         │  393.2M   │      │  83.9M  │
    │ 15.0% │         │   70.1%   │      │  15.0%  │
    └───────┘         └─────┬─────┘      └─────────┘
                            │
                    ┌───────▼────────┐
                    │  ONE BLOCK     │
                    │  19.66M params │
                    └───────┬────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
         ┌──────▼──────┐        ┌──────▼──────┐
         │  Attention  │        │     MLP     │
         │   6.55M     │        │   13.11M    │
         │   33.3%     │        │   66.7%     │
         └─────────────┘        └─────────────┘
```

---

## 🔑 Key Insights

### 1. MLP Dominates Each Block (2:1 ratio)
```
MLP params      = 8 × d_model²
Attention params = 4 × d_model²
Ratio = 2:1
```

This is why MLPs account for 66.7% of block parameters!

### 2. Embeddings are Symmetric (but untied)
```
Token embedding size = vocab_size × d_model
LM head size         = d_model × vocab_size

They're transposes, same size, but DIFFERENT weights!
```

### 3. Quadratic Scaling in d_model
```
Double d_model → 4× more parameters per block!

Example:
  d_model = 1,280 → block = 12 × 1,280² = 19.66M
  d_model = 2,560 → block = 12 × 2,560² = 78.64M (4× larger!)
```

### 4. Linear Scaling in Vocabulary
```
Double vocab_size → 2× more embedding parameters

Example:
  vocab = 32K → embeddings = 2 × (32K × 1280) = 81.9M
  vocab = 64K → embeddings = 2 × (64K × 1280) = 163.8M (2× larger)
```

---

## 🧪 Quick Formula for Different Configurations

```python
def count_parameters(vocab_size, d_model, n_layers, expansion=4):
    """Calculate transformer parameters."""
    # Embeddings (untied)
    embeddings = 2 * vocab_size * d_model
    
    # One block = attention + MLP
    attention = 4 * d_model * d_model
    mlp = 2 * expansion * d_model * d_model
    block = attention + mlp
    
    # All blocks
    all_blocks = n_layers * block
    
    return embeddings + all_blocks

# Examples:
print(f"d20:  {count_parameters(65536, 1280, 20) / 1e6:.0f}M")  # 561M
print(f"d12:  {count_parameters(65536, 768, 12) / 1e6:.0f}M")   # 186M
print(f"d32:  {count_parameters(65536, 2048, 32) / 1e6:.0f}M")  # 1,879M
```

---

## 🎓 Common Mistakes to Avoid

### ❌ Mistake 1: Forgetting bias=False
```python
# WRONG (includes bias):
params = in_features * out_features + out_features

# CORRECT (no bias in nanochat):
params = in_features * out_features
```

### ❌ Mistake 2: Counting activation functions
```python
# Activations (ReLU, GELU, etc.) have 0 parameters
# RMSNorm (functional) has 0 parameters
# Only Linear layers and Embeddings have parameters!
```

### ❌ Mistake 3: Assuming tied embeddings
```python
# Many older models tie embeddings:
total = vocab × d_model + blocks  # Only ONE embedding

# Nanochat uses UNTIED embeddings:
total = 2 × (vocab × d_model) + blocks  # TWO separate embeddings
```

### ❌ Mistake 4: Wrong MLP expansion calculation
```python
# WRONG:
mlp_params = d_model × (4 × d_model)  # Only up projection

# CORRECT:
mlp_params = 2 × d_model × (4 × d_model)  # Both up and down
```

---

## 📈 Scaling Comparison

| Model | d_model | Layers | Block Params | Total Params | Memory (fp32) |
|-------|---------|--------|--------------|--------------|---------------|
| d4    | 256     | 4      | 0.79M        | 37M          | 148 MB        |
| d8    | 512     | 8      | 3.15M        | 92M          | 368 MB        |
| d12   | 768     | 12     | 7.08M        | 186M         | 744 MB        |
| d16   | 1024    | 16     | 12.58M       | 336M         | 1.34 GB       |
| **d20** | **1280** | **20** | **19.66M** | **561M** | **2.24 GB** |
| d26   | 1664    | 26     | 33.22M       | 1.08B        | 4.32 GB       |
| d32   | 2048    | 32     | 50.33M       | 1.88B        | 7.52 GB       |

Memory calculation: `params × 4 bytes (fp32)`

---

## 🔬 Verify Your Understanding: Practice Problems

### Problem 1
Calculate parameters for a model with:
- vocab_size = 32,000
- d_model = 512
- n_layers = 6
- expansion = 4

<details>
<summary>Solution</summary>

```
Embeddings: 2 × 32,000 × 512 = 32,768,000
One block: 12 × 512² = 3,145,728
All blocks: 6 × 3,145,728 = 18,874,368
Total: 32,768,000 + 18,874,368 = 51,642,368 ≈ 51.6M
```
</details>

### Problem 2
If d20 used **Multi-Query Attention** (n_kv_heads = 1), how many parameters would it save?

<details>
<summary>Solution</summary>

```
Original K params: 1,280 × 1,280 = 1,638,400
Original V params: 1,280 × 1,280 = 1,638,400

MQA K params: 1,280 × 128 = 163,840  (10× smaller!)
MQA V params: 1,280 × 128 = 163,840  (10× smaller!)

Saved per block: (1,638,400 - 163,840) × 2 = 2,949,120
Saved total: 20 × 2,949,120 = 58,982,400 ≈ 59M

New total: 561M - 59M = 502M (10.5% reduction)
```
</details>

### Problem 3
What percentage of total parameters are in the attention sublayers?

<details>
<summary>Solution</summary>

```
Attention per block: 6,553,600
All attention: 20 × 6,553,600 = 131,072,000
Total model: 560,988,160

Percentage: 131,072,000 / 560,988,160 = 23.4%
```
</details>

---

## 🎯 Summary Formulas

### For Quick Calculations:
```
Given: vocab_size (V), d_model (D), n_layers (L)

Token Embedding:  V × D
One Block:        12 × D²
All Blocks:       L × 12 × D²
LM Head:          D × V

TOTAL = 2(V × D) + L × 12 × D²
```

### Simplified (when V is large):
```
TOTAL ≈ 2VD + 12LD²

For nanochat d20:
= 2(65536)(1280) + 12(20)(1280²)
= 167,772,160 + 393,216,000
= 560,988,160 ✓
```

---

**Congratulations!** You now know how to calculate transformer parameters by hand! 🎉
