# 🧮 Manual Parameter Calculation Guide

A **complete guide** to calculating transformer parameters by hand!

---

## 🎯 The Golden Rule

For any `nn.Linear(in_features, out_features, bias=False)`:
```
Parameters = in_features × out_features
```

For any `nn.Embedding(num_embeddings, embedding_dim)`:
```
Parameters = num_embeddings × embedding_dim
```

**That's it!** Everything else builds on this.

---

## 📐 Step-by-Step: nanochat d20 Model

### Given Information
```
depth = 20
model_dim = depth × 64 = 20 × 64 = 1,280
num_heads = ceil(1280 / 128) = 10
head_dim = 1280 / 10 = 128
vocab_size = 65,536 (2^16)
```

---

## 1️⃣ Token Embedding (wte)

```python
nn.Embedding(vocab_size=65536, embedding_dim=1280)
```

**Calculation:**
```
Parameters = 65,536 × 1,280
           = 83,886,080
           ≈ 84M
```

**Mental shortcut:**
```
65K × 1.3K ≈ 84M
```

---

## 2️⃣ Single Transformer Block

### 2a. Attention Layer

The attention has **4 weight matrices**:

#### Q (Query) Projection
```python
nn.Linear(in=1280, out=10×128=1280, bias=False)
```
```
Params = 1,280 × 1,280 = 1,638,400
```

#### K (Key) Projection
```python
nn.Linear(in=1280, out=10×128=1280, bias=False)
```
```
Params = 1,280 × 1,280 = 1,638,400
```

#### V (Value) Projection
```python
nn.Linear(in=1280, out=10×128=1280, bias=False)
```
```
Params = 1,280 × 1,280 = 1,638,400
```

#### Output Projection
```python
nn.Linear(in=1280, out=1280, bias=False)
```
```
Params = 1,280 × 1,280 = 1,638,400
```

**Total Attention:**
```
= 4 × (1,280 × 1,280)
= 4 × 1,638,400
= 6,553,600
≈ 6.6M per block
```

**Mental formula for attention:**
```
If using MHA (n_head = n_kv_head):
  Attention params = 4 × d_model²

If using MQA (n_kv_head = 1):
  Attention params = d_model × (2×d_model + 2×head_dim)
```

---

### 2b. MLP Layer

The MLP has **2 weight matrices** with 4× expansion:

#### First Linear (expansion)
```python
nn.Linear(in=1280, out=4×1280=5120, bias=False)
```
```
Params = 1,280 × 5,120
       = 6,553,600
```

#### Activation (ReLU²)
```
Params = 0  (activations have no parameters!)
```

#### Second Linear (projection back)
```python
nn.Linear(in=5120, out=1280, bias=False)
```
```
Params = 5,120 × 1,280
       = 6,553,600
```

**Total MLP:**
```
= (1,280 × 5,120) + (5,120 × 1,280)
= 2 × (1,280 × 5,120)
= 2 × 6,553,600
= 13,107,200
≈ 13.1M per block
```

**Mental formula for MLP:**
```
MLP params = 2 × (d_model × expansion × d_model)
           = 2 × d_model × (expansion × d_model)
           = 2 × 1,280 × 5,120
           
With expansion=4:
  MLP params = 8 × d_model²
```

---

### 2c. RMSNorm

```python
# Functional normalization - no parameters!
F.rms_norm(x, (x.size(-1),))
```
```
Params = 0
```

**Why?** No learnable γ (scale) or β (shift) parameters.

---

### Total per Block

```
Single block = Attention + MLP + RMSNorm
             = 6,553,600 + 13,107,200 + 0
             = 19,660,800
             ≈ 19.7M per block
```

**Mental formula:**
```
Block params = 4×d² + 8×d² = 12×d²
             = 12 × 1,280²
             = 12 × 1,638,400
             = 19,660,800
```

---

## 3️⃣ All Transformer Layers

```
Total layers = 20 blocks × 19,660,800 params/block
             = 393,216,000
             ≈ 393M
```

**Mental shortcut:**
```
20 × 20M ≈ 400M
```

---

## 4️⃣ LM Head (Unembedding)

```python
nn.Linear(in=1280, out=65536, bias=False)
```

```
Params = 1,280 × 65,536
       = 83,886,080
       ≈ 84M
```

**Note:** This is the SAME size as the token embedding!
```
wte params = vocab × d_model
lm_head params = d_model × vocab
```
They're transposes of each other (but untied = separate weights).

---

## 5️⃣ Total Model

```
TOTAL = Token Embedding + All Layers + LM Head
      = 83,886,080 + 393,216,000 + 83,886,080
      = 560,988,160
      ≈ 561M parameters
      ≈ 0.56B parameters
```

---

## 🧠 Mental Math Shortcuts

### Quick Approximation Formula
```
For d_model = D, num_layers = L, vocab_size = V:

Total ≈ 2×(V×D) + L×(12×D²)
      ≈ embeddings + layers

For d20:
  ≈ 2×(65K×1.3K) + 20×(12×1.3K²)
  ≈ 2×84M + 20×20M
  ≈ 168M + 400M
  ≈ 568M ✓
```

### What Dominates?

**For small vocab (like 32K):**
```
Layers dominate: 12×D²×L >> 2×V×D
```

**For large vocab (like 256K):**
```
Embeddings can compete with layers!
```

**For d20 (vocab=65K):**
```
Embeddings: 168M (30%)
Layers:     393M (70%)
```

---

## 📊 Component Breakdown

```
┌─────────────────────────────────────────────┐
│ Token Embedding (wte)        84M  (15.0%)  │
├─────────────────────────────────────────────┤
│ Transformer Layers          393M  (70.1%)  │
│   ├─ Attention (per block)  6.6M           │
│   │   ├─ Q projection       1.6M           │
│   │   ├─ K projection       1.6M           │
│   │   ├─ V projection       1.6M           │
│   │   └─ Output proj        1.6M           │
│   │                                         │
│   └─ MLP (per block)       13.1M           │
│       ├─ Expand            6.6M            │
│       └─ Project           6.6M            │
│                                             │
│   20 blocks × 19.7M = 393M                 │
├─────────────────────────────────────────────┤
│ LM Head (unembedding)        84M  (15.0%)  │
├─────────────────────────────────────────────┤
│ TOTAL                       561M           │
└─────────────────────────────────────────────┘
```

---

## 🔢 Compare Different Sizes

| Model | Layers | d_model | Parameters | Formula |
|-------|--------|---------|------------|---------|
| d4    | 4      | 256     | ~36M       | 2×(65K×256) + 4×(12×256²) |
| d12   | 12     | 768     | ~203M      | 2×(65K×768) + 12×(12×768²) |
| d20   | 20     | 1,280   | **561M**   | 2×(65K×1280) + 20×(12×1280²) |
| d26   | 26     | 1,664   | ~1.05B     | 2×(65K×1664) + 26×(12×1664²) |
| d32   | 32     | 2,048   | ~1.9B      | 2×(65K×2048) + 32×(12×2048²) |

---

## 🎓 Practice Problems

### Problem 1: d8 Model
```
depth = 8
model_dim = 8 × 64 = 512
num_heads = ceil(512/128) = 4
vocab = 65,536

Calculate total parameters!
```

<details>
<summary>Solution</summary>

```
Token embedding = 65,536 × 512 = 33,554,432

Per block:
  Attention = 4 × 512² = 1,048,576
  MLP = 8 × 512² = 2,097,152
  Total = 12 × 512² = 3,145,728

All layers = 8 × 3,145,728 = 25,165,824

LM head = 512 × 65,536 = 33,554,432

TOTAL = 33.6M + 25.2M + 33.6M = 92.3M
```
</details>

---

### Problem 2: With MQA
```
Same d20, but with Multi-Query Attention:
  n_head = 10
  n_kv_head = 1  ← Only 1 set of K,V for all Q heads!

How many attention params now?
```

<details>
<summary>Solution</summary>

```
Q projection: 1,280 × 1,280 = 1,638,400 (unchanged)
K projection: 1,280 × 128 = 163,840 (much smaller!)
V projection: 1,280 × 128 = 163,840 (much smaller!)
Output proj: 1,280 × 1,280 = 1,638,400 (unchanged)

Total attention = 1,638,400 + 163,840 + 163,840 + 1,638,400
                = 3,604,480
                ≈ 3.6M (was 6.6M with MHA!)

Saved ~45% of attention parameters!
```
</details>

---

## 🚀 Key Insights

1. **MLP is 2× the attention params** (with 4× expansion)
   ```
   Attention = 4×d²
   MLP = 8×d²
   ```

2. **Embeddings scale linearly with vocab**
   - Double vocab → double embedding params
   - Double depth → embedding params unchanged!

3. **Layers scale quadratically with d_model**
   - Double d_model → 4× more params per layer!
   - This is why width is expensive

4. **MQA/GQA saves inference memory, not training params much**
   - Main benefit: smaller KV cache during generation
   - Training param reduction: only ~25% of attention params

5. **No bias = ~d_model fewer params per linear layer**
   - For d20: saves ~1,280 params × (4 attn + 2 mlp) × 20 layers
   - = ~150K params (negligible)
   - But conceptually simpler!

---

## 💡 Why This Matters

**For Research:**
- Understand scaling laws (compute vs parameters)
- Design architecture variants (GQA ratios, etc.)
- Estimate memory requirements

**For Implementation:**
- Debug shape mismatches
- Verify model loaded correctly
- Calculate memory usage (params × bytes_per_param)

**For Optimization:**
- Know where parameters are (MLP-heavy!)
- Understand what to prune/quantize
- Calculate FLOPS from parameters

---

## 🎯 Next Steps

Try calculating yourself:
1. Different depths (d=4, 8, 16, 32)
2. Different vocab sizes (32K, 128K, 256K)
3. With GQA (n_kv_head = 2, 4)
4. Different MLP expansion (2×, 8×)

The formula is always:
```
Total ≈ 2×(vocab×d_model) + num_layers×(4+2×expansion)×d_model²
```

Master this, and you'll understand any transformer architecture! 🎓
