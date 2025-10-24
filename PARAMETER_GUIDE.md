# 🎓 Complete Parameter Calculation Guide for nanochat

## 📖 Quick Summary

For the **d20 speedrun model**:

```
Total: 561M parameters
├─ Token Embedding (wte):     84M  (15%)
├─ 20 × Transformer Blocks:  393M  (70%)
│   ├─ Attention per block:  6.6M
│   └─ MLP per block:       13.1M
└─ LM Head (unembedding):     84M  (15%)
```

---

## 🧮 The Master Formula

For any nanochat model with depth `d`, vocab `V`, model_dim `D`:

```python
# Dimensions
model_dim = d × 64
num_heads = ceil(model_dim / 128)
vocab_size = 65,536  # 2^16

# Components
token_embedding = V × D
per_block = 12 × D²  # (4×D² attention + 8×D² MLP)
all_blocks = d × (12 × D²)
lm_head = D × V

# Total
TOTAL = 2×(V×D) + d×(12×D²)
```

---

## 🔍 Step-by-Step Calculation for d20

### Step 1: Determine model dimensions

```python
depth = 20
model_dim = 20 × 64 = 1,280
num_heads = ceil(1,280 / 128) = 10
head_dim = 1,280 / 10 = 128
vocab_size = 65,536
```

### Step 2: Token Embedding

```python
wte = nn.Embedding(65536, 1280)
params = 65,536 × 1,280 = 83,886,080
```

### Step 3: Single Transformer Block

#### Attention (4 matrices):
```python
# Q, K, V, Output projections
each_projection = 1,280 × 1,280 = 1,638,400
total_attention = 4 × 1,638,400 = 6,553,600
```

**Why 4 matrices?**
- Q: `Linear(1280 → 1280)` = 1,638,400
- K: `Linear(1280 → 1280)` = 1,638,400
- V: `Linear(1280 → 1280)` = 1,638,400
- Output: `Linear(1280 → 1280)` = 1,638,400

#### MLP (2 matrices with 4× expansion):
```python
# First: expand 4×
first = 1,280 × (4 × 1,280) = 1,280 × 5,120 = 6,553,600

# Activation: 0 params

# Second: project back
second = (4 × 1,280) × 1,280 = 5,120 × 1,280 = 6,553,600

total_mlp = 6,553,600 + 6,553,600 = 13,107,200
```

#### RMSNorm: 0 params (functional, no learnable weights)

#### Total per block:
```python
block = attention + mlp + rmsnorm
      = 6,553,600 + 13,107,200 + 0
      = 19,660,800
```

### Step 4: All 20 blocks

```python
all_blocks = 20 × 19,660,800 = 393,216,000
```

### Step 5: LM Head

```python
lm_head = nn.Linear(1280, 65536)
params = 1,280 × 65,536 = 83,886,080
```

### Step 6: Total

```python
TOTAL = token_emb + all_blocks + lm_head
      = 83,886,080 + 393,216,000 + 83,886,080
      = 560,988,160
      ≈ 561M
```

---

## 💡 Key Insights

### 1. **MLP dominates each block** (2:1 ratio)
```
Attention: 4×d²  = 4 × 1,280² = 6.6M
MLP:       8×d²  = 8 × 1,280² = 13.1M
Ratio:     1:2
```

Why? MLP expands 4× then projects back = `2 × (d × 4d) = 8d²`

### 2. **Embeddings are symmetric** (but untied!)
```
Token embedding = vocab × d_model = 84M
LM head         = d_model × vocab = 84M
```

They're the same size (transposes), but **separate weights**.

### 3. **Layers scale quadratically, embeddings linearly**
```
Double d_model:  4× more params per layer! (d² → (2d)² = 4d²)
Double vocab:    2× more embedding params  (v → 2v)
```

This is why wide models are expensive!

### 4. **No "hidden" parameters**

These have **0 learnable parameters**:
- RMSNorm (functional)
- RoPE (precomputed lookup)
- Activations (ReLU²)
- Attention mechanism itself (uses QKV projections)

### 5. **Optimization split**

```
Muon optimizer:   393M (70%) - all transformer blocks
AdamW optimizer:  168M (30%) - embeddings + lm_head
```

This is why untied embeddings matter - different optimization!

---

## 📊 Comparison: How Size Scales

| Depth | d_model | Block | Layers | Emb+LM | Total | Cost |
|-------|---------|-------|--------|--------|-------|------|
| d4    | 256     | 0.8M  | 3M     | 34M    | 37M   | <$1  |
| d8    | 512     | 3.1M  | 25M    | 67M    | 92M   | ~$5  |
| d12   | 768     | 7.1M  | 85M    | 101M   | 186M  | ~$30 |
| **d20**| **1280** | **19.7M** | **393M** | **168M** | **561M** | **~$100** |
| d26   | 1664    | 33.2M | 863M   | 218M   | 1.08B | ~$300|
| d32   | 2048    | 50.3M | 1.61B  | 268M   | 1.88B | ~$800|

**Observation:** 
- d4 → d8: 2× depth, 4× d_model → 2.5× params
- d20 → d26: 1.3× depth, 1.3× d_model → 1.9× params

It's more efficient to go deeper than wider!

---

## 🎯 Practice Examples

### Example 1: What if we used MQA?

**Question:** d20 model with `n_kv_head=1` (Multi-Query Attention)

**Answer:**
```python
# Attention changes:
Q: 1,280 × 1,280 = 1,638,400  (unchanged)
K: 1,280 × 128   = 163,840    (10× smaller!)
V: 1,280 × 128   = 163,840    (10× smaller!)
Output: 1,280 × 1,280 = 1,638,400  (unchanged)

New attention = 1,638,400 + 163,840 + 163,840 + 1,638,400
              = 3,604,480  (was 6,553,600)

Saved per block = 6.6M - 3.6M = 3M
Saved total = 20 × 3M = 60M

New total = 561M - 60M = 501M

Reduction: 10.7%
```

**Takeaway:** MQA saves 10% params, but HUGE KV cache savings for inference!

---

### Example 2: What if vocab was 256K (like GPT-4)?

**Question:** Same d20, but `vocab_size=256K`

**Answer:**
```python
Old embeddings = 2 × (65K × 1,280) = 168M
New embeddings = 2 × (256K × 1,280) = 655M

Increase = 655M - 168M = 487M

New total = 561M + 487M = 1,048M ≈ 1.05B

Parameters increased by 87%!
```

**Takeaway:** Large vocab is expensive! That's why 65K is sweet spot.

---

### Example 3: Calculate params for d16

**Question:** What's the parameter count for d16?

**Solution:**
```python
depth = 16
model_dim = 16 × 64 = 1,024
num_heads = ceil(1,024 / 128) = 8
vocab = 65,536

# Embeddings
emb = 2 × (65,536 × 1,024) = 134,217,728

# Per block
block = 12 × 1,024² = 12 × 1,048,576 = 12,582,912

# All blocks
blocks = 16 × 12,582,912 = 201,326,592

# Total
total = 134,217,728 + 201,326,592 = 335,544,320
      ≈ 336M params
```

---

## 🧪 DIY Calculator

Try this Python function:

```python
def count_params(depth, vocab=65536):
    """Calculate nanochat parameters for any depth."""
    d = depth * 64  # model_dim
    emb = 2 * vocab * d
    blocks = depth * 12 * d * d
    return emb + blocks

# Examples
print(f"d10:  {count_params(10)/1e6:.0f}M")
print(f"d15:  {count_params(15)/1e6:.0f}M")
print(f"d20:  {count_params(20)/1e6:.0f}M")  # 561M
print(f"d25:  {count_params(25)/1e6:.0f}M")
print(f"d30:  {count_params(30)/1e6:.0f}M")
```

---

## 🎓 Why This Matters

### For Understanding:
- Know where parameters live (70% in transformers!)
- Understand memory requirements (param count × bytes per param)
- Debug shape mismatches in code

### For Research:
- Design efficient architectures (MQA, smaller vocab, etc.)
- Calculate compute budgets (FLOPs ∝ 6 × params × tokens)
- Apply scaling laws (bigger vs wider vs more data)

### For Implementation:
- Verify model loaded correctly
- Estimate training time/cost
- Plan inference optimization (quantization targets)

---

## 📚 References

Scripts to check out:
1. `calculate_params.py` - Detailed step-by-step calculation
2. `visualize_params.py` - ASCII art diagrams
3. `manual_param_calculation.md` - This guide with examples

Model files:
- `nanochat/gpt.py` - See the actual architecture
- `scripts/base_train.py` - See how dimensions are derived

---

## ✅ Quick Self-Test

**Can you answer these?**

1. How many parameters in the Q projection for d20?
   <details><summary>Answer</summary>1,280 × 1,280 = 1,638,400</details>

2. What's the attention:MLP ratio?
   <details><summary>Answer</summary>1:2 (4×d² vs 8×d²)</details>

3. Why are embeddings 30% of total params?
   <details><summary>Answer</summary>Large vocab (65K) + untied weights doubles embedding cost</details>

4. How many params if we used GQA with 2 KV heads?
   <details><summary>Answer</summary>~535M (saves ~26M from attention)</details>

5. What has 0 parameters?
   <details><summary>Answer</summary>RMSNorm, RoPE, activations, attention mechanism</details>

---

**You now know how to calculate transformer parameters by hand!** 🎉

Try calculating for different configurations and see what you get!
