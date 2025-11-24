# nanochat Training Pipeline & Optimizer Deep Dive

## Table of Contents
1. [Training Pipeline Overview](#training-pipeline-overview)
2. [Optimizer Architecture: AdamW + Muon](#optimizer-architecture)
3. [AdamW Optimizer Deep Dive](#adamw-optimizer)
4. [Muon Optimizer Deep Dive](#muon-optimizer)
5. [Key Insights & Modern Techniques](#key-insights)
6. [PyTorch Implementation Patterns](#pytorch-patterns)

---

## Training Pipeline Overview

nanochat uses a **hybrid optimizer strategy** that splits model parameters into different groups, each optimized with the most suitable algorithm:

```
Model Parameters
├── Matrix Parameters (Transformer layers)  → Muon Optimizer
├── Embedding Parameters (wte)              → AdamW Optimizer  
└── LM Head Parameters (output projection)  → AdamW Optimizer
```

### Training Flow (from `base_train.py`)

```python
# 1. Model Initialization
model = GPT(config)
model.init_weights()
model = torch.compile(model)  # JIT compilation for speed

# 2. Optimizer Setup (Hybrid Strategy)
optimizers = model.setup_optimizers(
    unembedding_lr=0.004,   # AdamW for lm_head
    embedding_lr=0.2,       # AdamW for embeddings
    matrix_lr=0.02,         # Muon for transformer matrices
    weight_decay=0.0
)
adamw_optimizer, muon_optimizer = optimizers

# 3. Training Loop with Gradient Accumulation
for step in range(num_iterations):
    # Forward/backward with gradient accumulation
    for micro_step in range(grad_accum_steps):
        loss = model(x, y) / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)  # Prefetch next batch
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Learning rate scheduling
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    # Momentum scheduling (Muon specific)
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    
    # Optimizer step
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
```

---

## Optimizer Architecture

### Why Multiple Optimizers?

Different parameter types have different optimization characteristics:

| Parameter Type | Properties | Best Optimizer | Reasoning |
|---------------|------------|----------------|-----------|
| **Transformer Matrices** | 2D, large, well-conditioned | **Muon** | Benefits from orthogonalization; natural 2D structure |
| **Embeddings** | 2D but sparse updates | **AdamW** | Sparse gradient patterns need adaptive learning |
| **LM Head** | Output projection | **AdamW** | Similar to embeddings; needs adaptive rates |

### Key Innovation: Parameter-Specific Optimization

```python
def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, 
                     matrix_lr=0.02, weight_decay=0.0):
    # Separate parameters by type
    matrix_params = list(self.transformer.h.parameters())      # Muon
    embedding_params = list(self.transformer.wte.parameters()) # AdamW
    lm_head_params = list(self.lm_head.parameters())          # AdamW
    
    # Scale LR inversely with sqrt of model dimension
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    
    # AdamW for embeddings and output
    adam_groups = [
        dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
    ]
    adamw_optimizer = DistAdamW(adam_groups, betas=(0.8, 0.95), 
                                 eps=1e-10, weight_decay=weight_decay)
    
    # Muon for transformer matrices
    muon_optimizer = DistMuon(matrix_params, lr=matrix_lr, momentum=0.95)
    
    return [adamw_optimizer, muon_optimizer]
```

---

## AdamW Optimizer

### Mathematical Foundation

AdamW (Adam with decoupled Weight Decay) maintains two moment estimates:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          # First moment (momentum)
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²         # Second moment (variance)

m̂_t = m_t / (1 - β₁ᵗ)                        # Bias correction
v̂_t = v_t / (1 - β₂ᵗ)                        # Bias correction

θ_t = θ_{t-1} · (1 - λ · η)                  # Weight decay (decoupled!)
      - η · m̂_t / (√v̂_t + ε)                # Adam update
```

Where:
- `g_t` = gradient at time t
- `β₁, β₂` = exponential decay rates for moments (0.8, 0.95 in nanochat)
- `η` = learning rate
- `λ` = weight decay coefficient
- `ε` = numerical stability term (1e-10)

### Implementation: DistAdamW (ZeRO-2 Style)

```python
class DistAdamW(torch.optim.Optimizer):
    """Distributed AdamW with sharded optimizer states (ZeRO-2)"""
    
    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Phase 1: Reduce-scatter gradients (average across ranks)
        for group in self.param_groups:
            for param in group["params"]:
                grad = param.grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                
                # Each rank gets 1/world_size of the gradient
                dist.reduce_scatter_tensor(
                    grad_slice, grad, 
                    op=dist.ReduceOp.AVG, 
                    async_op=True
                )
        
        # Phase 2: Each rank updates its slice
        for group in self.param_groups:
            for param in group["params"]:
                p_slice = param[rank*rank_size:(rank+1)*rank_size]
                g_slice = grad_slices[idx]
                
                # Initialize state
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                
                # Weight decay (decoupled - applied directly to params)
                if wd != 0:
                    p_slice.mul_(1 - lr * wd)
                
                # Update moments
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, 
                                                value=1 - beta2)
                
                # Bias correction
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                
                # Compute update
                step_size = lr * (sqrt(bias2) / bias1)
                update = exp_avg / (exp_avg_sq.sqrt() + eps) * step_size
                p_slice.add_(update, alpha=-1.0)
        
        # Phase 3: All-gather to replicate updated parameters
        dist.all_gather_into_tensor(param, p_slice, async_op=True)
```

### Key Features of DistAdamW

1. **ZeRO-2 Optimization**: Shards optimizer states across GPUs
   - Memory savings: ~2x reduction in optimizer memory
   - Each rank stores only 1/N of the optimizer state
   
2. **Decoupled Weight Decay**: 
   - Applied directly to parameters: `p *= (1 - lr * wd)`
   - More effective than L2 regularization for adaptive optimizers
   
3. **Async Communication**:
   - `reduce_scatter` and `all_gather` use `async_op=True`
   - Overlaps communication with computation

4. **@torch.compile**: JIT compilation for ~10-20% speedup

### Advantages of AdamW

✅ **Adaptive learning rates** per parameter  
✅ **Handles sparse gradients** well (embeddings)  
✅ **Robust to hyperparameter choices**  
✅ **Proven track record** in LLM training  

❌ Memory intensive (2x params for moments)  
❌ Not exploiting structure of 2D matrices  

---

## Muon Optimizer

### The Big Idea

**Muon = Momentum + Orthogonalization via Newton-Schulz**

Key insight: For 2D weight matrices in neural networks, updates should preserve the linear transformation's "shape" while changing its direction. Orthogonalization achieves this by projecting updates onto the nearest orthogonal matrix.

### Mathematical Foundation

#### Step 1: Standard SGD with Momentum

```
g_t = ∇L(θ_{t-1})                            # Gradient
m_t = β · m_{t-1} + (1 - β) · g_t           # Momentum buffer
u_t = g_t + β · m_t  (if Nesterov)          # Nesterov acceleration
```

#### Step 2: Orthogonalization via Newton-Schulz

For a matrix U, find the nearest orthogonal matrix V (where V^T V = I):

```python
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Compute orthogonalization of G via quintic Newton-Schulz iteration.
    Returns approximately U·V^T where U·S·V^T is the SVD of G.
    """
    X = G / (||G|| + ε)  # Normalize to spectral norm ≤ 1
    
    # Quintic iteration (5th order convergence)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X^T
        B = b·A + c·A²
        X = a·X + B @ X
    
    return X
```

This computes `G → U·V^T` where `G = U·S·V^T` is the SVD, effectively "removing" the singular values and keeping only the orthogonal components.

#### Step 3: Aspect-Ratio Scaled Step

```
θ_t = θ_{t-1} - η · √max(1, h/w) · NS₅(u_t)
```

The `√(h/w)` factor accounts for rectangular matrices where height ≠ width.

### Implementation: DistMuon

```python
class DistMuon(torch.optim.Optimizer):
    """Distributed Muon with block-cyclic parameter assignment"""
    
    def __init__(self, params, lr=0.02, momentum=0.95, 
                 nesterov=True, ns_steps=5):
        # Group parameters by shape for efficient batching
        shapes = sorted({p.shape for p in params})
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            param_groups.append(dict(
                params=group_params,
                zero_buffer=torch.zeros_like(group_params[0])
            ))
        super().__init__(param_groups, defaults)
    
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Phase 1: Reduce-scatter gradients (block-cyclic assignment)
        for group in self.param_groups:
            params = group["params"]
            for base_i in range(0, len(params), world_size):
                owner_idx = base_i + rank  # This rank owns param[owner_idx]
                
                # Reduce-scatter: average gradients
                rs_input = [p.grad for p in params[base_i:base_i+world_size]]
                rs_output = params[owner_idx].grad
                dist.reduce_scatter(rs_output, rs_input, 
                                   op=dist.ReduceOp.AVG)
        
        # Phase 2: Owner ranks compute Muon update
        for group in self.param_groups:
            for base_i in range(0, len(params), world_size):
                owner_idx = base_i + rank
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # Averaged gradient
                    
                    # Initialize momentum buffer
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    
                    buf = state["momentum_buffer"]
                    
                    # Update momentum: buf = β·buf + (1-β)·g
                    buf.lerp_(g, 1 - momentum)
                    
                    # Nesterov: g = lerp(g, buf, β)
                    if nesterov:
                        g = g.lerp_(buf, momentum)
                    else:
                        g = buf
                    
                    # Orthogonalize via Newton-Schulz
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    
                    # Aspect-ratio scaled update
                    scale = sqrt(max(1, p.size(0) / p.size(1)))
                    p.add_(g, alpha=-lr * scale)
        
        # Phase 3: All-gather updated parameters
        for group in self.param_groups:
            for base_i in range(0, len(params), world_size):
                owner_idx = base_i + rank
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i+world_size]
                dist.all_gather(ag_output, ag_input)
```

### Why Newton-Schulz Works

The Newton-Schulz iteration computes the "zero-power" of a matrix (i.e., projection onto orthogonal matrices):

- **Quintic iteration**: 5th-order convergence → only 5 steps needed
- **Numerically stable** in bfloat16
- **No eigendecomposition** required (unlike SVD)
- **Batched implementation** possible (efficient on GPU)

### Advantages of Muon

✅ **Exploits 2D structure** of weight matrices  
✅ **Memory efficient**: Only stores momentum buffer (1x params)  
✅ **Orthogonalization** = implicit preconditioning  
✅ **Higher learning rates** possible (0.02 vs 0.001 for Adam)  
✅ **Faster convergence** on transformer matrices  
✅ **bfloat16 stable**: NS iteration works in low precision  

❌ Only works for 2D parameters  
❌ More computation per step (NS iterations)  
❌ Less understood than AdamW  

---

## Key Insights & Modern Techniques

### 1. **Hybrid Optimizer Strategy**

nanochat's key innovation is **parameter-type-specific optimization**:

- **Muon for structured matrices**: Exploits 2D geometry
- **AdamW for unstructured params**: Handles sparse updates

This is better than using a single optimizer for everything.

### 2. **Learning Rate Scaling**

```python
# Scale LR with model dimension: η ∝ 1/√d_model
dmodel_lr_scale = (model_dim / 768) ** -0.5
```

**Rationale**: Larger models need smaller learning rates to maintain stability. This follows the **µP (Maximal Update Parameterization)** principle.

### 3. **Momentum Scheduling**

```python
def get_muon_momentum(step):
    # Warm up momentum from 0.85 → 0.95 over 300 steps
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
```

**Rationale**: Start with lower momentum for exploration, increase for stability.

### 4. **Learning Rate Schedule**

```python
def get_lr_multiplier(step):
    # Warmup (0%) → Constant (80%) → Cosine decay (20%)
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    elif step <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac
```

**Modern pattern**: 
- No warmup (0% vs typical 5%)
- 80% constant LR (vs typical 0%)
- 20% cosine decay

### 5. **Gradient Accumulation**

```python
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

for micro_step in range(grad_accum_steps):
    loss = model(x, y) / grad_accum_steps  # Scale loss!
    loss.backward()
```

**Key**: Divide loss by accumulation steps so gradients have correct scale.

### 6. **Mixed Precision Training**

```python
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

with autocast_ctx:
    loss = model(x, y)
```

**bfloat16 advantages**:
- 2x memory savings
- 2x compute speedup (on Ampere+ GPUs)
- Better numeric range than fp16
- No loss scaling needed

### 7. **ZeRO-2 Distributed Training**

Both optimizers implement ZeRO-2 (Zero Redundancy Optimizer, Stage 2):

```
Traditional DDP: Each GPU stores full model + full optimizer states
ZeRO-2: Each GPU stores full model + 1/N optimizer states

Memory savings: ~2x reduction in optimizer memory
Communication: reduce_scatter (gradients) + all_gather (params)
```

### 8. **Efficient Data Loading**

```python
# CPU → GPU transfer optimizations
scratch = torch.tensor(tokens, pin_memory=True)  # Pinned memory
inputs = inputs_cpu.to(device, non_blocking=True)  # Async transfer
```

### 9. **Model Compilation**

```python
model = torch.compile(model, dynamic=False)
```

**Benefits**:
- 10-30% speedup via JIT compilation
- Kernel fusion (reduce memory bandwidth)
- `dynamic=False` = fixed input shapes → more optimization

### 10. **@torch.compile on Optimizer**

```python
@torch.compile
@torch.no_grad()
def step(self):
    # optimizer logic
```

Even optimizers can be compiled! Reduces Python overhead.

---

## PyTorch Implementation Patterns

### Pattern 1: Efficient In-Place Operations

```python
# ❌ Slow: Creates new tensors
exp_avg = beta1 * exp_avg + (1 - beta1) * grad

# ✅ Fast: In-place operations
exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
```

### Pattern 2: Async Distributed Communication

```python
# ❌ Blocking
dist.reduce_scatter_tensor(output, input)
dist.all_gather_into_tensor(output, input)

# ✅ Non-blocking (overlap with compute)
future1 = dist.reduce_scatter_tensor(..., async_op=True).get_future()
future2 = dist.all_gather_into_tensor(..., async_op=True).get_future()
# ... do computation ...
future1.wait()
future2.wait()
```

### Pattern 3: Memory-Efficient Gradient Accumulation

```python
# ✅ Correct scaling
for micro_step in range(grad_accum_steps):
    loss = model(x, y) / grad_accum_steps  # Scale BEFORE backward!
    loss.backward()

# ❌ Wrong: Would need to scale gradients manually
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    loss.backward()
```

### Pattern 4: Gradient Clipping

```python
# Clip gradient norms to prevent instability
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Pattern 5: Zero Gradients Efficiently

```python
# ✅ Faster: Frees memory immediately
model.zero_grad(set_to_none=True)

# ❌ Slower: Keeps tensors, fills with zeros
model.zero_grad()
```

---

## Comparison: AdamW vs Muon

| Aspect | AdamW | Muon |
|--------|-------|------|
| **Memory** | 2x params (m, v) | 1x params (momentum) |
| **Computation** | Low (elementwise ops) | Medium (matrix mult in NS) |
| **Learning Rate** | ~0.001-0.01 | ~0.01-0.1 (10x higher!) |
| **Convergence** | Stable, proven | Faster on transformers |
| **Applicability** | Any parameter | Only 2D matrices |
| **Adaptivity** | Per-parameter | Per-matrix (structured) |
| **Theory** | Well-understood | Newer (2024) |

---

## Summary: Why This Design?

nanochat's training pipeline embodies several modern best practices:

1. **Hybrid optimization**: Match optimizer to parameter structure
2. **Distributed efficiency**: ZeRO-2 for memory, async ops for speed
3. **Mixed precision**: bfloat16 for 2x speedup with numeric stability
4. **Compilation**: JIT compile model + optimizers
5. **Orthogonalization**: Novel technique (Muon) for structured parameters
6. **Careful scheduling**: LR, momentum, warmup/decay
7. **Memory optimization**: In-place ops, gradient accumulation, zero_grad(set_to_none=True)

The result: **Train a 2B parameter LLM for $100 in 4 hours** 🚀

---

## Further Reading

- **Muon Optimizer**: https://kellerjordan.github.io/posts/muon/
- **AdamW Paper**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
- **ZeRO Paper**: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020)
- **Newton-Schulz**: "Functions of Matrices: Theory and Computation" (Higham, 2008)
- **µP (Maximal Update Parameterization)**: "Tensor Programs V" (Yang & Hu, 2021)

---

*This guide was created for learning purposes based on the nanochat codebase.*
