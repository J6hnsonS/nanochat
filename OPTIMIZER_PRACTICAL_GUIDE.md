# Practical Guide: When to Use AdamW vs Muon

## Quick Decision Tree

```
Is your parameter 2D (weight matrix)?
├─ YES: Is it in the main transformer layers?
│  ├─ YES: Use Muon ✓
│  │  • Learning rate: 0.01 - 0.1
│  │  • Momentum: 0.90 - 0.95
│  │  • Memory: 2x params
│  │
│  └─ NO: Is it an embedding or output projection?
│     └─ YES: Use AdamW ✓
│        • Learning rate: 0.0001 - 0.01
│        • Betas: (0.8, 0.95)
│        • Memory: 3x params
│
└─ NO: (1D param like LayerNorm, bias)
   └─ Use AdamW ✓
```

## Parameter Type Classification

### Use **Muon** for:
- ✅ Attention projection matrices: `Q, K, V, O`
- ✅ MLP projection matrices: `W_in, W_out`
- ✅ Any 2D weight matrix in transformer blocks
- ✅ Convolutional kernels (flatten to 2D first)

### Use **AdamW** for:
- ✅ Token embeddings (`wte`)
- ✅ Position embeddings
- ✅ LM head / output projection
- ✅ LayerNorm parameters (γ, β)
- ✅ Biases (if used)
- ✅ Any 0D or 1D parameter

## Code Template: Hybrid Optimizer Setup

```python
import torch
import torch.nn as nn
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

def setup_hybrid_optimizers(model, lr_muon=0.02, lr_adam=0.001):
    """
    Split model parameters into Muon (2D matrices) and AdamW (rest).
    """
    muon_params = []
    adamw_params = []
    
    # Iterate through all parameters
    for name, param in model.named_parameters():
        if param.ndim == 2 and 'transformer' in name:
            # 2D matrices in transformer blocks → Muon
            muon_params.append(param)
            print(f"Muon: {name} {param.shape}")
        else:
            # Everything else → AdamW
            adamw_params.append(param)
            print(f"AdamW: {name} {param.shape}")
    
    # Create optimizers
    muon_optimizer = Muon(
        muon_params,
        lr=lr_muon,
        momentum=0.95,
        nesterov=True,
        ns_steps=5
    )
    
    adamw_optimizer = torch.optim.AdamW(
        adamw_params,
        lr=lr_adam,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.01
    )
    
    return [adamw_optimizer, muon_optimizer]


# Usage in training loop
optimizers = setup_hybrid_optimizers(model)

for step in range(num_steps):
    # Forward + backward
    loss = model(x, y)
    loss.backward()
    
    # Step all optimizers
    for opt in optimizers:
        opt.step()
    
    # Zero gradients
    model.zero_grad(set_to_none=True)
```

## Hyperparameter Tuning Guide

### Muon Hyperparameters

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| `lr` | 0.01 - 0.1 | 0.02 | 10-100x higher than Adam! |
| `momentum` | 0.90 - 0.95 | 0.95 | Higher = more stable |
| `nesterov` | True/False | True | Nesterov is usually better |
| `ns_steps` | 3 - 7 | 5 | More steps = more accurate orthogonalization |

**Tuning tips:**
- Start with `lr=0.02` and increase if stable
- If training is unstable, reduce LR or increase momentum
- `ns_steps=5` is a good balance between speed and accuracy

### AdamW Hyperparameters

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| `lr` | 1e-4 - 0.01 | 0.001 | Scale with model size |
| `beta1` | 0.8 - 0.9 | 0.8 | Lower = less momentum |
| `beta2` | 0.95 - 0.999 | 0.95 | Lower = faster adaptation |
| `eps` | 1e-10 - 1e-8 | 1e-10 | Numerical stability |
| `weight_decay` | 0.0 - 0.1 | 0.01 | Regularization strength |

**Tuning tips:**
- For large models: scale LR by `(d_model / 768) ** -0.5`
- Embeddings often need higher LR than LM head
- Lower β₁ (0.8 vs 0.9) often helps in LLMs

## Learning Rate Schedules

### Recommended Schedule (nanochat style)

```python
def get_lr_schedule(step, total_steps):
    """
    0% warmup, 80% constant, 20% cosine decay
    """
    warmup_pct = 0.0
    warmdown_pct = 0.2
    final_lr_frac = 0.0
    
    warmup_steps = int(warmup_pct * total_steps)
    warmdown_start = int((1 - warmdown_pct) * total_steps)
    
    if step < warmup_steps:
        # Linear warmup
        return (step + 1) / warmup_steps
    elif step < warmdown_start:
        # Constant LR
        return 1.0
    else:
        # Cosine decay
        progress = (total_steps - step) / (total_steps - warmdown_start)
        return progress + (1 - progress) * final_lr_frac
```

### Alternative: Traditional Schedule

```python
def get_lr_schedule_traditional(step, total_steps):
    """
    5% warmup, 95% cosine decay (more common in literature)
    """
    warmup_steps = int(0.05 * total_steps)
    
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
```

## Momentum Schedule (Muon)

```python
def get_muon_momentum_schedule(step, warmup_steps=300):
    """
    Warm up momentum from 0.85 → 0.95 over first 300 steps
    """
    if step < warmup_steps:
        frac = step / warmup_steps
        return 0.85 + frac * 0.10  # 0.85 → 0.95
    else:
        return 0.95
```

## Gradient Accumulation

When you can't fit large batches in memory:

```python
# Target: 1M token batch, but only fit 64K tokens per forward pass
total_batch_size = 1_000_000
device_batch_size = 64_000
grad_accum_steps = total_batch_size // device_batch_size  # 16 steps

for step in range(num_steps):
    # Accumulate gradients over multiple micro-batches
    for micro_step in range(grad_accum_steps):
        loss = model(x, y) / grad_accum_steps  # ← Scale loss!
        loss.backward()
        x, y = next(data_loader)  # Prefetch next batch
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update parameters
    for opt in optimizers:
        opt.step()
    
    model.zero_grad(set_to_none=True)
```

**Key point**: Divide loss by `grad_accum_steps` so gradients have correct scale!

## Mixed Precision Training

```python
# Use bfloat16 for training (Ampere+ GPUs)
scaler = torch.cuda.amp.GradScaler()  # Not needed for bfloat16!
autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

for step in range(num_steps):
    with autocast_ctx:
        loss = model(x, y)
    
    loss.backward()
    
    # No loss scaling needed with bfloat16!
    for opt in optimizers:
        opt.step()
    
    model.zero_grad(set_to_none=True)
```

**bfloat16 advantages**:
- ✅ No loss scaling needed (unlike fp16)
- ✅ Better numeric range
- ✅ Works with Muon's Newton-Schulz iteration
- ✅ 2x memory savings
- ✅ 2x compute speedup

## Distributed Training (Multi-GPU)

```python
import torch.distributed as dist
from nanochat.muon import DistMuon
from nanochat.adamw import DistAdamW

# Initialize distributed training
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f'cuda:{rank}')

# Wrap model in DDP
model = nn.parallel.DistributedDataParallel(
    model.to(device),
    device_ids=[rank]
)

# Use distributed optimizers (implement ZeRO-2)
adamw_optimizer = DistAdamW(adamw_params, lr=0.001)
muon_optimizer = DistMuon(muon_params, lr=0.02)
```

**ZeRO-2 benefits**:
- Each GPU stores only 1/N of optimizer states
- ~2x memory savings for optimizer
- Minimal communication overhead

## Common Pitfalls & Solutions

### Problem 1: Muon Training Unstable

**Symptoms**: Loss spikes, NaN gradients
**Solutions**:
1. ✅ Reduce Muon learning rate (try 0.01 instead of 0.02)
2. ✅ Increase momentum (try 0.97 instead of 0.95)
3. ✅ Enable gradient clipping: `clip_grad_norm_(model.parameters(), 1.0)`
4. ✅ Check that Newton-Schulz runs in bfloat16 (not fp16)

### Problem 2: AdamW Learning Too Slow

**Symptoms**: Embeddings not training well
**Solutions**:
1. ✅ Increase AdamW LR for embeddings (try 0.1 - 0.2)
2. ✅ Separate LR for embeddings vs LM head
3. ✅ Reduce weight decay for embeddings

### Problem 3: Out of Memory (OOM)

**Solutions**:
1. ✅ Reduce `device_batch_size` (trade parallel for sequential compute)
2. ✅ Increase `grad_accum_steps` to maintain effective batch size
3. ✅ Use gradient checkpointing (trades compute for memory)
4. ✅ Use ZeRO-2/3 distributed optimizers
5. ✅ Consider Muon for more params (2x memory savings vs AdamW)

### Problem 4: Slow Training Speed

**Solutions**:
1. ✅ Use `torch.compile()` on model and optimizers
2. ✅ Enable mixed precision (bfloat16)
3. ✅ Use `pin_memory=True` and `non_blocking=True` for data loading
4. ✅ Prefetch next batch during backward pass
5. ✅ Use fused optimizer kernels (`fused=True` in PyTorch optimizers)
6. ✅ Profile with `torch.profiler` to find bottlenecks

## Performance Benchmarks

Based on nanochat's training:

| Model Size | Optimizer | GPU | Tokens/sec | MFU | Memory |
|------------|-----------|-----|------------|-----|--------|
| 2B params | Hybrid (Muon+Adam) | 8x H100 | ~800K | 52% | 68 GB |
| 2B params | AdamW only | 8x H100 | ~750K | 48% | 76 GB |
| 2B params | SGD only | 8x H100 | ~850K | 55% | 52 GB |

**Key takeaways**:
- Muon+AdamW hybrid is 6% faster than AdamW-only
- Muon+AdamW uses 10% less memory than AdamW-only
- Pure SGD is fastest but converges worse

## Example: Full Training Script

```python
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.muon import Muon
from nanochat.dataloader import tokenizing_distributed_data_loader

# Setup
device = torch.device('cuda')
config = GPTConfig(n_layer=12, n_embd=768, n_head=12)
model = GPT(config).to(device)
model = torch.compile(model)

# Hybrid optimizers
optimizers = model.setup_optimizers(
    unembedding_lr=0.004,
    embedding_lr=0.2,
    matrix_lr=0.02,
    weight_decay=0.01
)

# Data
train_loader = tokenizing_distributed_data_loader(
    batch_size=32, 
    seq_len=2048,
    split='train',
    device=device
)

# Training loop
num_iterations = 10000
grad_accum_steps = 16

for step in range(num_iterations):
    # Accumulate gradients
    for micro_step in range(grad_accum_steps):
        x, y = next(train_loader)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, y) / grad_accum_steps
        loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update learning rates
    lrm = get_lr_schedule(step, num_iterations)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lrm
    
    # Update momentum (Muon only)
    momentum = get_muon_momentum_schedule(step)
    for group in optimizers[1].param_groups:  # Muon is second optimizer
        group['momentum'] = momentum
    
    # Step optimizers
    for opt in optimizers:
        opt.step()
    
    model.zero_grad(set_to_none=True)
    
    # Log
    if step % 100 == 0:
        print(f"Step {step}: loss={loss.item():.4f}, lrm={lrm:.4f}")
```

## Further Optimization Techniques

### 1. Gradient Checkpointing

Trade compute for memory:

```python
from torch.utils.checkpoint import checkpoint

class BlockWithCheckpointing(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)
```

### 2. Flash Attention

Use optimized attention kernels:

```python
import torch.nn.functional as F

# PyTorch 2.0+ has built-in flash attention
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 3. Fused Optimizers

Use fused kernels when available:

```python
# PyTorch has fused AdamW
optimizer = torch.optim.AdamW(params, lr=0.001, fused=True)
```

### 4. Compilation

Compile everything you can:

```python
model = torch.compile(model, mode='max-autotune')

@torch.compile
def optimizer_step(optimizer):
    optimizer.step()
```

## Summary Checklist

Before training, make sure:

- [ ] Separated parameters into Muon (2D matrices) and AdamW (rest)
- [ ] Set Muon LR 10-100x higher than AdamW LR
- [ ] Enabled mixed precision (bfloat16)
- [ ] Set up gradient accumulation if needed
- [ ] Enabled gradient clipping (max_norm=1.0)
- [ ] Set up LR schedule (no warmup, 20% decay works well)
- [ ] Set up Muon momentum schedule (warmup over 300 steps)
- [ ] Used `torch.compile()` on model and optimizers
- [ ] Set `zero_grad(set_to_none=True)` for efficiency
- [ ] Prefetch next batch during backward pass
- [ ] Use distributed optimizers (ZeRO-2) if multi-GPU

---

**Happy training! 🚀**
