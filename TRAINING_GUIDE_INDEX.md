# 📚 nanochat Training Pipeline & Optimizer Study Guide

Welcome! This collection of documents will help you master nanochat's training pipeline and understand modern optimizer techniques.

## 📖 Reading Order

### For Beginners: Start Here

1. **[LEARNING_SUMMARY.md](LEARNING_SUMMARY.md)** ⭐ START HERE
   - Overview of key concepts
   - High-level understanding of the training pipeline
   - Core insights and takeaways
   - ~15 min read

2. **[TRAINING_PIPELINE_EXPLAINED.md](TRAINING_PIPELINE_EXPLAINED.md)** 
   - Detailed technical deep dive
   - Mathematical foundations of AdamW and Muon
   - Implementation analysis with code
   - Modern techniques (ZeRO-2, mixed precision, etc.)
   - ~45 min read

3. **[OPTIMIZER_PRACTICAL_GUIDE.md](OPTIMIZER_PRACTICAL_GUIDE.md)**
   - Hands-on guide: when to use which optimizer
   - Hyperparameter tuning recommendations
   - Code templates and examples
   - Common pitfalls and solutions
   - ~30 min read

### For Hands-On Learning

4. **[optimizer_comparison_demo.py](optimizer_comparison_demo.py)**
   - Runnable Python script
   - Visualizes AdamW vs Muon behavior
   - Demonstrates Newton-Schulz orthogonalization
   - Generates comparison plots
   - ~10 min to run

```bash
# Install dependencies if needed
pip install matplotlib numpy

# Run the demo
python optimizer_comparison_demo.py
```

## 🎯 Learning Paths

### Path 1: Quick Overview (1 hour)
```
1. Read LEARNING_SUMMARY.md (15 min)
2. Skim TRAINING_PIPELINE_EXPLAINED.md (20 min)
3. Run optimizer_comparison_demo.py (10 min)
4. Read "Quick Decision Tree" in OPTIMIZER_PRACTICAL_GUIDE.md (15 min)
```

### Path 2: Deep Understanding (3-4 hours)
```
1. Read LEARNING_SUMMARY.md thoroughly (30 min)
2. Study TRAINING_PIPELINE_EXPLAINED.md in detail (90 min)
3. Read OPTIMIZER_PRACTICAL_GUIDE.md (45 min)
4. Run and modify optimizer_comparison_demo.py (30 min)
5. Experiment with nanochat training scripts (60 min)
```

### Path 3: Practical Implementation (2 hours)
```
1. Read "Quick Decision Tree" in OPTIMIZER_PRACTICAL_GUIDE.md (10 min)
2. Study "Code Template: Hybrid Optimizer Setup" (20 min)
3. Read "Hyperparameter Tuning Guide" (30 min)
4. Implement hybrid optimizers in your own project (60 min)
```

## 📊 Document Summary

| Document | Purpose | Difficulty | Time |
|----------|---------|------------|------|
| LEARNING_SUMMARY.md | Conceptual overview | ⭐ Easy | 15-30 min |
| TRAINING_PIPELINE_EXPLAINED.md | Technical deep dive | ⭐⭐⭐ Advanced | 45-90 min |
| OPTIMIZER_PRACTICAL_GUIDE.md | Implementation guide | ⭐⭐ Intermediate | 30-45 min |
| optimizer_comparison_demo.py | Interactive demo | ⭐⭐ Intermediate | 10-30 min |

## 🎓 What You'll Learn

### Core Concepts
- ✅ Why nanochat uses multiple optimizers (Muon + AdamW)
- ✅ How Muon's orthogonalization works (Newton-Schulz iteration)
- ✅ When to use AdamW vs Muon for different parameter types
- ✅ ZeRO-2 distributed training pattern
- ✅ Modern LLM training techniques

### PyTorch Skills
- ✅ Custom optimizer implementation
- ✅ Distributed training with torch.distributed
- ✅ Mixed precision training (bfloat16)
- ✅ Gradient accumulation
- ✅ torch.compile optimization
- ✅ Efficient in-place operations

### Practical Knowledge
- ✅ Hyperparameter tuning for both optimizers
- ✅ Learning rate scheduling strategies
- ✅ Memory optimization techniques
- ✅ Debugging training instabilities
- ✅ Performance profiling and optimization

## 🔍 Quick Reference

### Key Files in nanochat Codebase

```
nanochat/
├── adamw.py           → DistAdamW implementation (ZeRO-2)
├── muon.py            → Muon + DistMuon implementation
├── gpt.py             → Model architecture + setup_optimizers()
├── engine.py          → Inference engine with KV caching
├── dataloader.py      → Efficient data loading
└── common.py          → Utility functions

scripts/
├── base_train.py      → Main training script
├── mid_train.py       → Mid-training (domain adaptation)
├── chat_sft.py        → Supervised fine-tuning
└── chat_rl.py         → RL fine-tuning
```

### Important Code Sections

| Concept | File | Line Range |
|---------|------|------------|
| Optimizer setup | `gpt.py` | 213-242 |
| Training loop | `base_train.py` | 176-315 |
| DistAdamW step | `adamw.py` | 21-77 |
| DistMuon step | `muon.py` | 126-187 |
| Newton-Schulz | `muon.py` | 10-36 |
| LR schedule | `base_train.py` | 158-167 |
| Gradient accum | `base_train.py` | 268-275 |

## 🚀 Quick Start Examples

### Example 1: Basic Hybrid Optimizer Setup

```python
from nanochat.muon import Muon

# Split parameters
matrix_params = [p for p in model.transformer.h.parameters()]
other_params = [p for p in model.transformer.wte.parameters()] + \
               [p for p in model.lm_head.parameters()]

# Create optimizers
muon_opt = Muon(matrix_params, lr=0.02, momentum=0.95)
adam_opt = torch.optim.AdamW(other_params, lr=0.001)

# Training loop
for step in range(num_steps):
    loss = model(x, y)
    loss.backward()
    for opt in [adam_opt, muon_opt]:
        opt.step()
    model.zero_grad(set_to_none=True)
```

### Example 2: With Gradient Accumulation

```python
grad_accum_steps = 16

for step in range(num_steps):
    # Accumulate
    for micro_step in range(grad_accum_steps):
        loss = model(x, y) / grad_accum_steps
        loss.backward()
        x, y = next(data_loader)
    
    # Clip and step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
```

### Example 3: With Learning Rate Schedule

```python
def get_lr(step, total_steps):
    warmdown_start = int(0.8 * total_steps)
    if step < warmdown_start:
        return 1.0
    else:
        progress = (total_steps - step) / (total_steps - warmdown_start)
        return progress

for step in range(num_steps):
    # ... forward + backward ...
    
    # Update LR
    lrm = get_lr(step, num_steps)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lrm
    
    # Step
    for opt in optimizers:
        opt.step()
```

## 🔧 Debugging Tips

### Training Instability
```python
# Check gradient norms
grad_norm = torch.sqrt(sum(
    (p.grad ** 2).sum() for p in model.parameters() if p.grad is not None
))
print(f"Grad norm: {grad_norm:.2f}")

# If > 100: Reduce LR or increase clipping
# If NaN: Check for inf/nan in data or model
```

### Memory Issues
```python
# Track memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Solutions:
# 1. Reduce device_batch_size
# 2. Increase grad_accum_steps
# 3. Enable gradient checkpointing
# 4. Use ZeRO-2/3 optimizers
```

### Slow Training
```python
# Profile the code
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        loss = model(x, y)
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 📚 Additional Resources

### Papers
- **Muon**: https://kellerjordan.github.io/posts/muon/
- **AdamW**: "Decoupled Weight Decay Regularization" (ICLR 2019)
- **ZeRO**: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (SC 2020)
- **Flash Attention**: "FlashAttention: Fast and Memory-Efficient Exact Attention" (NeurIPS 2022)

### External Guides
- PyTorch Distributed: https://pytorch.org/tutorials/beginner/dist_overview.html
- Mixed Precision: https://pytorch.org/docs/stable/amp.html
- torch.compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

### Related Projects
- nanoGPT: https://github.com/karpathy/nanoGPT
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt
- Llama recipes: https://github.com/meta-llama/llama-recipes

## ❓ FAQ

### Q: Can I use Muon for convolutional networks?
**A:** Yes! Flatten the convolutional kernels to 2D: `[out_channels, in_channels * kernel_h * kernel_w]`

### Q: Should I always use Muon over AdamW?
**A:** Only for 2D weight matrices. Use AdamW for embeddings, LayerNorm, and biases.

### Q: What if I only have 1 GPU?
**A:** Regular Muon and AdamW work fine. No need for Dist* versions. Increase `grad_accum_steps` to maintain batch size.

### Q: Can I use fp16 instead of bfloat16?
**A:** Possible but not recommended. Newton-Schulz is more stable in bfloat16. If using fp16, add loss scaling.

### Q: How do I choose learning rates?
**A:** Start with Muon LR = 0.02, AdamW LR = 0.001. Scale both by `(d_model / 768) ** -0.5` for larger models.

## 🎯 Learning Checklist

Track your progress:

- [ ] Understand why nanochat uses multiple optimizers
- [ ] Can explain Newton-Schulz iteration in your own words
- [ ] Know when to use Muon vs AdamW
- [ ] Understand ZeRO-2 pattern (reduce-scatter + all-gather)
- [ ] Can implement gradient accumulation correctly
- [ ] Understand learning rate scaling with model size
- [ ] Can set up hybrid optimizers in your own code
- [ ] Know how to debug training instabilities
- [ ] Understand memory-compute tradeoffs
- [ ] Can profile and optimize PyTorch code

## 🏆 Next Challenges

Once you've mastered the basics:

1. **Implement a new optimizer**
   - Try SOAP, Schedule-Free, or others
   - Compare against Muon+AdamW

2. **Port to different architecture**
   - Vision Transformer (ViT)
   - Diffusion model
   - Multi-modal model

3. **Optimize further**
   - Implement ZeRO-3
   - Add gradient checkpointing
   - Try different orthogonalization methods

4. **Contribute to nanochat**
   - Run ablation studies
   - Document findings
   - Submit improvements

## 📞 Getting Help

- **GitHub Discussions**: https://github.com/karpathy/nanochat/discussions
- **Issues**: https://github.com/karpathy/nanochat/issues
- **Discord**: Check the repo for community links

## 🙏 Acknowledgments

This study guide is based on the excellent nanochat codebase by Andrej Karpathy and the Muon optimizer by Keller Jordan.

---

**Ready to dive in? Start with [LEARNING_SUMMARY.md](LEARNING_SUMMARY.md)! 🚀**
