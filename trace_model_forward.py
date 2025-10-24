#!/usr/bin/env python3
"""
Trace through a forward pass of nanochat d20 showing exact structure.
This script shows what happens at each step with actual tensor shapes.
"""

def trace_forward_pass():
    """Simulate a forward pass showing all operations and shapes."""
    
    print("="*80)
    print("NANOCHAT D20 MODEL - FORWARD PASS TRACE")
    print("="*80)
    print()
    
    # Input
    B, T = 2, 4
    vocab_size = 65536
    d_model = 1280
    n_heads = 10
    head_dim = 128
    n_layers = 20
    
    print(f"📥 INPUT")
    print(f"   Token IDs: shape ({B}, {T})")
    print(f"   Example: [[1, 2, 3, 4], [5, 6, 7, 8]]")
    print()
    
    # Step 1: Token Embedding
    print(f"{'─'*80}")
    print(f"STEP 1: Token Embedding")
    print(f"{'─'*80}")
    print(f"   Operation: nn.Embedding(vocab_size={vocab_size}, embedding_dim={d_model})")
    print(f"   Input:  ({B}, {T})")
    print(f"   Output: ({B}, {T}, {d_model})")
    print(f"   Params: {vocab_size * d_model:,}")
    print()
    
    # Step 2: Initial RMSNorm
    print(f"{'─'*80}")
    print(f"STEP 2: Initial RMSNorm")
    print(f"{'─'*80}")
    print(f"   Operation: x = x / sqrt(mean(x²))")
    print(f"   Input:  ({B}, {T}, {d_model})")
    print(f"   Output: ({B}, {T}, {d_model})")
    print(f"   Params: 0 (functional normalization)")
    print()
    
    # Step 3-22: Transformer Blocks
    print(f"{'═'*80}")
    print(f"STEPS 3-22: Transformer Blocks (×{n_layers})")
    print(f"{'═'*80}")
    print()
    
    for block_idx in range(min(2, n_layers)):  # Show first 2 blocks in detail
        print(f"╔{'═'*78}╗")
        print(f"║ BLOCK {block_idx:<71} ║")
        print(f"╠{'═'*78}╣")
        print()
        
        # Attention sub-layer
        print(f"  ┌{'─'*76}┐")
        print(f"  │ ATTENTION SUB-LAYER{' '*53}│")
        print(f"  └{'─'*76}┘")
        print()
        
        print(f"  Step {block_idx}.1: Save residual")
        print(f"     x_residual = x  (save for later)")
        print()
        
        print(f"  Step {block_idx}.2: Pre-norm")
        print(f"     x_normed = RMSNorm(x)")
        print(f"     Shape: ({B}, {T}, {d_model}) → ({B}, {T}, {d_model})")
        print()
        
        print(f"  Step {block_idx}.3: Project to Q, K, V")
        print(f"     Q = Linear(x_normed)  [{d_model} → {d_model}]  ({d_model * d_model:,} params)")
        print(f"     K = Linear(x_normed)  [{d_model} → {d_model}]  ({d_model * d_model:,} params)")
        print(f"     V = Linear(x_normed)  [{d_model} → {d_model}]  ({d_model * d_model:,} params)")
        print(f"     Shape: ({B}, {T}, {d_model}) → ({B}, {T}, {d_model}) for each")
        print()
        
        print(f"  Step {block_idx}.4: Reshape to multi-head")
        print(f"     Q = Q.reshape({B}, {T}, {n_heads}, {head_dim})")
        print(f"     K = K.reshape({B}, {T}, {n_heads}, {head_dim})")
        print(f"     V = V.reshape({B}, {T}, {n_heads}, {head_dim})")
        print(f"     Shape: ({B}, {T}, {d_model}) → ({B}, {T}, {n_heads}, {head_dim})")
        print()
        
        print(f"  Step {block_idx}.5: Apply RoPE (Rotary Position Embeddings)")
        print(f"     Q = apply_rotary_emb(Q, cos, sin)")
        print(f"     K = apply_rotary_emb(K, cos, sin)")
        print(f"     (Rotates vectors based on position - NO parameters!)")
        print()
        
        print(f"  Step {block_idx}.6: QK Normalization")
        print(f"     Q = RMSNorm(Q)")
        print(f"     K = RMSNorm(K)")
        print(f"     (Stabilizes attention - NO parameters!)")
        print()
        
        print(f"  Step {block_idx}.7: Transpose for batch matmul")
        print(f"     Q, K, V = transpose(1, 2)")
        print(f"     Shape: ({B}, {T}, {n_heads}, {head_dim}) → ({B}, {n_heads}, {T}, {head_dim})")
        print()
        
        print(f"  Step {block_idx}.8: Scaled Dot-Product Attention")
        print(f"     scores = Q @ K.T / sqrt({head_dim})")
        print(f"     scores = causal_mask(scores)  (mask future tokens)")
        print(f"     attn_weights = softmax(scores)")
        print(f"     attn_out = attn_weights @ V")
        print(f"     Shape: ({B}, {n_heads}, {T}, {head_dim})")
        print()
        
        print(f"  Step {block_idx}.9: Concatenate heads")
        print(f"     attn_out = transpose(1, 2).reshape({B}, {T}, {d_model})")
        print(f"     Shape: ({B}, {n_heads}, {T}, {head_dim}) → ({B}, {T}, {d_model})")
        print()
        
        print(f"  Step {block_idx}.10: Output projection")
        print(f"     attn_out = Linear(attn_out)  [{d_model} → {d_model}]")
        print(f"     Params: {d_model * d_model:,}")
        print()
        
        print(f"  Step {block_idx}.11: 🔵 RESIDUAL CONNECTION 🔵")
        print(f"     x = x_residual + attn_out")
        print(f"     Shape: ({B}, {T}, {d_model})")
        print()
        
        # MLP sub-layer
        print(f"  ┌{'─'*76}┐")
        print(f"  │ MLP SUB-LAYER{' '*59}│")
        print(f"  └{'─'*76}┘")
        print()
        
        print(f"  Step {block_idx}.12: Save residual (again!)")
        print(f"     x_residual = x  (NEW residual, not the original!)")
        print()
        
        print(f"  Step {block_idx}.13: Pre-norm")
        print(f"     x_normed = RMSNorm(x)")
        print(f"     Shape: ({B}, {T}, {d_model}) → ({B}, {T}, {d_model})")
        print()
        
        print(f"  Step {block_idx}.14: Expand 4×")
        print(f"     hidden = Linear(x_normed)  [{d_model} → {4*d_model}]")
        print(f"     Params: {d_model * 4 * d_model:,}")
        print(f"     Shape: ({B}, {T}, {d_model}) → ({B}, {T}, {4*d_model})")
        print()
        
        print(f"  Step {block_idx}.15: ReLU² Activation")
        print(f"     hidden = relu(hidden).square()")
        print(f"     (Element-wise: max(0,x)² - NO parameters!)")
        print()
        
        print(f"  Step {block_idx}.16: Project back")
        print(f"     mlp_out = Linear(hidden)  [{4*d_model} → {d_model}]")
        print(f"     Params: {4 * d_model * d_model:,}")
        print(f"     Shape: ({B}, {T}, {4*d_model}) → ({B}, {T}, {d_model})")
        print()
        
        print(f"  Step {block_idx}.17: 🔵 RESIDUAL CONNECTION 🔵")
        print(f"     x = x_residual + mlp_out")
        print(f"     Shape: ({B}, {T}, {d_model})")
        print()
        
        print(f"╚{'═'*78}╝")
        print()
    
    if n_layers > 2:
        print(f"   ... (Blocks 2-{n_layers-1} follow same pattern) ...")
        print()
    
    # Final steps
    print(f"{'─'*80}")
    print(f"STEP 23: Final RMSNorm")
    print(f"{'─'*80}")
    print(f"   Operation: x = RMSNorm(x)")
    print(f"   Shape: ({B}, {T}, {d_model}) → ({B}, {T}, {d_model})")
    print()
    
    print(f"{'─'*80}")
    print(f"STEP 24: LM Head")
    print(f"{'─'*80}")
    print(f"   Operation: logits = Linear(x)  [{d_model} → {vocab_size}]")
    print(f"   Params: {d_model * vocab_size:,}")
    print(f"   Shape: ({B}, {T}, {d_model}) → ({B}, {T}, {vocab_size})")
    print()
    
    print(f"{'─'*80}")
    print(f"STEP 25: Logit Softcapping")
    print(f"{'─'*80}")
    print(f"   Operation: logits = 15 * tanh(logits / 15)")
    print(f"   Effect: Bounds logits to range [-15, 15]")
    print(f"   Shape: ({B}, {T}, {vocab_size}) → ({B}, {T}, {vocab_size})")
    print()
    
    print(f"{'─'*80}")
    print(f"STEP 26: Float32 Conversion")
    print(f"{'─'*80}")
    print(f"   Operation: logits = logits.float()")
    print(f"   Reason: Numerical stability for cross-entropy")
    print()
    
    print(f"{'─'*80}")
    print(f"STEP 27: Loss Computation (if training)")
    print(f"{'─'*80}")
    print(f"   Operation: loss = F.cross_entropy(logits, targets)")
    print(f"   Logits: ({B}, {T}, {vocab_size}) → flatten to ({B*T}, {vocab_size})")
    print(f"   Targets: ({B}, {T}) → flatten to ({B*T},)")
    print(f"   Output: scalar loss value")
    print()
    
    print(f"📤 OUTPUT")
    if True:  # Training mode
        print(f"   Loss: scalar (single number)")
    else:  # Inference mode
        print(f"   Logits: ({B}, {T}, {vocab_size})")
    print()
    
    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print()
    print(f"Total steps: 27")
    print(f"  - 1 embedding")
    print(f"  - 1 initial norm")
    print(f"  - {n_layers} × (attention + MLP) = {n_layers} × 2 = {n_layers*2} sub-layers")
    print(f"  - 1 final norm")
    print(f"  - 1 LM head")
    print(f"  - 1 softcapping")
    print(f"  - 1 float conversion")
    print(f"  - 1 loss computation")
    print()
    
    print(f"Key patterns:")
    print(f"  ✓ Pre-norm: RMSNorm BEFORE each sub-layer")
    print(f"  ✓ Residual: Add back AFTER each sub-layer")
    print(f"  ✓ Two residuals per block: one for attn, one for MLP")
    print(f"  ✓ RoPE applied INSIDE attention (not a separate layer)")
    print(f"  ✓ ReLU² applied INSIDE MLP (not a separate layer)")
    print()
    
    print(f"Components with 0 parameters:")
    print(f"  • RMSNorm (functional, no γ/β)")
    print(f"  • RoPE (precomputed cos/sin tables)")
    print(f"  • ReLU² activation")
    print(f"  • Attention mechanism itself (uses QKV projections)")
    print(f"  • Softcapping (just tanh transformation)")
    print()
    
    print(f"Parameter distribution:")
    print(f"  • Token embedding: {vocab_size * d_model:>12,} (15.0%)")
    print(f"  • Transformer layers: {n_layers * 12 * d_model * d_model:>12,} (70.1%)")
    print(f"  • LM head: {d_model * vocab_size:>12,} (15.0%)")
    print(f"  • TOTAL: {2 * vocab_size * d_model + n_layers * 12 * d_model * d_model:>12,}")
    print()


def show_code_structure():
    """Show the actual code structure from gpt.py."""
    
    print()
    print("="*80)
    print("ACTUAL CODE STRUCTURE (from nanochat/gpt.py)")
    print("="*80)
    print()
    
    code = '''
# Block.forward (lines 132-135)
def forward(self, x, cos_sin, kv_cache):
    x = x + self.attn(norm(x), cos_sin, kv_cache)  # Pre-norm + residual
    x = x + self.mlp(norm(x))                      # Pre-norm + residual
    return x

# GPT.forward (lines 244-276, simplified)
def forward(self, idx, targets=None, kv_cache=None):
    # Get rotary embeddings
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
    
    # Forward the trunk of the Transformer
    x = self.transformer.wte(idx)         # Token embedding
    x = norm(x)                           # Initial norm
    for block in self.transformer.h:
        x = block(x, cos_sin, kv_cache)   # Each block: 2 residuals
    x = norm(x)                           # Final norm
    
    # Forward the lm_head (compute logits)
    softcap = 15
    logits = self.lm_head(x)
    logits = softcap * torch.tanh(logits / softcap)  # Logits softcap
    logits = logits.float()                          # Use fp32
    
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                               targets.view(-1))
        return loss
    else:
        return logits
'''
    
    print(code)
    print()


if __name__ == "__main__":
    trace_forward_pass()
    show_code_structure()
    
    print("="*80)
    print("🎓 Now you know the EXACT structure of nanochat d20!")
    print("="*80)
    print()
    print("Key takeaway: The structure is:")
    print()
    print("  Embedding → Norm → 20×(Norm→Attn→Add, Norm→MLP→Add) → Norm → LM_head → Softcap")
    print()
    print("Each block has TWO residual connections (one for attention, one for MLP)!")
