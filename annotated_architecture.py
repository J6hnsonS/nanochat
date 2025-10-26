"""
Annotated Architecture - Shows EXACTLY where parameters live in the code

This walks through nanochat/gpt.py with parameter counts for d20 model.
"""

print("=" * 80)
print("NANOCHAT d20 MODEL - ANNOTATED ARCHITECTURE")
print("=" * 80)
print()

# Configuration for d20
vocab_size = 65536
d_model = 1280
n_layers = 20
n_heads = 10
head_dim = 128

print(f"Configuration: vocab={vocab_size:,}, d_model={d_model}, layers={n_layers}")
print()
print("=" * 80)

# =============================================================================
# From nanochat/gpt.py, line 138-146
# =============================================================================
print("CLASS GPT - Main Model")
print("=" * 80)
print()

print("def __init__(self, config):")
print("    self.config = config")
print("    self.transformer = nn.ModuleDict({")
print()

# Token Embedding
wte_params = vocab_size * d_model
print(f"        'wte': nn.Embedding({vocab_size:,}, {d_model}),")
print(f"               ↑ Parameters: {wte_params:,}")
print()

# Transformer blocks
print(f"        'h': nn.ModuleList([Block(config) for _ in range({n_layers})]),")
print(f"             ↑ See Block details below")
print()

print("    })")
print()

# LM Head
lm_head_params = d_model * vocab_size
print(f"    self.lm_head = nn.Linear({d_model}, {vocab_size:,}, bias=False)")
print(f"                   ↑ Parameters: {lm_head_params:,}")
print()

print("=" * 80)
print()

# =============================================================================
# From nanochat/gpt.py, line 126-135
# =============================================================================
print("CLASS BLOCK - One Transformer Layer")
print("=" * 80)
print()

print("def __init__(self, config, layer_idx):")
print(f"    self.attn = CausalSelfAttention(config, layer_idx)")
print(f"    self.mlp = MLP(config)")
print()

print("def forward(self, x, cos_sin, kv_cache):")
print("    # First sublayer: Attention with residual")
print("    x = x + self.attn(norm(x), cos_sin, kv_cache)")
print("           ↑            ↑")
print("         residual   RMSNorm (0 params!)")
print()
print("    # Second sublayer: MLP with residual")
print("    x = x + self.mlp(norm(x))")
print("           ↑         ↑")
print("         residual  RMSNorm (0 params!)")
print("    return x")
print()

print("=" * 80)
print()

# =============================================================================
# From nanochat/gpt.py, line 51-110
# =============================================================================
print("CLASS CausalSelfAttention - Attention Mechanism")
print("=" * 80)
print()

print("def __init__(self, config, layer_idx):")
print(f"    self.n_head = {n_heads}")
print(f"    self.n_kv_head = {n_heads}  # Same as n_head for standard MHA")
print(f"    self.n_embd = {d_model}")
print(f"    self.head_dim = {head_dim}")
print()

# Q projection
q_params = d_model * d_model
print(f"    self.c_q = nn.Linear({d_model}, {n_heads} * {head_dim}, bias=False)")
print(f"               = nn.Linear({d_model}, {d_model}, bias=False)")
print(f"               ↑ Parameters: {q_params:,}")
print()

# K projection
k_params = d_model * d_model
print(f"    self.c_k = nn.Linear({d_model}, {n_heads} * {head_dim}, bias=False)")
print(f"               = nn.Linear({d_model}, {d_model}, bias=False)")
print(f"               ↑ Parameters: {k_params:,}")
print()

# V projection
v_params = d_model * d_model
print(f"    self.c_v = nn.Linear({d_model}, {n_heads} * {head_dim}, bias=False)")
print(f"               = nn.Linear({d_model}, {d_model}, bias=False)")
print(f"               ↑ Parameters: {v_params:,}")
print()

# Output projection
o_params = d_model * d_model
print(f"    self.c_proj = nn.Linear({d_model}, {d_model}, bias=False)")
print(f"                  ↑ Parameters: {o_params:,}")
print()

attn_total = q_params + k_params + v_params + o_params
print(f"    TOTAL ATTENTION PARAMS: {attn_total:,}")
print()

print("def forward(self, x, cos_sin, kv_cache):")
print("    # 1. Project to Q, K, V")
print("    q = self.c_q(x)  # Shape: [B, T, d_model] -> [B, T, n_heads * head_dim]")
print("    k = self.c_k(x)")
print("    v = self.c_v(x)")
print()
print("    # 2. Reshape to multi-head format")
print(f"    q = q.view(B, T, {n_heads}, {head_dim})  # [B, T, {n_heads}, {head_dim}]")
print(f"    k = k.view(B, T, {n_heads}, {head_dim})")
print(f"    v = v.view(B, T, {n_heads}, {head_dim})")
print()
print("    # 3. Apply RoPE (Rotary Position Embeddings)")
print("    cos, sin = cos_sin")
print("    q = apply_rotary_emb(q, cos, sin)")
print("    k = apply_rotary_emb(k, cos, sin)")
print("        ↑ 0 learnable parameters! Just rotation of existing vectors")
print()
print("    # 4. Apply QK normalization")
print("    q = norm(q)  # RMSNorm, 0 parameters!")
print("    k = norm(k)")
print()
print("    # 5. Transpose for attention: [B, T, H, D] -> [B, H, T, D]")
print("    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)")
print()
print("    # 6. Scaled dot-product attention")
print("    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)")
print("        ↑ 0 parameters! Just matrix multiplications")
print()
print("    # 7. Reshape back and project")
print("    y = y.transpose(1, 2).contiguous().view(B, T, d_model)")
print("    y = self.c_proj(y)")
print("    return y")
print()

print("=" * 80)
print()

# =============================================================================
# From nanochat/gpt.py, line 113-123
# =============================================================================
print("CLASS MLP - Feed-Forward Network")
print("=" * 80)
print()

mlp_hidden = 4 * d_model

print("def __init__(self, config):")
fc_params = d_model * mlp_hidden
print(f"    self.c_fc = nn.Linear({d_model}, {mlp_hidden}, bias=False)")
print(f"                ↑ Expand 4×")
print(f"                ↑ Parameters: {fc_params:,}")
print()

proj_params = mlp_hidden * d_model
print(f"    self.c_proj = nn.Linear({mlp_hidden}, {d_model}, bias=False)")
print(f"                  ↑ Project back")
print(f"                  ↑ Parameters: {proj_params:,}")
print()

mlp_total = fc_params + proj_params
print(f"    TOTAL MLP PARAMS: {mlp_total:,}")
print()

print("def forward(self, x):")
print("    x = self.c_fc(x)        # [B, T, 1280] -> [B, T, 5120]")
print("    x = F.relu(x).square()  # ReLU² activation (0 params!)")
print("    x = self.c_proj(x)      # [B, T, 5120] -> [B, T, 1280]")
print("    return x")
print()

print("=" * 80)
print()

# =============================================================================
# Summary
# =============================================================================
print("FINAL PARAMETER SUMMARY")
print("=" * 80)
print()

block_params = attn_total + mlp_total
all_blocks = n_layers * block_params
total = wte_params + all_blocks + lm_head_params

print(f"Token Embedding (wte):          {wte_params:>15,}  (15.0%)")
print()
print(f"Transformer Blocks (×{n_layers}):")
print(f"  One block:")
print(f"    - Attention:                {attn_total:>15,}  (33.3% of block)")
print(f"    - MLP:                      {mlp_total:>15,}  (66.7% of block)")
print(f"    - RMSNorm (×2):                           0  (functional)")
print(f"  Block subtotal:               {block_params:>15,}")
print(f"  All {n_layers} blocks:                 {all_blocks:>15,}  (70.1%)")
print()
print(f"LM Head (unembedding):          {lm_head_params:>15,}  (15.0%)")
print()
print("─" * 80)
print(f"TOTAL MODEL PARAMETERS:         {total:>15,}")
print(f"                                {total/1e6:>15.1f}M")
print("─" * 80)
print()

print("=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print()
print("1. NO PARAMETERS in:")
print("   - RMSNorm (functional, no learnable weights)")
print("   - RoPE (precomputed rotation, no learning)")
print("   - Attention mechanism itself (just uses Q, K, V)")
print("   - Activations (ReLU²)")
print()
print("2. NO BIAS in any Linear layer:")
print("   - All Linear layers have bias=False")
print("   - This saves n_out parameters per Linear layer")
print()
print("3. UNTIED embeddings:")
print("   - Token embedding (wte) and LM head are SEPARATE")
print("   - Same size, but different weights")
print("   - Optimized differently (AdamW with different LRs)")
print()
print("4. MLP dominates each block:")
print("   - MLP: 8 × d_model² = 13.1M params")
print("   - Attention: 4 × d_model² = 6.6M params")
print("   - Ratio: 2:1")
print()
print("5. Simple scaling formula:")
print("   ONE BLOCK = 12 × d_model²")
print("   For d20: 12 × 1280² = 19,660,800 ✓")
print()

print("=" * 80)
print("FORMULA FOR ANY MODEL")
print("=" * 80)
print()
print("Given: vocab_size (V), d_model (D), n_layers (L)")
print()
print("TOTAL = 2(V × D) + L × 12 × D²")
print()
print("Where:")
print("  2(V × D)     = Token embedding + LM head (untied)")
print("  12 × D²      = One block (4×D² attention + 8×D² MLP)")
print("  L × 12 × D²  = All transformer blocks")
print()
print("For d20:")
print(f"  = 2 × ({vocab_size:,} × {d_model})")
print(f"    + {n_layers} × 12 × {d_model}²")
print(f"  = {2 * vocab_size * d_model:,}")
print(f"    + {n_layers * 12 * d_model * d_model:,}")
print(f"  = {total:,} ✓")
print()

print("=" * 80)
print("END OF ANNOTATED ARCHITECTURE")
print("=" * 80)
