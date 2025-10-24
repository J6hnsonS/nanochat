"""
Complete Parameter Calculation Guide for nanochat d20 Model

This script shows EXACTLY how to count parameters in a transformer,
with detailed breakdowns for each component.
"""

# ============================================================================
# STEP 1: Define the model configuration
# ============================================================================
print("=" * 70)
print("STEP 1: Model Configuration")
print("=" * 70)

depth = 20
vocab_size = 65536  # 2^16
model_dim = depth * 64  # = 1280
num_heads = 10
head_dim = model_dim // num_heads  # = 128
sequence_len = 2048

print(f"Depth (n_layer):              {depth}")
print(f"Vocabulary size (vocab_size): {vocab_size:,}")
print(f"Model dimension (n_embd):     {model_dim}")
print(f"Number of attention heads:    {num_heads}")
print(f"Head dimension:               {head_dim}")
print(f"Sequence length:              {sequence_len}")
print()

# ============================================================================
# STEP 2: Token Embedding Layer
# ============================================================================
print("=" * 70)
print("STEP 2: Token Embedding (wte)")
print("=" * 70)

# nn.Embedding(vocab_size, model_dim)
# This creates a lookup table: each of vocab_size tokens maps to a model_dim vector
token_embedding_params = vocab_size * model_dim

print(f"nn.Embedding({vocab_size:,}, {model_dim})")
print(f"Shape: [{vocab_size:,} × {model_dim}]")
print(f"Parameters: {vocab_size:,} × {model_dim} = {token_embedding_params:,}")
print()

# ============================================================================
# STEP 3: Single Transformer Block - ATTENTION
# ============================================================================
print("=" * 70)
print("STEP 3: Attention in ONE Transformer Block")
print("=" * 70)

print("\nThe attention mechanism has 4 weight matrices:")
print("  1. Q (Query) projection")
print("  2. K (Key) projection")
print("  3. V (Value) projection")
print("  4. Output projection")
print()

# Q projection: nn.Linear(model_dim, num_heads * head_dim, bias=False)
q_params = model_dim * (num_heads * head_dim)
print(f"1. Q projection: nn.Linear({model_dim}, {num_heads * head_dim}, bias=False)")
print(f"   Shape: [{model_dim} × {num_heads * head_dim}]")
print(f"   Parameters: {model_dim} × {num_heads * head_dim} = {q_params:,}")
print()

# K projection: nn.Linear(model_dim, num_kv_heads * head_dim, bias=False)
# For d20, num_kv_heads = num_heads = 10 (standard MHA, not MQA)
num_kv_heads = num_heads
k_params = model_dim * (num_kv_heads * head_dim)
print(f"2. K projection: nn.Linear({model_dim}, {num_kv_heads * head_dim}, bias=False)")
print(f"   Shape: [{model_dim} × {num_kv_heads * head_dim}]")
print(f"   Parameters: {model_dim} × {num_kv_heads * head_dim} = {k_params:,}")
print()

# V projection: nn.Linear(model_dim, num_kv_heads * head_dim, bias=False)
v_params = model_dim * (num_kv_heads * head_dim)
print(f"3. V projection: nn.Linear({model_dim}, {num_kv_heads * head_dim}, bias=False)")
print(f"   Shape: [{model_dim} × {num_kv_heads * head_dim}]")
print(f"   Parameters: {model_dim} × {num_kv_heads * head_dim} = {v_params:,}")
print()

# Output projection: nn.Linear(model_dim, model_dim, bias=False)
output_params = model_dim * model_dim
print(f"4. Output projection: nn.Linear({model_dim}, {model_dim}, bias=False)")
print(f"   Shape: [{model_dim} × {model_dim}]")
print(f"   Parameters: {model_dim} × {model_dim} = {output_params:,}")
print()

attention_params = q_params + k_params + v_params + output_params
print(f"{'─' * 70}")
print(f"TOTAL ATTENTION PARAMETERS: {attention_params:,}")
print()

# ============================================================================
# STEP 4: Single Transformer Block - MLP
# ============================================================================
print("=" * 70)
print("STEP 4: MLP in ONE Transformer Block")
print("=" * 70)

print("\nThe MLP has 2 weight matrices with 4× expansion:")
print("  1. Expand: model_dim → 4 × model_dim")
print("  2. Project back: 4 × model_dim → model_dim")
print()

mlp_hidden_dim = 4 * model_dim  # = 5120

# First linear layer (expansion): nn.Linear(model_dim, 4 * model_dim, bias=False)
mlp_fc_params = model_dim * mlp_hidden_dim
print(f"1. c_fc (expand): nn.Linear({model_dim}, {mlp_hidden_dim}, bias=False)")
print(f"   Shape: [{model_dim} × {mlp_hidden_dim}]")
print(f"   Parameters: {model_dim} × {mlp_hidden_dim} = {mlp_fc_params:,}")
print()

# Activation: ReLU^2 (0 parameters!)
print(f"2. Activation: F.relu(x).square()")
print(f"   Parameters: 0 (no learnable parameters in activation functions)")
print()

# Second linear layer (project back): nn.Linear(4 * model_dim, model_dim, bias=False)
mlp_proj_params = mlp_hidden_dim * model_dim
print(f"3. c_proj (project): nn.Linear({mlp_hidden_dim}, {model_dim}, bias=False)")
print(f"   Shape: [{mlp_hidden_dim} × {model_dim}]")
print(f"   Parameters: {mlp_hidden_dim} × {model_dim} = {mlp_proj_params:,}")
print()

mlp_params = mlp_fc_params + mlp_proj_params
print(f"{'─' * 70}")
print(f"TOTAL MLP PARAMETERS: {mlp_params:,}")
print()

# ============================================================================
# STEP 5: Single Transformer Block - TOTAL
# ============================================================================
print("=" * 70)
print("STEP 5: ONE Complete Transformer Block")
print("=" * 70)

# RMSNorm has 0 parameters (it's functional, no learnable weights)
rmsnorm_params = 0

block_params = attention_params + mlp_params + rmsnorm_params
print(f"Attention:       {attention_params:,}")
print(f"MLP:             {mlp_params:,}")
print(f"RMSNorm (×2):    {rmsnorm_params} (functional, no learnable parameters)")
print(f"{'─' * 70}")
print(f"ONE BLOCK TOTAL: {block_params:,}")
print()

# ============================================================================
# STEP 6: All Transformer Blocks
# ============================================================================
print("=" * 70)
print("STEP 6: ALL Transformer Blocks")
print("=" * 70)

all_blocks_params = depth * block_params
print(f"Number of blocks: {depth}")
print(f"Parameters per block: {block_params:,}")
print(f"Total parameters: {depth} × {block_params:,} = {all_blocks_params:,}")
print()

# ============================================================================
# STEP 7: LM Head (Unembedding)
# ============================================================================
print("=" * 70)
print("STEP 7: LM Head (Output Projection)")
print("=" * 70)

# nn.Linear(model_dim, vocab_size, bias=False)
lm_head_params = model_dim * vocab_size

print(f"nn.Linear({model_dim}, {vocab_size:,}, bias=False)")
print(f"Shape: [{model_dim} × {vocab_size:,}]")
print(f"Parameters: {model_dim} × {vocab_size:,} = {lm_head_params:,}")
print()

# NOTE: This is the TRANSPOSE of the token embedding!
# But they are UNTIED (separate weights), not shared.
print("NOTE: LM head is the same size as token embedding,")
print("      but they have SEPARATE weights (untied).")
print()

# ============================================================================
# STEP 8: GRAND TOTAL
# ============================================================================
print("=" * 70)
print("STEP 8: FINAL MODEL PARAMETER COUNT")
print("=" * 70)

total_params = token_embedding_params + all_blocks_params + lm_head_params

print(f"Token Embedding (wte):         {token_embedding_params:,}")
print(f"Transformer Blocks (×{depth}):       {all_blocks_params:,}")
print(f"LM Head (unembedding):         {lm_head_params:,}")
print(f"{'═' * 70}")
print(f"TOTAL MODEL PARAMETERS:        {total_params:,}")
print(f"                               {total_params / 1e6:.1f}M")
print(f"                               {total_params / 1e9:.2f}B")
print()

# ============================================================================
# STEP 9: Percentage Breakdown
# ============================================================================
print("=" * 70)
print("STEP 9: Parameter Distribution")
print("=" * 70)

emb_pct = 100 * token_embedding_params / total_params
blocks_pct = 100 * all_blocks_params / total_params
lm_pct = 100 * lm_head_params / total_params

print(f"Token Embedding:     {emb_pct:5.1f}%  ({token_embedding_params:,})")
print(f"Transformer Blocks:  {blocks_pct:5.1f}%  ({all_blocks_params:,})")
print(f"LM Head:             {lm_pct:5.1f}%  ({lm_head_params:,})")
print(f"{'─' * 70}")
print(f"TOTAL:              100.0%  ({total_params:,})")
print()

# Within blocks, attention vs MLP
attn_pct = 100 * attention_params / block_params
mlp_pct = 100 * mlp_params / block_params
print("Within each block:")
print(f"  Attention:  {attn_pct:5.1f}%  ({attention_params:,})")
print(f"  MLP:        {mlp_pct:5.1f}%  ({mlp_params:,})")
print()

# ============================================================================
# STEP 10: Verification with PyTorch
# ============================================================================
print("=" * 70)
print("STEP 10: Verify with Actual Model")
print("=" * 70)

try:
    import torch
    from nanochat.gpt import GPT, GPTConfig
    
    config = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim
    )
    
    # Create model on CPU to avoid CUDA issues
    with torch.device('cpu'):
        model = GPT(config)
    
    # Count parameters
    actual_params = sum(p.numel() for p in model.parameters())
    
    print(f"Calculated parameters: {total_params:,}")
    print(f"Actual model parameters: {actual_params:,}")
    
    if total_params == actual_params:
        print("✓ MATCH! Calculation is correct.")
    else:
        print(f"✗ MISMATCH! Difference: {abs(total_params - actual_params):,}")
        print("  (This might be due to implementation details)")
    
    print()
    
    # Show parameter breakdown from actual model
    print("Actual parameter breakdown:")
    print(f"  wte (embedding):    {model.transformer.wte.weight.numel():,}")
    print(f"  lm_head:            {model.lm_head.weight.numel():,}")
    
    # First block breakdown
    if len(model.transformer.h) > 0:
        block0 = model.transformer.h[0]
        block0_params = sum(p.numel() for p in block0.parameters())
        attn_params = sum(p.numel() for p in block0.attn.parameters())
        mlp_params = sum(p.numel() for p in block0.mlp.parameters())
        print(f"  Block[0] total:     {block0_params:,}")
        print(f"    - attention:      {attn_params:,}")
        print(f"    - mlp:            {mlp_params:,}")
    
except ImportError:
    print("(Skipping verification - nanochat module not available)")

print()
print("=" * 70)
print("CALCULATION COMPLETE!")
print("=" * 70)
