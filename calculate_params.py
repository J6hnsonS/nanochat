"""
Step-by-step parameter calculation for nanochat d20 model.
This script breaks down EXACTLY where every parameter comes from.
"""

def calculate_transformer_params(depth, vocab_size=65536):
    """
    Calculate parameters for a nanochat model given depth.
    
    Architecture rules (from base_train.py):
    - model_dim = depth × 64
    - num_heads = ceil(model_dim / 128)  # head_dim fixed at 128
    - num_kv_heads = num_heads  # MHA by default (can be changed for MQA/GQA)
    """
    
    # Model dimensions
    num_layers = depth
    model_dim = depth * 64  # embedding dimension / hidden size
    num_heads = max(1, (model_dim + 127) // 128)  # ceiling division
    num_kv_heads = num_heads  # 1:1 ratio for MHA
    head_dim = model_dim // num_heads
    
    print("="*80)
    print(f"MODEL ARCHITECTURE: d{depth}")
    print("="*80)
    print(f"Vocab size:        {vocab_size:,}")
    print(f"Model dimension:   {model_dim:,} (depth × 64 = {depth} × 64)")
    print(f"Number of layers:  {num_layers}")
    print(f"Number of heads:   {num_heads}")
    print(f"Head dimension:    {head_dim}")
    print(f"KV heads:          {num_kv_heads} (MHA)")
    print()
    
    # =========================================================================
    # 1. TOKEN EMBEDDING (wte)
    # =========================================================================
    print("="*80)
    print("1. TOKEN EMBEDDING (wte)")
    print("="*80)
    print(f"Shape: Embedding(vocab_size={vocab_size:,}, embedding_dim={model_dim:,})")
    print(f"Calculation: vocab_size × model_dim")
    print(f"           = {vocab_size:,} × {model_dim:,}")
    
    wte_params = vocab_size * model_dim
    print(f"           = {wte_params:,}")
    print()
    
    # =========================================================================
    # 2. SINGLE TRANSFORMER BLOCK
    # =========================================================================
    print("="*80)
    print("2. SINGLE TRANSFORMER BLOCK")
    print("="*80)
    print()
    
    # -------------------------------------------------------------------------
    # 2a. Attention Layer
    # -------------------------------------------------------------------------
    print("-" * 40)
    print("2a. ATTENTION LAYER")
    print("-" * 40)
    print()
    
    # Query projection: model_dim -> (num_heads × head_dim)
    print(f"Q projection: Linear(in={model_dim:,}, out={num_heads} × {head_dim} = {num_heads * head_dim:,}, bias=False)")
    print(f"  Parameters = in_features × out_features")
    print(f"             = {model_dim:,} × {num_heads * head_dim:,}")
    c_q_params = model_dim * (num_heads * head_dim)
    print(f"             = {c_q_params:,}")
    print()
    
    # Key projection: model_dim -> (num_kv_heads × head_dim)
    print(f"K projection: Linear(in={model_dim:,}, out={num_kv_heads} × {head_dim} = {num_kv_heads * head_dim:,}, bias=False)")
    print(f"  Parameters = in_features × out_features")
    print(f"             = {model_dim:,} × {num_kv_heads * head_dim:,}")
    c_k_params = model_dim * (num_kv_heads * head_dim)
    print(f"             = {c_k_params:,}")
    print()
    
    # Value projection: model_dim -> (num_kv_heads × head_dim)
    print(f"V projection: Linear(in={model_dim:,}, out={num_kv_heads} × {head_dim} = {num_kv_heads * head_dim:,}, bias=False)")
    print(f"  Parameters = in_features × out_features")
    print(f"             = {model_dim:,} × {num_kv_heads * head_dim:,}")
    c_v_params = model_dim * (num_kv_heads * head_dim)
    print(f"             = {c_v_params:,}")
    print()
    
    # Output projection: model_dim -> model_dim
    print(f"Output projection: Linear(in={model_dim:,}, out={model_dim:,}, bias=False)")
    print(f"  Parameters = in_features × out_features")
    print(f"             = {model_dim:,} × {model_dim:,}")
    c_proj_attn_params = model_dim * model_dim
    print(f"             = {c_proj_attn_params:,}")
    print()
    
    # Total attention params
    attn_params = c_q_params + c_k_params + c_v_params + c_proj_attn_params
    print(f"TOTAL ATTENTION PARAMS = Q + K + V + Output")
    print(f"                       = {c_q_params:,} + {c_k_params:,} + {c_v_params:,} + {c_proj_attn_params:,}")
    print(f"                       = {attn_params:,}")
    print()
    
    # -------------------------------------------------------------------------
    # 2b. MLP Layer
    # -------------------------------------------------------------------------
    print("-" * 40)
    print("2b. MLP LAYER (with 4× expansion)")
    print("-" * 40)
    print()
    
    # First linear layer: model_dim -> 4 × model_dim
    mlp_hidden_dim = 4 * model_dim
    print(f"First layer: Linear(in={model_dim:,}, out=4 × {model_dim:,} = {mlp_hidden_dim:,}, bias=False)")
    print(f"  Parameters = in_features × out_features")
    print(f"             = {model_dim:,} × {mlp_hidden_dim:,}")
    c_fc_params = model_dim * mlp_hidden_dim
    print(f"             = {c_fc_params:,}")
    print()
    
    # Activation: ReLU² (no parameters!)
    print(f"Activation: ReLU² (relu(x).square())")
    print(f"  Parameters = 0 (activation functions have no parameters)")
    print()
    
    # Second linear layer: 4 × model_dim -> model_dim
    print(f"Second layer: Linear(in={mlp_hidden_dim:,}, out={model_dim:,}, bias=False)")
    print(f"  Parameters = in_features × out_features")
    print(f"             = {mlp_hidden_dim:,} × {model_dim:,}")
    c_proj_mlp_params = mlp_hidden_dim * model_dim
    print(f"             = {c_proj_mlp_params:,}")
    print()
    
    # Total MLP params
    mlp_params = c_fc_params + c_proj_mlp_params
    print(f"TOTAL MLP PARAMS = First layer + Second layer")
    print(f"                 = {c_fc_params:,} + {c_proj_mlp_params:,}")
    print(f"                 = {mlp_params:,}")
    print()
    
    # -------------------------------------------------------------------------
    # 2c. Normalization Layers
    # -------------------------------------------------------------------------
    print("-" * 40)
    print("2c. NORMALIZATION (RMSNorm)")
    print("-" * 40)
    print(f"RMSNorm: NO learnable parameters!")
    print(f"  - No γ (scale) parameter")
    print(f"  - No β (shift) parameter")
    print(f"  - Just functional: x / sqrt(mean(x²))")
    print(f"  Parameters = 0")
    print()
    
    # -------------------------------------------------------------------------
    # Total per block
    # -------------------------------------------------------------------------
    block_params = attn_params + mlp_params
    print("-" * 40)
    print("TOTAL PER TRANSFORMER BLOCK")
    print("-" * 40)
    print(f"Block params = Attention + MLP + RMSNorm")
    print(f"             = {attn_params:,} + {mlp_params:,} + 0")
    print(f"             = {block_params:,}")
    print()
    
    # =========================================================================
    # 3. ALL TRANSFORMER LAYERS
    # =========================================================================
    print("="*80)
    print("3. ALL TRANSFORMER LAYERS")
    print("="*80)
    print(f"Number of blocks: {num_layers}")
    print(f"Params per block: {block_params:,}")
    print(f"Total = {num_layers} × {block_params:,}")
    all_layers_params = num_layers * block_params
    print(f"      = {all_layers_params:,}")
    print()
    
    # =========================================================================
    # 4. LM HEAD (unembedding)
    # =========================================================================
    print("="*80)
    print("4. LM HEAD (unembedding layer)")
    print("="*80)
    print(f"Shape: Linear(in={model_dim:,}, out={vocab_size:,}, bias=False)")
    print(f"Calculation: model_dim × vocab_size")
    print(f"           = {model_dim:,} × {vocab_size:,}")
    lm_head_params = model_dim * vocab_size
    print(f"           = {lm_head_params:,}")
    print()
    print("NOTE: lm_head is UNTIED from token embedding (separate weights)")
    print()
    
    # =========================================================================
    # 5. ROTARY EMBEDDINGS
    # =========================================================================
    print("="*80)
    print("5. ROTARY EMBEDDINGS (RoPE)")
    print("="*80)
    print("Rotary embeddings: NO learnable parameters!")
    print("  - Precomputed cos/sin tables")
    print("  - Stored as buffers (not saved in checkpoint)")
    print("  - Just a lookup table based on position")
    print("  Parameters = 0")
    print()
    
    # =========================================================================
    # TOTAL MODEL PARAMETERS
    # =========================================================================
    print("="*80)
    print("TOTAL MODEL PARAMETERS")
    print("="*80)
    print(f"Token embedding (wte):     {wte_params:,}")
    print(f"Transformer layers:        {all_layers_params:,}")
    print(f"  └─ {num_layers} blocks × {block_params:,} params/block")
    print(f"LM head (unembedding):     {lm_head_params:,}")
    print(f"RMSNorm:                   0")
    print(f"RoPE:                      0")
    print("-" * 80)
    
    total_params = wte_params + all_layers_params + lm_head_params
    print(f"TOTAL:                     {total_params:,}")
    print(f"                           ≈ {total_params/1e6:.1f} million")
    print(f"                           ≈ {total_params/1e9:.2f} billion")
    print()
    
    # =========================================================================
    # PARAMETER BREAKDOWN BY TYPE
    # =========================================================================
    print("="*80)
    print("PARAMETER BREAKDOWN BY TYPE")
    print("="*80)
    
    # Attention vs MLP split
    total_attn_params = num_layers * attn_params
    total_mlp_params = num_layers * mlp_params
    
    print(f"Attention parameters:      {total_attn_params:,} ({100*total_attn_params/total_params:.1f}%)")
    print(f"MLP parameters:            {total_mlp_params:,} ({100*total_mlp_params/total_params:.1f}%)")
    print(f"Embedding + Unembedding:   {wte_params + lm_head_params:,} ({100*(wte_params + lm_head_params)/total_params:.1f}%)")
    print()
    
    # 2D vs other
    matrix_params = all_layers_params  # All transformer layers are 2D matrices
    embedding_params = wte_params + lm_head_params
    
    print("OPTIMIZATION STRATEGY:")
    print(f"Muon (2D matrices):        {matrix_params:,} ({100*matrix_params/total_params:.1f}%)")
    print(f"AdamW (embeddings):        {embedding_params:,} ({100*embedding_params/total_params:.1f}%)")
    print()
    
    return total_params


def compare_with_actual_model():
    """Verify our calculations match the actual model."""
    print("="*80)
    print("VERIFICATION: Loading actual model and counting parameters")
    print("="*80)
    
    import torch
    from nanochat.gpt import GPT, GPTConfig
    
    # Create d20 model
    depth = 20
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    vocab_size = 65536
    
    config = GPTConfig(
        sequence_len=2048,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim
    )
    
    with torch.device("meta"):
        model = GPT(config)
    
    # Count actual parameters
    actual_total = sum(p.numel() for p in model.parameters())
    
    print(f"Actual model parameters: {actual_total:,}")
    print()
    
    # Our calculation
    calculated_total = calculate_transformer_params(depth, vocab_size)
    
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Calculated:  {calculated_total:,}")
    print(f"Actual:      {actual_total:,}")
    print(f"Difference:  {abs(calculated_total - actual_total):,}")
    print()
    
    if calculated_total == actual_total:
        print("✅ PERFECT MATCH! Our calculation is correct!")
    else:
        print("❌ Mismatch - need to debug")
        
        # Debug: show per-parameter breakdown
        print("\nDETAILED BREAKDOWN:")
        for name, param in model.named_parameters():
            print(f"  {name:40s}: {param.numel():>12,} params, shape {list(param.shape)}")


if __name__ == "__main__":
    # Calculate for d20 model
    calculate_transformer_params(depth=20, vocab_size=65536)
    
    print("\n\n")
    
    # Verify with actual model
    compare_with_actual_model()
    
    print("\n\n")
    print("="*80)
    print("TRY OTHER SIZES")
    print("="*80)
    print("\nYou can calculate parameters for other depths:")
    print("  - d4  (toy model for CPU):     ", end="")
    tiny_params = 4 * 64 * 4 * 64 * (4 * 4 * 4 + 8 * 4 * 4) + 2 * 65536 * (4 * 64)
    print(f"~{calculate_transformer_params(4, 65536)/1e6:.0f}M params")
    
    print("  - d12 (small):                 ", end="")
    print(f"~{calculate_transformer_params(12, 65536)/1e6:.0f}M params")
    
    print("  - d20 (speedrun $100):         ", end="")
    print(f"~{calculate_transformer_params(20, 65536)/1e6:.0f}M params")
    
    print("  - d26 (GPT-2 performance):     ", end="")
    print(f"~{calculate_transformer_params(26, 65536)/1e9:.2f}B params")
    
    print("  - d32 (mentioned in README):   ", end="")
    print(f"~{calculate_transformer_params(32, 65536)/1e9:.2f}B params")
