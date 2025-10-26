#!/usr/bin/env python3
"""
Interactive Parameter Calculator for Transformer Models

Usage:
    python parameter_calculator.py
    
Or import and use:
    from parameter_calculator import calculate_params
    params = calculate_params(vocab_size=65536, d_model=1280, n_layers=20)
"""

def calculate_params(vocab_size, d_model, n_layers, n_kv_heads=None, tied_embeddings=False, expansion=4, include_bias=False):
    """
    Calculate the total number of parameters in a transformer model.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension (hidden size)
        n_layers: Number of transformer blocks
        n_kv_heads: Number of KV heads for GQA/MQA (None = use n_heads, standard MHA)
        tied_embeddings: Whether to tie input and output embeddings
        expansion: MLP expansion factor (typically 4)
        include_bias: Whether Linear layers have bias (typically False in modern models)
    
    Returns:
        dict with parameter breakdown
    """
    n_heads = max(1, (d_model + 127) // 128)  # Head dim = 128
    head_dim = d_model // n_heads
    
    if n_kv_heads is None:
        n_kv_heads = n_heads  # Standard multi-head attention
    
    # Embeddings
    token_embedding = vocab_size * d_model
    lm_head = 0 if tied_embeddings else (d_model * vocab_size)
    embeddings_total = token_embedding + lm_head
    
    # Attention in one block
    q_params = d_model * (n_heads * head_dim)
    k_params = d_model * (n_kv_heads * head_dim)
    v_params = d_model * (n_kv_heads * head_dim)
    o_params = (n_heads * head_dim) * d_model
    
    if include_bias:
        q_params += n_heads * head_dim
        k_params += n_kv_heads * head_dim
        v_params += n_kv_heads * head_dim
        o_params += d_model
    
    attention_params = q_params + k_params + v_params + o_params
    
    # MLP in one block
    mlp_hidden = expansion * d_model
    mlp_fc = d_model * mlp_hidden
    mlp_proj = mlp_hidden * d_model
    
    if include_bias:
        mlp_fc += mlp_hidden
        mlp_proj += d_model
    
    mlp_params = mlp_fc + mlp_proj
    
    # Blocks
    block_params = attention_params + mlp_params
    all_blocks = n_layers * block_params
    
    # Total
    total = embeddings_total + all_blocks
    
    return {
        'total': total,
        'token_embedding': token_embedding,
        'lm_head': lm_head,
        'embeddings_total': embeddings_total,
        'all_blocks': all_blocks,
        'block_params': block_params,
        'attention_params': attention_params,
        'mlp_params': mlp_params,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'head_dim': head_dim,
    }


def print_params(config_name, vocab_size, d_model, n_layers, **kwargs):
    """Print parameter breakdown for a configuration."""
    params = calculate_params(vocab_size, d_model, n_layers, **kwargs)
    
    print(f"\n{'=' * 70}")
    print(f"Configuration: {config_name}")
    print(f"{'=' * 70}")
    print(f"Vocab size:           {vocab_size:,}")
    print(f"Model dimension:      {d_model}")
    print(f"Number of layers:     {n_layers}")
    print(f"Attention heads:      {params['n_heads']}")
    print(f"KV heads:             {params['n_kv_heads']}")
    print(f"Head dimension:       {params['head_dim']}")
    print()
    
    print(f"{'─' * 70}")
    print(f"Token Embedding:      {params['token_embedding']:>15,}")
    print(f"LM Head:              {params['lm_head']:>15,}")
    print(f"Embeddings subtotal:  {params['embeddings_total']:>15,}  ({100*params['embeddings_total']/params['total']:5.1f}%)")
    print()
    print(f"One block:")
    print(f"  Attention:          {params['attention_params']:>15,}  ({100*params['attention_params']/params['block_params']:5.1f}%)")
    print(f"  MLP:                {params['mlp_params']:>15,}  ({100*params['mlp_params']/params['block_params']:5.1f}%)")
    print(f"  Block total:        {params['block_params']:>15,}")
    print()
    print(f"All {n_layers} blocks:         {params['all_blocks']:>15,}  ({100*params['all_blocks']/params['total']:5.1f}%)")
    print(f"{'─' * 70}")
    print(f"TOTAL PARAMETERS:     {params['total']:>15,}")
    print(f"                      {params['total']/1e6:>15.1f}M")
    print(f"                      {params['total']/1e9:>15.2f}B")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("TRANSFORMER PARAMETER CALCULATOR")
    print("=" * 70)
    
    # Nanochat models
    print("\n" + "=" * 70)
    print("NANOCHAT MODELS")
    print("=" * 70)
    
    print_params("d4 (tiny)",    65536, 256,  4)
    print_params("d8 (small)",   65536, 512,  8)
    print_params("d12 (medium)", 65536, 768,  12)
    print_params("d16",          65536, 1024, 16)
    print_params("d20 (speedrun)", 65536, 1280, 20)
    print_params("d26",          65536, 1664, 26)
    print_params("d32 (run1000)", 65536, 2048, 32)
    
    # Comparison with famous models
    print("\n" + "=" * 70)
    print("COMPARISON: FAMOUS MODELS (approximate)")
    print("=" * 70)
    
    # GPT-2 Small (124M params)
    print_params("GPT-2 Small", 50257, 768, 12, tied_embeddings=True)
    
    # GPT-2 Medium (355M params)
    print_params("GPT-2 Medium", 50257, 1024, 24, tied_embeddings=True)
    
    # GPT-2 Large (774M params)
    print_params("GPT-2 Large", 50257, 1280, 36, tied_embeddings=True)
    
    # LLaMA 7B (approximate)
    print_params("LLaMA-like 7B", 32000, 4096, 32, tied_embeddings=False)
    
    # Efficiency comparison
    print("\n" + "=" * 70)
    print("EFFICIENCY EXPERIMENTS")
    print("=" * 70)
    
    print_params("d20 Standard", 65536, 1280, 20, n_kv_heads=10)
    print_params("d20 with MQA (1 KV head)", 65536, 1280, 20, n_kv_heads=1)
    print_params("d20 with GQA (2 KV heads)", 65536, 1280, 20, n_kv_heads=2)
    print_params("d20 Tied Embeddings", 65536, 1280, 20, tied_embeddings=True)
    print_params("d20 32K vocab", 32000, 1280, 20)
    
    # Show the simple formula
    print("\n" + "=" * 70)
    print("QUICK FORMULA")
    print("=" * 70)
    print()
    print("For standard MHA with untied embeddings:")
    print()
    print("  TOTAL = 2(V × D) + L × 12 × D²")
    print()
    print("Where:")
    print("  V = vocab_size")
    print("  D = d_model")
    print("  L = n_layers")
    print()
    print("Example for d20:")
    print("  = 2(65,536 × 1,280) + 20 × 12 × 1,280²")
    print("  = 167,772,160 + 393,216,000")
    print("  = 560,988,160")
    print()
    
    # Memory estimation
    print("\n" + "=" * 70)
    print("MEMORY ESTIMATION (d20 model)")
    print("=" * 70)
    total_params = 560_988_160
    print()
    print(f"Parameters: {total_params:,}")
    print()
    print("Storage (weights only):")
    print(f"  FP32 (4 bytes):     {total_params * 4 / 1e9:>8.2f} GB")
    print(f"  FP16 (2 bytes):     {total_params * 2 / 1e9:>8.2f} GB")
    print(f"  BF16 (2 bytes):     {total_params * 2 / 1e9:>8.2f} GB")
    print(f"  INT8 (1 byte):      {total_params * 1 / 1e9:>8.2f} GB")
    print(f"  INT4 (0.5 bytes):   {total_params * 0.5 / 1e9:>8.2f} GB")
    print()
    print("Training memory (rough estimate, FP32 + gradients + optimizer states):")
    print(f"  AdamW (fp32):       {total_params * 16 / 1e9:>8.2f} GB  (4x params + optimizer)")
    print(f"  Mixed precision:    {total_params * 12 / 1e9:>8.2f} GB  (approx)")
    print()
    print("Note: Actual memory usage is higher due to:")
    print("  - Activations (depends on batch size and sequence length)")
    print("  - KV cache (for inference)")
    print("  - Optimizer states")
    print()
    
    print("=" * 70)
    print("END OF CALCULATIONS")
    print("=" * 70)
