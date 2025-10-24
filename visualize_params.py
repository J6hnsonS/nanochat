"""
Visual ASCII diagram of parameter flow through a transformer block.
Shows where all the parameters are and their sizes.
"""

def draw_transformer_block(depth=20):
    """Draw a detailed diagram of a single transformer block with parameter counts."""
    
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    head_dim = model_dim // num_heads
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SINGLE TRANSFORMER BLOCK (d{depth})                        ║
║                      Input: (batch, seq_len, {model_dim})                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────────┐
  │                        INPUT TENSOR                           │
  │                   Shape: (B, T, {model_dim})                          │
  └──────────────────────────────────────────────────────────────┘
                                  │
                                  │ (residual connection saves this)
                                  ├──────────────────────────┐
                                  │                          │
                                  ▼                          │
  ┌──────────────────────────────────────────────────────────────┐
  │                         RMSNorm                              │
  │                  Params: 0 (no learnable params)             │
  └──────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ╔══════════════════════════════════════════════════════════════╗
  ║                    ATTENTION MECHANISM                       ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         ║
  ║  │  Q Proj     │  │  K Proj     │  │  V Proj     │         ║
  ║  │  Linear     │  │  Linear     │  │  Linear     │         ║
  ║  │  {model_dim:4d}→{model_dim:4d}  │  │  {model_dim:4d}→{model_dim:4d}  │  │  {model_dim:4d}→{model_dim:4d}  │         ║
  ║  │             │  │             │  │             │         ║
  ║  │  {format_params(model_dim * model_dim):>10s}  │  │  {format_params(model_dim * model_dim):>10s}  │  │  {format_params(model_dim * model_dim):>10s}  │         ║
  ║  │  params     │  │  params     │  │  params     │         ║
  ║  └─────────────┘  └─────────────┘  └─────────────┘         ║
  ║         │                │                │                 ║
  ║         └────────────────┴────────────────┘                 ║
  ║                         │                                   ║
  ║                         ▼                                   ║
  ║         ┌──────────────────────────────┐                    ║
  ║         │  Reshape to {num_heads} heads         │                    ║
  ║         │  ({model_dim}) → ({num_heads}, {head_dim})        │                    ║
  ║         └──────────────────────────────┘                    ║
  ║                         │                                   ║
  ║                         ▼                                   ║
  ║         ┌──────────────────────────────┐                    ║
  ║         │   Apply RoPE (no params!)   │                    ║
  ║         │   QK Normalization           │                    ║
  ║         └──────────────────────────────┘                    ║
  ║                         │                                   ║
  ║                         ▼                                   ║
  ║         ┌──────────────────────────────┐                    ║
  ║         │  Scaled Dot-Product Attn    │                    ║
  ║         │  softmax(QK^T/√d_k) V       │                    ║
  ║         └──────────────────────────────┘                    ║
  ║                         │                                   ║
  ║                         ▼                                   ║
  ║         ┌──────────────────────────────┐                    ║
  ║         │  Concatenate heads           │                    ║
  ║         │  ({num_heads}, {head_dim}) → ({model_dim})        │                    ║
  ║         └──────────────────────────────┘                    ║
  ║                         │                                   ║
  ║                         ▼                                   ║
  ║         ┌──────────────────────────────┐                    ║
  ║         │    Output Projection         │                    ║
  ║         │    Linear {model_dim:4d}→{model_dim:4d}          │                    ║
  ║         │    {format_params(model_dim * model_dim):>10s} params             │                    ║
  ║         └──────────────────────────────┘                    ║
  ║                                                              ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Total Attention Params: {format_params(4 * model_dim * model_dim):>10s}                      ║
  ╚══════════════════════════════════════════════════════════════╝
                                  │
                                  │
                                  ├◄─────────────────────────┘
                                  │  (add residual)
                                  ▼
  ┌──────────────────────────────────────────────────────────────┐
  │               OUTPUT: (B, T, {model_dim})                            │
  └──────────────────────────────────────────────────────────────┘
                                  │
                                  │ (residual connection saves this)
                                  ├──────────────────────────┐
                                  │                          │
                                  ▼                          │
  ┌──────────────────────────────────────────────────────────────┐
  │                         RMSNorm                              │
  │                  Params: 0 (no learnable params)             │
  └──────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ╔══════════════════════════════════════════════════════════════╗
  ║                        MLP (Feed-Forward)                    ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║         ┌──────────────────────────────┐                    ║
  ║         │    First Linear (Expand)    │                    ║
  ║         │    {model_dim:4d} → {4*model_dim:4d}             │                    ║
  ║         │    {format_params(model_dim * 4 * model_dim):>10s} params             │                    ║
  ║         └──────────────────────────────┘                    ║
  ║                         │                                   ║
  ║                         ▼                                   ║
  ║         ┌──────────────────────────────┐                    ║
  ║         │       ReLU² Activation       │                    ║
  ║         │      relu(x).square()        │                    ║
  ║         │      Params: 0               │                    ║
  ║         └──────────────────────────────┘                    ║
  ║                         │                                   ║
  ║                         ▼                                   ║
  ║         ┌──────────────────────────────┐                    ║
  ║         │   Second Linear (Project)    │                    ║
  ║         │    {4*model_dim:4d} → {model_dim:4d}             │                    ║
  ║         │    {format_params(model_dim * 4 * model_dim):>10s} params             │                    ║
  ║         └──────────────────────────────┘                    ║
  ║                                                              ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Total MLP Params: {format_params(2 * model_dim * 4 * model_dim):>10s}                          ║
  ╚══════════════════════════════════════════════════════════════╝
                                  │
                                  │
                                  ├◄─────────────────────────┘
                                  │  (add residual)
                                  ▼
  ┌──────────────────────────────────────────────────────────────┐
  │           BLOCK OUTPUT: (B, T, {model_dim})                          │
  └──────────────────────────────────────────────────────────────┘

  ╔══════════════════════════════════════════════════════════════╗
  ║            TOTAL PARAMS PER BLOCK: {format_params((4 + 8) * model_dim * model_dim):>10s}                   ║
  ║            (Attention: {format_params(4 * model_dim * model_dim):>10s}, MLP: {format_params(8 * model_dim * model_dim):>10s})              ║
  ╚══════════════════════════════════════════════════════════════╝
""")


def format_params(n):
    """Format parameter count nicely."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return f"{n}"


def draw_full_model(depth=20, vocab_size=65536):
    """Draw the complete model architecture."""
    
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    
    emb_params = vocab_size * model_dim
    block_params = 12 * model_dim * model_dim
    all_blocks = depth * block_params
    lm_head = model_dim * vocab_size
    total = emb_params + all_blocks + lm_head
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         FULL MODEL ARCHITECTURE                              ║
║                               nanochat d{depth}                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Input: Token IDs (batch, seq_len)
                    │
                    ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                     TOKEN EMBEDDING (wte)                      │
  │                  Embedding({vocab_size}, {model_dim})                    │
  │                  Params: {format_params(emb_params):>10s}                              │
  └────────────────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                         RMSNorm                                │
  │                    Params: 0                                   │
  └────────────────────────────────────────────────────────────────┘
                    │
                    ▼
  ╔════════════════════════════════════════════════════════════════╗
  ║                    TRANSFORMER LAYERS                          ║
  ║                                                                ║
  ║   ┌──────────────────────────────────────────────────┐        ║
  ║   │         Transformer Block 0                      │        ║
  ║   │         Params: {format_params(block_params):>10s}                     │        ║
  ║   │         (Attention + MLP)                        │        ║
  ║   └──────────────────────────────────────────────────┘        ║
  ║                         │                                     ║
  ║                         ▼                                     ║
  ║   ┌──────────────────────────────────────────────────┐        ║
  ║   │         Transformer Block 1                      │        ║
  ║   │         Params: {format_params(block_params):>10s}                     │        ║
  ║   └──────────────────────────────────────────────────┘        ║
  ║                         │                                     ║
  ║                        ...                                    ║
  ║                         │                                     ║
  ║   ┌──────────────────────────────────────────────────┐        ║
  ║   │         Transformer Block {depth-1}                     │        ║
  ║   │         Params: {format_params(block_params):>10s}                     │        ║
  ║   └──────────────────────────────────────────────────┘        ║
  ║                                                                ║
  ╠════════════════════════════════════════════════════════════════╣
  ║   Total: {depth} blocks × {format_params(block_params):>10s} = {format_params(all_blocks):>10s}             ║
  ╚════════════════════════════════════════════════════════════════╝
                    │
                    ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                         RMSNorm                                │
  │                    Params: 0                                   │
  └────────────────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                   LM HEAD (unembedding)                        │
  │                  Linear({model_dim}, {vocab_size})                       │
  │                  Params: {format_params(lm_head):>10s}                              │
  │                  (UNTIED from token embedding)                 │
  └────────────────────────────────────────────────────────────────┘
                    │
                    ▼
  Output: Logits (batch, seq_len, {vocab_size})
  
  ╔════════════════════════════════════════════════════════════════╗
  ║                      TOTAL PARAMETERS                          ║
  ╠════════════════════════════════════════════════════════════════╣
  ║  Token Embedding:    {format_params(emb_params):>10s}  ({100*emb_params/total:>5.1f}%)                ║
  ║  Transformer Layers: {format_params(all_blocks):>10s}  ({100*all_blocks/total:>5.1f}%)                ║
  ║  LM Head:            {format_params(lm_head):>10s}  ({100*lm_head/total:>5.1f}%)                ║
  ║  ─────────────────────────────────────────────────────────     ║
  ║  TOTAL:              {format_params(total):>10s}                          ║
  ╚════════════════════════════════════════════════════════════════╝
  
  Key Facts:
  ─────────────────────────────────────────────────────────────────
  • MLP has 2× more params than Attention (8×d² vs 4×d²)
  • Token embedding = LM head size (but UNTIED weights!)
  • RMSNorm and RoPE: 0 learnable parameters!
  • No bias terms in any Linear layers
  • Model optimized with: Muon ({100*all_blocks/total:.1f}%) + AdamW ({100*(emb_params+lm_head)/total:.1f}%)
""")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("               NANOCHAT d20 PARAMETER VISUALIZATION")
    print("="*80 + "\n")
    
    # Show single block detail
    draw_transformer_block(depth=20)
    
    print("\n\n")
    
    # Show full model
    draw_full_model(depth=20, vocab_size=65536)
    
    print("\n\n" + "="*80)
    print("                     COMPARISON TABLE")
    print("="*80 + "\n")
    
    # Comparison table
    configs = [
        (4, "Tiny (CPU)"),
        (12, "Small"),
        (20, "Speedrun ($100)"),
        (26, "GPT-2 level"),
        (32, "Hosted demo"),
    ]
    
    print(f"{'Model':<18} {'Layers':<8} {'d_model':<10} {'Params':<12} {'$/train':<10}")
    print("-" * 70)
    
    for depth, name in configs:
        model_dim = depth * 64
        vocab_size = 65536
        
        emb = vocab_size * model_dim
        blocks = depth * 12 * model_dim * model_dim
        lm = model_dim * vocab_size
        total = emb + blocks + lm
        
        # Rough cost estimate (assuming $3/GPU-hour, 8 GPUs, Chinchilla optimal)
        # tokens = 20 × total_params
        # time_hours = (tokens × 6 × total_params) / (8 × 3.12e14)  # 8×H100 = 2.5 PFLOPS
        # cost = time_hours × 24
        costs = {4: "<$1", 12: "~$30", 20: "~$100", 26: "~$300", 32: "~$800"}
        
        print(f"{name:<18} {depth:<8} {model_dim:<10} {format_params(total):<12} {costs.get(depth, '?'):<10}")
    
    print("\n" + "="*80)
    print("Try different configurations yourself!")
    print("="*80)
