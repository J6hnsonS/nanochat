"""
Visual demonstration of AdamW vs Muon optimizer behavior
Run with: python optimizer_comparison_demo.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Simple toy problem to visualize optimizer behavior
class SimpleTransformer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

def demo_optimizer_comparison():
    """Compare AdamW and Muon on a simple optimization problem"""
    
    torch.manual_seed(42)
    dim = 64
    batch_size = 32
    
    # Create two identical models
    model_adam = SimpleTransformer(dim)
    model_muon = SimpleTransformer(dim)
    
    # Copy weights to ensure identical initialization
    model_muon.load_state_dict(model_adam.state_dict())
    
    # Setup optimizers
    from nanochat.muon import Muon
    
    adam_optimizer = torch.optim.AdamW(
        model_adam.parameters(), 
        lr=0.001, 
        betas=(0.8, 0.95),
        weight_decay=0.01
    )
    
    muon_optimizer = Muon(
        [p for p in model_muon.parameters()],
        lr=0.02,  # Note: 20x higher LR!
        momentum=0.95
    )
    
    # Generate synthetic data
    X_train = torch.randn(1000, dim)
    y_train = torch.randn(1000, dim)
    
    # Training loop
    losses_adam = []
    losses_muon = []
    gradient_norms_adam = []
    gradient_norms_muon = []
    
    for epoch in range(100):
        # AdamW step
        adam_optimizer.zero_grad()
        pred_adam = model_adam(X_train[:batch_size])
        loss_adam = nn.functional.mse_loss(pred_adam, y_train[:batch_size])
        loss_adam.backward()
        
        # Track gradient norm
        grad_norm_adam = torch.sqrt(sum(
            (p.grad ** 2).sum() for p in model_adam.parameters() if p.grad is not None
        ))
        gradient_norms_adam.append(grad_norm_adam.item())
        
        adam_optimizer.step()
        losses_adam.append(loss_adam.item())
        
        # Muon step
        muon_optimizer.zero_grad()
        pred_muon = model_muon(X_train[:batch_size])
        loss_muon = nn.functional.mse_loss(pred_muon, y_train[:batch_size])
        loss_muon.backward()
        
        # Track gradient norm
        grad_norm_muon = torch.sqrt(sum(
            (p.grad ** 2).sum() for p in model_muon.parameters() if p.grad is not None
        ))
        gradient_norms_muon.append(grad_norm_muon.item())
        
        muon_optimizer.step()
        losses_muon.append(loss_muon.item())
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss comparison
    axes[0, 0].plot(losses_adam, label='AdamW (LR=0.001)', linewidth=2)
    axes[0, 0].plot(losses_muon, label='Muon (LR=0.02)', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Comparison: AdamW vs Muon')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss (log scale)
    axes[0, 1].semilogy(losses_adam, label='AdamW', linewidth=2)
    axes[0, 1].semilogy(losses_muon, label='Muon', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].set_title('Loss Comparison (Log Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gradient norms
    axes[1, 0].plot(gradient_norms_adam, label='AdamW', linewidth=2)
    axes[1, 0].plot(gradient_norms_muon, label='Muon', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('Gradient Norm Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final statistics
    stats_text = f"""
    Final Comparison:
    
    AdamW:
    - Final Loss: {losses_adam[-1]:.6f}
    - Mean Grad Norm: {np.mean(gradient_norms_adam):.4f}
    - Learning Rate: 0.001
    
    Muon:
    - Final Loss: {losses_muon[-1]:.6f}
    - Mean Grad Norm: {np.mean(gradient_norms_muon):.4f}
    - Learning Rate: 0.02 (20x higher!)
    
    Muon Advantages:
    ✓ Higher learning rate possible
    ✓ Faster convergence
    ✓ 2x less memory (no second moment)
    ✓ Orthogonalization = implicit preconditioning
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                    verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison plot to optimizer_comparison.png")
    
    return losses_adam, losses_muon


def visualize_newton_schulz():
    """Visualize the Newton-Schulz orthogonalization process"""
    from nanochat.muon import zeropower_via_newtonschulz5
    
    # Create a random 2D matrix
    torch.manual_seed(42)
    G = torch.randn(32, 32)
    
    # Compute SVD
    U, S, Vh = torch.linalg.svd(G)
    
    # Apply Newton-Schulz
    G_ortho = zeropower_via_newtonschulz5(G, steps=5)
    
    # Verify orthogonality: G_ortho @ G_ortho^T should be identity
    ortho_check = G_ortho @ G_ortho.T
    identity_error = torch.norm(ortho_check - torch.eye(32))
    
    print("\n" + "="*60)
    print("Newton-Schulz Orthogonalization Demo")
    print("="*60)
    print(f"\nOriginal matrix G shape: {G.shape}")
    print(f"Singular values of G: {S[:5].tolist()}")
    print(f"\nAfter Newton-Schulz (5 steps):")
    print(f"Orthogonalized matrix shape: {G_ortho.shape}")
    print(f"Orthogonality error ||G_ortho @ G_ortho^T - I||: {identity_error.item():.2e}")
    print(f"✓ Matrix is orthogonal!" if identity_error < 0.01 else "✗ Not orthogonal")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original matrix
    im1 = axes[0].imshow(G.numpy(), cmap='RdBu', vmin=-3, vmax=3)
    axes[0].set_title('Original Matrix G')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0])
    
    # Orthogonalized matrix
    im2 = axes[1].imshow(G_ortho.numpy(), cmap='RdBu', vmin=-3, vmax=3)
    axes[1].set_title('Orthogonalized Matrix (NS5)')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[1])
    
    # G_ortho @ G_ortho^T (should be identity)
    im3 = axes[2].imshow(ortho_check.numpy(), cmap='RdBu', vmin=-0.5, vmax=1.5)
    axes[2].set_title('G_ortho @ G_ortho^T\n(Should be Identity)')
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('newton_schulz_visualization.png', dpi=150, bbox_inches='tight')
    print("\nSaved Newton-Schulz visualization to newton_schulz_visualization.png")


def memory_comparison():
    """Compare memory usage of AdamW vs Muon"""
    print("\n" + "="*60)
    print("Memory Usage Comparison")
    print("="*60)
    
    # Model with 1B parameters (similar to GPT-2)
    num_params = 1_000_000_000
    param_size_bytes = 4  # fp32
    
    # AdamW memory
    adamw_params = num_params * param_size_bytes
    adamw_first_moment = num_params * param_size_bytes  # m
    adamw_second_moment = num_params * param_size_bytes  # v
    adamw_total = adamw_params + adamw_first_moment + adamw_second_moment
    
    # Muon memory
    muon_params = num_params * param_size_bytes
    muon_momentum = num_params * param_size_bytes
    muon_total = muon_params + muon_momentum
    
    print(f"\nFor a 1B parameter model:")
    print(f"\nAdamW:")
    print(f"  - Parameters: {adamw_params / 1e9:.2f} GB")
    print(f"  - First moment (m): {adamw_first_moment / 1e9:.2f} GB")
    print(f"  - Second moment (v): {adamw_second_moment / 1e9:.2f} GB")
    print(f"  - Total: {adamw_total / 1e9:.2f} GB")
    
    print(f"\nMuon:")
    print(f"  - Parameters: {muon_params / 1e9:.2f} GB")
    print(f"  - Momentum buffer: {muon_momentum / 1e9:.2f} GB")
    print(f"  - Total: {muon_total / 1e9:.2f} GB")
    
    savings = (adamw_total - muon_total) / adamw_total * 100
    print(f"\n✓ Muon saves {savings:.1f}% optimizer memory vs AdamW!")


def learning_rate_schedule_viz():
    """Visualize the learning rate schedule used in nanochat"""
    num_iterations = 10000
    warmup_ratio = 0.0
    warmdown_ratio = 0.2
    final_lr_frac = 0.0
    
    def get_lr_multiplier(it):
        warmup_iters = round(warmup_ratio * num_iterations)
        warmdown_iters = round(warmdown_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac
    
    # Momentum schedule for Muon
    def get_muon_momentum(it):
        frac = min(it / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95
    
    iterations = np.arange(num_iterations)
    lr_mults = [get_lr_multiplier(i) for i in iterations]
    momentums = [get_muon_momentum(i) for i in iterations]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Learning rate schedule
    axes[0].plot(iterations, lr_mults, linewidth=2, color='blue')
    axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Max LR')
    axes[0].axvline(x=num_iterations * (1 - warmdown_ratio), color='orange', 
                    linestyle='--', alpha=0.5, label='Warmdown starts')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('LR Multiplier')
    axes[0].set_title('Learning Rate Schedule (No Warmup, 20% Cosine Decay)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(-0.05, 1.1)
    
    # Momentum schedule
    axes[1].plot(iterations[:1000], momentums[:1000], linewidth=2, color='green')
    axes[1].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Final momentum')
    axes[1].axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='Initial momentum')
    axes[1].axvline(x=300, color='purple', linestyle='--', alpha=0.5, label='Warmup complete')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Momentum')
    axes[1].set_title('Muon Momentum Schedule (300 step warmup)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(0.84, 0.96)
    
    plt.tight_layout()
    plt.savefig('lr_momentum_schedule.png', dpi=150, bbox_inches='tight')
    print("\nSaved LR/momentum schedule to lr_momentum_schedule.png")


if __name__ == "__main__":
    print("\n🚀 nanochat Optimizer Deep Dive Demo\n")
    
    # Run demonstrations
    print("1. Running optimizer comparison...")
    try:
        losses_adam, losses_muon = demo_optimizer_comparison()
        print("   ✓ Complete")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n2. Visualizing Newton-Schulz orthogonalization...")
    try:
        visualize_newton_schulz()
        print("   ✓ Complete")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Computing memory comparison...")
    try:
        memory_comparison()
        print("   ✓ Complete")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n4. Visualizing LR/momentum schedules...")
    try:
        learning_rate_schedule_viz()
        print("   ✓ Complete")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Demo complete! Check the generated PNG files.")
    print("="*60)
