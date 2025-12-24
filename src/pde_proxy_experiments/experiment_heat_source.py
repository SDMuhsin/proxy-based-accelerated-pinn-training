"""
Experiment 1B: Source Training on 1D Heat Equation

Train Physics Operator Network on Heat equation u_t = α·u_xx with α=0.01 (source domain).

Objective:
- Validate that two-network architecture can solve linear Heat equation accurately
- Establish baseline accuracy before attempting FD proxy transfer
- Compare to known analytical solution: u(x,t) = exp(-α·π²·t)·sin(πx)

Success Criteria:
- L2 relative error < 1% (much easier than Burgers due to linearity)
- PDE residual < 1e-4
- Both networks converge stably
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from heat_physics_operator import HeatPhysicsOperatorPINN


class HeatDataGenerator:
    """Generate training/test data using analytical solution"""

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def analytical_solution(self, x, t):
        """u(x,t) = exp(-α·π²·t)·sin(πx)"""
        return np.exp(-self.alpha * np.pi**2 * t) * np.sin(np.pi * x)

    def get_ic_data(self, n_points=100):
        """Initial condition: u(x,0) = sin(πx)"""
        x = np.random.uniform(-1, 1, (n_points, 1))
        t = np.zeros((n_points, 1))
        u = np.sin(np.pi * x)
        return x, t, u

    def get_bc_data(self, n_points=100):
        """Boundary conditions: u(±1,t) = 0"""
        n_per_boundary = n_points // 2

        # Left boundary x = -1
        x_left = np.full((n_per_boundary, 1), -1.0)
        t_left = np.random.uniform(0, 1, (n_per_boundary, 1))
        u_left = np.zeros((n_per_boundary, 1))

        # Right boundary x = 1
        x_right = np.full((n_per_boundary, 1), 1.0)
        t_right = np.random.uniform(0, 1, (n_per_boundary, 1))
        u_right = np.zeros((n_per_boundary, 1))

        x = np.vstack([x_left, x_right])
        t = np.vstack([t_left, t_right])
        u = np.vstack([u_left, u_right])

        return x, t, u

    def get_collocation_data(self, n_points=2000):
        """Random collocation points for physics loss"""
        x = np.random.uniform(-1, 1, (n_points, 1))
        t = np.random.uniform(0, 1, (n_points, 1))
        return x, t

    def get_test_data(self, nx=101, nt=51):
        """Dense grid for testing"""
        x = np.linspace(-1, 1, nx)
        t = np.linspace(0, 1, nt)
        X, T = np.meshgrid(x, t)
        x_test = X.flatten()[:, None]
        t_test = T.flatten()[:, None]
        u_test = self.analytical_solution(x_test, t_test)

        return x_test, t_test, u_test


def train_source_domain(
    model,
    data_generator,
    n_epochs=10000,
    n_ic=100,
    n_bc=100,
    n_collocation=2000,
    lr_solution=1e-3,
    lr_physics=1e-3,
    log_interval=100,
    save_dir="results"
):
    """
    Train Physics Operator Network on source domain (α = 0.01)

    Loss = λ_ic * Loss_IC + λ_bc * Loss_BC + λ_physics * Loss_physics
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Separate optimizers for solution and physics networks
    optimizer_sol = optim.Adam(model.solution_net.parameters(), lr=lr_solution)
    optimizer_phys = optim.Adam(model.physics_net.parameters(), lr=lr_physics)

    # Learning rate schedulers
    scheduler_sol = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_sol, mode='min', factor=0.5, patience=500, verbose=True
    )
    scheduler_phys = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phys, mode='min', factor=0.5, patience=500, verbose=True
    )

    # Loss weights
    lambda_ic = 10.0
    lambda_bc = 10.0
    lambda_physics = 1.0

    # Training history
    history = {
        'epoch': [],
        'loss_total': [],
        'loss_ic': [],
        'loss_bc': [],
        'loss_physics': [],
        'test_l2_relative': [],
        'test_l2_absolute': [],
        'pde_residual': [],
        'time_per_epoch': []
    }

    # Get test data once
    x_test, t_test, u_test = data_generator.get_test_data()
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(model.device)
    t_test_t = torch.tensor(t_test, dtype=torch.float32).to(model.device)
    u_test_t = torch.tensor(u_test, dtype=torch.float32).to(model.device)

    print(f"\nStarting Heat equation source domain training...")
    print(f"  Epochs: {n_epochs}")
    print(f"  IC points: {n_ic}")
    print(f"  BC points: {n_bc}")
    print(f"  Collocation points: {n_collocation}")
    print(f"  Loss weights - IC: {lambda_ic}, BC: {lambda_bc}, Physics: {lambda_physics}")
    print(f"  Learning rates - Solution: {lr_solution}, Physics: {lr_physics}\n")

    best_test_error = float('inf')
    pbar = tqdm(range(n_epochs), desc="Training")

    for epoch in pbar:
        epoch_start = time.time()

        # Generate training data
        x_ic, t_ic, u_ic = data_generator.get_ic_data(n_ic)
        x_bc, t_bc, u_bc = data_generator.get_bc_data(n_bc)
        x_col, t_col = data_generator.get_collocation_data(n_collocation)

        # Convert to tensors
        x_ic_t = torch.tensor(x_ic, dtype=torch.float32, requires_grad=True).to(model.device)
        t_ic_t = torch.tensor(t_ic, dtype=torch.float32, requires_grad=True).to(model.device)
        u_ic_t = torch.tensor(u_ic, dtype=torch.float32).to(model.device)

        x_bc_t = torch.tensor(x_bc, dtype=torch.float32, requires_grad=True).to(model.device)
        t_bc_t = torch.tensor(t_bc, dtype=torch.float32, requires_grad=True).to(model.device)
        u_bc_t = torch.tensor(u_bc, dtype=torch.float32).to(model.device)

        x_col_t = torch.tensor(x_col, dtype=torch.float32, requires_grad=True).to(model.device)
        t_col_t = torch.tensor(t_col, dtype=torch.float32, requires_grad=True).to(model.device)

        # Zero gradients
        optimizer_sol.zero_grad()
        optimizer_phys.zero_grad()

        # 1. Initial condition loss
        u_ic_pred = model.solution_net(x_ic_t, t_ic_t)
        loss_ic = torch.mean((u_ic_pred - u_ic_t)**2)

        # 2. Boundary condition loss
        u_bc_pred = model.solution_net(x_bc_t, t_bc_t)
        loss_bc = torch.mean((u_bc_pred - u_bc_t)**2)

        # 3. Physics loss (using exact AD)
        result = model.forward_exact(x_col_t, t_col_t)
        loss_physics = result['physics_loss']

        # Total loss
        loss_total = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_physics * loss_physics

        # Backward pass
        loss_total.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.solution_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.physics_net.parameters(), max_norm=1.0)

        # Optimizer steps
        optimizer_sol.step()
        optimizer_phys.step()

        epoch_time = time.time() - epoch_start

        # Evaluation
        if epoch % log_interval == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                # Test error
                u_test_pred = model.solution_net(x_test_t, t_test_t)
                l2_absolute = torch.mean((u_test_pred - u_test_t)**2).sqrt().item()
                l2_relative = (l2_absolute / torch.mean(u_test_t**2).sqrt().item()) * 100

                # PDE residual
                pde_residual = model.compute_pde_residual(
                    x_test_t[:100], t_test_t[:100], use_fd=False
                )

                # Update schedulers
                scheduler_sol.step(loss_total)
                scheduler_phys.step(loss_physics)

                # Record history
                history['epoch'].append(epoch)
                history['loss_total'].append(loss_total.item())
                history['loss_ic'].append(loss_ic.item())
                history['loss_bc'].append(loss_bc.item())
                history['loss_physics'].append(loss_physics.item())
                history['test_l2_relative'].append(l2_relative)
                history['test_l2_absolute'].append(l2_absolute)
                history['pde_residual'].append(pde_residual)
                history['time_per_epoch'].append(epoch_time)

                # Save best model
                if l2_relative < best_test_error:
                    best_test_error = l2_relative
                    model.save_checkpoint(
                        save_dir / "best_model.pt",
                        epoch,
                        optimizer_sol,
                        optimizer_phys,
                        metadata={'test_l2_relative': l2_relative}
                    )

                pbar.set_postfix({
                    'Loss': f"{loss_total.item():.2e}",
                    'L2_rel': f"{l2_relative:.2f}%",
                    'PDE_res': f"{pde_residual:.2e}"
                })

    # Save final checkpoint
    model.save_checkpoint(
        save_dir / "final_model.pt",
        n_epochs,
        optimizer_sol,
        optimizer_phys,
        metadata={'test_l2_relative': history['test_l2_relative'][-1]}
    )

    # Save training history
    with open(save_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[OK] Training complete!")
    print(f"  Best test L2 relative error: {best_test_error:.4f}%")
    print(f"  Final test L2 relative error: {history['test_l2_relative'][-1]:.4f}%")
    print(f"  Final PDE residual: {history['pde_residual'][-1]:.2e}")

    return history


def plot_results(history, save_dir="results"):
    """Plot training history"""
    save_dir = Path(save_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves
    axes[0, 0].semilogy(history['epoch'], history['loss_total'], label='Total')
    axes[0, 0].semilogy(history['epoch'], history['loss_ic'], label='IC', alpha=0.7)
    axes[0, 0].semilogy(history['epoch'], history['loss_bc'], label='BC', alpha=0.7)
    axes[0, 0].semilogy(history['epoch'], history['loss_physics'], label='Physics', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Test error
    axes[0, 1].plot(history['epoch'], history['test_l2_relative'], color='red')
    axes[0, 1].axhline(y=1.0, color='green', linestyle='--', label='Target: 1%')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('L2 Relative Error (%)')
    axes[0, 1].set_title('Test Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # PDE residual
    axes[1, 0].semilogy(history['epoch'], history['pde_residual'], color='green')
    axes[1, 0].axhline(y=1e-4, color='red', linestyle='--', label='Target: 1e-4')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PDE Residual')
    axes[1, 0].set_title('Physics Equation Residual')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Time per epoch
    axes[1, 1].plot(history['epoch'], history['time_per_epoch'], color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].set_title('Time per Epoch')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    print(f"\n  Plots saved to {save_dir / 'training_curves.png'}")


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize model
    alpha_source = 0.01
    model = HeatPhysicsOperatorPINN(
        alpha=alpha_source,
        solution_hidden=[64, 64, 64, 64],
        physics_hidden=[32, 32],
    )

    # Data generator
    data_gen = HeatDataGenerator(alpha=alpha_source)

    # Train
    save_dir = Path("results/heat_experiment1_source")
    history = train_source_domain(
        model=model,
        data_generator=data_gen,
        n_epochs=5000,
        n_ic=100,
        n_bc=100,
        n_collocation=2000,
        lr_solution=1e-3,
        lr_physics=1e-3,
        log_interval=100,
        save_dir=save_dir
    )

    # Plot results
    plot_results(history, save_dir)

    print("\n" + "="*60)
    print("EXPERIMENT 1B: HEAT EQUATION SOURCE TRAINING - COMPLETE")
    print("="*60)
