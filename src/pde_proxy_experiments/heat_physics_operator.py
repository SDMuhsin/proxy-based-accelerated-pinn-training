"""
Physics Operator Network Architecture for 1D Heat Equation

Heat equation: u_t = α·u_xx

Where:
- u(x,t): temperature field
- α: thermal diffusivity (parameterizable for transfer learning)
- Domain: x ∈ [-1, 1], t ∈ [0, 1]

Initial condition: u(x,0) = sin(πx)
Boundary conditions: u(±1,t) = 0

Analytical solution: u(x,t) = exp(-α·π²·t)·sin(πx)

This is MUCH simpler than Burgers (linear vs nonlinear) and has exact solution for validation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional


class SolutionNetwork(nn.Module):
    """Neural network predicting u(x,t) for Heat equation"""

    def __init__(self, hidden_dims=[64, 64, 64, 64], activation='tanh'):
        super().__init__()

        layers = []
        input_dim = 2  # (x, t)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        """
        Args:
            x: Spatial coordinate (batch_size, 1)
            t: Temporal coordinate (batch_size, 1)
        Returns:
            u: Temperature (batch_size, 1)
        """
        xt = torch.cat([x, t], dim=1)
        return self.network(xt)


class PhysicsOperatorNetwork(nn.Module):
    """
    Neural network learning the Heat equation operator: (u, u_xx, α) → u_t

    For Heat equation: u_t = α·u_xx

    The physics network learns this linear relationship from data.
    """

    def __init__(self, hidden_dims=[32, 32], activation='tanh'):
        super().__init__()

        layers = []
        input_dim = 3  # (u, u_xx, α)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, u, u_xx, alpha):
        """
        Args:
            u: Solution value (batch_size, 1)
            u_xx: Second spatial derivative (batch_size, 1)
            alpha: Thermal diffusivity (batch_size, 1) or scalar
        Returns:
            u_t: Predicted temporal derivative (batch_size, 1)
        """
        if isinstance(alpha, float):
            alpha = torch.full_like(u, alpha)
        elif alpha.dim() == 0:
            alpha = alpha.expand_as(u)

        features = torch.cat([u, u_xx, alpha], dim=1)
        return self.network(features)


class HeatPhysicsOperatorPINN:
    """
    Complete Physics-Informed Neural Network for 1D Heat equation.

    Two training modes:
    1. forward_exact: Source training with exact AD derivatives
    2. forward_proxy: Target transfer with FD approximation (frozen F_φ)
    """

    def __init__(
        self,
        alpha: float = 0.01,
        solution_hidden=[64, 64, 64, 64],
        physics_hidden=[32, 32],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.alpha = alpha
        self.device = device

        self.solution_net = SolutionNetwork(hidden_dims=solution_hidden).to(device)
        self.physics_net = PhysicsOperatorNetwork(hidden_dims=physics_hidden).to(device)

        print(f"Initialized Heat Equation Physics Operator PINN:")
        print(f"  Solution network: {sum(p.numel() for p in self.solution_net.parameters())} params")
        print(f"  Physics network: {sum(p.numel() for p in self.physics_net.parameters())} params")
        print(f"  Device: {device}")
        print(f"  Thermal diffusivity α: {alpha:.6f}")

    def analytical_solution(self, x, t):
        """
        Exact solution: u(x,t) = exp(-α·π²·t)·sin(πx)

        Useful for validation and comparison.
        """
        if isinstance(x, torch.Tensor):
            return torch.exp(-self.alpha * np.pi**2 * t) * torch.sin(np.pi * x)
        else:
            return np.exp(-self.alpha * np.pi**2 * t) * np.sin(np.pi * x)

    def compute_derivatives_exact(self, x, t):
        """
        Compute spatial derivatives using exact automatic differentiation.
        Used during source training.

        Returns:
            u, u_xx, u_t (all with gradients)
        """
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)

        u = self.solution_net(x, t)

        # First spatial derivative
        u_x = torch.autograd.grad(
            outputs=u.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]

        # Second spatial derivative
        u_xx = torch.autograd.grad(
            outputs=u_x.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]

        # Temporal derivative
        u_t = torch.autograd.grad(
            outputs=u.sum(),
            inputs=t,
            create_graph=True,
            retain_graph=True
        )[0]

        return u, u_xx, u_t

    def compute_derivatives_fd(self, x, t, epsilon=1e-4):
        """
        Compute spatial derivatives using finite differences.
        Used during target transfer with frozen physics network.
        """
        u = self.solution_net(x, t)

        # Spatial FD
        with torch.no_grad():
            x_plus = x + epsilon
            x_minus = x - epsilon

        u_plus = self.solution_net(x_plus, t)
        u_minus = self.solution_net(x_minus, t)
        u_xx_fd = (u_plus - 2*u + u_minus) / (epsilon**2)

        # Temporal FD (for ground truth during training)
        with torch.no_grad():
            t_plus = t + epsilon
        u_t_plus = self.solution_net(x, t_plus)
        u_t_fd = (u_t_plus - u) / epsilon

        return u, u_xx_fd, u_t_fd

    def forward_exact(self, x, t):
        """
        Source domain training mode: exact automatic differentiation

        Returns:
            Dict with losses and predictions
        """
        u, u_xx, u_t_true = self.compute_derivatives_exact(x, t)

        # Physics operator prediction
        alpha_tensor = torch.full_like(u, self.alpha)
        u_t_pred = self.physics_net(u, u_xx, alpha_tensor)

        # Physics loss: F_φ should predict true temporal evolution
        physics_loss = torch.mean((u_t_pred - u_t_true)**2)

        return {
            'u': u,
            'u_t_true': u_t_true,
            'u_t_pred': u_t_pred,
            'physics_loss': physics_loss
        }

    def forward_proxy(self, x, t, epsilon=1e-4):
        """
        Target domain transfer mode: finite difference approximation with frozen physics

        Returns:
            Dict with losses and predictions
        """
        u, u_xx_fd, u_t_fd = self.compute_derivatives_fd(x, t, epsilon)

        # Frozen physics operator (no gradient updates)
        with torch.no_grad():
            physics_net_frozen = self.physics_net

        # Predict using frozen physics with FD inputs
        alpha_tensor = torch.full_like(u, self.alpha)
        u_t_pred = physics_net_frozen(u, u_xx_fd, alpha_tensor)

        # Physics loss (for regularization, physics net frozen)
        physics_loss = torch.mean((u_t_pred - u_t_fd)**2)

        return {
            'u': u,
            'u_t_fd': u_t_fd,
            'u_t_pred': u_t_pred,
            'physics_loss': physics_loss
        }

    def compute_pde_residual(self, x, t, use_fd=False, epsilon=1e-4):
        """
        Compute PDE residual for validation: u_t - α·u_xx

        Args:
            use_fd: If True, use finite differences; else use AD
        """
        if not use_fd:
            with torch.enable_grad():
                x = x.detach().clone().requires_grad_(True)
                t = t.detach().clone().requires_grad_(True)
                u, u_xx, u_t = self.compute_derivatives_exact(x, t)
                residual = u_t - self.alpha * u_xx
                return torch.mean(residual**2).item()
        else:
            u, u_xx, u_t = self.compute_derivatives_fd(x, t, epsilon)
            residual = u_t - self.alpha * u_xx
            return torch.mean(residual**2).item()

    def save_checkpoint(self, path, epoch, optimizer_sol, optimizer_phys, metadata=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'solution_net_state': self.solution_net.state_dict(),
            'physics_net_state': self.physics_net.state_dict(),
            'optimizer_sol_state': optimizer_sol.state_dict() if optimizer_sol else None,
            'optimizer_phys_state': optimizer_phys.state_dict() if optimizer_phys else None,
            'alpha': self.alpha,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, load_optimizers=True, optimizer_sol=None, optimizer_phys=None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.solution_net.load_state_dict(checkpoint['solution_net_state'])
        self.physics_net.load_state_dict(checkpoint['physics_net_state'])

        if load_optimizers and optimizer_sol and optimizer_phys:
            optimizer_sol.load_state_dict(checkpoint['optimizer_sol_state'])
            optimizer_phys.load_state_dict(checkpoint['optimizer_phys_state'])

        print(f"Checkpoint loaded: {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Metadata: {checkpoint.get('metadata', {})}")

        return checkpoint

    def freeze_physics_network(self):
        """Freeze physics network for transfer learning"""
        for param in self.physics_net.parameters():
            param.requires_grad = False
        print("Physics network frozen for transfer learning")

    def unfreeze_physics_network(self):
        """Unfreeze physics network"""
        for param in self.physics_net.parameters():
            param.requires_grad = True
        print("Physics network unfrozen")


if __name__ == "__main__":
    import time

    print("Testing Heat Equation Physics Operator Network architecture...")

    model = HeatPhysicsOperatorPINN(alpha=0.01)

    # Test data
    x = torch.randn(100, 1, requires_grad=True).to(model.device)
    t = torch.randn(100, 1, requires_grad=True).abs().to(model.device)  # t >= 0

    print("\n1. Testing exact forward pass...")
    start = time.time()
    result_exact = model.forward_exact(x, t)
    print(f"   Physics loss: {result_exact['physics_loss'].item():.6f}")
    print(f"   Time: {time.time() - start:.4f}s")

    print("\n2. Testing proxy forward pass...")
    model.freeze_physics_network()
    start = time.time()
    result_proxy = model.forward_proxy(x, t)
    print(f"   Physics loss: {result_proxy['physics_loss'].item():.6f}")
    print(f"   Time: {time.time() - start:.4f}s")

    print("\n3. Testing PDE residual...")
    residual_exact = model.compute_pde_residual(x[:10], t[:10], use_fd=False)
    residual_fd = model.compute_pde_residual(x[:10], t[:10], use_fd=True)
    print(f"   Exact AD residual: {residual_exact:.6f}")
    print(f"   FD residual: {residual_fd:.6f}")

    print("\n4. Testing analytical solution...")
    x_test = torch.tensor([[0.5]], dtype=torch.float32)
    t_test = torch.tensor([[0.1]], dtype=torch.float32)
    u_analytical = model.analytical_solution(x_test, t_test)
    print(f"   u(0.5, 0.1) = {u_analytical.item():.6f}")
    print(f"   Expected: exp(-{model.alpha}·π²·0.1)·sin(π·0.5) = {np.exp(-model.alpha*np.pi**2*0.1)*np.sin(np.pi*0.5):.6f}")

    print("\n[OK] Architecture test complete!")
