"""
Physics Operator Network Architecture for Burgers 1D PDE

This implements a two-network architecture compatible with finite difference proxy methods:
1. Solution Network: u_θ(x,t) predicts solution u at spacetime coordinates
2. Physics Operator Network: F_φ(u, u_x, u_xx, ν) predicts temporal evolution u_t

Key difference from traditional PINNs:
- Traditional: Derivatives computed via AD and used DIRECTLY in loss → FD breaks gradient flow
- Our approach: Derivatives are INPUT FEATURES to differentiable F_φ → FD preserves gradient flow

This enables:
- Exact training: Use AD for accurate source training
- Proxy transfer: Use FD approximation with frozen F_φ for efficient target adaptation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import time


class SolutionNetwork(nn.Module):
    """Neural network predicting u(x,t) for Burgers equation"""

    def __init__(self, hidden_dims=[64, 64, 64], activation='tanh'):
        super().__init__()

        # Input: (x, t) - 2D spacetime coordinates
        # Output: u - scalar field value
        layers = []
        input_dim = 2

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sin':
                layers.append(torch.sin)
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

        # Xavier initialization
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
            u: Solution value (batch_size, 1)
        """
        xt = torch.cat([x, t], dim=1)
        return self.network(xt)


class PhysicsOperatorNetwork(nn.Module):
    """
    Neural network learning the PDE operator: (u, u_x, u_xx, ν) → u_t

    For Burgers equation: u_t = -u·u_x + ν·u_xx

    Instead of encoding this directly in loss, F_φ LEARNS the mapping from
    spatial configuration to temporal evolution.
    """

    def __init__(self, hidden_dims=[32, 32], activation='tanh'):
        super().__init__()

        # Input: (u, u_x, u_xx, ν) - 4 features
        # Output: u_t - temporal derivative
        layers = []
        input_dim = 4

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

    def forward(self, u, u_x, u_xx, nu):
        """
        Args:
            u: Solution value (batch_size, 1)
            u_x: First spatial derivative (batch_size, 1)
            u_xx: Second spatial derivative (batch_size, 1)
            nu: Viscosity parameter (batch_size, 1) or scalar
        Returns:
            u_t: Predicted temporal derivative (batch_size, 1)
        """
        # Ensure nu is broadcastable
        if isinstance(nu, float):
            nu = torch.full_like(u, nu)
        elif nu.dim() == 0:
            nu = nu.expand_as(u)

        features = torch.cat([u, u_x, u_xx, nu], dim=1)
        return self.network(features)


class BurgersPhysicsOperatorPINN:
    """
    Complete Physics-Informed Neural Network for Burgers equation using Physics Operator architecture.

    Two training modes:
    1. forward_exact: Source training with exact AD derivatives
    2. forward_proxy: Target transfer with FD approximation (frozen F_φ)
    """

    def __init__(
        self,
        nu: float = 0.01 / np.pi,
        solution_hidden=[64, 64, 64],
        physics_hidden=[32, 32],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.nu = nu
        self.device = device

        # Initialize networks
        self.solution_net = SolutionNetwork(hidden_dims=solution_hidden).to(device)
        self.physics_net = PhysicsOperatorNetwork(hidden_dims=physics_hidden).to(device)

        print(f"Initialized Physics Operator PINN:")
        print(f"  Solution network: {sum(p.numel() for p in self.solution_net.parameters())} params")
        print(f"  Physics network: {sum(p.numel() for p in self.physics_net.parameters())} params")
        print(f"  Device: {device}")
        print(f"  Viscosity ν: {nu:.6f}")

    def compute_derivatives_exact(self, x, t):
        """
        Compute spatial derivatives using exact automatic differentiation.
        Used during source training.

        Returns:
            u, u_x, u_xx, u_t (all with gradients)
        """
        # Ensure requires_grad
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)

        # Forward pass
        u = self.solution_net(x, t)

        # First-order derivatives
        u_x = torch.autograd.grad(
            outputs=u.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]

        u_t = torch.autograd.grad(
            outputs=u.sum(),
            inputs=t,
            create_graph=True,
            retain_graph=True
        )[0]

        # Second-order spatial derivative
        u_xx = torch.autograd.grad(
            outputs=u_x.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]

        return u, u_x, u_xx, u_t

    def compute_derivatives_fd(self, x, t, epsilon=1e-4):
        """
        Compute spatial derivatives using finite differences.
        Used during target transfer with frozen physics network.

        Critical: No gradients through finite difference computation,
        but gradients still flow through solution network and physics network!
        """
        # Solution at query point
        u = self.solution_net(x, t)

        # Perturbed evaluations (detached x to compute FD)
        with torch.no_grad():
            x_plus = x + epsilon
            x_minus = x - epsilon

        u_plus = self.solution_net(x_plus, t)
        u_minus = self.solution_net(x_minus, t)

        # Finite difference approximations
        u_x_fd = (u_plus - u_minus) / (2 * epsilon)
        u_xx_fd = (u_plus - 2*u + u_minus) / (epsilon**2)

        # Temporal derivative via FD (for ground truth during training)
        with torch.no_grad():
            t_plus = t + epsilon
        u_t_plus = self.solution_net(x, t_plus)
        u_t_fd = (u_t_plus - u) / epsilon

        return u, u_x_fd, u_xx_fd, u_t_fd

    def forward_exact(self, x, t, mode='train'):
        """
        Source domain training mode: exact automatic differentiation

        Returns:
            Dict with losses and predictions
        """
        u, u_x, u_xx, u_t_true = self.compute_derivatives_exact(x, t)

        # Physics operator prediction
        nu_tensor = torch.full_like(u, self.nu)
        u_t_pred = self.physics_net(u, u_x, u_xx, nu_tensor)

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
        u, u_x_fd, u_xx_fd, u_t_fd = self.compute_derivatives_fd(x, t, epsilon)

        # Frozen physics operator (no gradient updates)
        with torch.no_grad():
            physics_net_frozen = self.physics_net

        # Predict using frozen physics with FD inputs
        nu_tensor = torch.full_like(u, self.nu)
        u_t_pred = physics_net_frozen(u, u_x_fd, u_xx_fd, nu_tensor)

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
        Compute traditional PDE residual for validation: u_t + u·u_x - ν·u_xx

        Args:
            use_fd: If True, use finite differences; else use AD
        """
        # For PDE residual computation with AD, we need gradients enabled
        # even if we're inside a no_grad context
        if not use_fd:
            with torch.enable_grad():
                # Detach from previous computations and create new tensors with grad
                x = x.detach().clone().requires_grad_(True)
                t = t.detach().clone().requires_grad_(True)
                u, u_x, u_xx, u_t = self.compute_derivatives_exact(x, t)
                residual = u_t + u * u_x - self.nu * u_xx
                return torch.mean(residual**2).item()
        else:
            u, u_x, u_xx, u_t = self.compute_derivatives_fd(x, t, epsilon)
            residual = u_t + u * u_x - self.nu * u_xx
            return torch.mean(residual**2).item()

    def save_checkpoint(self, path, epoch, optimizer_sol, optimizer_phys, metadata=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'solution_net_state': self.solution_net.state_dict(),
            'physics_net_state': self.physics_net.state_dict(),
            'optimizer_sol_state': optimizer_sol.state_dict(),
            'optimizer_phys_state': optimizer_phys.state_dict(),
            'nu': self.nu,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

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
    # Quick test
    print("Testing Physics Operator Network architecture...")

    model = BurgersPhysicsOperatorPINN(nu=0.01/np.pi)

    # Test data
    x = torch.randn(100, 1, requires_grad=True).cuda()
    t = torch.randn(100, 1, requires_grad=True).cuda()

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

    print("\n[OK] Architecture test complete!")
