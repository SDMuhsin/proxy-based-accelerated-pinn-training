"""
Fast Approximate Autograd for PINNs

Goal: Replace torch.autograd.grad() with faster approximations
that maintain acceptable accuracy.

Strategy: Use finite differences with optimizations:
1. Vectorized computation (no loops)
2. Forward differences (1 extra forward pass per dimension)
3. Adaptive step size
4. Optional caching/reuse

This directly targets the 40% autograd bottleneck.
"""

import torch
import torch.nn as nn


class FastAutogradConfig:
    """Configuration for fast autograd approximation"""

    def __init__(self,
                 method='forward',           # 'forward' or 'central'
                 step_size=1e-5,            # Finite difference step size
                 adaptive_step=False,        # Adapt step size to input magnitude
                 vectorized=True,            # Use vectorized computation
                 ):
        self.method = method
        self.step_size = step_size
        self.adaptive_step = adaptive_step
        self.vectorized = vectorized


class FastAutograd:
    """
    Fast approximate gradient computation using finite differences.

    Key optimizations:
    1. Vectorized batched computation (no Python loops)
    2. Forward differences only (cheaper than central)
    3. Single concatenated forward pass for all perturbations
    4. Reuse predictions where possible
    """

    def __init__(self, config=None):
        self.config = config or FastAutogradConfig()

    @staticmethod
    def compute_gradient_forward_vectorized(model, xt, h=1e-5):
        """
        Compute gradients using vectorized forward differences.

        Standard approach (SLOW):
        - For each dimension i:
          - xt_plus = xt.clone(); xt_plus[:, i] += h
          - u_plus = model(xt_plus)
          - grad_i = (u_plus - u) / h
        - Total: n_dims + 1 forward passes

        Vectorized approach (FAST):
        - Stack all perturbations: [xt, xt+h_0, xt+h_1, ..., xt+h_n]
        - Single batched forward pass
        - Compute all gradients at once
        - Total: 1 forward pass (on larger batch)

        Args:
            model: Neural network (solution_u)
            xt: Input [batch_size, n_dims]
            h: Step size

        Returns:
            u: Predictions [batch_size, 1]
            grads: Gradients [batch_size, n_dims]
        """
        batch_size, n_dims = xt.shape
        device = xt.device

        # Create perturbation matrix: identity * h
        # Shape: [n_dims, n_dims] - each row is a unit vector scaled by h
        perturbations = torch.eye(n_dims, device=device) * h  # [n_dims, n_dims]

        # Expand to batch size
        # xt: [batch_size, n_dims]
        # We want: [batch_size, n_dims+1, n_dims]
        #   First slice: original xt
        #   Next n_dims slices: xt perturbed in each dimension

        # Stack: [batch_size, n_dims, n_dims] where xt is broadcast
        xt_expanded = xt.unsqueeze(1)  # [batch_size, 1, n_dims]
        perturbations_expanded = perturbations.unsqueeze(0)  # [1, n_dims, n_dims]

        # All perturbed versions
        xt_perturbed = xt_expanded + perturbations_expanded  # [batch_size, n_dims, n_dims]

        # Add original
        xt_all = torch.cat([xt.unsqueeze(1), xt_perturbed], dim=1)  # [batch_size, n_dims+1, n_dims]

        # Reshape for batched forward pass
        xt_flat = xt_all.reshape(-1, n_dims)  # [batch_size * (n_dims+1), n_dims]

        # Single batched forward pass
        with torch.no_grad():
            u_flat = model(xt_flat)  # [batch_size * (n_dims+1), 1]

        # Reshape back
        u_all = u_flat.reshape(batch_size, n_dims + 1, -1)  # [batch_size, n_dims+1, out_dim]

        # Extract base predictions and perturbed predictions
        u = u_all[:, 0, :]  # [batch_size, out_dim]
        u_perturbed = u_all[:, 1:, :]  # [batch_size, n_dims, out_dim]

        # Compute gradients via finite differences
        grads = (u_perturbed - u.unsqueeze(1)) / h  # [batch_size, n_dims, out_dim]

        # If output is scalar (out_dim=1), squeeze
        if grads.shape[-1] == 1:
            grads = grads.squeeze(-1)  # [batch_size, n_dims]

        return u, grads

    @staticmethod
    def compute_gradient_forward_separate(model, xt, h=1e-5):
        """
        Compute time and space gradients separately (PINN-specific optimization).

        For PINNs, we need:
        - u_t: gradient w.r.t. time (last dimension)
        - u_x: gradient w.r.t. features (all other dimensions)

        We can optimize by only perturbing what we need.

        Args:
            model: Neural network
            xt: Input [batch_size, 17] (16 features + 1 time)
            h: Step size

        Returns:
            u: Predictions [batch_size, 1]
            u_x: Spatial gradients [batch_size, 16]
            u_t: Time gradient [batch_size, 1]
        """
        batch_size, n_dims = xt.shape
        n_features = n_dims - 1  # Last dim is time
        device = xt.device

        # Base prediction
        u = model(xt)  # [batch_size, 1]

        # Perturb time (last dimension)
        xt_t_plus = xt.clone()
        xt_t_plus[:, -1] += h
        u_t_plus = model(xt_t_plus)
        u_t = (u_t_plus - u) / h  # [batch_size, 1]

        # Perturb features (vectorized)
        # Create all feature perturbations at once
        perturbations = torch.eye(n_features, device=device) * h  # [n_features, n_features]

        # Expand and apply
        xt_expanded = xt[:, :-1].unsqueeze(1)  # [batch_size, 1, n_features]
        xt_perturbed = xt_expanded + perturbations.unsqueeze(0)  # [batch_size, n_features, n_features]

        # Add time dimension back
        time = xt[:, -1:].unsqueeze(1).expand(-1, n_features, -1)  # [batch_size, n_features, 1]
        xt_perturbed_full = torch.cat([xt_perturbed, time], dim=2)  # [batch_size, n_features, n_dims]

        # Batched forward pass
        xt_perturbed_flat = xt_perturbed_full.reshape(-1, n_dims)
        u_perturbed_flat = model(xt_perturbed_flat)
        u_perturbed = u_perturbed_flat.reshape(batch_size, n_features, -1)

        # Compute spatial gradients
        u_x = (u_perturbed - u.unsqueeze(1)) / h  # [batch_size, n_features, 1]
        u_x = u_x.squeeze(-1)  # [batch_size, n_features]

        return u, u_x, u_t

    def compute_pinn_gradients(self, model, xt, separate=True):
        """
        Main interface for computing PINN gradients.

        Args:
            model: solution_u network
            xt: Input [batch_size, 17]
            separate: If True, compute time/space separately (faster for PINNs)

        Returns:
            u: Predictions
            u_x: Spatial gradients
            u_t: Time gradient
        """
        h = self.config.step_size

        if separate:
            return self.compute_gradient_forward_separate(model, xt, h=h)
        else:
            u, grads = self.compute_gradient_forward_vectorized(model, xt, h=h)
            u_x = grads[:, :-1]  # All but last
            u_t = grads[:, -1:]  # Last dim
            return u, u_x, u_t


class FastPINN(nn.Module):
    """
    PINN using fast approximate autograd.

    Drop-in replacement for standard PINN forward pass.
    """

    def __init__(self, solution_u, dynamical_F, fast_autograd=None):
        super().__init__()
        self.solution_u = solution_u
        self.dynamical_F = dynamical_F
        self.fast_autograd = fast_autograd or FastAutograd()

    def forward(self, xt):
        """
        Forward pass using fast approximate gradients.

        Args:
            xt: Input [batch_size, 17]

        Returns:
            u: Predictions [batch_size, 1]
            f: Physics residuals [batch_size, 1]
        """
        # Compute gradients using fast autograd
        u, u_x, u_t = self.fast_autograd.compute_pinn_gradients(
            self.solution_u, xt, separate=True
        )

        # Compute physics residual (this part stays the same)
        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))
        f = u_t - F

        return u, f

    def forward_with_grad(self, xt):
        """
        Forward pass that also returns gradients (for compatibility).
        """
        u, u_x, u_t = self.fast_autograd.compute_pinn_gradients(
            self.solution_u, xt, separate=True
        )
        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))
        f = u_t - F
        return u, f, u_x, u_t


# ==============================================================================
# PERFORMANCE COMPARISON UTILITIES
# ==============================================================================

class AutogradBenchmark:
    """Compare standard autograd vs fast autograd"""

    @staticmethod
    def standard_autograd(model, xt):
        """Standard torch.autograd.grad() approach"""
        xt.requires_grad = True
        x = xt[:, :-1]
        t = xt[:, -1:]

        u = model(torch.cat((x, t), dim=1))

        u_t = torch.autograd.grad(u.sum(), t,
                                  create_graph=True,
                                  only_inputs=True,
                                  allow_unused=True)[0]
        u_x = torch.autograd.grad(u.sum(), x,
                                  create_graph=True,
                                  only_inputs=True,
                                  allow_unused=True)[0]

        return u, u_x, u_t

    @staticmethod
    def benchmark(model, batch_size=64, n_dims=17, n_runs=100, device='cuda'):
        """
        Benchmark standard vs fast autograd.

        Returns:
            results: Dict with timing and speedup info
        """
        import time

        # Create dummy input
        xt = torch.randn(batch_size, n_dims, device=device)

        # Warmup
        for _ in range(10):
            u1, ux1, ut1 = AutogradBenchmark.standard_autograd(model, xt.clone())

        fast_ag = FastAutograd()
        for _ in range(10):
            u2, ux2, ut2 = fast_ag.compute_pinn_gradients(model, xt.clone(), separate=True)

        # Benchmark standard autograd
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(n_runs):
            u1, ux1, ut1 = AutogradBenchmark.standard_autograd(model, xt.clone())
        torch.cuda.synchronize() if device == 'cuda' else None
        time_standard = (time.time() - start) / n_runs

        # Benchmark fast autograd
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(n_runs):
            u2, ux2, ut2 = fast_ag.compute_pinn_gradients(model, xt.clone(), separate=True)
        torch.cuda.synchronize() if device == 'cuda' else None
        time_fast = (time.time() - start) / n_runs

        # Compare accuracy
        with torch.no_grad():
            u_diff = (u1 - u2).abs().mean().item()
            ux_diff = (ux1 - ux2).abs().mean().item()
            ut_diff = (ut1 - ut2).abs().mean().item()

        results = {
            'time_standard': time_standard * 1000,  # ms
            'time_fast': time_fast * 1000,  # ms
            'speedup': time_standard / time_fast,
            'accuracy': {
                'u_mae': u_diff,
                'u_x_mae': ux_diff,
                'u_t_mae': ut_diff,
            }
        }

        return results


# ==============================================================================
# INTEGRATION WITH EXISTING PINN
# ==============================================================================

def convert_pinn_to_fast(pinn_model, fast_autograd_config=None):
    """
    Convert existing PINN to use fast autograd.

    Args:
        pinn_model: Existing PINN model
        fast_autograd_config: Optional config for fast autograd

    Returns:
        fast_pinn: PINN using fast autograd
    """
    config = fast_autograd_config or FastAutogradConfig()
    fast_ag = FastAutograd(config)

    fast_pinn = FastPINN(
        solution_u=pinn_model.solution_u,
        dynamical_F=pinn_model.dynamical_F,
        fast_autograd=fast_ag
    )

    # Copy other attributes
    fast_pinn.loss_func = pinn_model.loss_func
    fast_pinn.relu = pinn_model.relu
    fast_pinn.alpha = pinn_model.alpha
    fast_pinn.beta = pinn_model.beta

    return fast_pinn


# ==============================================================================
# EXPECTED PERFORMANCE
# ==============================================================================

"""
Performance Analysis:

Standard autograd (create_graph=True):
- Time gradient: 1 backward pass with graph building
- Space gradients (16 dims): 1 backward pass with graph building
- Total: 2 expensive backward passes
- Estimated time: ~15-20ms per batch (based on profiling)

Fast autograd (vectorized finite differences):
- Time gradient: 1 forward pass (perturbed input)
- Space gradients: 16 forward passes (batched together)
- Total: 17 forward passes (but batched into 1-2 calls)
- Estimated time: ~3-5ms per batch

Expected speedup on autograd: 3-5x
Since autograd is 40% of total time:
- New time: 0.6 + 0.4/4 = 0.7 (using 4x speedup estimate)
- Total speedup: 1/0.7 = 1.43x

Conservative estimate: 1.3-1.5x total speedup
Optimistic estimate: 1.5-1.8x total speedup

Accuracy impact:
- Finite differences with h=1e-5 are accurate to ~1e-4 to 1e-5
- Should maintain acceptable physics constraint accuracy
- May see small degradation in final MAE (< 5-10%)

Key advantages:
1. No overhead from graph building
2. Forward passes can use FP16 (if desired)
3. Vectorized = better GPU utilization
4. No special requirements (works everywhere)

Key risks:
1. Numerical instability if h too small
2. Inaccuracy if h too large
3. Need to tune h for this specific problem
4. May hurt convergence

This is our BEST CHANCE at real speedup.
"""
