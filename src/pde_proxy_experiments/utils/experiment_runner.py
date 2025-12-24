"""
Modular experiment runner for reproducible multi-seed experiments
Supports parallel execution and statistical analysis
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import sys
sys.path.append('..')

from heat_physics_operator import HeatPhysicsOperatorPINN
from experiment_heat_source import HeatDataGenerator


def run_transfer_experiment(
    source_checkpoint_path,
    target_alpha,
    method='exact',  # 'exact' or 'proxy'
    seed=42,
    n_epochs=2000,
    lr_solution=1e-4,
    lr_physics=1e-4,
    epsilon=1e-4,
    freeze_physics=False,
    save_dir=None,
    verbose=True
):
    """
    Run a single transfer learning experiment

    Args:
        source_checkpoint_path: Path to source model checkpoint
        target_alpha: Target thermal diffusivity parameter
        method: 'exact' (AD) or 'proxy' (FD)
        seed: Random seed for reproducibility
        n_epochs: Number of training epochs
        lr_solution: Learning rate for solution network
        lr_physics: Learning rate for physics network (ignored if frozen)
        epsilon: FD epsilon for proxy method
        freeze_physics: Whether to freeze physics network
        save_dir: Directory to save results (None = don't save)
        verbose: Print progress

    Returns:
        results: Dict with metrics (best_l2, final_l2, times, etc.)
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize model and load source checkpoint
    model = HeatPhysicsOperatorPINN(alpha=target_alpha)
    model.load_checkpoint(source_checkpoint_path, load_optimizers=False)

    # Freeze physics if requested
    if freeze_physics or method == 'proxy':
        model.freeze_physics_network()

    # Setup optimizers
    optimizer_sol = optim.Adam(model.solution_net.parameters(), lr=lr_solution)
    optimizer_phys = None if freeze_physics else optim.Adam(
        model.physics_net.parameters(), lr=lr_physics
    )

    # Data generator
    data_gen = HeatDataGenerator(alpha=target_alpha)

    # Loss weights
    lambda_ic = 10.0
    lambda_bc = 10.0
    lambda_physics = 1.0

    # History tracking
    history = {
        'epoch': [], 'loss_total': [], 'loss_ic': [], 'loss_bc': [],
        'loss_physics': [], 'test_l2_relative': [], 'pde_residual': [],
        'time_per_epoch': []
    }

    # Test data (fixed for all epochs)
    x_test, t_test, u_test = data_gen.get_test_data()
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(model.device)
    t_test_t = torch.tensor(t_test, dtype=torch.float32).to(model.device)
    u_test_t = torch.tensor(u_test, dtype=torch.float32).to(model.device)

    if verbose:
        desc = f"{method.upper()} Transfer (seed={seed})"
        pbar = tqdm(range(n_epochs), desc=desc)
    else:
        pbar = range(n_epochs)

    best_error = float('inf')
    best_epoch = 0

    for epoch in pbar:
        start = time.time()

        # Generate training data
        x_ic, t_ic, u_ic = data_gen.get_ic_data(100)
        x_bc, t_bc, u_bc = data_gen.get_bc_data(100)
        x_col, t_col = data_gen.get_collocation_data(2000)

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
        if optimizer_phys:
            optimizer_phys.zero_grad()

        # IC loss
        u_ic_pred = model.solution_net(x_ic_t, t_ic_t)
        loss_ic = torch.mean((u_ic_pred - u_ic_t)**2)

        # BC loss
        u_bc_pred = model.solution_net(x_bc_t, t_bc_t)
        loss_bc = torch.mean((u_bc_pred - u_bc_t)**2)

        # Physics loss
        if method == 'exact':
            result = model.forward_exact(x_col_t, t_col_t)
        else:  # proxy
            result = model.forward_proxy(x_col_t, t_col_t, epsilon=epsilon)
        loss_physics = result['physics_loss']

        # Total loss
        loss_total = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_physics * loss_physics
        loss_total.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.solution_net.parameters(), max_norm=1.0)
        if optimizer_phys:
            torch.nn.utils.clip_grad_norm_(model.physics_net.parameters(), max_norm=1.0)

        # Optimizer steps
        optimizer_sol.step()
        if optimizer_phys:
            optimizer_phys.step()

        epoch_time = time.time() - start

        # Evaluation every 100 epochs
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                u_test_pred = model.solution_net(x_test_t, t_test_t)
                l2_absolute = torch.mean((u_test_pred - u_test_t)**2).sqrt().item()
                l2_relative = (l2_absolute / torch.mean(u_test_t**2).sqrt().item()) * 100

                use_fd = (method == 'proxy')
                pde_residual = model.compute_pde_residual(
                    x_test_t[:100], t_test_t[:100], use_fd=use_fd, epsilon=epsilon
                )

                history['epoch'].append(epoch)
                history['loss_total'].append(loss_total.item())
                history['loss_ic'].append(loss_ic.item())
                history['loss_bc'].append(loss_bc.item())
                history['loss_physics'].append(loss_physics.item())
                history['test_l2_relative'].append(l2_relative)
                history['pde_residual'].append(pde_residual)
                history['time_per_epoch'].append(epoch_time)

                if l2_relative < best_error:
                    best_error = l2_relative
                    best_epoch = epoch

                if verbose and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'Loss': f"{loss_total.item():.2e}",
                        'L2': f"{l2_relative:.2f}%",
                        'Best': f"{best_error:.2f}%@{best_epoch}"
                    })

    # Compile results
    results = {
        'seed': seed,
        'method': method,
        'target_alpha': target_alpha,
        'freeze_physics': freeze_physics,
        'epsilon': epsilon if method == 'proxy' else None,
        'n_epochs': n_epochs,
        'best_l2_error': best_error,
        'best_epoch': best_epoch,
        'final_l2_error': history['test_l2_relative'][-1],
        'avg_time_per_epoch': float(np.mean(history['time_per_epoch'][1:])),  # Exclude first
        'history': history
    }

    # Save if requested
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        with open(save_dir / f"results_seed{seed}.json", 'w') as f:
            # Save without full history for summary
            summary = {k: v for k, v in results.items() if k != 'history'}
            json.dump(summary, f, indent=2)

        with open(save_dir / f"history_seed{seed}.json", 'w') as f:
            json.dump(history, f, indent=2)

    return results


def run_multi_seed_comparison(
    source_checkpoint_path,
    target_alpha,
    seeds,
    n_epochs=2000,
    epsilon=1e-4,
    save_dir='results/multi_seed_validation',
    verbose=True
):
    """
    Run both exact and proxy methods across multiple seeds

    Returns:
        exact_results: List of results for exact AD method
        proxy_results: List of results for FD proxy method
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    exact_results = []
    proxy_results = []

    for seed in seeds:
        if verbose:
            print(f"\n{'='*70}")
            print(f"SEED {seed}: Running Exact AD Transfer")
            print(f"{'='*70}")

        # Exact AD transfer
        result_exact = run_transfer_experiment(
            source_checkpoint_path=source_checkpoint_path,
            target_alpha=target_alpha,
            method='exact',
            seed=seed,
            n_epochs=n_epochs,
            freeze_physics=False,
            save_dir=save_dir / 'exact',
            verbose=verbose
        )
        exact_results.append(result_exact)

        if verbose:
            print(f"\n{'='*70}")
            print(f"SEED {seed}: Running FD Proxy Transfer")
            print(f"{'='*70}")

        # FD Proxy transfer
        result_proxy = run_transfer_experiment(
            source_checkpoint_path=source_checkpoint_path,
            target_alpha=target_alpha,
            method='proxy',
            seed=seed,
            n_epochs=n_epochs,
            epsilon=epsilon,
            freeze_physics=True,
            save_dir=save_dir / 'proxy',
            verbose=verbose
        )
        proxy_results.append(result_proxy)

        if verbose:
            print(f"\nSeed {seed} Summary:")
            print(f"  Exact AD:  Best={result_exact['best_l2_error']:.2f}%, "
                  f"Final={result_exact['final_l2_error']:.2f}%")
            print(f"  FD Proxy: Best={result_proxy['best_l2_error']:.2f}%, "
                  f"Final={result_proxy['final_l2_error']:.2f}%")
            improvement = ((result_exact['best_l2_error'] - result_proxy['best_l2_error'])
                          / result_exact['best_l2_error'] * 100)
            print(f"  Improvement: {improvement:+.1f}%")

    # Save aggregated results
    summary = {
        'seeds': seeds,
        'target_alpha': target_alpha,
        'n_epochs': n_epochs,
        'exact_results': [
            {k: v for k, v in r.items() if k != 'history'}
            for r in exact_results
        ],
        'proxy_results': [
            {k: v for k, v in r.items() if k != 'history'}
            for r in proxy_results
        ]
    }

    with open(save_dir / 'aggregated_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return exact_results, proxy_results


if __name__ == "__main__":
    # Quick test
    print("Testing experiment runner...")

    source_checkpoint = "../results/heat_experiment1_source/best_model.pt"

    result = run_transfer_experiment(
        source_checkpoint_path=source_checkpoint,
        target_alpha=0.02,
        method='exact',
        seed=42,
        n_epochs=100,
        verbose=True
    )

    print(f"\nTest Result: Best L2 = {result['best_l2_error']:.2f}%")
