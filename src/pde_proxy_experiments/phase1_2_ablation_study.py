"""
Phase 1.2: Core Ablation Study

Tests 5 key variants to identify which components are critical:
1. Full method (proxy): FD approximation + frozen physics [BASELINE]
2. Exact AD + frozen physics: Tests if freezing alone helps
3. FD proxy + trainable physics: Tests if FD approximation alone helps
4. FD proxy + frozen physics + frozen solution (0 trainable): Tests if any learning needed
5. Random initialization (no transfer): Tests value of transfer learning

Expected time: ~35 minutes (5 variants × 3 seeds × 2.5 min)
Uses 3 seeds per variant for statistical confidence
"""

import sys
sys.path.append('.')

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from heat_physics_operator import HeatPhysicsOperatorPINN
from experiment_heat_source import HeatDataGenerator

def run_ablation_variant(
    variant_name,
    source_checkpoint_path,
    target_alpha,
    use_fd,  # True = FD proxy, False = exact AD
    freeze_physics,
    freeze_solution,
    use_transfer,  # False = random init
    seed=42,
    n_epochs=2000,
    epsilon=1e-4,
    save_dir=None
):
    """Run a single ablation variant"""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize model
    model = HeatPhysicsOperatorPINN(alpha=target_alpha)

    if use_transfer:
        model.load_checkpoint(source_checkpoint_path, load_optimizers=False)
    # else: random initialization (default)

    # Freeze networks as specified
    if freeze_physics:
        model.freeze_physics_network()
    if freeze_solution:
        for param in model.solution_net.parameters():
            param.requires_grad = False

    # Setup optimizers (only for trainable networks)
    optimizer_sol = None if freeze_solution else optim.Adam(
        model.solution_net.parameters(), lr=1e-4
    )
    optimizer_phys = None if freeze_physics else optim.Adam(
        model.physics_net.parameters(), lr=1e-4
    )

    # Data generator
    data_gen = HeatDataGenerator(alpha=target_alpha)

    # Loss weights
    lambda_ic = 10.0
    lambda_bc = 10.0
    lambda_physics = 1.0

    # Test data
    x_test, t_test, u_test = data_gen.get_test_data()
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(model.device)
    t_test_t = torch.tensor(t_test, dtype=torch.float32).to(model.device)
    u_test_t = torch.tensor(u_test, dtype=torch.float32).to(model.device)

    best_error = float('inf')
    best_epoch = 0

    pbar = tqdm(range(n_epochs), desc=f"{variant_name} (seed={seed})")

    for epoch in pbar:
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
        if optimizer_sol:
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
        if use_fd:
            result = model.forward_proxy(x_col_t, t_col_t, epsilon=epsilon)
        else:
            result = model.forward_exact(x_col_t, t_col_t)
        loss_physics = result['physics_loss']

        # Total loss
        loss_total = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_physics * loss_physics

        # Only backprop if there are trainable parameters
        if optimizer_sol or optimizer_phys:
            loss_total.backward()

            # Gradient clipping
            if optimizer_sol:
                torch.nn.utils.clip_grad_norm_(model.solution_net.parameters(), max_norm=1.0)
            if optimizer_phys:
                torch.nn.utils.clip_grad_norm_(model.physics_net.parameters(), max_norm=1.0)

            # Optimizer steps
            if optimizer_sol:
                optimizer_sol.step()
            if optimizer_phys:
                optimizer_phys.step()

        # Evaluation every 100 epochs
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                u_test_pred = model.solution_net(x_test_t, t_test_t)
                l2_absolute = torch.mean((u_test_pred - u_test_t)**2).sqrt().item()
                l2_relative = (l2_absolute / torch.mean(u_test_t**2).sqrt().item()) * 100

                if l2_relative < best_error:
                    best_error = l2_relative
                    best_epoch = epoch

                pbar.set_postfix({
                    'Loss': f"{loss_total.item():.2e}",
                    'L2': f"{l2_relative:.2f}%",
                    'Best': f"{best_error:.2f}%@{best_epoch}"
                })

    results = {
        'variant': variant_name,
        'seed': seed,
        'use_fd': use_fd,
        'freeze_physics': freeze_physics,
        'freeze_solution': freeze_solution,
        'use_transfer': use_transfer,
        'best_l2_error': best_error,
        'best_epoch': best_epoch
    }

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        with open(save_dir / f'results_seed{seed}.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results


def run_ablation_study(
    source_checkpoint_path,
    target_alpha,
    seeds=[42, 123, 456],  # 3 seeds per variant
    n_epochs=2000,
    epsilon=1e-4,
    save_dir='results/phase1_2_ablation'
):
    """Run full ablation study with 5 variants"""

    variants = [
        {
            'name': 'Full Method (Proxy + Frozen Physics)',
            'use_fd': True,
            'freeze_physics': True,
            'freeze_solution': False,
            'use_transfer': True
        },
        {
            'name': 'Exact AD + Frozen Physics',
            'use_fd': False,
            'freeze_physics': True,
            'freeze_solution': False,
            'use_transfer': True
        },
        {
            'name': 'Proxy + Trainable Physics',
            'use_fd': True,
            'freeze_physics': False,
            'freeze_solution': False,
            'use_transfer': True
        },
        {
            'name': 'Proxy + All Frozen (No Learning)',
            'use_fd': True,
            'freeze_physics': True,
            'freeze_solution': True,
            'use_transfer': True
        },
        {
            'name': 'No Transfer (Random Init)',
            'use_fd': True,
            'freeze_physics': True,
            'freeze_solution': False,
            'use_transfer': False
        }
    ]

    print("="*80)
    print("PHASE 1.2: ABLATION STUDY")
    print("="*80)
    print(f"\nRunning {len(variants)} variants × {len(seeds)} seeds = {len(variants) * len(seeds)} experiments")
    print(f"Expected time: ~{len(variants) * len(seeds) * 1.5:.0f} minutes")
    print()

    all_results = {}

    for variant in variants:
        print(f"\n{'='*80}")
        print(f"VARIANT: {variant['name']}")
        print(f"{'='*80}")

        variant_results = []
        for seed in seeds:
            result = run_ablation_variant(
                variant_name=variant['name'],
                source_checkpoint_path=source_checkpoint_path,
                target_alpha=target_alpha,
                use_fd=variant['use_fd'],
                freeze_physics=variant['freeze_physics'],
                freeze_solution=variant['freeze_solution'],
                use_transfer=variant['use_transfer'],
                seed=seed,
                n_epochs=n_epochs,
                epsilon=epsilon,
                save_dir=Path(save_dir) / variant['name'].replace(' ', '_')
            )
            variant_results.append(result)

            print(f"  Seed {seed}: Best L2 = {result['best_l2_error']:.2f}%")

        # Aggregate results
        best_errors = [r['best_l2_error'] for r in variant_results]
        all_results[variant['name']] = best_errors

        print(f"  Mean ± Std: {np.mean(best_errors):.2f} ± {np.std(best_errors):.2f}%")

    # Summary analysis
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)

    for variant_name, errors in all_results.items():
        print(f"\n{variant_name}:")
        print(f"  Mean: {np.mean(errors):.2f}%")
        print(f"  Std:  {np.std(errors):.2f}%")
        print(f"  Individual: {[f'{e:.2f}%' for e in errors]}")

    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    summary = {
        'variants': variants,
        'seeds': seeds,
        'results': all_results,
        'statistics': {
            variant_name: {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'min': float(np.min(errors)),
                'max': float(np.max(errors)),
                'values': [float(e) for e in errors]
            }
            for variant_name, errors in all_results.items()
        }
    }

    with open(save_dir / 'ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {save_dir}/ablation_summary.json")

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    baseline = np.mean(all_results['Full Method (Proxy + Frozen Physics)'])
    print(f"\nBaseline (Full Method): {baseline:.2f}%")

    for variant_name, errors in all_results.items():
        if variant_name == 'Full Method (Proxy + Frozen Physics)':
            continue
        mean_err = np.mean(errors)
        diff = mean_err - baseline
        pct_change = (diff / baseline) * 100
        print(f"  vs {variant_name}: {mean_err:.2f}% ({diff:+.2f}%, {pct_change:+.1f}%)")

    return all_results


if __name__ == "__main__":
    source_checkpoint = "results/heat_experiment1_source/best_model.pt"

    ablation_results = run_ablation_study(
        source_checkpoint_path=source_checkpoint,
        target_alpha=0.02,
        seeds=[42, 123, 456],
        n_epochs=2000,
        epsilon=1e-4,
        save_dir='results/phase1_2_ablation'
    )

    print("\nPhase 1.2 COMPLETE")
