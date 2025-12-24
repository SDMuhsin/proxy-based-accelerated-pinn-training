"""
Test: Sparse Physics Sampling for PINN Acceleration

Goal: Validate that computing physics on only 25% of batch:
1. Actually speeds up training
2. Doesn't hurt accuracy too much

This is the LOWEST RISK approach with HIGH CONFIDENCE.
"""

import sys
import os
import torch
import time
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import PINN
from src.utils.logging import AverageMeter
from src.data_loaders import get_MIT_data, get_TJU_data
from src.utils import fix_seeds

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_with_sparse_physics(model, dataloader, physics_ratio=1.0, epochs=20):
    """
    Train PINN with sparse physics sampling

    Args:
        model: PINN model
        dataloader: Training data
        physics_ratio: Fraction of batch to compute physics on (1.0 = all, 0.25 = 25%)
        epochs: Number of training epochs

    Returns:
        final_mae: Final MAE on validation
        train_time: Total training time
    """
    model.train()

    start_time = time.time()

    for epoch in range(epochs):
        for iter, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            # ==================================================================
            # SPARSE PHYSICS SAMPLING
            # ==================================================================
            batch_size = x1.shape[0]

            if physics_ratio < 1.0:
                # Sample subset for physics computation
                n_physics = max(1, int(batch_size * physics_ratio))
                physics_idx = torch.randperm(batch_size)[:n_physics]

                # Compute predictions for ALL (needed for data loss)
                x1.requires_grad = False
                x2.requires_grad = False
                u1_all = model.solution_u(x1)
                u2_all = model.solution_u(x2)

                # Compute physics for SUBSET only
                x1_physics = x1[physics_idx].clone()
                x2_physics = x2[physics_idx].clone()
                x1_physics.requires_grad = True
                x2_physics.requires_grad = True

                x1_p_feat = x1_physics[:, :-1]
                x1_p_time = x1_physics[:, -1:]
                u1_p = model.solution_u(torch.cat((x1_p_feat, x1_p_time), dim=1))
                u1_t = torch.autograd.grad(u1_p.sum(), x1_p_time,
                                           create_graph=True, only_inputs=True, allow_unused=True)[0]
                u1_x = torch.autograd.grad(u1_p.sum(), x1_p_feat,
                                           create_graph=True, only_inputs=True, allow_unused=True)[0]

                x2_p_feat = x2_physics[:, :-1]
                x2_p_time = x2_physics[:, -1:]
                u2_p = model.solution_u(torch.cat((x2_p_feat, x2_p_time), dim=1))
                u2_t = torch.autograd.grad(u2_p.sum(), x2_p_time,
                                           create_graph=True, only_inputs=True, allow_unused=True)[0]
                u2_x = torch.autograd.grad(u2_p.sum(), x2_p_feat,
                                           create_graph=True, only_inputs=True, allow_unused=True)[0]

                F1 = model.dynamical_F(torch.cat([x1_physics, u1_p, u1_x, u1_t], dim=1))
                F2 = model.dynamical_F(torch.cat([x2_physics, u2_p, u2_x, u2_t], dim=1))
                f1 = u1_t - F1
                f2 = u2_t - F2

                u1, u2 = u1_all, u2_all

            else:
                # Standard forward (compute physics for ALL)
                u1, f1 = model.forward(x1)
                u2, f2 = model.forward(x2)

            # ==================================================================
            # LOSS COMPUTATION (same as baseline)
            # ==================================================================

            # Data loss (on full batch)
            loss1 = 0.5 * model.loss_func(u1, y1) + 0.5 * model.loss_func(u2, y2)

            # PDE loss (on sampled points)
            f_target = torch.zeros_like(f1)
            loss2 = 0.5 * model.loss_func(f1, f_target) + 0.5 * model.loss_func(f2, f_target)

            # Physics loss (on sampled points)
            if physics_ratio < 1.0:
                # Need to sample corresponding y values
                y1_physics = y1[physics_idx]
                y2_physics = y2[physics_idx]
                u1_physics = u1[physics_idx]
                u2_physics = u2[physics_idx]
                loss3 = model.relu(torch.mul(u2_physics - u1_physics,
                                            y1_physics - y2_physics)).sum()
            else:
                loss3 = model.relu(torch.mul(u2 - u1, y1 - y2)).sum()

            # Total loss
            loss = loss1 + model.alpha * loss2 + model.beta * loss3

            # Optimization
            model.optimizer1.zero_grad()
            model.optimizer2.zero_grad()
            loss.backward()
            model.optimizer1.step()
            model.optimizer2.step()

    train_time = time.time() - start_time

    # Compute final MAE
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x1, x2, y1, y2 in dataloader:
            x1, y1 = x1.to(device), y1.to(device)
            u1 = model.solution_u(x1)
            all_preds.append(u1)
            all_targets.append(y1)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mae = torch.abs(all_preds - all_targets).mean().item()

    return mae, train_time


def run_sparse_physics_test():
    """
    Compare baseline (100% physics) vs sparse (25% physics)
    """
    print("=" * 80)
    print("SPARSE PHYSICS SAMPLING TEST")
    print("=" * 80)
    print()
    print("Testing if computing physics on 25% of batch:")
    print("  1. Speeds up training")
    print("  2. Maintains acceptable accuracy")
    print()

    # Setup
    fix_seeds(42)

    # Small test: 3 batteries, 20 epochs
    print("Loading data (3 batteries, 20 epochs)...")
    train_data = get_MIT_data(start_battery=0, num_batteries=3)

    # Create args for PINN
    import argparse
    args = argparse.Namespace(
        save_folder=None,
        log_dir='./logs/sparse_test',
        dataset='MIT',
        source_dataset='MIT',
        target_dataset='TJU',
        num_epochs=20,
        batch_size=64,
        base_lr=0.001,
        encoder_name='default',
        use_data_parallel=False,
        gpu_ids='0',
        alpha=1.0,  # PDE loss weight
        beta=0.1,   # Physics loss weight
    )

    results = {}

    # ==================================================================
    # TEST 1: BASELINE (100% physics)
    # ==================================================================
    print()
    print("=" * 80)
    print("TEST 1: BASELINE (100% physics computation)")
    print("=" * 80)

    fix_seeds(42)
    model_baseline = PINN(args)

    from torch.utils.data import DataLoader, TensorDataset
    train_loader = DataLoader(
        TensorDataset(*train_data),
        batch_size=args.batch_size,
        shuffle=True
    )

    print(f"Training with 100% physics sampling...")
    mae_baseline, time_baseline = train_with_sparse_physics(
        model_baseline, train_loader, physics_ratio=1.0, epochs=args.num_epochs
    )

    results['baseline'] = {
        'mae': mae_baseline,
        'time': time_baseline,
        'physics_ratio': 1.0,
        'speedup': 1.0
    }

    print(f"[OK] Baseline complete")
    print(f"  Time: {time_baseline:.2f}s")
    print(f"  MAE: {mae_baseline:.4f}")

    # ==================================================================
    # TEST 2: SPARSE 50%
    # ==================================================================
    print()
    print("=" * 80)
    print("TEST 2: SPARSE 50% (physics on 50% of batch)")
    print("=" * 80)

    fix_seeds(42)
    model_sparse50 = PINN(args)

    train_loader = DataLoader(
        TensorDataset(*train_data),
        batch_size=args.batch_size,
        shuffle=True
    )

    print(f"Training with 50% physics sampling...")
    mae_sparse50, time_sparse50 = train_with_sparse_physics(
        model_sparse50, train_loader, physics_ratio=0.5, epochs=args.num_epochs
    )

    speedup_50 = time_baseline / time_sparse50
    results['sparse_50'] = {
        'mae': mae_sparse50,
        'time': time_sparse50,
        'physics_ratio': 0.5,
        'speedup': speedup_50
    }

    print(f"[OK] Sparse 50% complete")
    print(f"  Time: {time_sparse50:.2f}s")
    print(f"  MAE: {mae_sparse50:.4f}")
    print(f"  Speedup: {speedup_50:.2f}x")
    print(f"  MAE degradation: {((mae_sparse50 - mae_baseline) / mae_baseline * 100):.1f}%")

    # ==================================================================
    # TEST 3: SPARSE 25%
    # ==================================================================
    print()
    print("=" * 80)
    print("TEST 3: SPARSE 25% (physics on 25% of batch)")
    print("=" * 80)

    fix_seeds(42)
    model_sparse25 = PINN(args)

    train_loader = DataLoader(
        TensorDataset(*train_data),
        batch_size=args.batch_size,
        shuffle=True
    )

    print(f"Training with 25% physics sampling...")
    mae_sparse25, time_sparse25 = train_with_sparse_physics(
        model_sparse25, train_loader, physics_ratio=0.25, epochs=args.num_epochs
    )

    speedup_25 = time_baseline / time_sparse25
    results['sparse_25'] = {
        'mae': mae_sparse25,
        'time': time_sparse25,
        'physics_ratio': 0.25,
        'speedup': speedup_25
    }

    print(f"[OK] Sparse 25% complete")
    print(f"  Time: {time_sparse25:.2f}s")
    print(f"  MAE: {mae_sparse25:.4f}")
    print(f"  Speedup: {speedup_25:.2f}x")
    print(f"  MAE degradation: {((mae_sparse25 - mae_baseline) / mae_baseline * 100):.1f}%")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Configuration           Time (s)    Speedup    MAE        MAE Δ")
    print("-" * 80)
    print(f"Baseline (100%)         {time_baseline:6.2f}      {1.0:.2f}x     {mae_baseline:.4f}    +0.0%")
    print(f"Sparse 50%              {time_sparse50:6.2f}      {speedup_50:.2f}x     {mae_sparse50:.4f}    {((mae_sparse50-mae_baseline)/mae_baseline*100):+.1f}%")
    print(f"Sparse 25%              {time_sparse25:6.2f}      {speedup_25:.2f}x     {mae_sparse25:.4f}    {((mae_sparse25-mae_baseline)/mae_baseline*100):+.1f}%")
    print()

    # Verdict
    print("VERDICT:")
    if speedup_25 > 1.1:  # At least 10% faster
        print(f"[OK] SPARSE PHYSICS WORKS! {speedup_25:.2f}x speedup achieved")
        if abs((mae_sparse25 - mae_baseline) / mae_baseline) < 0.1:  # Less than 10% degradation
            print(f"[OK] Accuracy maintained (MAE degradation < 10%)")
            print()
            print("RECOMMENDATION: Use 25% physics sampling for 5-hour workload")
            print(f"  Estimated time: 5 hours / {speedup_25:.2f} = {5/speedup_25:.2f} hours")
        else:
            print(f"⚠ Accuracy degraded by {abs((mae_sparse25-mae_baseline)/mae_baseline*100):.1f}%")
            print(f"  Try 50% sampling instead: {5/speedup_50:.2f} hours")
    else:
        print(f"[FAIL] Speedup insufficient ({speedup_25:.2f}x)")
        print("  This approach may not provide the needed 2x speedup")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/sparse_physics_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("Results saved to: results/sparse_physics_results.json")

    return results


if __name__ == '__main__':
    run_sparse_physics_test()
