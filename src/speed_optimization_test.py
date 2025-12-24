"""
Speed Optimization Test: Validate Real Speedup for PINN Fine-Tuning

Tests realistic optimizations that should work for PINNs:
1. Baseline (current implementation)
2. Mixed Precision (FP16 with AMP)
3. Mixed Precision + torch.compile()
4. Mixed Precision + torch.compile() + Larger Batch Size

Goal: Achieve 2x+ speedup for 5-hour fine-tuning → 2.5 hours

Usage:
    source env/bin/activate && python3 src/speed_optimization_test.py
"""

import os
import sys
import time
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loaders import XJTUdata, TJUdata
from src.models import PINN
from src.utils.logging import eval_metrix
import torch
from torch.cuda.amp import autocast, GradScaler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Args:
    pass


def create_args(dataset, epochs=20, batch_size=256):
    args = Args()
    args.batch = 0
    args.batch_size = batch_size
    args.normalization_method = 'min-max'
    args.epochs = epochs
    args.early_stop = 5
    args.warmup_epochs = 3
    args.warmup_lr = 0.002
    args.lr = 0.01
    args.final_lr = 0.0002
    args.lr_F = 0.001
    args.F_layers_num = 3
    args.F_hidden_dim = 60

    configs = {
        'XJTU': {'alpha': 0.7, 'beta': 0.2},
        'TJU': {'alpha': 1.0, 'beta': 0.05}
    }
    args.alpha = configs[dataset]['alpha']
    args.beta = configs[dataset]['beta']
    args.save_folder = f'results/speed_test_{dataset}'
    args.log_dir = 'logging.txt'
    return args


def load_dataset_subset(dataset, args, n_batteries=8):
    """Load realistic subset (8 batteries like honest POC)"""
    os.makedirs(args.save_folder, exist_ok=True)

    if dataset == 'XJTU':
        root = 'data/XJTU data'
        data = XJTUdata(root=root, args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root) if '2C' in f])[:n_batteries]
    elif dataset == 'TJU':
        root = 'data/TJU data/Dataset_1_NCA_battery'
        data = TJUdata(root='data/TJU data', args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root)])[:n_batteries]

    loader = data.read_all(specific_path_list=files)
    return {'train': loader['train'], 'valid': loader['valid'], 'test': loader['test']}


def pretrain_baseline(dataset, epochs=20, batch_size=256, n_batteries=8):
    """Pretrain using baseline (for loading into fine-tuning tests)"""
    args = create_args(dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_dataset_subset(dataset, args, n_batteries=n_batteries)

    model = PINN(args)
    model.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test']
    )

    return model


def finetune_baseline(pretrained_model, target_dataset, epochs=20, batch_size=256, n_batteries=8):
    """Baseline fine-tuning (current implementation)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_dataset_subset(target_dataset, args, n_batteries=n_batteries)

    # Freeze dynamical_F
    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    loss_func = torch.nn.MSELoss()
    relu = torch.nn.ReLU()

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        pretrained_model.train()
        for x1, x2, y1, y2 in dataloader['train']:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            # Standard forward/backward
            u1, f1 = pretrained_model.forward(x1)
            u2, f2 = pretrained_model.forward(x2)

            loss_data = 0.5 * loss_func(u1, y1) + 0.5 * loss_func(u2, y2)
            f_target = torch.zeros_like(f1)
            loss_pde = 0.5 * loss_func(f1, f_target) + 0.5 * loss_func(f2, f_target)
            loss_physics = relu(torch.mul(u2 - u1, y1 - y2)).sum()
            loss = loss_data + args.alpha * loss_pde + args.beta * loss_physics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time


def finetune_mixed_precision(pretrained_model, target_dataset, epochs=20, batch_size=256, n_batteries=8):
    """Fine-tuning with Mixed Precision (FP16)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_dataset_subset(target_dataset, args, n_batteries=n_batteries)

    # Freeze dynamical_F
    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    # Initialize gradient scaler for AMP
    scaler = GradScaler()

    loss_func = torch.nn.MSELoss()
    relu = torch.nn.ReLU()

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        pretrained_model.train()
        for x1, x2, y1, y2 in dataloader['train']:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            # Forward pass in FP16
            with autocast():
                u1, f1 = pretrained_model.forward(x1)
                u2, f2 = pretrained_model.forward(x2)

                loss_data = 0.5 * loss_func(u1, y1) + 0.5 * loss_func(u2, y2)
                f_target = torch.zeros_like(f1)
                loss_pde = 0.5 * loss_func(f1, f_target) + 0.5 * loss_func(f2, f_target)
                loss_physics = relu(torch.mul(u2 - u1, y1 - y2)).sum()
                loss = loss_data + args.alpha * loss_pde + args.beta * loss_physics

            # Backward with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time


def finetune_compiled(pretrained_model, target_dataset, epochs=20, batch_size=256, n_batteries=8):
    """Fine-tuning with Mixed Precision + torch.compile()"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_dataset_subset(target_dataset, args, n_batteries=n_batteries)

    # Freeze dynamical_F
    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    # Compile the model
    print("  Compiling model (first epoch will be slow)...")
    try:
        pretrained_model = torch.compile(pretrained_model, mode='reduce-overhead')
        compiled = True
    except Exception as e:
        print(f"  WARNING: torch.compile() failed: {e}")
        print("  Falling back to non-compiled version")
        compiled = False

    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    scaler = GradScaler()
    loss_func = torch.nn.MSELoss()
    relu = torch.nn.ReLU()

    start_time = time.time()
    first_epoch_time = None

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        pretrained_model.train()

        for x1, x2, y1, y2 in dataloader['train']:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            with autocast():
                u1, f1 = pretrained_model.forward(x1)
                u2, f2 = pretrained_model.forward(x2)

                loss_data = 0.5 * loss_func(u1, y1) + 0.5 * loss_func(u2, y2)
                f_target = torch.zeros_like(f1)
                loss_pde = 0.5 * loss_func(f1, f_target) + 0.5 * loss_func(f2, f_target)
                loss_physics = relu(torch.mul(u2 - u1, y1 - y2)).sum()
                loss = loss_data + args.alpha * loss_pde + args.beta * loss_physics

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        if epoch == 1:
            first_epoch_time = time.time() - epoch_start
            if compiled:
                print(f"  First epoch (with compilation): {first_epoch_time:.1f}s")

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time, first_epoch_time


def finetune_large_batch(pretrained_model, target_dataset, epochs=20, batch_size=512, n_batteries=8):
    """Fine-tuning with Mixed Precision + Larger Batch Size"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_dataset_subset(target_dataset, args, n_batteries=n_batteries)

    # Freeze dynamical_F
    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    # Scale learning rate with batch size (linear scaling rule)
    base_lr = 0.001
    scaled_lr = base_lr * (batch_size / 256)
    print(f"  Scaled LR: {base_lr} → {scaled_lr:.4f} (batch {batch_size})")

    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=scaled_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    scaler = GradScaler()
    loss_func = torch.nn.MSELoss()
    relu = torch.nn.ReLU()

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        pretrained_model.train()
        for x1, x2, y1, y2 in dataloader['train']:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            with autocast():
                u1, f1 = pretrained_model.forward(x1)
                u2, f2 = pretrained_model.forward(x2)

                loss_data = 0.5 * loss_func(u1, y1) + 0.5 * loss_func(u2, y2)
                f_target = torch.zeros_like(f1)
                loss_pde = 0.5 * loss_func(f1, f_target) + 0.5 * loss_func(f2, f_target)
                loss_physics = relu(torch.mul(u2 - u1, y1 - y2)).sum()
                loss = loss_data + args.alpha * loss_pde + args.beta * loss_physics

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time


def run_speed_test():
    print("="*80)
    print("SPEED OPTIMIZATION TEST: Real Speedup Validation")
    print("="*80)
    print("\nObjective: Validate 2x+ speedup for PINN fine-tuning")
    print("Target: 5 hours → 2.5 hours (50% reduction)\n")

    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_info}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    print()

    results = {}

    # Pretrain once
    print("="*80)
    print("PRETRAINING (shared for all tests)")
    print("="*80)
    print("\nPretraining on XJTU (8 batteries, 20 epochs)...")
    source_model = pretrain_baseline('XJTU', epochs=20, n_batteries=8)
    print("[OK] Pretraining complete\n")

    # Test 1: Baseline
    print("="*80)
    print("TEST 1: BASELINE (Current Implementation)")
    print("="*80)
    print("\nFine-tuning on TJU (8 batteries, 20 epochs, batch_size=256)...")

    model_baseline = pretrain_baseline('XJTU', epochs=20, n_batteries=8)
    baseline_mae, baseline_time = finetune_baseline(
        model_baseline, 'TJU', epochs=20, batch_size=256, n_batteries=8
    )

    results['baseline'] = {
        'mae': baseline_mae,
        'time': baseline_time,
        'batch_size': 256,
        'speedup': 1.0
    }

    print(f"[OK] Baseline complete")
    print(f"  Time: {baseline_time:.1f}s ({baseline_time/60:.2f} min)")
    print(f"  MAE: {baseline_mae:.4f}")

    # Test 2: Mixed Precision
    print("\n" + "="*80)
    print("TEST 2: MIXED PRECISION (FP16 with AMP)")
    print("="*80)
    print("\nFine-tuning with FP16 automatic mixed precision...")

    model_amp = pretrain_baseline('XJTU', epochs=20, n_batteries=8)
    amp_mae, amp_time = finetune_mixed_precision(
        model_amp, 'TJU', epochs=20, batch_size=256, n_batteries=8
    )

    amp_speedup = baseline_time / amp_time
    results['mixed_precision'] = {
        'mae': amp_mae,
        'time': amp_time,
        'batch_size': 256,
        'speedup': amp_speedup
    }

    print(f"[OK] Mixed precision complete")
    print(f"  Time: {amp_time:.1f}s ({amp_time/60:.2f} min)")
    print(f"  MAE: {amp_mae:.4f}")
    print(f"  Speedup: {amp_speedup:.2f}x")
    print(f"  MAE change: {((amp_mae - baseline_mae)/baseline_mae)*100:+.1f}%")

    # Test 3: Mixed Precision + torch.compile()
    print("\n" + "="*80)
    print("TEST 3: MIXED PRECISION + torch.compile()")
    print("="*80)
    print("\nFine-tuning with FP16 + compiled autograd...")

    model_compiled = pretrain_baseline('XJTU', epochs=20, n_batteries=8)
    compiled_mae, compiled_time, first_epoch = finetune_compiled(
        model_compiled, 'TJU', epochs=20, batch_size=256, n_batteries=8
    )

    compiled_speedup = baseline_time / compiled_time
    results['compiled'] = {
        'mae': compiled_mae,
        'time': compiled_time,
        'first_epoch_time': first_epoch,
        'batch_size': 256,
        'speedup': compiled_speedup
    }

    print(f"[OK] Compiled version complete")
    print(f"  Time: {compiled_time:.1f}s ({compiled_time/60:.2f} min)")
    print(f"  MAE: {compiled_mae:.4f}")
    print(f"  Speedup: {compiled_speedup:.2f}x")
    print(f"  MAE change: {((compiled_mae - baseline_mae)/baseline_mae)*100:+.1f}%")

    # Test 4: Mixed Precision + Larger Batch
    print("\n" + "="*80)
    print("TEST 4: MIXED PRECISION + LARGER BATCH (512)")
    print("="*80)
    print("\nFine-tuning with FP16 + batch_size=512...")

    model_large_batch = pretrain_baseline('XJTU', epochs=20, n_batteries=8)
    large_batch_mae, large_batch_time = finetune_large_batch(
        model_large_batch, 'TJU', epochs=20, batch_size=512, n_batteries=8
    )

    large_batch_speedup = baseline_time / large_batch_time
    results['large_batch'] = {
        'mae': large_batch_mae,
        'time': large_batch_time,
        'batch_size': 512,
        'speedup': large_batch_speedup
    }

    print(f"[OK] Large batch complete")
    print(f"  Time: {large_batch_time:.1f}s ({large_batch_time/60:.2f} min)")
    print(f"  MAE: {large_batch_mae:.4f}")
    print(f"  Speedup: {large_batch_speedup:.2f}x")
    print(f"  MAE change: {((large_batch_mae - baseline_mae)/baseline_mae)*100:+.1f}%")

    # Summary
    print("\n" + "="*80)
    print("SPEED OPTIMIZATION SUMMARY")
    print("="*80)

    print(f"\n{'Configuration':<30} {'Time (s)':<12} {'Speedup':<10} {'MAE':<10} {'MAE Change'}")
    print("-"*80)
    print(f"{'Baseline':<30} {baseline_time:<12.1f} {1.0:<10.2f} {baseline_mae:<10.4f} -")
    print(f"{'Mixed Precision (FP16)':<30} {amp_time:<12.1f} {amp_speedup:<10.2f} {amp_mae:<10.4f} {((amp_mae-baseline_mae)/baseline_mae)*100:+.1f}%")
    print(f"{'+ torch.compile()':<30} {compiled_time:<12.1f} {compiled_speedup:<10.2f} {compiled_mae:<10.4f} {((compiled_mae-baseline_mae)/baseline_mae)*100:+.1f}%")
    print(f"{'+ Larger Batch (512)':<30} {large_batch_time:<12.1f} {large_batch_speedup:<10.2f} {large_batch_mae:<10.4f} {((large_batch_mae-baseline_mae)/baseline_mae)*100:+.1f}%")

    # Find best configuration
    best_config = max(
        [('Mixed Precision', amp_speedup, amp_mae),
         ('Compiled', compiled_speedup, compiled_mae),
         ('Large Batch', large_batch_speedup, large_batch_mae)],
        key=lambda x: x[1]  # Sort by speedup
    )

    print("\n" + "="*80)
    print("RECOMMENDATION FOR CONSORTIUM")
    print("="*80)

    print(f"\nBest configuration: {best_config[0]}")
    print(f"  Speedup: {best_config[1]:.2f}x")
    print(f"  MAE: {best_config[2]:.4f} (baseline: {baseline_mae:.4f})")

    # Extrapolate to 5-hour training
    extrapolated_time = 5 * 60 / best_config[1]  # hours to minutes
    print(f"\nExtrapolation to full-scale fine-tuning:")
    print(f"  Current: 5.0 hours")
    print(f"  Optimized: {extrapolated_time:.1f} minutes ({extrapolated_time/60:.2f} hours)")
    print(f"  Reduction: {((5 - extrapolated_time/60) / 5)*100:.1f}%")

    if extrapolated_time/60 <= 2.5:
        print(f"\n[OK] TARGET MET: {extrapolated_time/60:.2f} hours ≤ 2.5 hours")
    else:
        print(f"\n[FAIL] TARGET MISSED: {extrapolated_time/60:.2f} hours > 2.5 hours")
        print(f"  Additional speedup needed: {(extrapolated_time/60) / 2.5:.2f}x")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/speed_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: results/speed_optimization_results.json")

    return results


if __name__ == '__main__':
    results = run_speed_test()
