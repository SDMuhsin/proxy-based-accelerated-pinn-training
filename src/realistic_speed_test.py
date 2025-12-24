"""
Realistic Speed Test: Full-Scale Simulation

Simulates the consortium's actual 5-hour fine-tuning workload by:
1. Using ALL available batteries (not 8 subset)
2. Running realistic number of epochs (100-200)
3. Testing optimizations at scale where they should work

Goal: Find optimizations that work for 5-hour → 2.5-hour reduction

Usage:
    source env/bin/activate && python3 src/realistic_speed_test.py
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
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Args:
    pass


def create_args(dataset, epochs=100, batch_size=256):
    args = Args()
    args.batch = 0
    args.batch_size = batch_size
    args.normalization_method = 'min-max'
    args.epochs = epochs
    args.early_stop = 15  # More patience for full-scale
    args.warmup_epochs = 5
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
    args.save_folder = f'results/realistic_test_{dataset}'
    args.log_dir = 'logging.txt'
    return args


def load_full_dataset(dataset, args, max_batteries=None):
    """Load FULL dataset (or large subset)"""
    os.makedirs(args.save_folder, exist_ok=True)

    if dataset == 'XJTU':
        root = 'data/XJTU data'
        data = XJTUdata(root=root, args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root) if '2C' in f])
        if max_batteries:
            files = files[:max_batteries]
        print(f"  Loading {len(files)} XJTU batteries from {root}")
    elif dataset == 'TJU':
        root = 'data/TJU data/Dataset_1_NCA_battery'
        data = TJUdata(root='data/TJU data', args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root)])
        if max_batteries:
            files = files[:max_batteries]
        print(f"  Loading {len(files)} TJU batteries from {root}")

    loader = data.read_all(specific_path_list=files)
    return {'train': loader['train'], 'valid': loader['valid'], 'test': loader['test']}, len(files)


def pretrain_baseline(dataset, epochs=50, batch_size=256, max_batteries=20):
    """Pretrain on realistic dataset size"""
    args = create_args(dataset, epochs=epochs, batch_size=batch_size)
    dataloader, n_batteries = load_full_dataset(dataset, args, max_batteries=max_batteries)

    print(f"  Dataset: {n_batteries} batteries, batch_size={batch_size}, epochs={epochs}")

    model = PINN(args)
    start_time = time.time()
    model.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test']
    )
    elapsed_time = time.time() - start_time

    true_label, pred_label = model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return model, float(metrics[0]), elapsed_time


def finetune_baseline(pretrained_model, target_dataset, epochs=100, batch_size=256, max_batteries=30):
    """Baseline fine-tuning at realistic scale"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader, n_batteries = load_full_dataset(target_dataset, args, max_batteries=max_batteries)

    print(f"  Dataset: {n_batteries} batteries, batch_size={batch_size}, epochs={epochs}")

    # Freeze dynamical_F
    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    loss_func = nn.MSELoss()
    relu = nn.ReLU()

    best_val_loss = float('inf')
    patience_counter = 0

    start_time = time.time()
    actual_epochs = 0

    for epoch in range(1, epochs + 1):
        pretrained_model.train()
        for x1, x2, y1, y2 in dataloader['train']:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

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
        actual_epochs = epoch

        # Early stopping check
        if epoch % 5 == 0:
            val_loss = pretrained_model.Valid(dataloader['valid'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Progress every 10 epochs
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}/{epochs}, elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time, actual_epochs


def finetune_mixed_precision(pretrained_model, target_dataset, epochs=100, batch_size=256, max_batteries=30):
    """Mixed precision at realistic scale"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader, n_batteries = load_full_dataset(target_dataset, args, max_batteries=max_batteries)

    print(f"  Dataset: {n_batteries} batteries, batch_size={batch_size}, epochs={epochs}")

    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    scaler = GradScaler()
    loss_func = nn.MSELoss()
    relu = nn.ReLU()

    best_val_loss = float('inf')
    patience_counter = 0

    start_time = time.time()
    actual_epochs = 0

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
        actual_epochs = epoch

        if epoch % 5 == 0:
            val_loss = pretrained_model.Valid(dataloader['valid'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}/{epochs}, elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time, actual_epochs


def finetune_large_batch(pretrained_model, target_dataset, epochs=100, batch_size=512, max_batteries=30):
    """Large batch size at realistic scale"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader, n_batteries = load_full_dataset(target_dataset, args, max_batteries=max_batteries)

    print(f"  Dataset: {n_batteries} batteries, batch_size={batch_size}, epochs={epochs}")

    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    # Scale LR with batch size
    base_lr = 0.001
    scaled_lr = base_lr * (batch_size / 256)
    print(f"  Scaled LR: {scaled_lr:.4f} (batch_size={batch_size})")

    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=scaled_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    scaler = GradScaler()
    loss_func = nn.MSELoss()
    relu = nn.ReLU()

    best_val_loss = float('inf')
    patience_counter = 0

    start_time = time.time()
    actual_epochs = 0

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
        actual_epochs = epoch

        if epoch % 5 == 0:
            val_loss = pretrained_model.Valid(dataloader['valid'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}/{epochs}, elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time, actual_epochs


def run_realistic_test():
    print("="*80)
    print("REALISTIC SPEED TEST: Full-Scale Simulation")
    print("="*80)
    print("\nSimulating consortium's 5-hour fine-tuning workload")
    print("Testing optimizations at scale where they should work\n")

    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_info}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print()

    results = {}

    # Use larger dataset + more epochs to simulate 5-hour workload
    # Scaled down to fit in reasonable time for POC (~30 minutes total)
    PRETRAIN_BATTERIES = 20  # XJTU has 55 total
    FINETUNE_BATTERIES = 30  # TJU has 130 total
    PRETRAIN_EPOCHS = 50
    FINETUNE_EPOCHS = 100

    print("="*80)
    print(f"TEST CONFIGURATION (Scaled to ~30 minute POC)")
    print("="*80)
    print(f"Pretraining: {PRETRAIN_BATTERIES} batteries, {PRETRAIN_EPOCHS} epochs")
    print(f"Fine-tuning: {FINETUNE_BATTERIES} batteries, {FINETUNE_EPOCHS} epochs")
    print(f"Batch size: 256 (baseline), 512 (large batch test)")
    print()

    # Pretrain once
    print("="*80)
    print("PRETRAINING")
    print("="*80)
    print(f"\nPretraining on XJTU...")
    pretrain_start = time.time()
    source_model = pretrain_baseline('XJTU', epochs=PRETRAIN_EPOCHS, max_batteries=PRETRAIN_BATTERIES)
    pretrain_time = time.time() - pretrain_start
    print(f"[OK] Pretraining complete: {pretrain_time:.1f}s ({pretrain_time/60:.1f} min)\n")

    # Test 1: Baseline
    print("="*80)
    print("TEST 1: BASELINE")
    print("="*80)
    print(f"\nFine-tuning on TJU (batch_size=256)...")

    model_baseline = pretrain_baseline('XJTU', epochs=PRETRAIN_EPOCHS, max_batteries=PRETRAIN_BATTERIES)[0]
    baseline_mae, baseline_time, baseline_epochs = finetune_baseline(
        model_baseline, 'TJU', epochs=FINETUNE_EPOCHS, batch_size=256, max_batteries=FINETUNE_BATTERIES
    )

    results['baseline'] = {
        'mae': baseline_mae,
        'time': baseline_time,
        'epochs': baseline_epochs,
        'batch_size': 256,
        'speedup': 1.0
    }

    print(f"\n[OK] Baseline complete")
    print(f"  Time: {baseline_time:.1f}s ({baseline_time/60:.1f} min)")
    print(f"  Epochs: {baseline_epochs}")
    print(f"  MAE: {baseline_mae:.4f}")

    # Test 2: Mixed Precision
    print("\n" + "="*80)
    print("TEST 2: MIXED PRECISION (FP16)")
    print("="*80)
    print(f"\nFine-tuning with FP16...")

    model_amp = pretrain_baseline('XJTU', epochs=PRETRAIN_EPOCHS, max_batteries=PRETRAIN_BATTERIES)[0]
    amp_mae, amp_time, amp_epochs = finetune_mixed_precision(
        model_amp, 'TJU', epochs=FINETUNE_EPOCHS, batch_size=256, max_batteries=FINETUNE_BATTERIES
    )

    amp_speedup = baseline_time / amp_time
    results['mixed_precision'] = {
        'mae': amp_mae,
        'time': amp_time,
        'epochs': amp_epochs,
        'batch_size': 256,
        'speedup': amp_speedup
    }

    print(f"\n[OK] Mixed precision complete")
    print(f"  Time: {amp_time:.1f}s ({amp_time/60:.1f} min)")
    print(f"  Epochs: {amp_epochs}")
    print(f"  MAE: {amp_mae:.4f}")
    print(f"  Speedup: {amp_speedup:.2f}x")

    # Test 3: Large Batch with FP16
    print("\n" + "="*80)
    print("TEST 3: LARGE BATCH (512) + FP16")
    print("="*80)
    print(f"\nFine-tuning with batch_size=512 + FP16...")

    model_large = pretrain_baseline('XJTU', epochs=PRETRAIN_EPOCHS, max_batteries=PRETRAIN_BATTERIES)[0]
    large_mae, large_time, large_epochs = finetune_large_batch(
        model_large, 'TJU', epochs=FINETUNE_EPOCHS, batch_size=512, max_batteries=FINETUNE_BATTERIES
    )

    large_speedup = baseline_time / large_time
    results['large_batch'] = {
        'mae': large_mae,
        'time': large_time,
        'epochs': large_epochs,
        'batch_size': 512,
        'speedup': large_speedup
    }

    print(f"\n[OK] Large batch complete")
    print(f"  Time: {large_time:.1f}s ({large_time/60:.1f} min)")
    print(f"  Epochs: {large_epochs}")
    print(f"  MAE: {large_mae:.4f}")
    print(f"  Speedup: {large_speedup:.2f}x")

    # Summary
    print("\n" + "="*80)
    print("REALISTIC SCALE SUMMARY")
    print("="*80)

    print(f"\n{'Configuration':<30} {'Time (min)':<12} {'Speedup':<10} {'Epochs':<10} {'MAE'}")
    print("-"*80)
    print(f"{'Baseline (batch=256)':<30} {baseline_time/60:<12.1f} {1.0:<10.2f} {baseline_epochs:<10} {baseline_mae:.4f}")
    print(f"{'FP16 (batch=256)':<30} {amp_time/60:<12.1f} {amp_speedup:<10.2f} {amp_epochs:<10} {amp_mae:.4f}")
    print(f"{'FP16 + Large Batch (512)':<30} {large_time/60:<12.1f} {large_speedup:<10.2f} {large_epochs:<10} {large_mae:.4f}")

    best_speedup = max(amp_speedup, large_speedup)
    best_config = "FP16" if amp_speedup > large_speedup else "FP16 + Large Batch"

    print("\n" + "="*80)
    print("EXTRAPOLATION TO 5-HOUR WORKLOAD")
    print("="*80)

    print(f"\nBest configuration: {best_config}")
    print(f"Best speedup: {best_speedup:.2f}x")

    extrapolated_time = 5.0 / best_speedup
    print(f"\n5-hour baseline → {extrapolated_time:.2f} hours with {best_config}")

    if extrapolated_time <= 2.5:
        print(f"[OK] TARGET MET: {extrapolated_time:.2f} hours ≤ 2.5 hours")
    else:
        print(f"[FAIL] TARGET MISSED: {extrapolated_time:.2f} hours > 2.5 hours")
        print(f"  Need additional {(extrapolated_time / 2.5):.2f}x speedup")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/realistic_speed_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: results/realistic_speed_results.json")

    return results


if __name__ == '__main__':
    results = run_realistic_test()
