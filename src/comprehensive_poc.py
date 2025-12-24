"""
Comprehensive POC: Baseline vs LoRA Fine-tuning Comparison

Tests multiple transfer scenarios with different LoRA configurations.
Time budget: 2 hours total

Scenarios tested:
1. XJTU → TJU (cross-chemistry)
2. XJTU → MIT (cross-capacity)
3. HUST → MIT (same capacity, different conditions)

LoRA configurations:
- Baseline (full fine-tuning)
- LoRA r=4
- LoRA r=8
- LoRA r=16

Usage:
    source env/bin/activate && python3 src/comprehensive_poc.py
"""

import argparse
import os
import sys
import time
import json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loaders import XJTUdata, MITdata, HUSTdata, TJUdata
from src.models import PINN
from src.lora_layers import add_lora_to_model, get_lora_parameters, print_trainable_parameters
from src.utils.logging import eval_metrix
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Args:
    """Argument container"""
    pass


def create_args(dataset, epochs=15):
    """Create args for model initialization"""
    args = Args()
    args.batch = 0
    args.batch_size = 128
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

    # Dataset-specific hyperparameters
    configs = {
        'XJTU': {'alpha': 0.7, 'beta': 0.2},
        'MIT': {'alpha': 1.0, 'beta': 0.02},
        'TJU': {'alpha': 1.0, 'beta': 0.05},
        'HUST': {'alpha': 0.5, 'beta': 0.2}
    }
    args.alpha = configs[dataset]['alpha']
    args.beta = configs[dataset]['beta']
    args.save_folder = f'results/comprehensive_poc_{dataset}'
    args.log_dir = 'logging.txt'

    return args


def load_dataset_subset(dataset, args, n_batteries=2):
    """Load small subset of dataset"""
    os.makedirs(args.save_folder, exist_ok=True)

    if dataset == 'XJTU':
        root = 'data/XJTU data'
        data = XJTUdata(root=root, args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root) if '2C' in f])[:n_batteries]
    elif dataset == 'MIT':
        root = 'data/MIT data/2017-05-12'
        data = MITdata(root='data/MIT data', args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root)])[:n_batteries]
    elif dataset == 'HUST':
        root = 'data/HUST data'
        data = HUSTdata(root=root, args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root)])[:n_batteries]
    elif dataset == 'TJU':
        root = 'data/TJU data/Dataset_1_NCA_battery'
        data = TJUdata(root='data/TJU data', args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root)])[:n_batteries]

    print(f"  Using {len(files)} batteries: {[os.path.basename(f) for f in files]}")
    loader = data.read_all(specific_path_list=files)
    return {'train': loader['train'], 'valid': loader['valid'], 'test': loader['test']}


def quick_pretrain(dataset, n_batteries=2, epochs=15):
    """Quick pretraining on subset"""
    print(f"\n{'='*70}")
    print(f"PRETRAINING: {dataset}")
    print(f"{'='*70}")

    args = create_args(dataset, epochs=epochs)
    dataloader = load_dataset_subset(dataset, args, n_batteries=n_batteries)

    print(f"\nTraining for {epochs} epochs...")
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

    print(f"\nPretraining completed:")
    print(f"  MAE: {metrics[0]:.4f}")
    print(f"  Time: {elapsed_time:.1f}s ({elapsed_time/60:.2f}min)")

    return model, metrics, elapsed_time


def finetune_baseline(pretrained_model, target_dataset, n_batteries=2, epochs=15):
    """Baseline fine-tuning (freeze dynamical_F, train solution_u)"""
    print(f"\n{'='*70}")
    print(f"BASELINE FINE-TUNING: {target_dataset}")
    print(f"{'='*70}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs)
    dataloader = load_dataset_subset(target_dataset, args, n_batteries=n_batteries)

    # Freeze dynamical_F
    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    trainable_before = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_before:,}")

    # Optimizer
    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    loss_func = torch.nn.MSELoss()
    relu = torch.nn.ReLU()

    best_metrics = None
    best_valid_mse = float('inf')

    print(f"Training for {epochs} epochs...")
    start_time = time.time()

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

        # Validation
        if epoch % 5 == 0 or epoch == epochs:
            valid_mse = pretrained_model.Valid(dataloader['valid'])
            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                true_label, pred_label = pretrained_model.Test(dataloader['test'])
                metrics = eval_metrix(pred_label, true_label)
                best_metrics = {'MAE': float(metrics[0]), 'RMSE': float(metrics[3]), 'epoch': epoch}

    elapsed_time = time.time() - start_time

    print(f"\nBaseline fine-tuning completed:")
    print(f"  Best epoch: {best_metrics['epoch']}")
    print(f"  MAE: {best_metrics['MAE']:.4f}")
    print(f"  Time: {elapsed_time:.1f}s ({elapsed_time/60:.2f}min)")

    return best_metrics, elapsed_time, trainable_before


def finetune_lora(pretrained_model, target_dataset, lora_r=4, n_batteries=2, epochs=15):
    """LoRA fine-tuning"""
    print(f"\n{'='*70}")
    print(f"LoRA FINE-TUNING (r={lora_r}): {target_dataset}")
    print(f"{'='*70}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs)
    dataloader = load_dataset_subset(target_dataset, args, n_batteries=n_batteries)

    # Add LoRA
    lora_alpha = lora_r * 2  # Common practice: alpha = 2*r
    lora_params = add_lora_to_model(pretrained_model, r=lora_r, lora_alpha=lora_alpha)

    trainable, total = print_trainable_parameters(pretrained_model)

    # Optimizer for LoRA parameters only
    lora_param_list = get_lora_parameters(pretrained_model)
    optimizer = torch.optim.Adam(lora_param_list, lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    loss_func = torch.nn.MSELoss()
    relu = torch.nn.ReLU()

    best_metrics = None
    best_valid_mse = float('inf')

    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

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

        # Validation
        if epoch % 5 == 0 or epoch == epochs:
            valid_mse = pretrained_model.Valid(dataloader['valid'])
            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                true_label, pred_label = pretrained_model.Test(dataloader['test'])
                metrics = eval_metrix(pred_label, true_label)
                best_metrics = {'MAE': float(metrics[0]), 'RMSE': float(metrics[3]), 'epoch': epoch}

    elapsed_time = time.time() - start_time

    print(f"\nLoRA fine-tuning completed:")
    print(f"  Best epoch: {best_metrics['epoch']}")
    print(f"  MAE: {best_metrics['MAE']:.4f}")
    print(f"  Time: {elapsed_time:.1f}s ({elapsed_time/60:.2f}min)")

    return best_metrics, elapsed_time, trainable


def run_comprehensive_poc():
    """Run comprehensive POC with multiple scenarios and LoRA configurations"""
    print("="*70)
    print("COMPREHENSIVE POC: Baseline vs LoRA Fine-tuning")
    print("="*70)
    print("\nTime Budget: 2 hours")
    print("Test Configuration: 2 batteries, 15 epochs per stage")
    print("\nScenarios:")
    print("  1. XJTU → TJU (cross-chemistry: NCM → NCA)")
    print("  2. XJTU → MIT (cross-capacity: 2.0Ah → 1.1Ah)")
    print("  3. HUST → MIT (same capacity, different conditions)")
    print("\nConfigurations:")
    print("  - Baseline (full fine-tuning)")
    print("  - LoRA r=4")
    print("  - LoRA r=8")
    print("  - LoRA r=16")

    results = {}
    total_start = time.time()

    # Scenario 1: XJTU → TJU
    print(f"\n{'#'*70}")
    print("SCENARIO 1: XJTU → TJU (Cross-Chemistry Transfer)")
    print(f"{'#'*70}")

    xjtu_model, xjtu_metrics, xjtu_time = quick_pretrain('XJTU', n_batteries=2, epochs=15)

    # Baseline
    baseline_tju, time_baseline_tju, params_baseline = finetune_baseline(
        xjtu_model, 'TJU', n_batteries=2, epochs=15
    )

    # LoRA r=4
    xjtu_model_lora4, _, _ = quick_pretrain('XJTU', n_batteries=2, epochs=15)
    lora4_tju, time_lora4_tju, params_lora4 = finetune_lora(
        xjtu_model_lora4, 'TJU', lora_r=4, n_batteries=2, epochs=15
    )

    # LoRA r=8
    xjtu_model_lora8, _, _ = quick_pretrain('XJTU', n_batteries=2, epochs=15)
    lora8_tju, time_lora8_tju, params_lora8 = finetune_lora(
        xjtu_model_lora8, 'TJU', lora_r=8, n_batteries=2, epochs=15
    )

    # LoRA r=16
    xjtu_model_lora16, _, _ = quick_pretrain('XJTU', n_batteries=2, epochs=15)
    lora16_tju, time_lora16_tju, params_lora16 = finetune_lora(
        xjtu_model_lora16, 'TJU', lora_r=16, n_batteries=2, epochs=15
    )

    results['XJTU_to_TJU'] = {
        'pretrain': {'metrics': xjtu_metrics, 'time': xjtu_time},
        'baseline': {'metrics': baseline_tju, 'time': time_baseline_tju, 'params': params_baseline},
        'lora_r4': {'metrics': lora4_tju, 'time': time_lora4_tju, 'params': params_lora4},
        'lora_r8': {'metrics': lora8_tju, 'time': time_lora8_tju, 'params': params_lora8},
        'lora_r16': {'metrics': lora16_tju, 'time': time_lora16_tju, 'params': params_lora16}
    }

    # Scenario 2: XJTU → MIT
    print(f"\n{'#'*70}")
    print("SCENARIO 2: XJTU → MIT (Cross-Capacity Transfer)")
    print(f"{'#'*70}")

    # Baseline
    baseline_mit, time_baseline_mit, _ = finetune_baseline(
        xjtu_model, 'MIT', n_batteries=2, epochs=15
    )

    # LoRA r=8 (best from scenario 1)
    xjtu_model_lora8_mit, _, _ = quick_pretrain('XJTU', n_batteries=2, epochs=15)
    lora8_mit, time_lora8_mit, _ = finetune_lora(
        xjtu_model_lora8_mit, 'MIT', lora_r=8, n_batteries=2, epochs=15
    )

    results['XJTU_to_MIT'] = {
        'pretrain': {'metrics': xjtu_metrics, 'time': xjtu_time},
        'baseline': {'metrics': baseline_mit, 'time': time_baseline_mit, 'params': params_baseline},
        'lora_r8': {'metrics': lora8_mit, 'time': time_lora8_mit, 'params': params_lora8}
    }

    # Scenario 3: HUST → MIT
    print(f"\n{'#'*70}")
    print("SCENARIO 3: HUST → MIT (Same Capacity Transfer)")
    print(f"{'#'*70}")

    hust_model, hust_metrics, hust_time = quick_pretrain('HUST', n_batteries=2, epochs=15)

    # Baseline
    baseline_hust_mit, time_baseline_hust_mit, _ = finetune_baseline(
        hust_model, 'MIT', n_batteries=2, epochs=15
    )

    # LoRA r=8
    hust_model_lora8, _, _ = quick_pretrain('HUST', n_batteries=2, epochs=15)
    lora8_hust_mit, time_lora8_hust_mit, _ = finetune_lora(
        hust_model_lora8, 'MIT', lora_r=8, n_batteries=2, epochs=15
    )

    results['HUST_to_MIT'] = {
        'pretrain': {'metrics': hust_metrics, 'time': hust_time},
        'baseline': {'metrics': baseline_hust_mit, 'time': time_baseline_hust_mit, 'params': params_baseline},
        'lora_r8': {'metrics': lora8_hust_mit, 'time': time_lora8_hust_mit, 'params': params_lora8}
    }

    total_elapsed = time.time() - total_start

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/comprehensive_poc_results.json', 'w') as f:
        # Convert numpy types to Python types
        results_serializable = {}
        for scenario, data in results.items():
            results_serializable[scenario] = {}
            for stage, info in data.items():
                results_serializable[scenario][stage] = {}
                for key, value in info.items():
                    if key == 'metrics':
                        if isinstance(value, (list, tuple)):
                            results_serializable[scenario][stage][key] = [float(v) for v in value]
                        else:
                            results_serializable[scenario][stage][key] = value
                    else:
                        results_serializable[scenario][stage][key] = float(value) if isinstance(value, (np.floating, np.integer)) else value

        json.dump(results_serializable, f, indent=2)

    print(f"\n{'='*70}")
    print("COMPREHENSIVE POC COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"Results saved to: results/comprehensive_poc_results.json")

    return results, total_elapsed


if __name__ == '__main__':
    results, elapsed = run_comprehensive_poc()
