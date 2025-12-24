"""
Test LoRA Speed Scaling: Does LoRA get faster with larger models?

This script tests the hypothesis that LoRA provides speed benefits
at larger model scales (even if accuracy suffers from overfitting).

Comparison:
- Small model (~8K params): We know LoRA is 23% slower
- Large model (~500K params): Test if LoRA is faster

Usage:
    source env/bin/activate && python3 src/test_lora_scaling.py
"""

import os
import sys
import time
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loaders import XJTUdata, TJUdata
from src.models import PINN
from src.models_large import PINN_Large
from src.lora_layers import add_lora_to_model, get_lora_parameters
from src.utils.logging import eval_metrix
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Args:
    pass


def create_args(dataset, epochs=10, batch_size=256):
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
    args.save_folder = f'results/scaling_test_{dataset}'
    args.log_dir = 'logging.txt'
    return args


def load_dataset_subset(dataset, args, n_batteries=5):
    """Load subset for quick testing"""
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


def pretrain_model(ModelClass, dataset, epochs=10, batch_size=256, n_batteries=5):
    """Pretrain a model (small or large)"""
    args = create_args(dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_dataset_subset(dataset, args, n_batteries=n_batteries)

    model = ModelClass(args)
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


def finetune_baseline(pretrained_model, target_dataset, epochs=10, batch_size=256, n_batteries=5):
    """Baseline fine-tuning: freeze dynamical_F"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_dataset_subset(target_dataset, args, n_batteries=n_batteries)

    # Freeze dynamical_F
    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=0.001)
    trainable_params = sum(p.numel() for p in pretrained_model.solution_u.parameters() if p.requires_grad)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    loss_func = torch.nn.MSELoss()
    relu = torch.nn.ReLU()

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

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time, trainable_params


def finetune_lora(pretrained_model, target_dataset, lora_r=16, epochs=10, batch_size=256, n_batteries=5):
    """LoRA fine-tuning"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_dataset_subset(target_dataset, args, n_batteries=n_batteries)

    # Add LoRA
    lora_alpha = lora_r * 2
    add_lora_to_model(pretrained_model, r=lora_r, lora_alpha=lora_alpha)
    lora_param_list = get_lora_parameters(pretrained_model)
    optimizer = torch.optim.Adam(lora_param_list, lr=0.001)
    trainable_params = sum(p.numel() for p in lora_param_list)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    loss_func = torch.nn.MSELoss()
    relu = torch.nn.ReLU()

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

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time, trainable_params


def run_scaling_test():
    print("="*80)
    print("LORA SCALING TEST: Does LoRA Get Faster at Larger Scales?")
    print("="*80)
    print("\nHypothesis: LoRA overhead is constant, but savings scale with model size")
    print("Expected: Small model = LoRA slower, Large model = LoRA faster\n")

    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_info}\n")

    results = {}

    # Test 1: Small model (baseline - we know LoRA is slower)
    print("="*80)
    print("TEST 1: SMALL MODEL (~8K parameters)")
    print("="*80)

    print("\n[1/2] Pretraining small model on XJTU...")
    small_model, _, _ = pretrain_model(PINN, 'XJTU', epochs=10, n_batteries=5)

    print("\n[2/2] Fine-tuning comparisons on TJU...\n")

    # Baseline
    print("  Baseline (freeze dynamical_F)...")
    small_model_baseline, _, _ = pretrain_model(PINN, 'XJTU', epochs=10, n_batteries=5)
    small_baseline_mae, small_baseline_time, small_baseline_params = finetune_baseline(
        small_model_baseline, 'TJU', epochs=10, n_batteries=5
    )
    print(f"    [OK] Time: {small_baseline_time:.1f}s, Params: {small_baseline_params:,}, MAE: {small_baseline_mae:.4f}")

    # LoRA
    print("\n  LoRA r=16...")
    small_model_lora, _, _ = pretrain_model(PINN, 'XJTU', epochs=10, n_batteries=5)
    small_lora_mae, small_lora_time, small_lora_params = finetune_lora(
        small_model_lora, 'TJU', lora_r=16, epochs=10, n_batteries=5
    )
    print(f"    [OK] Time: {small_lora_time:.1f}s, Params: {small_lora_params:,}, MAE: {small_lora_mae:.4f}")

    small_speedup = small_baseline_time / small_lora_time
    results['small_model'] = {
        'baseline': {'time': small_baseline_time, 'params': small_baseline_params, 'mae': small_baseline_mae},
        'lora_r16': {'time': small_lora_time, 'params': small_lora_params, 'mae': small_lora_mae},
        'speedup': small_speedup
    }

    print(f"\n  Result: LoRA is {small_speedup:.2f}x speed " +
          ("(FASTER [OK])" if small_speedup > 1 else "(SLOWER [FAIL])"))

    # Test 2: Large model
    print("\n" + "="*80)
    print("TEST 2: LARGE MODEL (~500K parameters)")
    print("="*80)

    print("\n[1/2] Pretraining large model on XJTU...")
    large_model, _, _ = pretrain_model(PINN_Large, 'XJTU', epochs=10, n_batteries=5)

    print("\n[2/2] Fine-tuning comparisons on TJU...\n")

    # Baseline
    print("  Baseline (freeze dynamical_F)...")
    large_model_baseline, _, _ = pretrain_model(PINN_Large, 'XJTU', epochs=10, n_batteries=5)
    large_baseline_mae, large_baseline_time, large_baseline_params = finetune_baseline(
        large_model_baseline, 'TJU', epochs=10, n_batteries=5
    )
    print(f"    [OK] Time: {large_baseline_time:.1f}s, Params: {large_baseline_params:,}, MAE: {large_baseline_mae:.4f}")

    # LoRA
    print("\n  LoRA r=16...")
    large_model_lora, _, _ = pretrain_model(PINN_Large, 'XJTU', epochs=10, n_batteries=5)
    large_lora_mae, large_lora_time, large_lora_params = finetune_lora(
        large_model_lora, 'TJU', lora_r=16, epochs=10, n_batteries=5
    )
    print(f"    [OK] Time: {large_lora_time:.1f}s, Params: {large_lora_params:,}, MAE: {large_lora_mae:.4f}")

    large_speedup = large_baseline_time / large_lora_time
    results['large_model'] = {
        'baseline': {'time': large_baseline_time, 'params': large_baseline_params, 'mae': large_baseline_mae},
        'lora_r16': {'time': large_lora_time, 'params': large_lora_params, 'mae': large_lora_mae},
        'speedup': large_speedup
    }

    print(f"\n  Result: LoRA is {large_speedup:.2f}x speed " +
          ("(FASTER [OK])" if large_speedup > 1 else "(SLOWER [FAIL])"))

    # Summary
    print("\n" + "="*80)
    print("SCALING TEST SUMMARY")
    print("="*80)

    print(f"\nSmall Model (~8K params):")
    print(f"  Baseline: {small_baseline_time:.1f}s")
    print(f"  LoRA:     {small_lora_time:.1f}s")
    print(f"  Speedup:  {small_speedup:.2f}x " + ("[OK] FASTER" if small_speedup > 1 else "[FAIL] SLOWER"))

    print(f"\nLarge Model (~500K params):")
    print(f"  Baseline: {large_baseline_time:.1f}s")
    print(f"  LoRA:     {large_lora_time:.1f}s")
    print(f"  Speedup:  {large_speedup:.2f}x " + ("[OK] FASTER" if large_speedup > 1 else "[FAIL] SLOWER"))

    print("\n" + "="*80)
    print("HYPOTHESIS TEST RESULT:")
    print("="*80)

    if large_speedup > 1.0 and small_speedup < 1.0:
        print("[OK] HYPOTHESIS CONFIRMED")
        print("  LoRA overhead is constant, but benefits scale with model size.")
        print(f"  Crossover point: somewhere between 8K and 500K parameters")
    elif large_speedup > small_speedup:
        print("⚠ HYPOTHESIS PARTIALLY CONFIRMED")
        print("  LoRA gets relatively faster with scale, but may need even larger models")
        print(f"  Improvement: {small_speedup:.2f}x → {large_speedup:.2f}x")
    else:
        print("[FAIL] HYPOTHESIS REJECTED")
        print("  LoRA does not get faster at larger scales for PINNs")
        print("  Possible reasons:")
        print("    - Automatic differentiation overhead dominates at all scales")
        print("    - Physics constraints prevent LoRA speedup benefits")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/lora_scaling_test.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: results/lora_scaling_test.json")

    return results


if __name__ == '__main__':
    results = run_scaling_test()
