"""
Quick Comprehensive POC: Baseline vs LoRA (optimized for 2-hour runtime)

Focus: Performance comparison (MAE) and speed comparison
Scenarios: 3 transfer scenarios × (1 baseline + 3 LoRA configs) = 12 experiments
Time per experiment: ~6-10 minutes → Total: ~90-120 minutes

Usage:
    source env/bin/activate && python3 src/quick_comprehensive_poc.py
"""

import os
import sys
import time
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loaders import XJTUdata, MITdata, HUSTdata, TJUdata
from src.models import PINN
from src.lora_layers import add_lora_to_model, get_lora_parameters, print_trainable_parameters
from src.utils.logging import eval_metrix
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Args:
    pass


def create_args(dataset, epochs=15):
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

    configs = {
        'XJTU': {'alpha': 0.7, 'beta': 0.2},
        'MIT': {'alpha': 1.0, 'beta': 0.02},
        'TJU': {'alpha': 1.0, 'beta': 0.05},
        'HUST': {'alpha': 0.5, 'beta': 0.2}
    }
    args.alpha = configs[dataset]['alpha']
    args.beta = configs[dataset]['beta']
    args.save_folder = f'results/quick_poc_{dataset}'
    args.log_dir = 'logging.txt'
    return args


def load_dataset_subset(dataset, args, n_batteries=2):
    os.makedirs(args.save_folder, exist_ok=True)

    if dataset == 'XJTU':
        root = 'data/XJTU data'
        data = XJTUdata(root=root, args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root) if '2C' in f])[:n_batteries]
    elif dataset == 'MIT':
        root = 'data/MIT data/2017-05-12'
        data = MITdata(root='data/MIT data', args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root)])[:n_batteries]
    elif dataset == 'TJU':
        root = 'data/TJU data/Dataset_1_NCA_battery'
        data = TJUdata(root='data/TJU data', args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root)])[:n_batteries]

    loader = data.read_all(specific_path_list=files)
    return {'train': loader['train'], 'valid': loader['valid'], 'test': loader['test']}


def quick_pretrain(dataset, epochs=15):
    args = create_args(dataset, epochs=epochs)
    dataloader = load_dataset_subset(dataset, args)

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


def finetune_with_config(pretrained_model, target_dataset, use_lora=False, lora_r=0, epochs=15):
    """Fine-tune with or without LoRA"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args(target_dataset, epochs=epochs)
    dataloader = load_dataset_subset(target_dataset, args)

    if use_lora:
        # Add LoRA
        lora_alpha = lora_r * 2
        add_lora_to_model(pretrained_model, r=lora_r, lora_alpha=lora_alpha)
        lora_param_list = get_lora_parameters(pretrained_model)
        optimizer = torch.optim.Adam(lora_param_list, lr=0.001)
        trainable_params = sum(p.numel() for p in lora_param_list)
    else:
        # Baseline: freeze dynamical_F
        for param in pretrained_model.dynamical_F.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=0.001)
        trainable_params = sum(p.numel() for p in pretrained_model.solution_u.parameters() if p.requires_grad)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    loss_func = torch.nn.MSELoss()
    relu = torch.nn.ReLU()

    best_mae = float('inf')

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

        # Check every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            true_label, pred_label = pretrained_model.Test(dataloader['test'])
            metrics = eval_metrix(pred_label, true_label)
            if metrics[0] < best_mae:
                best_mae = metrics[0]

    elapsed_time = time.time() - start_time

    return float(best_mae), elapsed_time, trainable_params


def run_quick_poc():
    print("="*70)
    print("QUICK COMPREHENSIVE POC: Baseline vs LoRA")
    print("="*70)
    print("\nScenarios: 3 transfer scenarios")
    print("Configurations: 1 baseline + 3 LoRA (r=4, 8, 16)")
    print("Focus: Performance (MAE) vs Training Speed\n")

    results = {}
    total_start = time.time()

    scenarios = [
        ('XJTU', 'TJU', 'Cross-Chemistry: NCM → NCA'),
        ('XJTU', 'MIT', 'Cross-Capacity: 2.0Ah → 1.1Ah'),
        ('MIT', 'TJU', 'Different conditions')
    ]

    for i, (source, target, desc) in enumerate(scenarios, 1):
        print(f"\n{'#'*70}")
        print(f"SCENARIO {i}: {source} → {target} ({desc})")
        print(f"{'#'*70}\n")

        # Pretrain
        print(f"[1/5] Pretraining on {source}...")
        source_model, pretrain_mae, pretrain_time = quick_pretrain(source, epochs=15)
        print(f"  [OK] MAE: {pretrain_mae:.4f}, Time: {pretrain_time:.1f}s")

        # Baseline fine-tuning
        print(f"\n[2/5] Baseline fine-tuning on {target}...")
        baseline_mae, baseline_time, baseline_params = finetune_with_config(
            source_model, target, use_lora=False, epochs=15
        )
        print(f"  [OK] MAE: {baseline_mae:.4f}, Time: {baseline_time:.1f}s, Params: {baseline_params:,}")

        # LoRA r=4
        print(f"\n[3/5] LoRA r=4 fine-tuning on {target}...")
        source_model_r4, _, _ = quick_pretrain(source, epochs=15)
        lora4_mae, lora4_time, lora4_params = finetune_with_config(
            source_model_r4, target, use_lora=True, lora_r=4, epochs=15
        )
        print(f"  [OK] MAE: {lora4_mae:.4f}, Time: {lora4_time:.1f}s, Params: {lora4_params:,}")

        # LoRA r=8
        print(f"\n[4/5] LoRA r=8 fine-tuning on {target}...")
        source_model_r8, _, _ = quick_pretrain(source, epochs=15)
        lora8_mae, lora8_time, lora8_params = finetune_with_config(
            source_model_r8, target, use_lora=True, lora_r=8, epochs=15
        )
        print(f"  [OK] MAE: {lora8_mae:.4f}, Time: {lora8_time:.1f}s, Params: {lora8_params:,}")

        # LoRA r=16
        print(f"\n[5/5] LoRA r=16 fine-tuning on {target}...")
        source_model_r16, _, _ = quick_pretrain(source, epochs=15)
        lora16_mae, lora16_time, lora16_params = finetune_with_config(
            source_model_r16, target, use_lora=True, lora_r=16, epochs=15
        )
        print(f"  [OK] MAE: {lora16_mae:.4f}, Time: {lora16_time:.1f}s, Params: {lora16_params:,}")

        # Store results
        scenario_key = f"{source}_to_{target}"
        results[scenario_key] = {
            'description': desc,
            'pretrain': {'mae': pretrain_mae, 'time': pretrain_time},
            'baseline': {'mae': baseline_mae, 'time': baseline_time, 'params': baseline_params},
            'lora_r4': {'mae': lora4_mae, 'time': lora4_time, 'params': lora4_params},
            'lora_r8': {'mae': lora8_mae, 'time': lora8_time, 'params': lora8_params},
            'lora_r16': {'mae': lora16_mae, 'time': lora16_time, 'params': lora16_params}
        }

        # Print summary for this scenario
        print(f"\n{'-'*70}")
        print(f"SCENARIO {i} SUMMARY:")
        print(f"{'-'*70}")
        print(f"{'Config':<15} {'MAE':<10} {'Time (s)':<12} {'Params':<12} {'vs Baseline'}")
        print(f"{'-'*70}")
        print(f"{'Baseline':<15} {baseline_mae:<10.4f} {baseline_time:<12.1f} {baseline_params:<12,} -")
        print(f"{'LoRA r=4':<15} {lora4_mae:<10.4f} {lora4_time:<12.1f} {lora4_params:<12,} {lora4_time/baseline_time:.2f}x speed")
        print(f"{'LoRA r=8':<15} {lora8_mae:<10.4f} {lora8_time:<12.1f} {lora8_params:<12,} {lora8_time/baseline_time:.2f}x speed")
        print(f"{'LoRA r=16':<15} {lora16_mae:<10.4f} {lora16_time:<12.1f} {lora16_params:<12,} {lora16_time/baseline_time:.2f}x speed")

    total_elapsed = time.time() - total_start

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/quick_comprehensive_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"POC COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"Results saved to: results/quick_comprehensive_results.json")

    return results, total_elapsed


if __name__ == '__main__':
    results, elapsed = run_quick_poc()
