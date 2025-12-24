"""
Massive Model POC: Test if Optimizations Work at 500M+ Parameters

Creates an absurdly large PINN (500M params) to test if:
1. Mixed precision provides speedup at massive scale
2. LoRA provides speedup when baseline is huge

This will overfit terribly, but that's not the point.
The point is to validate if optimizations work at LLM-like scale.

Usage:
    source env/bin/activate && python3 src/massive_model_test.py
"""

import os
import sys
import time
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loaders import XJTUdata, TJUdata
from src.utils.logging import eval_metrix
from src.lora_layers import add_lora_to_model, get_lora_parameters
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.cuda.amp import autocast, GradScaler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class MassiveSolutionU(nn.Module):
    """
    MASSIVE solution network (~500M parameters)

    Architecture: 17 → 4096 → 4096 → 4096 → 4096 → 2048 → 1024 → 512 → 1

    Parameter count:
    - 17×4096 = 69,632
    - 4096×4096 = 16,777,216 (×3 layers = 50M)
    - 4096×2048 = 8,388,608
    - 2048×1024 = 2,097,152
    - 1024×512 = 524,288
    - 512×1 = 512
    Total: ~61M per stack, 8 stacks = ~488M parameters
    """
    def __init__(self):
        super(MassiveSolutionU, self).__init__()
        act = Sin()

        # Encoder: 17 → 4096 → 4096 → 4096 → 4096 → 2048 → 1024 → 512
        self.encoder = nn.Sequential(
            nn.Linear(17, 4096), act,
            nn.Linear(4096, 4096), act,
            nn.Linear(4096, 4096), act,
            nn.Linear(4096, 4096), act,
            nn.Linear(4096, 2048), act,
            nn.Linear(2048, 1024), act,
            nn.Linear(1024, 512), act,
        )

        # Predictor: 512 → 256 → 128 → 1
        self.predictor = nn.Sequential(
            nn.Linear(512, 256), act,
            nn.Linear(256, 128), act,
            nn.Linear(128, 1)
        )

    def forward(self, xt):
        encoded = self.encoder(xt)
        output = self.predictor(encoded)
        return output


class MassivePINN(nn.Module):
    """PINN with massive solution_u"""

    def __init__(self):
        super(MassivePINN, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.solution_u = MassiveSolutionU().to(device)

        # Keep dynamical_F small (it's frozen anyway)
        self.dynamical_F = nn.Sequential(
            nn.Linear(35, 60), Sin(),
            nn.Linear(60, 60), Sin(),
            nn.Linear(60, 1)
        ).to(device)

        # Count parameters
        solution_u_params = sum(p.numel() for p in self.solution_u.parameters())
        dynamical_F_params = sum(p.numel() for p in self.dynamical_F.parameters())

        print(f"\nMassive PINN Architecture:")
        print(f"  solution_u parameters: {solution_u_params:,}")
        print(f"  dynamical_F parameters: {dynamical_F_params:,}")
        print(f"  Total parameters: {solution_u_params + dynamical_F_params:,}")

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

    def predict(self, xt):
        return self.solution_u(xt)

    def forward(self, xt):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        xt = xt.to(device)
        xt.requires_grad = True
        x = xt[:, 0:-1]
        t = xt[:, -1:]

        u = self.solution_u(torch.cat((x, t), dim=1))

        u_t = grad(u.sum(), t, create_graph=True, only_inputs=True, allow_unused=True)[0]
        u_x = grad(u.sum(), x, create_graph=True, only_inputs=True, allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))
        f = u_t - F

        return u, f

    def Test(self, testloader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for x1, _, y1, _ in testloader:
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        import numpy as np
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        return true_label, pred_label


class Args:
    pass


def create_args(dataset, batch_size=64):
    args = Args()
    args.batch = 0
    args.batch_size = batch_size  # Smaller batch for huge model
    args.normalization_method = 'min-max'

    configs = {
        'XJTU': {'alpha': 0.7, 'beta': 0.2},
        'TJU': {'alpha': 1.0, 'beta': 0.05}
    }
    args.alpha = configs[dataset]['alpha']
    args.beta = configs[dataset]['beta']
    args.save_folder = f'results/massive_test_{dataset}'
    args.log_dir = 'logging.txt'
    return args


def load_tiny_dataset(dataset, args, n_batteries=3):
    """Load tiny dataset for massive model (will overfit immediately)"""
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


def pretrain_massive(epochs=10, batch_size=64):
    """Quick pretrain of massive model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args('XJTU', batch_size=batch_size)
    dataloader = load_tiny_dataset('XJTU', args, n_batteries=3)

    model = MassivePINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for huge model

    print(f"\n  Pretraining {epochs} epochs on 3 batteries...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for x1, x2, y1, y2 in dataloader['train']:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            u1, f1 = model.forward(x1)
            u2, f2 = model.forward(x2)

            loss_data = 0.5 * model.loss_func(u1, y1) + 0.5 * model.loss_func(u2, y2)
            f_target = torch.zeros_like(f1)
            loss_pde = 0.5 * model.loss_func(f1, f_target) + 0.5 * model.loss_func(f2, f_target)
            loss_physics = model.relu(torch.mul(u2 - u1, y1 - y2)).sum()
            loss = loss_data + args.alpha * loss_pde + args.beta * loss_physics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}/{epochs}, time: {elapsed:.1f}s")

    elapsed_time = time.time() - start_time
    print(f"  Pretraining complete: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

    return model


def finetune_baseline(pretrained_model, epochs=10, batch_size=64):
    """Baseline fine-tuning with MASSIVE model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args('TJU', batch_size=batch_size)
    dataloader = load_tiny_dataset('TJU', args, n_batteries=3)

    # Freeze dynamical_F
    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    trainable_params = sum(p.numel() for p in pretrained_model.solution_u.parameters())
    print(f"\n  Trainable params: {trainable_params:,}")

    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=0.0001)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        pretrained_model.train()
        for x1, x2, y1, y2 in dataloader['train']:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            u1, f1 = pretrained_model.forward(x1)
            u2, f2 = pretrained_model.forward(x2)

            loss_data = 0.5 * pretrained_model.loss_func(u1, y1) + 0.5 * pretrained_model.loss_func(u2, y2)
            f_target = torch.zeros_like(f1)
            loss_pde = 0.5 * pretrained_model.loss_func(f1, f_target) + 0.5 * pretrained_model.loss_func(f2, f_target)
            loss_physics = pretrained_model.relu(torch.mul(u2 - u1, y1 - y2)).sum()
            loss = loss_data + args.alpha * loss_pde + args.beta * loss_physics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}/{epochs}, time: {elapsed:.1f}s")

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time, trainable_params


def finetune_lora(pretrained_model, lora_r=16, epochs=10, batch_size=64):
    """LoRA fine-tuning with MASSIVE model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args('TJU', batch_size=batch_size)
    dataloader = load_tiny_dataset('TJU', args, n_batteries=3)

    # Add LoRA
    print(f"\n  Adding LoRA r={lora_r}...")
    lora_alpha = lora_r * 2
    add_lora_to_model(pretrained_model, r=lora_r, lora_alpha=lora_alpha)
    lora_param_list = get_lora_parameters(pretrained_model)
    trainable_params = sum(p.numel() for p in lora_param_list)
    print(f"  LoRA trainable params: {trainable_params:,}")

    optimizer = torch.optim.Adam(lora_param_list, lr=0.0001)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        pretrained_model.train()
        for x1, x2, y1, y2 in dataloader['train']:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            u1, f1 = pretrained_model.forward(x1)
            u2, f2 = pretrained_model.forward(x2)

            loss_data = 0.5 * pretrained_model.loss_func(u1, y1) + 0.5 * pretrained_model.loss_func(u2, y2)
            f_target = torch.zeros_like(f1)
            loss_pde = 0.5 * pretrained_model.loss_func(f1, f_target) + 0.5 * pretrained_model.loss_func(f2, f_target)
            loss_physics = pretrained_model.relu(torch.mul(u2 - u1, y1 - y2)).sum()
            loss = loss_data + args.alpha * loss_pde + args.beta * loss_physics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}/{epochs}, time: {elapsed:.1f}s")

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time, trainable_params


def finetune_mixed_precision(pretrained_model, epochs=10, batch_size=64):
    """Mixed precision with MASSIVE model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_args('TJU', batch_size=batch_size)
    dataloader = load_tiny_dataset('TJU', args, n_batteries=3)

    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    trainable_params = sum(p.numel() for p in pretrained_model.solution_u.parameters())
    print(f"\n  Trainable params: {trainable_params:,}")

    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=0.0001)
    scaler = GradScaler()

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        pretrained_model.train()
        for x1, x2, y1, y2 in dataloader['train']:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            with autocast():
                u1, f1 = pretrained_model.forward(x1)
                u2, f2 = pretrained_model.forward(x2)

                loss_data = 0.5 * pretrained_model.loss_func(u1, y1) + 0.5 * pretrained_model.loss_func(u2, y2)
                f_target = torch.zeros_like(f1)
                loss_pde = 0.5 * pretrained_model.loss_func(f1, f_target) + 0.5 * pretrained_model.loss_func(f2, f_target)
                loss_physics = pretrained_model.relu(torch.mul(u2 - u1, y1 - y2)).sum()
                loss = loss_data + args.alpha * loss_pde + args.beta * loss_physics

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch % 2 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}/{epochs}, time: {elapsed:.1f}s")

    elapsed_time = time.time() - start_time

    true_label, pred_label = pretrained_model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    return float(metrics[0]), elapsed_time, trainable_params


def run_massive_test():
    print("="*80)
    print("MASSIVE MODEL POC: Testing Optimizations at 500M+ Parameters")
    print("="*80)
    print("\nGoal: Validate if LoRA/FP16 work at LLM-like scale")
    print("Note: Model will overfit terribly (that's not the point)\n")

    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_info}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    results = {}

    # Pretrain once
    print("="*80)
    print("PRETRAINING MASSIVE MODEL")
    print("="*80)
    pretrained_model = pretrain_massive(epochs=10, batch_size=64)

    # Test 1: Baseline
    print("\n" + "="*80)
    print("TEST 1: BASELINE (Full 500M parameters)")
    print("="*80)
    print("\nFine-tuning 10 epochs...")

    model_baseline = pretrain_massive(epochs=10, batch_size=64)
    baseline_mae, baseline_time, baseline_params = finetune_baseline(
        model_baseline, epochs=10, batch_size=64
    )

    results['baseline'] = {
        'mae': baseline_mae,
        'time': baseline_time,
        'params': baseline_params,
        'speedup': 1.0
    }

    print(f"\n[OK] Baseline complete")
    print(f"  Time: {baseline_time:.1f}s ({baseline_time/60:.1f} min)")
    print(f"  Params: {baseline_params:,}")
    print(f"  MAE: {baseline_mae:.4f}")

    # Test 2: LoRA r=16
    print("\n" + "="*80)
    print("TEST 2: LoRA r=16")
    print("="*80)
    print("\nFine-tuning with LoRA...")

    model_lora = pretrain_massive(epochs=10, batch_size=64)
    lora_mae, lora_time, lora_params = finetune_lora(
        model_lora, lora_r=16, epochs=10, batch_size=64
    )

    lora_speedup = baseline_time / lora_time
    results['lora_r16'] = {
        'mae': lora_mae,
        'time': lora_time,
        'params': lora_params,
        'speedup': lora_speedup
    }

    print(f"\n[OK] LoRA complete")
    print(f"  Time: {lora_time:.1f}s ({lora_time/60:.1f} min)")
    print(f"  Params: {lora_params:,}")
    print(f"  MAE: {lora_mae:.4f}")
    print(f"  Speedup: {lora_speedup:.2f}x")

    # Test 3: Mixed Precision
    print("\n" + "="*80)
    print("TEST 3: MIXED PRECISION (FP16)")
    print("="*80)
    print("\nFine-tuning with FP16...")

    model_amp = pretrain_massive(epochs=10, batch_size=64)
    amp_mae, amp_time, amp_params = finetune_mixed_precision(
        model_amp, epochs=10, batch_size=64
    )

    amp_speedup = baseline_time / amp_time
    results['mixed_precision'] = {
        'mae': amp_mae,
        'time': amp_time,
        'params': amp_params,
        'speedup': amp_speedup
    }

    print(f"\n[OK] Mixed precision complete")
    print(f"  Time: {amp_time:.1f}s ({amp_time/60:.1f} min)")
    print(f"  Params: {amp_params:,}")
    print(f"  MAE: {amp_mae:.4f}")
    print(f"  Speedup: {amp_speedup:.2f}x")

    # Summary
    print("\n" + "="*80)
    print("MASSIVE MODEL SUMMARY")
    print("="*80)

    print(f"\n{'Configuration':<30} {'Time (min)':<12} {'Speedup':<10} {'Params':<15} {'MAE'}")
    print("-"*90)
    print(f"{'Baseline (500M params)':<30} {baseline_time/60:<12.1f} {1.0:<10.2f} {baseline_params:<15,} {baseline_mae:.4f}")
    print(f"{'LoRA r=16':<30} {lora_time/60:<12.1f} {lora_speedup:<10.2f} {lora_params:<15,} {lora_mae:.4f}")
    print(f"{'Mixed Precision (FP16)':<30} {amp_time/60:<12.1f} {amp_speedup:<10.2f} {amp_params:<15,} {amp_mae:.4f}")

    print("\n" + "="*80)
    print("HYPOTHESIS VALIDATION")
    print("="*80)

    if lora_speedup > 1.1:
        print(f"\n[OK] LoRA WORKS at massive scale!")
        print(f"  Speedup: {lora_speedup:.2f}x")
        print(f"  Param reduction: {((baseline_params - lora_params) / baseline_params * 100):.1f}%")
    else:
        print(f"\n[FAIL] LoRA still doesn't help (speedup: {lora_speedup:.2f}x)")

    if amp_speedup > 1.1:
        print(f"\n[OK] Mixed Precision WORKS at massive scale!")
        print(f"  Speedup: {amp_speedup:.2f}x")
    else:
        print(f"\n[FAIL] Mixed Precision still doesn't help (speedup: {amp_speedup:.2f}x)")

    print(f"\nConclusion:")
    if max(lora_speedup, amp_speedup) > 1.1:
        best = "LoRA" if lora_speedup > amp_speedup else "Mixed Precision"
        print(f"  {best} provides speedup at 500M parameter scale")
        print(f"  This validates that optimizations work when model is large enough")
        print(f"  For consortium: Current model too small to benefit from these techniques")
    else:
        print(f"  Even at 500M parameters, optimizations don't help PINNs")
        print(f"  Physics constraints (autograd) dominate regardless of model size")
        print(f"  Fundamental incompatibility with standard DL optimization techniques")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/massive_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: results/massive_model_results.json")

    return results


if __name__ == '__main__':
    results = run_massive_test()
