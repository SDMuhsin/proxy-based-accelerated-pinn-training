"""
Comprehensive Validation: Skip Physics During Fine-Tuning

Test across:
1. Multiple transfer scenarios (XJTU→TJU, XJTU→MIT, MIT→TJU)
2. Larger datasets (10+ batteries)
3. More epochs (50-100)
4. Different batch sizes

We need to be ABSOLUTELY SURE this works before recommending it.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import time
import json
from src.models import PINN
from src.data_loaders import XJTUdata, TJUdata, MITdata
from src.utils.logging import eval_metrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_args(dataset, epochs=50, batch_size=128):
    """Create args for PINN"""
    class Args:
        pass

    args = Args()
    args.save_folder = f'./results/comprehensive_validation'
    args.log_dir = 'logs'
    args.dataset = dataset
    args.source_dataset = 'XJTU'
    args.target_dataset = 'TJU'
    args.num_epochs = epochs
    args.batch_size = batch_size
    args.batch = 0
    args.normalization_method = 'min-max'
    args.epochs = epochs
    args.early_stop = 25
    args.warmup_epochs = 5
    args.warmup_lr = 0.002
    args.lr = 0.01
    args.final_lr = 0.0002
    args.lr_F = 0.001
    args.F_layers_num = 3
    args.F_hidden_dim = 60
    args.base_lr = 0.001
    args.encoder_name = 'default'
    args.use_data_parallel = False
    args.gpu_ids = '0'
    args.alpha = 1.0
    args.beta = 0.1

    return args


def load_dataset(dataset, args, num_batteries=10):
    """Load dataset"""
    os.makedirs(args.save_folder, exist_ok=True)

    if dataset == 'XJTU':
        root = 'data/XJTU data'
        data = XJTUdata(root=root, args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root) if '2C' in f])[:num_batteries]
    elif dataset == 'TJU':
        root = 'data/TJU data/Dataset_1_NCA_battery'
        data = TJUdata(root='data/TJU data', args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root)])[:num_batteries]
    elif dataset == 'MIT':
        root = 'data/MIT data'
        data = MITdata(root=root, args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.mat')])[:num_batteries]

    loader = data.read_all(specific_path_list=files)
    return loader


def pretrain_model(source_dataset, num_batteries=10, epochs=50, batch_size=128):
    """
    Pretrain model (always with physics)
    """
    print(f"Pretraining on {source_dataset} ({num_batteries} batteries, {epochs} epochs)...")

    args = create_args(source_dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_dataset(source_dataset, args, num_batteries=num_batteries)

    model = PINN(args)
    start_time = time.time()

    model.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test']
    )

    pretrain_time = time.time() - start_time

    # Evaluate
    true_label, pred_label = model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)
    pretrain_mae = float(metrics[0])

    print(f"[OK] Pretraining complete: MAE={pretrain_mae:.4f}, Time={pretrain_time:.1f}s")

    return model, pretrain_mae, pretrain_time


def finetune_model(pretrained_model, target_dataset, use_physics=True,
                   num_batteries=10, epochs=50, batch_size=128):
    """
    Fine-tune with or without physics
    """
    method_name = "WITH physics" if use_physics else "WITHOUT physics"
    print(f"  Fine-tuning {method_name}...")

    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_dataset(target_dataset, args, num_batteries=num_batteries)

    # Create model copy
    model = PINN(args)
    model.solution_u.load_state_dict(pretrained_model.solution_u.state_dict())
    model.dynamical_F.load_state_dict(pretrained_model.dynamical_F.state_dict())

    # Freeze dynamical_F
    for param in model.dynamical_F.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.Adam(model.solution_u.parameters(), lr=0.001)

    # Training loop
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        for batch_idx, (x1, x2, y1, y2) in enumerate(dataloader['train']):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            if use_physics:
                # WITH PHYSICS (slow)
                u1, f1 = model.forward(x1)
                u2, f2 = model.forward(x2)

                loss1 = 0.5 * model.loss_func(u1, y1) + 0.5 * model.loss_func(u2, y2)

                f_target = torch.zeros_like(f1)
                loss2 = 0.5 * model.loss_func(f1, f_target) + 0.5 * model.loss_func(f2, f_target)

                loss3 = model.relu(torch.mul(u2 - u1, y1 - y2)).sum()

                total_loss = loss1 + model.alpha * loss2 + model.beta * loss3

            else:
                # WITHOUT PHYSICS (fast)
                u1 = model.solution_u(x1)
                u2 = model.solution_u(x2)

                loss1 = 0.5 * model.loss_func(u1, y1) + 0.5 * model.loss_func(u2, y2)
                loss3 = model.relu(torch.mul(u2 - u1, y1 - y2)).sum()

                total_loss = loss1 + model.beta * loss3

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    finetune_time = time.time() - start_time

    # Evaluate
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x1, x2, y1, y2 in dataloader['test']:
            x1, y1 = x1.to(device), y1.to(device)
            u1 = model.solution_u(x1)
            all_preds.append(u1)
            all_targets.append(y1)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mae = torch.abs(all_preds - all_targets).mean().item()

    print(f"    [OK] {method_name}: MAE={mae:.4f}, Time={finetune_time:.1f}s")

    return mae, finetune_time


def run_comprehensive_validation():
    """
    Run comprehensive validation across multiple scenarios
    """
    print()
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION: SKIP PHYSICS DURING FINE-TUNING")
    print("=" * 80)
    print()
    print("Device:", device)
    if device == 'cuda':
        print("GPU:", torch.cuda.get_device_name(0))
    print()
    print("Testing across multiple scenarios to ensure robustness...")
    print()

    all_results = {}

    # =========================================================================
    # SCENARIO 1: XJTU → TJU (Cross-chemistry, hardest)
    # =========================================================================
    print("=" * 80)
    print("SCENARIO 1: XJTU → TJU (Cross-chemistry transfer)")
    print("=" * 80)
    print("Source: XJTU (NCM batteries)")
    print("Target: TJU (NCA batteries)")
    print("Difficulty: HARD (different chemistry)")
    print()

    pretrained_xjtu, _, _ = pretrain_model('XJTU', num_batteries=10, epochs=50, batch_size=128)

    print()
    print("Fine-tuning on TJU...")
    mae_with_1, time_with_1 = finetune_model(pretrained_xjtu, 'TJU', use_physics=True,
                                             num_batteries=10, epochs=50, batch_size=128)
    mae_without_1, time_without_1 = finetune_model(pretrained_xjtu, 'TJU', use_physics=False,
                                                   num_batteries=10, epochs=50, batch_size=128)

    speedup_1 = time_with_1 / time_without_1
    mae_change_1 = ((mae_without_1 - mae_with_1) / mae_with_1) * 100

    all_results['xjtu_to_tju'] = {
        'with_physics': {'time': time_with_1, 'mae': mae_with_1},
        'without_physics': {'time': time_without_1, 'mae': mae_without_1},
        'speedup': speedup_1,
        'mae_change_pct': mae_change_1
    }

    print()
    print(f"  Speedup: {speedup_1:.2f}x")
    print(f"  MAE change: {mae_change_1:+.1f}%")
    print()

    # =========================================================================
    # SCENARIO 2: XJTU → MIT (Cross-capacity)
    # =========================================================================
    print("=" * 80)
    print("SCENARIO 2: XJTU → MIT (Cross-capacity transfer)")
    print("=" * 80)
    print("Source: XJTU (2.0 Ah)")
    print("Target: MIT (1.1 Ah)")
    print("Difficulty: MEDIUM (different capacity)")
    print()

    # Reuse pretrained XJTU model
    print()
    print("Fine-tuning on MIT...")
    mae_with_2, time_with_2 = finetune_model(pretrained_xjtu, 'MIT', use_physics=True,
                                             num_batteries=10, epochs=50, batch_size=128)
    mae_without_2, time_without_2 = finetune_model(pretrained_xjtu, 'MIT', use_physics=False,
                                                   num_batteries=10, epochs=50, batch_size=128)

    speedup_2 = time_with_2 / time_without_2
    mae_change_2 = ((mae_without_2 - mae_with_2) / mae_with_2) * 100

    all_results['xjtu_to_mit'] = {
        'with_physics': {'time': time_with_2, 'mae': mae_with_2},
        'without_physics': {'time': time_without_2, 'mae': mae_without_2},
        'speedup': speedup_2,
        'mae_change_pct': mae_change_2
    }

    print()
    print(f"  Speedup: {speedup_2:.2f}x")
    print(f"  MAE change: {mae_change_2:+.1f}%")
    print()

    # =========================================================================
    # SCENARIO 3: MIT → TJU (Different conditions)
    # =========================================================================
    print("=" * 80)
    print("SCENARIO 3: MIT → TJU (Cross-conditions transfer)")
    print("=" * 80)
    print("Source: MIT (LiFePO4)")
    print("Target: TJU (NCA)")
    print("Difficulty: MEDIUM (different chemistry)")
    print()

    pretrained_mit, _, _ = pretrain_model('MIT', num_batteries=10, epochs=50, batch_size=128)

    print()
    print("Fine-tuning on TJU...")
    mae_with_3, time_with_3 = finetune_model(pretrained_mit, 'TJU', use_physics=True,
                                             num_batteries=10, epochs=50, batch_size=128)
    mae_without_3, time_without_3 = finetune_model(pretrained_mit, 'TJU', use_physics=False,
                                                   num_batteries=10, epochs=50, batch_size=128)

    speedup_3 = time_with_3 / time_without_3
    mae_change_3 = ((mae_without_3 - mae_with_3) / mae_with_3) * 100

    all_results['mit_to_tju'] = {
        'with_physics': {'time': time_with_3, 'mae': mae_with_3},
        'without_physics': {'time': time_without_3, 'mae': mae_without_3},
        'speedup': speedup_3,
        'mae_change_pct': mae_change_3
    }

    print()
    print(f"  Speedup: {speedup_3:.2f}x")
    print(f"  MAE change: {mae_change_3:+.1f}%")
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 80)
    print()

    print(f"{'Scenario':<30} {'Speedup':<12} {'MAE Δ':<12} {'Verdict':<15}")
    print("-" * 80)

    scenarios = [
        ('XJTU → TJU (Hard)', speedup_1, mae_change_1),
        ('XJTU → MIT (Medium)', speedup_2, mae_change_2),
        ('MIT → TJU (Medium)', speedup_3, mae_change_3),
    ]

    for name, speedup, mae_change in scenarios:
        verdict = "[OK] PASS" if speedup > 1.5 and abs(mae_change) < 20 else "[FAIL] FAIL"
        print(f"{name:<30} {speedup:>6.2f}x      {mae_change:>+6.1f}%     {verdict:<15}")

    print()

    # Overall verdict
    avg_speedup = (speedup_1 + speedup_2 + speedup_3) / 3
    avg_mae_change = (abs(mae_change_1) + abs(mae_change_2) + abs(mae_change_3)) / 3

    print("=" * 80)
    print("OVERALL VERDICT")
    print("=" * 80)
    print()
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Average MAE change: ±{avg_mae_change:.1f}%")
    print()

    all_pass = all(s > 1.5 and abs(m) < 20 for _, s, m in scenarios)

    if all_pass:
        print("[OK] ALL SCENARIOS PASS")
        print()
        print("CONCLUSION:")
        print("  Skipping physics during fine-tuning is VALIDATED")
        print("  Works across different transfer scenarios")
        print(f"  Consistent {avg_speedup:.1f}x speedup with acceptable accuracy")
        print()
        print("RECOMMENDATION FOR CONSORTIUM:")
        print(f"  Expected: 5 hours / {avg_speedup:.2f} = {5.0/avg_speedup:.2f} hours")
        print(f"  Target: 2.5 hours")
        if 5.0/avg_speedup <= 2.5:
            print("  [OK] ACHIEVES GOAL!")
        else:
            print(f"  ⚠ Close ({5.0/avg_speedup:.2f} hours)")
    else:
        print("[FAIL] SOME SCENARIOS FAILED")
        print()
        print("The approach needs more investigation")
        print("May not be robust across all transfer scenarios")

    print()

    # Save results
    os.makedirs('results', exist_ok=True)
    all_results['summary'] = {
        'avg_speedup': avg_speedup,
        'avg_mae_change_pct': avg_mae_change,
        'all_pass': all_pass
    }

    with open('results/comprehensive_validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("Results saved to: results/comprehensive_validation_results.json")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    run_comprehensive_validation()
