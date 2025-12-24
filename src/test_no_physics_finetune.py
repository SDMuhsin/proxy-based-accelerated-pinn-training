"""
Test: Skip Physics Constraints During Fine-Tuning

Hypothesis: Physics constraints learned during pretraining carry over.
           We can skip them during fine-tuning for 2x speedup.

This is our BEST CHANCE at real speedup without complex changes.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import time
import json
from src.models import PINN
from src.data_loaders import XJTUdata, TJUdata
from src.utils.logging import eval_metrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_args(dataset, epochs=20, batch_size=64):
    """Create args for PINN"""
    class Args:
        pass

    args = Args()
    args.save_folder = f'./results/no_physics_test'
    args.log_dir = 'logs'
    args.dataset = dataset
    args.source_dataset = 'XJTU'
    args.target_dataset = 'TJU'
    args.num_epochs = epochs
    args.batch_size = batch_size
    args.batch = 0
    args.normalization_method = 'min-max'
    args.epochs = epochs
    args.early_stop = 15
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


def load_small_dataset(dataset, args, num_batteries=3):
    """Load small dataset for testing"""
    os.makedirs(args.save_folder, exist_ok=True)

    if dataset == 'XJTU':
        root = 'data/XJTU data'
        data = XJTUdata(root=root, args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root) if '2C' in f])[:num_batteries]
    elif dataset == 'TJU':
        root = 'data/TJU data/Dataset_1_NCA_battery'
        data = TJUdata(root='data/TJU data', args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root)])[:num_batteries]

    loader = data.read_all(specific_path_list=files)
    return loader


def finetune_with_physics_control(pretrained_model, target_dataset,
                                  use_physics=True,
                                  epochs=20, batch_size=64, num_batteries=3):
    """
    Fine-tune with optional physics constraints.

    Args:
        pretrained_model: Pretrained PINN
        target_dataset: Target dataset
        use_physics: If True, use full physics. If False, data loss only.
        epochs: Number of epochs
        batch_size: Batch size
        num_batteries: Number of batteries

    Returns:
        mae: Final MAE
        train_time: Training time
    """
    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_small_dataset(target_dataset, args, num_batteries=num_batteries)

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
                # STANDARD: Full physics constraints (slow)
                u1, f1 = model.forward(x1)  # Includes autograd
                u2, f2 = model.forward(x2)

                # Data loss
                loss1 = 0.5 * model.loss_func(u1, y1) + 0.5 * model.loss_func(u2, y2)

                # PDE loss
                f_target = torch.zeros_like(f1)
                loss2 = 0.5 * model.loss_func(f1, f_target) + 0.5 * model.loss_func(f2, f_target)

                # Physics loss
                loss3 = model.relu(torch.mul(u2 - u1, y1 - y2)).sum()

                # Total loss
                total_loss = loss1 + model.alpha * loss2 + model.beta * loss3

            else:
                # FAST: Data loss only (no physics!)
                # Just compute predictions, no gradient computation
                u1 = model.solution_u(x1)  # Direct forward, no autograd
                u2 = model.solution_u(x2)

                # Data loss only
                loss1 = 0.5 * model.loss_func(u1, y1) + 0.5 * model.loss_func(u2, y2)

                # Physics loss on predictions (not gradients)
                # We can still check if u2 < u1 (monotonicity on values)
                loss3 = model.relu(torch.mul(u2 - u1, y1 - y2)).sum()

                # Total loss (no PDE constraint!)
                total_loss = loss1 + model.beta * loss3

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    train_time = time.time() - start_time

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

    return mae, train_time


def run_test():
    """
    Compare fine-tuning with vs without physics constraints.
    """
    print("=" * 80)
    print("TEST: SKIP PHYSICS DURING FINE-TUNING")
    print("=" * 80)
    print()
    print("Device:", device)
    if device == 'cuda':
        print("GPU:", torch.cuda.get_device_name(0))
    print()
    print("Hypothesis: Physics learned during pretraining carries over")
    print("           Can skip during fine-tuning for speedup")
    print()

    # Setup
    source_dataset = 'XJTU'
    target_dataset = 'TJU'
    num_batteries_pretrain = 3
    num_batteries_finetune = 3
    epochs_pretrain = 20
    epochs_finetune = 20
    batch_size = 64

    # Pretrain (shared, with full physics)
    print("=" * 80)
    print("PRETRAINING (with physics - done once)")
    print("=" * 80)
    print()

    args = create_args(source_dataset, epochs=epochs_pretrain, batch_size=batch_size)
    dataloader = load_small_dataset(source_dataset, args, num_batteries=num_batteries_pretrain)

    pretrained_model = PINN(args)
    pretrained_model.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test']
    )
    print("[OK] Pretraining complete")
    print()

    # Test 1: Fine-tune WITH physics (baseline)
    print("=" * 80)
    print("TEST 1: FINE-TUNE WITH PHYSICS (baseline)")
    print("=" * 80)
    print()

    mae_with_physics, time_with_physics = finetune_with_physics_control(
        pretrained_model, target_dataset,
        use_physics=True,
        epochs=epochs_finetune,
        batch_size=batch_size,
        num_batteries=num_batteries_finetune
    )

    print(f"[OK] With physics complete")
    print(f"  Time: {time_with_physics:.2f}s ({time_with_physics/60:.2f} min)")
    print(f"  MAE: {mae_with_physics:.4f}")
    print()

    # Test 2: Fine-tune WITHOUT physics (proposed)
    print("=" * 80)
    print("TEST 2: FINE-TUNE WITHOUT PHYSICS (proposed)")
    print("=" * 80)
    print()

    mae_without_physics, time_without_physics = finetune_with_physics_control(
        pretrained_model, target_dataset,
        use_physics=False,
        epochs=epochs_finetune,
        batch_size=batch_size,
        num_batteries=num_batteries_finetune
    )

    print(f"[OK] Without physics complete")
    print(f"  Time: {time_without_physics:.2f}s ({time_without_physics/60:.2f} min)")
    print(f"  MAE: {mae_without_physics:.4f}")
    print()

    # Results
    speedup = time_with_physics / time_without_physics
    mae_change = ((mae_without_physics - mae_with_physics) / mae_with_physics) * 100

    results = {
        'with_physics': {
            'time': time_with_physics,
            'mae': mae_with_physics,
            'speedup': 1.0
        },
        'without_physics': {
            'time': time_without_physics,
            'mae': mae_without_physics,
            'speedup': speedup
        }
    }

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"{'Method':<25} {'Time (s)':<12} {'Speedup':<10} {'MAE':<10} {'MAE Δ':<10}")
    print("-" * 80)
    print(f"{'With Physics':<25} {time_with_physics:>8.2f}     {1.0:>6.2f}x    {mae_with_physics:.4f}     +0.0%")
    print(f"{'Without Physics':<25} {time_without_physics:>8.2f}     {speedup:>6.2f}x    {mae_without_physics:.4f}    {mae_change:>+5.1f}%")
    print()

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    if speedup > 1.5:
        print(f"[OK] SKIPPING PHYSICS WORKS!")
        print(f"  Speedup: {speedup:.2f}x")
        print()

        if abs(mae_change) < 15:
            print(f"[OK] Accuracy acceptable (MAE change: {mae_change:+.1f}%)")
            print()
            print("RECOMMENDATION FOR 5-HOUR WORKLOAD:")
            print(f"  Skip physics constraints during fine-tuning")
            print(f"  Estimated time: 5.0 hours / {speedup:.2f} = {5.0/speedup:.2f} hours")
            print()

            if 5.0/speedup <= 2.5:
                print("  [OK] ACHIEVES CONSORTIUM GOAL (< 2.5 hours)!")
            else:
                print(f"  ⚠ Close but not quite 2x")
                print(f"  Might need to combine with other techniques")

        else:
            print(f"⚠ Accuracy degraded significantly ({abs(mae_change):.1f}%)")
            print(f"  This approach may not be viable")
            print(f"  Physics constraints are critical for fine-tuning")

    else:
        print(f"[FAIL] Insufficient speedup ({speedup:.2f}x)")
        print(f"  Expected ~2x from removing 40% physics overhead")
        print(f"  Other factors may be limiting performance")

    print()

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/no_physics_finetune_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: results/no_physics_finetune_results.json")
    print("=" * 80)

    return results


if __name__ == '__main__':
    run_test()
