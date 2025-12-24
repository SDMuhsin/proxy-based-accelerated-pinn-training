"""
Simple Sparse Physics Test

Test sparse physics sampling on the existing PINN code.
This is the most practical approach with highest confidence (90%).

Goal: Prove that computing physics on 25-50% of batch gives real speedup.
"""

import os
import sys
import time
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_loaders import XJTUdata, TJUdata
from src.models import PINN
from src.utils.logging import eval_metrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SparsePINN(PINN):
    """PINN with sparse physics sampling"""

    def __init__(self, args, physics_ratio=1.0):
        super().__init__(args)
        self.physics_ratio = physics_ratio

    def forward_sparse(self, xt, compute_physics=True):
        """
        Forward with optional sparse physics

        Args:
            xt: Input batch
            compute_physics: Whether to compute physics this batch

        Returns:
            u: Predictions (full batch)
            f: Physics residuals (sampled or None)
        """
        if not compute_physics or self.physics_ratio == 0:
            # Skip physics entirely
            xt.requires_grad = False
            u = self.solution_u(xt)
            return u, None

        batch_size = xt.shape[0]

        if self.physics_ratio >= 1.0:
            # Standard forward (full physics)
            return super().forward(xt)

        # Sparse physics sampling
        n_physics = max(1, int(batch_size * self.physics_ratio))
        physics_idx = torch.randperm(batch_size, device=device)[:n_physics]

        # Predictions for ALL samples (needed for data loss)
        with torch.no_grad():
            u_all = self.solution_u(xt)

        # Physics for SAMPLED points only
        xt_physics = xt[physics_idx].clone()
        xt_physics.requires_grad = True
        x_physics = xt_physics[:, :-1]
        t_physics = xt_physics[:, -1:]

        u_physics = self.solution_u(torch.cat((x_physics, t_physics), dim=1))

        # Autograd (expensive) only on subset
        u_t = torch.autograd.grad(u_physics.sum(), t_physics,
                                  create_graph=True, only_inputs=True, allow_unused=True)[0]
        u_x = torch.autograd.grad(u_physics.sum(), x_physics,
                                  create_graph=True, only_inputs=True, allow_unused=True)[0]

        F = self.dynamical_F(torch.cat([xt_physics, u_physics, u_x, u_t], dim=1))
        f = u_t - F

        return u_all, f, physics_idx

    def train_one_epoch_sparse(self, epoch, dataloader):
        """Training loop with sparse physics"""
        self.train()

        for iter, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            # Forward with sparse physics
            if self.physics_ratio < 1.0:
                u1, f1, idx1 = self.forward_sparse(x1, compute_physics=True)
                u2, f2, idx2 = self.forward_sparse(x2, compute_physics=True)
            else:
                u1, f1 = self.forward(x1)
                u2, f2 = self.forward(x2)
                idx1 = idx2 = None

            # Data loss (full batch)
            loss1 = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)

            # PDE loss (sampled points)
            f_target = torch.zeros_like(f1)
            loss2 = 0.5 * self.loss_func(f1, f_target) + 0.5 * self.loss_func(f2, f_target)

            # Physics loss (sampled points)
            if self.physics_ratio < 1.0 and idx1 is not None:
                # Use sampled indices
                loss3 = self.relu(torch.mul(u2[idx2] - u1[idx1],
                                           y1[idx1] - y2[idx2])).sum()
            else:
                loss3 = self.relu(torch.mul(u2 - u1, y1 - y2)).sum()

            # Total loss
            loss = loss1 + self.alpha * loss2 + self.beta * loss3

            # Optimization
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            if self.scheduler1 is not None:
                self.scheduler1.step()
            if self.scheduler2 is not None:
                self.scheduler2.step()


def create_args(dataset, epochs=20, batch_size=64, physics_ratio=1.0):
    """Create args for PINN"""
    class Args:
        pass

    args = Args()
    args.save_folder = f'./results/sparse_test_{physics_ratio}'
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


def finetune_with_sparse_physics(pretrained_model, target_dataset, physics_ratio=1.0,
                                 epochs=20, batch_size=64, num_batteries=3):
    """
    Fine-tune with sparse physics sampling

    Args:
        pretrained_model: Pretrained PINN
        target_dataset: Target dataset name
        physics_ratio: Fraction of batch to compute physics on
        epochs: Number of epochs
        batch_size: Batch size
        num_batteries: Number of batteries to use

    Returns:
        mae: Final MAE
        train_time: Training time in seconds
    """
    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size, physics_ratio=physics_ratio)
    dataloader = load_small_dataset(target_dataset, args, num_batteries=num_batteries)

    # Create sparse PINN from pretrained model
    model = SparsePINN(args, physics_ratio=physics_ratio)
    model.solution_u.load_state_dict(pretrained_model.solution_u.state_dict())
    model.dynamical_F.load_state_dict(pretrained_model.dynamical_F.state_dict())

    # Freeze dynamical_F
    for param in model.dynamical_F.parameters():
        param.requires_grad = False

    # Reset optimizer for fine-tuning
    model.optimizer1 = torch.optim.Adam(model.solution_u.parameters(), lr=0.001)
    model.optimizer2 = torch.optim.Adam(model.dynamical_F.parameters(), lr=0.001)
    model.scheduler1 = None
    model.scheduler2 = None

    # Train
    start_time = time.time()
    for epoch in range(epochs):
        model.train_one_epoch_sparse(epoch, dataloader['train'])
    train_time = time.time() - start_time

    # Evaluate
    true_label, pred_label = model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)
    mae = float(metrics[0])

    return mae, train_time


def run_sparse_physics_test():
    """
    Run comprehensive sparse physics test
    """
    print("=" * 80)
    print("SPARSE PHYSICS SAMPLING TEST")
    print("=" * 80)
    print()
    print("Device:", device)
    if device == 'cuda':
        print("GPU:", torch.cuda.get_device_name(0))
    print()
    print("Goal: Test if computing physics on subset of batch gives speedup")
    print()

    # Configuration
    source_dataset = 'XJTU'
    target_dataset = 'TJU'
    num_batteries_pretrain = 3
    num_batteries_finetune = 3
    epochs_pretrain = 20
    epochs_finetune = 20
    batch_size = 64

    # Step 1: Pretrain (shared for all tests)
    print("=" * 80)
    print("STEP 1: PRETRAINING (shared)")
    print("=" * 80)
    print(f"Dataset: {source_dataset}, {num_batteries_pretrain} batteries, {epochs_pretrain} epochs")

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

    results = {}

    # Step 2: Baseline fine-tuning (100% physics)
    print("=" * 80)
    print("TEST 1: BASELINE (100% physics)")
    print("=" * 80)

    mae_baseline, time_baseline = finetune_with_sparse_physics(
        pretrained_model, target_dataset,
        physics_ratio=1.0,
        epochs=epochs_finetune,
        batch_size=batch_size,
        num_batteries=num_batteries_finetune
    )

    results['baseline'] = {
        'physics_ratio': 1.0,
        'mae': mae_baseline,
        'time': time_baseline,
        'speedup': 1.0
    }

    print(f"[OK] Baseline complete")
    print(f"  Time: {time_baseline:.2f}s ({time_baseline/60:.2f} min)")
    print(f"  MAE: {mae_baseline:.4f}")
    print()

    # Step 3: Sparse 50%
    print("=" * 80)
    print("TEST 2: SPARSE 50% (physics on 50% of batch)")
    print("=" * 80)

    mae_sparse50, time_sparse50 = finetune_with_sparse_physics(
        pretrained_model, target_dataset,
        physics_ratio=0.5,
        epochs=epochs_finetune,
        batch_size=batch_size,
        num_batteries=num_batteries_finetune
    )

    speedup_50 = time_baseline / time_sparse50
    results['sparse_50'] = {
        'physics_ratio': 0.5,
        'mae': mae_sparse50,
        'time': time_sparse50,
        'speedup': speedup_50
    }

    print(f"[OK] Sparse 50% complete")
    print(f"  Time: {time_sparse50:.2f}s ({time_sparse50/60:.2f} min)")
    print(f"  MAE: {mae_sparse50:.4f}")
    print(f"  Speedup: {speedup_50:.2f}x")
    print(f"  MAE change: {((mae_sparse50 - mae_baseline) / mae_baseline * 100):+.1f}%")
    print()

    # Step 4: Sparse 25%
    print("=" * 80)
    print("TEST 3: SPARSE 25% (physics on 25% of batch)")
    print("=" * 80)

    mae_sparse25, time_sparse25 = finetune_with_sparse_physics(
        pretrained_model, target_dataset,
        physics_ratio=0.25,
        epochs=epochs_finetune,
        batch_size=batch_size,
        num_batteries=num_batteries_finetune
    )

    speedup_25 = time_baseline / time_sparse25
    results['sparse_25'] = {
        'physics_ratio': 0.25,
        'mae': mae_sparse25,
        'time': time_sparse25,
        'speedup': speedup_25
    }

    print(f"[OK] Sparse 25% complete")
    print(f"  Time: {time_sparse25:.2f}s ({time_sparse25/60:.2f} min)")
    print(f"  MAE: {mae_sparse25:.4f}")
    print(f"  Speedup: {speedup_25:.2f}x")
    print(f"  MAE change: {((mae_sparse25 - mae_baseline) / mae_baseline * 100):+.1f}%")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Configuration':<20} {'Time (s)':<12} {'Speedup':<10} {'MAE':<10} {'MAE Δ':<10}")
    print("-" * 80)
    print(f"{'Baseline (100%)':<20} {time_baseline:>8.2f}     {1.0:>6.2f}x    {mae_baseline:.4f}     +0.0%")
    print(f"{'Sparse 50%':<20} {time_sparse50:>8.2f}     {speedup_50:>6.2f}x    {mae_sparse50:.4f}    {((mae_sparse50-mae_baseline)/mae_baseline*100):>+5.1f}%")
    print(f"{'Sparse 25%':<20} {time_sparse25:>8.2f}     {speedup_25:>6.2f}x    {mae_sparse25:.4f}    {((mae_sparse25-mae_baseline)/mae_baseline*100):>+5.1f}%")
    print()

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    if speedup_25 > 1.15:  # At least 15% faster
        print(f"[OK] SPARSE PHYSICS WORKS!")
        print(f"  Speedup achieved: {speedup_25:.2f}x")
        print()

        if abs((mae_sparse25 - mae_baseline) / mae_baseline) < 0.15:  # Less than 15% degradation
            print(f"[OK] Accuracy acceptable (MAE change < 15%)")
            print()
            print("RECOMMENDATION FOR 5-HOUR WORKLOAD:")
            print(f"  Use 25% physics sampling")
            print(f"  Estimated time: 5.0 hours / {speedup_25:.2f} = {5.0/speedup_25:.2f} hours")
            print()
            if 5.0/speedup_25 <= 2.5:
                print("  [OK] ACHIEVES CONSORTIUM GOAL (< 2.5 hours)!")
            else:
                print(f"  ⚠ Close but not quite 2x (need {5.0/speedup_25:.2f} vs target 2.5 hours)")
                print(f"  Consider combining with adaptive frequency for additional speedup")
        else:
            print(f"⚠ Accuracy degraded significantly ({abs((mae_sparse25-mae_baseline)/mae_baseline*100):.1f}%)")
            print(f"  Recommendation: Use 50% sampling instead")
            print(f"  Estimated time: 5.0 hours / {speedup_50:.2f} = {5.0/speedup_50:.2f} hours")
    else:
        print(f"[FAIL] Insufficient speedup ({speedup_25:.2f}x)")
        print(f"  Sparse physics alone may not achieve 2x goal")
        print(f"  Consider combining with adaptive frequency strategy")

    print()

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/sparse_physics_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: results/sparse_physics_results.json")
    print("=" * 80)

    return results


if __name__ == '__main__':
    run_sparse_physics_test()
