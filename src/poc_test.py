"""
POC Test Script: Quick pretraining + fine-tuning demonstration

This script runs a quick proof-of-concept with:
1. Pretraining on XJTU batch 0 (smallest subset, few epochs)
2. Fine-tuning to TJU batch 1 (cross-chemistry transfer)
3. Timing and performance metrics collection

Usage:
    source env/bin/activate && python3 src/poc_test.py
"""

import argparse
import os
import sys
import time
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loaders import XJTUdata, TJUdata
from src.models import PINN, count_parameters
from src.utils.logging import eval_metrix
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def create_args(stage='pretrain', epochs=20):
    """Create argument object for model initialization"""
    class Args:
        pass

    args = Args()

    # Dataset args
    args.batch = '2C' if stage == 'pretrain' else 1
    args.batch_size = 128  # Smaller for faster training
    args.normalization_method = 'min-max'

    # Training args
    args.epochs = epochs
    args.early_stop = 10
    args.warmup_epochs = 5
    args.warmup_lr = 0.002
    args.lr = 0.01
    args.final_lr = 0.0002
    args.lr_F = 0.001

    # Model args
    args.F_layers_num = 3
    args.F_hidden_dim = 60

    # Loss args
    if stage == 'pretrain':
        args.alpha = 0.7  # XJTU
        args.beta = 0.2
    else:
        args.alpha = 1.0  # TJU
        args.beta = 0.05

    # Output args
    args.save_folder = f'results/poc_{stage}'
    args.log_dir = 'logging.txt'

    return args


def load_xjtu_subset(args, n_batteries=3):
    """Load small subset of XJTU data for quick testing"""
    # Create save folder first
    os.makedirs(args.save_folder, exist_ok=True)

    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=args)

    # Get only first N batteries from 2C batch
    files = sorted([f for f in os.listdir(root) if '2C' in f])
    train_files = [os.path.join(root, f) for f in files[:n_batteries]]

    print(f"  Using {len(train_files)} batteries: {[os.path.basename(f) for f in train_files]}")

    loader = data.read_all(specific_path_list=train_files)
    return {'train': loader['train'], 'valid': loader['valid'], 'test': loader['test']}


def load_tju_subset(args, n_batteries=3):
    """Load small subset of TJU data for quick testing"""
    # Create save folder first
    os.makedirs(args.save_folder, exist_ok=True)

    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)

    # Get only first N batteries from batch 1 (NCA)
    batch_dir = os.path.join(root, 'Dataset_1_NCA_battery')
    files = sorted(os.listdir(batch_dir))[:n_batteries]
    train_files = [os.path.join(batch_dir, f) for f in files]

    print(f"  Using {len(train_files)} batteries: {[os.path.basename(f) for f in train_files]}")

    loader = data.read_all(specific_path_list=train_files)
    return {'train': loader['train'], 'valid': loader['valid'], 'test': loader['test']}


def run_pretraining(args, dataloader):
    """Run pretraining and return metrics"""
    print("\n" + "="*70)
    print("STAGE 1: PRETRAINING")
    print("="*70)

    os.makedirs(args.save_folder, exist_ok=True)

    print(f"\nCreating PINN model...")
    model = PINN(args)

    print(f"\nModel architecture:")
    print(f"  solution_u parameters: {count_parameters(model.solution_u)}")
    print(f"  dynamical_F parameters: {count_parameters(model.dynamical_F)}")

    print(f"\nTraining configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Alpha (PDE): {args.alpha}")
    print(f"  Beta (physics): {args.beta}")

    start_time = time.time()

    print(f"\nStarting pretraining...")
    model.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test']
    )

    elapsed_time = time.time() - start_time

    # Get final metrics
    true_label, pred_label = model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)

    results = {
        'MAE': float(metrics[0]),
        'MAPE': float(metrics[1]),
        'MSE': float(metrics[2]),
        'RMSE': float(metrics[3]),
        'time_seconds': elapsed_time,
        'time_minutes': elapsed_time / 60,
        'epochs': args.epochs
    }

    print(f"\n" + "="*70)
    print(f"PRETRAINING COMPLETED")
    print(f"="*70)
    print(f"Final metrics:")
    print(f"  MAE:  {results['MAE']:.6f}")
    print(f"  RMSE: {results['RMSE']:.6f}")
    print(f"  Time: {results['time_minutes']:.2f} minutes")

    return results, model


def run_finetuning(pretrained_model, args, dataloader, finetune_epochs=20):
    """Run fine-tuning with frozen physics network"""
    print("\n" + "="*70)
    print("STAGE 2: FINE-TUNING (Transfer Learning)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Freeze physics network
    for param in pretrained_model.dynamical_F.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in pretrained_model.solution_u.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in pretrained_model.dynamical_F.parameters())

    print(f"\nFine-tuning configuration:")
    print(f"  Frozen parameters (dynamical_F): {frozen}")
    print(f"  Trainable parameters (solution_u): {trainable}")
    print(f"  Reduction: {frozen/(trainable+frozen)*100:.1f}% of model frozen")

    # New optimizer for solution_u only
    finetune_lr = 0.001
    optimizer = torch.optim.Adam(pretrained_model.solution_u.parameters(), lr=finetune_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs, eta_min=finetune_lr * 0.1
    )

    loss_func = torch.nn.MSELoss()
    relu = torch.nn.ReLU()

    best_valid_mse = float('inf')
    best_metrics = None
    early_stop_counter = 0

    print(f"\nStarting fine-tuning for {finetune_epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, finetune_epochs + 1):
        pretrained_model.train()
        train_loss = 0
        n_batches = 0

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

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == finetune_epochs:
            valid_mse = pretrained_model.Valid(dataloader['valid'])

            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                true_label, pred_label = pretrained_model.Test(dataloader['test'])
                metrics = eval_metrix(pred_label, true_label)
                best_metrics = {
                    'MAE': float(metrics[0]),
                    'MAPE': float(metrics[1]),
                    'MSE': float(metrics[2]),
                    'RMSE': float(metrics[3]),
                    'epoch': epoch
                }
                early_stop_counter = 0
                print(f"Epoch {epoch:3d}: Valid MSE={valid_mse:.6f}, Test MAE={metrics[0]:.4f} [BEST]")
            else:
                early_stop_counter += 1

            if early_stop_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

    elapsed_time = time.time() - start_time
    best_metrics['time_seconds'] = elapsed_time
    best_metrics['time_minutes'] = elapsed_time / 60

    print(f"\n" + "="*70)
    print(f"FINE-TUNING COMPLETED")
    print(f"="*70)
    print(f"Best metrics (epoch {best_metrics['epoch']}):")
    print(f"  MAE:  {best_metrics['MAE']:.6f}")
    print(f"  RMSE: {best_metrics['RMSE']:.6f}")
    print(f"  Time: {best_metrics['time_minutes']:.2f} minutes")

    return best_metrics


def estimate_full_training(pretrain_results, finetune_results, pretrain_epochs_poc, finetune_epochs_poc):
    """Estimate time for full training based on POC results"""
    print("\n" + "="*70)
    print("FULL TRAINING TIME ESTIMATES")
    print("="*70)

    # Estimates for full training
    full_pretrain_epochs = 200
    full_finetune_epochs = 100

    time_per_epoch_pretrain = pretrain_results['time_seconds'] / pretrain_epochs_poc
    time_per_epoch_finetune = finetune_results['time_seconds'] / finetune_epochs_poc

    est_pretrain_time = time_per_epoch_pretrain * full_pretrain_epochs
    est_finetune_time = time_per_epoch_finetune * full_finetune_epochs

    print(f"\nPOC Results:")
    print(f"  Pretraining:  {pretrain_epochs_poc} epochs in {pretrain_results['time_minutes']:.2f} min")
    print(f"  Fine-tuning:  {finetune_epochs_poc} epochs in {finetune_results['time_minutes']:.2f} min")
    print(f"  Total POC:    {(pretrain_results['time_minutes'] + finetune_results['time_minutes']):.2f} min")

    print(f"\nFull Training Estimates (with early stopping):")
    print(f"  Pretraining:  {full_pretrain_epochs} epochs → ~{est_pretrain_time/60:.1f} minutes ({est_pretrain_time/3600:.2f} hours)")
    print(f"  Fine-tuning:  {full_finetune_epochs} epochs → ~{est_finetune_time/60:.1f} minutes ({est_finetune_time/3600:.2f} hours)")
    print(f"  Total:        ~{(est_pretrain_time + est_finetune_time)/60:.1f} minutes ({(est_pretrain_time + est_finetune_time)/3600:.2f} hours)")

    print(f"\nNote: Actual time may be shorter due to early stopping (patience=20-30 epochs)")

    return {
        'pretrain_est_hours': est_pretrain_time / 3600,
        'finetune_est_hours': est_finetune_time / 3600,
        'total_est_hours': (est_pretrain_time + est_finetune_time) / 3600
    }


def main():
    print("="*70)
    print("POC TEST: PINN Pretraining + Fine-tuning Pipeline")
    print("="*70)
    print("\nConfiguration:")
    print("  Source: XJTU batch 0 (2C charge rate, 3 batteries)")
    print("  Target: TJU batch 1 (NCA chemistry, 3 batteries)")
    print("  Pretraining epochs: 20 (vs. 200 full)")
    print("  Fine-tuning epochs: 20 (vs. 100 full)")

    # Stage 1: Pretraining
    print("\n" + "="*70)
    print("STAGE 1: PRETRAINING ON XJTU")
    print("="*70)

    pretrain_args = create_args(stage='pretrain', epochs=20)
    print(f"\nLoading XJTU subset...")
    xjtu_data = load_xjtu_subset(pretrain_args, n_batteries=3)

    pretrain_results, pretrained_model = run_pretraining(pretrain_args, xjtu_data)

    # Stage 2: Fine-tuning
    print("\n" + "="*70)
    print("STAGE 2: FINE-TUNING ON TJU")
    print("="*70)

    finetune_args = create_args(stage='finetune', epochs=20)
    print(f"\nLoading TJU subset...")
    tju_data = load_tju_subset(finetune_args, n_batteries=3)

    finetune_results = run_finetuning(pretrained_model, finetune_args, tju_data, finetune_epochs=20)

    # Estimates
    estimates = estimate_full_training(pretrain_results, finetune_results, 20, 20)

    # Save summary
    summary = {
        'poc_config': {
            'source': 'XJTU batch 0, 3 batteries',
            'target': 'TJU batch 1, 3 batteries',
            'pretrain_epochs': 20,
            'finetune_epochs': 20
        },
        'pretrain_results': pretrain_results,
        'finetune_results': finetune_results,
        'full_training_estimates': estimates
    }

    os.makedirs('results', exist_ok=True)
    with open('results/poc_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "="*70)
    print("POC TEST COMPLETED")
    print("="*70)
    print(f"Summary saved to: results/poc_summary.json")


if __name__ == '__main__':
    main()
