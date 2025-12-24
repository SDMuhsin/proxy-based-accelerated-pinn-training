"""
Fine-tuning script for PINN models (Stage 2: Transfer Learning)

This script handles Stage 2 training where a pretrained model is adapted
to a new battery domain with minimal target data by freezing the physics
network and only updating the solution network.

Usage:
    source env/bin/activate && python3 src/train_finetune.py \
        --source_dataset XJTU --source_batch 0 \
        --target_dataset TJU --target_batch 1 \
        --pretrained_model results/XJTU_batch0/model.pth
"""

import argparse
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loaders import XJTUdata, MITdata, HUSTdata, TJUdata
from src.models import PINN, count_parameters
from src.utils.logging import eval_metrix
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_data(dataset, batch, args):
    """Load dataset based on name and batch"""
    if dataset == 'XJTU':
        root = 'data/XJTU data'
        data = XJTUdata(root=root, args=args)
        batch_names = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
        train_list = []
        test_list = []
        files = os.listdir(root)
        batch_name = batch_names[batch]
        for file in files:
            if batch_name in file:
                if '4' in file or '8' in file:
                    test_list.append(os.path.join(root, file))
                else:
                    train_list.append(os.path.join(root, file))

        train_loader = data.read_all(specific_path_list=train_list)
        test_loader = data.read_all(specific_path_list=test_list)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
        return dataloader

    elif dataset == 'TJU':
        root = 'data/TJU data'
        data = TJUdata(root=root, args=args)
        loader = data.read_one_batch(batch)
        return {'train': loader['train_2'], 'valid': loader['valid_2'], 'test': loader['test_3']}

    elif dataset == 'MIT':
        root = 'data/MIT data'
        data = MITdata(root=root, args=args)
        loader = data.read_one_batch(batch)
        return {'train': loader['train_2'], 'valid': loader['valid_2'], 'test': loader['test_3']}

    elif dataset == 'HUST':
        root = 'data/HUST data'
        data = HUSTdata(root=root, args=args)
        test_ids = [13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        files = os.listdir(root)
        train_list = []
        test_list = []
        for file in files:
            battery_id = int(file.split('-')[1].split('.')[0])
            if battery_id in test_ids:
                test_list.append(os.path.join(root, file))
            else:
                train_list.append(os.path.join(root, file))
        train_loader = data.read_all(specific_path_list=train_list)
        test_loader = data.read_all(specific_path_list=test_list)
        return {'train': train_loader['train_2'], 'valid': train_loader['valid_2'], 'test': test_loader['test_3']}


def finetune_model(model, trainloader, validloader, testloader, args):
    """Fine-tune model with frozen physics network"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Freeze physics network
    for param in model.dynamical_F.parameters():
        param.requires_grad = False

    print(f"\nFine-tuning configuration:")
    print(f"  - Frozen parameters (dynamical_F): {sum(p.numel() for p in model.dynamical_F.parameters())}")
    print(f"  - Trainable parameters (solution_u): {sum(p.numel() for p in model.solution_u.parameters() if p.requires_grad)}")

    # Create new optimizer only for solution_u
    optimizer = torch.optim.Adam(model.solution_u.parameters(), lr=args.finetune_lr)

    # Simple learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs, eta_min=args.finetune_lr * 0.1
    )

    loss_func = torch.nn.MSELoss()
    relu = torch.nn.ReLU()

    best_valid_mse = float('inf')
    best_metrics = None
    early_stop_counter = 0

    print(f"\nStarting fine-tuning for {args.finetune_epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        train_loss = 0
        n_batches = 0

        for x1, x2, y1, y2 in trainloader:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            u1, f1 = model.forward(x1)
            u2, f2 = model.forward(x2)

            # Same loss components as pretraining
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

        train_loss /= n_batches
        scheduler.step()

        # Validation
        if epoch % 5 == 0 or epoch == args.finetune_epochs:
            valid_mse = model.Valid(validloader)

            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                true_label, pred_label = model.Test(testloader)
                metrics = eval_metrix(pred_label, true_label)
                best_metrics = {
                    'MAE': metrics[0],
                    'MAPE': metrics[1],
                    'MSE': metrics[2],
                    'RMSE': metrics[3],
                    'epoch': epoch
                }
                early_stop_counter = 0

                print(f"Epoch {epoch:3d}: Valid MSE={valid_mse:.6f}, Test MAE={metrics[0]:.4f}, RMSE={metrics[3]:.4f} [BEST]")
            else:
                early_stop_counter += 1
                print(f"Epoch {epoch:3d}: Valid MSE={valid_mse:.6f}")

            if early_stop_counter >= args.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    elapsed_time = time.time() - start_time

    return best_metrics, elapsed_time


def get_args():
    parser = argparse.ArgumentParser(description='PINN Fine-tuning for Transfer Learning')

    # Source and target datasets
    parser.add_argument('--source_dataset', type=str, required=True,
                        choices=['XJTU', 'MIT', 'HUST', 'TJU'])
    parser.add_argument('--source_batch', type=int, required=True)
    parser.add_argument('--target_dataset', type=str, required=True,
                        choices=['XJTU', 'MIT', 'HUST', 'TJU'])
    parser.add_argument('--target_batch', type=int, required=True)

    # Model loading
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path to pretrained model.pth file')

    # Fine-tuning hyperparameters
    parser.add_argument('--finetune_epochs', type=int, default=100,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--finetune_lr', type=float, default=0.001,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='Early stopping patience')

    # Model architecture (must match pretrained model)
    parser.add_argument('--F_layers_num', type=int, default=3)
    parser.add_argument('--F_hidden_dim', type=int, default=60)

    # Loss weights (use target dataset defaults)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)

    # Data processing
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--normalization_method', type=str, default='min-max')

    # Output
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='logging.txt')

    # Dummy args for compatibility
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--warmup_lr', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--final_lr', type=float, default=0.0001)
    parser.add_argument('--lr_F', type=float, default=0.001)

    return parser.parse_args()


def main():
    args = get_args()

    # Set default save folder
    if args.save_folder is None:
        args.save_folder = f'results/finetune_{args.source_dataset}{args.source_batch}_to_{args.target_dataset}{args.target_batch}'

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # Set target dataset-specific hyperparameters
    dataset_configs = {
        'XJTU': {'alpha': 0.7, 'beta': 0.2, 'batch_size': 256},
        'MIT': {'alpha': 1.0, 'beta': 0.02, 'batch_size': 512},
        'TJU': {'alpha': 1.0, 'beta': 0.05, 'batch_size': 512},
        'HUST': {'alpha': 0.5, 'beta': 0.2, 'batch_size': 512}
    }

    config = dataset_configs[args.target_dataset]
    if args.alpha is None:
        args.alpha = config['alpha']
    if args.beta is None:
        args.beta = config['beta']
    args.batch_size = config['batch_size']

    print(f"\n{'='*70}")
    print(f"PINN Fine-Tuning: {args.source_dataset} batch {args.source_batch} → {args.target_dataset} batch {args.target_batch}")
    print(f"{'='*70}")

    # Load target dataset
    print(f"\nLoading target dataset ({args.target_dataset})...")
    dataloader = load_data(args.target_dataset, args.target_batch, args)
    print(f"  Train batches: {len(dataloader['train'])}")
    print(f"  Valid batches: {len(dataloader['valid'])}")
    print(f"  Test batches: {len(dataloader['test'])}")

    # Load pretrained model
    print(f"\nLoading pretrained model from {args.pretrained_model}...")
    model = PINN(args)
    model.load_model(args.pretrained_model)
    print(f"  Model loaded successfully")

    # Fine-tune
    best_metrics, elapsed_time = finetune_model(
        model, dataloader['train'], dataloader['valid'], dataloader['test'], args
    )

    # Print results
    print(f"\n{'='*70}")
    print(f"Fine-tuning completed!")
    print(f"{'='*70}")
    print(f"Best results (epoch {best_metrics['epoch']}):")
    print(f"  MAE:  {best_metrics['MAE']:.6f}")
    print(f"  MAPE: {best_metrics['MAPE']:.6f}")
    print(f"  MSE:  {best_metrics['MSE']:.6f}")
    print(f"  RMSE: {best_metrics['RMSE']:.6f}")
    print(f"\nTime: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Results saved to: {args.save_folder}")

    # Save metrics
    with open(os.path.join(args.save_folder, 'finetune_metrics.txt'), 'w') as f:
        f.write(f"Source: {args.source_dataset} batch {args.source_batch}\n")
        f.write(f"Target: {args.target_dataset} batch {args.target_batch}\n")
        f.write(f"Best epoch: {best_metrics['epoch']}\n")
        f.write(f"MAE: {best_metrics['MAE']:.6f}\n")
        f.write(f"MAPE: {best_metrics['MAPE']:.6f}\n")
        f.write(f"MSE: {best_metrics['MSE']:.6f}\n")
        f.write(f"RMSE: {best_metrics['RMSE']:.6f}\n")
        f.write(f"Time: {elapsed_time:.1f} seconds\n")


if __name__ == '__main__':
    main()
