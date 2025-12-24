"""
Pretraining script for PINN models on battery datasets

This script handles Stage 1 training (pretraining) where the model learns
general battery degradation patterns from large labeled datasets.

Usage:
    source env/bin/activate && python3 src/train_pretrain.py --dataset XJTU --batch 0
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loaders import XJTUdata, MITdata, HUSTdata, TJUdata
from src.models import PINN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_xjtu_data(args):
    """Load XJTU dataset with train/valid/test splits"""
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    for file in files:
        if args.batch in file:
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


def load_mit_data(args):
    """Load MIT dataset"""
    root = 'data/MIT data'
    data = MITdata(root=root, args=args)
    loader = data.read_one_batch(args.batch)
    dataloader = {'train': loader['train_2'],
                  'valid': loader['valid_2'],
                  'test': loader['test_3']}
    return dataloader


def load_hust_data(args):
    """Load HUST dataset"""
    root = 'data/HUST data'
    data = HUSTdata(root=root, args=args)

    # Define test battery IDs (from original code)
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
    dataloader = {'train': train_loader['train_2'],
                  'valid': train_loader['valid_2'],
                  'test': test_loader['test_3']}
    return dataloader


def load_tju_data(args):
    """Load TJU dataset"""
    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)
    loader = data.read_one_batch(args.batch)
    dataloader = {'train': loader['train_2'],
                  'valid': loader['valid_2'],
                  'test': loader['test_3']}
    return dataloader


def get_args():
    parser = argparse.ArgumentParser(description='PINN Pretraining for Battery SOH Prediction')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='XJTU',
                        choices=['XJTU', 'MIT', 'HUST', 'TJU'],
                        help='Dataset to use for training')
    parser.add_argument('--batch', type=int, default=0,
                        help='Batch index (XJTU: 0-5, MIT: 1-3, TJU: 1-3, HUST: not used)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--normalization_method', type=str, default='min-max',
                        choices=['min-max', 'z-score'],
                        help='Normalization method for features')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--early_stop', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='Warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='Warmup learning rate')
    parser.add_argument('--lr', type=float, default=0.01, help='Base learning rate')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='Final learning rate')
    parser.add_argument('--lr_F', type=float, default=0.001, help='Learning rate for physics network')

    # Model architecture
    parser.add_argument('--F_layers_num', type=int, default=3, help='Number of layers in physics network')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='Hidden dimension of physics network')

    # Loss function weights
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight for PDE loss (loss = l_data + alpha * l_PDE + beta * l_physics)')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Weight for physics constraint loss')

    # Output paths
    parser.add_argument('--save_folder', type=str, default=None,
                        help='Folder to save results (default: results/{dataset}_batch{batch})')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='Log file name')

    return parser.parse_args()


def main():
    args = get_args()

    # Set default save folder if not specified
    if args.save_folder is None:
        args.save_folder = f'results/{args.dataset}_batch{args.batch}'

    # Create save folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # Dataset-specific configurations
    dataset_configs = {
        'XJTU': {
            'batch_names': ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite'],
            'alpha': 0.7,
            'beta': 0.2,
            'batch_size': 256
        },
        'MIT': {
            'alpha': 1.0,
            'beta': 0.02,
            'batch_size': 512
        },
        'TJU': {
            'alpha': 1.0,
            'beta': 0.05,
            'batch_size': 512
        },
        'HUST': {
            'alpha': 0.5,
            'beta': 0.2,
            'batch_size': 512
        }
    }

    # Apply dataset-specific configurations
    config = dataset_configs[args.dataset]
    args.alpha = config['alpha']
    args.beta = config['beta']
    args.batch_size = config['batch_size']

    # Convert batch index to batch name for XJTU
    if args.dataset == 'XJTU':
        args.batch = config['batch_names'][args.batch]

    # Load data
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'XJTU':
        dataloader = load_xjtu_data(args)
    elif args.dataset == 'MIT':
        dataloader = load_mit_data(args)
    elif args.dataset == 'HUST':
        dataloader = load_hust_data(args)
    elif args.dataset == 'TJU':
        dataloader = load_tju_data(args)

    # Create and train model
    print(f"Creating PINN model...")
    print(f"Hyperparameters: alpha={args.alpha}, beta={args.beta}, batch_size={args.batch_size}")
    pinn = PINN(args)

    print(f"Starting training...")
    print(f"Results will be saved to: {args.save_folder}")
    pinn.Train(trainloader=dataloader['train'],
               validloader=dataloader['valid'],
               testloader=dataloader['test'])

    print(f"Training completed! Results saved to {args.save_folder}")


if __name__ == '__main__':
    main()
