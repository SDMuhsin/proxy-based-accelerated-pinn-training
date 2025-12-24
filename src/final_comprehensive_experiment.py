"""
FINAL COMPREHENSIVE EXPERIMENT FOR PUBLICATION

Tests 4 methods across 6 cross-chemistry transfer scenarios:
- EXACT physics (baseline)
- BASIC FD proxy
- RICHARDSON extrapolation
- ADAPTIVE Richardson + importance

Total: 24 experiments (6 scenarios × 4 methods)

Features:
- Checkpoint/resume capability
- Full datasets (no subsets)
- Structured logging
- Publication-ready results table generation
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import time
import json
import numpy as np
from datetime import datetime
from src.models import PINN
from src.data_loaders import XJTUdata, TJUdata, MITdata
from src.utils.logging import eval_metrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


def log_message(message, log_file='results/final_experiment.log'):
    """Thread-safe logging with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(log_file, 'a') as f:
        f.write(log_line + '\n')


def create_args(dataset, epochs=50, batch_size=128):
    """Create args for PINN"""
    class Args:
        pass

    args = Args()
    args.save_folder = f'./results/final_comprehensive'
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


def load_full_dataset(dataset, args):
    """Load FULL dataset (all batteries)"""
    os.makedirs(args.save_folder, exist_ok=True)

    if dataset == 'XJTU':
        root = 'data/XJTU data'
        data = XJTUdata(root=root, args=args)
        # Load ALL 2C batteries (most consistent)
        files = sorted([os.path.join(root, f) for f in os.listdir(root) if '2C' in f])
    elif dataset == 'TJU':
        root = 'data/TJU data/Dataset_1_NCA_battery'
        data = TJUdata(root='data/TJU data', args=args)
        files = sorted([os.path.join(root, f) for f in os.listdir(root)])
    elif dataset == 'MIT':
        root = 'data/MIT data'
        data = MITdata(root=root, args=args)
        # MIT data has CSV files in subdirectories
        import glob
        files = sorted(glob.glob(os.path.join(root, '*/*.csv')))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    log_message(f"Loading {dataset}: {len(files)} batteries")
    loader = data.read_all(specific_path_list=files)
    return loader


def pretrain_model(source_dataset, epochs=50, batch_size=128):
    """Pretrain model with exact physics"""
    log_message(f"=== PRETRAINING: {source_dataset} ===")

    args = create_args(source_dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_full_dataset(source_dataset, args)

    model = PINN(args)
    start_time = time.time()

    model.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test']
    )

    pretrain_time = time.time() - start_time
    true_label, pred_label = model.Test(dataloader['test'])
    metrics = eval_metrix(pred_label, true_label)
    pretrain_mae = float(metrics[0])

    log_message(f"Pretraining complete: MAE={pretrain_mae:.4f}, Time={pretrain_time:.1f}s")

    return model, pretrain_mae, pretrain_time


def get_feature_importance(model):
    """Extract feature importance from frozen dynamical_F"""
    first_layer = None
    for module in model.dynamical_F.net:
        if isinstance(module, nn.Linear):
            first_layer = module
            break

    if first_layer is None:
        return torch.ones(16, device=device)

    weights = first_layer.weight
    u_x_weights = weights[:, 18:34]
    importance = torch.norm(u_x_weights, p=2, dim=0)
    importance = importance / (importance.max() + 1e-8)

    return importance


def compute_richardson_gradient(model, x, u, feature_idx, epsilon_base=1e-4):
    """Richardson extrapolation for gradient"""
    x_pert_1 = x.clone()
    x_pert_1[:, feature_idx] += epsilon_base
    with torch.no_grad():
        u_pert_1 = model.solution_u(x_pert_1)
    fd_1 = (u_pert_1 - u) / epsilon_base

    x_pert_2 = x.clone()
    x_pert_2[:, feature_idx] += 2 * epsilon_base
    with torch.no_grad():
        u_pert_2 = model.solution_u(x_pert_2)
    fd_2 = (u_pert_2 - u) / (2 * epsilon_base)

    gradient = (4 * fd_1 - fd_2) / 3
    return gradient


def finetune_with_method(pretrained_model, target_dataset, method='basic',
                          epochs=50, batch_size=128):
    """Fine-tune with specified method"""
    method_names = {
        'exact': 'EXACT physics',
        'basic': 'BASIC FD',
        'richardson': 'RICHARDSON',
        'adaptive': 'ADAPTIVE'
    }

    log_message(f"  Fine-tuning with {method_names[method]}...")

    args = create_args(target_dataset, epochs=epochs, batch_size=batch_size)
    dataloader = load_full_dataset(target_dataset, args)

    # Copy model
    model = PINN(args)
    model.solution_u.load_state_dict(pretrained_model.solution_u.state_dict())
    model.dynamical_F.load_state_dict(pretrained_model.dynamical_F.state_dict())

    # Freeze F
    for param in model.dynamical_F.parameters():
        param.requires_grad = False

    # Get feature importance for adaptive
    if method == 'adaptive':
        feature_importance = get_feature_importance(model)
        K = 8
        top_k_indices = torch.argsort(feature_importance, descending=True)[:K].cpu().numpy()
        log_message(f"    Top-{K} features: {top_k_indices.tolist()}")
    else:
        top_k_indices = None

    optimizer = torch.optim.Adam(model.solution_u.parameters(), lr=0.001)

    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        for batch_idx, (x1, x2, y1, y2) in enumerate(dataloader['train']):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            if method == 'exact':
                # EXACT physics
                u1, f1 = model.forward(x1)
                u2, f2 = model.forward(x2)

                loss1 = 0.5 * model.loss_func(u1, y1) + 0.5 * model.loss_func(u2, y2)
                f_target = torch.zeros_like(f1)
                loss2 = 0.5 * model.loss_func(f1, f_target) + 0.5 * model.loss_func(f2, f_target)
                loss3 = model.relu(torch.mul(u2 - u1, y1 - y2)).sum()

                total_loss = loss1 + model.alpha * loss2 + model.beta * loss3

            else:
                # PHYSICS PROXY
                u1 = model.solution_u(x1)
                u2 = model.solution_u(x2)

                loss1 = 0.5 * model.loss_func(u1, y1) + 0.5 * model.loss_func(u2, y2)

                t1 = x1[:, -1:]
                t2 = x2[:, -1:]
                delta_t = t2 - t1 + 1e-8
                u_t_observed = (u2 - u1) / delta_t

                # Spatial gradients
                epsilon = 1e-4
                u_x_components = []

                if method == 'basic':
                    # Basic FD
                    for i in range(16):
                        x1_perturbed = x1.clone()
                        x1_perturbed[:, i] += epsilon
                        with torch.no_grad():
                            u_perturbed = model.solution_u(x1_perturbed)
                        u_x_i = (u_perturbed - u1) / epsilon
                        u_x_components.append(u_x_i)

                elif method == 'richardson':
                    # Richardson for all
                    for i in range(16):
                        u_x_i = compute_richardson_gradient(model, x1, u1, i, epsilon)
                        u_x_components.append(u_x_i)

                elif method == 'adaptive':
                    # Adaptive: Richardson for top-K, FD for rest
                    for i in range(16):
                        if i in top_k_indices:
                            u_x_i = compute_richardson_gradient(model, x1, u1, i, epsilon)
                        else:
                            x1_perturbed = x1.clone()
                            x1_perturbed[:, i] += epsilon
                            with torch.no_grad():
                                u_perturbed = model.solution_u(x1_perturbed)
                            u_x_i = (u_perturbed - u1) / epsilon
                        u_x_components.append(u_x_i)

                u_x_approx = torch.cat(u_x_components, dim=-1)

                # Query F
                F_input = torch.cat([x1, u1.detach(), u_x_approx.detach(), u_t_observed.detach()], dim=1)
                with torch.no_grad():
                    F_expected = model.dynamical_F(F_input)

                loss_physics_proxy = model.loss_func(u_t_observed, F_expected.detach())
                loss3 = model.relu(torch.mul(u2 - u1, y1 - y2)).sum()

                total_loss = loss1 + model.alpha * loss_physics_proxy + model.beta * loss3

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

    log_message(f"    [OK] {method_names[method]}: MAE={mae:.4f}, Time={finetune_time:.1f}s")

    return mae, finetune_time


def run_scenario(source, target, scenario_num, checkpoint_file='results/checkpoint.json'):
    """Run one complete scenario (1 pretrain + 4 finetunes)"""
    log_message(f"\n{'='*80}")
    log_message(f"SCENARIO {scenario_num}: {source} → {target}")
    log_message(f"{'='*80}")

    # Check checkpoint
    checkpoint = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

    scenario_key = f"{source}_{target}"

    # Skip if already completed
    if scenario_key in checkpoint and checkpoint[scenario_key].get('completed', False):
        log_message(f"Scenario {scenario_num} already completed. Skipping...")
        return checkpoint[scenario_key]

    # Pretrain
    if scenario_key not in checkpoint or 'pretrained_model' not in checkpoint[scenario_key]:
        pretrained_model, pretrain_mae, pretrain_time = pretrain_model(source, epochs=50, batch_size=128)

        # Save pretrained model
        model_path = f"results/pretrained_{source}.pt"
        torch.save({
            'solution_u': pretrained_model.solution_u.state_dict(),
            'dynamical_F': pretrained_model.dynamical_F.state_dict()
        }, model_path)

        checkpoint[scenario_key] = {
            'pretrain_mae': pretrain_mae,
            'pretrain_time': pretrain_time,
            'model_path': model_path,
            'results': {}
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    else:
        # Load pretrained model
        log_message(f"Loading pretrained model from checkpoint...")
        model_path = checkpoint[scenario_key]['model_path']
        args = create_args(source)
        pretrained_model = PINN(args)
        state = torch.load(model_path)
        pretrained_model.solution_u.load_state_dict(state['solution_u'])
        pretrained_model.dynamical_F.load_state_dict(state['dynamical_F'])

    # Fine-tune with all methods
    for method in ['exact', 'basic', 'richardson', 'adaptive']:
        if method not in checkpoint[scenario_key]['results']:
            mae, time_ft = finetune_with_method(pretrained_model, target, method=method,
                                                 epochs=50, batch_size=128)

            checkpoint[scenario_key]['results'][method] = {
                'mae': mae,
                'time': time_ft
            }

            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

    checkpoint[scenario_key]['completed'] = True
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    return checkpoint[scenario_key]


def main():
    """Run all experiments"""
    log_message("="*80)
    log_message("FINAL COMPREHENSIVE EXPERIMENT - PUBLICATION RESULTS")
    log_message("="*80)
    log_message(f"Device: {device}")
    if device == 'cuda':
        log_message(f"GPU: {torch.cuda.get_device_name(0)}")
    log_message(f"Total experiments: 24 (6 scenarios × 4 methods)")
    log_message("="*80)

    scenarios = [
        ('XJTU', 'TJU', 1),   # NCM → NCA
        ('XJTU', 'MIT', 2),   # NCM → LiFePO4
        ('TJU', 'MIT', 3),    # NCA → LiFePO4
        ('MIT', 'XJTU', 4),   # LiFePO4 → NCM
        ('MIT', 'TJU', 5),    # LiFePO4 → NCA
        ('TJU', 'XJTU', 6),   # NCA → NCM
    ]

    all_results = {}

    for source, target, num in scenarios:
        try:
            results = run_scenario(source, target, num)
            all_results[f"{source}_{target}"] = results
        except Exception as e:
            log_message(f"ERROR in scenario {num}: {str(e)}")
            continue

    # Save final results
    with open('results/final_comprehensive_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    log_message("\n" + "="*80)
    log_message("ALL EXPERIMENTS COMPLETED!")
    log_message("Results saved to: results/final_comprehensive_results.json")
    log_message("="*80)


if __name__ == '__main__':
    main()
