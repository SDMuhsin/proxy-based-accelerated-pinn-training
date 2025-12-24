"""
Test Fast Autograd Implementation

CRITICAL: Test FIRST, make claims LATER (learning from previous mistakes)

Goals:
1. Verify fast autograd is actually faster (measure, don't assume)
2. Check accuracy degradation is acceptable
3. Test on real PINN training (not just microbenchmark)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import time
import json
from src.fast_autograd import FastAutograd, AutogradBenchmark, FastPINN, convert_pinn_to_fast, FastAutogradConfig
from src.models import PINN, Solution_u
from src.data_loaders import XJTUdata, TJUdata
from src.utils.logging import eval_metrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_microbenchmark():
    """
    Test 1: Microbenchmark - Pure autograd speed comparison
    """
    print("=" * 80)
    print("TEST 1: MICROBENCHMARK")
    print("=" * 80)
    print()
    print("Comparing standard autograd vs fast autograd...")
    print()

    # Create a dummy model (same size as PINN solution_u)
    model = Solution_u().to(device)

    # Run benchmark
    results = AutogradBenchmark.benchmark(
        model,
        batch_size=64,
        n_dims=17,
        n_runs=100,
        device=device
    )

    print(f"Standard autograd: {results['time_standard']:.3f} ms/batch")
    print(f"Fast autograd:     {results['time_fast']:.3f} ms/batch")
    print(f"Speedup:           {results['speedup']:.2f}x")
    print()
    print("Accuracy (MAE difference from standard autograd):")
    print(f"  u:   {results['accuracy']['u_mae']:.2e}")
    print(f"  u_x: {results['accuracy']['u_x_mae']:.2e}")
    print(f"  u_t: {results['accuracy']['u_t_mae']:.2e}")
    print()

    if results['speedup'] > 1.5:
        print(f"[OK] Fast autograd is {results['speedup']:.2f}x faster!")
    elif results['speedup'] > 1.1:
        print(f"[OK] Fast autograd is faster ({results['speedup']:.2f}x), but modest gain")
    else:
        print(f"[FAIL] Fast autograd NOT significantly faster ({results['speedup']:.2f}x)")

    print()
    return results


def create_args(dataset, epochs=20, batch_size=64):
    """Create args for PINN"""
    class Args:
        pass

    args = Args()
    args.save_folder = f'./results/fast_autograd_test'
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


def train_pinn_with_fast_autograd(pretrained_model, target_dataset,
                                  use_fast_autograd=False,
                                  epochs=20, batch_size=64, num_batteries=3):
    """
    Fine-tune PINN with or without fast autograd.

    Args:
        pretrained_model: Pretrained PINN
        target_dataset: Target dataset
        use_fast_autograd: If True, use fast autograd. If False, use standard.
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
    if use_fast_autograd:
        # Convert to fast PINN
        fast_config = FastAutogradConfig(
            method='forward',
            step_size=1e-5,
            vectorized=True
        )
        model = convert_pinn_to_fast(pretrained_model, fast_config)
        model_name = "FastPINN"
    else:
        # Standard PINN
        model = PINN(args)
        model.solution_u.load_state_dict(pretrained_model.solution_u.state_dict())
        model.dynamical_F.load_state_dict(pretrained_model.dynamical_F.state_dict())
        model_name = "StandardPINN"

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

            # Forward pass
            u1, f1 = model.forward(x1)
            u2, f2 = model.forward(x2)

            # Data loss
            loss1 = 0.5 * nn.MSELoss()(u1, y1) + 0.5 * nn.MSELoss()(u2, y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5 * nn.MSELoss()(f1, f_target) + 0.5 * nn.MSELoss()(f2, f_target)

            # Physics loss
            loss3 = nn.ReLU()(torch.mul(u2 - u1, y1 - y2)).sum()

            # Total loss
            loss = loss1 + model.alpha * loss2 + model.beta * loss3

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time

    # Evaluate
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x1, x2, y1, y2 in dataloader['test']:
            x1, y1 = x1.to(device), y1.to(device)
            if use_fast_autograd:
                u1, _ = model.forward(x1)
            else:
                u1 = model.solution_u(x1)
            all_preds.append(u1)
            all_targets.append(y1)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mae = torch.abs(all_preds - all_targets).mean().item()

    return mae, train_time


def test_full_training():
    """
    Test 2: Full training comparison
    """
    print("=" * 80)
    print("TEST 2: FULL TRAINING COMPARISON")
    print("=" * 80)
    print()
    print("Training PINN with standard vs fast autograd...")
    print()

    # Setup
    source_dataset = 'XJTU'
    target_dataset = 'TJU'
    num_batteries_pretrain = 3
    num_batteries_finetune = 3
    epochs_pretrain = 20
    epochs_finetune = 20
    batch_size = 64

    # Pretrain (shared)
    print("Pretraining (shared)...")
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

    # Test 1: Standard autograd
    print("-" * 80)
    print("Fine-tuning with STANDARD autograd...")
    mae_standard, time_standard = train_pinn_with_fast_autograd(
        pretrained_model, target_dataset,
        use_fast_autograd=False,
        epochs=epochs_finetune,
        batch_size=batch_size,
        num_batteries=num_batteries_finetune
    )
    print(f"[OK] Standard complete")
    print(f"  Time: {time_standard:.2f}s")
    print(f"  MAE: {mae_standard:.4f}")
    print()

    # Test 2: Fast autograd
    print("-" * 80)
    print("Fine-tuning with FAST autograd...")
    mae_fast, time_fast = train_pinn_with_fast_autograd(
        pretrained_model, target_dataset,
        use_fast_autograd=True,
        epochs=epochs_finetune,
        batch_size=batch_size,
        num_batteries=num_batteries_finetune
    )
    print(f"[OK] Fast complete")
    print(f"  Time: {time_fast:.2f}s")
    print(f"  MAE: {mae_fast:.4f}")
    print()

    # Results
    speedup = time_standard / time_fast
    mae_change = ((mae_fast - mae_standard) / mae_standard) * 100

    results = {
        'standard': {
            'time': time_standard,
            'mae': mae_standard,
            'speedup': 1.0
        },
        'fast_autograd': {
            'time': time_fast,
            'mae': mae_fast,
            'speedup': speedup
        }
    }

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"{'Method':<20} {'Time (s)':<12} {'Speedup':<10} {'MAE':<10} {'MAE Δ':<10}")
    print("-" * 80)
    print(f"{'Standard':<20} {time_standard:>8.2f}     {1.0:>6.2f}x    {mae_standard:.4f}     +0.0%")
    print(f"{'Fast Autograd':<20} {time_fast:>8.2f}     {speedup:>6.2f}x    {mae_fast:.4f}    {mae_change:>+5.1f}%")
    print()

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    if speedup > 1.3:
        print(f"[OK] FAST AUTOGRAD WORKS!")
        print(f"  Speedup: {speedup:.2f}x")
        print()

        if abs(mae_change) < 15:
            print(f"[OK] Accuracy acceptable (MAE change: {mae_change:+.1f}%)")
            print()
            print("RECOMMENDATION:")
            print(f"  Use fast autograd for 5-hour workload")
            print(f"  Estimated time: 5.0 hours / {speedup:.2f} = {5.0/speedup:.2f} hours")
            print()

            if 5.0/speedup <= 2.5:
                print("  [OK] ACHIEVES CONSORTIUM GOAL!")
                print(f"  Target: 2.5 hours, Estimated: {5.0/speedup:.2f} hours")
            else:
                print(f"  ⚠ Close but not quite 2x")
                print(f"  Need additional optimizations")

        else:
            print(f"⚠ Accuracy degraded ({abs(mae_change):.1f}%)")
            print(f"  May need to tune step size or combine with other methods")

    elif speedup > 1.1:
        print(f"[OK] Fast autograd is faster ({speedup:.2f}x)")
        print(f"  But speedup is modest, may need to combine with other techniques")

    else:
        print(f"[FAIL] Fast autograd NOT faster ({speedup:.2f}x)")
        print(f"  Implementation overhead likely canceling benefits")

    print()

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/fast_autograd_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: results/fast_autograd_results.json")
    print("=" * 80)

    return results


def run_all_tests():
    """Run all tests"""
    print()
    print("=" * 80)
    print("FAST AUTOGRAD VALIDATION")
    print("=" * 80)
    print()
    print("Device:", device)
    if device == 'cuda':
        print("GPU:", torch.cuda.get_device_name(0))
    print()
    print("Testing custom fast autograd implementation...")
    print()

    # Test 1: Microbenchmark
    micro_results = test_microbenchmark()

    # Test 2: Full training (only if microbenchmark shows promise)
    if micro_results['speedup'] > 1.1:
        print()
        print("Microbenchmark shows promise, proceeding to full training test...")
        print()
        training_results = test_full_training()
    else:
        print()
        print("⚠ Microbenchmark shows no speedup, skipping full training test")
        print("  Fast autograd implementation needs optimization")
        training_results = None

    return micro_results, training_results


if __name__ == '__main__':
    run_all_tests()
