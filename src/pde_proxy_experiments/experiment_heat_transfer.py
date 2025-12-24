"""
Experiments 2B & 3B: Transfer Learning on Heat Equation

2B: Exact Transfer (Baseline)
   - Load source model trained on α=0.01
   - Transfer to target α=0.02
   - Use exact AD for derivatives
   - Fine-tune both networks

3B: FD Proxy Transfer (THE KEY TEST)
   - Load source model trained on α=0.01
   - FREEZE physics network
   - Transfer to target α=0.02
   - Use FD approximation for spatial derivatives
   - Fine-tune solution network only

Success Criteria:
- Experiment 2B: Achieve <20% error (baseline)
- Experiment 3B: Degradation <10% vs Experiment 2B AND faster training
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm

from heat_physics_operator import HeatPhysicsOperatorPINN
from experiment_heat_source import HeatDataGenerator


def transfer_exact(
    model,
    data_generator_target,
    n_epochs=2000,
    lr_solution=1e-4,
    lr_physics=1e-4,
    save_dir="results"
):
    """
    Experiment 2B: Transfer with exact AD (baseline)
    Both networks trainable, exact derivatives
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    optimizer_sol = optim.Adam(model.solution_net.parameters(), lr=lr_solution)
    optimizer_phys = optim.Adam(model.physics_net.parameters(), lr=lr_physics)

    lambda_ic = 10.0
    lambda_bc = 10.0
    lambda_physics = 1.0

    history = {
        'epoch': [], 'loss_total': [], 'loss_ic': [], 'loss_bc': [],
        'loss_physics': [], 'test_l2_relative': [], 'pde_residual': [], 'time_per_epoch': []
    }

    x_test, t_test, u_test = data_generator_target.get_test_data()
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(model.device)
    t_test_t = torch.tensor(t_test, dtype=torch.float32).to(model.device)
    u_test_t = torch.tensor(u_test, dtype=torch.float32).to(model.device)

    print(f"\n🔄 Experiment 2B: Exact Transfer (Baseline)")
    print(f"  Source α: 0.01 → Target α: {model.alpha:.4f}")
    print(f"  Mode: Exact AD, both networks trainable")
    print(f"  Epochs: {n_epochs}\n")

    best_error = float('inf')
    pbar = tqdm(range(n_epochs), desc="Exact Transfer")

    for epoch in pbar:
        start = time.time()

        x_ic, t_ic, u_ic = data_generator_target.get_ic_data(100)
        x_bc, t_bc, u_bc = data_generator_target.get_bc_data(100)
        x_col, t_col = data_generator_target.get_collocation_data(2000)

        x_ic_t = torch.tensor(x_ic, dtype=torch.float32, requires_grad=True).to(model.device)
        t_ic_t = torch.tensor(t_ic, dtype=torch.float32, requires_grad=True).to(model.device)
        u_ic_t = torch.tensor(u_ic, dtype=torch.float32).to(model.device)

        x_bc_t = torch.tensor(x_bc, dtype=torch.float32, requires_grad=True).to(model.device)
        t_bc_t = torch.tensor(t_bc, dtype=torch.float32, requires_grad=True).to(model.device)
        u_bc_t = torch.tensor(u_bc, dtype=torch.float32).to(model.device)

        x_col_t = torch.tensor(x_col, dtype=torch.float32, requires_grad=True).to(model.device)
        t_col_t = torch.tensor(t_col, dtype=torch.float32, requires_grad=True).to(model.device)

        optimizer_sol.zero_grad()
        optimizer_phys.zero_grad()

        u_ic_pred = model.solution_net(x_ic_t, t_ic_t)
        loss_ic = torch.mean((u_ic_pred - u_ic_t)**2)

        u_bc_pred = model.solution_net(x_bc_t, t_bc_t)
        loss_bc = torch.mean((u_bc_pred - u_bc_t)**2)

        result = model.forward_exact(x_col_t, t_col_t)
        loss_physics = result['physics_loss']

        loss_total = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_physics * loss_physics
        loss_total.backward()

        torch.nn.utils.clip_grad_norm_(model.solution_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.physics_net.parameters(), max_norm=1.0)

        optimizer_sol.step()
        optimizer_phys.step()

        epoch_time = time.time() - start

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                u_test_pred = model.solution_net(x_test_t, t_test_t)
                l2_absolute = torch.mean((u_test_pred - u_test_t)**2).sqrt().item()
                l2_relative = (l2_absolute / torch.mean(u_test_t**2).sqrt().item()) * 100

                pde_residual = model.compute_pde_residual(x_test_t[:100], t_test_t[:100], use_fd=False)

                history['epoch'].append(epoch)
                history['loss_total'].append(loss_total.item())
                history['loss_ic'].append(loss_ic.item())
                history['loss_bc'].append(loss_bc.item())
                history['loss_physics'].append(loss_physics.item())
                history['test_l2_relative'].append(l2_relative)
                history['pde_residual'].append(pde_residual)
                history['time_per_epoch'].append(epoch_time)

                if l2_relative < best_error:
                    best_error = l2_relative
                    model.save_checkpoint(
                        save_dir / "best_model.pt", epoch,
                        optimizer_sol, optimizer_phys,
                        metadata={'test_l2_relative': l2_relative}
                    )

                pbar.set_postfix({'Loss': f"{loss_total.item():.2e}", 'L2': f"{l2_relative:.2f}%"})

    with open(save_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[OK] Exact Transfer Complete!")
    print(f"  Best L2 error: {best_error:.4f}%")
    print(f"  Final L2 error: {history['test_l2_relative'][-1]:.4f}%")

    return history, best_error


def transfer_proxy(
    model,
    data_generator_target,
    n_epochs=2000,
    lr_solution=1e-4,
    epsilon=1e-4,
    save_dir="results"
):
    """
    Experiment 3B: Transfer with FD proxy (THE KEY TEST)
    Physics network FROZEN, FD approximation, solution network only
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Freeze physics network
    model.freeze_physics_network()

    optimizer_sol = optim.Adam(model.solution_net.parameters(), lr=lr_solution)

    lambda_ic = 10.0
    lambda_bc = 10.0
    lambda_physics = 1.0

    history = {
        'epoch': [], 'loss_total': [], 'loss_ic': [], 'loss_bc': [],
        'loss_physics': [], 'test_l2_relative': [], 'pde_residual': [], 'time_per_epoch': []
    }

    x_test, t_test, u_test = data_generator_target.get_test_data()
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(model.device)
    t_test_t = torch.tensor(t_test, dtype=torch.float32).to(model.device)
    u_test_t = torch.tensor(u_test, dtype=torch.float32).to(model.device)

    print(f"\n🚀 Experiment 3B: FD Proxy Transfer (THE KEY TEST)")
    print(f"  Source α: 0.01 → Target α: {model.alpha:.4f}")
    print(f"  Mode: FD approximation (ε={epsilon}), physics network FROZEN")
    print(f"  Epochs: {n_epochs}\n")

    best_error = float('inf')
    pbar = tqdm(range(n_epochs), desc="FD Proxy Transfer")

    for epoch in pbar:
        start = time.time()

        x_ic, t_ic, u_ic = data_generator_target.get_ic_data(100)
        x_bc, t_bc, u_bc = data_generator_target.get_bc_data(100)
        x_col, t_col = data_generator_target.get_collocation_data(2000)

        x_ic_t = torch.tensor(x_ic, dtype=torch.float32, requires_grad=True).to(model.device)
        t_ic_t = torch.tensor(t_ic, dtype=torch.float32, requires_grad=True).to(model.device)
        u_ic_t = torch.tensor(u_ic, dtype=torch.float32).to(model.device)

        x_bc_t = torch.tensor(x_bc, dtype=torch.float32, requires_grad=True).to(model.device)
        t_bc_t = torch.tensor(t_bc, dtype=torch.float32, requires_grad=True).to(model.device)
        u_bc_t = torch.tensor(u_bc, dtype=torch.float32).to(model.device)

        x_col_t = torch.tensor(x_col, dtype=torch.float32, requires_grad=True).to(model.device)
        t_col_t = torch.tensor(t_col, dtype=torch.float32, requires_grad=True).to(model.device)

        optimizer_sol.zero_grad()

        u_ic_pred = model.solution_net(x_ic_t, t_ic_t)
        loss_ic = torch.mean((u_ic_pred - u_ic_t)**2)

        u_bc_pred = model.solution_net(x_bc_t, t_bc_t)
        loss_bc = torch.mean((u_bc_pred - u_bc_t)**2)

        # Use FD proxy for physics loss
        result = model.forward_proxy(x_col_t, t_col_t, epsilon=epsilon)
        loss_physics = result['physics_loss']

        loss_total = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_physics * loss_physics
        loss_total.backward()

        torch.nn.utils.clip_grad_norm_(model.solution_net.parameters(), max_norm=1.0)

        optimizer_sol.step()

        epoch_time = time.time() - start

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                u_test_pred = model.solution_net(x_test_t, t_test_t)
                l2_absolute = torch.mean((u_test_pred - u_test_t)**2).sqrt().item()
                l2_relative = (l2_absolute / torch.mean(u_test_t**2).sqrt().item()) * 100

                pde_residual = model.compute_pde_residual(x_test_t[:100], t_test_t[:100], use_fd=True, epsilon=epsilon)

                history['epoch'].append(epoch)
                history['loss_total'].append(loss_total.item())
                history['loss_ic'].append(loss_ic.item())
                history['loss_bc'].append(loss_bc.item())
                history['loss_physics'].append(loss_physics.item())
                history['test_l2_relative'].append(l2_relative)
                history['pde_residual'].append(pde_residual)
                history['time_per_epoch'].append(epoch_time)

                if l2_relative < best_error:
                    best_error = l2_relative
                    model.save_checkpoint(
                        save_dir / "best_model.pt", epoch,
                        optimizer_sol, None,
                        metadata={'test_l2_relative': l2_relative}
                    )

                pbar.set_postfix({'Loss': f"{loss_total.item():.2e}", 'L2': f"{l2_relative:.2f}%"})

    with open(save_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[OK] FD Proxy Transfer Complete!")
    print(f"  Best L2 error: {best_error:.4f}%")
    print(f"  Final L2 error: {history['test_l2_relative'][-1]:.4f}%")

    return history, best_error


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Load source model
    print("="*70)
    print("HEAT EQUATION TRANSFER LEARNING EXPERIMENTS")
    print("="*70)

    alpha_source = 0.01
    alpha_target = 0.02

    # Experiment 2B: Exact Transfer
    print("\n" + "="*70)
    print("EXPERIMENT 2B: EXACT TRANSFER BASELINE")
    print("="*70)

    model_exact = HeatPhysicsOperatorPINN(alpha=alpha_target)
    checkpoint = model_exact.load_checkpoint(
        "results/heat_experiment1_source/best_model.pt",
        load_optimizers=False
    )

    data_gen_target = HeatDataGenerator(alpha=alpha_target)

    history_exact, best_exact = transfer_exact(
        model_exact,
        data_gen_target,
        n_epochs=2000,
        lr_solution=1e-4,
        lr_physics=1e-4,
        save_dir="results/heat_experiment2_exact"
    )

    # Experiment 3B: FD Proxy Transfer
    print("\n" + "="*70)
    print("EXPERIMENT 3B: FD PROXY TRANSFER")
    print("="*70)

    model_proxy = HeatPhysicsOperatorPINN(alpha=alpha_target)
    model_proxy.load_checkpoint(
        "results/heat_experiment1_source/best_model.pt",
        load_optimizers=False
    )

    history_proxy, best_proxy = transfer_proxy(
        model_proxy,
        data_gen_target,
        n_epochs=2000,
        lr_solution=1e-4,
        epsilon=1e-4,
        save_dir="results/heat_experiment3_proxy"
    )

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON: EXACT VS FD PROXY")
    print("="*70)

    print(f"\n📊 Best L2 Errors:")
    print(f"  Source (α=0.01):        16.68%")
    print(f"  Exact Transfer (α=0.02): {best_exact:.2f}%")
    print(f"  FD Proxy (α=0.02):       {best_proxy:.2f}%")

    degradation = ((best_proxy - best_exact) / best_exact) * 100
    print(f"\n📉 Proxy Degradation: {degradation:+.1f}%")

    avg_time_exact = np.mean(history_exact['time_per_epoch'])
    avg_time_proxy = np.mean(history_proxy['time_per_epoch'])
    speedup = avg_time_exact / avg_time_proxy

    print(f"\n⏱️  Training Speed:")
    print(f"  Exact:  {avg_time_exact:.4f}s/epoch")
    print(f"  Proxy:  {avg_time_proxy:.4f}s/epoch")
    print(f"  Speedup: {speedup:.2f}×")

    print(f"\n🎯 Success Criteria:")
    print(f"  Degradation < 10%: {'[PASS] PASS' if abs(degradation) < 10 else '[FAIL] FAIL'}")
    print(f"  Speedup > 1.3×:    {'[PASS] PASS' if speedup > 1.3 else '[FAIL] FAIL'}")

    if abs(degradation) < 10 and speedup > 1.3:
        print(f"\n🎉 SUCCESS! FD proxy method works for Heat equation!")
    else:
        print(f"\n[WARN]  FD proxy method shows limitations")

    # Save comparison
    comparison = {
        'source_error': 16.68,
        'exact_transfer_error': best_exact,
        'proxy_transfer_error': best_proxy,
        'degradation_percent': degradation,
        'exact_time_per_epoch': avg_time_exact,
        'proxy_time_per_epoch': avg_time_proxy,
        'speedup': speedup
    }

    with open("results/comparison_exact_vs_proxy.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
