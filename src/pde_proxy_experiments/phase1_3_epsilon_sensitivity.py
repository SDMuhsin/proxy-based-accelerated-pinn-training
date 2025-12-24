"""
Phase 1.3: Epsilon Sensitivity Analysis

Tests how FD proxy performance depends on epsilon (step size) for finite differences.
Tests 7 epsilon values: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

Expected time: ~30 minutes (7 epsilons × 3 seeds × 1.5 min)
Uses 3 seeds per epsilon for statistical confidence
"""

import sys
sys.path.append('.')

from utils.experiment_runner import run_transfer_experiment
import numpy as np
from pathlib import Path
import json

def run_epsilon_sweep(
    source_checkpoint_path,
    target_alpha,
    epsilons,
    seeds=[42, 123, 456],
    n_epochs=2000,
    save_dir='results/phase1_3_epsilon_sensitivity'
):
    """Run epsilon sensitivity analysis"""

    print("="*80)
    print("PHASE 1.3: EPSILON SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"\nTesting {len(epsilons)} epsilon values × {len(seeds)} seeds = {len(epsilons) * len(seeds)} experiments")
    print(f"Epsilon values: {epsilons}")
    print(f"Expected time: ~{len(epsilons) * len(seeds) * 1.5:.0f} minutes")
    print()

    results_by_epsilon = {}

    for epsilon in epsilons:
        print(f"\n{'='*80}")
        print(f"EPSILON = {epsilon:.0e}")
        print(f"{'='*80}")

        epsilon_results = []

        for seed in seeds:
            result = run_transfer_experiment(
                source_checkpoint_path=source_checkpoint_path,
                target_alpha=target_alpha,
                method='proxy',
                seed=seed,
                n_epochs=n_epochs,
                epsilon=epsilon,
                freeze_physics=True,
                save_dir=Path(save_dir) / f'epsilon_{epsilon:.0e}',
                verbose=True
            )
            epsilon_results.append(result)

            print(f"  Seed {seed}: Best L2 = {result['best_l2_error']:.2f}%")

        # Aggregate for this epsilon
        best_errors = [r['best_l2_error'] for r in epsilon_results]
        results_by_epsilon[epsilon] = best_errors

        print(f"  Mean ± Std: {np.mean(best_errors):.2f} ± {np.std(best_errors):.2f}%")

    # Summary analysis
    print("\n" + "="*80)
    print("EPSILON SENSITIVITY SUMMARY")
    print("="*80)

    print(f"\n{'Epsilon':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 52)

    for epsilon, errors in sorted(results_by_epsilon.items()):
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        min_err = np.min(errors)
        max_err = np.max(errors)
        print(f"{epsilon:<12.0e} {mean_err:<10.2f} {std_err:<10.2f} {min_err:<10.2f} {max_err:<10.2f}")

    # Find optimal epsilon
    mean_errors = {eps: np.mean(errs) for eps, errs in results_by_epsilon.items()}
    optimal_epsilon = min(mean_errors, key=mean_errors.get)
    optimal_error = mean_errors[optimal_epsilon]

    print(f"\n{'='*80}")
    print(f"OPTIMAL EPSILON: {optimal_epsilon:.0e} (Mean L2 = {optimal_error:.2f}%)")
    print(f"{'='*80}")

    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    summary = {
        'epsilons': [float(eps) for eps in epsilons],
        'seeds': seeds,
        'n_epochs': n_epochs,
        'results_by_epsilon': {
            f'{eps:.0e}': [float(e) for e in errors]
            for eps, errors in results_by_epsilon.items()
        },
        'statistics': {
            f'{eps:.0e}': {
                'epsilon': float(eps),
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'min': float(np.min(errors)),
                'max': float(np.max(errors)),
                'values': [float(e) for e in errors]
            }
            for eps, errors in results_by_epsilon.items()
        },
        'optimal': {
            'epsilon': float(optimal_epsilon),
            'mean_error': float(optimal_error),
            'std_error': float(np.std(results_by_epsilon[optimal_epsilon]))
        }
    }

    with open(save_dir / 'epsilon_sensitivity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {save_dir}/epsilon_sensitivity_summary.json")

    # Robustness analysis
    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS")
    print("="*80)

    print("\nEpsilons within 10% of optimal performance:")
    threshold = optimal_error * 1.1  # Within 10% of best

    robust_epsilons = []
    for epsilon, mean_err in sorted(mean_errors.items()):
        if mean_err <= threshold:
            robust_epsilons.append(epsilon)
            print(f"  {epsilon:.0e}: {mean_err:.2f}% ({((mean_err / optimal_error - 1) * 100):+.1f}%)")

    print(f"\nRobust epsilon range: [{min(robust_epsilons):.0e}, {max(robust_epsilons):.0e}]")
    print(f"Order of magnitude span: {np.log10(max(robust_epsilons) / min(robust_epsilons)):.1f} decades")

    summary['robustness'] = {
        'threshold': float(threshold),
        'robust_epsilons': [float(eps) for eps in robust_epsilons],
        'robust_range': [float(min(robust_epsilons)), float(max(robust_epsilons))],
        'magnitude_span': float(np.log10(max(robust_epsilons) / min(robust_epsilons)))
    }

    with open(save_dir / 'epsilon_sensitivity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return results_by_epsilon


if __name__ == "__main__":
    source_checkpoint = "results/heat_experiment1_source/best_model.pt"

    # Test 7 epsilon values spanning 3 orders of magnitude
    epsilons = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    epsilon_results = run_epsilon_sweep(
        source_checkpoint_path=source_checkpoint,
        target_alpha=0.02,
        epsilons=epsilons,
        seeds=[42, 123, 456],
        n_epochs=2000,
        save_dir='results/phase1_3_epsilon_sensitivity'
    )

    print("\nPhase 1.3 COMPLETE")
