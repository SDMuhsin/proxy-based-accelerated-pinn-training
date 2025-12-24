"""
Visualization utilities for creating publication-quality figures
All figures follow IEEE/scientific publication standards
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import json

# Set publication-quality defaults
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'


def significance_stars(p_value):
    """Convert p-value to significance stars"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def plot_multi_seed_comparison(exact_results, proxy_results, save_path=None):
    """
    Create box plot comparing exact vs proxy across multiple seeds

    Returns: figure object
    """
    exact_best = [r['best_l2_error'] for r in exact_results]
    proxy_best = [r['best_l2_error'] for r in proxy_results]

    fig, ax = plt.subplots(figsize=(4, 4))

    # Box plots
    positions = [1, 2]
    bp = ax.boxplot([exact_best, proxy_best],
                     positions=positions,
                     widths=0.6,
                     patch_artist=True,
                     showfliers=True,
                     notch=True)

    # Color boxes
    colors = ['#3498db', '#e74c3c']  # Blue for exact, red for proxy
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual data points
    np.random.seed(42)
    for i, data in enumerate([exact_best, proxy_best]):
        x = np.random.normal(positions[i], 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.5, s=30, color='black', zorder=3)

    # Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(exact_best, proxy_best)
    stars = significance_stars(p_value)

    # Add significance bracket
    y_max = max(max(exact_best), max(proxy_best))
    y_bracket = y_max + (y_max - min(min(exact_best), min(proxy_best))) * 0.1
    ax.plot([1, 2], [y_bracket, y_bracket], 'k-', linewidth=1)
    ax.text(1.5, y_bracket + 0.3, stars, ha='center', va='bottom', fontsize=12)

    # Labels and formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(['Exact AD\n(Both trainable)', 'FD Proxy\n(Physics frozen)'])
    ax.set_ylabel('Best L2 Error (%)')
    ax.set_title('Transfer Learning Performance\n(Heat Equation, α: 0.01→0.02)')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics text
    text_str = (f'Exact: {np.mean(exact_best):.2f} ± {np.std(exact_best):.2f}%\n'
                f'Proxy: {np.mean(proxy_best):.2f} ± {np.std(proxy_best):.2f}%\n'
                f'p = {p_value:.4f}')
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_ablation_study(ablation_results, save_path=None):
    """
    Create bar chart comparing all ablation variants

    Args:
        ablation_results: Dict with keys = variant names, values = list of errors across seeds
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    variants = list(ablation_results.keys())
    means = [np.mean(ablation_results[v]) for v in variants]
    stds = [np.std(ablation_results[v]) for v in variants]

    x = np.arange(len(variants))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=colors[:len(variants)])

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.3, f'{mean:.2f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.set_ylabel('Best L2 Error (%)')
    ax.set_title('Ablation Study: Identifying Key Components')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_epsilon_sensitivity(epsilon_results, save_path=None):
    """
    Plot L2 error vs epsilon with confidence intervals

    Args:
        epsilon_results: Dict with epsilon values as keys, lists of errors as values
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    epsilons = sorted(epsilon_results.keys())
    means = [np.mean(epsilon_results[eps]) for eps in epsilons]
    stds = [np.std(epsilon_results[eps]) for eps in epsilons]
    ci_lower = [m - 1.96 * s / np.sqrt(len(epsilon_results[eps]))
                for m, s, eps in zip(means, stds, epsilons)]
    ci_upper = [m + 1.96 * s / np.sqrt(len(epsilon_results[eps]))
                for m, s, eps in zip(means, stds, epsilons)]

    # Left plot: Error vs epsilon (log scale)
    ax1.plot(epsilons, means, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax1.fill_between(epsilons, ci_lower, ci_upper, alpha=0.3, color='#e74c3c')
    ax1.set_xscale('log')
    ax1.set_xlabel('Epsilon (FD step size)')
    ax1.set_ylabel('Best L2 Error (%)')
    ax1.set_title('Epsilon Sensitivity Analysis')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Mark optimal epsilon
    optimal_idx = np.argmin(means)
    optimal_eps = epsilons[optimal_idx]
    ax1.axvline(optimal_eps, color='green', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_eps}')
    ax1.legend()

    # Right plot: Computational cost (placeholder - would need timing data)
    # For now, show standard deviation as robustness metric
    ax2.plot(epsilons, stds, 's-', color='#3498db', linewidth=2, markersize=6)
    ax2.set_xscale('log')
    ax2.set_xlabel('Epsilon (FD step size)')
    ax2.set_ylabel('Standard Deviation (%)')
    ax2.set_title('Robustness Across Seeds')
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_learning_curves(histories, labels, save_path=None):
    """
    Plot training dynamics with confidence bands

    Args:
        histories: List of history dicts (each from different seeds)
        labels: List of method labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = ['#3498db', '#e74c3c']

    for ax, metric, ylabel in zip(axes, ['test_l2_relative', 'loss_total'],
                                   ['Test L2 Error (%)', 'Total Loss']):
        for hist_list, label, color in zip(histories, labels, colors):
            # Get common epochs
            epochs = hist_list[0]['history']['epoch']

            # Stack all runs
            values = np.array([h['history'][metric] for h in hist_list])

            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)

            ax.plot(epochs, mean, '-', label=label, color=color, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        if metric == 'loss_total':
            ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')

    axes[0].set_title('Test Error Evolution')
    axes[1].set_title('Training Loss Evolution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_transfer_distance_analysis(transfer_results, save_path=None):
    """
    Plot error vs parameter transfer distance

    Args:
        transfer_results: Dict with (source_alpha, target_alpha) tuples as keys
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Extract transfer ratios and errors
    exact_points = []
    proxy_points = []

    for (source_alpha, target_alpha), results in transfer_results.items():
        ratio = target_alpha / source_alpha
        exact_errs = [r['exact_best_error'] for r in results]
        proxy_errs = [r['proxy_best_error'] for r in results]

        exact_points.append((ratio, np.mean(exact_errs), np.std(exact_errs)))
        proxy_points.append((ratio, np.mean(proxy_errs), np.std(proxy_errs)))

    # Sort by ratio
    exact_points = sorted(exact_points)
    proxy_points = sorted(proxy_points)

    # Plot
    exact_ratios, exact_means, exact_stds = zip(*exact_points)
    proxy_ratios, proxy_means, proxy_stds = zip(*proxy_points)

    ax.errorbar(exact_ratios, exact_means, yerr=exact_stds, fmt='o-',
                label='Exact AD', color='#3498db', linewidth=2, markersize=8, capsize=5)
    ax.errorbar(proxy_ratios, proxy_means, yerr=proxy_stds, fmt='s-',
                label='FD Proxy', color='#e74c3c', linewidth=2, markersize=8, capsize=5)

    ax.set_xlabel('Parameter Transfer Ratio (α_target / α_source)')
    ax.set_ylabel('Best L2 Error (%)')
    ax.set_title('Generalization vs Transfer Distance')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add reference line at ratio=1 (no transfer)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='No transfer')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def create_all_publication_figures(results_dir, output_dir='figures'):
    """
    Generate all publication-quality figures from experiment results

    Args:
        results_dir: Directory containing all experimental results
        output_dir: Directory to save figures
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Creating publication figures...")

    # Figure 1: Main comparison (multi-seed)
    if (results_dir / 'multi_seed_validation' / 'aggregated_results.json').exists():
        with open(results_dir / 'multi_seed_validation' / 'aggregated_results.json') as f:
            data = json.load(f)
        exact_results = data['exact_results']
        proxy_results = data['proxy_results']

        plot_multi_seed_comparison(exact_results, proxy_results,
                                   save_path=output_dir / 'fig1_main_comparison.pdf')
        plot_multi_seed_comparison(exact_results, proxy_results,
                                   save_path=output_dir / 'fig1_main_comparison.png')

    # Figure 2: Ablation study
    # Would load ablation results here

    # Figure 3: Epsilon sensitivity
    # Would load epsilon sweep results here

    # Figure 4: Learning curves
    # Would load training histories here

    print(f"Figures saved to {output_dir}/")

    return output_dir


if __name__ == "__main__":
    # Test visualization with simulated data
    print("Testing visualization utilities...")

    np.random.seed(42)

    # Simulate multi-seed results
    n_seeds = 10
    exact_sim = [{'best_l2_error': np.random.normal(6.3, 0.5)}
                 for _ in range(n_seeds)]
    proxy_sim = [{'best_l2_error': np.random.normal(4.0, 0.4)}
                 for _ in range(n_seeds)]

    fig = plot_multi_seed_comparison(exact_sim, proxy_sim,
                                     save_path='test_comparison.png')
    plt.show()

    print("Test figure created: test_comparison.png")
