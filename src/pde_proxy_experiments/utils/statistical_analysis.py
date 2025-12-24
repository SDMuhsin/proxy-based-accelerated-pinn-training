"""
Statistical analysis utilities for experimental validation
Includes t-tests, confidence intervals, effect sizes, and multiple comparison corrections
"""

import numpy as np
from scipy import stats
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
import json
from pathlib import Path


def compute_summary_statistics(values, name="Metric"):
    """
    Compute comprehensive summary statistics

    Args:
        values: List or array of values
        name: Name of the metric

    Returns:
        dict with mean, std, CI, min, max, etc.
    """
    values = np.array(values)
    n = len(values)

    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    sem = stats.sem(values)  # Standard error of mean

    # 95% confidence interval
    ci_95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)

    return {
        'name': name,
        'n': n,
        'mean': float(mean),
        'std': float(std),
        'sem': float(sem),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'ci_95_lower': float(ci_95[0]),
        'ci_95_upper': float(ci_95[1]),
        'cv': float(std / mean) if mean != 0 else None,  # Coefficient of variation
        'values': values.tolist()
    }


def paired_comparison(values_a, values_b, name_a="Method A", name_b="Method B"):
    """
    Perform paired statistical comparison between two methods

    Args:
        values_a: Results from method A (same seeds as B)
        values_b: Results from method B
        name_a: Name of method A
        name_b: Name of method B

    Returns:
        dict with test results and effect sizes
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    assert len(values_a) == len(values_b), "Values must be paired"

    # Differences
    diffs = values_a - values_b
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values_a, values_b)

    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, p_value_wilcoxon = stats.wilcoxon(values_a, values_b)

    # Effect size (Cohen's d for paired samples)
    cohen_d = mean_diff / std_diff if std_diff > 0 else 0

    # Percentage improvement (assuming lower is better)
    pct_improvement = -mean_diff / np.mean(values_a) * 100

    # 95% CI for difference
    n = len(diffs)
    sem_diff = stats.sem(diffs)
    ci_95_diff = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=sem_diff)

    return {
        'method_a': name_a,
        'method_b': name_b,
        'n_pairs': n,
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'pct_improvement': float(pct_improvement),
        'ci_95_diff_lower': float(ci_95_diff[0]),
        'ci_95_diff_upper': float(ci_95_diff[1]),
        'ttest': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_0_05': bool(p_value < 0.05),
            'significant_at_0_01': bool(p_value < 0.01),
            'significant_at_0_001': bool(p_value < 0.001)
        },
        'wilcoxon': {
            'w_statistic': float(w_stat),
            'p_value': float(p_value_wilcoxon)
        },
        'effect_size': {
            'cohen_d': float(cohen_d),
            'interpretation': interpret_cohen_d(cohen_d)
        }
    }


def interpret_cohen_d(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    elif abs_d < 1.2:
        return "large"
    else:
        return "very large"


def multiple_comparison_correction(p_values, alpha=0.05, method='holm'):
    """
    Apply multiple comparison correction to p-values

    Args:
        p_values: List of p-values
        alpha: Significance level
        method: Correction method ('bonferroni', 'holm', 'fdr_bh', etc.)

    Returns:
        dict with corrected results
    """
    if not HAS_STATSMODELS:
        # Fallback to simple Bonferroni correction
        alpha_bonf = alpha / len(p_values)
        reject = [p < alpha_bonf for p in p_values]
        pvals_corrected = [min(p * len(p_values), 1.0) for p in p_values]

        return {
            'method': 'bonferroni (manual)',
            'alpha': alpha,
            'n_tests': len(p_values),
            'original_p_values': [float(p) for p in p_values],
            'corrected_p_values': [float(p) for p in pvals_corrected],
            'reject_null': [bool(r) for r in reject],
            'n_significant': int(sum(reject)),
            'alpha_bonferroni': float(alpha_bonf),
            'alpha_sidak': None
        }

    reject, pvals_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=alpha, method=method
    )

    return {
        'method': method,
        'alpha': alpha,
        'n_tests': len(p_values),
        'original_p_values': [float(p) for p in p_values],
        'corrected_p_values': [float(p) for p in pvals_corrected],
        'reject_null': [bool(r) for r in reject],
        'n_significant': int(sum(reject)),
        'alpha_bonferroni': float(alpha_bonf),
        'alpha_sidak': float(alpha_sidak)
    }


def analyze_multi_seed_results(exact_results, proxy_results, save_path=None):
    """
    Comprehensive analysis of multi-seed experimental results

    Args:
        exact_results: List of result dicts from exact AD method
        proxy_results: List of result dicts from proxy method
        save_path: Path to save analysis JSON (optional)

    Returns:
        Complete statistical analysis dict
    """
    # Extract metrics
    exact_best = [r['best_l2_error'] for r in exact_results]
    proxy_best = [r['best_l2_error'] for r in proxy_results]
    exact_final = [r['final_l2_error'] for r in exact_results]
    proxy_final = [r['final_l2_error'] for r in proxy_results]
    exact_time = [r['avg_time_per_epoch'] for r in exact_results]
    proxy_time = [r['avg_time_per_epoch'] for r in proxy_results]

    # Summary statistics
    analysis = {
        'exact_best_l2': compute_summary_statistics(exact_best, "Exact AD Best L2 Error (%)"),
        'proxy_best_l2': compute_summary_statistics(proxy_best, "FD Proxy Best L2 Error (%)"),
        'exact_final_l2': compute_summary_statistics(exact_final, "Exact AD Final L2 Error (%)"),
        'proxy_final_l2': compute_summary_statistics(proxy_final, "FD Proxy Final L2 Error (%)"),
        'exact_time': compute_summary_statistics(exact_time, "Exact AD Time/Epoch (s)"),
        'proxy_time': compute_summary_statistics(proxy_time, "FD Proxy Time/Epoch (s)")
    }

    # Paired comparisons
    analysis['comparison_best_error'] = paired_comparison(
        exact_best, proxy_best, "Exact AD", "FD Proxy"
    )

    analysis['comparison_final_error'] = paired_comparison(
        exact_final, proxy_final, "Exact AD", "FD Proxy"
    )

    # Speedup analysis
    speedups = [e / p for e, p in zip(exact_time, proxy_time)]
    analysis['speedup'] = compute_summary_statistics(speedups, "Speedup (Proxy vs Exact)")

    # Count wins
    proxy_wins_best = sum(1 for e, p in zip(exact_best, proxy_best) if p < e)
    proxy_wins_final = sum(1 for e, p in zip(exact_final, proxy_final) if p < e)

    analysis['win_statistics'] = {
        'total_seeds': len(exact_results),
        'proxy_wins_best_error': proxy_wins_best,
        'proxy_wins_final_error': proxy_wins_final,
        'win_rate_best': proxy_wins_best / len(exact_results),
        'win_rate_final': proxy_wins_final / len(exact_results)
    }

    # Overall assessment
    best_comparison = analysis['comparison_best_error']
    is_significant = best_comparison['ttest']['significant_at_0_05']
    effect_large = abs(best_comparison['effect_size']['cohen_d']) > 0.8

    analysis['assessment'] = {
        'statistically_significant': is_significant,
        'large_effect_size': effect_large,
        'proxy_better': best_comparison['mean_diff'] > 0,  # Assuming lower error is better
        'recommendation': get_recommendation(
            is_significant, effect_large,
            best_comparison['mean_diff'] > 0,
            proxy_wins_best
        )
    }

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)

    return analysis


def get_recommendation(significant, large_effect, proxy_better, n_wins):
    """Generate recommendation based on statistical results"""
    if significant and large_effect and proxy_better and n_wins >= 0.7 * 10:
        return "PROCEED WITH FULL PHASE 1: Results replicate strongly across seeds"
    elif significant and proxy_better and n_wins >= 0.6 * 10:
        return "PROCEED WITH CAUTION: Results replicate but need more validation"
    elif proxy_better and n_wins >= 0.5 * 10:
        return "MIXED RESULTS: Run additional seeds before committing to full study"
    else:
        return "DO NOT PROCEED: Results do not replicate reliably across seeds"


def print_analysis_summary(analysis):
    """Print human-readable summary of statistical analysis"""
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*70)

    # Best error comparison
    best = analysis['comparison_best_error']
    print(f"\nBest L2 Error Comparison:")
    print(f"  Exact AD:  {analysis['exact_best_l2']['mean']:.2f} ± "
          f"{analysis['exact_best_l2']['std']:.2f}% "
          f"(95% CI: [{analysis['exact_best_l2']['ci_95_lower']:.2f}, "
          f"{analysis['exact_best_l2']['ci_95_upper']:.2f}])")
    print(f"  FD Proxy:  {analysis['proxy_best_l2']['mean']:.2f} ± "
          f"{analysis['proxy_best_l2']['std']:.2f}% "
          f"(95% CI: [{analysis['proxy_best_l2']['ci_95_lower']:.2f}, "
          f"{analysis['proxy_best_l2']['ci_95_upper']:.2f}])")
    print(f"  Difference: {best['mean_diff']:+.2f}% "
          f"(95% CI: [{best['ci_95_diff_lower']:+.2f}, {best['ci_95_diff_upper']:+.2f}])")
    print(f"  Improvement: {best['pct_improvement']:+.1f}%")
    print(f"  p-value: {best['ttest']['p_value']:.4f} "
          f"({'***' if best['ttest']['significant_at_0_001'] else '**' if best['ttest']['significant_at_0_01'] else '*' if best['ttest']['significant_at_0_05'] else 'ns'})")
    print(f"  Cohen's d: {best['effect_size']['cohen_d']:.2f} "
          f"({best['effect_size']['interpretation']})")

    # Win statistics
    wins = analysis['win_statistics']
    print(f"\nWin Statistics:")
    print(f"  Proxy better (best error): {wins['proxy_wins_best_error']}/{wins['total_seeds']} "
          f"({wins['win_rate_best']:.1%})")
    print(f"  Proxy better (final error): {wins['proxy_wins_final_error']}/{wins['total_seeds']} "
          f"({wins['win_rate_final']:.1%})")

    # Speedup
    speedup = analysis['speedup']
    print(f"\nSpeedup:")
    print(f"  {speedup['mean']:.2f}x ± {speedup['std']:.2f}x "
          f"(95% CI: [{speedup['ci_95_lower']:.2f}, {speedup['ci_95_upper']:.2f}])")

    # Overall assessment
    assessment = analysis['assessment']
    print(f"\n{'='*70}")
    print("ASSESSMENT")
    print(f"{'='*70}")
    print(f"  Statistically significant: {assessment['statistically_significant']}")
    print(f"  Large effect size: {assessment['large_effect_size']}")
    print(f"  Proxy performs better: {assessment['proxy_better']}")
    print(f"\nRECOMMENDATION: {assessment['recommendation']}")
    print()


if __name__ == "__main__":
    # Example usage
    print("Testing statistical analysis utilities...")

    # Simulated data
    np.random.seed(42)
    exact_sim = np.random.normal(6.3, 0.5, 10)
    proxy_sim = np.random.normal(4.0, 0.4, 10)

    exact_results = [{'best_l2_error': e, 'final_l2_error': e+0.5,
                     'avg_time_per_epoch': 0.022} for e in exact_sim]
    proxy_results = [{'best_l2_error': p, 'final_l2_error': p+0.8,
                     'avg_time_per_epoch': 0.018} for p in proxy_sim]

    analysis = analyze_multi_seed_results(exact_results, proxy_results)
    print_analysis_summary(analysis)
