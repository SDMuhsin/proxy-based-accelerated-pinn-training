"""
Phase 1.1: Full Multi-Seed Statistical Validation (10 seeds)

This runs the complete 10-seed validation experiment with:
- Both exact AD and FD proxy methods
- Comprehensive statistical analysis
- Publication-quality figures
- Decision on proceeding to ablation/sensitivity studies

Total time: ~30 minutes (10 seeds × 2 methods × 1.5 min)
"""

import sys
sys.path.append('.')

from utils.experiment_runner import run_multi_seed_comparison
from utils.statistical_analysis import analyze_multi_seed_results, print_analysis_summary
from utils.visualization import plot_multi_seed_comparison
import numpy as np
from pathlib import Path

print("="*80)
print("PHASE 1.1: FULL MULTI-SEED STATISTICAL VALIDATION")
print("="*80)
print("\nThis experiment will:")
print("  1. Run 10 seeds (5 new + 5 from quick validation)")
print("  2. Compute comprehensive statistical analysis")
print("  3. Generate publication-quality box plot figures")
print("  4. Make GO/NO-GO decision for Phase 1.2 and 1.3")
print()
print("Expected time: ~15 minutes (5 new seeds × 2 methods × 1.5 min)")
print("="*80)
print()

# We already have seeds [123, 456, 789, 1011, 1213] from quick validation
# Add 5 more seeds to reach 10 total
new_seeds = [42, 999, 2024, 3141, 5678]  # Include original seed 42

print(f"Running 5 additional seeds: {new_seeds}")
print(f"(Already have seeds: [123, 456, 789, 1011, 1213] from quick validation)")
print()

# Run the 5 new seeds
exact_new, proxy_new = run_multi_seed_comparison(
    source_checkpoint_path="results/heat_experiment1_source/best_model.pt",
    target_alpha=0.02,
    seeds=new_seeds,
    n_epochs=2000,
    epsilon=1e-4,
    save_dir="results/phase1_1_multi_seed/new_seeds",
    verbose=True
)

print("\n" + "="*80)
print("LOADING RESULTS FROM QUICK VALIDATION")
print("="*80)

# Load the 5 seeds from quick validation
import json
quick_val_dir = Path("results/quick_validation_5seeds")

quick_seeds = [123, 456, 789, 1011, 1213]
exact_quick = []
proxy_quick = []

for seed in quick_seeds:
    # Load exact results
    with open(quick_val_dir / 'exact' / f'results_seed{seed}.json') as f:
        exact_quick.append(json.load(f))

    # Load proxy results
    with open(quick_val_dir / 'proxy' / f'results_seed{seed}.json') as f:
        proxy_quick.append(json.load(f))

print(f"Loaded {len(exact_quick)} exact results and {len(proxy_quick)} proxy results")

# Combine all 10 seeds
all_seeds = quick_seeds + new_seeds
exact_all = exact_quick + exact_new
proxy_all = proxy_quick + proxy_new

print(f"\nTotal seeds: {len(all_seeds)}")
print(f"Seed list: {all_seeds}")

# Save combined results
combined_dir = Path("results/phase1_1_multi_seed")
combined_dir.mkdir(exist_ok=True, parents=True)

summary = {
    'seeds': all_seeds,
    'n_seeds': len(all_seeds),
    'target_alpha': 0.02,
    'n_epochs': 2000,
    'exact_results': [
        {k: v for k, v in r.items() if k != 'history'}
        for r in exact_all
    ],
    'proxy_results': [
        {k: v for k, v in r.items() if k != 'history'}
        for r in proxy_all
    ]
}

with open(combined_dir / 'aggregated_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Comprehensive statistical analysis
print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

analysis = analyze_multi_seed_results(
    exact_all,
    proxy_all,
    save_path=combined_dir / 'statistical_analysis.json'
)

print_analysis_summary(analysis)

# Generate publication figure
print("\n" + "="*80)
print("GENERATING PUBLICATION FIGURES")
print("="*80)

fig = plot_multi_seed_comparison(
    exact_all,
    proxy_all,
    save_path=combined_dir / 'fig_multi_seed_comparison.pdf'
)
plot_multi_seed_comparison(
    exact_all,
    proxy_all,
    save_path=combined_dir / 'fig_multi_seed_comparison.png'
)
print("Figures saved to:", combined_dir)

# Decision criteria for proceeding
print("\n" + "="*80)
print("PHASE 1.1 DECISION: PROCEED TO ABLATION & SENSITIVITY?")
print("="*80)

best_comp = analysis['comparison_best_error']
is_significant = best_comp['ttest']['significant_at_0_05']
is_highly_significant = best_comp['ttest']['significant_at_0_01']
large_effect = abs(best_comp['effect_size']['cohen_d']) > 0.8
proxy_better = best_comp['mean_diff'] > 0
win_rate = analysis['win_statistics']['win_rate_best']

print(f"\nCriteria for proceeding to Phase 1.2 & 1.3:")
print(f"  1. Statistically significant (p < 0.05): {is_significant}")
print(f"  2. Highly significant (p < 0.01): {is_highly_significant}")
print(f"  3. Large effect size (|d| > 0.8): {large_effect}")
print(f"  4. Proxy better than exact: {proxy_better}")
print(f"  5. Win rate >= 70%: {win_rate >= 0.7} ({win_rate:.1%})")
print()

if is_highly_significant and large_effect and proxy_better and win_rate >= 0.7:
    print("[OK][OK][OK] STRONG GO: All criteria met!")
    print("  → Proceed with Phase 1.2 (Ablation Study)")
    print("  → Proceed with Phase 1.3 (Epsilon Sensitivity)")
    decision = "STRONG_GO"
elif is_significant and proxy_better and win_rate >= 0.6:
    print("[OK] GO: Core criteria met")
    print("  → Proceed with Phase 1.2 and 1.3, but results may be marginal")
    decision = "GO"
elif proxy_better:
    print("⚠ MARGINAL: Proxy shows improvement but not statistically strong")
    print("  → Consider Phase 1.2 ablation to understand mechanisms")
    print("  → May skip Phase 1.3 sensitivity analysis")
    decision = "MARGINAL"
else:
    print("[FAIL] NO GO: Proxy does not consistently outperform exact AD")
    print("  → Do NOT proceed to Phase 1.2 or 1.3")
    print("  → Investigate why proxy underperforms")
    decision = "NO_GO"

# Save decision
summary['decision'] = {
    'verdict': decision,
    'criteria': {
        'significant_p005': is_significant,
        'significant_p001': is_highly_significant,
        'large_effect_size': large_effect,
        'proxy_better': proxy_better,
        'win_rate_above_70pct': win_rate >= 0.7
    },
    'recommendation': analysis['assessment']['recommendation']
}

with open(combined_dir / 'aggregated_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{''*80}")
print("PHASE 1.1 COMPLETE")
print(f"Results saved to: {combined_dir}/")
print("="*80)
