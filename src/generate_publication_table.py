"""
Generate publication-ready results table from comprehensive experiment
"""
import json
import pandas as pd
import numpy as np

# Load results
with open('results/checkpoint.json', 'r') as f:
    data = json.load(f)

# Prepare data for table
rows = []

scenario_names = {
    'XJTU_TJU': 'NCM→NCA',
    'XJTU_MIT': 'NCM→LFP',
    'TJU_MIT': 'NCA→LFP',
    'TJU_XJTU': 'NCA→NCM',
    'MIT_XJTU': 'LFP→NCM',
    'MIT_TJU': 'LFP→NCA',
}

for scenario_key in ['XJTU_TJU', 'XJTU_MIT', 'TJU_MIT', 'TJU_XJTU', 'MIT_XJTU', 'MIT_TJU']:
    if scenario_key not in data:
        continue

    scenario_data = data[scenario_key]
    scenario_name = scenario_names[scenario_key]

    results = scenario_data['results']

    # Get baseline (EXACT) time
    exact_time = results['exact']['time']
    exact_mae = results['exact']['mae']

    for method in ['exact', 'basic', 'richardson', 'adaptive']:
        if method not in results:
            continue

        mae = results[method]['mae']
        time = results[method]['time']
        speedup = exact_time / time
        mae_change = ((mae - exact_mae) / exact_mae) * 100

        rows.append({
            'Scenario': scenario_name,
            'Method': method.upper(),
            'MAE': mae,
            'Time (s)': time,
            'Speedup': speedup,
            'MAE Change (%)': mae_change
        })

df = pd.DataFrame(rows)

# Print summary table
print("="*80)
print("COMPREHENSIVE EXPERIMENT RESULTS - PUBLICATION TABLE")
print("="*80)
print()

# Per-scenario table
for scenario in scenario_names.values():
    scenario_df = df[df['Scenario'] == scenario]
    print(f"\n{scenario}:")
    print(scenario_df.to_string(index=False))
    print()

# Aggregate statistics
print("\n" + "="*80)
print("AGGREGATE STATISTICS (across all scenarios)")
print("="*80)

method_stats = df.groupby('Method').agg({
    'Speedup': ['mean', 'std', 'min', 'max'],
    'MAE Change (%)': ['mean', 'std', 'min', 'max']
}).round(3)

print(method_stats)

# LaTeX table
print("\n" + "="*80)
print("LATEX TABLE")
print("="*80)
print()

latex_table = []
latex_table.append("\\begin{table}[h]")
latex_table.append("\\centering")
latex_table.append("\\caption{Cross-Chemistry Transfer Learning Results: Physics Proxy Methods}")
latex_table.append("\\label{tab:comprehensive_results}")
latex_table.append("\\begin{tabular}{llcccc}")
latex_table.append("\\hline")
latex_table.append("Scenario & Method & MAE & Time (s) & Speedup & MAE Change (\\%) \\\\")
latex_table.append("\\hline")

for scenario in scenario_names.values():
    scenario_df = df[df['Scenario'] == scenario].sort_values('Method')
    first = True
    for _, row in scenario_df.iterrows():
        if first:
            scenario_col = f"\\multirow{{4}}{{*}}{{{scenario}}}"
            first = False
        else:
            scenario_col = ""

        latex_table.append(
            f"{scenario_col} & {row['Method']} & {row['MAE']:.4f} & "
            f"{row['Time (s)']:.1f} & {row['Speedup']:.2f}x & {row['MAE Change (%)']:+.1f} \\\\"
        )
    latex_table.append("\\hline")

latex_table.append("\\end{tabular}")
latex_table.append("\\end{table}")

for line in latex_table:
    print(line)

# CSV export
df.to_csv('results/comprehensive_results_table.csv', index=False)
print("\n[OK] Results exported to: results/comprehensive_results_table.csv")

# Key findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

basic_stats = df[df['Method'] == 'BASIC'].describe()
richardson_stats = df[df['Method'] == 'RICHARDSON'].describe()
adaptive_stats = df[df['Method'] == 'ADAPTIVE'].describe()

print(f"\n1. BASIC FD Proxy:")
print(f"   - Average speedup: {df[df['Method'] == 'BASIC']['Speedup'].mean():.2f}x")
print(f"   - Speedup range: {df[df['Method'] == 'BASIC']['Speedup'].min():.2f}x - {df[df['Method'] == 'BASIC']['Speedup'].max():.2f}x")
print(f"   - Average MAE change: {df[df['Method'] == 'BASIC']['MAE Change (%)'].mean():+.1f}%")

print(f"\n2. RICHARDSON Extrapolation:")
print(f"   - Average speedup: {df[df['Method'] == 'RICHARDSON']['Speedup'].mean():.2f}x")
print(f"   - Speedup range: {df[df['Method'] == 'RICHARDSON']['Speedup'].min():.2f}x - {df[df['Method'] == 'RICHARDSON']['Speedup'].max():.2f}x")
print(f"   - Average MAE change: {df[df['Method'] == 'RICHARDSON']['MAE Change (%)'].mean():+.1f}%")

print(f"\n3. ADAPTIVE (Hybrid):")
print(f"   - Average speedup: {df[df['Method'] == 'ADAPTIVE']['Speedup'].mean():.2f}x")
print(f"   - Speedup range: {df[df['Method'] == 'ADAPTIVE']['Speedup'].min():.2f}x - {df[df['Method'] == 'ADAPTIVE']['Speedup'].max():.2f}x")
print(f"   - Average MAE change: {df[df['Method'] == 'ADAPTIVE']['MAE Change (%)'].mean():+.1f}%")

# Best method by scenario
print("\n4. Best Speedup by Scenario:")
for scenario in scenario_names.values():
    scenario_df = df[(df['Scenario'] == scenario) & (df['Method'] != 'EXACT')]
    best = scenario_df.loc[scenario_df['Speedup'].idxmax()]
    print(f"   {scenario}: {best['Method']} ({best['Speedup']:.2f}x speedup, MAE change: {best['MAE Change (%)']:+.1f}%)")

print("\n" + "="*80)
