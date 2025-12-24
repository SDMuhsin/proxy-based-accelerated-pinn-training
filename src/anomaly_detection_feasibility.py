"""
Feasibility Study: Physics Residuals as Anomaly Scores for Battery Degradation

This script tests whether physics residuals from a PINN can detect abnormal
battery degradation patterns, supporting industrial anomaly detection applications.

Usage:
    python src/anomaly_detection_feasibility.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import glob
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# Import existing infrastructure
from src.models import PINN
from src.data_loaders import MITdata

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Args:
    """Configuration for PINN and data loading"""
    def __init__(self):
        self.batch = 0
        self.batch_size = 128
        self.normalization_method = 'min-max'
        self.epochs = 50
        self.early_stop = 25
        self.warmup_epochs = 5
        self.warmup_lr = 0.002
        self.lr = 0.01
        self.final_lr = 0.0002
        self.lr_F = 0.001
        self.F_layers_num = 3
        self.F_hidden_dim = 60
        self.alpha = 1.0
        self.beta = 0.02
        self.save_folder = './results/anomaly_feasibility'
        self.log_dir = 'logs'


# ============================================================================
# BATTERY STATISTICS COMPUTATION
# ============================================================================

def get_battery_files(dataset='MIT'):
    """Get all battery CSV files with their cycle counts for a given dataset"""

    if dataset == 'MIT':
        data_root = 'data/MIT data'
        files = glob.glob(os.path.join(data_root, '*/*.csv'))
    elif dataset == 'TJU':
        # Use Dataset_1_NCA_battery (single chemistry, cleanest)
        data_root = 'data/TJU data/Dataset_1_NCA_battery'
        files = glob.glob(os.path.join(data_root, '*.csv'))
    elif dataset == 'XJTU':
        data_root = 'data/XJTU data'
        files = glob.glob(os.path.join(data_root, '*.csv'))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    battery_info = []
    for f in files:
        try:
            # Count lines (cycles) - subtract 1 for header
            with open(f, 'r') as fp:
                cycle_count = sum(1 for _ in fp) - 1

            # Get final capacity from last row
            df = pd.read_csv(f)
            if len(df) > 0:
                final_capacity = df['capacity'].iloc[-1]
                initial_capacity = df['capacity'].iloc[0]
                degradation_rate = (initial_capacity - final_capacity) / max(cycle_count, 1)
            else:
                continue

            battery_info.append({
                'file': f,
                'battery_id': os.path.basename(f).replace('.csv', ''),
                'cycle_count': cycle_count,
                'final_capacity': final_capacity,
                'initial_capacity': initial_capacity,
                'degradation_rate': degradation_rate,
                'dataset': dataset
            })
        except Exception as e:
            print(f"  Warning: Could not load {f}: {e}")

    return pd.DataFrame(battery_info)


def label_anomalies(battery_df, method='combined'):
    """
    Label batteries as normal (0) or anomalous (1)

    Methods:
    - 'early_failure': Bottom 15% by cycle life
    - 'low_soh': Final capacity < 0.70
    - 'fast_degradation': Degradation rate > mean + 2*std
    - 'combined': OR of all above
    """
    labels = np.zeros(len(battery_df), dtype=int)

    # Criterion 1: Early failure (bottom 15% by cycle count)
    threshold_cycles = np.percentile(battery_df['cycle_count'], 15)
    early_failure = battery_df['cycle_count'] < threshold_cycles

    # Criterion 2: Low final SOH (< 0.70)
    low_soh = battery_df['final_capacity'] < 0.70

    # Criterion 3: Fast degradation (> mean + 2*std)
    mean_rate = battery_df['degradation_rate'].mean()
    std_rate = battery_df['degradation_rate'].std()
    fast_degradation = battery_df['degradation_rate'] > (mean_rate + 2 * std_rate)

    if method == 'early_failure':
        labels = early_failure.astype(int).values
    elif method == 'low_soh':
        labels = low_soh.astype(int).values
    elif method == 'fast_degradation':
        labels = fast_degradation.astype(int).values
    elif method == 'combined':
        labels = (early_failure | low_soh | fast_degradation).astype(int).values

    return labels, {
        'early_failure_threshold': threshold_cycles,
        'low_soh_threshold': 0.70,
        'fast_degradation_threshold': mean_rate + 2 * std_rate,
        'n_early_failure': early_failure.sum(),
        'n_low_soh': low_soh.sum(),
        'n_fast_degradation': fast_degradation.sum()
    }


# ============================================================================
# PHYSICS RESIDUAL EXTRACTION
# ============================================================================

def load_pretrained_model(model_path, args):
    """Load a pretrained PINN model"""
    model = PINN(args)
    checkpoint = torch.load(model_path, map_location=device)
    model.solution_u.load_state_dict(checkpoint['solution_u'])
    model.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
    model.solution_u.to(device)
    model.dynamical_F.to(device)
    model.solution_u.eval()
    model.dynamical_F.eval()
    return model


def extract_residuals_for_battery(model, battery_file, args):
    """
    Extract physics residuals for a single battery.

    Returns:
        dict with residual statistics and predictions
    """
    # Load battery data
    df = pd.read_csv(battery_file)
    features = df.iloc[:, :-1].values  # All columns except capacity
    targets = df.iloc[:, -1].values    # Capacity column

    # Add cycle index as last feature (normalized 0-1)
    n_cycles = len(df)
    cycle_indices = np.arange(n_cycles) / max(n_cycles - 1, 1)

    # Create input tensor: [features, cycle_index]
    # The model expects 17 features + time, but MIT data has 16 features
    # We treat cycle index as the temporal coordinate
    features_with_time = np.column_stack([features, cycle_indices])

    # Normalize features (min-max per feature)
    feat_min = features_with_time.min(axis=0, keepdims=True)
    feat_max = features_with_time.max(axis=0, keepdims=True)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1  # Avoid division by zero

    # Handle NaN values
    features_with_time = np.nan_to_num(features_with_time, nan=0.0)
    feat_min = np.nan_to_num(feat_min, nan=0.0)
    feat_range = np.nan_to_num(feat_range, nan=1.0)

    features_norm = (features_with_time - feat_min) / feat_range
    features_norm = np.nan_to_num(features_norm, nan=0.0)  # Final safety check

    # Convert to tensor
    xt = torch.tensor(features_norm, dtype=torch.float32, device=device)
    xt.requires_grad = True

    # Forward pass to get physics residuals
    with torch.enable_grad():
        u, f = model.forward(xt)

    # Extract values
    predictions = u.detach().cpu().numpy().flatten()
    residuals = f.detach().cpu().numpy().flatten()

    # Compute reconstruction error (need to normalize targets same way)
    targets_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)
    reconstruction_error = np.abs(predictions - targets_norm)

    return {
        'residuals': residuals,
        'predictions': predictions,
        'targets': targets,
        'targets_norm': targets_norm,
        'reconstruction_error': reconstruction_error,
        'residual_mean': np.mean(np.abs(residuals)),
        'residual_max': np.max(np.abs(residuals)),
        'residual_std': np.std(residuals),
        'recon_error_mean': np.mean(reconstruction_error),
        'recon_error_max': np.max(reconstruction_error),
        'n_cycles': n_cycles
    }


# ============================================================================
# QUICK SANITY CHECK
# ============================================================================

def run_sanity_check(model, battery_df, n_samples=6):
    """
    Quick sanity check: Compare residuals for short-lived vs long-lived batteries.
    """
    print("\n" + "="*60)
    print("PHASE 1: QUICK SANITY CHECK")
    print("="*60)

    # Sort by cycle count
    sorted_df = battery_df.sort_values('cycle_count')

    # Select 2 shortest, 2 longest, 2 median
    shortest = sorted_df.head(2)
    longest = sorted_df.tail(2)
    median_idx = len(sorted_df) // 2
    median = sorted_df.iloc[median_idx-1:median_idx+1]

    sample_batteries = pd.concat([shortest, median, longest])

    print(f"\nSelected {len(sample_batteries)} batteries for sanity check:")
    print(sample_batteries[['battery_id', 'cycle_count', 'final_capacity']].to_string())

    results = []
    for _, row in sample_batteries.iterrows():
        print(f"\nProcessing {row['battery_id']}...")
        try:
            res = extract_residuals_for_battery(model, row['file'], Args())
            results.append({
                'battery_id': row['battery_id'],
                'cycle_count': row['cycle_count'],
                'final_capacity': row['final_capacity'],
                'residual_mean': res['residual_mean'],
                'residual_max': res['residual_max'],
                'residual_std': res['residual_std'],
                'recon_error_mean': res['recon_error_mean'],
                'category': 'short' if row['cycle_count'] < 200 else ('long' if row['cycle_count'] > 1500 else 'median')
            })
        except Exception as e:
            print(f"  Error: {e}")

    results_df = pd.DataFrame(results)

    print("\n" + "-"*60)
    print("SANITY CHECK RESULTS:")
    print("-"*60)
    print(results_df.to_string())

    # Check if short-lived batteries have higher residuals
    short_residuals = results_df[results_df['category'] == 'short']['residual_mean'].mean()
    long_residuals = results_df[results_df['category'] == 'long']['residual_mean'].mean()

    print(f"\n{'='*40}")
    print(f"Short-lived batteries mean residual: {short_residuals:.6f}")
    print(f"Long-lived batteries mean residual:  {long_residuals:.6f}")
    print(f"Ratio (short/long): {short_residuals/long_residuals:.2f}x")
    print(f"{'='*40}")

    if short_residuals > long_residuals:
        print("\n[OK] SANITY CHECK PASSED: Short-lived batteries have higher residuals!")
        return True, results_df
    else:
        print("\n[FAIL] SANITY CHECK FAILED: No clear pattern. Investigating...")
        return False, results_df


# ============================================================================
# FULL EVALUATION
# ============================================================================

def run_full_evaluation(model, battery_df, labels, args):
    """
    Run full anomaly detection evaluation on all batteries.
    """
    print("\n" + "="*60)
    print("PHASE 3-4: FULL EVALUATION")
    print("="*60)

    all_scores = []

    for idx, row in battery_df.iterrows():
        print(f"\rProcessing {idx+1}/{len(battery_df)}: {row['battery_id']}", end='')
        try:
            res = extract_residuals_for_battery(model, row['file'], args)
            all_scores.append({
                'battery_id': row['battery_id'],
                'label': labels[idx],
                'residual_mean': res['residual_mean'],
                'residual_max': res['residual_max'],
                'residual_std': res['residual_std'],
                'recon_error_mean': res['recon_error_mean'],
                'recon_error_max': res['recon_error_max'],
                'cycle_count': row['cycle_count'],
                'final_capacity': row['final_capacity']
            })
        except Exception as e:
            print(f"\n  Error for {row['battery_id']}: {e}")

    print("\n")
    scores_df = pd.DataFrame(all_scores)

    # Filter out rows with NaN values for metrics computation
    scores_df_valid = scores_df.dropna(subset=['residual_mean', 'recon_error_mean'])
    print(f"  Valid batteries for evaluation: {len(scores_df_valid)}/{len(scores_df)}")

    # Compute metrics for different scoring methods
    y_true = scores_df_valid['label'].values

    metrics = {}
    scoring_methods = {
        'Physics Residual (mean)': 'residual_mean',
        'Physics Residual (max)': 'residual_max',
        'Physics Residual (std)': 'residual_std',
        'Reconstruction Error (mean)': 'recon_error_mean',
        'Reconstruction Error (max)': 'recon_error_max'
    }

    for method_name, col in scoring_methods.items():
        y_score = scores_df_valid[col].values

        try:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)

            # Precision@K
            n = len(y_true)
            k10 = max(1, int(0.1 * n))
            k20 = max(1, int(0.2 * n))

            top_k10_idx = np.argsort(y_score)[-k10:]
            top_k20_idx = np.argsort(y_score)[-k20:]

            prec_10 = np.mean(y_true[top_k10_idx])
            prec_20 = np.mean(y_true[top_k20_idx])

            metrics[method_name] = {
                'AUROC': auroc,
                'AUPRC': auprc,
                'Precision@10%': prec_10,
                'Precision@20%': prec_20
            }
        except Exception as e:
            print(f"Error computing metrics for {method_name}: {e}")
            metrics[method_name] = {'error': str(e)}

    return scores_df, metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_visualizations(scores_df, metrics, output_dir):
    """Generate plots for the feasibility study"""
    os.makedirs(output_dir, exist_ok=True)

    # Drop rows with NaN values for visualization
    scores_df_clean = scores_df.dropna(subset=['residual_mean', 'recon_error_mean'])
    if len(scores_df_clean) == 0:
        print("  WARNING: No valid data for visualization")
        return

    # 1. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))

    y_true = scores_df_clean['label'].values

    for method_name, col in [
        ('Physics Residual (mean)', 'residual_mean'),
        ('Reconstruction Error (mean)', 'recon_error_mean')
    ]:
        if method_name not in metrics or 'error' in metrics.get(method_name, {}):
            continue
        y_score = scores_df_clean[col].values
        if np.isnan(y_score).any():
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = metrics[method_name]['AUROC']
        ax.plot(fpr, tpr, label=f'{method_name} (AUROC={auroc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Physics Residual vs Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150)
    plt.close()

    # 2. Score Distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    normal = scores_df_clean[scores_df_clean['label'] == 0]
    anomalous = scores_df_clean[scores_df_clean['label'] == 1]

    # Physics residual distribution
    axes[0].hist(normal['residual_mean'], bins=20, alpha=0.7, label=f'Normal (n={len(normal)})')
    axes[0].hist(anomalous['residual_mean'], bins=20, alpha=0.7, label=f'Anomalous (n={len(anomalous)})')
    axes[0].set_xlabel('Physics Residual (mean)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Physics Residual Distribution')
    axes[0].legend()

    # Reconstruction error distribution
    axes[1].hist(normal['recon_error_mean'], bins=20, alpha=0.7, label=f'Normal (n={len(normal)})')
    axes[1].hist(anomalous['recon_error_mean'], bins=20, alpha=0.7, label=f'Anomalous (n={len(anomalous)})')
    axes[1].set_xlabel('Reconstruction Error (mean)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Reconstruction Error Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=150)
    plt.close()

    # 3. Scatter plot: Cycle count vs Residual
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        scores_df_clean['cycle_count'],
        scores_df_clean['residual_mean'],
        c=scores_df_clean['label'],
        cmap='RdYlGn_r',
        alpha=0.7,
        s=50
    )
    ax.set_xlabel('Cycle Count')
    ax.set_ylabel('Physics Residual (mean)')
    ax.set_title('Cycle Life vs Physics Residual\n(Red = Anomalous, Green = Normal)')
    plt.colorbar(scatter, label='Anomaly Label')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cycle_vs_residual.png'), dpi=150)
    plt.close()

    print(f"\nVisualizations saved to {output_dir}/")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def get_model_path(dataset):
    """Get pretrained model path for a dataset"""
    model_paths = {
        'MIT': 'pretrained model/model_MIT.pth',
        'TJU': 'pretrained model/model_TJU_0.pth',
        'XJTU': 'pretrained model/model_XJTU_0.pth'
    }
    return model_paths.get(dataset)


def run_single_dataset_experiment(dataset='MIT', output_suffix=''):
    """Run anomaly detection experiment on a single dataset"""
    print("="*70)
    print(f"SINGLE-DATASET EXPERIMENT: {dataset}")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    args = Args()
    output_dir = f'./results/anomaly_feasibility_{dataset.lower()}{output_suffix}'
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load battery statistics
    print("\n[1/5] Loading battery statistics...")
    battery_df = get_battery_files(dataset)
    print(f"  Found {len(battery_df)} batteries")
    print(f"  Cycle range: {battery_df['cycle_count'].min()} - {battery_df['cycle_count'].max()}")
    print(f"  Final capacity range: {battery_df['final_capacity'].min():.3f} - {battery_df['final_capacity'].max():.3f}")

    # Step 2: Load pretrained model
    print("\n[2/5] Loading pretrained model...")
    model_path = get_model_path(dataset)
    model = load_pretrained_model(model_path, args)
    print(f"  Loaded model from {model_path}")

    # Step 3: Quick sanity check
    sanity_passed, sanity_results = run_sanity_check(model, battery_df)

    if not sanity_passed:
        print("\nWARNING: Sanity check did not show expected pattern.")
        print("Proceeding with full evaluation anyway...")

    # Step 4: Label anomalies
    print("\n[3/5] Labeling anomalies...")
    labels, label_stats = label_anomalies(battery_df, method='combined')
    n_anomalies = labels.sum()
    print(f"  Labeled {n_anomalies} anomalies ({100*n_anomalies/len(labels):.1f}%)")
    print(f"  Label stats: {label_stats}")

    # Step 5: Full evaluation
    print("\n[4/5] Running full evaluation...")
    scores_df, metrics = run_full_evaluation(model, battery_df, labels, args)

    # Step 6: Generate visualizations
    print("\n[5/5] Generating visualizations...")
    generate_visualizations(scores_df, metrics, output_dir)

    # Print final results
    print("\n" + "="*70)
    print(f"FINAL RESULTS: {dataset}")
    print("="*70)

    results_table = []
    for method, m in metrics.items():
        if 'error' not in m:
            results_table.append({
                'Method': method,
                'AUROC': f"{m['AUROC']:.3f}",
                'AUPRC': f"{m['AUPRC']:.3f}",
                'Prec@10%': f"{m['Precision@10%']:.3f}",
                'Prec@20%': f"{m['Precision@20%']:.3f}"
            })

    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))

    # Determine pass/fail
    physics_auroc = metrics['Physics Residual (mean)']['AUROC']
    recon_auroc = metrics['Reconstruction Error (mean)']['AUROC']

    print("\n" + "-"*70)
    print("VERDICT:")

    if physics_auroc > 0.75 and physics_auroc > recon_auroc:
        verdict = "STRONG PASS"
        print(f"  [OK] {verdict}: AUROC = {physics_auroc:.3f} > 0.75 AND beats reconstruction error")
    elif physics_auroc > 0.65 and physics_auroc > recon_auroc:
        verdict = "PASS"
        print(f"  [OK] {verdict}: AUROC = {physics_auroc:.3f} > 0.65 AND beats reconstruction error")
    elif physics_auroc > 0.55:
        verdict = "MARGINAL"
        print(f"  ~ {verdict}: AUROC = {physics_auroc:.3f} shows some signal but weak")
    else:
        verdict = "FAIL"
        print(f"  [FAIL] {verdict}: AUROC = {physics_auroc:.3f} near random")

    print("-"*70)

    # Save results
    results = {
        'dataset': dataset,
        'experiment_type': 'single_dataset',
        'verdict': verdict,
        'n_batteries': len(battery_df),
        'n_anomalies': int(n_anomalies),
        'label_stats': label_stats,
        'metrics': metrics,
        'physics_auroc': physics_auroc,
        'recon_auroc': recon_auroc,
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    scores_df.to_csv(os.path.join(output_dir, 'all_scores.csv'), index=False)

    print(f"\nResults saved to {output_dir}/")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


def run_transfer_experiment(source_dataset, target_dataset):
    """
    Run cross-chemistry transfer anomaly detection experiment.

    Train model on source chemistry, use it to detect anomalies in target chemistry.
    This tests whether physics learned on one chemistry can detect violations in another.
    """
    print("="*70)
    print(f"TRANSFER EXPERIMENT: {source_dataset} → {target_dataset}")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    args = Args()
    output_dir = f'./results/anomaly_transfer_{source_dataset.lower()}_to_{target_dataset.lower()}'
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load TARGET battery statistics (where we detect anomalies)
    print(f"\n[1/5] Loading target batteries ({target_dataset})...")
    battery_df = get_battery_files(target_dataset)
    print(f"  Found {len(battery_df)} batteries")
    print(f"  Cycle range: {battery_df['cycle_count'].min()} - {battery_df['cycle_count'].max()}")

    # Step 2: Load SOURCE model (trained on different chemistry)
    print(f"\n[2/5] Loading source model ({source_dataset})...")
    model_path = get_model_path(source_dataset)
    model = load_pretrained_model(model_path, args)
    print(f"  Loaded model from {model_path}")
    print(f"  NOTE: Model trained on {source_dataset}, evaluating on {target_dataset}")

    # Step 3: Label anomalies in target dataset
    print("\n[3/5] Labeling anomalies in target dataset...")
    labels, label_stats = label_anomalies(battery_df, method='combined')
    n_anomalies = labels.sum()
    print(f"  Labeled {n_anomalies} anomalies ({100*n_anomalies/len(labels):.1f}%)")

    # Step 4: Extract residuals and evaluate
    print("\n[4/5] Extracting physics residuals with cross-chemistry model...")
    scores_df, metrics = run_full_evaluation(model, battery_df, labels, args)

    # Step 5: Generate visualizations
    print("\n[5/5] Generating visualizations...")
    generate_visualizations(scores_df, metrics, output_dir)

    # Print final results
    print("\n" + "="*70)
    print(f"TRANSFER RESULTS: {source_dataset} → {target_dataset}")
    print("="*70)

    results_table = []
    for method, m in metrics.items():
        if 'error' not in m:
            results_table.append({
                'Method': method,
                'AUROC': f"{m['AUROC']:.3f}",
                'AUPRC': f"{m['AUPRC']:.3f}",
                'Prec@10%': f"{m['Precision@10%']:.3f}",
                'Prec@20%': f"{m['Precision@20%']:.3f}"
            })

    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))

    # Determine pass/fail
    physics_auroc = metrics['Physics Residual (mean)']['AUROC']
    recon_auroc = metrics['Reconstruction Error (mean)']['AUROC']

    print("\n" + "-"*70)
    print("VERDICT:")

    if physics_auroc > 0.75 and physics_auroc > recon_auroc:
        verdict = "STRONG PASS"
        print(f"  [OK] {verdict}: AUROC = {physics_auroc:.3f} > 0.75 AND beats reconstruction error")
    elif physics_auroc > 0.65 and physics_auroc > recon_auroc:
        verdict = "PASS"
        print(f"  [OK] {verdict}: AUROC = {physics_auroc:.3f} > 0.65 AND beats reconstruction error")
    elif physics_auroc > 0.55:
        verdict = "MARGINAL"
        print(f"  ~ {verdict}: AUROC = {physics_auroc:.3f} shows some signal but weak")
    else:
        verdict = "FAIL"
        print(f"  [FAIL] {verdict}: AUROC = {physics_auroc:.3f} near random")

    print("-"*70)

    # Save results
    results = {
        'source_dataset': source_dataset,
        'target_dataset': target_dataset,
        'experiment_type': 'transfer',
        'verdict': verdict,
        'n_batteries': len(battery_df),
        'n_anomalies': int(n_anomalies),
        'label_stats': label_stats,
        'metrics': metrics,
        'physics_auroc': physics_auroc,
        'recon_auroc': recon_auroc,
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    scores_df.to_csv(os.path.join(output_dir, 'all_scores.csv'), index=False)

    print(f"\nResults saved to {output_dir}/")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


def run_all_experiments():
    """Run complete experiment suite: single-dataset + transfer experiments"""
    all_results = {}

    print("\n" + "#"*70)
    print("# PART A: SINGLE-DATASET VALIDATION")
    print("#"*70)

    # Part A: Single-dataset validation
    for dataset in ['MIT', 'TJU', 'XJTU']:
        try:
            result = run_single_dataset_experiment(dataset)
            all_results[f'single_{dataset}'] = result
        except Exception as e:
            print(f"ERROR in {dataset}: {e}")
            all_results[f'single_{dataset}'] = {'error': str(e)}

    print("\n" + "#"*70)
    print("# PART B: CROSS-CHEMISTRY TRANSFER")
    print("#"*70)

    # Part B: Transfer experiments
    transfer_pairs = [
        ('MIT', 'TJU'),    # LFP → NCA
        ('MIT', 'XJTU'),   # LFP → NCM
        ('TJU', 'MIT'),    # NCA → LFP
        ('TJU', 'XJTU'),   # NCA → NCM
        ('XJTU', 'MIT'),   # NCM → LFP
        ('XJTU', 'TJU'),   # NCM → NCA
    ]

    for source, target in transfer_pairs:
        try:
            result = run_transfer_experiment(source, target)
            all_results[f'transfer_{source}_to_{target}'] = result
        except Exception as e:
            print(f"ERROR in {source}→{target}: {e}")
            all_results[f'transfer_{source}_to_{target}'] = {'error': str(e)}

    # Save master results
    with open('./results/anomaly_detection_master_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*70)
    print("MASTER SUMMARY")
    print("="*70)

    print("\nPart A: Single-Dataset Validation")
    print("-"*50)
    for dataset in ['MIT', 'TJU', 'XJTU']:
        key = f'single_{dataset}'
        if key in all_results and 'error' not in all_results[key]:
            r = all_results[key]
            print(f"  {dataset}: AUROC={r['physics_auroc']:.3f} vs Recon={r['recon_auroc']:.3f} → {r['verdict']}")
        else:
            print(f"  {dataset}: ERROR")

    print("\nPart B: Cross-Chemistry Transfer")
    print("-"*50)
    for source, target in transfer_pairs:
        key = f'transfer_{source}_to_{target}'
        if key in all_results and 'error' not in all_results[key]:
            r = all_results[key]
            print(f"  {source}→{target}: AUROC={r['physics_auroc']:.3f} vs Recon={r['recon_auroc']:.3f} → {r['verdict']}")
        else:
            print(f"  {source}→{target}: ERROR")

    return all_results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            results = run_all_experiments()
        elif sys.argv[1] == 'transfer' and len(sys.argv) == 4:
            results = run_transfer_experiment(sys.argv[2], sys.argv[3])
        else:
            results = run_single_dataset_experiment(sys.argv[1])
    else:
        # Default: run MIT only
        results = run_single_dataset_experiment('MIT')
