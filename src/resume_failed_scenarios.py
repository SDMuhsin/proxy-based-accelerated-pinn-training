"""
RESUME FAILED SCENARIOS - Complete the MIT experiments

This script will rerun only the scenarios that failed due to MIT dataset loading issue:
- Scenario 2: XJTU → MIT
- Scenario 3: TJU → MIT
- Scenario 4: MIT → XJTU
- Scenario 5: MIT → TJU
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the fixed script
from src.final_comprehensive_experiment import (
    run_scenario, log_message, device, torch
)

def main():
    log_message("="*80)
    log_message("RESUMING FAILED MIT SCENARIOS")
    log_message("="*80)
    log_message(f"Device: {device}")
    if device == 'cuda':
        log_message(f"GPU: {torch.cuda.get_device_name(0)}")

    failed_scenarios = [
        ('XJTU', 'MIT', 2),   # NCM → LiFePO4
        ('TJU', 'MIT', 3),    # NCA → LiFePO4
        ('MIT', 'XJTU', 4),   # LiFePO4 → NCM
        ('MIT', 'TJU', 5),    # LiFePO4 → NCA
    ]

    all_results = {}

    for source, target, num in failed_scenarios:
        try:
            results = run_scenario(source, target, num)
            all_results[f"{source}_{target}"] = results
        except Exception as e:
            log_message(f"ERROR in scenario {num}: {str(e)}")
            import traceback
            log_message(traceback.format_exc())
            continue

    log_message("\\n" + "="*80)
    log_message("FAILED SCENARIOS RERUN COMPLETED!")
    log_message("="*80)

if __name__ == '__main__':
    main()
