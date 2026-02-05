#!/usr/bin/env python3
"""
RMSE Consistency Validation Script
===================================

This script validates that RMSE calculations are consistent across all
outputs, figures, and tables in the research project.

It checks:
1. That all scripts use the unified metrics module
2. That RMSE values match across different output files
3. That no inconsistent RMSE values exist in results

Run: python scripts/validate_rmse_consistency.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
import warnings

import config
from src.unified_metrics import calculate_all_metrics, calculate_improvement


def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_section(title):
    print("\n" + "─"*70)
    print(title)
    print("─"*70)


def extract_rmse_from_csv_files() -> Dict[str, Dict]:
    """Extract RMSE values from all CSV output files."""
    rmse_values = {}
    
    # Check results directory
    results_dir = config.RESULTS_DIR
    if not results_dir.exists():
        print(f"⚠️  Results directory not found: {results_dir}")
        return rmse_values
    
    # Look for CSV files with metrics
    csv_files = list(results_dir.glob('**/*.csv'))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check if RMSE column exists
            rmse_cols = [col for col in df.columns if 'RMSE' in col or 'rmse' in col]
            
            if rmse_cols:
                file_key = csv_file.stem
                rmse_values[file_key] = {
                    'file': str(csv_file),
                    'data': df[rmse_cols].to_dict('records') if len(rmse_cols) > 0 else {}
                }
                print(f"   ✅ Found RMSE in: {csv_file.name}")
        except Exception as e:
            print(f"   ⚠️  Error reading {csv_file.name}: {e}")
    
    return rmse_values


def extract_rmse_from_tables() -> Dict[str, Dict]:
    """Extract RMSE values from table files."""
    rmse_values = {}
    
    tables_dir = config.TABLES_DIR
    if not tables_dir.exists():
        print(f"⚠️  Tables directory not found: {tables_dir}")
        return rmse_values
    
    csv_files = list(tables_dir.glob('*.csv'))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            rmse_cols = [col for col in df.columns if 'RMSE' in col or 'rmse' in col]
            
            if rmse_cols:
                file_key = csv_file.stem
                rmse_values[file_key] = {
                    'file': str(csv_file),
                    'data': df[rmse_cols].to_dict('records') if len(rmse_cols) > 0 else {}
                }
                print(f"   ✅ Found RMSE in: {csv_file.name}")
        except Exception as e:
            print(f"   ⚠️  Error reading {csv_file.name}: {e}")
    
    return rmse_values


def extract_rmse_from_final_results() -> Dict[str, Dict]:
    """Extract RMSE values from FINAL_RESULTS directory."""
    rmse_values = {}
    
    final_dir = config.PROJECT_ROOT / 'FINAL_RESULTS'
    if not final_dir.exists():
        print(f"⚠️  FINAL_RESULTS directory not found: {final_dir}")
        return rmse_values
    
    csv_files = list(final_dir.glob('*.csv'))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            rmse_cols = [col for col in df.columns if 'RMSE' in col or 'rmse' in col]
            
            if rmse_cols:
                file_key = csv_file.stem
                rmse_values[file_key] = {
                    'file': str(csv_file),
                    'data': df[rmse_cols].to_dict('records') if len(rmse_cols) > 0 else {}
                }
                print(f"   ✅ Found RMSE in: {csv_file.name}")
        except Exception as e:
            print(f"   ⚠️  Error reading {csv_file.name}: {e}")
    
    return rmse_values


def compare_rmse_values(all_rmse: Dict[str, Dict], tolerance: float = 0.01) -> List[Dict]:
    """
    Compare RMSE values across different sources and identify inconsistencies.
    
    Parameters
    ----------
    all_rmse : dict
        Dictionary of RMSE values from different sources
    tolerance : float
        Maximum allowed difference between RMSE values (default: 0.01)
        
    Returns
    -------
    inconsistencies : list
        List of detected inconsistencies
    """
    inconsistencies = []
    
    # Extract RMSE values by horizon and model
    rmse_by_horizon_model = {}
    
    for source, data in all_rmse.items():
        if not data.get('data'):
            continue
        
        records = data['data']
        if isinstance(records, list):
            for record in records:
                # Try to identify horizon and model
                horizon = None
                model = None
                rmse_value = None
                
                for key, value in record.items():
                    if 'horizon' in key.lower():
                        horizon = str(value)
                    if 'model' in key.lower():
                        model = str(value)
                    if 'RMSE' in key and 'improvement' not in key.lower():
                        try:
                            rmse_value = float(value) if not pd.isna(value) else None
                        except (ValueError, TypeError):
                            # Try to extract numeric value from string
                            if isinstance(value, str):
                                try:
                                    rmse_value = float(value.replace('%', '').replace('+', '').strip())
                                except:
                                    pass
                
                if horizon and model and rmse_value is not None:
                    key = f"{horizon}_{model}"
                    if key not in rmse_by_horizon_model:
                        rmse_by_horizon_model[key] = []
                    rmse_by_horizon_model[key].append({
                        'source': source,
                        'file': data['file'],
                        'rmse': rmse_value
                    })
    
    # Check for inconsistencies
    for key, values in rmse_by_horizon_model.items():
        if len(values) > 1:
            rmse_vals = [v['rmse'] for v in values]
            min_rmse = min(rmse_vals)
            max_rmse = max(rmse_vals)
            
            if max_rmse - min_rmse > tolerance:
                inconsistencies.append({
                    'horizon_model': key,
                    'min_rmse': min_rmse,
                    'max_rmse': max_rmse,
                    'difference': max_rmse - min_rmse,
                    'sources': values
                })
    
    return inconsistencies


def check_script_imports() -> Dict[str, bool]:
    """Check which scripts import the unified metrics module."""
    scripts_dir = config.PROJECT_ROOT / 'scripts'
    script_files = list(scripts_dir.glob('*.py'))
    
    import_status = {}
    
    for script_file in script_files:
        if script_file.name.startswith('__'):
            continue
        
        try:
            with open(script_file, 'r') as f:
                content = f.read()
                
            uses_unified = 'from src.unified_metrics import' in content or 'import src.unified_metrics' in content
            uses_sklearn_direct = 'from sklearn.metrics import mean_squared_error' in content
            calculates_rmse = 'np.sqrt(mean_squared_error' in content or 'sqrt(mean_squared_error' in content
            
            import_status[script_file.name] = {
                'uses_unified_metrics': uses_unified,
                'uses_sklearn_direct': uses_sklearn_direct,
                'calculates_rmse_inline': calculates_rmse,
                'needs_update': calculates_rmse and not uses_unified
            }
        except Exception as e:
            print(f"   ⚠️  Error checking {script_file.name}: {e}")
    
    return import_status


def generate_consistency_report(output_path: Path = None):
    """Generate comprehensive consistency report."""
    
    if output_path is None:
        output_path = config.RESULTS_DIR / 'rmse_consistency_report.txt'
    
    report_lines = []
    
    report_lines.append("="*70)
    report_lines.append("RMSE CONSISTENCY VALIDATION REPORT")
    report_lines.append("="*70)
    report_lines.append("")
    
    # 1. Check script imports
    print_section("1. Checking Script Imports")
    import_status = check_script_imports()
    
    needs_update = []
    for script, status in import_status.items():
        if status['needs_update']:
            needs_update.append(script)
            print(f"   ⚠️  {script}: Uses inline RMSE calculation (needs update)")
        elif status['uses_unified_metrics']:
            print(f"   ✅ {script}: Uses unified metrics module")
        elif status['calculates_rmse_inline']:
            print(f"   ⚠️  {script}: Calculates RMSE inline")
    
    report_lines.append("1. SCRIPT IMPORT STATUS")
    report_lines.append("-" * 70)
    for script, status in import_status.items():
        report_lines.append(f"{script}:")
        report_lines.append(f"  - Uses unified metrics: {status['uses_unified_metrics']}")
        report_lines.append(f"  - Calculates RMSE inline: {status['calculates_rmse_inline']}")
        report_lines.append(f"  - Needs update: {status['needs_update']}")
    report_lines.append("")
    
    # 2. Extract RMSE from outputs
    print_section("2. Extracting RMSE from Output Files")
    
    all_rmse = {}
    
    print("\n   Results directory:")
    results_rmse = extract_rmse_from_csv_files()
    all_rmse.update(results_rmse)
    
    print("\n   Tables directory:")
    tables_rmse = extract_rmse_from_tables()
    all_rmse.update(tables_rmse)
    
    print("\n   FINAL_RESULTS directory:")
    final_rmse = extract_rmse_from_final_results()
    all_rmse.update(final_rmse)
    
    report_lines.append("2. RMSE VALUES FOUND IN OUTPUT FILES")
    report_lines.append("-" * 70)
    report_lines.append(f"Total files with RMSE: {len(all_rmse)}")
    for source, data in all_rmse.items():
        report_lines.append(f"\n{source}:")
        report_lines.append(f"  File: {data['file']}")
        if data['data']:
            report_lines.append(f"  Records: {len(data['data']) if isinstance(data['data'], list) else 'N/A'}")
    report_lines.append("")
    
    # 3. Compare RMSE values
    print_section("3. Checking for Inconsistencies")
    
    inconsistencies = compare_rmse_values(all_rmse, tolerance=0.01)
    
    if inconsistencies:
        print(f"\n   ❌ Found {len(inconsistencies)} inconsistencies!")
        report_lines.append("3. INCONSISTENCIES DETECTED")
        report_lines.append("-" * 70)
        
        for inc in inconsistencies:
            print(f"\n   Inconsistency in: {inc['horizon_model']}")
            print(f"      Min RMSE: {inc['min_rmse']:.4f}")
            print(f"      Max RMSE: {inc['max_rmse']:.4f}")
            print(f"      Difference: {inc['difference']:.4f}")
            print(f"      Sources:")
            for src in inc['sources']:
                print(f"         - {src['source']}: {src['rmse']:.4f}")
            
            report_lines.append(f"\n{inc['horizon_model']}:")
            report_lines.append(f"  Min RMSE: {inc['min_rmse']:.4f}")
            report_lines.append(f"  Max RMSE: {inc['max_rmse']:.4f}")
            report_lines.append(f"  Difference: {inc['difference']:.4f}")
            report_lines.append(f"  Sources:")
            for src in inc['sources']:
                report_lines.append(f"    - {src['source']}: {src['rmse']:.4f} ({Path(src['file']).name})")
    else:
        print("\n   ✅ No inconsistencies detected (or insufficient data to compare)")
        report_lines.append("3. INCONSISTENCIES")
        report_lines.append("-" * 70)
        report_lines.append("No inconsistencies detected.")
    
    report_lines.append("")
    
    # 4. Recommendations
    report_lines.append("4. RECOMMENDATIONS")
    report_lines.append("-" * 70)
    
    if needs_update:
        report_lines.append("\nScripts that need to be updated to use unified metrics:")
        for script in needs_update:
            report_lines.append(f"  - {script}")
        report_lines.append("\nAction: Update these scripts to import and use src.unified_metrics")
    
    if inconsistencies:
        report_lines.append("\nInconsistent RMSE values detected:")
        report_lines.append("Action: Re-run all analysis scripts using unified metrics module")
        report_lines.append("        to regenerate consistent outputs.")
    
    if not needs_update and not inconsistencies:
        report_lines.append("\n✅ All checks passed! RMSE calculations appear consistent.")
    
    report_lines.append("")
    report_lines.append("="*70)
    
    # Write report
    report_text = "\n".join(report_lines)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n   📄 Report saved to: {output_path}")
    
    return {
        'needs_update': needs_update,
        'inconsistencies': inconsistencies,
        'total_files_checked': len(all_rmse)
    }


def main():
    """Main validation routine."""
    print_header("RMSE CONSISTENCY VALIDATION")
    
    print("This script validates RMSE calculation consistency across all outputs.")
    print("It will:")
    print("  1. Check which scripts use the unified metrics module")
    print("  2. Extract RMSE values from all output files")
    print("  3. Identify any inconsistencies")
    print("  4. Generate a detailed report")
    
    results = generate_consistency_report()
    
    print_section("Summary")
    print(f"   Files checked: {results['total_files_checked']}")
    print(f"   Scripts needing update: {len(results['needs_update'])}")
    print(f"   Inconsistencies found: {len(results['inconsistencies'])}")
    
    if results['needs_update'] or results['inconsistencies']:
        print("\n   ⚠️  ACTION REQUIRED: See report for details")
        return 1
    else:
        print("\n   ✅ All checks passed!")
        return 0


if __name__ == '__main__':
    exit(main())
