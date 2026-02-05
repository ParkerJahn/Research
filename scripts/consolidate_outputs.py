#!/usr/bin/env python3
"""
Consolidate All Outputs to FINAL_RESULTS
=========================================

This script ensures all publication-ready outputs are consolidated into
the FINAL_RESULTS directory, regardless of which script generated them.

It will:
1. Copy all relevant tables from results/tables/ to FINAL_RESULTS/tables/
2. Copy all relevant figures from results/figures/ to FINAL_RESULTS/figures/
3. Preserve existing FINAL_RESULTS content (no overwriting unless newer)
4. Generate a manifest of all consolidated files
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def print_header(text):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"   {text}")
    print("="*70 + "\n")

def copy_if_newer(src, dst):
    """Copy file only if source is newer than destination or dst doesn't exist."""
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)
        return True
    return False

def consolidate_tables():
    """Consolidate all tables to FINAL_RESULTS/tables/"""
    print("   Consolidating tables...")
    
    src_dir = config.TABLES_DIR
    dst_dir = config.PROJECT_ROOT / 'FINAL_RESULTS' / 'tables'
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Priority tables to consolidate (most recent/canonical versions)
    priority_tables = [
        'forecast_performance_final.csv',
        'dm_test_results_final.csv',
        'rolling_pca_loadings_mean_std.csv',
        'forecast_summary.csv',
        'forecast_cv_results.csv',
        'stationarity_tests.csv',
        'pca_loadings.csv',
    ]
    
    copied = []
    skipped = []
    
    for table_name in priority_tables:
        src = src_dir / table_name
        if src.exists():
            dst = dst_dir / table_name
            if copy_if_newer(src, dst):
                copied.append(table_name)
                print(f"      ✅ Copied: {table_name}")
            else:
                skipped.append(table_name)
                print(f"      ⏭️  Skipped (up-to-date): {table_name}")
        else:
            print(f"      ⚠️  Not found: {table_name}")
    
    return copied, skipped

def consolidate_figures():
    """Consolidate all figures to FINAL_RESULTS/figures/"""
    print("\n   Consolidating figures...")
    
    src_dir = config.FIGURES_DIR
    dst_dir = config.PROJECT_ROOT / 'FINAL_RESULTS' / 'figures'
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Priority figures to consolidate
    priority_figures = [
        'forecast/figure1_forecast_vs_actual_1d_improved.png',
        'forecast/figure1_forecast_vs_actual_1d_improved.pdf',
        'forecast/figure1_forecast_zoom_2024.png',
        'forecast/figure1_forecast_zoom_2025.png',
        'forecast/pca_loadings_rolling_averaged.png',
        'forecast/forecast_vs_actual_1d.png',
        'forecast/forecast_vs_actual_5d.png',
        'forecast/forecast_vs_actual_22d.png',
        'forecast/pca_loadings_heatmap.png',
        'forecast/sentiment_coefficients.png',
        'forecast/performance_comparison.png',
    ]
    
    copied = []
    skipped = []
    
    for fig_path in priority_figures:
        src = src_dir / fig_path
        if src.exists():
            # Keep same filename, flatten directory structure
            dst = dst_dir / Path(fig_path).name
            if copy_if_newer(src, dst):
                copied.append(Path(fig_path).name)
                print(f"      ✅ Copied: {Path(fig_path).name}")
            else:
                skipped.append(Path(fig_path).name)
                print(f"      ⏭️  Skipped (up-to-date): {Path(fig_path).name}")
        else:
            print(f"      ⚠️  Not found: {fig_path}")
    
    return copied, skipped

def generate_manifest():
    """Generate a manifest of all files in FINAL_RESULTS."""
    print("\n   Generating manifest...")
    
    final_dir = config.PROJECT_ROOT / 'FINAL_RESULTS'
    manifest_path = final_dir / 'MANIFEST.txt'
    
    manifest_lines = [
        "="*70,
        "FINAL_RESULTS MANIFEST",
        "="*70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This directory contains all publication-ready outputs.",
        "All files use the unified metrics module for consistency.",
        "",
        "="*70,
        "TABLES",
        "="*70,
    ]
    
    tables_dir = final_dir / 'tables'
    if tables_dir.exists():
        for f in sorted(tables_dir.glob('*')):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                manifest_lines.append(f"  • {f.name} ({size_kb:.1f} KB)")
    
    manifest_lines.extend([
        "",
        "="*70,
        "FIGURES",
        "="*70,
    ])
    
    figures_dir = final_dir / 'figures'
    if figures_dir.exists():
        for f in sorted(figures_dir.glob('*')):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                manifest_lines.append(f"  • {f.name} ({size_kb:.1f} KB)")
    
    manifest_lines.extend([
        "",
        "="*70,
        "DOCUMENTATION",
        "="*70,
    ])
    
    for f in sorted(final_dir.glob('*.md')):
        size_kb = f.stat().st_size / 1024
        manifest_lines.append(f"  • {f.name} ({size_kb:.1f} KB)")
    
    manifest_lines.extend([
        "",
        "="*70,
        "LOGS",
        "="*70,
    ])
    
    logs_dir = final_dir / 'logs'
    if logs_dir.exists():
        for f in sorted(logs_dir.glob('*')):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                manifest_lines.append(f"  • {f.name} ({size_kb:.1f} KB)")
    
    manifest_lines.append("\n" + "="*70)
    
    with open(manifest_path, 'w') as f:
        f.write('\n'.join(manifest_lines))
    
    print(f"      💾 Saved: MANIFEST.txt")
    return manifest_path

def main():
    """Main consolidation routine."""
    print_header("📦 CONSOLIDATING OUTPUTS TO FINAL_RESULTS")
    
    # Ensure FINAL_RESULTS exists
    final_dir = config.PROJECT_ROOT / 'FINAL_RESULTS'
    final_dir.mkdir(exist_ok=True)
    
    # Consolidate tables
    print_header("STEP 1: Tables")
    tables_copied, tables_skipped = consolidate_tables()
    
    # Consolidate figures
    print_header("STEP 2: Figures")
    figures_copied, figures_skipped = consolidate_figures()
    
    # Generate manifest
    print_header("STEP 3: Manifest")
    manifest_path = generate_manifest()
    
    # Summary
    print_header("📋 CONSOLIDATION SUMMARY")
    print(f"   Tables copied: {len(tables_copied)}")
    print(f"   Tables skipped (up-to-date): {len(tables_skipped)}")
    print(f"   Figures copied: {len(figures_copied)}")
    print(f"   Figures skipped (up-to-date): {len(figures_skipped)}")
    print(f"\n   📁 All outputs consolidated to: FINAL_RESULTS/")
    print(f"   📄 Manifest: {manifest_path.name}")
    
    print("\n" + "="*70)
    print("   ✅ CONSOLIDATION COMPLETE")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
