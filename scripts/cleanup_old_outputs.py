#!/usr/bin/env python3
"""
Clean Up Old Outputs (Pre-Unified Metrics)
===========================================

This script removes all outputs generated before the unified metrics module
was implemented. It will:

1. Archive old tables from results/tables/archive/ (pre-unified metrics)
2. Remove duplicate figures from results/figures/ (now in FINAL_RESULTS/)
3. Keep only the canonical outputs in FINAL_RESULTS/
4. Generate a cleanup report

SAFE: Creates a backup archive before deletion.
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

def create_backup_archive():
    """Create a timestamped backup of files to be deleted."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = config.PROJECT_ROOT / f'OLD_OUTPUTS_BACKUP_{timestamp}'
    backup_dir.mkdir(exist_ok=True)
    
    print(f"   📦 Creating backup: {backup_dir.name}")
    return backup_dir

def cleanup_old_tables(backup_dir):
    """Remove old tables from archive directory."""
    print("   Cleaning up old tables...")
    
    archive_dir = config.TABLES_DIR / 'archive'
    if not archive_dir.exists():
        print("      ℹ️  No archive directory found")
        return []
    
    removed = []
    old_tables = [
        'diebold_mariano_tests.csv',
        'dm_test_results.csv',
        'forecast_performance.csv',
        'forecast_results_cv.csv',
        'forecast_results_final.csv',
        'improved_forecast_results.csv',
        'pca_var_model_comparison.csv',
        'pca_var_model_comparison_fixed.csv',
    ]
    
    # Backup first
    backup_tables = backup_dir / 'tables_archive'
    backup_tables.mkdir(exist_ok=True)
    
    for table_name in old_tables:
        src = archive_dir / table_name
        if src.exists():
            # Backup
            shutil.copy2(src, backup_tables / table_name)
            # Remove
            src.unlink()
            removed.append(table_name)
            print(f"      ✅ Removed: {table_name}")
    
    # Remove archive directory if empty
    if archive_dir.exists() and not list(archive_dir.iterdir()):
        archive_dir.rmdir()
        print(f"      🗑️  Removed empty archive directory")
    
    return removed

def cleanup_duplicate_figures(backup_dir):
    """Remove duplicate figures from results/figures/ (now in FINAL_RESULTS/)."""
    print("\n   Cleaning up duplicate figures...")
    
    figures_dir = config.FIGURES_DIR
    if not figures_dir.exists():
        print("      ℹ️  No figures directory found")
        return []
    
    removed = []
    
    # Backup first
    backup_figures = backup_dir / 'figures'
    backup_figures.mkdir(exist_ok=True)
    
    # Remove all figures that are now in FINAL_RESULTS
    duplicate_patterns = [
        'forecast/*.png',
        'forecast/*.pdf',
        'corsi_style/*.png',
        'corsi_style/*.pdf',
    ]
    
    for pattern in duplicate_patterns:
        for fig_file in figures_dir.glob(pattern):
            # Backup
            backup_subdir = backup_figures / fig_file.parent.name
            backup_subdir.mkdir(exist_ok=True)
            shutil.copy2(fig_file, backup_subdir / fig_file.name)
            # Remove
            fig_file.unlink()
            removed.append(str(fig_file.relative_to(figures_dir)))
            print(f"      ✅ Removed: {fig_file.relative_to(figures_dir)}")
    
    # Remove empty subdirectories
    for subdir in figures_dir.iterdir():
        if subdir.is_dir() and not list(subdir.iterdir()):
            subdir.rmdir()
            print(f"      🗑️  Removed empty directory: {subdir.name}/")
    
    return removed

def cleanup_old_results_tables(backup_dir):
    """Remove old table files from results/tables/ that are superseded."""
    print("\n   Cleaning up superseded tables...")
    
    tables_dir = config.TABLES_DIR
    if not tables_dir.exists():
        print("      ℹ️  No tables directory found")
        return []
    
    removed = []
    
    # Backup first
    backup_tables = backup_dir / 'tables'
    backup_tables.mkdir(exist_ok=True)
    
    # Tables that are superseded by FINAL_RESULTS versions
    superseded_tables = [
        'dm_test_results.csv',  # Superseded by dm_test_results_final.csv
        'pca_loadings.csv',  # Now in FINAL_RESULTS with better version
    ]
    
    for table_name in superseded_tables:
        src = tables_dir / table_name
        if src.exists():
            # Check if we have the final version
            final_version = table_name.replace('.csv', '_final.csv')
            if (tables_dir / final_version).exists() or table_name == 'pca_loadings.csv':
                # Backup
                shutil.copy2(src, backup_tables / table_name)
                # Remove
                src.unlink()
                removed.append(table_name)
                print(f"      ✅ Removed: {table_name}")
    
    return removed

def generate_cleanup_report(backup_dir, removed_tables, removed_figures, removed_superseded):
    """Generate a report of what was cleaned up."""
    report_path = config.PROJECT_ROOT / 'CLEANUP_REPORT.txt'
    
    report_lines = [
        "="*70,
        "CLEANUP REPORT - Old Outputs Removed",
        "="*70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This report documents the removal of all outputs generated before",
        "the unified metrics module was implemented.",
        "",
        f"Backup location: {backup_dir.name}/",
        "",
        "="*70,
        "ARCHIVED TABLES REMOVED",
        "="*70,
    ]
    
    if removed_tables:
        for table in removed_tables:
            report_lines.append(f"  • {table}")
    else:
        report_lines.append("  (none)")
    
    report_lines.extend([
        "",
        "="*70,
        "SUPERSEDED TABLES REMOVED",
        "="*70,
    ])
    
    if removed_superseded:
        for table in removed_superseded:
            report_lines.append(f"  • {table}")
    else:
        report_lines.append("  (none)")
    
    report_lines.extend([
        "",
        "="*70,
        "DUPLICATE FIGURES REMOVED",
        "="*70,
    ])
    
    if removed_figures:
        for fig in removed_figures:
            report_lines.append(f"  • {fig}")
    else:
        report_lines.append("  (none)")
    
    report_lines.extend([
        "",
        "="*70,
        "CURRENT STATE",
        "="*70,
        "",
        "All canonical outputs are now in:",
        "  📁 FINAL_RESULTS/",
        "     ├── tables/     (CSV + LaTeX, using unified metrics)",
        "     ├── figures/    (PNG + PDF, using unified metrics)",
        "     └── MANIFEST.txt (complete inventory)",
        "",
        "All removed files are backed up in:",
        f"  📦 {backup_dir.name}/",
        "",
        "="*70,
    ])
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n   💾 Saved: CLEANUP_REPORT.txt")
    return report_path

def main():
    """Main cleanup routine."""
    print_header("🧹 CLEANING UP OLD OUTPUTS")
    print("   This will remove all pre-unified metrics outputs")
    print("   A backup will be created before deletion")
    
    # Create backup
    print_header("STEP 1: Creating Backup")
    backup_dir = create_backup_archive()
    
    # Cleanup old tables from archive
    print_header("STEP 2: Archived Tables")
    removed_tables = cleanup_old_tables(backup_dir)
    
    # Cleanup superseded tables
    print_header("STEP 3: Superseded Tables")
    removed_superseded = cleanup_old_results_tables(backup_dir)
    
    # Cleanup duplicate figures
    print_header("STEP 4: Duplicate Figures")
    removed_figures = cleanup_duplicate_figures(backup_dir)
    
    # Generate report
    print_header("STEP 5: Cleanup Report")
    report_path = generate_cleanup_report(backup_dir, removed_tables, removed_figures, removed_superseded)
    
    # Summary
    print_header("📋 CLEANUP SUMMARY")
    print(f"   Archived tables removed: {len(removed_tables)}")
    print(f"   Superseded tables removed: {len(removed_superseded)}")
    print(f"   Duplicate figures removed: {len(removed_figures)}")
    print(f"\n   📦 Backup: {backup_dir.name}/")
    print(f"   📄 Report: {report_path.name}")
    print(f"\n   ✅ All canonical outputs are in: FINAL_RESULTS/")
    
    print("\n" + "="*70)
    print("   ✅ CLEANUP COMPLETE")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
