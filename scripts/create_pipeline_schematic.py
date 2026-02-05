#!/usr/bin/env python3
"""
Research Pipeline Schematic
============================
Creates a visual flowchart of the research methodology pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def create_box(ax, x, y, width, height, text, color, text_color='white', fontsize=11, fontweight='bold'):
    """Create a rounded box with text."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05", 
        edgecolor='black', 
        facecolor=color,
        linewidth=2,
        zorder=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=fontsize, fontweight=fontweight, color=text_color,
            zorder=3, wrap=True)

def create_arrow(ax, x1, y1, x2, y2, label='', color='black', linewidth=2):
    """Create an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color=color,
        linewidth=linewidth,
        zorder=1
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def create_pipeline_schematic():
    """Create a clean, streamlined pipeline schematic."""
    
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    # Simplified color scheme
    colors = {
        'data': '#3498db',      # Blue
        'process': '#16a085',   # Teal
        'model': '#e67e22',     # Orange
        'output': '#27ae60',    # Green
    }
    
    y_pos = 12.5
    
    # Title
    ax.text(5, y_pos, 'VIX Forecasting Pipeline', 
            ha='center', fontsize=18, fontweight='bold')
    
    y_pos -= 1.5
    
    # ========== DATA ==========
    create_box(ax, 5, y_pos, 6, 0.8, 
               'Data Sources\nVIX • ETFs (SMH, SOXX) • Commodities (Gold, Oil, Copper) • News Sentiment (AV, FinBERT)', 
               colors['data'], fontsize=10)
    
    y_pos -= 1.3
    create_arrow(ax, 5, y_pos + 0.8, 5, y_pos + 0.3)
    
    # ========== FEATURE ENGINEERING ==========
    create_box(ax, 5, y_pos, 6, 0.8, 
               'Feature Engineering\nReturns & Realized Volatility • HAR Components (1d, 5d, 22d) • Residualized Sentiment', 
               colors['process'], fontsize=10)
    
    y_pos -= 1.3
    create_arrow(ax, 5, y_pos + 0.8, 5, y_pos + 0.3)
    
    # ========== PCA ==========
    create_box(ax, 5, y_pos, 6, 0.8, 
               'Dimensionality Reduction (PCA)\n9 Features → 3 Principal Components (Returns, Volatility, Sentiment)', 
               colors['process'], fontsize=10)
    
    y_pos -= 1.3
    create_arrow(ax, 5, y_pos + 0.8, 5, y_pos + 0.3)
    
    # ========== MODELS (SIDE BY SIDE) ==========
    create_box(ax, 2.8, y_pos, 2.8, 0.9, 
               'Baseline Model\nHAR-IV\n(VIX lags only)', 
               colors['model'], fontsize=10)
    create_box(ax, 7.2, y_pos, 2.8, 0.9, 
               'Augmented Model\nHAR-IV + PCA + Sentiment\n(Baseline + 3 PCs + Shocks)', 
               colors['model'], fontsize=10)
    
    y_pos -= 1.4
    
    # Arrows from both models
    create_arrow(ax, 2.8, y_pos + 0.9, 5, y_pos + 0.3)
    create_arrow(ax, 7.2, y_pos + 0.9, 5, y_pos + 0.3)
    
    # ========== VALIDATION ==========
    create_box(ax, 5, y_pos, 6, 0.8, 
               'Rolling Cross-Validation\nTrain: 200 days • Test: 50 days • Step: 25 days • Horizons: 1d, 5d, 22d', 
               colors['process'], fontsize=10)
    
    y_pos -= 1.3
    create_arrow(ax, 5, y_pos + 0.8, 5, y_pos + 0.3)
    
    # ========== EVALUATION ==========
    create_box(ax, 5, y_pos, 6, 0.8, 
               'Evaluation\nRMSE, MAE, R² • Diebold-Mariano Test • Coefficient Stability Analysis', 
               colors['output'], fontsize=10)
    
    y_pos -= 1.3
    create_arrow(ax, 5, y_pos + 0.8, 5, y_pos + 0.3)
    
    # ========== RESULTS ==========
    create_box(ax, 5, y_pos, 6, 0.7, 
               'Publication Figures\nForecast Comparison • RMSE Charts • PCA Loadings • Coefficient Stability', 
               colors['output'], fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_dir = config.PROJECT_ROOT / 'FINAL_RESULTS' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'pipeline_schematic.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'pipeline_schematic.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("\n" + "="*70)
    print("   ✅ PIPELINE SCHEMATIC CREATED")
    print("="*70)
    print(f"\n   Output directory: {output_dir}")
    print("\n   Files created:")
    print("      • pipeline_schematic.pdf")
    print("      • pipeline_schematic.png")
    print()

if __name__ == '__main__':
    create_pipeline_schematic()
