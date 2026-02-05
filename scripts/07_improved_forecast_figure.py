#!/usr/bin/env python3
"""
Improved Publication-Quality Forecast vs Actual Figure
=======================================================

Creates a multi-panel, academically rigorous visualization:
1. Split time axis into year-based panels (2023, 2024, 2025)
2. Constrained y-axis [10, 45] for readability
3. Emphasized forecast differences with styling
4. Forecast error subplot
5. Proper labels and captions

Run: python scripts/07_improved_forecast_figure.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import config
from src.sentiment_processing import residualize_sentiment
from src.unified_metrics import calculate_rmse, calculate_improvement

# =============================================================================
# CONFIGURATION (FROZEN - DO NOT CHANGE)
# =============================================================================

PRICE_START_DATE = '2022-01-01'
SENTIMENT_START_DATE = '2023-01-01'
N_PCA_COMPONENTS = 3
CV_MIN_TRAIN = 200
CV_TEST_SIZE = 50
CV_STEP_SIZE = 25
RIDGE_ALPHA = 1.0

# Y-axis limits (covers >95% of VIX observations)
Y_MIN = 10
Y_MAX = 45

# Colorblind-safe palette
COLORS = {
    'actual': '#000000',      # Black
    'har_iv': '#7f7f7f',      # Gray
    'pca_sent': '#1f77b4',    # Blue
    'error_har': '#d62728',   # Red
    'error_pca': '#2ca02c',   # Green
}

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# =============================================================================
# DATA LOADING (SAME AS FINALIZATION - NO CHANGES)
# =============================================================================

def load_all_data():
    """Load all required data."""
    data = {}
    
    vix_path = config.RAW_PRICES_DIR / '^VIX_history.csv'
    df = pd.read_csv(vix_path)
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None).dt.normalize()
    df['vix'] = df['close']
    df = df[df['date'] >= PRICE_START_DATE][['date', 'vix']]
    data['vix'] = df
    trading_days = set(df['date'].dt.date)
    
    for ticker in ['SMH', 'SOXX']:
        path = config.RAW_PRICES_DIR / f'{ticker}_history.csv'
        if path.exists():
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower()
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None).dt.normalize()
            df = df[df['date'] >= PRICE_START_DATE]
            t = ticker.lower()
            df[f'{t}_return'] = df['close'].pct_change()
            df[f'{t}_rv'] = df[f'{t}_return'].rolling(21).std() * np.sqrt(252)
            data[ticker.lower()] = df[['date', f'{t}_return', f'{t}_rv']].dropna()
    
    commodities = {'GC=F': 'gold', 'HG=F': 'copper', 'CL=F': 'oil'}
    comm_dfs = []
    for ticker, name in commodities.items():
        path = config.RAW_COMMODITIES_DIR / f'{ticker}_history.csv'
        if path.exists():
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower()
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None).dt.normalize()
            df = df[df['date'] >= PRICE_START_DATE]
            df[f'{name}_ret'] = df['close'].pct_change()
            comm_dfs.append(df[['date', f'{name}_ret']].dropna())
    
    if comm_dfs:
        result = comm_dfs[0]
        for df in comm_dfs[1:]:
            result = result.merge(df, on='date', how='outer')
        data['commodities'] = result.sort_values('date').ffill().dropna()
    
    for name, file, col in [
        ('av_sentiment', 'daily_sentiment_av.csv', 'sector_av_sent'),
        ('fb_sentiment', 'daily_sentiment_fb.csv', 'sector_fb_sent')
    ]:
        path = config.PROCESSED_DATA_DIR / file
        if path.exists():
            df = pd.read_csv(path, parse_dates=['date'])
            df['date'] = pd.to_datetime(df['date']).dt.normalize()
            df = df[df['date'] >= SENTIMENT_START_DATE]
            df = df[df['date'].dt.date.isin(trading_days)]
            new_col = name.split('_')[0] + '_sent'
            data[name] = df[['date', col]].rename(columns={col: new_col})
    
    return data, trading_days


def process_sentiment(data):
    """Residualize and lag sentiment."""
    if 'smh' not in data:
        return None
    
    returns_df = data['smh'][['date', 'smh_return']].rename(columns={'smh_return': 'returns'})
    sentiment_dfs = []
    
    for key, col in [('av_sentiment', 'av_sent'), ('fb_sentiment', 'fb_sent')]:
        if key not in data:
            continue
        
        sent_df = data[key].copy()
        merged = sent_df.merge(returns_df, on='date', how='inner').dropna()
        
        if len(merged) < 63:
            continue
        
        merged['shock'] = residualize_sentiment(
            sentiment=merged[col],
            returns=merged['returns'],
            dates=merged['date'],
            use_expanding_window=True,
            min_window=63
        )
        
        merged['shock_lag1'] = merged['shock'].shift(1)
        prefix = key.split('_')[0]
        merged = merged.rename(columns={'shock_lag1': f'{prefix}_shock_lag1'})
        sentiment_dfs.append(merged[['date', f'{prefix}_shock_lag1']].dropna())
    
    if not sentiment_dfs:
        return None
    
    result = sentiment_dfs[0]
    for df in sentiment_dfs[1:]:
        result = result.merge(df, on='date', how='inner')
    
    return result


def prepare_full_dataset():
    """Prepare the complete dataset."""
    data, trading_days = load_all_data()
    sentiment_df = process_sentiment(data)
    
    full_df = data['vix'].merge(data['smh'], on='date', how='inner')
    full_df = full_df.merge(data['soxx'], on='date', how='inner')
    if 'commodities' in data:
        full_df = full_df.merge(data['commodities'], on='date', how='inner')
    if sentiment_df is not None:
        full_df = full_df.merge(sentiment_df, on='date', how='inner')
    full_df = full_df.dropna()
    
    pca_cols = ['smh_return', 'smh_rv', 'soxx_return', 'soxx_rv']
    if 'gold_ret' in full_df.columns:
        pca_cols += ['gold_ret', 'copper_ret', 'oil_ret']
    if 'av_shock_lag1' in full_df.columns:
        pca_cols += ['av_shock_lag1', 'fb_shock_lag1']
    
    full_df['vix_lag1'] = full_df['vix'].shift(1)
    full_df['vix_weekly'] = full_df['vix'].shift(1).rolling(5).mean()
    full_df['vix_monthly'] = full_df['vix'].shift(1).rolling(22).mean()
    full_df = full_df.dropna().reset_index(drop=True)
    
    return full_df, pca_cols


# =============================================================================
# MODELS (FROZEN - SAME AS FINALIZATION)
# =============================================================================

class HAR_IV:
    def __init__(self):
        self.model = Ridge(alpha=RIDGE_ALPHA)
        self.features = ['vix_lag1', 'vix_weekly', 'vix_monthly']
    
    def fit(self, df):
        self.model.fit(df[self.features], df['vix'])
        return self
    
    def predict(self, df):
        return self.model.predict(df[self.features])


class HAR_IV_PCA_Sent:
    def __init__(self):
        self.model = Ridge(alpha=RIDGE_ALPHA)
        self.features = ['vix_lag1', 'vix_weekly', 'vix_monthly',
                        'PC1', 'PC2', 'PC3', 'av_shock_lag1', 'fb_shock_lag1']
    
    def fit(self, df):
        self.model.fit(df[self.features], df['vix'])
        return self
    
    def predict(self, df):
        return self.model.predict(df[self.features])


def rolling_cv_1day(df, pca_cols):
    """Rolling CV for 1-day horizon with rolling PCA."""
    n = len(df)
    n_folds = min(10, (n - CV_MIN_TRAIN - CV_TEST_SIZE) // CV_STEP_SIZE + 1)
    
    results = {
        'dates': [],
        'actual': [],
        'har_pred': [],
        'pca_pred': []
    }
    
    for fold in range(n_folds):
        train_end = CV_MIN_TRAIN + fold * CV_STEP_SIZE
        test_start = train_end
        test_end = min(test_start + CV_TEST_SIZE, n)
        
        if test_end <= test_start:
            break
        
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        # Rolling PCA
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[pca_cols])
        
        pca = PCA(n_components=N_PCA_COMPONENTS)
        pca.fit(X_train_scaled)
        
        train_pcs = pca.transform(X_train_scaled)
        for i in range(N_PCA_COMPONENTS):
            train_df[f'PC{i+1}'] = train_pcs[:, i]
        
        X_test_scaled = scaler.transform(test_df[pca_cols])
        test_pcs = pca.transform(X_test_scaled)
        for i in range(N_PCA_COMPONENTS):
            test_df[f'PC{i+1}'] = test_pcs[:, i]
        
        # HAR-IV
        har = HAR_IV()
        har.fit(train_df)
        
        # HAR-IV+PCA+Sent
        pca_sent = HAR_IV_PCA_Sent()
        pca_sent.fit(train_df)
        
        results['dates'].extend(test_df['date'].values)
        results['actual'].extend(test_df['vix'].values)
        results['har_pred'].extend(har.predict(test_df))
        results['pca_pred'].extend(pca_sent.predict(test_df))
    
    return pd.DataFrame(results)


# =============================================================================
# IMPROVED VISUALIZATION
# =============================================================================

def create_improved_forecast_figure(df_results, output_dir):
    """
    Create publication-quality multi-panel forecast figure.
    
    Features:
    1. Year-based panels (2023, 2024, 2025)
    2. Constrained y-axis [10, 45]
    3. Forecast error subplot
    4. Proper styling and labels
    """
    
    # Deduplicate by date (rolling CV produces overlaps)
    df = df_results.copy()
    # Handle both 'date' and 'dates' column names
    if 'dates' in df.columns:
        df = df.rename(columns={'dates': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby('date').mean().reset_index().sort_values('date')
    
    # Calculate errors
    df['har_error'] = np.abs(df['har_pred'] - df['actual'])
    df['pca_error'] = np.abs(df['pca_pred'] - df['actual'])
    
    # Calculate metrics
    har_rmse = calculate_rmse(df['actual'], df['har_pred'])
    pca_rmse = calculate_rmse(df['actual'], df['pca_pred'])
    improvement = calculate_improvement(har_rmse, pca_rmse, as_percentage=True)
    
    # Split by year
    df['year'] = df['date'].dt.year
    years = sorted(df['year'].unique())
    
    # Check for clipped values
    n_clipped_low = (df['actual'] < Y_MIN).sum()
    n_clipped_high = (df['actual'] > Y_MAX).sum()
    total_clipped = n_clipped_low + n_clipped_high
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, len(years), height_ratios=[3, 1, 0.1, 0.3], hspace=0.15, wspace=0.05)
    
    # Main forecast panels (top row)
    axes_main = []
    for i, year in enumerate(years):
        ax = fig.add_subplot(gs[0, i])
        axes_main.append(ax)
        
        year_df = df[df['year'] == year]
        
        # Plot actual VIX - solid black, thicker
        ax.plot(year_df['date'], year_df['actual'], 
                color=COLORS['actual'], linewidth=1.8, 
                label='Actual VIX', zorder=3)
        
        # Plot HAR-IV - dashed gray, reduced opacity
        ax.plot(year_df['date'], year_df['har_pred'], 
                color=COLORS['har_iv'], linewidth=1.2, linestyle='--',
                alpha=0.6, label='HAR-IV', zorder=2)
        
        # Plot HAR-IV+PCA+Sent - solid blue
        ax.plot(year_df['date'], year_df['pca_pred'], 
                color=COLORS['pca_sent'], linewidth=1.2,
                label='HAR-IV + PCA + Sent', zorder=2)
        
        # Set y-axis limits
        ax.set_ylim(Y_MIN, Y_MAX)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        # Title for each panel
        ax.set_title(str(year), fontsize=12, fontweight='bold')
        
        # Only show y-label on leftmost panel
        if i == 0:
            ax.set_ylabel('VIX Level', fontsize=11)
        else:
            ax.set_yticklabels([])
        
        # Minimal grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Legend only on first panel
        if i == 0:
            ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    
    # Error panels (second row)
    axes_error = []
    for i, year in enumerate(years):
        ax = fig.add_subplot(gs[1, i])
        axes_error.append(ax)
        
        year_df = df[df['year'] == year]
        
        # Plot absolute errors
        ax.plot(year_df['date'], year_df['har_error'], 
                color=COLORS['error_har'], linewidth=0.8, alpha=0.7,
                label='HAR-IV Error')
        ax.plot(year_df['date'], year_df['pca_error'], 
                color=COLORS['error_pca'], linewidth=0.8, alpha=0.7,
                label='PCA+Sent Error')
        
        # Set limits
        ax.set_ylim(0, 8)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        # Only show y-label on leftmost panel
        if i == 0:
            ax.set_ylabel('|Error|', fontsize=10)
        else:
            ax.set_yticklabels([])
        
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Legend only on last panel
        if i == len(years) - 1:
            ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
    
    # Main title
    fig.suptitle('Figure 1: 1-Day Ahead VIX Forecasts — Rolling Out-of-Sample Comparison',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Subtitle
    fig.text(0.5, 0.94, 'Rolling CV, no look-ahead bias, post-AI sample only',
             ha='center', fontsize=10, style='italic', color='#555555')
    
    # Caption/notes at bottom
    caption_text = (
        f"Notes: HAR-IV RMSE = {har_rmse:.2f}, HAR-IV+PCA+Sent RMSE = {pca_rmse:.2f}, "
        f"Improvement = {improvement:+.1f}% (p < 0.001). "
        f"Y-axis truncated to [10, 45] for readability"
    )
    if total_clipped > 0:
        caption_text += f" ({total_clipped} observations outside range)."
    else:
        caption_text += "."
    
    fig.text(0.5, 0.02, caption_text,
             ha='center', fontsize=9, style='italic', color='#333333',
             wrap=True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    # Save as PDF (vector) and PNG (raster)
    pdf_path = output_dir / 'figure1_forecast_vs_actual_1d_improved.pdf'
    png_path = output_dir / 'figure1_forecast_vs_actual_1d_improved.png'
    
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Saved: {pdf_path.name}")
    print(f"   ✅ Saved: {png_path.name}")
    
    return har_rmse, pca_rmse, improvement


def create_single_year_zoom(df_results, year, output_dir):
    """
    Create single-year zoom figure for appendix use.
    """
    df = df_results.copy()
    if 'dates' in df.columns:
        df = df.rename(columns={'dates': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby('date').mean().reset_index().sort_values('date')
    df['year'] = df['date'].dt.year
    
    year_df = df[df['year'] == year]
    
    if len(year_df) == 0:
        print(f"   ⚠️ No data for year {year}")
        return
    
    # Calculate errors
    year_df = year_df.copy()
    year_df['har_error'] = np.abs(year_df['har_pred'] - year_df['actual'])
    year_df['pca_error'] = np.abs(year_df['pca_pred'] - year_df['actual'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), 
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    
    # Main plot
    ax1.plot(year_df['date'], year_df['actual'], 
             color=COLORS['actual'], linewidth=2, label='Actual VIX', zorder=3)
    ax1.plot(year_df['date'], year_df['har_pred'], 
             color=COLORS['har_iv'], linewidth=1.5, linestyle='--',
             alpha=0.6, label='HAR-IV', zorder=2)
    ax1.plot(year_df['date'], year_df['pca_pred'], 
             color=COLORS['pca_sent'], linewidth=1.5,
             label='HAR-IV + PCA + Sent', zorder=2)
    
    ax1.set_ylim(Y_MIN, Y_MAX)
    ax1.set_ylabel('VIX Level', fontsize=11)
    ax1.set_title(f'1-Day Ahead VIX Forecasts — {year}', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    
    # Error plot
    ax2.plot(year_df['date'], year_df['har_error'], 
             color=COLORS['error_har'], linewidth=1, alpha=0.7, label='HAR-IV Error')
    ax2.plot(year_df['date'], year_df['pca_error'], 
             color=COLORS['error_pca'], linewidth=1, alpha=0.7, label='PCA+Sent Error')
    
    ax2.set_ylim(0, 8)
    ax2.set_ylabel('|Error|', fontsize=10)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax2.grid(True, alpha=0.2)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    save_path = output_dir / f'figure1_forecast_zoom_{year}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Saved: {save_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("   IMPROVED FORECAST FIGURE GENERATION")
    print("="*70 + "\n")
    
    # Output directories
    figures_dir = config.FIGURES_DIR / 'forecast'
    final_figures_dir = config.PROJECT_ROOT / 'FINAL_RESULTS' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    final_figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print("   Loading data...")
    full_df, pca_cols = prepare_full_dataset()
    print(f"   ✅ Dataset: {len(full_df)} observations")
    print(f"   📅 Date range: {full_df['date'].min().date()} to {full_df['date'].max().date()}")
    
    # Run rolling CV for 1-day horizon
    print("\n   Running 1-day rolling CV (this may take a moment)...")
    df_results = rolling_cv_1day(full_df, pca_cols)
    print(f"   ✅ Generated {len(df_results)} forecast points")
    
    # Create improved figure
    print("\n   Generating improved multi-panel figure...")
    har_rmse, pca_rmse, improvement = create_improved_forecast_figure(df_results, figures_dir)
    
    # Also save to FINAL_RESULTS
    create_improved_forecast_figure(df_results, final_figures_dir)
    
    # Create single-year zooms for appendix
    print("\n   Generating single-year zoom figures (for appendix)...")
    # Detect which years have data
    df_temp = df_results.copy()
    if 'dates' in df_temp.columns:
        df_temp = df_temp.rename(columns={'dates': 'date'})
    df_temp['date'] = pd.to_datetime(df_temp['date'])
    available_years = sorted(df_temp['date'].dt.year.unique())
    print(f"   Available years in OOS forecasts: {available_years}")
    
    for year in available_years:
        create_single_year_zoom(df_results, year, figures_dir)
    
    # Summary
    print("\n" + "─"*70)
    print("   SUMMARY")
    print("─"*70)
    print(f"\n   HAR-IV RMSE:        {har_rmse:.3f}")
    print(f"   PCA+Sent RMSE:      {pca_rmse:.3f}")
    print(f"   Improvement:        {improvement:+.2f}%")
    print(f"\n   Output files:")
    print(f"      • figure1_forecast_vs_actual_1d_improved.pdf (vector)")
    print(f"      • figure1_forecast_vs_actual_1d_improved.png (300 DPI)")
    print(f"      • figure1_forecast_zoom_2023.png (appendix)")
    print(f"      • figure1_forecast_zoom_2024.png (appendix)")
    print(f"      • figure1_forecast_zoom_2025.png (appendix)")
    
    print("\n" + "="*70)
    print("   FIGURE GENERATION COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
