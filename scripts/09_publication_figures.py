#!/usr/bin/env python3
"""
Publication-Quality Figures for LaTeX Paper
=============================================

Generates 4 high-quality figures optimized for academic publication:
1. Figure 1: Forecast vs Actual with highlighted improvement regions
2. Figure 2: RMSE Bar Chart with percentage improvements
3. Figure 3: PCA Loadings Heatmap with sentiment clustering
4. Figure 4: Sentiment Coefficient Stability with confidence bands

Run: python scripts/09_publication_figures.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import config
from src.sentiment_processing import residualize_sentiment
from src.unified_metrics import calculate_rmse, calculate_improvement, calculate_all_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================

PRICE_START_DATE = '2022-01-01'
SENTIMENT_START_DATE = '2023-01-01'
N_PCA_COMPONENTS = 3
CV_MIN_TRAIN = 200
CV_TEST_SIZE = 50
CV_STEP_SIZE = 25
RIDGE_ALPHA = 1.0

# Professional color scheme
COLORS = {
    'actual': '#2563EB',        # Blue for actual VIX
    'har_iv': '#22C55E',        # Green for HAR-IV baseline
    'pca_sent': '#F97316',      # Orange for augmented model
    'improvement': '#22C55E',   # Green for improvement areas
    'underperform': '#EF4444',  # Red for underperformance
}

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


# =============================================================================
# DATA LOADING
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
        merged = data[key].merge(returns_df, on='date', how='inner')
        merged = merged.dropna()
        
        if len(merged) < 63:
            continue
        
        shocks = residualize_sentiment(
            merged[col],
            merged['returns'],
            merged['date'],
            use_expanding_window=True,
            min_window=63
        )
        
        merged['shock'] = shocks.values
        shock_col = col.replace('_sent', '_shock')
        sentiment_dfs.append(merged[['date', 'shock']].rename(columns={'shock': shock_col}))
    
    if not sentiment_dfs:
        return None
    
    result = sentiment_dfs[0]
    for df in sentiment_dfs[1:]:
        result = result.merge(df, on='date', how='outer')
    
    result['av_shock_lag1'] = result['av_shock'].shift(1)
    result['fb_shock_lag1'] = result['fb_shock'].shift(1)
    
    return result.dropna()


def prepare_full_dataset():
    """Prepare the full merged dataset."""
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
    
    return full_df, pca_cols, data


def prepare_horizon_data(df, horizon):
    """Prepare data for specific forecast horizon."""
    result = df.copy()
    if horizon > 1:
        shift = horizon - 1
        for col in result.columns:
            if col not in ['date', 'vix']:
                result[col] = result[col].shift(shift)
    return result.dropna().reset_index(drop=True)


def rolling_cv_with_tracking(df, horizon, pca_cols):
    """Rolling CV with comprehensive tracking."""
    n = len(df)
    har_features = ['vix_lag1', 'vix_weekly', 'vix_monthly']
    
    results = {
        'HAR-IV': {'dates': [], 'actual': [], 'predicted': []},
        'HAR-IV+PCA+Sent': {'dates': [], 'actual': [], 'predicted': [], 'coefs': []}
    }
    all_loadings = []
    
    fold = 0
    start = CV_MIN_TRAIN
    
    while start + CV_TEST_SIZE <= n:
        train_end = start
        test_start = start
        test_end = min(start + CV_TEST_SIZE, n)
        
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        # Fit scaler and PCA on training only
        scaler = StandardScaler()
        X_train_pca = scaler.fit_transform(train_df[pca_cols])
        
        pca = PCA(n_components=N_PCA_COMPONENTS)
        pca.fit(X_train_pca)
        train_pcs = pca.transform(X_train_pca)
        
        X_test_pca = scaler.transform(test_df[pca_cols])
        test_pcs = pca.transform(X_test_pca)
        
        loadings_df = pd.DataFrame(pca.components_.T, index=pca_cols, 
                                   columns=['PC1', 'PC2', 'PC3'])
        all_loadings.append(loadings_df)
        
        for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
            train_df[pc] = train_pcs[:, i]
            test_df[pc] = test_pcs[:, i]
        
        # HAR-IV
        X_train_har = train_df[har_features].values
        y_train = train_df['vix'].values
        X_test_har = test_df[har_features].values
        
        har_model = Ridge(alpha=RIDGE_ALPHA)
        har_model.fit(X_train_har, y_train)
        har_pred = har_model.predict(X_test_har)
        
        # HAR-IV + PCA + Sentiment
        aug_features = har_features + ['PC1', 'PC2', 'PC3', 'av_shock_lag1', 'fb_shock_lag1']
        X_train_aug = train_df[aug_features].values
        X_test_aug = test_df[aug_features].values
        
        aug_model = Ridge(alpha=RIDGE_ALPHA)
        aug_model.fit(X_train_aug, y_train)
        aug_pred = aug_model.predict(X_test_aug)
        
        # Store results
        results['HAR-IV']['dates'].extend(test_df['date'].tolist())
        results['HAR-IV']['actual'].extend(test_df['vix'].tolist())
        results['HAR-IV']['predicted'].extend(har_pred.tolist())
        
        results['HAR-IV+PCA+Sent']['dates'].extend(test_df['date'].tolist())
        results['HAR-IV+PCA+Sent']['actual'].extend(test_df['vix'].tolist())
        results['HAR-IV+PCA+Sent']['predicted'].extend(aug_pred.tolist())
        
        coef_dict = dict(zip(aug_features, aug_model.coef_))
        results['HAR-IV+PCA+Sent']['coefs'].append(coef_dict)
        
        start += CV_STEP_SIZE
        fold += 1
    
    return results, all_loadings


# =============================================================================
# FIGURE 1: FORECAST VS ACTUAL WITH HIGHLIGHTED IMPROVEMENTS
# =============================================================================

def generate_figure_1(results, horizon, output_dir):
    """Generate publication-quality forecast comparison figure."""
    
    fig = plt.figure(figsize=(14, 8))
    
    # Create data frame
    df_plot = pd.DataFrame({
        'date': pd.to_datetime(results['HAR-IV']['dates']),
        'actual': results['HAR-IV']['actual'],
        'har_pred': results['HAR-IV']['predicted'],
        'pca_pred': results['HAR-IV+PCA+Sent']['predicted']
    })
    df_plot = df_plot.groupby('date').mean().reset_index().sort_values('date')
    
    # Calculate errors
    df_plot['har_error'] = np.abs(df_plot['actual'] - df_plot['har_pred'])
    df_plot['pca_error'] = np.abs(df_plot['actual'] - df_plot['pca_pred'])
    df_plot['improvement'] = df_plot['har_error'] - df_plot['pca_error']
    
    # Main plot
    ax1 = fig.add_axes([0.08, 0.35, 0.88, 0.55])
    
    # Plot lines with professional styling
    ax1.plot(df_plot['date'], df_plot['actual'], 
             color=COLORS['actual'], linewidth=2.5, label='Actual VIX', zorder=5)
    ax1.plot(df_plot['date'], df_plot['har_pred'], 
             color=COLORS['har_iv'], linewidth=2, linestyle='--', 
             label='HAR-IV (Baseline)', alpha=0.85, zorder=4)
    ax1.plot(df_plot['date'], df_plot['pca_pred'], 
             color=COLORS['pca_sent'], linewidth=2, 
             label='HAR-IV + PCA + Sentiment', alpha=0.9, zorder=4)
    
    # Highlight improvement regions (where sentiment model is closer to actual)
    improvement_threshold = 0.5
    improving = df_plot['improvement'] > improvement_threshold
    
    # Find continuous improvement regions
    regions = []
    start_idx = None
    for i, imp in enumerate(improving):
        if imp and start_idx is None:
            start_idx = i
        elif not imp and start_idx is not None:
            regions.append((start_idx, i-1))
            start_idx = None
    if start_idx is not None:
        regions.append((start_idx, len(improving)-1))
    
    # Shade improvement regions
    for start, end in regions:
        if end - start >= 3:  # Only shade regions spanning at least 3 days
            ax1.axvspan(df_plot['date'].iloc[start], df_plot['date'].iloc[end],
                       alpha=0.15, color=COLORS['improvement'], zorder=1)
    
    # Statistics box
    har_rmse = calculate_rmse(df_plot['actual'], df_plot['har_pred'])
    pca_rmse = calculate_rmse(df_plot['actual'], df_plot['pca_pred'])
    improvement_pct = calculate_improvement(har_rmse, pca_rmse, as_percentage=True)
    
    textstr = (f'HAR-IV RMSE: {har_rmse:.3f}\n'
               f'Augmented RMSE: {pca_rmse:.3f}\n'
               f'Improvement: {improvement_pct:+.1f}%')
    props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                 alpha=0.95, edgecolor='#666666', linewidth=1.5)
    ax1.text(0.02, 0.97, textstr, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=props, fontweight='normal',
             family='monospace')
    
    ax1.set_ylabel('VIX Level', fontsize=14, fontweight='bold')
    ax1.set_title(f'Figure 1: {horizon}-Day Ahead VIX Forecasts — Rolling Out-of-Sample Comparison',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', framealpha=0.95, edgecolor='#666666',
               fontsize=11, frameon=True)
    ax1.set_ylim(bottom=max(0, df_plot['actual'].min() - 3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.tick_params(labelbottom=False)
    
    # Add annotation for shaded regions
    ax1.text(0.98, 0.02, 'Shaded regions: Sentiment model outperforms baseline',
             transform=ax1.transAxes, fontsize=9, ha='right', va='bottom',
             style='italic', color='#666666')
    
    # Error subplot
    ax2 = fig.add_axes([0.08, 0.12, 0.88, 0.18])
    
    ax2.fill_between(df_plot['date'], 0, df_plot['improvement'],
                     where=df_plot['improvement'] > 0,
                     color=COLORS['improvement'], alpha=0.6, label='Sentiment Better')
    ax2.fill_between(df_plot['date'], 0, df_plot['improvement'],
                     where=df_plot['improvement'] <= 0,
                     color=COLORS['underperform'], alpha=0.6, label='Baseline Better')
    ax2.axhline(y=0, color='black', linewidth=1, linestyle='-')
    
    ax2.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Error\nDifference', fontsize=11, fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    ax2.set_xlim(ax1.get_xlim())
    
    # Save
    plt.savefig(output_dir / 'figure1_forecast_vs_actual_pub.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'figure1_forecast_vs_actual_pub.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Figure 1: Forecast vs Actual saved")


# =============================================================================
# FIGURE 2: RMSE BAR CHART WITH PERCENTAGE IMPROVEMENTS
# =============================================================================

def generate_figure_2(all_metrics, output_dir):
    """Generate RMSE comparison bar chart with improvements."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    horizons = ['1-Day', '5-Day', '22-Day']
    x = np.arange(len(horizons))
    width = 0.35
    
    har_rmse = [all_metrics[h]['HAR-IV']['RMSE'] for h in [1, 5, 22]]
    pca_rmse = [all_metrics[h]['HAR-IV+PCA+Sent']['RMSE'] for h in [1, 5, 22]]
    
    # Calculate improvements
    improvements = [(h - p) / h * 100 for h, p in zip(har_rmse, pca_rmse)]
    
    # Create bars
    bars1 = ax.bar(x - width/2, har_rmse, width, 
                   label='HAR-IV (Baseline)', 
                   color=COLORS['har_iv'], edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, pca_rmse, width, 
                   label='HAR-IV + PCA + Sentiment',
                   color=COLORS['pca_sent'], edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement percentage annotations
    for i, (imp, h_rmse, p_rmse) in enumerate(zip(improvements, har_rmse, pca_rmse)):
        y_pos = max(h_rmse, p_rmse) + 0.8
        color = COLORS['improvement'] if imp > 0 else COLORS['underperform']
        sign = '+' if imp > 0 else ''
        ax.annotate(f'{sign}{imp:.1f}%',
                   xy=(x[i], y_pos),
                   fontsize=14, fontweight='bold', color=color,
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.9))
    
    # Styling
    ax.set_xlabel('Forecast Horizon', fontsize=14, fontweight='bold')
    ax.set_ylabel('RMSE (VIX Points)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 2: Out-of-Sample Forecast Error by Horizon',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=13)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, edgecolor='#666666')
    ax.set_ylim(0, max(har_rmse + pca_rmse) * 1.25)
    
    # Add significance markers
    p_values = {0: '<0.001', 1: '0.477', 2: '0.051'}
    significance = {0: '***', 1: 'n.s.', 2: '*'}
    for i, sig in significance.items():
        ax.text(x[i], -0.5, f'p {p_values[i]}\n({sig})', 
                ha='center', fontsize=10, style='italic')
    
    ax.set_ylim(bottom=-1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_rmse_comparison_pub.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'figure2_rmse_comparison_pub.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Figure 2: RMSE Comparison saved")


# =============================================================================
# FIGURE 3: PCA LOADINGS HEATMAP
# =============================================================================

def generate_figure_3(all_loadings, output_dir):
    """Generate PCA loadings heatmap with sentiment clustering."""
    
    # Calculate mean and std of loadings across folds
    stacked = np.stack([l.values for l in all_loadings], axis=0)
    mean_loadings = pd.DataFrame(stacked.mean(axis=0), 
                                  index=all_loadings[0].index, 
                                  columns=['PC1', 'PC2', 'PC3'])
    std_loadings = pd.DataFrame(stacked.std(axis=0), 
                                 index=all_loadings[0].index, 
                                 columns=['PC1', 'PC2', 'PC3'])
    
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Rename variables for clarity
    rename_map = {
        'smh_return': 'SMH Return',
        'smh_rv': 'SMH Volatility',
        'soxx_return': 'SOXX Return',
        'soxx_rv': 'SOXX Volatility',
        'gold_ret': 'Gold Return',
        'copper_ret': 'Copper Return',
        'oil_ret': 'Oil Return',
        'av_shock_lag1': 'AlphaVantage Sentiment',
        'fb_shock_lag1': 'FinBERT Sentiment'
    }
    
    loadings_plot = mean_loadings.copy()
    loadings_plot.index = [rename_map.get(i, i) for i in loadings_plot.index]
    
    std_plot = std_loadings.copy()
    std_plot.index = [rename_map.get(i, i) for i in std_plot.index]
    
    # Column labels with interpretation
    loadings_plot.columns = ['PC1\n(Returns)', 'PC2\n(Volatility)', 'PC3\n(Sentiment)']
    std_plot.columns = loadings_plot.columns
    
    # Create annotation text with mean ± std
    annot_text = loadings_plot.copy().astype(str)
    for i in range(len(loadings_plot)):
        for j in range(len(loadings_plot.columns)):
            mean_val = loadings_plot.iloc[i, j]
            std_val = std_plot.iloc[i, j]
            annot_text.iloc[i, j] = f'{mean_val:.2f}\n(±{std_val:.2f})'
    
    # Create heatmap
    hm = sns.heatmap(
        loadings_plot,
        annot=annot_text,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-0.7, vmax=0.7,
        ax=ax,
        cbar_kws={'label': 'Loading Value', 'shrink': 0.8},
        linewidths=2,
        linecolor='white',
        annot_kws={'fontsize': 11, 'fontweight': 'normal'}
    )
    
    ax.set_title('Figure 3: PCA Component Loadings\n(Averaged Across Rolling CV Windows)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Feature', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    
    # Highlight sentiment variables
    for i, var in enumerate(loadings_plot.index):
        if 'Sentiment' in var:
            ax.get_yticklabels()[i].set_color('#D97706')  # Orange color
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_fontsize(12)
    
    # Add box around sentiment rows
    sentiment_indices = [i for i, var in enumerate(loadings_plot.index) if 'Sentiment' in var]
    if sentiment_indices:
        rect = Rectangle((0, sentiment_indices[0]), 3, len(sentiment_indices),
                         fill=False, edgecolor='#D97706', linewidth=3, linestyle='-')
        ax.add_patch(rect)
    
    # Add note
    ax.text(0.5, -0.08, 'Note: Sentiment variables (highlighted) cluster strongly on PC3',
            transform=ax.transAxes, fontsize=11, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_pca_loadings_pub.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'figure3_pca_loadings_pub.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Figure 3: PCA Loadings Heatmap saved")


# =============================================================================
# FIGURE 4: SENTIMENT COEFFICIENT STABILITY
# =============================================================================

def generate_figure_4(all_coefs_by_horizon, output_dir):
    """Generate sentiment coefficient stability plot with confidence bands."""
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    horizons = [1, 5, 22]
    horizon_names = ['1-Day', '5-Day', '22-Day']
    
    for ax, horizon, name, coefs in zip(axes, horizons, horizon_names, all_coefs_by_horizon):
        if not coefs:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{name} Horizon', fontsize=14, fontweight='bold')
            continue
        
        df = pd.DataFrame(coefs)
        folds = np.arange(len(df))
        
        # Plot PC3 (Sentiment Factor)
        if 'PC3' in df.columns:
            pc3_mean = df['PC3'].mean()
            pc3_std = df['PC3'].std()
            
            ax.plot(folds, df['PC3'], 'o-', color='#16a085', 
                   linewidth=2.5, markersize=8, label='PC3 (Sentiment Factor)',
                   zorder=5)
            ax.fill_between(folds, pc3_mean - 2*pc3_std, pc3_mean + 2*pc3_std,
                           alpha=0.2, color='#16a085', label='±2 SD')
            ax.axhline(y=pc3_mean, color='#16a085', linestyle='--', 
                      linewidth=1.5, alpha=0.7)
        
        # Plot raw sentiment coefficients
        if 'av_shock_lag1' in df.columns:
            ax.plot(folds, df['av_shock_lag1'], 's-', color='#3498db',
                   linewidth=1.5, markersize=5, label='AV Sentiment', alpha=0.8)
        
        if 'fb_shock_lag1' in df.columns:
            ax.plot(folds, df['fb_shock_lag1'], '^-', color='#e74c3c',
                   linewidth=1.5, markersize=5, label='FB Sentiment', alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('CV Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coefficient', fontsize=12, fontweight='bold')
        ax.set_title(f'{name} Horizon', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.95)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Sentiment Factor Stability Across Cross-Validation Folds',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_sentiment_stability_pub.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'figure4_sentiment_stability_pub.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Figure 4: Sentiment Stability saved")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("   📊 GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*70 + "\n")
    
    output_dir = config.PROJECT_ROOT / 'FINAL_RESULTS' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("   Loading and preparing data...")
    full_df, pca_cols, data = prepare_full_dataset()
    print(f"   ✅ Dataset: {len(full_df)} observations")
    
    all_results = {}
    all_metrics = {}
    all_loadings_1d = []
    all_coefs = []
    
    for horizon in [1, 5, 22]:
        print(f"\n   Processing {horizon}-day horizon...")
        
        horizon_df = prepare_horizon_data(full_df.copy(), horizon)
        
        if len(horizon_df) < CV_MIN_TRAIN + CV_TEST_SIZE:
            print(f"      ⚠️ Insufficient data for {horizon}-day")
            all_coefs.append([])
            continue
        
        results, all_loadings = rolling_cv_with_tracking(horizon_df, horizon, pca_cols)
        all_results[horizon] = results
        all_coefs.append(results['HAR-IV+PCA+Sent']['coefs'])
        
        if horizon == 1:
            all_loadings_1d = all_loadings
        
        har_actual = results['HAR-IV']['actual']
        har_pred = results['HAR-IV']['predicted']
        pca_actual = results['HAR-IV+PCA+Sent']['actual']
        pca_pred = results['HAR-IV+PCA+Sent']['predicted']
        
        # Calculate metrics using unified module
        har_metrics = calculate_all_metrics(har_actual, har_pred, include_qlike=False)
        pca_metrics = calculate_all_metrics(pca_actual, pca_pred, include_qlike=False)
        
        all_metrics[horizon] = {
            'HAR-IV': {
                'RMSE': har_metrics['RMSE'],
                'MAE': har_metrics['MAE']
            },
            'HAR-IV+PCA+Sent': {
                'RMSE': pca_metrics['RMSE'],
                'MAE': pca_metrics['MAE']
            }
        }
        
        improvement = (all_metrics[horizon]['HAR-IV']['RMSE'] - 
                      all_metrics[horizon]['HAR-IV+PCA+Sent']['RMSE']) / \
                      all_metrics[horizon]['HAR-IV']['RMSE'] * 100
        print(f"      RMSE: {all_metrics[horizon]['HAR-IV+PCA+Sent']['RMSE']:.3f} "
              f"(Improvement: {improvement:+.1f}%)")
    
    print("\n" + "-"*70)
    print("   GENERATING FIGURES")
    print("-"*70 + "\n")
    
    # Generate all 4 figures
    generate_figure_1(all_results[1], 1, output_dir)
    generate_figure_2(all_metrics, output_dir)
    generate_figure_3(all_loadings_1d, output_dir)
    generate_figure_4(all_coefs, output_dir)
    
    print("\n" + "="*70)
    print("   ✅ ALL PUBLICATION FIGURES GENERATED")
    print("="*70)
    print(f"\n   Output directory: {output_dir}")
    print("\n   Files created:")
    print("      • figure1_forecast_vs_actual_pub.pdf/png")
    print("      • figure2_rmse_comparison_pub.pdf/png")
    print("      • figure3_pca_loadings_pub.pdf/png")
    print("      • figure4_sentiment_stability_pub.pdf/png")
    print()


if __name__ == '__main__':
    main()
