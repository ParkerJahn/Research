#!/usr/bin/env python3
"""
Publication-Quality Outputs for PCA-VARX VIX Forecasting
=========================================================

Generates all figures and tables for research paper:
1. Forecast vs Actual plots (1d, 5d, 22d)
2. PCA Loadings Heatmap
3. Sentiment Coefficient Over Time
4. Performance Comparison Chart
5. Summary Tables

IMPORTANT: This script uses ROLLING PCA - PCA is refitted at each cross-validation
fold using only training data to avoid look-ahead bias.

Run after: python scripts/03_pca_varx.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

import config
from src.sentiment_processing import residualize_sentiment
from src.unified_metrics import calculate_rmse, calculate_improvement, calculate_all_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================

PRICE_START_DATE = '2022-01-01'
SENTIMENT_START_DATE = '2023-01-01'
FORECAST_HORIZONS = [1, 5, 22]
N_PCA_COMPONENTS = 3
CV_MIN_TRAIN = 200
CV_TEST_SIZE = 50
CV_STEP_SIZE = 25
RIDGE_ALPHA = 1.0

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Color palette
COLORS = {
    'actual': '#1a1a2e',
    'HAR-IV': '#e94560',
    'HAR-IV+Sentiment': '#0f3460',
    'HAR-IV+PCA+Sent': '#16a085',
    'Random Walk': '#95a5a6',
    'AR(1)': '#8e44ad'
}


def print_header(title):
    print("\n" + "="*70)
    print(f"          {title}")
    print("="*70 + "\n")


def print_section(title):
    print("\n" + "─"*70)
    print(title)
    print("─"*70)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """Load all required data."""
    data = {}
    
    # VIX
    vix_path = config.RAW_PRICES_DIR / '^VIX_history.csv'
    df = pd.read_csv(vix_path)
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None).dt.normalize()
    df['vix'] = df['close']
    df = df[df['date'] >= PRICE_START_DATE][['date', 'vix']]
    data['vix'] = df
    
    # ETFs
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
    
    # Commodities
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
    
    # Sentiment
    for name, file, col in [
        ('av_sentiment', 'daily_sentiment_av.csv', 'sector_av_sent'),
        ('fb_sentiment', 'daily_sentiment_fb.csv', 'sector_fb_sent')
    ]:
        path = config.PROCESSED_DATA_DIR / file
        if path.exists():
            df = pd.read_csv(path, parse_dates=['date'])
            df['date'] = pd.to_datetime(df['date']).dt.normalize()
            df = df[df['date'] >= SENTIMENT_START_DATE]
            new_col = name.split('_')[0] + '_sent'
            data[name] = df[['date', col]].rename(columns={col: new_col})
    
    return data


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
    """
    Prepare the complete dataset WITHOUT pre-computed PCA features.
    PCA will be computed rolling inside the CV loop to avoid look-ahead bias.
    
    Returns:
        full_df: DataFrame with raw features (no PC columns)
        pca_cols: List of column names to use for PCA
        full_sample_loadings: PCA loadings from full sample (for visualization only)
    """
    print("   Loading and processing data...")
    data = load_all_data()
    sentiment_df = process_sentiment(data)
    
    full_df = data['vix'].merge(data['smh'], on='date', how='inner')
    full_df = full_df.merge(data['soxx'], on='date', how='inner')
    if 'commodities' in data:
        full_df = full_df.merge(data['commodities'], on='date', how='inner')
    if sentiment_df is not None:
        full_df = full_df.merge(sentiment_df, on='date', how='inner')
    full_df = full_df.dropna()
    
    # Define PCA columns (but don't compute PCA yet)
    pca_cols = ['smh_return', 'smh_rv', 'soxx_return', 'soxx_rv']
    if 'gold_ret' in full_df.columns:
        pca_cols += ['gold_ret', 'copper_ret', 'oil_ret']
    if 'av_shock_lag1' in full_df.columns:
        pca_cols += ['av_shock_lag1', 'fb_shock_lag1']
    
    # Compute full-sample PCA for visualization/description ONLY
    # This is NOT used in forecasting - just for the loadings heatmap
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(full_df[pca_cols])
    pca = PCA(n_components=N_PCA_COMPONENTS)
    pca.fit(X_scaled)
    
    full_sample_loadings = pd.DataFrame(
        pca.components_.T,
        index=pca_cols,
        columns=[f'PC{i+1}' for i in range(N_PCA_COMPONENTS)]
    )
    
    # HAR-IV features (these are fine - they use lagged values)
    full_df['vix_lag1'] = full_df['vix'].shift(1)
    full_df['vix_weekly'] = full_df['vix'].shift(1).rolling(5).mean()
    full_df['vix_monthly'] = full_df['vix'].shift(1).rolling(22).mean()
    full_df = full_df.dropna().reset_index(drop=True)
    
    return full_df, pca_cols, full_sample_loadings


# =============================================================================
# MODELS
# =============================================================================

class HAR_IV:
    """HAR-IV baseline model - uses only lagged VIX features."""
    def __init__(self):
        self.model = Ridge(alpha=RIDGE_ALPHA)
        self.features = ['vix_lag1', 'vix_weekly', 'vix_monthly']
        self.name = 'HAR-IV'
    
    def fit(self, df):
        self.model.fit(df[self.features], df['vix'])
        self.coef_ = dict(zip(self.features, self.model.coef_))
        return self
    
    def predict(self, df):
        return self.model.predict(df[self.features])


class HAR_IV_PCA_Sent:
    """HAR-IV + PCA + Sentiment model - expects PC columns to already exist in df."""
    def __init__(self):
        self.model = Ridge(alpha=RIDGE_ALPHA)
        self.features = ['vix_lag1', 'vix_weekly', 'vix_monthly',
                        'PC1', 'PC2', 'PC3', 'av_shock_lag1', 'fb_shock_lag1']
        self.name = 'HAR-IV+PCA+Sent'
    
    def fit(self, df):
        self.model.fit(df[self.features], df['vix'])
        self.coef_ = dict(zip(self.features, self.model.coef_))
        return self
    
    def predict(self, df):
        return self.model.predict(df[self.features])


def prepare_horizon_data(df, horizon):
    """Shift features for h-step forecast."""
    result = df.copy()
    if horizon > 1:
        shift = horizon - 1
        for col in result.columns:
            if col not in ['date', 'vix']:
                result[col] = result[col].shift(shift)
    return result.dropna().reset_index(drop=True)


def rolling_cv_with_coefficients(df, horizon, pca_cols):
    """
    Rolling CV with PROPER rolling PCA - no look-ahead bias.
    
    At each fold:
    1. Fit StandardScaler on training data only
    2. Fit PCA on training data only
    3. Transform both train and test using training-fitted scaler/PCA
    4. Fit forecasting models and predict
    """
    n = len(df)
    n_folds = min(10, (n - CV_MIN_TRAIN - CV_TEST_SIZE) // CV_STEP_SIZE + 1)
    
    results = {
        'HAR-IV': {'actual': [], 'predicted': [], 'dates': []},
        'HAR-IV+PCA+Sent': {'actual': [], 'predicted': [], 'dates': [], 'coefs': []}
    }
    
    for fold in range(n_folds):
        train_end = CV_MIN_TRAIN + fold * CV_STEP_SIZE
        test_start = train_end
        test_end = min(test_start + CV_TEST_SIZE, n)
        
        if test_end <= test_start:
            break
        
        # Split data
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        # === ROLLING PCA: Fit on training data ONLY ===
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[pca_cols])
        
        pca = PCA(n_components=N_PCA_COMPONENTS)
        pca.fit(X_train_scaled)
        
        # Transform training data
        train_pcs = pca.transform(X_train_scaled)
        for i in range(N_PCA_COMPONENTS):
            train_df[f'PC{i+1}'] = train_pcs[:, i]
        
        # Transform test data using TRAINING-fitted scaler and PCA
        X_test_scaled = scaler.transform(test_df[pca_cols])
        test_pcs = pca.transform(X_test_scaled)
        for i in range(N_PCA_COMPONENTS):
            test_df[f'PC{i+1}'] = test_pcs[:, i]
        
        # === HAR-IV (baseline - no PCA needed) ===
        har = HAR_IV()
        har.fit(train_df)
        results['HAR-IV']['actual'].extend(test_df['vix'].values)
        results['HAR-IV']['predicted'].extend(har.predict(test_df))
        results['HAR-IV']['dates'].extend(test_df['date'].values)
        
        # === HAR-IV+PCA+Sent ===
        pca_sent = HAR_IV_PCA_Sent()
        pca_sent.fit(train_df)
        results['HAR-IV+PCA+Sent']['actual'].extend(test_df['vix'].values)
        results['HAR-IV+PCA+Sent']['predicted'].extend(pca_sent.predict(test_df))
        results['HAR-IV+PCA+Sent']['dates'].extend(test_df['date'].values)
        results['HAR-IV+PCA+Sent']['coefs'].append({
            'fold': fold,
            'train_end_date': train_df['date'].iloc[-1],
            **pca_sent.coef_
        })
    
    # Convert to arrays
    for name in results:
        results[name]['actual'] = np.array(results[name]['actual'])
        results[name]['predicted'] = np.array(results[name]['predicted'])
        results[name]['dates'] = np.array(results[name]['dates'])
    
    return results


# =============================================================================
# PUBLICATION FIGURES
# =============================================================================

def plot_forecast_vs_actual_publication(results, horizon, save_path):
    """Publication-quality forecast vs actual plot."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Create DataFrame to deduplicate and sort by date
    # Rolling CV produces overlapping windows with duplicate dates - we average them
    df_plot = pd.DataFrame({
        'date': pd.to_datetime(results['HAR-IV']['dates']),
        'actual': results['HAR-IV']['actual'],
        'har_pred': results['HAR-IV']['predicted'],
        'pca_pred': results['HAR-IV+PCA+Sent']['predicted']
    })
    
    # Aggregate duplicates by averaging predictions for the same date
    df_plot = df_plot.groupby('date').mean().reset_index().sort_values('date')
    
    dates = df_plot['date']
    actual = df_plot['actual'].values
    har_pred = df_plot['har_pred'].values
    pca_pred = df_plot['pca_pred'].values
    
    # Actual VIX
    ax.plot(dates, actual, color=COLORS['actual'], linewidth=2, 
            label='Actual VIX', zorder=3)
    
    # HAR-IV baseline
    ax.plot(dates, har_pred, 
            color=COLORS['HAR-IV'], linewidth=1.5, alpha=0.8,
            label='HAR-IV (Baseline)', linestyle='--')
    
    # HAR-IV+PCA+Sentiment
    ax.plot(dates, pca_pred,
            color=COLORS['HAR-IV+PCA+Sent'], linewidth=1.5, alpha=0.8,
            label='HAR-IV + PCA + Sentiment')
    
    # Calculate metrics for annotation (using deduplicated data)
    har_rmse = calculate_rmse(actual, har_pred)
    pca_rmse = calculate_rmse(actual, pca_pred)
    improvement = calculate_improvement(har_rmse, pca_rmse, as_percentage=True)
    
    # Add metrics box
    textstr = f'HAR-IV RMSE: {har_rmse:.2f}\nPCA+Sent RMSE: {pca_rmse:.2f}\nImprovement: {improvement:+.1f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('VIX (Volatility Points)')
    ax.set_title(f'VIX {horizon}-Day Ahead Forecasts: Model Comparison', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    ax.set_ylim(bottom=max(0, actual.min() - 5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   📊 Saved: {save_path.name}")


def plot_pca_loadings_publication(loadings, save_path):
    """
    Publication-quality PCA loadings heatmap.
    
    IMPORTANT: This uses full-sample PCA for DESCRIPTIVE purposes only.
    For publication, use 05_audit_and_fix.py which generates rolling-averaged loadings.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Rename variables for publication
    rename_map = {
        'smh_return': 'SMH Return',
        'smh_rv': 'SMH Volatility',
        'soxx_return': 'SOXX Return',
        'soxx_rv': 'SOXX Volatility',
        'gold_ret': 'Gold Return',
        'copper_ret': 'Copper Return',
        'oil_ret': 'Oil Return',
        'av_shock_lag1': 'AV Sentiment (t-1)',
        'fb_shock_lag1': 'FB Sentiment (t-1)'
    }
    
    loadings_plot = loadings.copy()
    loadings_plot.index = [rename_map.get(i, i) for i in loadings_plot.index]
    loadings_plot.columns = ['PC1\n(Returns)', 'PC2\n(Volatility)', 'PC3\n(Sentiment)']
    
    # Create heatmap
    sns.heatmap(
        loadings_plot,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-0.7, vmax=0.7,
        ax=ax,
        cbar_kws={'label': 'Loading', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='white'
    )
    
    ax.set_title('PCA Component Loadings\n(Full Sample - Descriptive Only)\n⚠️ For publication: use rolling-averaged loadings', 
                 fontweight='bold', pad=20, fontsize=12)
    ax.set_ylabel('Feature')
    ax.set_xlabel('')
    
    # Highlight sentiment variables
    for i, var in enumerate(loadings_plot.index):
        if 'Sentiment' in var:
            ax.get_yticklabels()[i].set_color('#16a085')
            ax.get_yticklabels()[i].set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   📊 Saved: {save_path.name}")
    print(f"   ⚠️ NOTE: For publication, use 05_audit_and_fix.py for rolling-averaged PCA loadings")


def plot_sentiment_coefficients(all_coefs, save_path):
    """Plot sentiment-related coefficients over rolling CV folds."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    horizons = [1, 5, 22]
    
    for ax, (horizon, coefs) in zip(axes, zip(horizons, all_coefs)):
        if not coefs:
            continue
        
        df = pd.DataFrame(coefs)
        
        # Plot PC3 and sentiment coefficients
        if 'PC3' in df.columns:
            ax.plot(df['fold'], df['PC3'], 'o-', color='#16a085', 
                   label='PC3 (Sentiment Factor)', linewidth=2, markersize=4)
        
        if 'av_shock_lag1' in df.columns:
            ax.plot(df['fold'], df['av_shock_lag1'], 's--', color='#3498db',
                   label='AV Sentiment', linewidth=1.5, markersize=3, alpha=0.7)
        
        if 'fb_shock_lag1' in df.columns:
            ax.plot(df['fold'], df['fb_shock_lag1'], '^--', color='#e74c3c',
                   label='FB Sentiment', linewidth=1.5, markersize=3, alpha=0.7)
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('CV Fold')
        ax.set_ylabel('Coefficient')
        ax.set_title(f'{horizon}-Day Horizon', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle('Sentiment Coefficient Evolution Across Cross-Validation Folds\n(Rolling PCA - No Look-Ahead Bias)', 
                 fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   📊 Saved: {save_path.name}")


def plot_performance_comparison_publication(metrics_all, save_path):
    """Publication-quality performance comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    metrics_list = ['RMSE', 'MAE', 'Improvement (%)']
    
    for ax, metric in zip(axes, metrics_list):
        horizons = ['1-Day', '5-Day', '22-Day']
        x = np.arange(len(horizons))
        width = 0.35
        
        if metric == 'Improvement (%)':
            improvements = []
            for h in [1, 5, 22]:
                if h in metrics_all:
                    har = metrics_all[h]['HAR-IV']['RMSE']
                    pca = metrics_all[h]['HAR-IV+PCA+Sent']['RMSE']
                    improvements.append((har - pca) / har * 100)
                else:
                    improvements.append(0)
            
            colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
            bars = ax.bar(x, improvements, width*1.5, color=colors, edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax.annotate(f'{imp:+.1f}%',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3 if height >= 0 else -12),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=10, fontweight='bold')
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylabel('RMSE Improvement (%)')
            
        else:
            har_vals = [metrics_all[h]['HAR-IV'][metric] if h in metrics_all else 0 for h in [1, 5, 22]]
            pca_vals = [metrics_all[h]['HAR-IV+PCA+Sent'][metric] if h in metrics_all else 0 for h in [1, 5, 22]]
            
            bars1 = ax.bar(x - width/2, har_vals, width, label='HAR-IV', 
                          color=COLORS['HAR-IV'], edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, pca_vals, width, label='HAR-IV+PCA+Sent',
                          color=COLORS['HAR-IV+PCA+Sent'], edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
            
            ax.set_ylabel(f'{metric} (VIX Points)')
            ax.legend(loc='upper left', fontsize=9)
        
        ax.set_xlabel('Forecast Horizon')
        ax.set_title(metric, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
    
    fig.suptitle('Forecast Performance Comparison: HAR-IV vs HAR-IV+PCA+Sentiment\n(Rolling PCA - No Look-Ahead Bias)',
                 fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   📊 Saved: {save_path.name}")


def create_summary_table(metrics_all, dm_results, save_path):
    """Create comprehensive summary table."""
    rows = []
    
    for horizon in [1, 5, 22]:
        if horizon not in metrics_all:
            continue
        
        har = metrics_all[horizon]['HAR-IV']
        pca = metrics_all[horizon]['HAR-IV+PCA+Sent']
        dm = dm_results.get(horizon, {})
        
        improvement = (har['RMSE'] - pca['RMSE']) / har['RMSE'] * 100
        
        rows.append({
            'Horizon': f'{horizon}-Day',
            'HAR-IV RMSE': f"{har['RMSE']:.3f}",
            'HAR-IV MAE': f"{har['MAE']:.3f}",
            'PCA+Sent RMSE': f"{pca['RMSE']:.3f}",
            'PCA+Sent MAE': f"{pca['MAE']:.3f}",
            'Improvement': f"{improvement:+.2f}%",
            'DM p-value': f"{dm.get('p_value', 'N/A'):.4f}" if isinstance(dm.get('p_value'), float) else 'N/A',
            'Significant': '✓' if dm.get('significant', False) else ''
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"   💾 Saved: {save_path.name}")
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("📊 PUBLICATION-QUALITY OUTPUTS")
    print("   ⚠️  Using ROLLING PCA to avoid look-ahead bias")
    print("   ⚠️  For final publication outputs, also run 05_audit_and_fix.py")
    
    figures_dir = config.FIGURES_DIR / 'forecast'
    tables_dir = config.TABLES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # === LOAD DATA ===
    print_section("STEP 1: Preparing Data")
    full_df, pca_cols, full_sample_loadings = prepare_full_dataset()
    print(f"   ✅ Dataset ready: {len(full_df)} observations")
    print(f"   ✅ PCA columns: {pca_cols}")
    
    # === RUN FORECASTING FOR ALL HORIZONS ===
    print_section("STEP 2: Running Multi-Horizon Forecasts (Rolling PCA)")
    
    all_results = {}
    all_coefs = []
    metrics_all = {}
    dm_results = {}
    
    for horizon in FORECAST_HORIZONS:
        print(f"\n   Horizon: {horizon}-day")
        
        horizon_df = prepare_horizon_data(full_df.copy(), horizon)
        
        if len(horizon_df) < CV_MIN_TRAIN + CV_TEST_SIZE:
            print(f"      ⚠️ Insufficient data")
            all_coefs.append([])
            continue
        
        # Run rolling CV with proper rolling PCA
        results = rolling_cv_with_coefficients(horizon_df, horizon, pca_cols)
        all_results[horizon] = results
        all_coefs.append(results['HAR-IV+PCA+Sent']['coefs'])
        
        # Calculate metrics
        har_actual = results['HAR-IV']['actual']
        har_pred = results['HAR-IV']['predicted']
        pca_actual = results['HAR-IV+PCA+Sent']['actual']
        pca_pred = results['HAR-IV+PCA+Sent']['predicted']
        
        # Calculate metrics using unified module
        har_metrics = calculate_all_metrics(har_actual, har_pred, include_qlike=False)
        pca_metrics = calculate_all_metrics(pca_actual, pca_pred, include_qlike=False)
        
        metrics_all[horizon] = {
            'HAR-IV': {
                'RMSE': har_metrics['RMSE'],
                'MAE': har_metrics['MAE']
            },
            'HAR-IV+PCA+Sent': {
                'RMSE': pca_metrics['RMSE'],
                'MAE': pca_metrics['MAE']
            }
        }
        
        # DM test with horizon-adjusted Newey-West variance
        n_common = min(len(har_actual), len(pca_actual))
        e1 = (pca_actual[:n_common] - pca_pred[:n_common]) ** 2
        e2 = (har_actual[:n_common] - har_pred[:n_common]) ** 2
        d = e1 - e2
        
        n = len(d)
        d_mean = np.mean(d)
        
        # Newey-West variance with lag = h-1 (horizon adjustment)
        nw_lag = max(0, horizon - 1)
        gamma0 = np.var(d, ddof=1)
        gamma = []
        for k in range(1, nw_lag + 1):
            if k < n:
                cov_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
                gamma.append(cov_k)
        
        nw_variance = gamma0
        for k, g in enumerate(gamma, 1):
            weight = 1 - k / (nw_lag + 1)  # Bartlett kernel
            nw_variance += 2 * weight * g
        nw_variance = max(nw_variance, 1e-10)
        
        dm_stat = d_mean / np.sqrt(nw_variance / n)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        
        dm_results[horizon] = {
            'statistic': dm_stat,
            'p_value': p_value,
            'significant': p_value < 0.05 and dm_stat < 0,
            'nw_lag': nw_lag
        }
        
        print(f"      HAR-IV RMSE: {metrics_all[horizon]['HAR-IV']['RMSE']:.3f}")
        print(f"      PCA+Sent RMSE: {metrics_all[horizon]['HAR-IV+PCA+Sent']['RMSE']:.3f}")
        print(f"      DM p-value: {p_value:.4f}")
    
    # === GENERATE FIGURES ===
    print_section("STEP 3: Generating Publication Figures")
    
    # 1. Forecast vs Actual plots
    for horizon in FORECAST_HORIZONS:
        if horizon in all_results:
            plot_forecast_vs_actual_publication(
                all_results[horizon],
                horizon,
                figures_dir / f'forecast_vs_actual_{horizon}d.png'
            )
    
    # 2. PCA Loadings Heatmap (descriptive only)
    plot_pca_loadings_publication(full_sample_loadings, figures_dir / 'pca_loadings_heatmap.png')
    
    # 3. Sentiment Coefficients
    plot_sentiment_coefficients(all_coefs, figures_dir / 'sentiment_coefficients.png')
    
    # 4. Performance Comparison
    plot_performance_comparison_publication(metrics_all, figures_dir / 'performance_comparison.png')
    
    # === GENERATE TABLES ===
    print_section("STEP 4: Generating Tables")
    
    # Summary table
    summary_df = create_summary_table(metrics_all, dm_results, tables_dir / 'forecast_summary.csv')
    
    print("\n   Summary Table:")
    print(summary_df.to_string(index=False))
    
    # === FINAL SUMMARY ===
    print_section("📋 OUTPUT FILES")
    
    print("\n   Figures:")
    print(f"      • forecast_vs_actual_1d.png")
    print(f"      • forecast_vs_actual_5d.png")
    print(f"      • forecast_vs_actual_22d.png")
    print(f"      • pca_loadings_heatmap.png")
    print(f"      • sentiment_coefficients.png")
    print(f"      • performance_comparison.png")
    
    print("\n   Tables:")
    print(f"      • forecast_summary.csv")
    
    print("\n   ✅ Methodology: Rolling PCA (no look-ahead bias)")
    print("\n   ⚠️ IMPORTANT NOTES:")
    print("      - DM tests use horizon-adjusted Newey-West variance")
    print("      - PCA loadings shown are full-sample (descriptive only)")
    print("      - For publication, run 05_audit_and_fix.py for:")
    print("        • Rolling-averaged PCA loadings (mean ± std)")
    print("        • Canonical DM test results")
    print("        • LIMITATIONS.md with all methodological caveats")
    
    print("\n" + "="*70)
    print("   PUBLICATION OUTPUTS COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
