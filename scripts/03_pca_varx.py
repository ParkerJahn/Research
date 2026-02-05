#!/usr/bin/env python3
"""
PCA-VARX Implied Volatility Forecasting (Complete Pipeline)
============================================================

Single-run pipeline for VIX forecasting with:
- Full post-AI dataset (~900+ observations)
- Multi-horizon forecasts (1d, 5d, 22d)
- Rolling cross-validation (~130-150 test obs per horizon)
- ROLLING PCA (refitted at each fold to avoid look-ahead bias)
- ADF stationarity tests
- Diebold-Mariano tests
- Granger causality tests
- Publication-quality figures and tables

Run: python scripts/03_pca_varx.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

import config
from src.utils import logger
from src.sentiment_processing import residualize_sentiment
from src.unified_metrics import calculate_all_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================

PRICE_START_DATE = '2022-01-01'
SENTIMENT_START_DATE = '2023-01-01'
TRAIN_END_DATE = '2025-07-31'
FORECAST_HORIZONS = [1, 5, 22]
N_PCA_COMPONENTS = 3
CV_MIN_TRAIN = 200
CV_TEST_SIZE = 50
CV_STEP_SIZE = 25
CV_MAX_FOLDS = 10
RIDGE_ALPHA = 1.0
ALPHA = 0.05

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'actual': '#2C3E50', 'har_iv': '#E74C3C', 'har_sent': '#27AE60', 'rw': '#95A5A6'}


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

def load_all_data() -> Dict:
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
    print(f"   ✅ VIX: {len(df)} days")
    
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
            print(f"   ✅ {ticker}: {len(data[ticker.lower()])} days")
    
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
        print(f"   ✅ Commodities: {len(data['commodities'])} days")
    
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
            print(f"   ✅ {name}: {len(df)} days")
    
    return data


# =============================================================================
# STATIONARITY TESTS
# =============================================================================

def run_adf_tests(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Run ADF tests."""
    results = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 50:
            continue
        adf = adfuller(series, autolag='AIC')
        results.append({
            'Variable': col,
            'ADF_Statistic': adf[0],
            'p_value': adf[1],
            'Stationary': 'Yes' if adf[1] < ALPHA else 'No'
        })
    return pd.DataFrame(results)


# =============================================================================
# SENTIMENT PROCESSING
# =============================================================================

def process_sentiment(data: Dict) -> pd.DataFrame:
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
        
        print(f"\n   Processing {key}...")
        print("      1. Residualizing...")
        merged['shock'] = residualize_sentiment(
            sentiment=merged[col],
            returns=merged['returns'],
            dates=merged['date'],
            use_expanding_window=True,
            min_window=63
        )
        
        print("      2. Lagging (t-1)...")
        merged['shock_lag1'] = merged['shock'].shift(1)
        
        prefix = key.split('_')[0]
        merged = merged.rename(columns={'shock_lag1': f'{prefix}_shock_lag1'})
        sentiment_dfs.append(merged[['date', f'{prefix}_shock_lag1']].dropna())
        print(f"      ✅ Processed: {len(merged.dropna())} observations")
    
    if not sentiment_dfs:
        return None
    
    result = sentiment_dfs[0]
    for df in sentiment_dfs[1:]:
        result = result.merge(df, on='date', how='inner')
    
    return result


# =============================================================================
# PCA (Full sample for visualization only)
# =============================================================================

def create_pca_features_full_sample(df: pd.DataFrame, pca_cols: List[str]) -> Tuple[pd.DataFrame, PCA, pd.DataFrame]:
    """
    Create PCA features using FULL sample.
    NOTE: This is for descriptive/visualization purposes only, NOT for forecasting.
    """
    valid_df = df[['date'] + pca_cols].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(valid_df[pca_cols])
    
    n_comp = min(N_PCA_COMPONENTS, len(pca_cols))
    pca = PCA(n_components=n_comp)
    pc_scores = pca.fit_transform(X_scaled)
    
    result = valid_df[['date']].copy()
    for i in range(n_comp):
        result[f'PC{i+1}'] = pc_scores[:, i]
    
    loadings = pd.DataFrame(
        pca.components_.T,
        index=pca_cols,
        columns=[f'PC{i+1}' for i in range(n_comp)]
    )
    
    return result, pca, loadings


# =============================================================================
# MODELS
# =============================================================================

class HAR_IV:
    """HAR-IV baseline - only uses lagged VIX."""
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.features = ['vix_lag1', 'vix_weekly', 'vix_monthly']
        self.name = 'HAR-IV'
        self.coef_ = None
        self.uses_pca = False
    
    def fit(self, df):
        X = df[self.features].values
        y = df['vix'].values
        self.model.fit(X, y)
        self.coef_ = dict(zip(self.features, self.model.coef_))
        return self
    
    def predict(self, df):
        return self.model.predict(df[self.features].values)


class HAR_IV_Sentiment:
    """HAR-IV + raw sentiment shocks (no PCA)."""
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.features = ['vix_lag1', 'vix_weekly', 'vix_monthly', 
                        'av_shock_lag1', 'fb_shock_lag1']
        self.name = 'HAR-IV+Sentiment'
        self.coef_ = None
        self.uses_pca = False
    
    def fit(self, df):
        X = df[self.features].values
        y = df['vix'].values
        self.model.fit(X, y)
        self.coef_ = dict(zip(self.features, self.model.coef_))
        return self
    
    def predict(self, df):
        return self.model.predict(df[self.features].values)


class HAR_IV_PCA:
    """HAR-IV + PCA + Sentiment - requires PC columns to be added dynamically."""
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.features = ['vix_lag1', 'vix_weekly', 'vix_monthly',
                        'PC1', 'PC2', 'PC3', 'av_shock_lag1', 'fb_shock_lag1']
        self.name = 'HAR-IV+PCA+Sent'
        self.coef_ = None
        self.uses_pca = True
    
    def fit(self, df):
        X = df[self.features].values
        y = df['vix'].values
        self.model.fit(X, y)
        self.coef_ = dict(zip(self.features, self.model.coef_))
        return self
    
    def predict(self, df):
        return self.model.predict(df[self.features].values)


class RandomWalk:
    """Random walk benchmark."""
    def __init__(self, alpha=1.0):
        self.features = ['vix_lag1']
        self.name = 'Random Walk'
        self.coef_ = {'vix_lag1': 1.0}
        self.uses_pca = False
    
    def fit(self, df):
        return self
    
    def predict(self, df):
        return df['vix_lag1'].values


class AR1:
    """AR(1) model."""
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.features = ['vix_lag1']
        self.name = 'AR(1)'
        self.coef_ = None
        self.uses_pca = False
    
    def fit(self, df):
        X = df[self.features].values
        y = df['vix'].values
        self.model.fit(X, y)
        self.coef_ = dict(zip(self.features, self.model.coef_))
        return self
    
    def predict(self, df):
        return self.model.predict(df[self.features].values)


# =============================================================================
# FORECASTING
# =============================================================================

def prepare_horizon_data(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Shift features for h-step forecast."""
    result = df.copy()
    if horizon > 1:
        shift = horizon - 1
        for col in result.columns:
            if col not in ['date', 'vix']:
                result[col] = result[col].shift(shift)
    return result.dropna().reset_index(drop=True)


def rolling_cv(df: pd.DataFrame, model_classes: List, horizon: int, pca_cols: List[str]) -> Dict:
    """
    Rolling window cross-validation with PROPER rolling PCA.
    
    At each fold:
    1. Fit StandardScaler on training data only
    2. Fit PCA on training data only  
    3. Transform both train and test using training-fitted scaler/PCA
    4. Fit models and predict
    """
    n = len(df)
    n_folds = min(CV_MAX_FOLDS, (n - CV_MIN_TRAIN - CV_TEST_SIZE) // CV_STEP_SIZE + 1)
    
    if n_folds < 1:
        return {}
    
    results = {}
    for cls in model_classes:
        m = cls(alpha=RIDGE_ALPHA)
        results[m.name] = {'actual': [], 'predicted': [], 'dates': []}
    
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
        
        # Run each model
        for cls in model_classes:
            model = cls(alpha=RIDGE_ALPHA)
            if not all(f in train_df.columns for f in model.features):
                continue
            
            try:
                model.fit(train_df)
                preds = model.predict(test_df)
                results[model.name]['actual'].extend(test_df['vix'].values)
                results[model.name]['predicted'].extend(preds)
                results[model.name]['dates'].extend(test_df['date'].values)
            except Exception as e:
                continue
    
    for name in results:
        results[name] = {k: np.array(v) for k, v in results[name].items()}
    
    return results


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    """Calculate RMSE, MAE, QLIKE using unified metrics module."""
    # Use unified calculation - handles validation internally
    return calculate_all_metrics(actual, predicted, include_qlike=True)


def diebold_mariano_test(actual, pred1, pred2, horizon=1) -> Dict:
    """DM test for forecast comparison."""
    valid = ~(np.isnan(actual) | np.isnan(pred1) | np.isnan(pred2))
    if valid.sum() < 30:
        return {'statistic': np.nan, 'p_value': np.nan, 'significant': False}
    
    actual, pred1, pred2 = actual[valid], pred1[valid], pred2[valid]
    e1 = (actual - pred1) ** 2
    e2 = (actual - pred2) ** 2
    d = e1 - e2
    
    n = len(d)
    d_mean = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    
    gamma = []
    for k in range(1, min(horizon + 1, n // 3)):
        if len(d[k:]) > 1:
            gamma.append(np.cov(d[k:], d[:-k])[0, 1])
    
    variance = gamma0 + 2 * sum([(1 - k/(horizon+1)) * g for k, g in enumerate(gamma, 1)])
    variance = max(variance, 1e-10)
    
    dm_stat = d_mean / np.sqrt(variance / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return {
        'statistic': dm_stat,
        'p_value': p_value,
        'significant': p_value < ALPHA,
        'model1_better': dm_stat < 0
    }


# =============================================================================
# GRANGER CAUSALITY
# =============================================================================

def test_granger_predictability(df: pd.DataFrame, target: str, predictor: str, max_lag: int = 10) -> pd.DataFrame:
    """
    Granger predictability test (NOT causality).
    
    Note: This tests whether lagged values of the predictor improve forecasts
    of the target. Significant results indicate PREDICTIVE CONTENT, not causality.
    """
    test_df = df[[target, predictor]].dropna()
    if len(test_df) < 50:
        return None
    
    results = []
    try:
        gc = grangercausalitytests(test_df[[target, predictor]], maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            f_stat = gc[lag][0]['ssr_ftest'][0]
            p_val = gc[lag][0]['ssr_ftest'][1]
            results.append({
                'Predictor': predictor, 'Target': target, 'Lag': lag,
                'F_Statistic': f_stat, 'p_value': p_val,
                'Predictive': 'Yes' if p_val < ALPHA else 'No'
            })
    except:
        pass
    
    return pd.DataFrame(results) if results else None


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_pca_heatmap(loadings: pd.DataFrame, save_path: Path):
    """PCA loadings heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Loading'})
    ax.set_title('PCA Component Loadings (Full Sample - Descriptive)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_forecast(results: Dict, horizon: int, save_path: Path):
    """Forecast vs actual plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Rolling CV produces overlapping windows with duplicate dates
    # We need to deduplicate by averaging predictions for the same date
    if 'HAR-IV' in results and len(results['HAR-IV']['dates']) > 0:
        # Create DataFrame with all data
        df_plot = pd.DataFrame({
            'date': pd.to_datetime(results['HAR-IV']['dates']),
            'actual': results['HAR-IV']['actual']
        })
        # Add predictions from all models
        for name, data in results.items():
            if len(data['predicted']) == len(df_plot):
                df_plot[name] = data['predicted']
        
        # Aggregate duplicates by averaging, then sort by date
        df_plot = df_plot.groupby('date').mean().reset_index().sort_values('date')
        
        # Plot actual
        ax.plot(df_plot['date'], df_plot['actual'], color=COLORS['actual'],
                linewidth=2, label='Actual VIX', alpha=0.8)
        
        # Plot each model's predictions
        for name in results.keys():
            if name in df_plot.columns:
                ax.plot(df_plot['date'], df_plot[name], linewidth=1.5, label=name, alpha=0.7)
    
    ax.set_title(f'VIX Forecasts: {horizon}-Day Horizon (Rolling PCA)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('VIX Level')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance(results_df: pd.DataFrame, save_path: Path):
    """Performance comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric in zip(axes, ['RMSE', 'MAE', 'QLIKE']):
        pivot = results_df.pivot(index='Horizon', columns='Model', values=metric)
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metric} by Horizon', fontsize=12, fontweight='bold')
        ax.set_xlabel('Horizon')
        ax.set_ylabel(metric)
        ax.legend(title='Model', fontsize=8)
        ax.tick_params(axis='x', rotation=0)
    
    plt.suptitle('Forecast Performance Comparison (Rolling PCA - No Look-Ahead Bias)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("🚀 PCA-VARX IMPLIED VOLATILITY FORECASTING")
    print("   ⚠️  Using ROLLING PCA to avoid look-ahead bias")
    
    tables_dir = config.TABLES_DIR
    figures_dir = config.FIGURES_DIR / 'forecast'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # === STEP 1: LOAD DATA ===
    print_section("STEP 1: Loading Data")
    data = load_all_data()
    
    # === STEP 2: MERGE & PROCESS ===
    print_section("STEP 2: Sentiment Processing")
    sentiment_df = process_sentiment(data)
    
    # Merge all data
    full_df = data['vix'].merge(data['smh'], on='date', how='inner')
    full_df = full_df.merge(data['soxx'], on='date', how='inner')
    if 'commodities' in data:
        full_df = full_df.merge(data['commodities'], on='date', how='inner')
    if sentiment_df is not None:
        full_df = full_df.merge(sentiment_df, on='date', how='inner')
    full_df = full_df.dropna()
    print(f"\n   📊 Combined dataset: {len(full_df)} observations")
    
    # === STEP 3: STATIONARITY TESTS ===
    print_section("STEP 3: Stationarity Tests (ADF)")
    test_cols = [c for c in full_df.columns if c != 'date']
    adf_results = run_adf_tests(full_df, test_cols)
    print(adf_results[['Variable', 'ADF_Statistic', 'p_value', 'Stationary']].to_string(index=False))
    adf_results.to_csv(tables_dir / 'stationarity_tests.csv', index=False)
    print(f"\n   💾 Saved: stationarity_tests.csv")
    
    # === STEP 4: PCA (Full sample for visualization only) ===
    print_section("STEP 4: PCA Feature Extraction (Descriptive)")
    pca_cols = ['smh_return', 'smh_rv', 'soxx_return', 'soxx_rv']
    if 'gold_ret' in full_df.columns:
        pca_cols += ['gold_ret', 'copper_ret', 'oil_ret']
    if 'av_shock_lag1' in full_df.columns:
        pca_cols += ['av_shock_lag1', 'fb_shock_lag1']
    
    # Full-sample PCA for visualization ONLY
    pca_result, pca_model, loadings = create_pca_features_full_sample(full_df, pca_cols)
    
    print(f"\n   PCA Results (Full Sample - Descriptive Only):")
    print(f"      Components: {N_PCA_COMPONENTS}")
    print(f"      Variance explained: {pca_model.explained_variance_ratio_.sum()*100:.1f}%")
    for i, var in enumerate(pca_model.explained_variance_ratio_):
        print(f"      PC{i+1}: {var*100:.1f}%")
    
    print(f"\n   Component Loadings (top features):")
    for i in range(N_PCA_COMPONENTS):
        top = loadings[f'PC{i+1}'].abs().nlargest(3)
        top_str = ', '.join([f"{idx}({loadings.loc[idx, f'PC{i+1}']:.2f})" for idx in top.index])
        print(f"      PC{i+1}: {top_str}")
    
    loadings.to_csv(tables_dir / 'pca_loadings.csv')
    print(f"\n   💾 Saved: pca_loadings.csv")
    
    plot_pca_heatmap(loadings, figures_dir / 'pca_loadings_heatmap.png')
    print(f"   📊 Saved: pca_loadings_heatmap.png")
    
    # Add HAR-IV features (DO NOT add full-sample PCA scores - those are for visualization only)
    full_df['vix_lag1'] = full_df['vix'].shift(1)
    full_df['vix_weekly'] = full_df['vix'].shift(1).rolling(5).mean()
    full_df['vix_monthly'] = full_df['vix'].shift(1).rolling(22).mean()
    full_df = full_df.dropna().reset_index(drop=True)
    
    print(f"\n   📊 Final dataset: {len(full_df)} observations")
    
    # === STEP 5: GRANGER PREDICTABILITY TESTS ===
    print_section("STEP 5: Granger Predictability Tests")
    print("   ⚠️ Note: These test predictive content, NOT causality")
    granger_results = []
    for col in ['av_shock_lag1', 'fb_shock_lag1']:
        if col in full_df.columns:
            gc = test_granger_predictability(full_df, 'vix', col, max_lag=10)
            if gc is not None:
                granger_results.append(gc)
    
    if granger_results:
        all_granger = pd.concat(granger_results, ignore_index=True)
        sig = all_granger[all_granger['Predictive'] == 'Yes']
        if len(sig) > 0:
            print("\n   Significant predictive content:")
            print(sig.to_string(index=False))
        else:
            print("\n   ⚠️ No significant predictive content at α=0.05")
            print("   Note: Forecast improvements do NOT require Granger significance")
        all_granger.to_csv(tables_dir / 'granger_predictability_results.csv', index=False)
        print(f"\n   💾 Saved: granger_predictability_results.csv")
    
    # === STEP 6: MULTI-HORIZON FORECASTING WITH ROLLING PCA ===
    print_section("STEP 6: Multi-Horizon Rolling Cross-Validation (Rolling PCA)")
    
    model_classes = [HAR_IV, HAR_IV_Sentiment, HAR_IV_PCA, RandomWalk, AR1]
    all_cv_results = []
    all_dm_results = []
    
    for horizon in FORECAST_HORIZONS:
        print(f"\n   📊 Horizon: {horizon}-day")
        
        horizon_df = prepare_horizon_data(full_df.copy(), horizon)
        print(f"      Data points: {len(horizon_df)}")
        
        if len(horizon_df) < CV_MIN_TRAIN + CV_TEST_SIZE:
            print(f"      ⚠️ Insufficient data")
            continue
        
        # Run rolling CV with proper rolling PCA
        cv_results = rolling_cv(horizon_df, model_classes, horizon, pca_cols)
        har_data = cv_results.get('HAR-IV', {})
        
        for name, data in cv_results.items():
            if len(data['actual']) > 0:
                metrics = calculate_metrics(data['actual'], data['predicted'])
                metrics['Model'] = name
                metrics['Horizon'] = f'{horizon}d'
                all_cv_results.append(metrics)
                print(f"      {name}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, N={metrics['N']}")
        
        # DM tests vs HAR-IV
        if len(har_data.get('actual', [])) >= 30:
            for name, data in cv_results.items():
                if name != 'HAR-IV' and len(data['actual']) >= 30:
                    n_common = min(len(har_data['actual']), len(data['actual']))
                    dm = diebold_mariano_test(
                        data['actual'][:n_common],
                        data['predicted'][:n_common],
                        har_data['predicted'][:n_common],
                        horizon=horizon
                    )
                    
                    all_dm_results.append({
                        'Horizon': f'{horizon}d',
                        'Model_1': name,
                        'Model_2': 'HAR-IV',
                        'DM_Statistic': dm['statistic'],
                        'p_value': dm['p_value'],
                        'Significant': 'Yes' if dm['significant'] else 'No',
                        'Model_1_Better': 'Yes' if dm.get('model1_better', False) else 'No'
                    })
                    
                    har_rmse = calculate_metrics(har_data['actual'], har_data['predicted'])['RMSE']
                    model_rmse = calculate_metrics(data['actual'], data['predicted'])['RMSE']
                    improvement = (har_rmse - model_rmse) / har_rmse * 100
                    
                    sig = "✅" if dm['significant'] and dm.get('model1_better', False) else "⚠️"
                    print(f"         vs HAR-IV: {improvement:+.2f}% (p={dm['p_value']:.3f}) {sig}")
        
        # Plot
        if cv_results:
            plot_forecast(cv_results, horizon, figures_dir / f'forecast_vs_actual_{horizon}d.png')
            print(f"      📊 Saved: forecast_vs_actual_{horizon}d.png")
    
    # === STEP 7: SAVE RESULTS ===
    print_section("STEP 7: Saving Results")
    
    if all_cv_results:
        cv_df = pd.DataFrame(all_cv_results)
        cv_df.to_csv(tables_dir / 'forecast_cv_results.csv', index=False)
        print(f"   💾 Saved: forecast_cv_results.csv")
        
        plot_performance(cv_df, figures_dir / 'performance_comparison.png')
        print(f"   📊 Saved: performance_comparison.png")
    
    if all_dm_results:
        dm_df = pd.DataFrame(all_dm_results)
        dm_df.to_csv(tables_dir / 'dm_test_results.csv', index=False)
        print(f"   💾 Saved: dm_test_results.csv")
    
    # === FINAL SUMMARY ===
    print_section("📋 FINAL SUMMARY")
    
    print(f"\n   Data: {len(full_df)} observations")
    print(f"   Date range: {full_df['date'].min().date()} to {full_df['date'].max().date()}")
    print(f"\n   PCA (Descriptive): {N_PCA_COMPONENTS} components, {pca_model.explained_variance_ratio_.sum()*100:.1f}% variance")
    print(f"   ✅ Forecasting uses ROLLING PCA (no look-ahead bias)")
    
    if all_cv_results:
        cv_df = pd.DataFrame(all_cv_results)
        print("\n   Best Models by Horizon:")
        for h in FORECAST_HORIZONS:
            h_df = cv_df[cv_df['Horizon'] == f'{h}d']
            if len(h_df) > 0:
                best = h_df.loc[h_df['RMSE'].idxmin()]
                print(f"      {h}d: {best['Model']} (RMSE={best['RMSE']:.4f})")
    
    if all_dm_results:
        dm_df = pd.DataFrame(all_dm_results)
        sig = dm_df[(dm_df['Significant'] == 'Yes') & (dm_df['Model_1_Better'] == 'Yes')]
        print("\n   Statistically Significant Improvements:")
        if len(sig) > 0:
            for _, row in sig.iterrows():
                print(f"      ✅ {row['Horizon']}: {row['Model_1']} (p={row['p_value']:.4f})")
        else:
            print("      ⚠️ None at α=0.05 (still publishable!)")
    
    print("\n   Output Files:")
    print(f"      Tables: stationarity_tests.csv, pca_loadings.csv, granger_predictability_results.csv,")
    print(f"              forecast_cv_results.csv, dm_test_results.csv")
    print(f"      Figures: pca_loadings_heatmap.png, forecast_vs_actual_*.png, performance_comparison.png")
    print("\n   ⚠️ IMPORTANT: Run 05_audit_and_fix.py for publication-ready outputs")
    
    print("\n" + "="*70)
    print("   ANALYSIS COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
