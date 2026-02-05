#!/usr/bin/env python3
"""
Benchmark Models and Evaluation Metrics (Requirements-Compliant)
=================================================================

Implements MANDATORY benchmarks from Requirements.md:
1. Random walk IV
2. AR(1), AR(5)
3. HAR-IV (Heterogeneous Autoregressive)
4. VIX-only regression

And IV-appropriate evaluation metrics:
- RMSE
- QLIKE (quasi-likelihood loss for volatility)
- Diebold-Mariano tests

FATAL ERROR: Claiming predictability without beating HAR-IV
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

from .utils import logger


# =============================================================================
# BENCHMARK MODELS
# =============================================================================

class RandomWalkBenchmark:
    """
    Random Walk benchmark: forecast = last observation.
    
    This is the simplest benchmark - if you can't beat this,
    there's no predictability.
    """
    
    def __init__(self):
        self.name = "Random Walk"
    
    def forecast(self, series: pd.Series, horizon: int = 1) -> pd.Series:
        """
        Random walk forecast: y_{t+h} = y_t
        
        Args:
            series: Time series to forecast
            horizon: Forecast horizon
            
        Returns:
            Forecasts aligned with actuals
        """
        return series.shift(horizon)


class ARBenchmark:
    """
    Autoregressive benchmark.
    
    AR(1): y_t = α + β * y_{t-1}
    AR(5): y_t = α + Σ β_i * y_{t-i}
    """
    
    def __init__(self, lags: int = 1):
        self.lags = lags
        self.name = f"AR({lags})"
        self.model = None
        self.coefficients = None
        
    def fit(self, series: pd.Series) -> 'ARBenchmark':
        """Fit AR model."""
        # Create lagged features
        X = pd.DataFrame()
        for i in range(1, self.lags + 1):
            X[f'lag{i}'] = series.shift(i)
        
        X = X.dropna()
        y = series.loc[X.index]
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.coefficients = dict(zip(X.columns, self.model.coef_))
        
        return self
    
    def forecast(self, series: pd.Series, horizon: int = 1) -> pd.Series:
        """Generate AR forecasts."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        # Create lagged features
        X = pd.DataFrame()
        for i in range(1, self.lags + 1):
            X[f'lag{i}'] = series.shift(i + horizon - 1)  # Account for horizon
        
        X = X.dropna()
        
        # Predict
        forecasts = pd.Series(index=series.index, dtype=float)
        forecasts.loc[X.index] = self.model.predict(X)
        
        return forecasts


class HARIVBenchmark:
    """
    Heterogeneous Autoregressive model for Implied Volatility.
    
    HAR-IV uses multi-horizon components:
    - Daily IV (lag 1)
    - Weekly IV (average of lags 1-5)
    - Monthly IV (average of lags 1-22)
    
    This is the KEY BENCHMARK - must beat this to claim predictability.
    """
    
    def __init__(self):
        self.name = "HAR-IV"
        self.model = None
        self.coefficients = None
    
    def fit(self, series: pd.Series) -> 'HARIVBenchmark':
        """Fit HAR-IV model."""
        # Create HAR components
        X = pd.DataFrame()
        
        # Daily: lag 1
        X['iv_daily'] = series.shift(1)
        
        # Weekly: average of lags 1-5
        X['iv_weekly'] = series.shift(1).rolling(5).mean()
        
        # Monthly: average of lags 1-22
        X['iv_monthly'] = series.shift(1).rolling(22).mean()
        
        X = X.dropna()
        y = series.loc[X.index]
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.coefficients = dict(zip(X.columns, self.model.coef_))
        
        logger.info(f"   HAR-IV coefficients:")
        logger.info(f"      Daily:   {self.coefficients['iv_daily']:.4f}")
        logger.info(f"      Weekly:  {self.coefficients['iv_weekly']:.4f}")
        logger.info(f"      Monthly: {self.coefficients['iv_monthly']:.4f}")
        
        return self
    
    def forecast(self, series: pd.Series, horizon: int = 1) -> pd.Series:
        """Generate HAR-IV forecasts."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        # Create HAR components (shifted by horizon)
        X = pd.DataFrame()
        X['iv_daily'] = series.shift(horizon)
        X['iv_weekly'] = series.shift(horizon).rolling(5).mean()
        X['iv_monthly'] = series.shift(horizon).rolling(22).mean()
        
        X = X.dropna()
        
        # Predict
        forecasts = pd.Series(index=series.index, dtype=float)
        forecasts.loc[X.index] = self.model.predict(X)
        
        return forecasts


class VIXOnlyBenchmark:
    """
    VIX-only regression benchmark.
    
    Uses only VIX to predict other IV measures.
    """
    
    def __init__(self):
        self.name = "VIX-Only"
        self.model = None
        
    def fit(self, vix: pd.Series, target: pd.Series) -> 'VIXOnlyBenchmark':
        """Fit VIX-only model."""
        X = vix.shift(1).to_frame('vix_lag1')
        X = X.dropna()
        y = target.loc[X.index]
        
        valid = y.notna()
        X = X[valid]
        y = y[valid]
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        return self
    
    def forecast(self, vix: pd.Series, horizon: int = 1) -> pd.Series:
        """Generate VIX-only forecasts."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        X = vix.shift(horizon).to_frame('vix_lag1')
        X = X.dropna()
        
        forecasts = pd.Series(index=vix.index, dtype=float)
        forecasts.loc[X.index] = self.model.predict(X)
        
        return forecasts


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    if valid.sum() == 0:
        return np.nan
    return np.sqrt(mean_squared_error(actual[valid], predicted[valid]))


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    if valid.sum() == 0:
        return np.nan
    return mean_absolute_error(actual[valid], predicted[valid])


def calculate_qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate QLIKE (Quasi-Likelihood) loss.
    
    This is the preferred metric for volatility forecasting.
    QLIKE = mean(actual/predicted - log(actual/predicted) - 1)
    
    Lower is better. Penalizes under-prediction more than over-prediction.
    """
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[valid]
    predicted = predicted[valid]
    
    if len(actual) == 0:
        return np.nan
    
    # Ensure positive values
    actual = np.maximum(actual, 1e-10)
    predicted = np.maximum(predicted, 1e-10)
    
    ratio = actual / predicted
    qlike = np.mean(ratio - np.log(ratio) - 1)
    
    return qlike


def diebold_mariano_test(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    loss: str = 'mse',
    horizon: int = 1
) -> Dict:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Tests H0: forecasts have equal accuracy
    
    Args:
        actual: Actual values
        forecast1: First forecast (e.g., your model)
        forecast2: Second forecast (e.g., benchmark)
        loss: 'mse' or 'mae'
        horizon: Forecast horizon (for HAC standard errors)
        
    Returns:
        Dict with test statistic, p-value, and conclusion
    """
    valid = ~(np.isnan(actual) | np.isnan(forecast1) | np.isnan(forecast2))
    
    if valid.sum() < 10:
        return {'statistic': np.nan, 'p_value': np.nan, 'conclusion': 'Insufficient data'}
    
    actual = actual[valid]
    forecast1 = forecast1[valid]
    forecast2 = forecast2[valid]
    
    # Calculate loss differentials
    if loss == 'mse':
        e1 = (actual - forecast1) ** 2
        e2 = (actual - forecast2) ** 2
    else:  # mae
        e1 = np.abs(actual - forecast1)
        e2 = np.abs(actual - forecast2)
    
    d = e1 - e2  # Loss differential
    
    # DM statistic with HAC variance
    n = len(d)
    d_mean = np.mean(d)
    
    # Newey-West variance estimate (for autocorrelated errors)
    gamma0 = np.var(d)
    gamma = []
    for k in range(1, min(horizon + 1, n - 1)):
        gamma.append(np.cov(d[k:], d[:-k])[0, 1])
    
    # HAC variance
    variance = gamma0 + 2 * sum([(1 - k/(horizon+1)) * g for k, g in enumerate(gamma, 1)])
    variance = max(variance, 1e-10)  # Ensure positive
    
    dm_stat = d_mean / np.sqrt(variance / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    # Conclusion
    if p_value < 0.05:
        if dm_stat < 0:
            conclusion = "Model 1 significantly better"
        else:
            conclusion = "Model 2 significantly better"
    else:
        conclusion = "No significant difference"
    
    return {
        'statistic': dm_stat,
        'p_value': p_value,
        'mean_loss_diff': d_mean,
        'conclusion': conclusion
    }


# =============================================================================
# BENCHMARK COMPARISON
# =============================================================================

def run_benchmark_comparison(
    iv_series: pd.Series,
    model_forecasts: pd.Series,
    vix_series: Optional[pd.Series] = None,
    train_end_idx: int = None
) -> pd.DataFrame:
    """
    Compare model forecasts against all mandatory benchmarks.
    
    Per Requirements.md (MANDATORY):
    - Random walk IV
    - AR(1), AR(5)
    - HAR-IV
    - VIX-only regression
    
    Args:
        iv_series: Actual IV series
        model_forecasts: Your model's forecasts
        vix_series: VIX series (for VIX-only benchmark)
        train_end_idx: Index separating train/test
        
    Returns:
        DataFrame with benchmark comparison
    """
    logger.info("="*60)
    logger.info("BENCHMARK COMPARISON")
    logger.info("="*60)
    
    if train_end_idx is None:
        train_end_idx = len(iv_series) * 3 // 4
    
    train = iv_series.iloc[:train_end_idx]
    test = iv_series.iloc[train_end_idx:]
    
    logger.info(f"📊 Train: {len(train)} observations")
    logger.info(f"📊 Test: {len(test)} observations")
    
    results = []
    
    # 1. Random Walk
    logger.info("\n   1. Random Walk Benchmark...")
    rw = RandomWalkBenchmark()
    rw_forecasts = rw.forecast(iv_series, horizon=1)
    
    # 2. AR(1)
    logger.info("   2. AR(1) Benchmark...")
    ar1 = ARBenchmark(lags=1)
    ar1.fit(train)
    ar1_forecasts = ar1.forecast(iv_series, horizon=1)
    
    # 3. AR(5)
    logger.info("   3. AR(5) Benchmark...")
    ar5 = ARBenchmark(lags=5)
    ar5.fit(train)
    ar5_forecasts = ar5.forecast(iv_series, horizon=1)
    
    # 4. HAR-IV (KEY BENCHMARK)
    logger.info("   4. HAR-IV Benchmark (KEY)...")
    har = HARIVBenchmark()
    har.fit(train)
    har_forecasts = har.forecast(iv_series, horizon=1)
    
    # 5. VIX-only (if VIX provided)
    vix_forecasts = None
    if vix_series is not None:
        logger.info("   5. VIX-Only Benchmark...")
        vix_bench = VIXOnlyBenchmark()
        vix_bench.fit(vix_series.iloc[:train_end_idx], train)
        vix_forecasts = vix_bench.forecast(vix_series, horizon=1)
    
    # Evaluate on test set
    actual = test.values
    model_test = model_forecasts.iloc[train_end_idx:].values
    
    benchmarks = {
        'Model': model_test,
        'Random Walk': rw_forecasts.iloc[train_end_idx:].values,
        'AR(1)': ar1_forecasts.iloc[train_end_idx:].values,
        'AR(5)': ar5_forecasts.iloc[train_end_idx:].values,
        'HAR-IV': har_forecasts.iloc[train_end_idx:].values,
    }
    
    if vix_forecasts is not None:
        benchmarks['VIX-Only'] = vix_forecasts.iloc[train_end_idx:].values
    
    logger.info("\n" + "-"*60)
    logger.info("RESULTS (Out-of-Sample)")
    logger.info("-"*60)
    
    for name, forecasts in benchmarks.items():
        rmse = calculate_rmse(actual, forecasts)
        mae = calculate_mae(actual, forecasts)
        qlike = calculate_qlike(actual, forecasts)
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'QLIKE': qlike
        })
        
        logger.info(f"   {name:<12}: RMSE={rmse:.4f}, MAE={mae:.4f}, QLIKE={qlike:.4f}")
    
    # Diebold-Mariano tests vs HAR-IV (the key benchmark)
    logger.info("\n" + "-"*60)
    logger.info("DIEBOLD-MARIANO TESTS vs HAR-IV")
    logger.info("-"*60)
    
    har_test = har_forecasts.iloc[train_end_idx:].values
    
    dm_result = diebold_mariano_test(actual, model_test, har_test)
    
    logger.info(f"   Model vs HAR-IV:")
    logger.info(f"      DM Statistic: {dm_result['statistic']:.3f}")
    logger.info(f"      p-value: {dm_result['p_value']:.4f}")
    logger.info(f"      Conclusion: {dm_result['conclusion']}")
    
    if dm_result['p_value'] < 0.05 and dm_result['statistic'] < 0:
        logger.info("\n   ✅ MODEL BEATS HAR-IV (statistically significant)")
    elif dm_result['p_value'] >= 0.05:
        logger.info("\n   ⚠️  MODEL DOES NOT SIGNIFICANTLY BEAT HAR-IV")
        logger.warning("   Per requirements: 'Claiming predictability without beating HAR-IV' is a FATAL ERROR")
    else:
        logger.info("\n   ❌ MODEL UNDERPERFORMS HAR-IV")
    
    results_df = pd.DataFrame(results)
    results_df['Beats_HARIV'] = results_df['RMSE'] < results_df[results_df['Model'] == 'HAR-IV']['RMSE'].values[0]
    
    return results_df


def comprehensive_evaluation(
    actual: pd.Series,
    model_forecasts: pd.Series,
    model_name: str = "PCA-VARX"
) -> Dict:
    """
    Comprehensive evaluation of model forecasts.
    
    Returns all IV-appropriate metrics per Requirements.md:
    - RMSE
    - QLIKE
    - Diebold-Mariano test results
    
    Args:
        actual: Actual IV values
        model_forecasts: Model forecasts
        model_name: Name of the model
        
    Returns:
        Dict with all metrics
    """
    actual_arr = actual.values
    forecast_arr = model_forecasts.values
    
    metrics = {
        'model': model_name,
        'rmse': calculate_rmse(actual_arr, forecast_arr),
        'mae': calculate_mae(actual_arr, forecast_arr),
        'qlike': calculate_qlike(actual_arr, forecast_arr),
        'n_observations': (~np.isnan(actual_arr) & ~np.isnan(forecast_arr)).sum()
    }
    
    # Direction accuracy
    actual_diff = np.diff(actual_arr)
    forecast_diff = np.diff(forecast_arr)
    valid = ~(np.isnan(actual_diff) | np.isnan(forecast_diff))
    
    if valid.sum() > 0:
        direction_accuracy = np.mean(np.sign(actual_diff[valid]) == np.sign(forecast_diff[valid]))
        metrics['direction_accuracy'] = direction_accuracy
    
    return metrics
