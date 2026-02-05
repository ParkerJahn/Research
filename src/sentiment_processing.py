#!/usr/bin/env python3
"""
Sentiment Processing Module (Requirements-Compliant)
=====================================================

Implements the NON-NEGOTIABLE sentiment processing requirements:
1. Residualize sentiment (purge return information)
2. Lag sentiment BEFORE PCA (t-1, t-5)
3. Window-based standardization (z-score inside training window)

These steps are CRITICAL to avoid look-ahead bias and spurious correlations.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Optional, List, Tuple
import warnings

from .utils import logger


def residualize_sentiment(
    sentiment: pd.Series,
    returns: pd.Series,
    dates: pd.Series,
    use_expanding_window: bool = True,
    min_window: int = 63
) -> pd.Series:
    """
    Residualize sentiment by purging return information.
    
    This is NON-NEGOTIABLE per Requirements.md:
    > Sent_t = α + β·Returns_t + ε_t
    > Use ε_t (sentiment shock) only.
    
    Args:
        sentiment: Raw sentiment scores
        returns: Corresponding returns (same dates)
        dates: Date index for alignment
        use_expanding_window: If True, use expanding window regression
        min_window: Minimum observations for regression
        
    Returns:
        Residualized sentiment (sentiment shock)
        
    FATAL ERROR if skipped: Using raw sentiment scores
    """
    logger.info("🔧 Residualizing sentiment (purging return information)...")
    
    # Align data
    df = pd.DataFrame({
        'date': dates,
        'sentiment': sentiment.values,
        'returns': returns.values
    }).dropna()
    
    if len(df) < min_window:
        logger.warning(f"⚠️  Too few observations ({len(df)}) for residualization")
        return sentiment
    
    residuals = np.full(len(df), np.nan)
    
    if use_expanding_window:
        # Expanding window regression to avoid look-ahead
        for i in range(min_window, len(df)):
            # Fit on data up to (but not including) current observation
            X_train = df['returns'].iloc[:i].values.reshape(-1, 1)
            y_train = df['sentiment'].iloc[:i].values
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Get residual for current observation
            X_curr = df['returns'].iloc[i].reshape(-1, 1)
            y_curr = df['sentiment'].iloc[i]
            y_pred = model.predict(X_curr)[0]
            residuals[i] = y_curr - y_pred
    else:
        # Full sample regression (use only for diagnostics)
        X = df['returns'].values.reshape(-1, 1)
        y = df['sentiment'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        residuals = y - model.predict(X)
        
        logger.info(f"   Regression: Sent = {model.intercept_:.4f} + {model.coef_[0]:.4f}*Returns")
        logger.info(f"   R² = {model.score(X, y):.4f} (variance explained by returns)")
    
    # Create output series with original index
    result = pd.Series(residuals, index=df.index, name='sentiment_shock')
    
    # Stats
    valid_count = (~np.isnan(residuals)).sum()
    logger.info(f"   ✅ Residualized {valid_count} observations")
    logger.info(f"   Original sentiment std: {df['sentiment'].std():.4f}")
    logger.info(f"   Shock std: {pd.Series(residuals).std():.4f}")
    
    return result


def lag_sentiment(
    df: pd.DataFrame,
    sentiment_cols: List[str],
    lags: List[int] = [1, 5]
) -> pd.DataFrame:
    """
    Lag sentiment BEFORE using in PCA or models.
    
    This is NON-NEGOTIABLE per Requirements.md:
    > Use Sent_{t-1}, Sent_{t-5}
    > Never use contemporaneous sentiment
    
    Args:
        df: DataFrame with sentiment columns
        sentiment_cols: Names of sentiment columns to lag
        lags: Lag periods (default [1, 5] for 1-day and 1-week)
        
    Returns:
        DataFrame with lagged sentiment columns added
        
    FATAL ERROR if skipped: Look-ahead bias via Sent_t
    """
    logger.info(f"🔧 Lagging sentiment: {lags} periods...")
    
    df = df.copy()
    
    for col in sentiment_cols:
        if col not in df.columns:
            logger.warning(f"⚠️  Column '{col}' not found")
            continue
            
        for lag in lags:
            lagged_col = f"{col}_lag{lag}"
            df[lagged_col] = df[col].shift(lag)
            logger.info(f"   ✅ Created: {lagged_col}")
    
    return df


def window_standardize(
    series: pd.Series,
    window: int = 63,
    min_periods: int = 21
) -> pd.Series:
    """
    Z-score standardization using rolling window.
    
    This is NON-NEGOTIABLE per Requirements.md:
    > Z-score inside each training window
    > Never normalize using full-sample statistics
    
    Args:
        series: Series to standardize
        window: Rolling window size (default 63 = ~3 months)
        min_periods: Minimum periods for valid calculation
        
    Returns:
        Standardized series
        
    FATAL ERROR if skipped: Global normalization
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    standardized = (series - rolling_mean) / rolling_std
    
    return standardized


def process_sentiment_leakage_safe(
    sentiment_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    sentiment_col: str = 'sentiment',
    returns_col: str = 'returns',
    date_col: str = 'date',
    lags: List[int] = [1, 5],
    standardize_window: int = 63
) -> pd.DataFrame:
    """
    Complete leakage-safe sentiment processing pipeline.
    
    Implements all NON-NEGOTIABLE requirements:
    1. Residualize sentiment (purge returns)
    2. Lag sentiment before use
    3. Window-based standardization
    
    Args:
        sentiment_df: DataFrame with sentiment data
        returns_df: DataFrame with return data
        sentiment_col: Name of sentiment column
        returns_col: Name of returns column
        date_col: Name of date column
        lags: Lag periods for sentiment
        standardize_window: Window for z-score standardization
        
    Returns:
        DataFrame with processed sentiment features
    """
    logger.info("="*60)
    logger.info("LEAKAGE-SAFE SENTIMENT PROCESSING")
    logger.info("="*60)
    
    # Merge sentiment with returns
    df = sentiment_df[[date_col, sentiment_col]].merge(
        returns_df[[date_col, returns_col]],
        on=date_col,
        how='inner'
    ).sort_values(date_col).reset_index(drop=True)
    
    logger.info(f"📊 Combined dataset: {len(df)} observations")
    
    # Step 1: Residualize sentiment
    logger.info("\n" + "-"*60)
    logger.info("STEP 1: Residualizing Sentiment")
    logger.info("-"*60)
    
    df['sentiment_shock'] = residualize_sentiment(
        sentiment=df[sentiment_col],
        returns=df[returns_col],
        dates=df[date_col],
        use_expanding_window=True
    )
    
    # Step 2: Lag sentiment shocks
    logger.info("\n" + "-"*60)
    logger.info("STEP 2: Lagging Sentiment Shocks")
    logger.info("-"*60)
    
    df = lag_sentiment(df, ['sentiment_shock'], lags=lags)
    
    # Step 3: Window-based standardization
    logger.info("\n" + "-"*60)
    logger.info("STEP 3: Window-Based Standardization")
    logger.info("-"*60)
    
    lagged_cols = [f'sentiment_shock_lag{lag}' for lag in lags]
    
    for col in lagged_cols:
        z_col = f"{col}_z"
        df[z_col] = window_standardize(df[col], window=standardize_window)
        logger.info(f"   ✅ Created: {z_col}")
    
    # Final check
    logger.info("\n" + "="*60)
    logger.info("SENTIMENT PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Output columns: {[c for c in df.columns if 'shock' in c or '_z' in c]}")
    
    return df


class SentimentProcessor:
    """
    Stateful sentiment processor for production use.
    
    Maintains state for expanding-window operations.
    """
    
    def __init__(self, lags: List[int] = [1, 5], z_window: int = 63):
        self.lags = lags
        self.z_window = z_window
        self.regression_model = None
        self.training_data = []
        
    def update(self, sentiment: float, returns: float, date) -> dict:
        """
        Process a single new observation.
        
        Args:
            sentiment: Raw sentiment score
            returns: Corresponding returns
            date: Date of observation
            
        Returns:
            Dict with processed sentiment values
        """
        # Add to training history
        self.training_data.append({
            'date': date,
            'sentiment': sentiment,
            'returns': returns
        })
        
        if len(self.training_data) < 63:
            return {'sentiment_shock': np.nan}
        
        # Fit regression on all past data
        df = pd.DataFrame(self.training_data[:-1])  # Exclude current
        X = df['returns'].values.reshape(-1, 1)
        y = df['sentiment'].values
        
        self.regression_model = LinearRegression()
        self.regression_model.fit(X, y)
        
        # Get residual for current
        pred = self.regression_model.predict([[returns]])[0]
        shock = sentiment - pred
        
        # Add shock to data for lagging
        self.training_data[-1]['shock'] = shock
        
        # Get lagged values
        result = {'sentiment_shock': shock}
        
        for lag in self.lags:
            if len(self.training_data) > lag:
                lag_val = self.training_data[-lag-1].get('shock', np.nan)
                result[f'shock_lag{lag}'] = lag_val
            else:
                result[f'shock_lag{lag}'] = np.nan
        
        return result
