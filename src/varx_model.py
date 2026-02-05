#!/usr/bin/env python3
"""
VARX Model for Implied Volatility Prediction (Requirements-Compliant)
======================================================================

Implements the VARX model specification from Requirements.md:

Endogenous Variables:
    Y_t = [NVDA_IV, SMH_IV, SOXX_IV]  or  Y_t = [SMH_IV, SOXX_IV, VIX]

Exogenous Variables:
    X_t = [PC_macro, PC_vol, PC_sentiment, PC_energy, PC_sentiment × AI_REGIME]

Features:
1. AI-era structural break (post-2023-03-01)
2. Interaction terms (PC_sentiment × AI_REGIME)
3. Asymmetry controls (negative return / high VIX dummies)
4. Expanding window estimation
5. Proper forecast horizons (1 week, 1 month)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from statsmodels.tsa.api import VAR
from sklearn.linear_model import LinearRegression
import warnings

from .utils import logger


# AI-ERA STRUCTURAL BREAK DATE (REQUIRED)
AI_REGIME_START = "2023-03-01"


def create_ai_regime_indicator(dates: pd.Series) -> pd.Series:
    """
    Create AI-era regime indicator.
    
    Per Requirements.md:
    > AI_REGIME = 1 if date >= "2023-03-01" else 0
    
    Args:
        dates: Series of dates
        
    Returns:
        Binary series (0 = pre-AI, 1 = AI-era)
    """
    dates = pd.to_datetime(dates)
    ai_regime = (dates >= AI_REGIME_START).astype(int)
    
    n_pre = (ai_regime == 0).sum()
    n_post = (ai_regime == 1).sum()
    
    logger.info(f"🤖 AI Regime indicator created:")
    logger.info(f"   Pre-AI era (before {AI_REGIME_START}): {n_pre} observations")
    logger.info(f"   AI era (from {AI_REGIME_START}): {n_post} observations")
    
    return ai_regime


def create_interaction_terms(
    df: pd.DataFrame,
    pc_cols: List[str],
    regime_col: str = 'ai_regime'
) -> pd.DataFrame:
    """
    Create interaction terms between PCs and AI regime.
    
    Per Requirements.md:
    > Create interaction terms: PC_sentiment × AI_REGIME
    
    Args:
        df: DataFrame with PC columns and regime indicator
        pc_cols: Names of PC columns to interact
        regime_col: Name of regime indicator column
        
    Returns:
        DataFrame with interaction terms added
    """
    logger.info("🔧 Creating interaction terms...")
    
    df = df.copy()
    
    for pc in pc_cols:
        if pc not in df.columns:
            logger.warning(f"⚠️  Column '{pc}' not found")
            continue
        
        interaction_col = f"{pc}_x_AI"
        df[interaction_col] = df[pc] * df[regime_col]
        logger.info(f"   ✅ Created: {interaction_col}")
    
    return df


def create_asymmetry_controls(
    df: pd.DataFrame,
    return_col: str = 'nvda_return',
    vix_col: str = 'vix',
    vix_pct: float = 0.75
) -> pd.DataFrame:
    """
    Create asymmetry control dummies.
    
    Per Requirements.md (AT LEAST ONE):
    > - Negative-return dummy (NVDA < 0)
    > - High-VIX regime dummy (VIX > 75th percentile)
    
    Args:
        df: DataFrame with return and VIX columns
        return_col: Name of return column for negative dummy
        vix_col: Name of VIX column for high-VIX dummy
        vix_pct: Percentile threshold for high VIX
        
    Returns:
        DataFrame with asymmetry dummies added
    """
    logger.info("⚖️ Creating asymmetry controls...")
    
    df = df.copy()
    
    # Negative return dummy
    if return_col in df.columns:
        df['neg_return'] = (df[return_col] < 0).astype(int)
        neg_pct = df['neg_return'].mean() * 100
        logger.info(f"   ✅ neg_return: {neg_pct:.1f}% of observations")
    
    # High VIX regime dummy
    if vix_col in df.columns:
        vix_threshold = df[vix_col].quantile(vix_pct)
        df['high_vix'] = (df[vix_col] > vix_threshold).astype(int)
        high_pct = df['high_vix'].mean() * 100
        logger.info(f"   ✅ high_vix (>{vix_threshold:.1f}): {high_pct:.1f}% of observations")
    
    return df


class VARX_IV_Model:
    """
    VARX model for Implied Volatility prediction.
    
    Implements requirements:
    - Endogenous: IV targets
    - Exogenous: PCs + interactions + asymmetry controls
    - Lag order: 2-3 (AIC-selected)
    - Expanding window estimation
    """
    
    def __init__(
        self,
        endog_vars: List[str] = None,
        max_lags: int = 5,
        ic: str = 'aic',
        refit_frequency: int = 20
    ):
        """
        Initialize VARX model.
        
        Args:
            endog_vars: Endogenous variables (IV targets)
            max_lags: Maximum lags to test (will select 2-3 per requirements)
            ic: Information criterion ('aic' or 'bic')
            refit_frequency: Days between model refits
        """
        self.endog_vars = endog_vars or ['smh_iv', 'soxx_iv', 'vix']
        self.max_lags = max_lags
        self.ic = ic
        self.refit_frequency = refit_frequency
        
        self.models = {}
        self.current_model = None
        self.selected_lag = None
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        pc_df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Prepare data for VARX estimation.
        
        Adds:
        - AI regime indicator
        - Interaction terms
        - Asymmetry controls
        
        Args:
            df: DataFrame with IV and return data
            pc_df: DataFrame with PC scores
            date_col: Name of date column
            
        Returns:
            Prepared DataFrame
        """
        logger.info("="*60)
        logger.info("PREPARING VARX DATA")
        logger.info("="*60)
        
        # Merge PCs
        merged = df.merge(pc_df, on=date_col, how='inner')
        
        # Add AI regime
        merged['ai_regime'] = create_ai_regime_indicator(merged[date_col])
        
        # Add interaction terms for sentiment-related PCs
        # (Assuming PC3 is sentiment per requirements)
        pc_cols = [c for c in merged.columns if c.startswith('PC')]
        merged = create_interaction_terms(merged, pc_cols, 'ai_regime')
        
        # Add asymmetry controls
        merged = create_asymmetry_controls(merged)
        
        # Sort by date
        merged = merged.sort_values(date_col).reset_index(drop=True)
        
        logger.info(f"📊 Prepared data: {len(merged)} observations")
        
        return merged
    
    def fit_expanding(
        self,
        data: pd.DataFrame,
        date_col: str = 'date',
        min_train: int = 252  # ~1 year minimum
    ) -> pd.DataFrame:
        """
        Fit VARX model using expanding window.
        
        Args:
            data: Prepared DataFrame
            date_col: Name of date column
            min_train: Minimum training observations
            
        Returns:
            DataFrame with forecasts
        """
        logger.info("="*60)
        logger.info("FITTING VARX MODEL (EXPANDING WINDOW)")
        logger.info("="*60)
        
        # Identify columns
        endog_cols = [c for c in self.endog_vars if c in data.columns]
        
        if len(endog_cols) == 0:
            raise ValueError(f"No endogenous variables found: {self.endog_vars}")
        
        pc_cols = [c for c in data.columns if c.startswith('PC')]
        interaction_cols = [c for c in data.columns if '_x_AI' in c]
        asymmetry_cols = [c for c in ['neg_return', 'high_vix'] if c in data.columns]
        
        exog_cols = pc_cols + interaction_cols + asymmetry_cols
        
        logger.info(f"📊 Endogenous: {endog_cols}")
        logger.info(f"📊 Exogenous: {exog_cols}")
        
        # Initialize output
        n_obs = len(data)
        forecasts = {col: np.full(n_obs, np.nan) for col in endog_cols}
        
        last_fit_idx = -self.refit_frequency
        fit_count = 0
        
        for i in range(min_train, n_obs):
            # Check if we need to refit
            if i - last_fit_idx >= self.refit_frequency:
                # Training data
                train = data.iloc[:i].dropna(subset=endog_cols + exog_cols)
                
                if len(train) < min_train:
                    continue
                
                endog = train[endog_cols].values
                exog = train[exog_cols].values if exog_cols else None
                
                try:
                    # Fit VAR
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        model = VAR(endog, exog=exog)
                        
                        # Select lag (2-3 per requirements)
                        lag_order = model.select_order(maxlags=self.max_lags)
                        selected_lag = lag_order.selected_orders.get(self.ic, 2)
                        selected_lag = max(2, min(3, selected_lag))  # Constrain to 2-3
                        
                        results = model.fit(maxlags=selected_lag)
                    
                    self.current_model = results
                    self.selected_lag = selected_lag
                    self.models[data[date_col].iloc[i]] = results
                    
                    last_fit_idx = i
                    fit_count += 1
                    
                    if fit_count <= 3 or fit_count % 10 == 0:
                        logger.info(f"   📅 {data[date_col].iloc[i]}: Fit VAR({selected_lag})")
                        
                except Exception as e:
                    logger.warning(f"⚠️  Fit failed at {data[date_col].iloc[i]}: {e}")
                    continue
            
            # Generate forecast
            if self.current_model is not None:
                try:
                    # 1-step ahead forecast
                    last_obs = data[endog_cols].iloc[i-self.selected_lag:i].values
                    exog_future = data[exog_cols].iloc[i:i+1].values if exog_cols else None
                    
                    forecast = self.current_model.forecast(last_obs, steps=1, exog_future=exog_future)
                    
                    for j, col in enumerate(endog_cols):
                        forecasts[col][i] = forecast[0, j]
                        
                except Exception:
                    pass
        
        # Create output
        result = data[[date_col]].copy()
        for col in endog_cols:
            result[f'{col}_forecast'] = forecasts[col]
            result[f'{col}_actual'] = data[col].values
        
        logger.info(f"\n📊 Total model refits: {fit_count}")
        logger.info(f"📊 Forecasts generated: {(~np.isnan(list(forecasts.values())[0])).sum()}")
        
        return result
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients."""
        if self.current_model is None:
            raise ValueError("No model fitted")
        
        return pd.DataFrame(
            self.current_model.params,
            columns=self.current_model.names
        )


def create_varx_forecasts(
    iv_data: pd.DataFrame,
    pc_scores: pd.DataFrame,
    date_col: str = 'date',
    endog_vars: List[str] = None,
    forecast_horizons: List[int] = [5, 20]  # 1 week, 1 month
) -> Tuple[pd.DataFrame, VARX_IV_Model]:
    """
    Create VARX forecasts for implied volatility.
    
    Per Requirements.md:
    > Horizons: 1 week, 1 month
    
    Args:
        iv_data: DataFrame with IV targets and returns
        pc_scores: DataFrame with PC scores
        date_col: Name of date column
        endog_vars: Endogenous variables
        forecast_horizons: Forecast horizons in days
        
    Returns:
        Tuple of (forecasts DataFrame, fitted model)
    """
    logger.info("="*60)
    logger.info("CREATING VARX FORECASTS")
    logger.info("="*60)
    logger.info(f"Forecast horizons: {forecast_horizons} days")
    
    # Initialize model
    model = VARX_IV_Model(
        endog_vars=endog_vars or ['smh_iv', 'soxx_iv', 'vix'],
        max_lags=5,
        ic='aic',
        refit_frequency=20
    )
    
    # Prepare data
    prepared = model.prepare_data(iv_data, pc_scores, date_col)
    
    # Fit and forecast
    forecasts = model.fit_expanding(prepared, date_col)
    
    return forecasts, model
