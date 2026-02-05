# src/feature_engineering.py
"""
Feature engineering module for volatility and returns calculations.
Creates the master feature matrix for modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from src.utils import (
    logger, save_dataframe, load_dataframe,
    calculate_returns, calculate_realized_volatility,
    print_data_quality_report
)


def calculate_price_features(
    price_df: pd.DataFrame,
    ticker: str,
    volatility_window: int = None,
    date_column: str = 'date',
    close_column: str = 'close'
) -> pd.DataFrame:
    """
    Calculate returns and realized volatility from price data.
    
    Args:
        price_df: DataFrame with price data
        ticker: Ticker symbol for column naming
        volatility_window: Window for volatility calculation
        date_column: Name of date column
        close_column: Name of close price column
        
    Returns:
        DataFrame with date, returns, and volatility
    """
    volatility_window = volatility_window or config.VOLATILITY_WINDOW
    
    df = price_df[[date_column, close_column]].copy()
    # Convert to datetime and remove timezone info
    df[date_column] = pd.to_datetime(df[date_column], utc=True).dt.tz_localize(None)
    df = df.sort_values(date_column).reset_index(drop=True)
    
    # Calculate log returns
    returns = calculate_returns(df[close_column], method='log')
    
    # Calculate realized volatility
    volatility = calculate_realized_volatility(
        returns,
        window=volatility_window,
        annualize=True,
        annualization_factor=config.ANNUALIZATION_FACTOR
    )
    
    # Create output DataFrame
    result = pd.DataFrame({
        date_column: df[date_column],
        f'{ticker.lower()}_return': returns,
        f'{ticker.lower()}_vol': volatility,
        f'{ticker.lower()}_price': df[close_column]
    })
    
    return result


def process_all_price_data(
    price_data_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Process all price data to calculate features.
    
    Args:
        price_data_dict: Dictionary mapping ticker to price DataFrame
        
    Returns:
        Dictionary mapping ticker to feature DataFrame
    """
    logger.info("🔧 Processing price data for all tickers...")
    
    processed_data = {}
    
    for ticker, df in price_data_dict.items():
        if df.empty:
            logger.warning(f"⚠️  Skipping empty data for {ticker}")
            continue
        
        # Handle special ticker names (remove symbols)
        clean_ticker = ticker.replace('=F', '').replace('^', '')
        
        features_df = calculate_price_features(df, clean_ticker)
        processed_data[clean_ticker] = features_df
        
        logger.info(
            f"✅ {ticker} → {len(features_df)} days, "
            f"{features_df.isnull().sum().sum()} missing values"
        )
    
    return processed_data


def merge_all_features(
    price_features: Dict[str, pd.DataFrame],
    sentiment_df: Optional[pd.DataFrame] = None,
    date_column: str = 'date'
) -> pd.DataFrame:
    """
    Merge all features into master feature matrix.
    
    Args:
        price_features: Dictionary of price feature DataFrames
        sentiment_df: Optional sentiment DataFrame
        date_column: Name of date column
        
    Returns:
        Master feature DataFrame
    """
    logger.info("🔗 Merging all features into master matrix...")
    
    # Start with first DataFrame
    if not price_features:
        raise ValueError("No price features provided")
    
    # Get all dataframes
    dfs = list(price_features.values())
    
    # Start with first
    master_df = dfs[0].copy()
    
    # Merge with remaining price features
    for df in dfs[1:]:
        master_df = master_df.merge(df, on=date_column, how='outer')
    
    # Merge with sentiment if provided
    if sentiment_df is not None and not sentiment_df.empty:
        logger.info("  📊 Adding sentiment features...")
        
        # FIX: Normalize dates before merge to handle timezone/time differences
        logger.info("  🔧 Normalizing date formats for merge...")
        master_df[date_column] = pd.to_datetime(master_df[date_column]).dt.normalize()
        sentiment_df[date_column] = pd.to_datetime(sentiment_df[date_column]).dt.normalize()
        
        logger.info(f"  📅 Price date range: {master_df[date_column].min()} to {master_df[date_column].max()}")
        logger.info(f"  📅 Sentiment date range: {sentiment_df[date_column].min()} to {sentiment_df[date_column].max()}")
        
        master_df = master_df.merge(sentiment_df, on=date_column, how='left')
        
        # Check merge success
        sentiment_cols = [col for col in sentiment_df.columns if col != date_column]
        non_null_count = master_df[sentiment_cols[0]].notna().sum() if sentiment_cols else 0
        logger.info(f"  ✅ Sentiment merge: {non_null_count}/{len(master_df)} rows have sentiment data ({non_null_count/len(master_df)*100:.1f}%)")
    
    # Sort by date
    master_df = master_df.sort_values(date_column).reset_index(drop=True)
    
    # Forward fill missing values (weekends, holidays)
    logger.info("  🔄 Forward filling missing values...")
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    master_df[numeric_cols] = master_df[numeric_cols].fillna(method='ffill')
    
    logger.info(f"✅ Master feature matrix: {master_df.shape}")
    
    return master_df


def add_lagged_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    drop_na: bool = False
) -> pd.DataFrame:
    """
    Add lagged versions of specified columns.
    
    Args:
        df: DataFrame
        columns: Columns to lag
        lags: List of lag periods
        drop_na: Whether to drop rows with NaN
        
    Returns:
        DataFrame with lagged features added
    """
    logger.info(f"➕ Adding lagged features: {lags} periods for {len(columns)} columns...")
    
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"⚠️  Column '{col}' not found, skipping")
            continue
        
        for lag in lags:
            lagged_col_name = f"{col}_lag{lag}"
            df[lagged_col_name] = df[col].shift(lag)
            logger.info(f"  ✅ Created: {lagged_col_name}")
    
    if drop_na:
        df = df.dropna()
        logger.info(f"  🗑️  Dropped NaN rows: {len(df)} rows remaining")
    
    return df


def calculate_rolling_statistics(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int]
) -> pd.DataFrame:
    """
    Calculate rolling mean and std for specified columns.
    
    Args:
        df: DataFrame
        columns: Columns to calculate statistics for
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling statistics added
    """
    logger.info(f"📊 Calculating rolling statistics: windows={windows}...")
    
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        for window in windows:
            # Rolling mean
            df[f"{col}_ma{window}"] = df[col].rolling(window=window).mean()
            
            # Rolling std
            df[f"{col}_std{window}"] = df[col].rolling(window=window).std()
    
    logger.info(f"✅ Added {len(columns) * len(windows) * 2} rolling features")
    
    return df


def create_target_variables(
    df: pd.DataFrame,
    volatility_cols: List[str],
    horizons: List[int] = None
) -> pd.DataFrame:
    """
    Create forward-looking target variables for forecasting.
    
    Args:
        df: DataFrame with volatility columns
        volatility_cols: Names of volatility columns
        horizons: Forecast horizons (days ahead)
        
    Returns:
        DataFrame with target variables added
    """
    horizons = horizons or config.FORECAST_HORIZONS
    
    logger.info(f"🎯 Creating target variables: horizons={horizons}...")
    
    df = df.copy()
    
    for vol_col in volatility_cols:
        if vol_col not in df.columns:
            logger.warning(f"⚠️  Column '{vol_col}' not found, skipping")
            continue
        
        for horizon in horizons:
            target_col = f"{vol_col}_target_{horizon}d"
            df[target_col] = df[vol_col].shift(-horizon)
            logger.info(f"  ✅ Created: {target_col}")
    
    return df


def engineer_master_features(
    price_data_dict: Dict[str, pd.DataFrame],
    sentiment_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        price_data_dict: Dictionary of price DataFrames
        sentiment_df: Optional sentiment DataFrame
        output_path: Optional path to save master features
        
    Returns:
        Master feature DataFrame
    """
    logger.info("🚀 Starting feature engineering pipeline...")
    
    # Process all price data
    price_features = process_all_price_data(price_data_dict)
    
    # Merge all features
    master_df = merge_all_features(price_features, sentiment_df)
    
    # Add lagged sentiment features (if sentiment exists)
    if sentiment_df is not None and not sentiment_df.empty:
        sentiment_cols = [col for col in master_df.columns if 'sent' in col.lower()]
        if sentiment_cols:
            master_df = add_lagged_features(
                master_df,
                columns=sentiment_cols,
                lags=config.SENTIMENT_LAGS,
                drop_na=False
            )
    
    # Create target variables for key volatility metrics
    key_vol_cols = ['smh_vol', 'soxx_vol', 'vix']
    existing_vol_cols = [col for col in key_vol_cols if col in master_df.columns]
    
    if existing_vol_cols:
        master_df = create_target_variables(master_df, existing_vol_cols)
    
    # Drop rows with all NaN in volatility columns (due to rolling window)
    vol_cols = [col for col in master_df.columns if '_vol' in col and '_lag' not in col]
    if vol_cols:
        master_df = master_df.dropna(subset=vol_cols, how='all')
    
    # Print data quality report
    print_data_quality_report(master_df, "Master Feature Matrix")
    
    # Save if path provided
    if output_path:
        save_dataframe(master_df, output_path)
    
    logger.info("✅ Feature engineering complete")
    
    return master_df


if __name__ == '__main__':
    print("🧪 Testing feature engineering module...")
    
    # Create sample price data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    sample_df = pd.DataFrame({
        'date': dates,
        'close': sample_prices
    })
    
    print("\n" + "="*60)
    print("TEST: Price Feature Calculation")
    print("="*60)
    
    features = calculate_price_features(sample_df, 'TEST')
    print(features.head())
    print(f"\n✅ Generated {len(features.columns)} features")
    
    print("\n✅ Feature engineering module tests complete!")
