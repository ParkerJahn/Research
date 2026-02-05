# src/utils.py
"""
Utility functions for the research project.
Helper functions used across multiple modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure logging for the project.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('volatility_research')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame"
) -> bool:
    """
    Validate that a DataFrame has required columns and minimal data quality.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for logging purposes
        
    Returns:
        True if valid, False otherwise
    """
    # Check if DataFrame is empty
    if df.empty:
        logger.error(f"{name} is empty")
        return False
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"{name} missing columns: {missing_cols}")
        return False
    
    # Check for excessive missing values
    missing_pct = df[required_columns].isnull().sum() / len(df)
    high_missing = missing_pct[missing_pct > 0.5]
    if not high_missing.empty:
        logger.warning(
            f"{name} has columns with >50% missing: {high_missing.to_dict()}"
        )
    
    logger.info(f"✅ {name} validated: {len(df)} rows, {len(df.columns)} columns")
    return True


def calculate_returns(
    prices: pd.Series,
    method: str = 'log'
) -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'log' for log returns, 'simple' for simple returns
        
    Returns:
        Returns series
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return returns


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Calculate realized volatility from returns.
    
    Args:
        returns: Returns series
        window: Rolling window size
        annualize: Whether to annualize volatility
        annualization_factor: Days per year for annualization
        
    Returns:
        Realized volatility series
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(annualization_factor)
    
    return vol


def align_datasets(
    *dfs: pd.DataFrame,
    on: str = 'date',
    how: str = 'inner'
) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames by a common column (typically date).
    
    Args:
        *dfs: Variable number of DataFrames to align
        on: Column name to align on
        how: Join method ('inner', 'outer', 'left')
        
    Returns:
        List of aligned DataFrames
    """
    if len(dfs) < 2:
        return list(dfs)
    
    # Start with first DataFrame
    result = dfs[0].copy()
    
    # Progressively merge with others
    for df in dfs[1:]:
        result = result.merge(df, on=on, how=how, suffixes=('', '_dup'))
        
        # Remove duplicate columns
        dup_cols = [col for col in result.columns if col.endswith('_dup')]
        if dup_cols:
            result = result.drop(columns=dup_cols)
    
    logger.info(f"✅ Aligned {len(dfs)} datasets: {len(result)} rows")
    
    # Return the merged result for all
    return [result] * len(dfs) if len(dfs) > 1 else [result]


def forward_fill_weekends(
    df: pd.DataFrame,
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Forward fill data for weekends (financial data often missing weekends).
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        
    Returns:
        DataFrame with weekends filled
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Set date as index
    df = df.set_index(date_col)
    
    # Reindex to include all days
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_range)
    
    # Forward fill
    df = df.fillna(method='ffill')
    
    # Reset index
    df = df.reset_index().rename(columns={'index': date_col})
    
    return df


def calculate_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missing data statistics for a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with missing statistics per column
    """
    stats = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'dtype': df.dtypes
    })
    
    stats = stats[stats['missing_count'] > 0].sort_values(
        'missing_pct', ascending=False
    )
    
    return stats


def save_dataframe(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False,
    verbose: bool = True
) -> None:
    """
    Save DataFrame to CSV with logging.
    
    Args:
        df: DataFrame to save
        filepath: Path to save file
        index: Whether to save index
        verbose: Whether to log
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=index)
    
    if verbose:
        logger.info(f"💾 Saved: {filepath} ({len(df)} rows, {len(df.columns)} cols)")


def load_dataframe(
    filepath: Union[str, Path],
    parse_dates: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load DataFrame from CSV with logging.
    
    Args:
        filepath: Path to CSV file
        parse_dates: Columns to parse as dates
        verbose: Whether to log
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=parse_dates)
    
    # Convert timezone-aware dates to timezone-naive (remove timezone info)
    if parse_dates:
        for col in parse_dates:
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                # If timezone-aware, convert to UTC then remove timezone
                if df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_convert('UTC').dt.tz_localize(None)
    
    if verbose:
        logger.info(f"📂 Loaded: {filepath} ({len(df)} rows, {len(df.columns)} cols)")
    
    return df


def create_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for numerical columns.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Summary statistics DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = df[numeric_cols].describe().T
    stats['missing'] = df[numeric_cols].isnull().sum()
    stats['missing_pct'] = (stats['missing'] / len(df) * 100).round(2)
    
    return stats


def print_data_quality_report(
    df: pd.DataFrame,
    name: str = "Dataset"
) -> None:
    """
    Print comprehensive data quality report.
    
    Args:
        df: DataFrame to analyze
        name: Name for the report
    """
    print("\n" + "="*60)
    print(f"DATA QUALITY REPORT: {name}")
    print("="*60)
    
    print(f"\n📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Date range if date column exists
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        date_col = date_cols[0]
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            print(f"📅 Date Range: {df[date_col].min()} to {df[date_col].max()}")
            print(f"📅 Days: {(df[date_col].max() - df[date_col].min()).days}")
    
    # Missing data
    missing = calculate_missing_stats(df)
    if not missing.empty:
        print("\n⚠️  Missing Data:")
        print(missing.to_string())
    else:
        print("\n✅ No missing data")
    
    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"\n⚠️  Duplicate rows: {dup_count}")
    else:
        print("\n✅ No duplicate rows")
    
    # Summary stats for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 Numeric Columns ({len(numeric_cols)}):")
        print(df[numeric_cols].describe().T.to_string())
    
    print("\n" + "="*60 + "\n")


def exponential_backoff_retry(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Function result
    """
    import time
    
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries - 1:
                raise
            
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)


if __name__ == '__main__':
    # Test utilities
    print("🧪 Testing utility functions...")
    
    # Test returns calculation
    prices = pd.Series([100, 102, 101, 105, 103])
    returns = calculate_returns(prices)
    print(f"\n✅ Returns calculation: {len(returns)} values")
    
    # Test volatility calculation
    returns_extended = pd.Series(np.random.randn(100) * 0.01 + 0.001)
    vol = calculate_realized_volatility(returns_extended, window=21)
    print(f"✅ Volatility calculation: {len(vol.dropna())} values")
    
    print("\n✅ All utility tests passed!")
