# config.py
"""
Configuration file for Sentiment-Driven IMPLIED VOLATILITY Prediction Research Project.

This project predicts VIX (S&P 500 implied volatility) and VVIX (volatility of VIX)
using semiconductor news sentiment and commodity price factors.

Key targets:
- VIX: 30-day implied volatility of S&P 500 options
- VVIX: Implied volatility of VIX options (volatility of volatility)

All parameters, file paths, and API configurations defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API Keys
# ============================================================================
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')  # Note: env var is ALPHA_VANTAGE_API_KEY
USER_LOCATION = os.getenv('USER_LOCATION', 'Winter Park, Florida, US')
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))

# Validate critical API keys
if not NEWS_API_KEY:
    print("⚠️  WARNING: NEWS_API_KEY not found in .env file")
    print("    Some data collection features will not work.")

# ============================================================================
# Date Range (per Requirements.md: full sample 2022 → present)
# ============================================================================
START_DATE = '2022-01-01'  # Full sample starts 2022 for PCA stability
END_DATE = '2026-01-31'    # Through January 2026

# Train/Test Split
TRAIN_END_DATE = '2025-06-30'  # 2022 - mid-2025 = training (3.5 years)
TEST_START_DATE = '2025-07-01'  # Mid-2025 onwards = testing (out-of-sample)

# ============================================================================
# Tickers
# ============================================================================
# Semiconductor companies for sentiment analysis
SEMICONDUCTOR_TICKERS: List[str] = ['NVDA', 'AMD', 'INTC', 'TSM', 'MU']

# ETFs for volatility calculation
ETF_TICKERS: List[str] = ['SMH', 'SOXX']

# Commodity futures (used to create PCA "commodity stress" factor)
COMMODITY_TICKERS: List[str] = [
    # Energy
    'CL=F',  # WTI Crude Oil
    'NG=F',  # Natural Gas
    'BZ=F',  # Brent Crude Oil
    # Precious Metals
    'GC=F',  # Gold (safe haven)
    'SI=F',  # Silver
    'PA=F',  # Palladium
    'PL=F',  # Platinum
    # Industrial/Tech-related
    'HG=F',  # Copper (economic bellwether)
    'URA',   # Uranium ETF
    'REMX',  # Rare Earth ETF
]

# Implied Volatility Targets (what we're predicting)
IMPLIED_VOL_TICKERS: List[str] = ['^VIX', '^VVIX']

# All tickers used in analysis
ALL_TICKERS = SEMICONDUCTOR_TICKERS + ETF_TICKERS + COMMODITY_TICKERS + IMPLIED_VOL_TICKERS

# ============================================================================
# Feature Engineering Parameters
# ============================================================================
# Volatility calculation
VOLATILITY_WINDOW = 21  # 21-day rolling window for realized volatility
ANNUALIZATION_FACTOR = 252  # Trading days per year

# Sentiment lags
SENTIMENT_LAGS = [1, 5]  # 1-day and 5-day (1 week) lags

# Missing data threshold
MAX_MISSING_PCT = 0.10  # Maximum 10% missing values allowed

# ============================================================================
# NLP Configuration
# ============================================================================
# FinBERT model
FINBERT_MODEL = 'ProsusAI/finbert'
SENTIMENT_BATCH_SIZE = 32  # Process headlines in batches
MAX_HEADLINE_LENGTH = 512  # Token limit for FinBERT

# ============================================================================
# PCA Configuration (per Requirements.md: 70-85% variance)
# ============================================================================
PCA_VARIANCE_THRESHOLD = 0.80  # Retain components explaining 70-85% variance
PCA_MIN_COMPONENTS = 3
PCA_MAX_COMPONENTS = 5

# AI-Era Structural Break (REQUIRED per Requirements.md)
AI_REGIME_START_DATE = '2023-03-01'  # Post-ChatGPT era

# ============================================================================
# VAR Model Configuration
# ============================================================================
VAR_MAX_LAGS = 10  # Test lags 1-10
VAR_IC_CRITERION = 'aic'  # Information criterion: 'aic' or 'bic'

# Endogenous variables (to be predicted) - IMPLIED VOLATILITY
# VIX = S&P 500 implied volatility (30-day)
# VVIX = VIX of VIX (volatility of volatility)
VAR_ENDOG_VARS = ['vix', 'vvix']

# Granger causality test lags
# Extended range to capture delayed sentiment effects
GRANGER_TEST_LAGS = [1, 2, 3, 5, 7, 10, 15, 20]

# ============================================================================
# Forecasting Configuration
# ============================================================================
FORECAST_HORIZONS = [1, 5, 10]  # 1-day, 1-week, 2-week forecasts
ROLLING_WINDOW_DAYS = 20  # Re-estimate model every 20 days

# ============================================================================
# Backtesting Configuration
# ============================================================================
INITIAL_CAPITAL = 10000  # $10,000 starting capital
TRANSACTION_COST_BPS = 10  # 10 basis points per trade
VOL_THRESHOLD = 0.02  # 2% threshold for signal generation

# Signal thresholds for sensitivity analysis
THRESHOLD_GRID = [0.01, 0.02, 0.03, 0.05]
COST_GRID = [5, 10, 20]  # Transaction costs in bps

# ============================================================================
# File Paths
# ============================================================================
# Base directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'
MODELS_DIR = RESULTS_DIR / 'models'
DOCS_DIR = PROJECT_ROOT / 'docs'

# Raw data subdirectories
RAW_NEWS_DIR = RAW_DATA_DIR / 'news'
RAW_PRICES_DIR = RAW_DATA_DIR / 'prices'
RAW_COMMODITIES_DIR = RAW_DATA_DIR / 'commodities'

# Figure subdirectories
EDA_FIGURES_DIR = FIGURES_DIR / 'eda'
PCA_FIGURES_DIR = FIGURES_DIR / 'pca'
VAR_FIGURES_DIR = FIGURES_DIR / 'var'
FORECAST_FIGURES_DIR = FIGURES_DIR / 'forecast'
BACKTEST_FIGURES_DIR = FIGURES_DIR / 'backtest'

# ============================================================================
# Visualization Configuration
# ============================================================================
# Matplotlib settings
FIGURE_DPI = 100
FIGURE_SIZE = (10, 6)
SAVE_DPI = 300
SAVE_FORMAT = 'png'

# Color scheme
COLOR_PALETTE = 'Set2'

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_FILE = PROJECT_ROOT / 'project.log'
ERROR_LOG_FILE = PROJECT_ROOT / 'error_log.txt'

# ============================================================================
# API Configurations
# ============================================================================
NEWS_API_CONFIG: Dict = {
    'api_key': NEWS_API_KEY,
    'endpoint': 'https://newsapi.org/v2/everything',
    'params': {
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 100
    },
    'rate_limit_delay': 1.0  # Seconds between requests
}

YFINANCE_CONFIG: Dict = {
    'start_date': START_DATE,
    'end_date': END_DATE,
    'interval': '1d',
    'auto_adjust': True,
    'progress': False
}

# ============================================================================
# Validation Thresholds
# ============================================================================
MIN_HEADLINES_PER_TICKER = 10  # Minimum headlines required
MIN_TRADING_DAYS = 400  # Minimum trading days for analysis
MAX_PRICE_MISSING_PCT = 0.05  # Maximum 5% missing price data

# ============================================================================
# Helper Functions
# ============================================================================
def create_directory_structure():
    """Create all necessary directories for the project."""
    directories = [
        RAW_DATA_DIR,
        RAW_NEWS_DIR,
        RAW_PRICES_DIR,
        RAW_COMMODITIES_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        EDA_FIGURES_DIR,
        PCA_FIGURES_DIR,
        VAR_FIGURES_DIR,
        FORECAST_FIGURES_DIR,
        BACKTEST_FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        DOCS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directory structure created successfully!")
    return directories


def get_file_path(category: str, filename: str) -> Path:
    """
    Get full file path for a given category and filename.
    
    Args:
        category: One of 'raw_news', 'raw_prices', 'raw_commodities', 
                  'processed', 'figures', 'tables', 'models'
        filename: Name of the file
        
    Returns:
        Full Path object
    """
    path_map = {
        'raw_news': RAW_NEWS_DIR,
        'raw_prices': RAW_PRICES_DIR,
        'raw_commodities': RAW_COMMODITIES_DIR,
        'processed': PROCESSED_DATA_DIR,
        'figures_eda': EDA_FIGURES_DIR,
        'figures_pca': PCA_FIGURES_DIR,
        'figures_var': VAR_FIGURES_DIR,
        'figures_forecast': FORECAST_FIGURES_DIR,
        'figures_backtest': BACKTEST_FIGURES_DIR,
        'tables': TABLES_DIR,
        'results_tables': TABLES_DIR,  # Alias for 'tables'
        'models': MODELS_DIR,
        'docs': DOCS_DIR,
    }
    
    if category not in path_map:
        raise ValueError(f"Unknown category: {category}")
    
    return path_map[category] / filename


def validate_config():
    """Validate configuration and print summary."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Training Period: {START_DATE} to {TRAIN_END_DATE}")
    print(f"Testing Period: {TEST_START_DATE} to {END_DATE}")
    print(f"\nTickers:")
    print(f"  Semiconductors: {', '.join(SEMICONDUCTOR_TICKERS)}")
    print(f"  ETFs: {', '.join(ETF_TICKERS)}")
    print(f"  Commodities: {', '.join(COMMODITY_TICKERS)}")
    print(f"\nAPI Keys:")
    print(f"  NewsAPI: {'✅ Loaded' if NEWS_API_KEY else '❌ Missing'}")
    print(f"  Alpha Vantage: {'✅ Loaded' if ALPHA_VANTAGE_KEY else '❌ Missing'}")
    print(f"\nPaths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Results Dir: {RESULTS_DIR}")
    print("="*60 + "\n")


# ============================================================================
# Run validation if executed directly
# ============================================================================
if __name__ == '__main__':
    print("🚀 Volatility Research Project Configuration")
    validate_config()
    print("\nCreating directory structure...")
    create_directory_structure()
    print("\n✅ Configuration validated and directories created!")
    print("\nNext steps:")
    print("1. Copy .env.example to .env and add your API keys")
    print("2. Run: python scripts/01_collect_data.py")
