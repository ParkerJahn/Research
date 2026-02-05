# config_template.py
"""
Configuration Template for Sentiment-Augmented VIX Forecasting

INSTRUCTIONS:
1. Copy this file to 'config.py'
2. Fill in your API keys in the .env file (see .env.example)
3. Adjust parameters as needed for your analysis

DO NOT commit config.py to version control - it's in .gitignore
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API Keys (Set these in your .env file)
# ============================================================================
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))

# Validate critical API keys
if not ALPHA_VANTAGE_KEY:
    print("⚠️  WARNING: ALPHA_VANTAGE_API_KEY not found in .env file")
    print("    Sentiment data collection will not work.")
    print("    Get your free key at: https://www.alphavantage.co/support/#api-key")

# ============================================================================
# Date Range
# ============================================================================
START_DATE = '2022-01-01'  # Full sample starts 2022
END_DATE = '2026-01-31'    # Through January 2026

# ============================================================================
# Tickers
# ============================================================================
# Semiconductor companies for sentiment analysis
SEMICONDUCTOR_TICKERS: List[str] = ['NVDA', 'AMD', 'INTC', 'TSM', 'MU']

# ETFs for volatility calculation
ETF_TICKERS: List[str] = ['SMH', 'SOXX']

# Commodity futures
COMMODITY_TICKERS: List[str] = [
    'CL=F',  # WTI Crude Oil
    'GC=F',  # Gold
    'HG=F',  # Copper
]

# Implied Volatility Targets
IMPLIED_VOL_TICKERS: List[str] = ['^VIX']

# All tickers
ALL_TICKERS = SEMICONDUCTOR_TICKERS + ETF_TICKERS + COMMODITY_TICKERS + IMPLIED_VOL_TICKERS

# ============================================================================
# Model Parameters
# ============================================================================
# HAR-IV model
RIDGE_ALPHA = 1.0

# Cross-validation
CV_MIN_TRAIN = 200  # Minimum training window (days)
CV_TEST_SIZE = 50   # Test window size (days)
CV_STEP_SIZE = 25   # Rolling window step size (days)

# PCA
N_PCA_COMPONENTS = 3

# Forecast horizons
FORECAST_HORIZONS = [1, 5, 22]  # 1-day, 5-day, 22-day (1 month)

# ============================================================================
# Feature Engineering Parameters
# ============================================================================
VOLATILITY_WINDOW = 21  # 21-day rolling window for realized volatility
ANNUALIZATION_FACTOR = 252  # Trading days per year

# Sentiment orthogonalization
SENTIMENT_MIN_WINDOW = 63  # Minimum window for expanding OLS

# ============================================================================
# NLP Configuration
# ============================================================================
FINBERT_MODEL = 'ProsusAI/finbert'
SENTIMENT_BATCH_SIZE = 32
MAX_HEADLINE_LENGTH = 512

# ============================================================================
# File Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'
MODELS_DIR = RESULTS_DIR / 'models'

# Raw data subdirectories
RAW_NEWS_DIR = RAW_DATA_DIR / 'news'
RAW_PRICES_DIR = RAW_DATA_DIR / 'prices'
RAW_COMMODITIES_DIR = RAW_DATA_DIR / 'commodities'

# ============================================================================
# API Configurations
# ============================================================================
ALPHA_VANTAGE_CONFIG: Dict = {
    'api_key': ALPHA_VANTAGE_KEY,
    'rate_limit_delay': 12.0,  # 5 requests/minute = 12 seconds between requests
    'max_retries': 3,
}

YFINANCE_CONFIG: Dict = {
    'start_date': START_DATE,
    'end_date': END_DATE,
    'interval': '1d',
    'auto_adjust': True,
    'progress': False
}

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
        TABLES_DIR,
        MODELS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directory structure created successfully!")
    return directories


def validate_config():
    """Validate configuration and print summary."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"\nTickers:")
    print(f"  Semiconductors: {', '.join(SEMICONDUCTOR_TICKERS)}")
    print(f"  ETFs: {', '.join(ETF_TICKERS)}")
    print(f"  Commodities: {', '.join(COMMODITY_TICKERS)}")
    print(f"\nAPI Keys:")
    print(f"  Alpha Vantage: {'✅ Loaded' if ALPHA_VANTAGE_KEY else '❌ Missing'}")
    print(f"\nPaths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Results Dir: {RESULTS_DIR}")
    print("="*60 + "\n")


if __name__ == '__main__':
    print("🚀 VIX Forecasting Project Configuration")
    validate_config()
    print("\nCreating directory structure...")
    create_directory_structure()
    print("\n✅ Configuration validated and directories created!")
    print("\nNext steps:")
    print("1. Ensure .env file has ALPHA_VANTAGE_API_KEY set")
    print("2. Run: python scripts/01_collect.py")
