#!/usr/bin/env python3
"""
================================================================================
STEP 1: DATA COLLECTION
================================================================================

PURPOSE:
    Collect all raw data for PCA-VARX implied volatility prediction.

DATA COLLECTED (only what's used):
    
    1. VIX, VVIX - Target variables (implied volatility)
    2. SMH, SOXX - Semiconductor ETFs (returns, volatility proxies)
    3. Commodities - For PCA factor creation
    4. News - Semiconductor headlines for sentiment

OUTPUT FILES:
    data/raw/prices/^VIX_history.csv, ^VVIX_history.csv
    data/raw/prices/SMH_history.csv, SOXX_history.csv
    data/raw/commodities/*.csv
    data/raw/news/sentiment_raw_data.csv

DATE RANGE: 2022-01-01 to 2026-01-31 (per Requirements.md)

NEXT STEP:
    Run: python scripts/02_process.py
================================================================================
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

import config
from src.utils import logger, save_dataframe

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.error("❌ yfinance not installed. Run: pip install yfinance")


# ============================================================================
# DATA TO COLLECT (only what's used in 03_pca_varx.py)
# ============================================================================

# Target variables
IMPLIED_VOL_TICKERS = ['^VIX', '^VVIX']

# Semiconductor ETFs (for returns and volatility)
SEMICONDUCTOR_ETFS = ['SMH', 'SOXX']

# Commodities for PCA factor
COMMODITY_TICKERS = {
    'CL=F': 'WTI Crude Oil',
    'NG=F': 'Natural Gas',
    'BZ=F': 'Brent Crude Oil',
    'GC=F': 'Gold',
    'SI=F': 'Silver',
    'PA=F': 'Palladium',
    'PL=F': 'Platinum',
    'HG=F': 'Copper',
    'URA': 'Uranium ETF',
    'REMX': 'Rare Earth ETF',
}

# Semiconductor companies for news sentiment
SEMICONDUCTOR_TICKERS = ['NVDA', 'AMD', 'INTC', 'TSM', 'MU']

# Date range (from config)
START_DATE = config.START_DATE
END_DATE = config.END_DATE


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num: int, description: str):
    print(f"\n{'─' * 70}")
    print(f"STEP {step_num}: {description}")
    print(f"{'─' * 70}")


def fetch_ticker(ticker: str, start: str, end: str) -> Optional[dict]:
    """Fetch historical data from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start, end=end, auto_adjust=True)
        
        if df.empty:
            logger.warning(f"   ⚠️  {ticker}: No data returned")
            return None
        
        df = df.reset_index()
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        df['ticker'] = ticker
        
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'date'})
        
        return {
            'data': df,
            'rows': len(df),
            'start': df['date'].min(),
            'end': df['date'].max(),
            'price_range': (df['close'].min(), df['close'].max())
        }
        
    except Exception as e:
        logger.error(f"   ❌ {ticker}: Error - {e}")
        return None


# ============================================================================
# COLLECTION FUNCTIONS
# ============================================================================

def collect_implied_volatility() -> Dict:
    """Collect VIX and VVIX (target variables)."""
    print_step(1, "Collecting VIX & VVIX (Target Variables)")
    print("\n   These are what we predict - implied volatility indices.\n")
    
    results = {}
    
    for ticker in IMPLIED_VOL_TICKERS:
        result = fetch_ticker(ticker, START_DATE, END_DATE)
        
        if result:
            results[ticker] = result
            print(f"   ✅ {ticker}: {result['rows']} days, range [{result['price_range'][0]:.1f}, {result['price_range'][1]:.1f}]")
            
            filepath = config.RAW_PRICES_DIR / f"{ticker}_history.csv"
            save_dataframe(result['data'], filepath)
    
    return results


def collect_semiconductor_etfs() -> Dict:
    """Collect SMH and SOXX (for returns and volatility)."""
    print_step(2, "Collecting Semiconductor ETFs (SMH, SOXX)")
    print("\n   Used for sentiment residualization and as volatility proxies.\n")
    
    results = {}
    
    for ticker in SEMICONDUCTOR_ETFS:
        result = fetch_ticker(ticker, START_DATE, END_DATE)
        
        if result:
            results[ticker] = result
            print(f"   ✅ {ticker}: {result['rows']} days")
            
            filepath = config.RAW_PRICES_DIR / f"{ticker}_history.csv"
            save_dataframe(result['data'], filepath)
    
    return results


def collect_commodities() -> Dict:
    """Collect commodity prices for PCA factor."""
    print_step(3, "Collecting Commodities (for PCA Factor)")
    print("\n   These create the 'commodity stress' factor via PCA.\n")
    
    results = {}
    
    for ticker, name in COMMODITY_TICKERS.items():
        result = fetch_ticker(ticker, START_DATE, END_DATE)
        
        if result:
            results[ticker] = result
            print(f"   ✅ {name}: {result['rows']} days")
            
            filepath = config.RAW_COMMODITIES_DIR / f"{ticker}_history.csv"
            save_dataframe(result['data'], filepath)
        else:
            print(f"   ⚠️  {name}: Failed")
        
        time.sleep(0.3)  # Rate limiting
    
    return results


def collect_news_sentiment() -> bool:
    """Collect semiconductor news with Alpha Vantage sentiment."""
    print_step(4, "Collecting News Sentiment")
    print(f"\n   Tickers: {', '.join(SEMICONDUCTOR_TICKERS)}")
    print("   Source: Alpha Vantage News & Sentiment API\n")
    
    if not config.ALPHA_VANTAGE_KEY:
        print("   ⚠️  ALPHA_VANTAGE_API_KEY not found in .env")
        print("   ⚠️  Skipping news collection")
        print("\n   💡 To enable: Get free API key from alphavantage.co")
        return False
    
    try:
        from src.data_collection import AlphaVantageNewsCollector
        
        collector = AlphaVantageNewsCollector()
        
        print("   ⏰ This may take a few minutes (API rate limits)...\n")
        
        df = collector.collect_multiple_tickers(
            tickers=SEMICONDUCTOR_TICKERS,
            start_date=START_DATE,
            end_date=END_DATE,
            delay=15.0
        )
        
        if df.empty:
            print("   ⚠️  No news data returned")
            return False
        
        filepath = config.RAW_NEWS_DIR / "sentiment_raw_data.csv"
        save_dataframe(df, filepath)
        
        print(f"   ✅ Collected {len(df)} headlines")
        print(f"   💾 Saved to: {filepath}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ News collection failed: {e}")
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main data collection pipeline."""
    print_header("📥 STEP 1: DATA COLLECTION")
    print(f"\n   Date Range: {START_DATE} to {END_DATE}")
    
    # Ensure directories exist
    config.create_directory_structure()
    
    # Collect all data
    results = {
        'implied_vol': collect_implied_volatility(),
        'semi_etfs': collect_semiconductor_etfs(),
        'commodities': collect_commodities(),
        'news': collect_news_sentiment()
    }
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_header("📋 COLLECTION SUMMARY")
    
    print(f"\n   ✅ VIX/VVIX: {len(results['implied_vol'])}/2")
    print(f"   ✅ Semi ETFs: {len(results['semi_etfs'])}/2")
    print(f"   ✅ Commodities: {len(results['commodities'])}/{len(COMMODITY_TICKERS)}")
    print(f"   {'✅' if results['news'] else '⚠️ '} News: {'Collected' if results['news'] else 'Skipped'}")
    
    print("\n" + "=" * 70)
    print("   ✅ DATA COLLECTION COMPLETE")
    print("=" * 70)
    
    print(f"\n   📋 Next Step:")
    print(f"      Run: python scripts/02_process.py")
    print()
    
    # Check minimum requirements
    if len(results['implied_vol']) == 0:
        print("   ❌ ERROR: No VIX data. Cannot proceed.")
        return False
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
