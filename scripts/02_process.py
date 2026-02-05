#!/usr/bin/env python3
"""
================================================================================
STEP 2: FEATURE PROCESSING
================================================================================

PURPOSE:
    Process raw data into features for PCA-VARX analysis.

WHAT THIS SCRIPT DOES:
    1. Run FinBERT sentiment analysis on news headlines
    2. Aggregate sentiment to daily level
    3. Create processed sentiment files (AV + FinBERT)

INPUT:
    data/raw/news/sentiment_raw_data.csv (headlines with Alpha Vantage sentiment)

OUTPUT:
    data/processed/daily_sentiment_av.csv  (Alpha Vantage sentiment)
    data/processed/daily_sentiment_fb.csv  (FinBERT sentiment)

NEXT STEP:
    Run: python scripts/03_pca_varx.py
================================================================================
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

import config
from src.utils import logger, load_dataframe, save_dataframe


def run_finbert_analysis(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run FinBERT sentiment analysis on news headlines.
    
    Args:
        news_df: DataFrame with 'headline' column
        
    Returns:
        DataFrame with finbert_score column added
    """
    print("\n" + "─"*70)
    print("RUNNING FINBERT SENTIMENT ANALYSIS")
    print("─"*70)
    
    if 'headline' not in news_df.columns:
        logger.error("❌ No 'headline' column found in news data")
        return news_df
    
    try:
        from src.sentiment_analysis import SentimentAnalyzer
        
        print(f"   📊 Headlines to process: {len(news_df)}")
        print("   ⏰ This may take several minutes...\n")
        
        analyzer = SentimentAnalyzer()
        news_df = analyzer.analyze_dataframe(news_df, text_column='headline')
        
        # Rename to finbert_score for consistency
        news_df = news_df.rename(columns={'sentiment_score': 'finbert_score'})
        
        print(f"\n   ✅ FinBERT analysis complete")
        print(f"   📊 Score range: [{news_df['finbert_score'].min():.3f}, {news_df['finbert_score'].max():.3f}]")
        
        return news_df
        
    except ImportError as e:
        logger.error(f"❌ FinBERT dependencies not installed: {e}")
        logger.info("   Run: pip install transformers torch")
        return news_df
    except Exception as e:
        logger.error(f"❌ FinBERT analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return news_df


def aggregate_sentiment(news_df: pd.DataFrame, score_col: str, source_name: str) -> pd.DataFrame:
    """
    Aggregate headline-level sentiment to daily level.
    
    Args:
        news_df: DataFrame with sentiment scores
        score_col: Name of sentiment score column
        source_name: Name for output columns (e.g., 'av' or 'fb')
        
    Returns:
        Daily sentiment DataFrame
    """
    if score_col not in news_df.columns:
        return None
    
    # Aggregate by date and ticker
    daily = news_df.groupby(['date', 'ticker']).agg({
        score_col: ['mean', 'count']
    }).reset_index()
    
    daily.columns = ['date', 'ticker', f'{source_name}_sentiment_mean', 'headline_count']
    
    # Create sector-wide sentiment (average across all tickers)
    sector = daily.groupby('date').agg({
        f'{source_name}_sentiment_mean': 'mean',
        'headline_count': 'sum'
    }).reset_index()
    
    sector = sector.rename(columns={
        f'{source_name}_sentiment_mean': f'sector_{source_name}_sent',
        'headline_count': 'sector_headline_count'
    })
    
    # Pivot ticker sentiment to wide format
    ticker_wide = daily.pivot(
        index='date',
        columns='ticker',
        values=f'{source_name}_sentiment_mean'
    ).reset_index()
    
    ticker_wide.columns = ['date'] + [f'{col.lower()}_{source_name}_sent' for col in ticker_wide.columns[1:]]
    
    # Merge with sector sentiment
    result = ticker_wide.merge(sector[['date', f'sector_{source_name}_sent']], on='date', how='left')
    
    return result


def main():
    """Main processing pipeline."""
    
    print("\n" + "="*70)
    print("          🔧 STEP 2: FEATURE PROCESSING")
    print("="*70)
    print(f"\n   Date Range: {config.START_DATE} to {config.END_DATE}")
    
    # -------------------------------------------------------------------------
    # Load Raw News Data
    # -------------------------------------------------------------------------
    print("\n" + "─"*70)
    print("LOADING RAW NEWS DATA")
    print("─"*70)
    
    # Prefer sentiment_comparison.csv (has more historical data with FinBERT already)
    comparison_path = config.RAW_NEWS_DIR / 'sentiment_comparison.csv'
    raw_path = config.RAW_NEWS_DIR / 'sentiment_raw_data.csv'
    
    if comparison_path.exists():
        news_path = comparison_path
        print(f"   📁 Using: sentiment_comparison.csv (historical data with FinBERT)")
    elif raw_path.exists():
        news_path = raw_path
        print(f"   📁 Using: sentiment_raw_data.csv")
    else:
        logger.error(f"❌ No news file found")
        logger.info("   Run 01_collect.py first to collect news data")
        return False
    
    news_df = load_dataframe(news_path, parse_dates=['date'])
    news_df['date'] = pd.to_datetime(news_df['date']).dt.normalize()
    
    print(f"   ✅ Loaded {len(news_df)} headlines")
    print(f"   📅 Date range: {news_df['date'].min().date()} to {news_df['date'].max().date()}")
    
    # Check for existing sentiment scores
    has_av = 'sentiment_score' in news_df.columns
    has_finbert = 'finbert_score' in news_df.columns
    
    print(f"\n   Existing sentiment scores:")
    print(f"      Alpha Vantage: {'✅' if has_av else '❌'}")
    print(f"      FinBERT: {'✅' if has_finbert else '❌'}")
    
    # -------------------------------------------------------------------------
    # Run FinBERT Analysis (if not already done)
    # -------------------------------------------------------------------------
    if not has_finbert:
        news_df = run_finbert_analysis(news_df)
        has_finbert = 'finbert_score' in news_df.columns
        
        # Save updated news file with FinBERT scores
        if has_finbert:
            save_dataframe(news_df, news_path)
            logger.info(f"   💾 Updated {news_path} with FinBERT scores")
    else:
        print(f"\n   ✅ FinBERT scores already present - skipping analysis")
    
    # -------------------------------------------------------------------------
    # Aggregate Sentiment to Daily Level
    # -------------------------------------------------------------------------
    print("\n" + "─"*70)
    print("AGGREGATING SENTIMENT TO DAILY LEVEL")
    print("─"*70)
    
    # Alpha Vantage sentiment
    if has_av:
        av_daily = aggregate_sentiment(news_df, 'sentiment_score', 'av')
        if av_daily is not None:
            av_output = config.PROCESSED_DATA_DIR / 'daily_sentiment_av.csv'
            save_dataframe(av_daily, av_output)
            print(f"   ✅ Alpha Vantage: {len(av_daily)} days → {av_output}")
    
    # FinBERT sentiment
    if has_finbert:
        fb_daily = aggregate_sentiment(news_df, 'finbert_score', 'fb')
        if fb_daily is not None:
            fb_output = config.PROCESSED_DATA_DIR / 'daily_sentiment_fb.csv'
            save_dataframe(fb_daily, fb_output)
            print(f"   ✅ FinBERT: {len(fb_daily)} days → {fb_output}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("          ✅ PROCESSING COMPLETE")
    print("="*70)
    
    print(f"\n   Output files:")
    if has_av:
        print(f"      • data/processed/daily_sentiment_av.csv")
    if has_finbert:
        print(f"      • data/processed/daily_sentiment_fb.csv")
    
    print(f"\n   📋 Next Step:")
    print(f"      Run: python scripts/03_pca_varx.py")
    print()
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
