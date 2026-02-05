#!/usr/bin/env python3
"""
Fill Sentiment Data Gap Using GDELT + FinBERT
==============================================

This script fills the missing sentiment data (Jan 2025 - Oct 2025) by:
1. Querying GDELT for semiconductor-related news headlines
2. Running FinBERT locally to compute sentiment scores
3. Aggregating to daily sentiment and merging with existing data

GDELT (Global Database of Events, Language, and Tone):
- Free, open dataset of global news
- Historical data available back to 2015+
- API access without authentication

Requirements:
- pip install requests transformers torch pandas
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import config
from src.utils import logger

# Try to import transformers for FinBERT
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    print("⚠️  transformers/torch not installed. Run: pip install transformers torch")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date range to fill (the gap)
GAP_START = '2025-01-01'
GAP_END = '2025-10-11'

# Semiconductor search terms for GDELT
SEARCH_TERMS = [
    'NVIDIA semiconductor',
    'AMD chip',
    'Intel processor',
    'TSMC semiconductor',
    'Micron memory chip',
    'semiconductor industry',
    'chip shortage',
    'AI chip',
    'GPU market',
    'semiconductor stock'
]

# GDELT API configuration
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# FinBERT model
FINBERT_MODEL = "ProsusAI/finbert"


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}".center(70))
    print("=" * 70 + "\n")


def print_section(title: str):
    print("\n" + "─" * 70)
    print(f"  {title}")
    print("─" * 70)


# =============================================================================
# GDELT DATA COLLECTION
# =============================================================================

def query_gdelt(
    query: str,
    start_date: str,
    end_date: str,
    max_records: int = 250
) -> List[Dict]:
    """
    Query GDELT Document API for news articles.
    
    Args:
        query: Search query string
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_records: Maximum records to return
        
    Returns:
        List of article dictionaries
    """
    # Format dates for GDELT (YYYYMMDDHHMMSS)
    start_fmt = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d%H%M%S')
    end_fmt = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d%H%M%S')
    
    params = {
        'query': query,
        'mode': 'artlist',
        'format': 'json',
        'startdatetime': start_fmt,
        'enddatetime': end_fmt,
        'maxrecords': max_records,
        'sort': 'datedesc'
    }
    
    try:
        response = requests.get(GDELT_DOC_API, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get('articles', [])
        
        return articles
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"GDELT query failed for '{query}': {e}")
        return []
    except ValueError:
        logger.warning(f"Invalid JSON response for '{query}'")
        return []


def collect_gdelt_headlines(
    search_terms: List[str],
    start_date: str,
    end_date: str,
    delay: float = 1.0
) -> pd.DataFrame:
    """
    Collect headlines from GDELT for multiple search terms.
    
    Args:
        search_terms: List of search queries
        start_date: Start date
        end_date: End date
        delay: Delay between API calls (seconds)
        
    Returns:
        DataFrame with headlines
    """
    print(f"   Querying GDELT for {len(search_terms)} search terms...")
    print(f"   Date range: {start_date} to {end_date}")
    
    all_articles = []
    
    for i, term in enumerate(search_terms):
        print(f"      [{i+1}/{len(search_terms)}] Searching: {term}")
        
        articles = query_gdelt(term, start_date, end_date, max_records=250)
        
        for article in articles:
            all_articles.append({
                'date': article.get('seendate', '')[:8],  # YYYYMMDD
                'title': article.get('title', ''),
                'source': article.get('domain', ''),
                'url': article.get('url', ''),
                'search_term': term
            })
        
        print(f"         Found {len(articles)} articles")
        
        time.sleep(delay)  # Rate limiting
    
    if not all_articles:
        print("   ⚠️  No articles found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_articles)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date', 'title'])
    
    # Remove duplicates (same title)
    df = df.drop_duplicates(subset=['title'])
    
    # Filter to valid date range
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    print(f"\n   ✅ Collected {len(df)} unique headlines")
    
    return df


# =============================================================================
# FINBERT SENTIMENT SCORING
# =============================================================================

class FinBERTScorer:
    """FinBERT sentiment scorer."""
    
    def __init__(self, model_name: str = FINBERT_MODEL):
        print("   Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"      ✅ Model loaded on {self.device}")
        
        # Label mapping
        self.labels = ['positive', 'negative', 'neutral']
    
    def score_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Score a batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            List of score dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Extract scores
            for j, prob in enumerate(probs):
                pos_score = prob[0].item()
                neg_score = prob[1].item()
                neu_score = prob[2].item()
                
                # Compute sentiment score (-1 to +1)
                sentiment_score = pos_score - neg_score
                
                # Determine label
                label_idx = torch.argmax(prob).item()
                label = self.labels[label_idx]
                
                results.append({
                    'finbert_score': sentiment_score,
                    'positive_prob': pos_score,
                    'negative_prob': neg_score,
                    'neutral_prob': neu_score,
                    'sentiment_label': label,
                    'confidence': prob[label_idx].item()
                })
        
        return results


def score_headlines_finbert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score headlines using FinBERT.
    
    Args:
        df: DataFrame with 'title' column
        
    Returns:
        DataFrame with sentiment scores added
    """
    if not FINBERT_AVAILABLE:
        print("   ⚠️  FinBERT not available, skipping scoring")
        return df
    
    print_section("STEP 2: FinBERT Sentiment Scoring")
    
    scorer = FinBERTScorer()
    
    print(f"   Scoring {len(df)} headlines...")
    
    texts = df['title'].tolist()
    scores = scorer.score_batch(texts)
    
    # Add scores to dataframe
    score_df = pd.DataFrame(scores)
    df = pd.concat([df.reset_index(drop=True), score_df], axis=1)
    
    print(f"   ✅ Scoring complete")
    print(f"      Positive: {(df['sentiment_label'] == 'positive').sum()}")
    print(f"      Negative: {(df['sentiment_label'] == 'negative').sum()}")
    print(f"      Neutral: {(df['sentiment_label'] == 'neutral').sum()}")
    
    return df


# =============================================================================
# DAILY AGGREGATION
# =============================================================================

def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate headline-level sentiment to daily.
    
    Args:
        df: DataFrame with headline-level sentiment
        
    Returns:
        Daily aggregated sentiment
    """
    print_section("STEP 3: Daily Aggregation")
    
    # Group by date
    daily = df.groupby('date').agg({
        'finbert_score': ['mean', 'std', 'count'],
        'positive_prob': 'mean',
        'negative_prob': 'mean',
        'neutral_prob': 'mean'
    }).reset_index()
    
    # Flatten column names
    daily.columns = ['date', 'fb_sent_mean', 'fb_sent_std', 'headline_count',
                     'positive_avg', 'negative_avg', 'neutral_avg']
    
    # Rename for compatibility with existing data
    daily['sector_fb_sent'] = daily['fb_sent_mean']
    
    # Also create AV-style score (slightly different scaling)
    daily['sector_av_sent'] = daily['fb_sent_mean'] * 0.5  # Scale to match AV range
    
    print(f"   ✅ Aggregated to {len(daily)} daily observations")
    print(f"   📊 Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"   📊 Avg headlines/day: {daily['headline_count'].mean():.1f}")
    
    return daily


# =============================================================================
# MERGE WITH EXISTING DATA
# =============================================================================

def merge_with_existing(
    new_daily: pd.DataFrame,
    existing_av_path: Path,
    existing_fb_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge new daily sentiment with existing data.
    
    Args:
        new_daily: New daily sentiment from GDELT
        existing_av_path: Path to existing AV sentiment
        existing_fb_path: Path to existing FB sentiment
        
    Returns:
        Tuple of (merged AV df, merged FB df)
    """
    print_section("STEP 4: Merging with Existing Data")
    
    # Load existing
    av_df = pd.read_csv(existing_av_path, parse_dates=['date'])
    fb_df = pd.read_csv(existing_fb_path, parse_dates=['date'])
    
    print(f"   Existing AV data: {len(av_df)} days ({av_df['date'].min().date()} to {av_df['date'].max().date()})")
    print(f"   Existing FB data: {len(fb_df)} days ({fb_df['date'].min().date()} to {fb_df['date'].max().date()})")
    
    # Prepare new data for merging
    new_av = new_daily[['date', 'sector_av_sent']].copy()
    new_fb = new_daily[['date', 'sector_fb_sent']].copy()
    
    # Remove overlapping dates from new data
    existing_dates = set(av_df['date'].dt.date)
    new_av = new_av[~new_av['date'].dt.date.isin(existing_dates)]
    new_fb = new_fb[~new_fb['date'].dt.date.isin(existing_dates)]
    
    print(f"   New data to add: {len(new_av)} days")
    
    # Merge
    merged_av = pd.concat([av_df, new_av], ignore_index=True).sort_values('date')
    merged_fb = pd.concat([fb_df, new_fb], ignore_index=True).sort_values('date')
    
    # Fill any remaining gaps with forward fill (for weekends, etc.)
    merged_av = merged_av.drop_duplicates(subset=['date'])
    merged_fb = merged_fb.drop_duplicates(subset=['date'])
    
    print(f"\n   Merged AV data: {len(merged_av)} days ({merged_av['date'].min().date()} to {merged_av['date'].max().date()})")
    print(f"   Merged FB data: {len(merged_fb)} days ({merged_fb['date'].min().date()} to {merged_fb['date'].max().date()})")
    
    # Check for remaining gaps
    merged_av['gap'] = merged_av['date'].diff().dt.days
    gaps = merged_av[merged_av['gap'] > 5]
    if len(gaps) > 0:
        print(f"\n   ⚠️  Remaining gaps > 5 days:")
        for _, row in gaps.iterrows():
            print(f"      {row['date'].date()}: {row['gap']} day gap")
    else:
        print(f"\n   ✅ No significant gaps remaining!")
    
    return merged_av, merged_fb


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("🔧 Fill Sentiment Data Gap (GDELT + FinBERT)")
    
    # =========================================================================
    # STEP 1: Collect headlines from GDELT
    # =========================================================================
    print_section("STEP 1: Collecting Headlines from GDELT")
    
    headlines_df = collect_gdelt_headlines(
        search_terms=SEARCH_TERMS,
        start_date=GAP_START,
        end_date=GAP_END,
        delay=1.0
    )
    
    if headlines_df.empty:
        print("   ❌ No headlines collected. Cannot proceed.")
        return False
    
    # Save raw headlines
    raw_path = config.RAW_NEWS_DIR / 'gdelt_headlines_gap.csv'
    headlines_df.to_csv(raw_path, index=False)
    print(f"   💾 Saved raw headlines: {raw_path}")
    
    # =========================================================================
    # STEP 2: Score with FinBERT
    # =========================================================================
    scored_df = score_headlines_finbert(headlines_df)
    
    # Save scored headlines
    scored_path = config.RAW_NEWS_DIR / 'gdelt_headlines_scored.csv'
    scored_df.to_csv(scored_path, index=False)
    print(f"   💾 Saved scored headlines: {scored_path}")
    
    # =========================================================================
    # STEP 3: Aggregate to daily
    # =========================================================================
    daily_df = aggregate_daily_sentiment(scored_df)
    
    # Save daily aggregated
    daily_path = config.PROCESSED_DATA_DIR / 'daily_sentiment_gdelt.csv'
    daily_df.to_csv(daily_path, index=False)
    print(f"   💾 Saved daily sentiment: {daily_path}")
    
    # =========================================================================
    # STEP 4: Merge with existing data
    # =========================================================================
    merged_av, merged_fb = merge_with_existing(
        daily_df,
        config.PROCESSED_DATA_DIR / 'daily_sentiment_av.csv',
        config.PROCESSED_DATA_DIR / 'daily_sentiment_fb.csv'
    )
    
    # Backup existing files
    backup_dir = config.PROCESSED_DATA_DIR / 'backup'
    backup_dir.mkdir(exist_ok=True)
    
    import shutil
    shutil.copy(
        config.PROCESSED_DATA_DIR / 'daily_sentiment_av.csv',
        backup_dir / 'daily_sentiment_av_backup.csv'
    )
    shutil.copy(
        config.PROCESSED_DATA_DIR / 'daily_sentiment_fb.csv',
        backup_dir / 'daily_sentiment_fb_backup.csv'
    )
    print(f"\n   💾 Backed up original files to: {backup_dir}")
    
    # Save merged data
    merged_av.to_csv(config.PROCESSED_DATA_DIR / 'daily_sentiment_av.csv', index=False)
    merged_fb.to_csv(config.PROCESSED_DATA_DIR / 'daily_sentiment_fb.csv', index=False)
    print(f"   💾 Saved merged sentiment files")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("📋 SUMMARY")
    
    print(f"""
   Headlines collected: {len(headlines_df)}
   Days with sentiment: {len(daily_df)}
   Date range filled: {GAP_START} to {GAP_END}
   
   Files created:
      • {raw_path.name} (raw headlines)
      • {scored_path.name} (with FinBERT scores)
      • {daily_path.name} (daily aggregated)
   
   Files updated:
      • daily_sentiment_av.csv (merged)
      • daily_sentiment_fb.csv (merged)
   
   Backups saved to: {backup_dir}
    """)
    
    print("\n" + "=" * 70)
    print("   ✅ GAP FILLING COMPLETE")
    print("=" * 70)
    print("\n   Next step: Re-run your forecasting scripts!")
    print("      python scripts/06_forecast_enhanced.py")
    print()
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
