# src/sentiment_analysis.py
"""
Sentiment analysis module using FinBERT.
Processes news headlines and generates sentiment scores.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from src.utils import logger, save_dataframe, load_dataframe


class SentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
    """
    
    def __init__(self, model_name: str = None, batch_size: int = None):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name (default: from config)
            batch_size: Batch size for processing (default: from config)
        """
        self.model_name = model_name or config.FINBERT_MODEL
        self.batch_size = batch_size or config.SENTIMENT_BATCH_SIZE
        self.pipeline = None
        
        logger.info(f"🤖 Initializing FinBERT: {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load FinBERT model and tokenizer."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                truncation=True,
                device=-1  # CPU (-1), or 0 for GPU
            )
            
            logger.info("✅ FinBERT loaded successfully")
            
        except ImportError:
            logger.error("❌ Transformers not installed. Run: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading FinBERT: {e}")
            raise
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or pd.isna(text):
            return {'label': 'neutral', 'score': 0.0}
        
        try:
            result = self.pipeline(text)[0]
            
            # Convert label to score: positive=1, negative=-1, neutral=0
            label = result['label'].lower()
            confidence = result['score']
            
            if label == 'positive':
                sentiment_score = confidence
            elif label == 'negative':
                sentiment_score = -confidence
            else:  # neutral
                sentiment_score = 0.0
            
            return {
                'label': label,
                'confidence': confidence,
                'sentiment_score': sentiment_score
            }
            
        except Exception as e:
            logger.warning(f"⚠️  Error analyzing text: {e}")
            return {'label': 'neutral', 'confidence': 0.0, 'sentiment_score': 0.0}
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment dictionaries
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Filter out empty texts
            valid_texts = [t for t in batch if t and not pd.isna(t)]
            
            if not valid_texts:
                results.extend([{'label': 'neutral', 'confidence': 0.0, 'sentiment_score': 0.0}] * len(batch))
                continue
            
            try:
                # Run pipeline on batch
                batch_results = self.pipeline(valid_texts)
                
                # Convert to sentiment scores
                for result in batch_results:
                    label = result['label'].lower()
                    confidence = result['score']
                    
                    if label == 'positive':
                        sentiment_score = confidence
                    elif label == 'negative':
                        sentiment_score = -confidence
                    else:
                        sentiment_score = 0.0
                    
                    results.append({
                        'label': label,
                        'confidence': confidence,
                        'sentiment_score': sentiment_score
                    })
                    
            except Exception as e:
                logger.warning(f"⚠️  Error in batch processing: {e}")
                results.extend([{'label': 'neutral', 'confidence': 0.0, 'sentiment_score': 0.0}] * len(batch))
        
        return results
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'headline'
    ) -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df: DataFrame with text column
            text_column: Name of column containing text
            
        Returns:
            DataFrame with sentiment columns added
        """
        logger.info(f"🔬 Analyzing sentiment for {len(df)} texts...")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Get texts
        texts = df[text_column].fillna('').tolist()
        
        # Analyze with progress bar
        results = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing batches"):
            batch = texts[i:i + self.batch_size]
            batch_results = self.analyze_batch(batch)
            results.extend(batch_results)
        
        # Add results to DataFrame
        df = df.copy()
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_confidence'] = [r['confidence'] for r in results]
        df['sentiment_score'] = [r['sentiment_score'] for r in results]
        
        logger.info(f"✅ Sentiment analysis complete")
        logger.info(f"   Positive: {(df['sentiment_label'] == 'positive').sum()}")
        logger.info(f"   Negative: {(df['sentiment_label'] == 'negative').sum()}")
        logger.info(f"   Neutral: {(df['sentiment_label'] == 'neutral').sum()}")
        
        return df


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_column: str = 'date',
    ticker_column: str = 'ticker',
    sentiment_column: str = 'sentiment_score'
) -> pd.DataFrame:
    """
    Aggregate sentiment scores to daily level per ticker.
    
    Args:
        df: DataFrame with sentiment scores
        date_column: Name of date column
        ticker_column: Name of ticker column
        sentiment_column: Name of sentiment score column
        
    Returns:
        DataFrame with daily aggregated sentiment
    """
    logger.info("📊 Aggregating sentiment to daily level...")
    
    # Ensure date is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Aggregate by date and ticker
    daily_sentiment = df.groupby([date_column, ticker_column]).agg({
        sentiment_column: ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    daily_sentiment.columns = [
        date_column, ticker_column, 
        'sentiment_mean', 'sentiment_std', 'headline_count'
    ]
    
    # Fill missing std (when only 1 headline)
    daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
    
    logger.info(f"✅ Daily sentiment: {len(daily_sentiment)} date-ticker pairs")
    
    return daily_sentiment


def create_sector_sentiment(
    df: pd.DataFrame,
    date_column: str = 'date',
    ticker_column: str = 'ticker',
    sentiment_column: str = 'sentiment_mean'
) -> pd.DataFrame:
    """
    Create sector-wide sentiment by averaging across tickers.
    
    Args:
        df: DataFrame with daily ticker sentiment
        date_column: Name of date column
        ticker_column: Name of ticker column
        sentiment_column: Name of sentiment column to average
        
    Returns:
        DataFrame with sector sentiment
    """
    logger.info("🏭 Creating sector-wide sentiment...")
    
    sector_sentiment = df.groupby(date_column).agg({
        sentiment_column: 'mean',
        'headline_count': 'sum'
    }).reset_index()
    
    sector_sentiment = sector_sentiment.rename(columns={
        sentiment_column: 'sector_sentiment',
        'headline_count': 'sector_headline_count'
    })
    
    logger.info(f"✅ Sector sentiment: {len(sector_sentiment)} days")
    
    return sector_sentiment


def pivot_ticker_sentiment(
    df: pd.DataFrame,
    date_column: str = 'date',
    ticker_column: str = 'ticker',
    sentiment_column: str = 'sentiment_mean'
) -> pd.DataFrame:
    """
    Pivot sentiment data to wide format (one column per ticker).
    
    Args:
        df: DataFrame with daily ticker sentiment
        date_column: Name of date column
        ticker_column: Name of ticker column
        sentiment_column: Name of sentiment column
        
    Returns:
        DataFrame in wide format
    """
    logger.info("🔄 Pivoting sentiment to wide format...")
    
    # Pivot
    wide_df = df.pivot(
        index=date_column,
        columns=ticker_column,
        values=sentiment_column
    ).reset_index()
    
    # Rename columns (lowercase ticker names)
    wide_df.columns = [date_column] + [
        f"{col.lower()}_sent" for col in wide_df.columns[1:]
    ]
    
    logger.info(f"✅ Pivoted sentiment: {len(wide_df)} days, {len(wide_df.columns)-1} tickers")
    
    return wide_df


def process_news_sentiment(
    news_csv: Path,
    output_csv: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline: load news, analyze sentiment, aggregate.
    
    Args:
        news_csv: Path to raw news CSV
        output_csv: Optional path to save processed sentiment
        
    Returns:
        Tuple of (daily_sentiment_df, sector_sentiment_df)
    """
    logger.info("🚀 Starting sentiment processing pipeline...")
    
    # Load news data
    news_df = load_dataframe(news_csv, parse_dates=['date'])
    
    if news_df.empty:
        logger.error("❌ No news data to process")
        return pd.DataFrame(), pd.DataFrame()
    
    # Analyze sentiment
    analyzer = SentimentAnalyzer()
    news_with_sentiment = analyzer.analyze_dataframe(news_df, text_column='headline')
    
    # Aggregate to daily level
    daily_sentiment = aggregate_daily_sentiment(news_with_sentiment)
    
    # Create sector sentiment
    sector_sentiment = create_sector_sentiment(daily_sentiment)
    
    # Pivot to wide format
    daily_wide = pivot_ticker_sentiment(daily_sentiment)
    
    # Merge with sector sentiment
    daily_combined = daily_wide.merge(sector_sentiment, on='date', how='left')
    
    # Save if path provided
    if output_csv:
        save_dataframe(daily_combined, output_csv)
    
    logger.info("✅ Sentiment processing complete")
    
    return daily_combined, sector_sentiment


if __name__ == '__main__':
    print("🧪 Testing sentiment analysis module...")
    
    # Test on sample headlines
    sample_headlines = [
        "NVIDIA stock surges to record high on strong earnings",
        "Intel faces challenges as competition intensifies",
        "AMD announces new chip technology",
        "Semiconductor industry outlook remains uncertain"
    ]
    
    print("\n" + "="*60)
    print("TEST: Sentiment Analysis")
    print("="*60)
    
    analyzer = SentimentAnalyzer()
    
    for headline in sample_headlines:
        result = analyzer.analyze_text(headline)
        print(f"\nHeadline: {headline}")
        print(f"Sentiment: {result['label']} (score: {result['sentiment_score']:.3f})")
    
    print("\n✅ Sentiment analysis module tests complete!")
