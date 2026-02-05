# src/data_collection.py
"""
Data collection module for financial data and news headlines.
Interfaces with Yahoo Finance, NewsAPI, and Alpha Vantage.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import config
from src.utils import (
    logger, save_dataframe, exponential_backoff_retry, validate_dataframe
)


class YahooFinanceCollector:
    """Collect historical price data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize Yahoo Finance collector."""
        try:
            import yfinance as yf
            self.yf = yf
            logger.info("✅ Yahoo Finance initialized")
        except ImportError:
            logger.error("❌ yfinance not installed. Run: pip install yfinance")
            raise
    
    def fetch_ticker_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a single ticker.
        
        Args:
            ticker: Ticker symbol (e.g., 'NVDA', 'SMH')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            auto_adjust: Whether to use adjusted prices
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"📥 Fetching {ticker} from {start_date} to {end_date}")
        
        try:
            # Fetch data with retry logic
            def fetch():
                ticker_obj = self.yf.Ticker(ticker)
                df = ticker_obj.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=auto_adjust
                )
                return df
            
            df = exponential_backoff_retry(fetch, max_retries=3)
            
            if df.empty:
                logger.warning(f"⚠️  No data returned for {ticker}")
                return pd.DataFrame()
            
            # Clean and standardize column names
            df = df.reset_index()
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Rename 'index' to 'date' if needed
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'date'})
            
            # Ensure date is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"✅ {ticker}: {len(df)} rows retrieved")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        delay: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers with rate limiting.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            delay: Delay between requests (seconds)
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        
        logger.info(f"📥 Fetching {len(tickers)} tickers...")
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{len(tickers)}] {ticker}")
            
            df = self.fetch_ticker_data(ticker, start_date, end_date)
            
            if not df.empty:
                results[ticker] = df
            
            # Rate limiting
            if i < len(tickers):
                time.sleep(delay)
        
        logger.info(f"✅ Successfully fetched {len(results)}/{len(tickers)} tickers")
        return results


class AlphaVantageNewsCollector:
    """Collect news with sentiment scores from Alpha Vantage."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage News & Sentiment collector.
        
        Args:
            api_key: Alpha Vantage API key (defaults to config)
        """
        self.api_key = api_key or config.ALPHA_VANTAGE_KEY
        
        if not self.api_key:
            logger.error("❌ Alpha Vantage API key not found")
            raise ValueError("ALPHA_VANTAGE_KEY required")
        
        self.base_url = "https://www.alphavantage.co/query"
        logger.info("✅ Alpha Vantage News & Sentiment initialized")
    
    def fetch_news_sentiment(
        self,
        tickers: Union[str, List[str]],
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        limit: int = 1000,
        sort: str = 'LATEST'
    ) -> List[Dict]:
        """
        Fetch news with sentiment scores from Alpha Vantage.
        
        Args:
            tickers: Single ticker or list of tickers
            time_from: Start datetime 'YYYYMMDDTHHMM' (optional)
            time_to: End datetime 'YYYYMMDDTHHMM' (optional)
            limit: Number of articles (max 1000)
            sort: 'LATEST', 'EARLIEST', or 'RELEVANCE'
            
        Returns:
            List of article dictionaries with sentiment scores
        """
        if isinstance(tickers, list):
            tickers_str = ','.join(tickers)
        else:
            tickers_str = tickers
        
        logger.info(f"📰 Fetching Alpha Vantage news sentiment for: {tickers_str}")
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': tickers_str,
            'apikey': self.api_key,
            'limit': limit,
            'sort': sort
        }
        
        if time_from:
            params['time_from'] = time_from
        if time_to:
            params['time_to'] = time_to
        
        try:
            import requests
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"❌ Alpha Vantage error: {data['Error Message']}")
                return []
            
            if 'Note' in data:
                logger.warning(f"⚠️  API limit: {data['Note']}")
                return []
            
            articles = data.get('feed', [])
            logger.info(f"✅ Retrieved {len(articles)} articles")
            
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error fetching Alpha Vantage news: {e}")
            return []
    
    def parse_articles_to_dataframe(
        self,
        articles: List[Dict],
        target_tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Parse Alpha Vantage articles to DataFrame with sentiment scores.
        
        Args:
            articles: List of article dictionaries from API
            target_tickers: Filter to these tickers only
            
        Returns:
            DataFrame with columns: date, ticker, headline, sentiment_score, etc.
        """
        if not articles:
            return pd.DataFrame()
        
        rows = []
        
        for article in articles:
            # Parse timestamp
            time_published = article.get('time_published', '')
            try:
                # Format: YYYYMMDDTHHMMSS
                date = pd.to_datetime(time_published, format='%Y%m%dT%H%M%S').date()
            except:
                continue
            
            # Overall sentiment
            overall_sentiment = article.get('overall_sentiment_score', 0.0)
            overall_label = article.get('overall_sentiment_label', 'Neutral')
            
            # Ticker-specific sentiments
            ticker_sentiments = article.get('ticker_sentiment', [])
            
            if target_tickers:
                # Filter to target tickers only
                ticker_sentiments = [
                    ts for ts in ticker_sentiments 
                    if ts.get('ticker') in target_tickers
                ]
            
            # Create row for each ticker mentioned
            for ticker_sent in ticker_sentiments:
                ticker = ticker_sent.get('ticker')
                relevance = float(ticker_sent.get('relevance_score', 0.0))
                ticker_sentiment = float(ticker_sent.get('ticker_sentiment_score', overall_sentiment))
                ticker_label = ticker_sent.get('ticker_sentiment_label', overall_label)
                
                rows.append({
                    'date': date,
                    'ticker': ticker,
                    'headline': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'source': article.get('source', ''),
                    'url': article.get('url', ''),
                    'sentiment_score': ticker_sentiment,
                    'sentiment_label': ticker_label,
                    'relevance_score': relevance,
                    'overall_sentiment': overall_sentiment,
                    'overall_label': overall_label
                })
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✅ Parsed {len(df)} ticker-article pairs")
        return df
    
    def collect_ticker_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Collect news sentiment for a single ticker.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            
        Returns:
            DataFrame with news sentiment
        """
        # Convert dates to Alpha Vantage format: YYYYMMDDTHHMM
        time_from = pd.to_datetime(start_date).strftime('%Y%m%dT0000')
        time_to = pd.to_datetime(end_date).strftime('%Y%m%dT2359')
        
        articles = self.fetch_news_sentiment(
            tickers=ticker,
            time_from=time_from,
            time_to=time_to,
            limit=1000
        )
        
        df = self.parse_articles_to_dataframe(articles, target_tickers=[ticker])
        
        if not df.empty:
            logger.info(f"✅ {ticker}: {len(df)} articles with sentiment")
        else:
            logger.warning(f"⚠️  {ticker}: No articles found")
        
        return df
    
    def collect_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        delay: float = 12.0
    ) -> pd.DataFrame:
        """
        Collect news sentiment for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            delay: Delay between requests (free tier: 25/day, so ~1 per minute minimum)
            
        Returns:
            Combined DataFrame
        """
        all_news = []
        
        logger.info(f"📰 Collecting Alpha Vantage news for {len(tickers)} tickers...")
        logger.info(f"⏰ Rate limit: {delay}s between requests (~{3600/delay:.0f} per hour)")
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{i}/{len(tickers)}] {ticker}")
            
            df = self.collect_ticker_news(ticker, start_date, end_date)
            
            if not df.empty:
                all_news.append(df)
            
            # Rate limiting (critical for free tier)
            if i < len(tickers):
                logger.info(f"⏳ Waiting {delay}s (rate limit)...")
                time.sleep(delay)
        
        if not all_news:
            logger.error("❌ No news data collected from Alpha Vantage")
            return pd.DataFrame()
        
        # Combine all news
        combined_df = pd.concat(all_news, ignore_index=True)
        
        # Remove duplicates (same article for multiple tickers)
        combined_df = combined_df.drop_duplicates(subset=['date', 'ticker', 'headline'])
        combined_df = combined_df.sort_values(['date', 'ticker']).reset_index(drop=True)
        
        logger.info(f"\n✅ Total articles collected: {len(combined_df)}")
        logger.info(f"📊 Articles per ticker:")
        for ticker in tickers:
            count = len(combined_df[combined_df['ticker'] == ticker])
            logger.info(f"   {ticker}: {count} articles")
        
        return combined_df


class NewsAPICollector:
    """Collect news headlines from NewsAPI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsAPI collector.
        
        Args:
            api_key: NewsAPI key (defaults to config)
        """
        self.api_key = api_key or config.NEWS_API_KEY
        
        if not self.api_key:
            logger.error("❌ NewsAPI key not found")
            raise ValueError("NEWS_API_KEY required")
        
        try:
            from newsapi import NewsApiClient
            self.newsapi = NewsApiClient(api_key=self.api_key)
            logger.info("✅ NewsAPI initialized")
        except ImportError:
            logger.error("❌ newsapi-python not installed. Run: pip install newsapi-python")
            raise
    
    def fetch_headlines(
        self,
        query: str,
        from_date: str,
        to_date: str,
        language: str = 'en',
        sort_by: str = 'relevancy',
        page_size: int = 100
    ) -> List[Dict]:
        """
        Fetch news headlines for a query.
        
        Args:
            query: Search query (e.g., 'NVIDIA' or 'NVDA')
            from_date: Start date 'YYYY-MM-DD'
            to_date: End date 'YYYY-MM-DD'
            language: Language code
            sort_by: Sort method ('relevancy', 'popularity', 'publishedAt')
            page_size: Number of results per page
            
        Returns:
            List of article dictionaries
        """
        logger.info(f"📰 Fetching news: '{query}' from {from_date} to {to_date}")
        
        try:
            response = self.newsapi.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language=language,
                sort_by=sort_by,
                page_size=page_size
            )
            
            articles = response.get('articles', [])
            total_results = response.get('totalResults', 0)
            
            logger.info(
                f"✅ '{query}': {len(articles)} articles "
                f"(total available: {total_results})"
            )
            
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error fetching news for '{query}': {e}")
            return []
    
    def fetch_ticker_news(
        self,
        ticker: str,
        company_name: str,
        start_date: str,
        end_date: str,
        delay: float = 1.0
    ) -> pd.DataFrame:
        """
        Fetch news for a ticker using both ticker and company name.
        
        Args:
            ticker: Ticker symbol (e.g., 'NVDA')
            company_name: Company name (e.g., 'NVIDIA')
            start_date: Start date
            end_date: End date
            delay: Delay between requests
            
        Returns:
            DataFrame with news headlines
        """
        all_articles = []
        
        # Try ticker symbol
        articles_ticker = self.fetch_headlines(ticker, start_date, end_date)
        all_articles.extend(articles_ticker)
        
        time.sleep(delay)
        
        # Try company name
        articles_company = self.fetch_headlines(company_name, start_date, end_date)
        all_articles.extend(articles_company)
        
        if not all_articles:
            logger.warning(f"⚠️  No articles found for {ticker} ({company_name})")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': pd.to_datetime(article['publishedAt']).date(),
            'ticker': ticker,
            'headline': article['title'],
            'description': article.get('description', ''),
            'source': article['source']['name'],
            'url': article['url']
        } for article in all_articles])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['headline'])
        
        logger.info(f"✅ {ticker}: {len(df)} unique headlines")
        return df
    
    def fetch_multiple_tickers(
        self,
        tickers_info: Dict[str, str],
        start_date: str,
        end_date: str,
        delay: float = 2.0
    ) -> pd.DataFrame:
        """
        Fetch news for multiple tickers.
        
        Args:
            tickers_info: Dict mapping ticker to company name
            start_date: Start date
            end_date: End date
            delay: Delay between tickers
            
        Returns:
            Combined DataFrame
        """
        all_news = []
        
        logger.info(f"📰 Fetching news for {len(tickers_info)} tickers...")
        
        for i, (ticker, company_name) in enumerate(tickers_info.items(), 1):
            logger.info(f"[{i}/{len(tickers_info)}] {ticker} ({company_name})")
            
            df = self.fetch_ticker_news(
                ticker, company_name, start_date, end_date, delay=1.0
            )
            
            if not df.empty:
                all_news.append(df)
            
            # Rate limiting between tickers
            if i < len(tickers_info):
                time.sleep(delay)
        
        if not all_news:
            logger.error("❌ No news data collected")
            return pd.DataFrame()
        
        # Combine all news
        combined_df = pd.concat(all_news, ignore_index=True)
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✅ Total news collected: {len(combined_df)} headlines")
        return combined_df


def collect_all_price_data(
    save_to_disk: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Collect all price data (equities, commodities, indices).
    
    Args:
        save_to_disk: Whether to save CSVs
        
    Returns:
        Dictionary of DataFrames
    """
    logger.info("🚀 Starting price data collection...")
    
    collector = YahooFinanceCollector()
    all_data = {}
    
    # Collect individual semiconductor stocks
    logger.info("\n💻 Collecting Individual Semiconductor Stocks...")
    for ticker in config.SEMICONDUCTOR_TICKERS:
        df = collector.fetch_ticker_data(
            ticker, config.START_DATE, config.END_DATE
        )
        if not df.empty:
            all_data[ticker] = df
            if save_to_disk:
                filepath = config.get_file_path('raw_prices', f'{ticker}_history.csv')
                save_dataframe(df, filepath)
    
    # Collect equity ETFs
    logger.info("\n📊 Collecting Equity ETFs...")
    for ticker in config.ETF_TICKERS:
        df = collector.fetch_ticker_data(
            ticker, config.START_DATE, config.END_DATE
        )
        if not df.empty:
            all_data[ticker] = df
            if save_to_disk:
                filepath = config.get_file_path('raw_prices', f'{ticker}_history.csv')
                save_dataframe(df, filepath)
    
    # Collect commodities
    logger.info("\n🛢️ Collecting Commodity Futures...")
    for ticker in config.COMMODITY_TICKERS:
        df = collector.fetch_ticker_data(
            ticker, config.START_DATE, config.END_DATE
        )
        if not df.empty:
            all_data[ticker] = df
            if save_to_disk:
                filepath = config.get_file_path('raw_commodities', f'{ticker}_history.csv')
                save_dataframe(df, filepath)
    
    # Collect market indices
    logger.info("\n📈 Collecting Market Indices...")
    for ticker in config.MARKET_INDICES:
        df = collector.fetch_ticker_data(
            ticker, config.START_DATE, config.END_DATE
        )
        if not df.empty:
            all_data[ticker] = df
            if save_to_disk:
                filepath = config.get_file_path('raw_prices', f'{ticker}_history.csv')
                save_dataframe(df, filepath)
    
    logger.info(f"\n✅ Price data collection complete: {len(all_data)} tickers")
    return all_data


def collect_all_news_data(
    save_to_disk: bool = True,
    use_alpha_vantage: bool = True
) -> pd.DataFrame:
    """
    Collect all news data for semiconductor tickers.
    
    Args:
        save_to_disk: Whether to save CSV
        use_alpha_vantage: Use Alpha Vantage (True) or NewsAPI (False)
        
    Returns:
        Combined news DataFrame
    """
    logger.info("🚀 Starting news data collection...")
    
    try:
        if use_alpha_vantage and config.ALPHA_VANTAGE_KEY:
            # Use Alpha Vantage News & Sentiment API
            logger.info("📡 Using Alpha Vantage News & Sentiment API")
            collector = AlphaVantageNewsCollector()
            
            df = collector.collect_multiple_tickers(
                tickers=config.SEMICONDUCTOR_TICKERS,
                start_date=config.START_DATE,
                end_date=config.END_DATE,
                delay=15.0  # 15 seconds = ~240 requests/hour (well under limit)
            )
            
        else:
            # Fallback to NewsAPI
            logger.info("📡 Using NewsAPI")
            tickers_info = {
                'NVDA': 'NVIDIA',
                'AMD': 'AMD',
                'INTC': 'Intel',
                'TSM': 'TSMC',
                'MU': 'Micron'
            }
            
            collector = NewsAPICollector()
            df = collector.fetch_multiple_tickers(
                tickers_info,
                config.START_DATE,
                config.END_DATE,
                delay=2.0
            )
        
        if save_to_disk and not df.empty:
            filepath = config.get_file_path('raw_news', 'sentiment_raw_data.csv')
            save_dataframe(df, filepath)
        
        logger.info(f"✅ News data collection complete: {len(df)} articles")
        return df
        
    except Exception as e:
        logger.error(f"❌ News collection failed: {e}")
        logger.info("💡 Continuing without news data (will use price data only)")
        return pd.DataFrame()


if __name__ == '__main__':
    print("🧪 Testing data collection module...")
    
    # Test Yahoo Finance
    print("\n" + "="*60)
    print("TEST 1: Yahoo Finance - Single Ticker")
    print("="*60)
    collector = YahooFinanceCollector()
    df = collector.fetch_ticker_data('NVDA', '2024-01-01', '2024-01-31')
    if not df.empty:
        print(f"✅ Fetched {len(df)} rows for NVDA")
        print(df.head())
    
    print("\n✅ Data collection module tests complete!")
