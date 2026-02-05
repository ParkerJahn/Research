# Methodology and Implementation Details

## Overview

This document provides comprehensive technical documentation for the sentiment-augmented VIX forecasting methodology. The approach combines news sentiment analysis, PCA dimensionality reduction, and HAR-IV (Heterogeneous Autoregressive - Implied Volatility) models with rigorous out-of-sample validation.

## Table of Contents

1. [Data Collection](#data-collection)
2. [Feature Engineering](#feature-engineering)
3. [Sentiment Processing](#sentiment-processing)
4. [Dimensionality Reduction](#dimensionality-reduction)
5. [Model Specifications](#model-specifications)
6. [Cross-Validation Procedure](#cross-validation-procedure)
7. [Statistical Testing](#statistical-testing)
8. [Code Structure](#code-structure)
9. [Reproducibility Notes](#reproducibility-notes)
10. [Adapting to Other Assets](#adapting-to-other-assets)

---

## Data Collection

### Market Data Sources

All market data is collected via **Yahoo Finance** (free, no API key required):

#### VIX (Target Variable)
- **Ticker**: `^VIX`
- **Description**: CBOE Volatility Index (30-day implied volatility of S&P 500)
- **Frequency**: Daily closing values
- **Date Range**: 2022-01-01 to present

#### ETF Prices (Semiconductor Sector)
- **Tickers**: `SMH`, `SOXX`
- **Purpose**: Calculate realized volatility and returns
- **Frequency**: Daily OHLCV data

#### Commodity Futures
- **Gold**: `GC=F` (safe haven indicator)
- **Oil**: `CL=F` (energy/economic indicator)
- **Copper**: `HG=F` (industrial demand indicator)
- **Purpose**: Capture macroeconomic stress factors

### Sentiment Data Sources

#### Alpha Vantage News Sentiment API

**Setup Instructions:**
1. Register for free API key at: https://www.alphavantage.co/support/#api-key
2. Add to `.env` file: `ALPHA_VANTAGE_API_KEY=your_key_here`
3. **Rate Limits**: 5 requests per minute (free tier)
4. **Expected Collection Time**: ~4 days for full dataset

**API Call Example:**
```python
import requests
import time

API_KEY = "your_key_here"
ticker = "NVDA"
url = f"https://www.alphavantage.co/query"
params = {
    'function': 'NEWS_SENTIMENT',
    'tickers': ticker,
    'apikey': API_KEY,
    'limit': 1000
}

response = requests.get(url, params=params)
data = response.json()

# Rate limiting (CRITICAL)
time.sleep(12)  # 5 requests/min = 12 seconds between calls
```

**Tickers Collected:**
- NVDA (NVIDIA)
- AMD (Advanced Micro Devices)
- INTC (Intel)
- TSM (Taiwan Semiconductor)
- MU (Micron Technology)

#### FinBERT Sentiment Processing

**Model**: `ProsusAI/finbert` (Hugging Face)

**Installation:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
```

**Preprocessing Steps:**
1. Truncate headlines to 512 tokens (BERT limit)
2. Tokenize with FinBERT tokenizer
3. Run inference to get sentiment scores: [positive, negative, neutral]
4. Aggregate daily: `sentiment = (positive - negative) / (positive + negative + neutral)`

**Code Snippet:**
```python
def get_sentiment(headline):
    inputs = tokenizer(headline, return_tensors="pt", 
                      truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pos, neg, neu = probs[0].tolist()
    return (pos - neg) / (pos + neg + neu)
```

---

## Feature Engineering

### Returns and Realized Volatility

**Returns Calculation:**
```python
returns = df['close'].pct_change()
```

**Realized Volatility (21-day rolling):**
```python
realized_vol = returns.rolling(21).std() * np.sqrt(252)
```

- **Window**: 21 trading days (~1 month)
- **Annualization**: Multiply by √252 (trading days per year)

### HAR Components

The HAR (Heterogeneous Autoregressive) model uses three time scales:

```python
# Daily (1-day lag)
vix_lag1 = df['vix'].shift(1)

# Weekly (5-day average)
vix_weekly = df['vix'].shift(1).rolling(5).mean()

# Monthly (22-day average)
vix_monthly = df['vix'].shift(1).rolling(22).mean()
```

**Key Point**: All lags use `.shift(1)` to prevent look-ahead bias.

---

## Sentiment Processing

### Sentiment Orthogonalization (Critical Step)

**Problem**: Raw sentiment may simply reflect contemporaneous returns, not forward-looking information.

**Solution**: Residualize sentiment against returns using expanding window OLS.

**Procedure:**

```python
from sklearn.linear_model import LinearRegression

def residualize_sentiment(sentiment, returns, dates, min_window=63):
    """
    Purge return information from sentiment using expanding window.
    
    Args:
        sentiment: Raw sentiment scores
        returns: Contemporaneous returns
        dates: Date index
        min_window: Minimum observations for first regression (63 = 3 months)
    
    Returns:
        Orthogonalized sentiment shocks
    """
    shocks = pd.Series(index=dates, dtype=float)
    
    for i in range(min_window, len(sentiment)):
        # Training data: all observations up to (but not including) current
        X_train = returns.iloc[:i].values.reshape(-1, 1)
        y_train = sentiment.iloc[:i].values
        
        # Fit OLS
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict and residualize current observation
        X_current = returns.iloc[i:i+1].values.reshape(-1, 1)
        y_current = sentiment.iloc[i]
        y_pred = model.predict(X_current)[0]
        
        shocks.iloc[i] = y_current - y_pred
    
    return shocks
```

**Why Expanding Window?**
- Avoids look-ahead bias (only uses past data)
- Adapts to changing sentiment-return relationship over time
- More realistic than full-sample OLS

### Sentiment Lags

After orthogonalization, create 1-day lags:

```python
av_shock_lag1 = av_shock.shift(1)
fb_shock_lag1 = fb_shock.shift(1)
```

---

## Dimensionality Reduction

### Rolling PCA Implementation

**Critical**: PCA must be fit on training data only to avoid look-ahead bias.

**Procedure:**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def rolling_pca(df, pca_cols, train_end_idx, test_end_idx, n_components=3):
    """
    Fit PCA on training data, transform test data.
    
    Args:
        df: Full dataset
        pca_cols: List of feature column names
        train_end_idx: Last index of training data
        test_end_idx: Last index of test data
        n_components: Number of PCs to retain
    
    Returns:
        train_pcs, test_pcs, loadings
    """
    # Split data
    train_df = df.iloc[:train_end_idx]
    test_df = df.iloc[train_end_idx:test_end_idx]
    
    # Standardize on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[pca_cols])
    X_test = scaler.transform(test_df[pca_cols])  # Use training scaler
    
    # Fit PCA on training data
    pca = PCA(n_components=n_components)
    train_pcs = pca.fit_transform(X_train)
    test_pcs = pca.transform(X_test)  # Use training PCA
    
    # Extract loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=pca_cols,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return train_pcs, test_pcs, loadings
```

**Features Used (9 total):**
1. SMH returns
2. SMH realized volatility
3. SOXX returns
4. SOXX realized volatility
5. Gold returns
6. Oil returns
7. Copper returns
8. AlphaVantage sentiment shock (lag 1)
9. FinBERT sentiment shock (lag 1)

**Interpretation** (from empirical loadings):
- **PC1**: Primarily captures returns (SMH, SOXX returns load heavily)
- **PC2**: Primarily captures volatility (SMH, SOXX RV load heavily)
- **PC3**: Primarily captures sentiment (AV, FB shocks load heavily)

---

## Model Specifications

### HAR-IV Baseline

**Equation:**
```
VIX(t) = β₀ + β₁·VIX(t-1) + β₂·VIX_weekly(t-1) + β₃·VIX_monthly(t-1) + ε(t)
```

**Implementation:**
```python
from sklearn.linear_model import Ridge

# Features
X = df[['vix_lag1', 'vix_weekly', 'vix_monthly']]
y = df['vix']

# Ridge regression (α=1.0)
model = Ridge(alpha=1.0)
model.fit(X, y)
predictions = model.predict(X_test)
```

### Augmented Model (HAR-IV + PCA + Sentiment)

**Equation:**
```
VIX(t) = β₀ + β₁·VIX(t-1) + β₂·VIX_weekly(t-1) + β₃·VIX_monthly(t-1)
         + β₄·PC1(t-1) + β₅·PC2(t-1) + β₆·PC3(t-1)
         + β₇·AV_shock(t-1) + β₈·FB_shock(t-1) + ε(t)
```

**Implementation:**
```python
# Features (8 total)
X_aug = df[['vix_lag1', 'vix_weekly', 'vix_monthly',
            'PC1', 'PC2', 'PC3',
            'av_shock_lag1', 'fb_shock_lag1']]
y = df['vix']

model_aug = Ridge(alpha=1.0)
model_aug.fit(X_aug, y)
predictions_aug = model_aug.predict(X_test)
```

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Ridge α | 1.0 | Standard regularization, prevents overfitting |
| Training Window | 200 days | ~9 months, balances stability vs. adaptability |
| Test Window | 50 days | ~2 months, sufficient for statistical power |
| Step Size | 25 days | ~1 month, balances computation vs. granularity |
| PCA Components | 3 | Captures 85%+ variance, interpretable |

---

## Cross-Validation Procedure

### Rolling Window Design

```
Timeline:
[-------- Training (200 days) --------][-- Test (50 days) --]
                                       [-------- Training (200 days) --------][-- Test (50 days) --]
                                                                              [-------- Training (200 days) --------][-- Test (50 days) --]
```

**Pseudocode:**
```python
n = len(df)
min_train = 200
test_size = 50
step_size = 25

results = []
start = min_train

while start + test_size <= n:
    # Define windows
    train_end = start
    test_start = start
    test_end = min(start + test_size, n)
    
    # Split data
    train_df = df.iloc[:train_end]
    test_df = df.iloc[test_start:test_end]
    
    # Fit PCA on training only
    scaler = StandardScaler()
    X_train_pca = scaler.fit_transform(train_df[pca_cols])
    pca = PCA(n_components=3)
    pca.fit(X_train_pca)
    
    # Transform both sets
    train_pcs = pca.transform(X_train_pca)
    X_test_pca = scaler.transform(test_df[pca_cols])
    test_pcs = pca.transform(X_test_pca)
    
    # Add PCs to dataframes
    for i in range(3):
        train_df[f'PC{i+1}'] = train_pcs[:, i]
        test_df[f'PC{i+1}'] = test_pcs[:, i]
    
    # Fit models
    model_baseline = Ridge(alpha=1.0)
    model_baseline.fit(train_df[baseline_features], train_df['vix'])
    
    model_aug = Ridge(alpha=1.0)
    model_aug.fit(train_df[augmented_features], train_df['vix'])
    
    # Predict on test set
    pred_baseline = model_baseline.predict(test_df[baseline_features])
    pred_aug = model_aug.predict(test_df[augmented_features])
    
    # Store results
    results.append({
        'dates': test_df['date'],
        'actual': test_df['vix'],
        'pred_baseline': pred_baseline,
        'pred_aug': pred_aug
    })
    
    # Move window forward
    start += step_size
```

### Forecast Horizons

For multi-step forecasts (5-day, 22-day), shift features forward:

```python
def prepare_horizon_data(df, horizon):
    """Shift features for h-step ahead forecasting."""
    result = df.copy()
    if horizon > 1:
        shift = horizon - 1
        for col in result.columns:
            if col not in ['date', 'vix']:
                result[col] = result[col].shift(shift)
    return result.dropna()
```

---

## Statistical Testing

### Diebold-Mariano Test

**Purpose**: Test if forecast accuracy difference is statistically significant.

**Null Hypothesis**: Two forecasts have equal accuracy.

**Implementation:**
```python
from scipy import stats

def diebold_mariano_test(actual, pred1, pred2, h=1):
    """
    Diebold-Mariano test for forecast comparison.
    
    Args:
        actual: Actual values
        pred1: Forecast 1 (baseline)
        pred2: Forecast 2 (augmented)
        h: Forecast horizon (for Newey-West adjustment)
    
    Returns:
        dm_stat, p_value
    """
    # Loss differential (squared error)
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2
    
    # Mean loss differential
    d_mean = np.mean(d)
    
    # Newey-West variance estimator
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[:-k], d[k:])[0, 1]
        gamma_sum += 2 * gamma_k
    
    variance = (gamma_0 + gamma_sum) / len(d)
    
    # DM statistic
    dm_stat = d_mean / np.sqrt(variance)
    
    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value
```

**Interpretation:**
- **p < 0.05**: Significant difference in forecast accuracy
- **DM > 0**: Model 2 (augmented) is better
- **DM < 0**: Model 1 (baseline) is better

### Multiple Testing Adjustment

With 3 forecast horizons, use **Bonferroni correction**:

```python
alpha = 0.05
bonferroni_alpha = alpha / 3  # = 0.0167

# Reject null if p < 0.0167
```

---

## Code Structure

### Main Scripts (Execution Order)

1. **`scripts/01_collect.py`**
   - Collects VIX, ETF, commodity data from Yahoo Finance
   - Collects news headlines from Alpha Vantage
   - Saves to `data/raw/`

2. **`scripts/02_process.py`**
   - Processes sentiment with FinBERT
   - Calculates returns and realized volatility
   - Orthogonalizes sentiment against returns
   - Saves to `data/processed/`

3. **`scripts/03_pca_varx.py`**
   - Runs rolling PCA
   - Fits HAR-IV baseline and augmented models
   - Performs cross-validation
   - Computes evaluation metrics
   - Saves results to `results/`

4. **`scripts/09_publication_figures.py`**
   - Generates all publication-quality figures
   - Saves to `results/figures/`

### Helper Modules

- **`src/data_collection.py`**: API wrappers for Yahoo Finance, Alpha Vantage
- **`src/sentiment_analysis.py`**: FinBERT processing functions
- **`src/sentiment_processing.py`**: Orthogonalization implementation
- **`src/feature_engineering.py`**: Returns, volatility calculations
- **`src/pca_expanding.py`**: Rolling PCA implementation
- **`src/models.py`**: HAR-IV model classes
- **`src/unified_metrics.py`**: RMSE, MAE, Diebold-Mariano tests
- **`src/utils.py`**: Logging, data validation utilities

---

## Reproducibility Notes

### Random Seeds

Set at the top of each script:

```python
import numpy as np
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

### Computational Environment

**Tested On:**
- **OS**: macOS 14.0, Ubuntu 22.04
- **Python**: 3.10.12
- **Hardware**: 16GB RAM, 4-core CPU

**Approximate Runtime:**
- Data collection: ~4 days (API rate limits)
- Processing: ~5 minutes
- PCA + Forecasting: ~15 minutes
- Figure generation: ~2 minutes

### Known Variations

**Sources of Randomness:**
1. FinBERT model initialization (fixed by seed)
2. Ridge regression solver (deterministic for α>0)
3. PCA (deterministic)

**Platform Differences:**
- Yahoo Finance may have slight data differences if run on different dates
- Alpha Vantage news corpus grows over time

**Expected Tolerance:**
- RMSE values: ±0.05
- Improvement percentages: ±1%

---

## Adapting to Other Assets

### Changing the Sentiment Source

**Example: Technology Sector**

1. Edit `config.py`:
```python
SEMICONDUCTOR_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
ETF_TICKERS = ['QQQ', 'XLK']  # NASDAQ, Tech sector ETF
```

2. Re-run data collection:
```bash
python scripts/01_collect.py
```

### Changing the Target Variable

**Example: NASDAQ Volatility (VXN)**

1. Edit `config.py`:
```python
IMPLIED_VOL_TICKERS = ['^VXN']  # NASDAQ-100 volatility
```

2. Update scripts to use `vxn` instead of `vix` column names

### Changing Forecast Horizons

**Example: Weekly and Monthly Forecasts**

1. Edit `config.py`:
```python
FORECAST_HORIZONS = [5, 22]  # 1-week, 1-month
```

2. Re-run forecasting:
```bash
python scripts/03_pca_varx.py
```

---

## References

### Key Papers

1. **Corsi, F. (2009)**. "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics*, 7(2), 174-196.
   - Original HAR-RV model

2. **Diebold, F. X., & Mariano, R. S. (1995)**. "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics*, 13(3), 253-263.
   - Forecast comparison methodology

3. **Araci, D. (2019)**. "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models." *arXiv preprint arXiv:1908.10063*.
   - FinBERT model

### Data Sources

- **Yahoo Finance**: https://finance.yahoo.com/
- **Alpha Vantage**: https://www.alphavantage.co/
- **FinBERT Model**: https://huggingface.co/ProsusAI/finbert

---

## Troubleshooting

### Common Issues

**Issue**: `KeyError: 'vix'` during processing  
**Solution**: Ensure VIX data was successfully downloaded. Check `data/raw/prices/^VIX_history.csv` exists.

**Issue**: PCA returns NaN values  
**Solution**: Check for missing data in features. Run `df[pca_cols].isnull().sum()` to identify gaps.

**Issue**: Diebold-Mariano test returns NaN  
**Solution**: Ensure forecast arrays have same length as actual values. Check for alignment issues.

**Issue**: Memory error during cross-validation  
**Solution**: Reduce `CV_TEST_SIZE` from 50 to 25 in `config.py`.

---

## Contact

For questions about methodology or implementation:

**Parker Jahn**  
Email: jahnparker90@gmail.com  
Institution: Rollins College
