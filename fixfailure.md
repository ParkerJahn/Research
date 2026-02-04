# PCA-VARX Forecasting Pipeline (Yahoo Finance + Free Sentiment)

## Overview

This pipeline predicts implied volatility (IV) or realized volatility for NVIDIA and semiconductor ETFs (SMH, SOXX) using:

- Stock/ETF returns
- Realized volatility (as a proxy for IV)
- News sentiment aggregated into PCA factors
- VARX (Vector Autoregression with exogenous factors)

---

## 1. Data Collection

### Stock/ETF Prices

- **Sources**: Yahoo Finance (`yfinance`), Alpha Vantage
- **Assets**: NVDA, SMH, SOXX
- **Frequency**: Daily adjusted close
- **Processing**:
  - Compute **daily returns**:  
    `r_t = (P_t - P_{t-1}) / P_{t-1}`
  - Compute **realized volatility** (optional IV proxy):  
    `RV_t = sqrt(sum(r_{t-20}^2))` for a 21-day rolling window

⚠️ **Fatal Error to Avoid**: Do NOT mix adjusted close with raw close prices. Always adjust for splits/dividends.

---

### News Sentiment

- **Sources**: NewsAPI.org, Google News RSS, or other free news sources
- **Assets**: NVIDIA, semiconductor sector
- **Frequency**: Daily aggregation
- **Processing**:
  - Extract headlines
  - Apply sentiment scoring (e.g., VADER or TextBlob)
  - Aggregate daily sentiment scores per asset

⚠️ **Fatal Error to Avoid**: Align sentiment dates exactly with price/volatility data. Misalignment will break PCA and VAR.

---

## 2. Feature Engineering

- Combine:
  - Daily returns of NVDA, SMH, SOXX
  - Realized vol or ETF IV
  - Daily sentiment scores
- Standardize / normalize all features before PCA
- Perform PCA:
  - Extract top 3–5 components
  - Ensure they capture >50% variance if possible
  - Label components: e.g., "Market Factor", "Volatility Factor", "Sentiment Shock"

⚠️ **Fatal Error to Avoid**: Do NOT include non-stationary variables in PCA without transformation (e.g., returns or ΔIV). Stationarity is critical for VAR.

---

## 3. VARX Model

### Setup

- **Endogenous variables**: SMH vol, SOXX vol (or realized vol proxies)
- **Exogenous variables**: PCA factors (from returns + sentiment)
- **Lag selection**: 1–5, use AIC/BIC to select optimal
- **Train/Test Split**:
  - Use **recent AI-era data only** (post-2023)
  - Example: 70% train, 30% test
  - Ensure **no pre-AI era data** is included if sentiment is only post-AI

### Model Fitting

- Fit VARX with selected lags
- Test:
  - Granger causality
  - Impulse response functions (IRFs)
- Generate rolling forecasts (e.g., refit every 20 days)

⚠️ **Fatal Errors to Avoid**:
- Overfitting: do not use too many lags for a small dataset (<200 observations)
- Non-stationarity: check ADF tests for all endogenous variables
- Exogenous mismatch: PCA factors must be standardized and aligned with endogenous variables
- Mixing pre-AI and post-AI data for sentiment: will bias results

---

## 4. Forecast Evaluation

- Metrics:
  - RMSE
  - MAE
  - Directional Accuracy
- Evaluate across multiple horizons (1-day, 5-day, 10-day)
- Compare against **baseline models**:
  - HAR-IV
  - AR(1)
  - Random Walk

⚠️ **Fatal Error to Avoid**: Do not evaluate forecasts on overlapping windows with training data. Maintain strict out-of-sample test set.

---

## 5. Recommendations / Best Practices

1. Always check **stationarity** of all inputs
2. Align **dates for sentiment and market data**
3. Start simple: use **sector ETFs + sentiment**, avoid trying to model individual stock IV if data is sparse
4. Keep PCA factor count reasonable (3–5)
5. Refit VARX periodically to avoid stale coefficients in rolling forecasts
6. Clearly **document which data is post-AI era** to avoid regime bias
7. Ensure **train set > 100 observations** for each VARX fit

---

## 6. Optional Enhancements

- Once premium data is available:
  - Replace realized volatility with true **IV from OptionMetrics or CBOE**
  - Include **skew and term structure** as PCA features
- Consider **adding commodity/macro factors** to PCA for multivariate influence

---

## References / Tools

- `yfinance` or Alpha Vantage for prices
- `scikit-learn` PCA for dimensionality reduction
- `statsmodels.tsa.api.VAR` for VARX modeling


