# Sentiment-Driven Volatility Prediction

## Project Overview
Research project investigating whether sentiment shocks from financial news about semiconductor companies, combined with commodity market volatility, can predict short-term movements in semiconductor equity volatility.

**Timeline**: 3-Day Research Sprint  
**Status**: Active Development  
**Date**: January 31, 2026

## Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n volatility-research python=3.10 -y
conda activate volatility-research

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys:
# - NewsAPI: https://newsapi.org/
# - Alpha Vantage: https://www.alphavantage.co/
```

### 3. Run Setup
```bash
# Create directory structure and validate config
python setup_project.py
```

### 4. Start Data Collection
```bash
# Run Day 1: Data collection
python scripts/01_collect_data.py
```

## Project Structure
```
.
├── config.py                 # Configuration and parameters
├── setup_project.py          # Automated setup script
├── requirements.txt          # Python dependencies
├── data/                     # Data storage (not in git)
│   ├── raw/                  # Original downloaded data
│   │   ├── news/             # News headlines
│   │   ├── prices/           # Equity prices
│   │   └── commodities/      # Commodity futures
│   └── processed/            # Cleaned, aligned data
├── src/                      # Reusable Python modules
│   ├── data_collection.py    # API wrappers
│   ├── sentiment_analysis.py # FinBERT processing
│   ├── feature_engineering.py # Volatility calculations
│   ├── dimensionality_reduction.py # PCA
│   ├── models.py             # VAR, Granger tests
│   ├── forecasting.py        # Out-of-sample prediction
│   ├── backtesting.py        # Strategy simulation
│   ├── visualization.py      # Plotting functions
│   └── utils.py              # Helper functions
├── scripts/                  # Executable analysis scripts
│   ├── 01_collect_data.py    # Day 1: Data collection
│   ├── 02_process_features.py # Day 1: Feature engineering
│   ├── 03_run_pca.py         # Day 2: PCA analysis
│   ├── 04_estimate_var.py    # Day 2: VAR modeling
│   ├── 05_forecast.py        # Day 2: Forecasting
│   ├── 06_backtest.py        # Day 3: Backtesting
│   └── 07_create_plots.py    # Day 3: Visualization
├── results/                  # Output artifacts (not in git)
│   ├── figures/              # Plots and visualizations
│   │   ├── eda/              # Exploratory analysis
│   │   ├── pca/              # PCA visualizations
│   │   ├── var/              # VAR model results
│   │   └── backtest/         # Backtest performance
│   ├── tables/               # CSV result tables
│   └── models/               # Saved model objects
└── docs/                     # Documentation
    ├── FINDINGS.md           # Running notes
    └── research_paper.pdf    # Final deliverable

## Methodology

### Phase 1: Data Collection (Day 1)
- **News Sentiment**: FinBERT analysis of semiconductor company headlines
- **Equity Data**: SMH, SOXX ETF prices and volatility
- **Commodity Data**: WTI Crude, Natural Gas, Copper futures
- **Time Period**: 2023-2024 (2 years, ~500 trading days)

### Phase 2: Feature Engineering (Day 1)
- **Volatility Metrics**: 21-day rolling realized volatility
- **Sentiment Scores**: Daily aggregated sentiment per ticker
- **Lagged Features**: 1-day and 5-day sentiment lags

### Phase 3: Statistical Modeling (Day 2)
- **PCA**: Dimensionality reduction on sentiment-commodity factors
- **VAR Model**: Vector autoregression with lag selection
- **Granger Causality**: Test if sentiment/commodities predict volatility
- **IRF Analysis**: Impulse response functions

### Phase 4: Forecasting (Day 2)
- **Horizons**: 1-day, 5-day, 10-day ahead predictions
- **Validation**: Rolling-window out-of-sample forecasts
- **Baselines**: Random walk, historical average, AR(1)

### Phase 5: Backtesting (Day 3)
- **Strategy**: Long/short volatility based on forecasts
- **Transaction Costs**: 10 bps per trade
- **Metrics**: Sharpe ratio, max drawdown, win rate

## Key Technologies
- **Language**: Python 3.10
- **NLP**: FinBERT (transformers, PyTorch)
- **Econometrics**: statsmodels (VAR, Granger tests)
- **ML**: scikit-learn (PCA, preprocessing)
- **Data Collection**: yfinance, NewsAPI
- **Visualization**: matplotlib, seaborn

## Research Questions
1. **H1**: Do sentiment shocks Granger-cause changes in semiconductor volatility?
2. **H2**: Does commodity volatility provide incremental predictive power?
3. **H3**: Can sentiment-commodity factors enable profitable volatility trading?

## Success Metrics
- ✅ At least 1 Granger causality result with p < 0.05
- ✅ Out-of-sample RMSE beats random walk by ≥10%
- ✅ Backtest Sharpe ratio > 0.5 (if edge exists)
- ✅ Complete reproducible codebase
- ✅ 15-25 page research paper

## Timeline
| Day | Focus | Deliverables |
|-----|-------|--------------|
| **Day 1** | Data Foundation | Raw data, sentiment scores, master_features.csv |
| **Day 2** | Statistical Analysis | PCA loadings, VAR model, Granger tests, forecasts |
| **Day 3** | Application | Backtest results, all figures, research paper |

## Documentation
- **PRD.md**: Complete product requirements document
- **Setup.md**: Detailed technology stack setup guide
- **FINDINGS.md**: Running notes and observations
- **research_paper.pdf**: Final research deliverable

## License
MIT License

## Contact
Parker Jahn  
Winter Park, Florida, US
