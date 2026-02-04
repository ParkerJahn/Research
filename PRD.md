Product Requirements Document (PRD)
Sentiment-Driven Volatility Prediction Research Platform

Document Information
FieldValueProject NameSentiment-Commodity Factor Analysis for Semiconductor Volatility PredictionVersion1.0DateJanuary 31, 2026Author[Your Name]StatusActive DevelopmentTimeline3-Day SprintDeliverable TypeResearch Project + Trading Strategy Backtest

1. Executive Summary
1.1 Project Overview
This research project investigates whether sentiment shocks from financial news about semiconductor companies (NVIDIA, AMD, Intel, TSMC, Micron), combined with volatility in commodity markets (oil, natural gas, copper), can predict short-term movements in semiconductor equity volatility. The project will employ NLP-based sentiment analysis, dimensionality reduction via PCA, and vector autoregression (VAR) to forecast volatility metrics, culminating in a backtested trading strategy.
1.2 Business Objective

Primary Goal: Produce a complete, publishable research paper demonstrating a novel volatility forecasting approach
Secondary Goal: Validate whether sentiment-commodity factors provide actionable trading signals
Tertiary Goal: Create a reproducible, open-source research pipeline for financial ML applications

1.3 Success Metrics

✅ Complete working data pipeline (collection → processing → modeling → backtesting)
✅ Statistical significance: At least 1 Granger causality result with p < 0.05
✅ Forecast improvement: Out-of-sample RMSE at least 10% better than random walk baseline
✅ Backtest validation: Sharpe ratio > 0.5 (if positive edge exists) or clear explanation of when/why signals fail
✅ Reproducible codebase with documentation
✅ 15-25 page research paper with tables and figures


2. Problem Statement
2.1 Research Question
Can sentiment shocks and commodity volatility predict changes in semiconductor equity volatility better than traditional models?
2.2 Hypotheses

H1: Sentiment extracted from financial news about semiconductor companies Granger-causes changes in implied volatility
H2: Commodity market volatility (oil, gas, copper) provides incremental predictive power for semiconductor volatility beyond sentiment alone
H3: Joint sentiment-commodity factors enable profitable delta-hedged volatility trading strategies

2.3 Current Gaps

Most volatility forecasting relies on historical price data alone
Text-based sentiment is underutilized in volatility prediction
Limited research on commodity-semiconductor volatility linkages
Lack of open-source implementations combining NLP + time-series econometrics


3. Scope
3.1 In Scope
Data Sources
CategorySpecific AssetsSourceFrequencyEquity SentimentNVDA, AMD, INTC, TSM, MUNewsAPI, Alpha VantageDailyVolatility ProxiesVIX, SMH realized vol, SOXX realized volYahoo FinanceDailyCommoditiesWTI Crude (CL=F), Natural Gas (NG=F), Copper (HG=F)Yahoo FinanceDailyControl VariablesSPY returns, Treasury yields (optional)Yahoo FinanceDaily
Time Period

Primary Analysis: 2023-01-01 to 2024-12-31 (2 years, ~500 trading days)
Rationale: Recent data captures AI boom, energy volatility, supply chain normalization

Methodologies

NLP: FinBERT (ProsusAI/finbert) for sentiment scoring
Dimensionality Reduction: PCA (sklearn)
Time-Series Modeling: VAR (statsmodels)
Statistical Tests: Granger causality, ADF stationarity tests, Ljung-Box residual diagnostics
Backtesting: Simplified volatility trading strategy using VIX returns as proxy

Deliverables

Code Repository (GitHub)

Data collection scripts
Preprocessing pipeline
Modeling notebooks
Backtesting engine
Visualization scripts


Research Paper (PDF, 15-25 pages)

Introduction, literature review, methodology, results, discussion, conclusion
Minimum 6 tables, 8 figures


Supplementary Materials

README with reproduction instructions
requirements.txt
Sample data files (if permissible)



3.2 Out of Scope
Excluded from This Version

❌ Real options pricing (Black-Scholes Greeks, smile dynamics) — using realized vol instead
❌ High-frequency data (intraday) — daily only
❌ Machine learning models (LSTM, XGBoost) — staying with classical econometrics
❌ Multiple asset classes beyond semiconductors
❌ Live trading implementation or production deployment
❌ Comprehensive literature review (brief review only due to time constraint)
❌ Robustness checks across multiple sample periods (single 2023-2024 period)

Future Enhancements (Post-Sprint)

Expand to 2020-2024 with regime analysis (COVID, post-COVID, AI boom)
Incorporate actual options data (IV surface, skew, term structure)
Add earnings call transcript analysis
Test ML models (LSTM for sentiment, gradient boosting for forecasting)
Multi-asset extension (energy stocks, tech broadly)


4. User Personas
4.1 Primary User: Academic Researcher

Background: PhD student or professor in finance/economics
Goals: Publish in journal, validate new methodology, contribute to literature
Needs:

Rigorous statistical methodology
Clear documentation of data sources and transformations
Reproducible results
Proper citations and literature context


Pain Points:

Limited access to expensive data vendors
Time constraints for complex implementations
Need for novel contribution (not just replication)



4.2 Secondary User: Quantitative Trader/Analyst

Background: Works at hedge fund, prop trading firm, or asset manager
Goals: Identify alpha signals, improve volatility forecasting, reduce portfolio risk
Needs:

Practical trading strategy with realistic costs
Performance metrics (Sharpe, max drawdown, win rate)
Understanding of when signals work/fail
Code they can adapt for their own use


Pain Points:

Academic research often ignores transaction costs
Overfitting to historical data
Difficulty reproducing published results



4.3 Tertiary User: Data Science Student

Background: Learning financial ML, building portfolio
Goals: Understand end-to-end quant research workflow, learn best practices
Needs:

Well-commented code
Clear explanations of each step
Modular design they can extend


Pain Points:

Overwhelmed by complex codebases
Unclear how to get started with financial data
Gap between theory and implementation




5. Functional Requirements
5.1 Data Collection Module
FR-1.1: News Sentiment Data

Description: Collect financial news headlines for target tickers
Acceptance Criteria:

 Retrieve headlines for NVDA, AMD, INTC, TSM, MU from 2023-2024
 Minimum 50 headlines per ticker (total ~250-500 headlines)
 Each record includes: date, ticker, headline text, source
 Data stored in CSV format: sentiment_raw_data.csv


Data Schema:

  date (YYYY-MM-DD), ticker (str), headline (str), source (str)

Error Handling:

If API rate limit hit, implement exponential backoff
If ticker has <10 headlines, log warning but continue



FR-1.2: Equity Price Data

Description: Download historical OHLCV data for SMH, SOXX
Acceptance Criteria:

 Daily data from 2023-01-01 to 2024-12-31
 No missing dates (forward-fill holidays)
 Adjusted for splits/dividends
 Files: SMH_price_history.csv, SOXX_price_history.csv


Data Schema:

  date, open, high, low, close, volume, adj_close
FR-1.3: Commodity Futures Data

Description: Download WTI crude, natural gas, copper futures prices
Acceptance Criteria:

 Daily closing prices 2023-2024
 Files: WTI_history.csv, NatGas_history.csv, Copper_history.csv
 VIX index included: VIX_history.csv


Data Schema: Same as FR-1.2

FR-1.4: Data Validation

Description: Automated checks for data quality
Acceptance Criteria:

 No more than 5% missing values per series
 No duplicate dates
 Price data >0 (no negative prices)
 Date ranges align across all datasets
 Validation report saved to data_quality_report.txt



5.2 Feature Engineering Module
FR-2.1: Sentiment Scoring

Description: Apply FinBERT to headlines and aggregate to daily scores
Acceptance Criteria:

 Each headline scored with [positive, negative, neutral] probabilities
 Daily aggregation: average sentiment per ticker
 Composite sector sentiment: weighted average across all tickers
 Sentiment index: (positive - negative) ∈ [-1, 1]
 Output: daily_sentiment_scores.csv


Data Schema:

  date, nvda_sent, amd_sent, intc_sent, tsm_sent, mu_sent, sector_sent

Performance: Process all headlines in <30 minutes

FR-2.2: Volatility Metrics

Description: Calculate realized volatility from price data
Acceptance Criteria:

 21-day rolling realized volatility (annualized)
 Formula: σ = std(log_returns) × √252
 Calculated for: SMH, SOXX, WTI, NatGas, Copper
 Output: realized_volatility.csv


Data Schema:

  date, smh_vol, soxx_vol, wti_vol, natgas_vol, copper_vol
FR-2.3: Feature Matrix Assembly

Description: Merge all features into aligned dataset
Acceptance Criteria:

 All data aligned by date
 Weekends/holidays handled (forward-fill commodity data)
 Lagged features created: sentiment_t-1, sentiment_t-5
 Missing values <10% after alignment
 Output: master_features.csv with ~500 rows, ~15 columns


Data Schema:

  date, smh_vol, soxx_vol, vix, wti_vol, natgas_vol, copper_vol,
  nvda_sent, amd_sent, intc_sent, tsm_sent, mu_sent, sector_sent,
  smh_return, soxx_return, vix_change
5.3 Dimensionality Reduction Module
FR-3.1: Feature Standardization

Description: Z-score normalization of all features
Acceptance Criteria:

 Each feature has mean=0, std=1
 Scaler fit only on training data (2023)
 Same scaler applied to test data (2024)
 Scaler saved to disk: feature_scaler.pkl



FR-3.2: PCA Execution

Description: Perform principal component analysis
Acceptance Criteria:

 Retain components explaining 95% of variance
 Expected output: 3-5 principal components
 Scree plot generated showing variance explained
 Loadings matrix saved: pca_loadings.csv
 Component scores saved: pca_components.csv


Output Schema:

  # pca_components.csv
  date, PC1, PC2, PC3, PC4, ...
  
  # pca_loadings.csv
  feature, PC1_loading, PC2_loading, PC3_loading, ...
FR-3.3: Factor Interpretation

Description: Name factors based on loadings
Acceptance Criteria:

 Each PC has descriptive name (e.g., "Sentiment Factor", "Commodity Stress")
 Interpretation documented in pca_interpretation.md
 Visualization: heatmap of loadings saved to loadings_heatmap.png



5.4 Statistical Modeling Module
FR-4.1: Stationarity Testing

Description: Check if time series are stationary
Acceptance Criteria:

 Augmented Dickey-Fuller (ADF) test on all variables
 If non-stationary (p>0.05), apply first-differencing
 Stationarity results saved to stationarity_tests.csv


Output Schema:

  variable, adf_statistic, p_value, is_stationary, transformation_applied
FR-4.2: VAR Model Estimation

Description: Fit vector autoregression model
Acceptance Criteria:

 Endogenous variables: SMH_vol, SOXX_vol, VIX
 Exogenous variables: PC1, PC2, PC3
 Lag order selection via AIC/BIC (test lags 1-10)
 Model fit on training data (2023)
 Coefficient table saved: var_coefficients.csv
 Model object saved: var_model.pkl


Output Schema:

  # var_coefficients.csv
  equation, variable, lag, coefficient, std_error, t_stat, p_value
FR-4.3: Model Diagnostics

Description: Validate VAR model assumptions
Acceptance Criteria:

 Residual autocorrelation test (Ljung-Box, p>0.05 desired)
 Residual heteroskedasticity test (ARCH test)
 Residual normality test (Jarque-Bera)
 All diagnostic plots saved to diagnostics/ folder
 Diagnostic summary: model_diagnostics.txt



FR-4.4: Granger Causality Tests

Description: Test if PCs Granger-cause volatility
Acceptance Criteria:

 Test each PC → each volatility variable (9 tests total)
 Lags tested: 1, 3, 5
 Results table with F-statistics and p-values
 Output: granger_causality_results.csv


Output Schema:

  cause, effect, lag, f_statistic, p_value, significant (bool)

Success Criteria: At least 1 test with p < 0.05

FR-4.5: Impulse Response Functions

Description: Trace impact of shocks over time
Acceptance Criteria:

 1-std shock to each PC
 IRF calculated for 10 periods ahead
 95% confidence intervals included
 Plots saved for each PC-volatility pair: irf_plots/
 IRF data saved: irf_results.csv



5.5 Forecasting Module
FR-5.1: Out-of-Sample Forecast Generation

Description: Generate rolling-window forecasts on test data
Acceptance Criteria:

 Forecast horizon: 1-day, 5-day, 10-day ahead
 Re-estimate model every 20 trading days
 Forecasts generated for all of 2024
 Output: forecasts.csv


Output Schema:

  date, target_variable, forecast_horizon, actual_value, predicted_value
FR-5.2: Forecast Evaluation

Description: Calculate forecast accuracy metrics
Acceptance Criteria:

 Metrics: RMSE, MAE, MAPE, directional accuracy
 Calculated for each horizon (1, 5, 10 days)
 Comparison to baselines: random walk, historical average, AR(1)
 Output: forecast_performance.csv


Output Schema:

  model, horizon, rmse, mae, mape, directional_accuracy

Success Criteria: VAR+PCA model outperforms random walk by ≥10% on RMSE

FR-5.3: Factor Attribution

Description: Decompose forecast variance by factor
Acceptance Criteria:

 Variance decomposition at 1-day, 5-day, 10-day horizons
 % variance explained by each PC
 Output: variance_decomposition.csv
 Visualization: stacked area chart saved to variance_decomp.png



5.6 Backtesting Module
FR-6.1: Trading Signal Generation

Description: Convert forecasts to trading signals
Acceptance Criteria:

 Signal logic:

Long volatility if forecast_vol > current_vol × (1 + threshold)
Short volatility if forecast_vol < current_vol × (1 - threshold)
Neutral otherwise


 Default threshold: 2%
 Signals generated for all test period dates
 Output: trading_signals.csv


Output Schema:

  date, current_vol, forecast_vol, signal (1/0/-1), threshold_used
FR-6.2: Strategy Simulation

Description: Backtest volatility trading strategy
Acceptance Criteria:

 Position sizing: Fixed notional ($10,000 per trade)
 Returns proxy: signal × VIX daily return
 Transaction costs: 10 bps per trade
 Daily P&L calculation
 Cumulative returns tracked
 Output: backtest_pnl.csv, trade_log.csv


Output Schema:

  # backtest_pnl.csv
  date, signal, vix_return, gross_pnl, transaction_cost, net_pnl, cumulative_return
  
  # trade_log.csv
  trade_id, entry_date, exit_date, position, entry_vol, exit_vol, pnl
FR-6.3: Performance Metrics

Description: Calculate strategy performance statistics
Acceptance Criteria:

 Total return (%)
 Annualized return
 Sharpe ratio (rf=0 assumption)
 Maximum drawdown
 Win rate (% profitable trades)
 Average win/loss ratio
 Output: strategy_performance.csv


Output Schema:

  metric, value
  total_return, X.XX%
  sharpe_ratio, Y.YY
  ...
FR-6.4: Sensitivity Analysis

Description: Test strategy across parameter variations
Acceptance Criteria:

 Thresholds tested: 1%, 2%, 3%, 5%
 Transaction costs tested: 5bps, 10bps, 20bps
 Performance matrix: threshold × transaction cost
 Output: sensitivity_results.csv
 Heatmap visualization: sensitivity_heatmap.png



5.7 Visualization Module
FR-7.1: Exploratory Data Analysis Plots

Acceptance Criteria:

 Correlation heatmap (all features)
 Time series plots (volatility + sentiment)
 Distribution histograms (each feature)
 Saved to eda_plots/ folder



FR-7.2: PCA Visualizations

Acceptance Criteria:

 Scree plot (variance explained)
 Loadings heatmap
 Component time series (PC1, PC2, PC3 over time)
 Biplot (first 2 PCs)



FR-7.3: Model Results Visualizations

Acceptance Criteria:

 Granger causality heatmap (p-values)
 IRF plots (3 PCs × 3 targets = 9 plots)
 Forecast vs actual (time series, separate plot per horizon)
 Residual diagnostics (ACF, QQ plot, histogram)



FR-7.4: Backtest Visualizations

Acceptance Criteria:

 Equity curve (cumulative returns over time)
 Drawdown chart
 Monthly returns heatmap
 Win/loss distribution histogram
 Sensitivity heatmap




6. Non-Functional Requirements
6.1 Performance
NFR-1.1: Execution Time

Data collection: <2 hours total
Sentiment processing: <30 minutes for all headlines
PCA + VAR estimation: <10 minutes
Backtesting: <5 minutes
Total pipeline runtime: <4 hours (excluding manual steps)

NFR-1.2: Memory Usage

Peak memory: <8 GB (laptop-compatible)
No GPU required (CPU-only execution)

6.2 Scalability
NFR-2.1: Data Volume

Must handle 500-1000 trading days
Support 10-20 features without code changes
Extend to 5-10 tickers with config file modification only

NFR-2.2: Model Complexity

VAR with up to 10 lags
PCA with up to 20 input features

6.3 Reliability
NFR-3.1: Error Handling

All API calls wrapped in try-except with retry logic
Graceful degradation if data source unavailable
Logging: All errors written to error_log.txt

NFR-3.2: Data Integrity

Checksums for downloaded files (MD5 hashing)
Automated data validation before each modeling step
Version control for all data transformations (log transformations applied)

6.4 Usability
NFR-4.1: Code Quality

PEP 8 compliant Python code
Docstrings for all functions (Google style)
Type hints for function signatures
Maximum function length: 50 lines
Maximum cyclomatic complexity: 10

NFR-4.2: Documentation

README with setup instructions (5 steps max)
Inline comments for complex logic
Jupyter notebooks with markdown explanations
API reference (auto-generated from docstrings)

NFR-4.3: Reproducibility

Fixed random seeds for all stochastic processes
requirements.txt with exact versions
Docker container (optional, post-sprint)
Reproduction instructions tested on clean environment

6.5 Maintainability
NFR-5.1: Modularity

Separate scripts for each phase (collection, processing, modeling, backtest)
Config file for all parameters (no hard-coded values)
Utility functions in dedicated utils.py

NFR-5.2: Version Control

Git repository with meaningful commit messages
.gitignore for data files (too large for GitHub)
Branching strategy: main (stable), dev (active work)

6.6 Security & Compliance
NFR-6.1: API Keys

Never commit API keys to repository
Use environment variables or config file (in .gitignore)
Instructions for users to obtain their own keys

NFR-6.2: Data Privacy

No personally identifiable information (PII) in datasets
Public financial data only (no proprietary datasets)

NFR-6.3: Licensing

MIT License for code (open-source)
Proper attribution for data sources
Citation for FinBERT model


7. Technical Architecture
7.1 System Architecture
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                               │
├─────────────────────────────────────────────────────────────┤
│  External APIs          Local Storage        Processed Data  │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │ NewsAPI      │────▶│ raw/         │────▶│ processed/  │ │
│  │ Yahoo Finance│     │  - news.csv  │     │  - master_  │ │
│  │ Alpha Vantage│     │  - prices.csv│     │    features │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ NLP Module   │  │ Feature Eng  │  │ Validation       │  │
│  │ - FinBERT    │  │ - Vol Calc   │  │ - Quality Checks │  │
│  │ - Sentiment  │  │ - Alignment  │  │ - Stationarity   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ANALYTICS LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ PCA          │  │ VAR Model    │  │ Forecasting      │  │
│  │ - Dim Reduce │  │ - Estimation │  │ - OOS Predict    │  │
│  │ - Interpret  │  │ - Granger    │  │ - Evaluation     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Backtesting  │  │ Visualization│  │ Reporting        │  │
│  │ - Signals    │  │ - Plots      │  │ - Tables         │  │
│  │ - P&L Calc   │  │ - Dashboards │  │ - LaTeX Export   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
7.2 Technology Stack
LayerTechnologyVersionPurposeLanguagePython3.9+Core developmentEnvironmentJupyter Notebook-Interactive analysisNLPTransformers (HuggingFace)4.35+FinBERT sentimentPyTorch2.0+Model backendData ProcessingPandas2.0+Data manipulationNumPy1.24+Numerical computingStatisticsStatsmodels0.14+VAR, Granger testsSciPy1.11+Statistical functionsMachine Learningscikit-learn1.3+PCA, preprocessingVisualizationMatplotlib3.7+Static plotsSeaborn0.12+Statistical graphicsPlotly5.17+Interactive charts (optional)Data Sourcesyfinance0.2.28+Yahoo Finance APInewsapi-python0.2.7+News APIalpha_vantage2.3.1+Alternative dataUtilitiesrequests2.31+HTTP requestspython-dotenv1.0+Environment variablesVersion ControlGit2.40+Code versioningDocumentationSphinx7.1+API docs (optional)
7.3 Directory Structure
sentiment-volatility-research/
│
├── data/
│   ├── raw/                          # Original downloaded data
│   │   ├── news/
│   │   │   └── sentiment_raw_data.csv
│   │   ├── prices/
│   │   │   ├── SMH_price_history.csv
│   │   │   └── SOXX_price_history.csv
│   │   └── commodities/
│   │       ├── WTI_history.csv
│   │       └── VIX_history.csv
│   │
│   └── processed/                    # Cleaned, aligned data
│       ├── daily_sentiment_scores.csv
│       ├── realized_volatility.csv
│       └── master_features.csv
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_pca_analysis.ipynb
│   ├── 05_var_modeling.ipynb
│   ├── 06_forecasting.ipynb
│   └── 07_backtesting.ipynb
│
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── config.py                     # Configuration parameters
│   ├── data_collection.py            # API wrappers
│   ├── sentiment_analysis.py         # FinBERT processing
│   ├── feature_engineering.py        # Vol calc, alignment
│   ├── dimensionality_reduction.py   # PCA functions
│   ├── models.py                     # VAR estimation, Granger
│   ├── forecasting.py                # OOS prediction
│   ├── backtesting.py                # Strategy simulation
│   ├── visualization.py              # Plotting functions
│   └── utils.py                      # Helper functions
│
├── results/                          # Output artifacts
│   ├── figures/                      # All plots
│   │   ├── eda/
│   │   ├── pca/
│   │   ├── models/
│   │   └── backtest/
│   ├── tables/                       # CSV results
│   │   ├── pca_loadings.csv
│   │   ├── granger_causality_results.csv
│   │   └── forecast_performance.csv
│   └── models/                       # Saved model objects
│       ├── var_model.pkl
│       └── feature_scaler.pkl
│
├── tests/                            # Unit tests (optional)
│   ├── test_data_collection.py
│   ├── test_feature_engineering.py
│   └── test_models.py
│
├── docs/                             # Documentation
│   ├── setup_guide.md
│   ├── api_reference.md
│   └── research_paper.pdf            # Final deliverable
│
├── .env.example                      # Template for API keys
├── .gitignore                        # Git ignore rules
├── README.md                         # Project overview
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment (optional)
└── LICENSE                           # MIT License
7.4 Data Flow Diagram
[NewsAPI/Yahoo] → [Raw CSVs] → [Sentiment Analysis] → [Feature Matrix]
                                                            ↓
                                                       [PCA Transform]
                                                            ↓
                    [Forecasts] ← [VAR Model] ← [Principal Components]
                         ↓
                    [Signals] → [Backtest] → [Performance Metrics]

8. Data Schema Specifications
8.1 Input Data Schemas
News Headlines (sentiment_raw_data.csv)
python{
    'date': 'datetime64[ns]',       # YYYY-MM-DD format
    'ticker': 'str',                # NVDA, AMD, INTC, TSM, MU
    'headline': 'str',              # News headline text
    'source': 'str'                 # NewsAPI, AlphaVantage, etc.
}
Price History (SMH_price_history.csv)
python{
    'date': 'datetime64[ns]',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'adj_close': 'float64',
    'volume': 'int64'
}
8.2 Processed Data Schemas
Master Features (master_features.csv)
python{
    'date': 'datetime64[ns]',
    
    # Volatility metrics
    'smh_vol': 'float64',           # 21-day realized vol
    'soxx_vol': 'float64',
    'vix': 'float64',
    'wti_vol': 'float64',
    'natgas_vol': 'float64',
    'copper_vol': 'float64',
    
    # Sentiment scores
    'nvda_sent': 'float64',         # Range: [-1, 1]
    'amd_sent': 'float64',
    'intc_sent': 'float64',
    'tsm_sent': 'float64',
    'mu_sent': 'float64',
    'sector_sent': 'float64',
    
    # Returns
    'smh_return': 'float64',
    'soxx_return': 'float64',
    'vix_change': 'float64'
}
PCA Components (pca_components.csv)
python{
    'date': 'datetime64[ns]',
    'PC1': 'float64',
    'PC2': 'float64',
    'PC3': 'float64',
    # ... up to PCn
}
8.3 Output Data Schemas
Granger Causality Results (granger_causality_results.csv)
python{
    'cause': 'str',                 # PC1, PC2, PC3
    'effect': 'str',                # SMH_vol, SOXX_vol, VIX
    'lag': 'int',                   # 1, 3, 5
    'f_statistic': 'float64',
    'p_value': 'float64',
    'significant': 'bool'           # p < 0.05
}
Forecast Performance (forecast_performance.csv)
python{
    'model': 'str',                 # VAR+PCA, RandomWalk, AR1
    'horizon': 'int',               # 1, 5, 10 days
    'rmse': 'float64',
    'mae': 'float64',
    'mape': 'float64',              # Mean absolute % error
    'directional_accuracy': 'float64'  # [0, 1]
}
Backtest P&L (backtest_pnl.csv)
python{
    'date': 'datetime64[ns]',
    'signal': 'int',                # -1, 0, 1
    'vix_return': 'float64',
    'gross_pnl': 'float64',         # Before costs
    'transaction_cost': 'float64',
    'net_pnl': 'float64',           # After costs
    'cumulative_return': 'float64'
}

9. Interface Specifications
9.1 API Interfaces
NewsAPI Configuration
python# config.py
NEWS_API_CONFIG = {
    'api_key': os.getenv('NEWS_API_KEY'),
    'endpoint': 'https://newsapi.org/v2/everything',
    'params': {
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 100
    }
}
Yahoo Finance Configuration
pythonYFINANCE_CONFIG = {
    'tickers': ['SMH', 'SOXX', 'CL=F', 'NG=F', 'HG=F', '^VIX'],
    'start_date': '2023-01-01',
    'end_date': '2024-12-31',
    'interval': '1d'
}
9.2 Function Signatures
Data Collection
pythondef fetch_news_headlines(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str
) -> pd.DataFrame:
    """
    Fetch news headlines from NewsAPI.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'NVDA')
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        api_key: NewsAPI key
        
    Returns:
        DataFrame with columns: [date, ticker, headline, source]
    """
    pass
Sentiment Analysis
pythondef calculate_sentiment(
    headlines: pd.DataFrame,
    model_name: str = 'ProsusAI/finbert'
) -> pd.DataFrame:
    """
    Apply FinBERT to headlines and aggregate to daily sentiment.
    
    Args:
        headlines: DataFrame with 'date' and 'headline' columns
        model_name: HuggingFace model identifier
        
    Returns:
        DataFrame with columns: [date, ticker, sentiment_score]
        sentiment_score range: [-1, 1]
    """
    pass
PCA
pythondef perform_pca(
    features: pd.DataFrame,
    variance_threshold: float = 0.95
) -> Tuple[np.ndarray, PCA, pd.DataFrame]:
    """
    Perform PCA on feature matrix.
    
    Args:
        features: Standardized feature matrix
        variance_threshold: Cumulative variance to retain
        
    Returns:
        Tuple of:
        - components: Principal component scores (n_samples, n_components)
        - pca_model: Fitted PCA object
        - loadings: DataFrame of feature loadings
    """
    pass
VAR Modeling
pythondef estimate_var(
    endog_data: pd.DataFrame,
    exog_data: Optional[pd.DataFrame] = None,
    maxlags: int = 10
) -> VARResults:
    """
    Estimate VAR model with lag order selection.
    
    Args:
        endog_data: Endogenous variables (targets)
        exog_data: Exogenous variables (predictors)
        maxlags: Maximum lags to test
        
    Returns:
        Fitted VARResults object
    """
    pass
Backtesting
pythondef backtest_strategy(
    signals: pd.Series,
    returns: pd.Series,
    transaction_cost: float = 0.001
) -> Dict[str, float]:
    """
    Backtest volatility trading strategy.
    
    Args:
        signals: Trading signals (-1, 0, 1)
        returns: Asset returns (VIX daily returns)
        transaction_cost: Bps per trade (default 10bps)
        
    Returns:
        Dictionary with performance metrics:
        - total_return
        - sharpe_ratio
        - max_drawdown
        - win_rate
    """
    pass

10. Testing & Validation Strategy
10.1 Unit Tests
Critical Functions to Test

 fetch_news_headlines(): Mock API, verify schema
 calculate_sentiment(): Test on sample headlines, check range
 calculate_realized_volatility(): Verify formula, edge cases (zeros, NaN)
 perform_pca(): Check variance explained sums to ≤1.0
 granger_causality_test(): Verify p-value ranges

10.2 Integration Tests

 End-to-end pipeline: raw data → forecasts
 Data alignment: Ensure dates match across all datasets
 Model persistence: Save/load models without errors

10.3 Validation Checks
Data Quality Checks (Automated)
pythondef validate_data_quality(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Run automated data quality checks.
    
    Checks:
    - Missing values <10%
    - No duplicate dates
    - Price data >0
    - Date ranges align
    """
    checks = {
        'missing_ok': df.isnull().sum().sum() / df.size < 0.10,
        'no_duplicates': not df.index.duplicated().any(),
        'prices_positive': (df.select_dtypes(include='number') > 0).all().all(),
        'date_range_valid': df.index.is_monotonic_increasing
    }
    return checks
```

#### Model Diagnostics (Manual Review)
- [ ] Residual ACF plots show no autocorrelation
- [ ] Q-Q plots approximately linear
- [ ] Loadings interpretable (clear factor structure)
- [ ] IRFs decay to zero within 10 periods

### 10.4 Acceptance Testing

#### Minimum Viable Results
- [ ] ✅ At least 1 Granger causality p < 0.05
- [ ] ✅ Out-of-sample RMSE beats random walk
- [ ] ✅ Backtest Sharpe ratio >0 (even if not statistically significant)
- [ ] ✅ All code runs without errors on clean environment

#### Quality Thresholds
- [ ] Code coverage >50% (if unit tests implemented)
- [ ] Documentation: Every function has docstring
- [ ] Reproducibility: External tester can reproduce results

---

## 11. Risk Assessment & Mitigation

### 11.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **API rate limits hit** | High | Medium | Implement backoff, cache data, use multiple APIs |
| **Insufficient historical options data** | High | High | **PIVOT to VIX + realized vol as proxies** |
| **FinBERT model too slow** | Medium | Low | Batch processing, use smaller model if needed |
| **VAR model unstable/non-stationary** | Medium | Medium | Differencing, shorter sample period, use VECM |
| **No significant Granger causality** | Medium | High | Still publishable (negative result), explore different lags |
| **Overfitting in backtest** | Medium | Medium | Use walk-forward validation, test multiple thresholds |

### 11.2 Data Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Missing data >10%** | Medium | Medium | Forward-fill, linear interpolation, reduce sample period |
| **Low-quality news headlines** | Low | Medium | Filter by source credibility, manual review sample |
| **Data misalignment (dates)** | Low | High | Automated validation checks, visual inspection |

### 11.3 Scope Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Underestimated time requirements** | High | High | **3-day sprint already accounts for this by using simplified approach** |
| **Scope creep (adding features)** | Medium | Medium | Strict adherence to in-scope items, park ideas for future work |
| **Insufficient results for publication** | Medium | High | Focus on methodology contribution, negative results valid |

### 11.4 Contingency Plans

#### If NewsAPI doesn't provide enough data:
- **Fallback**: Use existing sentiment dataset from Kaggle (e.g., "Financial Sentiment Analysis")
- **Fallback 2**: Web scrape Yahoo Finance or SeekingAlpha (respect robots.txt)

#### If VAR shows no predictive power:
- **Alternative**: Switch to simpler linear regression forecasts
- **Alternative 2**: Focus paper on "limitations of sentiment for volatility prediction"

#### If backtesting shows losses:
- **Not a failure**: Document conditions where strategy fails
- **Analysis**: Time-varying performance (when does it work?)

---

## 12. Timeline & Milestones

### 12.1 Detailed Schedule

#### **Day 1: Data Foundation** (10 hours)
| Time Block | Duration | Tasks | Deliverables |
|------------|----------|-------|--------------|
| **Morning** | 4h | API setup, data downloads | Raw CSV files |
| **Afternoon** | 4h | Sentiment analysis, feature engineering | daily_sentiment_scores.csv, master_features.csv |
| **Evening** | 2h | Data cleaning, EDA | eda_report.html, correlation heatmap |

**Day 1 Milestone**: ✅ Clean, aligned dataset ready for modeling

---

#### **Day 2: Statistical Analysis** (12 hours)
| Time Block | Duration | Tasks | Deliverables |
|------------|----------|-------|--------------|
| **Morning** | 4h | PCA, VAR estimation | pca_loadings.csv, var_model.pkl |
| **Afternoon** | 4h | Granger tests, IRF analysis | granger_causality_results.csv, IRF plots |
| **Evening** | 4h | Forecasting, evaluation | forecasts.csv, forecast_performance.csv |

**Day 2 Milestone**: ✅ Statistical models estimated with diagnostic checks passed

---

#### **Day 3: Application & Documentation** (12 hours)
| Time Block | Duration | Tasks | Deliverables |
|------------|----------|-------|--------------|
| **Morning** | 4h | Backtesting, performance metrics | backtest_pnl.csv, equity_curve.png |
| **Afternoon** | 4h | All visualizations, results tables | figures/, tables/ |
| **Evening** | 4h | Write research paper, GitHub cleanup | research_paper.pdf, README.md |

**Day 3 Milestone**: ✅ Complete research package ready for submission/sharing

---

### 12.2 Critical Path
```
Data Collection → Feature Engineering → PCA → VAR → Forecasting → Backtesting → Paper
     (Day 1)          (Day 1)         (Day 2)  (Day 2)  (Day 2)      (Day 3)     (Day 3)
Bottleneck Identification:

Critical path: Sentiment analysis (if slow, entire pipeline delayed)
Mitigation: Start FinBERT processing early, run overnight if needed

12.3 Quality Gates
GateCriteriaAction if FailedGate 1 (End of Day 1)master_features.csv exists, <10% missingReduce sample period or simplify featuresGate 2 (End of Day 2 Morning)VAR model converges, residuals pass diagnosticsDifference variables, reduce lagsGate 3 (End of Day 2)At least 1 significant Granger result OR clear negative findingProceed with current results, focus on interpretationGate 4 (End of Day 3 Morning)Backtest runs without errorsSimplify strategy if needed

13. Success Criteria
13.1 Minimum Viable Product (MVP)
Required for successful 3-day completion:
✅ Data Pipeline

 500+ trading days of aligned data (2023-2024)
 Sentiment scores for at least 3 tickers
 Volatility metrics for SMH/SOXX + VIX

✅ Statistical Analysis

 PCA with 3-5 components, interpretable loadings
 VAR model with lag selection and diagnostics
 At least 3 Granger causality tests performed (even if not significant)

✅ Forecasting

 Out-of-sample forecasts generated for all test period
 Performance metrics calculated vs baseline

✅ Backtesting

 Trading strategy simulated with realistic costs
 Performance metrics reported

✅ Documentation

 Code repository with README
 Research paper draft (15+ pages)
 Minimum 6 figures, 4 tables

13.2 Stretch Goals (Nice-to-Have)
⭐ Enhanced Analysis

 Machine learning comparison (LSTM forecast vs VAR)
 Regime-dependent analysis (high/low VIX periods)
 Multiple strategy variations tested

⭐ Presentation

 Interactive dashboard (Plotly/Dash)
 Presentation slides for 15-min research talk
 Blog post summary

13.3 Success Metrics by Stakeholder
For Academic Researcher

✅ Methodology is sound and reproducible
✅ Results presented with appropriate statistical rigor
✅ Contribution is clear (even if results are negative)

For Quantitative Trader

✅ Strategy has clear entry/exit rules
✅ Transaction costs are realistic
✅ Performance attribution is transparent

For Data Science Student

✅ Code is well-commented and modular
✅ Each step is explained with rationale
✅ They can extend it for their own projects


14. Assumptions & Dependencies
14.1 Assumptions
Data Availability

NewsAPI provides sufficient historical headlines (2023-2024)
Yahoo Finance data is complete and accurate
VIX is an acceptable proxy for semiconductor volatility

Technical

FinBERT sentiment scores correlate with market-moving news
21-day realized volatility is a reasonable forecast target
Daily frequency is appropriate (not too coarse)

Statistical

Volatility is at least weakly predictable (not pure random walk)
Sentiment and commodity factors have some information content
VAR is an appropriate model (linearity assumption holds)

Strategic

VIX returns are a valid proxy for volatility P&L
10 bps transaction cost is realistic for retail trader
Daily rebalancing is feasible

14.2 Dependencies
External Services

NewsAPI uptime and rate limits
Yahoo Finance API availability
HuggingFace model hosting (FinBERT)

Software Libraries

No breaking changes in transformers, statsmodels during project

Computational Resources

Local machine has sufficient RAM (8 GB minimum)
Internet connection stable for API calls

14.3 Constraints
Time: 3-day hard deadline
Budget: $0 (free tier APIs only)
Team: Solo researcher (no collaboration)
Scope: Proof-of-concept, not production system

15. Appendices
Appendix A: Glossary
TermDefinitionATMAt-the-money; option strike price ≈ current stock priceGranger CausalityStatistical test: Does X help predict Y?IRFImpulse Response Function; traces shock impact over timePCAPrincipal Component Analysis; dimensionality reductionRealized VolatilityActual volatility computed from historical returnsSharpe RatioRisk-adjusted return metric: (return - rf) / stdVARVector Autoregression; multivariate time-series modelVIXCBOE Volatility Index; "fear gauge"
Appendix B: References
Academic Papers:

Tetlock, P. (2007). "Giving Content to Investor Sentiment" Journal of Finance
Antweiler, W. & Frank, M. (2004). "Is All That Talk Just Noise?" Journal of Finance

Technical Documentation:

FinBERT: https://huggingface.co/ProsusAI/finbert
Statsmodels VAR: https://www.statsmodels.org/stable/vector_ar.html

Data Sources:

NewsAPI: https://newsapi.org/docs
Yahoo Finance: https://python-yahoofinance.readthedocs.io/

Appendix C: Configuration Template
python# config.py

import os
from typing import Dict, List

# API Keys (store in .env file)
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')

# Date Range
START_DATE = '2023-01-01'
END_DATE = '2024-12-31'

# Tickers
SEMICONDUCTOR_TICKERS = ['NVDA', 'AMD', 'INTC', 'TSM', 'MU']
ETF_TICKERS = ['SMH', 'SOXX']
COMMODITY_TICKERS = ['CL=F', 'NG=F', 'HG=F']  # WTI, NatGas, Copper
MARKET_INDICES = ['^VIX', '^SPX']

# Feature Engineering
VOLATILITY_WINDOW = 21  # days
SENTIMENT_LAGS = [1, 5]  # 1-day, 1-week

# PCA
PCA_VARIANCE_THRESHOLD = 0.95

# VAR
VAR_MAX_LAGS = 10
VAR_IC_CRITERION = 'aic'  # or 'bic'

# Backtesting
INITIAL_CAPITAL = 10000
TRANSACTION_COST_BPS = 10
VOL_THRESHOLD = 0.02  # 2%

# File Paths
DATA_DIR = 'data/'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw/')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed/')
RESULTS_DIR = 'results/'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures/')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables/')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models/')
Appendix D: Example Notebook Structure
Notebook 1: Data Collection
markdown# 1. Data Collection

## 1.1 Setup
- Import libraries
- Load config
- Test API connections

## 1.2 News Headlines
- Fetch for each ticker
- Save to raw/

## 1.3 Price Data
- Download ETFs, commodities, VIX
- Save to raw/

## 1.4 Validation
- Check for missing dates
- Verify data ranges
- Generate data quality report
Notebook 5: VAR Modeling
markdown# 5. VAR Modeling

## 5.1 Stationarity Tests
- ADF test on each variable
- Difference if needed

## 5.2 Lag Selection
- Test lags 1-10
- AIC/BIC comparison

## 5.3 Model Estimation
- Fit VAR on training data
- Coefficient table

## 5.4 Diagnostics
- Residual ACF
- Heteroskedasticity test
- Normality test

## 5.5 Granger Causality
- Test each PC → volatility
- Results matrix

## 5.6 Impulse Responses
- Calculate IRFs
- Plot with confidence intervals

16. Approval & Sign-Off
16.1 Stakeholder Review
StakeholderRoleReview StatusDateComments[Your Name]Researcher✅ Approved2026-01-31Ready to proceed
16.2 Change Log
VersionDateAuthorChanges1.02026-01-31[Your Name]Initial PRD created
16.3 Next Steps

Immediate (Day 1 Morning):

Set up development environment
Obtain API keys
Run first data collection script


Day 1 Evening:

Review data quality checkpoint
Adjust scope if needed (e.g., reduce tickers, shorten period)


Day 2 Evening:

Validate statistical results meet MVP criteria
Make go/no-go decision on advanced analysis


Day 3 Evening:

Final review of research paper
Push to GitHub
Share with advisor/peers for feedback




Document Control
Document Status: Final
Approval Required: Self (solo project)
Distribution: Public (GitHub)
Review Cycle: N/A (3-day sprint, no revisions planned)

END OF PRODUCT REQUIREMENTS DOCUMENT