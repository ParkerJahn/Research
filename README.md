# Sentiment-Augmented Implied Volatility Forecasting

A methodologically rigorous implementation of sentiment-augmented VIX forecasting using news sentiment, PCA dimensionality reduction, and HAR-IV models with rolling cross-validation.

## Citation

If you use this code in your research, please cite:

```bibtex
@techreport{jahn2026sentiment,
  title={Sentiment-Augmented Implied Volatility Forecasting: A Methodologically Rigorous PCA-HAR Approach},
  author={Jahn, Parker},
  year={2026},
  institution={Rollins College},
  type={Working Paper}
}
```

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/sentiment-vix-forecasting.git
cd sentiment-vix-forecasting

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Access

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Alpha Vantage API key
# Get free key at: https://www.alphavantage.co/support/#api-key
```

### 3. Set Up Configuration

```bash
# Copy configuration template
cp config_template.py config.py

# (Optional) Edit config.py to adjust parameters
```

### 4. Run the Analysis

```bash
# Step 1: Collect data (WARNING: Takes ~4 days due to API rate limits)
python scripts/01_collect.py

# Step 2: Process features and sentiment
python scripts/02_process.py

# Step 3: Run PCA and forecasting models
python scripts/03_pca_varx.py

# Step 4: Generate publication-quality figures
python scripts/09_publication_figures.py
```

## Repository Structure

```
sentiment-vix-forecasting/
│
├── README.md                          # This file
├── METHODOLOGY.md                     # Detailed technical documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Files to exclude from git
├── .env.example                       # Environment variables template
├── config_template.py                 # Configuration template
│
├── src/                               # Source code modules
│   ├── data_collection.py            # Yahoo Finance + AlphaVantage APIs
│   ├── sentiment_analysis.py         # FinBERT sentiment processing
│   ├── sentiment_processing.py       # Sentiment orthogonalization
│   ├── feature_engineering.py        # Returns, volatility calculations
│   ├── pca_expanding.py              # Rolling PCA implementation
│   ├── models.py                     # HAR-IV model implementations
│   ├── unified_metrics.py            # Evaluation metrics
│   ├── benchmarks.py                 # Baseline model comparisons
│   └── utils.py                      # Helper functions
│
└── scripts/                           # Executable analysis scripts
    ├── 01_collect.py                 # Data collection pipeline
    ├── 02_process.py                 # Feature engineering
    ├── 03_pca_varx.py                # PCA + forecasting
    ├── 09_publication_figures.py     # Generate all figures
    └── create_pipeline_schematic.py  # Methodology flowchart
```

## Data Requirements

This project requires the following data sources:

### Market Data (Free - Yahoo Finance)
- **VIX**: CBOE Volatility Index (^VIX)
- **ETFs**: SMH (semiconductors), SOXX (semiconductors)
- **Commodities**: Gold (GC=F), Oil (CL=F), Copper (HG=F)
- **Date Range**: 2022-01-01 to present

### Sentiment Data (Free - Alpha Vantage API)
- **News Headlines**: Semiconductor companies (NVDA, AMD, INTC, TSM, MU)
- **API Key Required**: Free tier allows 5 requests/minute
- **Collection Time**: ~4 days for full dataset (due to rate limits)
- **Processing**: FinBERT sentiment analysis

## Expected Runtime

| Step | Description | Time |
|------|-------------|------|
| Data Collection | Yahoo Finance + Alpha Vantage | ~4 days* |
| Feature Engineering | Returns, volatility, sentiment | ~5 minutes |
| PCA + Forecasting | Rolling CV across 3 horizons | ~15 minutes |
| Figure Generation | All publication figures | ~2 minutes |

*Alpha Vantage free tier: 5 requests/minute. Can be reduced to ~2 hours with premium API.

## Results Replication

To verify your results match the paper:

### Expected Key Findings

**1-Day Horizon:**
- HAR-IV RMSE: ~2.28
- Augmented RMSE: ~1.89
- Improvement: ~17%
- Diebold-Mariano p-value: <0.001 (highly significant)

**5-Day Horizon:**
- HAR-IV RMSE: ~4.26
- Augmented RMSE: ~4.28
- Improvement: ~0% (not significant)

**22-Day Horizon:**
- HAR-IV RMSE: ~5.74
- Augmented RMSE: ~5.83
- Improvement: -1.5% (not significant)

### Verification Steps

```bash
# Run full pipeline
python scripts/03_pca_varx.py

# Check output in terminal for RMSE values
# Compare with expected values above (±0.05 tolerance)
```

## Troubleshooting

### API Rate Limit Errors
**Problem**: `429 Too Many Requests` from Alpha Vantage  
**Solution**: The script automatically handles rate limiting with 12-second delays. If you see this error, the free tier limit (5/min) may have changed. Increase delay in `config.py`.

### Missing Data
**Problem**: Gaps in sentiment data  
**Solution**: Run `scripts/08_fill_sentiment_gap.py` to interpolate missing values.

### FinBERT Model Download
**Problem**: First run downloads ~500MB FinBERT model  
**Solution**: Ensure stable internet connection. Model caches to `~/.cache/huggingface/`.

### Memory Issues
**Problem**: Out of memory during PCA  
**Solution**: Reduce `CV_TEST_SIZE` in `config.py` from 50 to 25.

## Methodology Overview

This project implements a sentiment-augmented HAR-IV (Heterogeneous Autoregressive - Implied Volatility) model:

1. **Data Collection**: Market data (VIX, ETFs, commodities) + news sentiment
2. **Sentiment Processing**: FinBERT → Orthogonalization (purge return information)
3. **Dimensionality Reduction**: Rolling PCA on 9 features → 3 principal components
4. **Forecasting**: HAR-IV baseline vs. HAR-IV + PCA + Sentiment
5. **Validation**: Rolling cross-validation (200-day train, 50-day test, 25-day step)
6. **Evaluation**: RMSE, MAE, Diebold-Mariano tests

For detailed methodology, see [METHODOLOGY.md](METHODOLOGY.md).

## Key Features

✅ **Methodologically Rigorous**: No look-ahead bias, proper cross-validation  
✅ **Reproducible**: Fixed random seeds, documented parameters  
✅ **Well-Documented**: Comprehensive docstrings, inline comments  
✅ **Publication-Ready**: Generates LaTeX-compatible figures  
✅ **Extensible**: Modular design for easy adaptation

## Adapting This Code

### Change Target Variable
Edit `config.py`:
```python
IMPLIED_VOL_TICKERS = ['^VXN']  # NASDAQ volatility instead of VIX
```

### Change Sentiment Source
Edit `config.py`:
```python
SEMICONDUCTOR_TICKERS = ['AAPL', 'MSFT', 'GOOGL']  # Tech giants
```

### Change Forecast Horizons
Edit `config.py`:
```python
FORECAST_HORIZONS = [1, 10, 30]  # 1-day, 2-week, 1-month
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

**Parker Jahn**  
Email: jahnparker90@gmail.com  
Institution: Rollins College

## Acknowledgments

- FinBERT model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- Data sources: Yahoo Finance, Alpha Vantage
- Methodology inspired by HAR-RV literature (Corsi, 2009)
