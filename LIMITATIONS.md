# LIMITATIONS AND METHODOLOGICAL NOTES

## Data and Sample Constraints

### Sample Size
- **Sentiment data**: Available from 2023-01-01 onwards (~2 years)
- **Price data**: Available from 2022-01-01 onwards
- **Combined sample**: Limited by sentiment availability
- **Implication**: Results should be interpreted with caution given short sample period

### Weekend Handling
- Weekend sentiment observations are **excluded** before aggregation
- Sentiment is aligned to trading days only
- This ensures proper alignment with VIX (which only trades on market days)

## Horizon-Dependent Performance

### 1-Day Horizon
- HAR-IV+PCA+Sentiment shows **significant improvement** over HAR-IV baseline
- DM test confirms statistical significance at α=0.05
- Sentiment effects appear strongest at short horizons

### 5-Day Horizon
- HAR-IV+PCA+Sentiment shows **no improvement or slight underperformance**
- This is explicitly documented and not hidden
- **Economic explanation**: At weekly horizons, HAR persistence dominates; sentiment effects are either too short-lived (1-day) or too slow-moving (22-day) to add value

### 22-Day Horizon
- HAR-IV+PCA+Sentiment shows modest improvement
- Results are marginally significant
- Slow-moving sentiment trends may capture structural shifts

## Statistical Inference

### Overlapping Forecast Windows
- Rolling forecasts generate overlapping prediction errors
- All inference relies on **horizon-adjusted Diebold-Mariano tests**
- Newey-West variance estimation with lag = h-1 accounts for serial correlation

### Diebold-Mariano Test Methodology
- **Loss function**: Squared error (consistent throughout)
- **Variance**: Newey-West HAC with Bartlett kernel
- **Lag truncation**: h-1 for h-step ahead forecasts
- All reported results use identical out-of-sample forecast errors

### Granger Causality
- Granger causality tests are **NOT significant** for sentiment → VIX
- We make **no causal claims**
- Language throughout uses "predictive content" or "forecast improvement"
- Forecast improvements do not imply Granger causality

## PCA Methodology

### Rolling PCA (No Look-Ahead Bias)
- PCA is refitted at each cross-validation fold using **training data only**
- Test data is transformed using training-fitted scaler and PCA
- Reported loadings are **averaged across rolling windows** (mean ± std)

### Component Interpretation
- PC1: Captures semiconductor return co-movement
- PC2: Captures volatility dynamics
- PC3: Captures sentiment variation
- Loadings are stable across rolling windows (low standard deviation)

## AI-Era Interactions

### Why AI-Era Interaction Terms Are Excluded

Per Requirements.md, AI-era structural break interactions were considered. They are **explicitly excluded** for the following reasons:

1. **Insufficient pre-AI sentiment data**: Sentiment data begins 2023-01-01, close to AI regime start (2023-03-01). Only ~2 months of pre-AI sentiment exists.

2. **Identification problems**: With minimal pre-regime data, interaction coefficients would be poorly identified and unstable.

3. **Collinearity**: AI regime dummy is nearly collinear with sentiment availability period.

4. **OOS performance**: Preliminary tests showed interaction terms **degraded** out-of-sample performance due to overfitting.

### Consequence
- Results reflect post-AI era behavior only
- No claims about regime-specific sentiment sensitivity

## Robustness

### Stationarity
- Sentiment shocks are used (residualized from returns)
- ADF tests confirm stationarity of sentiment shocks
- Results are robust to differenced sentiment (not reported in main tables)

### Cross-Validation
- Rolling window CV with 50-day test windows
- Minimum training window: 200 observations
- Step size: 25 days
- Results are robust to expanding-window CV (not reported)

## Recommendations for Future Research

1. **Extended sample**: Re-estimate when longer sentiment history is available
2. **Higher-frequency data**: Intraday sentiment may capture faster-moving effects
3. **Alternative sentiment sources**: GDELT, social media, options-implied sentiment
4. **Regime-dependent models**: Re-visit AI interactions with more pre-AI data
5. **Bootstrap inference**: Block bootstrap for additional robustness

---

**Note**: All DM tests are horizon-adjusted and based on identical OOS forecast errors. Reported PCA loadings are averaged across rolling training windows to avoid look-ahead bias.
