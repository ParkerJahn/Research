# PCA–VAR with Sentiment for Implied Volatility

## Quant Research Implementation Guide (Leakage-Safe)

---

## 🎯 Research Objective

Build a **PCA–VARX framework** to forecast **implied volatility** (NVDA, SMH, SOXX, VIX) using:

* Semiconductor equity dynamics
* Macro and commodity factors
* **Orthogonalized sentiment shocks**
* Explicit handling of the **AI-era structural break (post-2023)**

The goal is **not** to add sentiment mechanically, but to identify whether **sentiment-driven shocks** contain *incremental predictive power* for IV beyond known volatility factors.

---

## 🧠 Data Usage (CRITICAL)

* Use **full sample: 2022 → present**
* Do **not** discard early data

**Why this matters**:

* PCA factor stability
* Regime contrast (pre-AI vs AI-era)
* Avoiding post-2023 overfitting

🚫 **Fatal Error**: Training only on post-2023 data

---

## 🧾 Sentiment Processing (NON-NEGOTIABLE)

### 1. Residualize Sentiment

Sentiment must be purged of return information:

```
Sent_t = α + β·Returns_t + ε_t
```

Use **ε_t (sentiment shock)** only.

🚫 **Fatal Error**: Using raw sentiment scores

---

### 2. Lag Sentiment BEFORE PCA

* Use `Sent_{t-1}`, `Sent_{t-5}`
* Never use contemporaneous sentiment

🚫 **Fatal Error**: Look-ahead bias via Sent_t

---

### 3. Window-Based Standardization

* Z-score **inside each training window**
* Never normalize using full-sample statistics

🚫 **Fatal Error**: Global normalization

---

## 📉 PCA Construction (Leakage Control)

* PCA must be **expanding-window**
* Fit PCA **only on training data**
* Freeze loadings when projecting forward
* Retain PCs explaining **70–85% variance**

### Expected PCA Interpretation

| PC  | Interpretation             |
| --- | -------------------------- |
| PC1 | Semiconductor equity level |
| PC2 | Market volatility          |
| PC3 | **Sentiment shock**        |
| PC4 | Energy / commodities       |
| PC5 | Cross-asset stress         |

🚫 **Fatal Error**: PCA fit on the full dataset

---

## 🤖 AI-Era Structural Break (REQUIRED)

Define regime indicator:

```python
AI_REGIME = 1 if date >= "2023-03-01" else 0
```

Create interaction terms:

```
PC_sentiment × AI_REGIME
```

🚫 **Fatal Errors**:

* Using the dummy without interactions
* Re-running PCA separately by regime

---

## 🔁 VARX Model Specification

### Endogenous Variables

```
Y_t = [NVDA_IV, SMH_IV, SOXX_IV]
# or
Y_t = [SMH_IV, SOXX_IV, VIX]
```

### Exogenous Variables

```
X_t = [
  PC_macro,
  PC_vol,
  PC_sentiment,
  PC_energy,
  PC_sentiment × AI_REGIME
]
```

* Lag order: **2–3 (AIC-selected)**
* Expanding window
* Monthly or 20-day refits

---

## ⚖️ Asymmetry Control (AT LEAST ONE)

Include one:

* Negative-return dummy (NVDA < 0)
* High-VIX regime dummy (VIX > 75th percentile)

Purpose: capture leverage and convexity effects in IV

---

## 🔮 Forecast Design

* Horizons:

  * **1 week**
  * **1 month**

🚫 **Fatal Errors**:

* Long-horizon IV forecasting
* Predicting returns instead of IV

---

## 🧪 Benchmarks (MANDATORY)

Compare against:

* Random walk IV
* AR(1), AR(5)
* **HAR-IV**
* VIX-only regression

🚫 **Fatal Error**: Claiming predictability without beating HAR-IV

---

## 📊 Evaluation Metrics (IV-Appropriate)

* RMSE
* **QLIKE**
* Diebold–Mariano tests

Optional (strong):

* Vega-weighted option P&L

---

## 🔴 The Three Fatal Errors

1. PCA on the full sample
2. Raw or contemporaneous sentiment in PCA/VAR
3. Ignoring AI-era interaction effects

Any of these invalidate the results.

---

## ✅ Expected Outcome

If implemented correctly:

* Sentiment emerges as an **independent PCA factor**
* Sentiment sensitivity increases post-AI regime
* PCA–VARX outperforms HAR-IV at short horizons
* Results are publishable and trading-relevant

---

*End of document
