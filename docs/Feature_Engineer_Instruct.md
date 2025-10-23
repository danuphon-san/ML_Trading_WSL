# 📘 Bias-Free Feature Engineering for Machine Learning in Stock Trading

This markdown file serves as a guide for designing machine learning features for stock trading models that minimize or eliminate common data biases.

---

## 🎯 Objective

To build a reliable machine learning model for trading, you must design **features** that:

* Are based only on information **available at time *t***.
* Do **not leak future data** (avoid look-ahead bias).
* Represent the **true market environment**, not an idealized one.
        
---

## 🧠 1. Common Biases to Avoid

| Bias Type              | Description                               | Prevention Strategy                           |
| ---------------------- | ----------------------------------------- | --------------------------------------------- |
| **Look-ahead Bias**    | Using future information in training data | Only use data available at or before time *t* |
| **Survivorship Bias**  | Ignoring delisted or failed stocks        | Include delisted/inactive stocks if possible  |
| **Data Snooping Bias** | Over-tuning on test data                  | Keep a final untouched test set               |
| **Temporal Leakage**   | Using data that was published later       | Align all features by release timestamp       |
| **Overfitting Bias**   | Model learns noise                        | Use regularization, cross-validation          |
| **Execution Bias**     | Unrealistic trade assumptions             | Add slippage, transaction cost modeling       |
| **Sampling Bias**      | Training on limited market conditions     | Include multiple market regimes               |
| **Modeler Bias**       | Human preference in feature choice        | Automate feature generation and selection     |

---

## 📊 2. Safe Feature Categories

All features listed below are **bias-free** if computed using **past or current information only**.

### 🔹 Price-Derived Features

| Feature                 | Description                    | Safe Condition             |
| ----------------------- | ------------------------------ | -------------------------- |
| Lagged Returns          | Returns from t−1, t−2, …       | Computed using past prices |
| Log Returns             | `ln(Pt / Pt−1)`                | Stationary and additive    |
| SMA / EMA               | Moving averages over n periods | Based on past data only    |
| Momentum Ratio          | `Close / SMA(n)`               | No future prices involved  |
| Price-to-MA Distance    | `(Close − SMA) / SMA`          | Uses historical prices     |
| MACD / RSI / Stochastic | Technical indicators           | Derived from past data     |

### 🔹 Volume-Based Features

| Feature                 | Description                                   |
| ----------------------- | --------------------------------------------- |
| Volume Change           | `(Vol_t - Vol_t-1)/Vol_t-1`                   |
| OBV (On-Balance Volume) | Cumulative volume aligned with price trend    |
| Volume-to-Price Ratio   | Detects accumulation or distribution behavior |

### 🔹 Volatility and Range Features

| Feature                  | Description                        |
| ------------------------ | ---------------------------------- |
| Rolling Std / Volatility | Standard deviation of past returns |
| ATR (Average True Range) | Average range over past n days     |
| High-Low % Range         | `(High - Low)/Close` per day       |

### 🔹 Statistical Features

| Feature             | Description                              |
| ------------------- | ---------------------------------------- |
| Rolling Mean / Std  | Captures trend & dispersion              |
| Skewness / Kurtosis | Measures return distribution shape       |
| Autocorrelation     | Lag correlation of returns               |
| Z-score             | `(Close - mean) / std` for normalization |

### 🔹 Composite & Engineered Features

| Feature              | Description                       |
| -------------------- | --------------------------------- |
| Momentum Score       | `(Close_t - Close_t-n)/Close_t-n` |
| Volatility Ratio     | `Vol_short / Vol_long`            |
| Rolling Sharpe Ratio | `mean(returns)/std(returns)`      |
| Return Z-score       | Normalized signal strength        |

### 🔹 Fundamental & Macro Features

| Type        | Example                                          | Safe Condition                  |
| ----------- | ------------------------------------------------ | ------------------------------- |
| Fundamental | EPS, PE ratio, revenue growth                    | Use value **at release date**   |
| Macro       | Interest rate, inflation, VIX, benchmark returns | Use **lagged or same-day** data |

---

## ⚙️ 3. Example: Safe Python Feature Pipeline

```python
import pandas as pd
import numpy as np

df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

# Rolling features (all use past data)
df['sma_10'] = df['Close'].rolling(10).mean()
df['sma_50'] = df['Close'].rolling(50).mean()
df['vol_10'] = df['log_ret'].rolling(10).std()
df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
df['rsi_14'] = compute_rsi(df['Close'], 14)
df['atr_14'] = compute_atr(df['High'], df['Low'], df['Close'], 14)

# Normalized signal
df['price_zscore'] = (df['Close'] - df['sma_50']) / df['vol_10']

# Drop NaN values from rolling calculations
df = df.dropna()
```

✅ **All features above are time-aligned and bias-free.**

---

## 🚫 4. Features to Avoid

| Feature                              | Reason                                     |                  |
| ------------------------------------ | ------------------------------------------ | ---------------- |
| Future Returns or Future Price       | Leaks future info (look-ahead bias)        |                  |
| Future-based Indicators              | e.g., using next-day SMA                   | Temporal leakage |
| Future Fundamentals                  | Data not yet released                      | Temporal leakage |
| Optimized Parameters (post-backtest) | Data snooping bias                         |                  |
| Current Index Membership Only        | Excludes failed stocks (survivorship bias) |                  |

---

## 🧩 5. Recommended ML Targets

* **Binary Classification:** Predict next-period direction (up/down)
* **Regression:** Predict next return magnitude (`r_t+1`)
* **Ranking:** Predict relative strength among multiple assets

Make sure labels are **shifted forward** (so the model predicts future outcome) but features stop at time *t*.

---

## 📈 6. Summary Table

| Group        | Example Features              | Bias Safety |
| ------------ | ----------------------------- | ----------- |
| Return-Based | Lag returns, rolling mean/std | ✅           |
| Trend        | SMA, EMA, MACD, RSI           | ✅           |
| Volatility   | ATR, Bollinger Band Width     | ✅           |
| Volume       | Volume change, OBV            | ✅           |
| Statistical  | Z-score, skewness             | ✅           |
| Fundamental  | PE, EPS (at release)          | ✅           |
| Macro        | Interest rate, VIX (lagged)   | ✅           |

---

## ✅ 7. Key Takeaways

* Only use **historical or lagged features**.
* Always **time-align** fundamentals and macro data.
* Simulate **real-world execution conditions** in backtesting.
* Avoid optimizing on test data — use **walk-forward validation**.

---

### 🧰 Bonus: Automation Tip

If your AI assistant supports coding automation, you can instruct it to:

1. Generate these features from OHLCV data.
2. Validate time alignment for each column.
3. Export the dataset as `bias_free_features.csv`.
4. Optionally, integrate with your trading simulation pipeline.

---

**Author:** Danuphon Santiwong
**Purpose:** Reliable ML feature engineering guideline to prevent data bias in stock trading simulations.