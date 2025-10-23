# 📘 Instruction: Integrating Fundamental (PIT-Aligned) Features via Yahoo Finance (yfinance)

This guide expands the previous integration instructions, focusing on using **Yahoo Finance (yfinance)** to fetch and structure fundamental data for your ML-driven trading pipeline. It ensures the data is aligned **Point-in-Time (PIT)** and stored in **Parquet** format for seamless use alongside your OHLCV data.

---

## 🎯 Objective

Enable your pipeline to use **financial statement data** and **derived valuation metrics** directly from Yahoo Finance, keeping everything consistent with your existing OHLCV ingestion system.

---

## 🧩 1. Yahoo Finance Data Sources Explained

Yahoo Finance provides several distinct categories of fundamental data via the `yfinance.Ticker()` class. Each must be handled appropriately depending on your use case.

| Category                             | yfinance Method                               | Contains                               | Historical?             | Use Case                      |
| ------------------------------------ | --------------------------------------------- | -------------------------------------- | ----------------------- | ----------------------------- |
| **Income Statement**                 | `.financials` / `.quarterly_financials`       | Revenue, Net Income, EPS               | ✅ Yes                   | Profitability & growth ratios |
| **Balance Sheet**                    | `.balance_sheet` / `.quarterly_balance_sheet` | Assets, Liabilities, Equity            | ✅ Yes                   | Leverage & efficiency ratios  |
| **Cash Flow Statement**              | `.cashflow` / `.quarterly_cashflow`           | OCF, CapEx, FCF                        | ✅ Yes                   | Quality & liquidity measures  |
| **Snapshot Ratios (Key Statistics)** | `.info`                                       | PE, PB, Beta, Profit Margin, EV/EBITDA | ❌ No (current snapshot) | Reference-only metrics        |

---

## ⚙️ 2. How to Fetch Financial Data from yfinance

Example for one company:

```python
import yfinance as yf
import pandas as pd

symbol = "NVDA"
t = yf.Ticker(symbol)

# Core financial statements
income = t.financials.T.reset_index()
balance = t.balance_sheet.T.reset_index()
cashflow = t.cashflow.T.reset_index()

# Quarterly statements for higher granularity
income_q = t.quarterly_financials.T.reset_index()
balance_q = t.quarterly_balance_sheet.T.reset_index()
cashflow_q = t.quarterly_cashflow.T.reset_index()

# Current valuation and stats (snapshot)
info = t.info
```

---

## 🧱 3. Building a Combined Fundamental Dataset

### Step 1 — Combine Financial Statements

You can merge quarterly versions for each ticker:

```python
def combine_statements(symbol):
    t = yf.Ticker(symbol)
    inc = t.quarterly_financials.T.reset_index()
    bal = t.quarterly_balance_sheet.T.reset_index()
    cf = t.quarterly_cashflow.T.reset_index()

    df = inc.merge(bal, on='index', how='outer').merge(cf, on='index', how='outer')
    df = df.rename(columns={'index': 'date'})
    df['symbol'] = symbol
    return df
```

### Step 2 — Batch Collect Multiple Symbols

```python
symbols = ["AAPL", "MSFT", "NVDA"]
fundamental_frames = [combine_statements(s) for s in symbols]
fundamentals = pd.concat(fundamental_frames, ignore_index=True)
```

### Step 3 — Add Publication Dates

Since Yahoo Finance does not provide explicit report publication dates, estimate using a default delay (e.g., +45 days after quarter end):

```python
fundamentals['public_date'] = pd.to_datetime(fundamentals['date']) + pd.Timedelta(days=45)
```

This parameter is configurable via your YAML config under:

```yaml
features:
  pit_alignment:
    default_public_lag_days: 45
```

### Step 4 — Save as Parquet

```python
from pathlib import Path
path = Path("data/fundamentals/fundamentals.parquet")
fundamentals.to_parquet(path, index=False, compression='snappy')
print(f"Saved fundamentals for {len(symbols)} symbols → {path}")
```

---

## 🧩 4. Integration in `01_interactive_pipeline.py`

After creating technical features, insert:

```python
# --- Add Fundamental Features ---
from fa_features import align_pit_fundamentals
from pathlib import Path
import pandas as pd

print("Aligning and computing fundamental features...")

fundamental_path = Path("data/fundamentals/fundamentals.parquet")
if fundamental_path.exists():
    fundamentals_df = pd.read_parquet(fundamental_path)
else:
    raise FileNotFoundError(f"Fundamental data not found: {fundamental_path}")

# Align using fa_features.py
df = align_pit_fundamentals(df, fundamentals_df, config)

from src.io.storage import save_dataframe
save_dataframe(df, 'data/features/all_features_with_fundamentals.parquet')
print("✓ Combined technical + fundamental features saved.")
```

---

## 🧠 5. Feature Categories You’ll Get

After alignment, your dataset will include:

| Category          | Derived Features            | Examples                                      |
| ----------------- | --------------------------- | --------------------------------------------- |
| **Valuation**     | P/E, P/B, P/S               | `pe_ratio_calc`, `pb_ratio_calc`              |
| **Profitability** | ROE, ROA, Margin            | `roe_calc`, `profit_margin`                   |
| **Leverage**      | Debt Ratios                 | `debt_to_equity_calc`, `debt_ratio`           |
| **Growth**        | Revenue & Net Income Growth | `revenue_qoq_growth`, `net_income_qoq_growth` |
| **Quality**       | Earnings Quality            | `cash_quality` (OCF / Net Income)             |

These features are automatically calculated in `fa_features.py` and aligned using `public_date` to prevent look-ahead bias.

---

## 🔁 6. Optional: Fundamental Ingester (for automation)

```python
class FundamentalIngester:
    def __init__(self, storage_path="data/fundamentals"):
        from pathlib import Path
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def fetch_from_yfinance(self, symbols):
        import yfinance as yf, pandas as pd
        all_frames = []
        for sym in symbols:
            t = yf.Ticker(sym)
            inc = t.quarterly_financials.T.reset_index()
            bal = t.quarterly_balance_sheet.T.reset_index()
            cf = t.quarterly_cashflow.T.reset_index()
            df = inc.merge(bal, on='index', how='outer').merge(cf, on='index', how='outer')
            df = df.rename(columns={'index': 'date'})
            df['symbol'] = sym
            df['public_date'] = pd.to_datetime(df['date']) + pd.Timedelta(days=45)
            all_frames.append(df)
        fundamentals = pd.concat(all_frames, ignore_index=True)
        path = self.storage_path / "fundamentals.parquet"
        fundamentals.to_parquet(path, index=False, compression='snappy')
        print(f"Saved fundamentals for {len(symbols)} symbols → {path}")
```

---

## ✅ 7. Verification Checklist

* [ ] Parquet file exists at `data/fundamentals/fundamentals.parquet`.
* [ ] Each row includes `symbol`, `date`, and `public_date`.
* [ ] Config includes PIT alignment parameters (`default_public_lag_days`, etc.).
* [ ] Final dataset `all_features_with_fundamentals.parquet` includes new ratios.

---

**Author:** Danuphon Santiwong
**Purpose:** Comprehensive guide to fetch, structure, and integrate Yahoo Finance fundamental data for bias-free feature engineering in ML-based portfolio systems.
