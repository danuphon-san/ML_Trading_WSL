# SimFin Integration Guide

## Overview

SimFin has been integrated into the ML Trading pipeline as an additional fundamental data provider option. SimFin offers:

- **Extended History**: 20+ years of quarterly/annual fundamental data
- **Bulk Loading**: Efficient loading of multiple symbols at once
- **Local Caching**: Automatic caching for faster subsequent access
- **Professional Quality**: High-quality, normalized financial data
- **Point-in-Time Data**: Built-in respect for publication dates

## Prerequisites

### 1. Install SimFin Package

```bash
pip install simfin
```

Or add to `environment.yml`:
```yaml
dependencies:
  - simfin
```

### 2. Get SimFin API Key

1. Sign up at [https://simfin.com/](https://simfin.com/)
2. Subscribe to a paid plan (you mentioned you have Start Plan)
3. Get your API key from the dashboard

### 3. Configure API Key

Edit `config/config.yaml`:

```yaml
fundamentals:
  provider: "simfin"  # Set as default provider
  
  simfin:
    api_key: "your-actual-api-key-here"  # Replace with your key
    data_dir: "data/simfin_data"  # Local cache directory
    market: "us"  # US market
    variant: "quarterly"  # quarterly or annual
    include_banks_insurance: false  # Set to true if needed
```

## Usage

### Option 1: Bulk Ingestion (Initial Setup)

For first-time setup or complete data refresh:

**IMPORTANT**: Run these commands from the project root directory, NOT from the scripts directory.

```bash
# Navigate to project root (if not already there)
cd /home/bawbawz/Project/MLTrading_WSL

# Fetch all fundamental data from SimFin
python scripts/ingest_fundamentals_simfin.py

# Fetch specific symbols only
python scripts/ingest_fundamentals_simfin.py --symbols AAPL MSFT GOOGL

# Fetch S&P 500 only
python scripts/ingest_fundamentals_simfin.py --sp500-only
```

**First Run**: Downloads and caches data locally (~few minutes)
**Subsequent Runs**: Uses local cache (instant)

### Option 2: Daily Updates (Interactive)

For daily data updates with provider selection:

```bash
python scripts/daily_update_data.py
```

You'll be prompted to select a provider:
```
Select fundamental data provider:
======================================================================
1. SimFin (paid tier, 20+ years history, bulk loading)
2. Alpha Vantage (free tier, rate limited, 20 years)
3. yfinance (free, limited to ~5 quarters)
4. Use default from config.yaml
======================================================================
Enter choice (1-4) [default: 4]:
```

### Option 3: Programmatic Usage

```python
from src.io.ingest_fundamentals import FundamentalsIngester

# Initialize with SimFin
ingester = FundamentalsIngester(
    storage_path="data/fundamentals",
    provider="simfin",
    api_key="your-api-key",
    simfin_data_dir="data/simfin_data",
    simfin_market="us",
    simfin_variant="quarterly"
)

# Fetch data for symbols
symbols = ['AAPL', 'MSFT', 'GOOGL']
data_dict = ingester.fetch_fundamentals(symbols)

# Save to parquet
ingester.save_parquet(data_dict)
```

## Data Schema

SimFin data is automatically mapped to the standard pipeline schema:

| SimFin Column | Pipeline Column | Description |
|--------------|-----------------|-------------|
| Revenue | revenue | Total revenue |
| Net Income | net_income | Net income |
| Gross Profit | gross_profit | Gross profit |
| Total Assets | total_assets | Total assets |
| Total Liabilities | total_liabilities | Total liabilities |
| Total Equity | shareholders_equity | Shareholders' equity |
| Total Current Assets | current_assets | Current assets |
| Total Current Liabilities | current_liabilities | Current liabilities |
| Net Cash from Operating Activities | operating_cash_flow | Operating cash flow |
| Net Cash from Investing Activities | investing_cash_flow | Investing cash flow |
| Net Cash from Financing Activities | financing_cash_flow | Financing cash flow |

### Calculated Metrics

The following metrics are automatically calculated if not provided:

- **current_ratio**: current_assets / current_liabilities
- **debt_to_equity**: total_liabilities / shareholders_equity
- **roe** (Return on Equity): net_income / shareholders_equity
- **roa** (Return on Assets): net_income / total_assets

### Point-in-Time Alignment

SimFin provides:
- **date**: Report date (fiscal period end)
- **public_date**: Publication date (when data became available)

This ensures proper point-in-time alignment in your features, preventing look-ahead bias.

## Integration with Pipeline

### Step 1: Data Ingestion

```bash
# Option A: Use SimFin from start
python run_core_pipeline.py --steps 1

# Option B: Bulk ingest with SimFin first
python scripts/ingest_fundamentals_simfin.py
python run_core_pipeline.py --skip 1
```

### Step 2: Run Full Pipeline

```bash
python run_core_pipeline.py
```

The pipeline will automatically:
1. Load fundamental data from parquet files
2. Apply point-in-time alignment using `public_date`
3. Generate fundamental features
4. Merge with technical features
5. Train models and generate signals

## Performance Comparison

| Provider | History | Speed (100 symbols) | Rate Limits | Cost |
|----------|---------|---------------------|-------------|------|
| **SimFin** | 20+ years | ~2-5 min (first)<br>~10 sec (cached) | None (paid) | Paid plan |
| Alpha Vantage | 20 years | ~20-40 min | 5 calls/min (free) | Free/Paid |
| yfinance | ~5 quarters | ~5-10 min | None | Free |

## Troubleshooting

### Issue: "SimFin package not available"

**Solution**: Install simfin package
```bash
pip install simfin
```

### Issue: "API key required for SimFin provider"

**Solution**: Set your API key in `config/config.yaml`:
```yaml
fundamentals:
  simfin:
    api_key: "your-actual-api-key-here"
```

### Issue: First run is slow

**Expected**: SimFin downloads and caches data on first run. Subsequent runs are instant.

**Progress**: You'll see progress messages:
```
- Loading income statements...
- Loading balance sheets...
- Loading cash flow statements...
```

### Issue: Missing symbols

**Check**: Not all symbols are available in SimFin. The script will log warnings:
```
WARNING: No income data for XYZ
```

### Issue: Data directory permissions

**Solution**: Ensure the data directory is writable:
```bash
mkdir -p data/simfin_data
chmod 755 data/simfin_data
```

## Data Quality Checks

After ingestion, verify data quality:

```python
import pandas as pd

# Load a sample symbol
df = pd.read_parquet('data/fundamentals/AAPL.parquet')

# Check data coverage
print(f"Periods: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")

# Verify point-in-time dates
print(f"\nPublic dates properly set: {df['public_date'].notna().all()}")
print(f"Public date after report date: {(df['public_date'] > df['date']).all()}")
```

## Switching Between Providers

You can switch providers at any time:

### Method 1: Update config.yaml

```yaml
fundamentals:
  provider: "simfin"  # or "alpha_vantage" or "yfinance"
```

### Method 2: Interactive selection

Run `python scripts/daily_update_data.py` and select provider when prompted.

### Method 3: Command-line override

```bash
# Use specific ingestion script
python scripts/ingest_fundamentals_simfin.py
# OR
python scripts/reingest_fundamentals_alphavantage.py
```

## Best Practices

1. **Initial Setup**: Use bulk ingestion script for first-time setup
   ```bash
   python scripts/ingest_fundamentals_simfin.py --sp500-only
   ```

2. **Daily Updates**: Use daily update script with auto-provider selection
   ```bash
   python scripts/daily_update_data.
