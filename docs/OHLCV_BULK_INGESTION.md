# OHLCV Bulk Ingestion Guide

## Overview

The bulk OHLCV ingestion script provides a powerful, efficient way to download historical price data for multiple symbols. Inspired by SimFin's robust download patterns, it includes:

- **Progress Tracking**: Real-time progress with symbol-by-symbol status
- **Resume Capability**: Continue from where you left off if interrupted
- **File Age Checking**: Skip recently downloaded data to save time
- **Error Handling**: Gracefully handles failures and allows retries
- **Flexible Options**: Multiple ways to specify symbols and date ranges

## Quick Start

### Basic Usage

```bash
# Navigate to project root
cd /home/bawbawz/Project/MLTrading_WSL

# Fetch 10 years of data for S&P 500
python scripts/ingest_ohlcv_bulk.py --sp500-only --years 10

# Fetch specific symbols with 5 years of history
python scripts/ingest_ohlcv_bulk.py --symbols AAPL MSFT GOOGL --years 5

# Custom date range
python scripts/ingest_ohlcv_bulk.py --symbols AAPL --start-date 2015-01-01 --end-date 2024-12-31
```

## Features

### 1. Progress Tracking

The script provides detailed progress information:
```
Downloading OHLCV data for 500 symbols...
Date range: 2015-10-28 to 2025-10-28
Frequency: 1d

Fetching OHLCV: 100%|████████████████| 500/500 [15:23<00:00, 1.85s/symbol]
```

### 2. Resume Capability

If the download is interrupted (network issue, Ctrl+C, etc.), simply resume:

```bash
# Original command gets interrupted
python scripts/ingest_ohlcv_bulk.py --sp500-only --years 10
# ... download interrupted at 250/500

# Resume from checkpoint
python scripts/ingest_ohlcv_bulk.py --resume
```

The checkpoint file (`data/.ohlcv_progress.json`) tracks:
- Completed symbols
- Failed symbols (with error details)
- Current progress
- Timestamp

### 3. File Age Checking

Skip recently downloaded data to save time:

```bash
# Only re-download files older than 7 days
python scripts/ingest_ohlcv_bulk.py --sp500-only --max-age-days 7
```

Output:
```
Checking existing data...
- Found 500 symbols to check
- 450 fresh (< 7 days), will skip
- 30 stale (>= 7 days), will refresh
- 20 missing, will download
```

### 4. Error Handling

The script continues downloading other symbols even if some fail:
```
✓ Download Complete!
  Total symbols: 500
  Downloaded: 485
  Skipped (fresh): 0
  Failed: 15
  
Checkpoint saved: data/.ohlcv_progress.json
Run with --resume to retry failed symbols
```

## Command-Line Options

### Symbol Selection

| Option | Description | Example |
|--------|-------------|---------|
| `--symbols` | Specific symbols to fetch | `--symbols AAPL MSFT GOOGL` |
| `--sp500-only` | Fetch all S&P 500 constituents | `--sp500-only` |

### Date Range

| Option | Description | Example |
|--------|-------------|---------|
| `--years` | Number of years of history | `--years 10` |
| `--start-date` | Custom start date | `--start-date 2015-01-01` |
| `--end-date` | Custom end date (default: today) | `--end-date 2024-12-31` |

### Provider & Frequency

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | Data provider | `yfinance` |
| `--frequency` | Data frequency | `1d` (daily) |

### Optimization

| Option | Description | Example |
|--------|-------------|---------|
| `--resume` | Resume from checkpoint | `--resume` |
| `--max-age-days` | Skip files newer than N days | `--max-age-days 7` |

### Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to config file | `config/config.yaml` |

## Usage Examples

### Example 1: Initial Setup (First Time)

```bash
# Download 10 years of daily data for S&P 500
python scripts/ingest_ohlcv_bulk.py --sp500-only --years 10
```

**Expected output:**
```
======================================================================
Bulk OHLCV Data Ingestion
======================================================================
Provider: yfinance
Symbols: 503
Date range: 2015-10-28 to 2025-10-28
Frequency: 1d
======================================================================

Downloading OHLCV data for 503 symbols...
Date range: 2015-10-28 to today
Frequency: 1d

Fetching OHLCV: 100%|████████████████| 503/503 [18:45<00:00, 2.24s/symbol]

======================================================================
✓ Download Complete!
======================================================================
  Total symbols: 503
  Downloaded: 498
  Skipped (fresh): 0
  Failed: 5
  Total bars: 1,254,764
  Storage: data/parquet/1d/
  Duration: 18.8 minutes
  Avg speed: 26.5 symbols/min
======================================================================

Checkpoint saved: data/.ohlcv_progress.json
Run with --resume to retry failed symbols
```

### Example 2: Daily Update

```bash
# Only re-download data older than 1 day
python scripts/ingest_ohlcv_bulk.py --sp500-only --max-age-days 1
```

**Expected output:**
```
Checking existing data...
- Found 503 symbols to check
- 498 fresh (< 1 days), will skip
- 5 stale (>= 1 days), will refresh
- 0 missing, will download

Downloading OHLCV data for 5 symbols...

Fetching OHLCV: 100%|████████████████| 5/5 [00:15<00:00, 3.12s/symbol]

✓ Download Complete!
  Total symbols: 503
  Downloaded: 5
  Skipped (fresh): 498
  Failed: 0
  Total bars: 12,545
  Duration: 0.3 minutes
  Avg speed: 20.0 symbols/min
```

### Example 3: Specific Symbols with Custom Date Range

```bash
# Fetch 5 specific symbols from 2020 onwards
python scripts/ingest_ohlcv_bulk.py \
  --symbols AAPL MSFT GOOGL AMZN TSLA \
  --start-date 2020-01-01 \
  --end-date 2024-12-31
```

### Example 4: Resume Interrupted Download

```bash
# Original command
python scripts/ingest_ohlcv_bulk.py --sp500-only --years 10
# ... Ctrl+C pressed, or network issue

# Resume
python scripts/ingest_ohlcv_bulk.py --resume
```

The script will:
1. Load the checkpoint file
2. Skip already-completed symbols
3. Retry failed symbols
4. Continue with remaining symbols

## Storage Format

Downloaded data is stored as partitioned Parquet files:

```
data/parquet/
└── 1d/                    # Daily frequency
    ├── AAPL.parquet      # One file per symbol
    ├── MSFT.parquet
    ├── GOOGL.parquet
    └── ...
```

Each file contains columns:
- `date`: Trading date
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume
- `dividends`: Dividend payments
- `splits`: Stock splits
- `symbol`: Stock ticker

## Performance Tips

### 1. Use `--max-age-days` for Daily Updates

Instead of re-downloading all data daily:
```bash
# Only update data older than 1 day
python scripts/ingest_ohlcv_bulk.py --sp500-only --max-age-days 1
```

### 2. Start with Small Test

Test with a few symbols first:
```bash
python scripts/ingest_ohlcv_bulk.py --symbols AAPL MSFT GOOGL --years 5
```

### 3. Use Resume for Large Downloads

For large downloads (500+ symbols), use resume capability:
```bash
# If interrupted, just add --resume
python scripts/ingest_ohlcv_bulk.py --sp500-only --years 10
# ... interrupted
python scripts/ingest_ohlcv_bulk.py --resume
```

## Integration with Pipeline

### Method 1: Use Bulk Script First

```bash
# 1. Bulk download all data
python scripts/ingest_ohlcv_bulk.py --sp500-only --years 10

# 2. Run pipeline (skip step 1 since data is ready)
python run_core_pipeline.py --skip 1
```

### Method 2: Use Pipeline's Built-in Ingestion

```bash
# Pipeline will handle ingestion (step 1)
python run_core_pipeline.py
```

### Method 3: Daily Updates

```bash
# Quick daily update (only fetch recent data)
python scripts/ingest_ohlcv_bulk.py --sp500-only --max-age-days 1

# Then run pipeline
python run_core_pipeline.py --skip 1
```

## Troubleshooting

### Issue: Download is slow

**Solution**: This is normal for first-time downloads. yfinance fetches data symbol-by-symbol.

**Expected speeds**:
- First download (10 years): ~2-3 seconds per symbol
- Incremental update (1 day): ~1-2 seconds per symbol
- 500 symbols × 2.5s = ~20 minutes for initial download

### Issue: Some symbols failed

**Check the error**: Look at the checkpoint file:
```bash
cat data/.ohlcv_progress.json
```

**Common reasons**:
- Symbol no longer exists (delisted)
- Symbol moved to different exchange
- Network timeout

**Retry failed symbols**:
```bash
python scripts/ingest_ohlcv_bulk.py --resume
```

### Issue: "No data returned for symbol"

**Cause**: yfinance couldn't find data for that ticker

**Solutions**:
- Verify the ticker symbol is correct
- Check if the company is still listed
- Some symbols may not have historical data available

### Issue: Checkpoint file corrupted

**Solution**: Delete and start fresh:
```bash
rm data/.ohlcv_progress.json
python scripts/ingest_ohlcv_bulk.py --sp500-only --years 10
```

### Issue: Want to force re-download all data

**Solution**: Delete existing data first:
```bash
# Backup if needed
mv data/parquet/1d data/parquet/1d.backup

# Re-download
python scripts/ingest_ohlcv_bulk.py --sp500-only --years 10
```

## Comparison with Other Methods

| Method | Speed | Resume | File Age Check | Best For |
|--------|-------|--------|----------------|----------|
| **Bulk Script** | ⭐⭐⭐⭐ | ✅ Yes | ✅ Yes | Initial setup, large downloads |
| Pipeline Step 1 | ⭐⭐⭐ | ❌ No | ❌ No | Running full pipeline |
| Daily Update Script | ⭐⭐⭐⭐⭐ | ❌ No | ⚠️ Basic | Quick daily updates |

## Best Practices

### 1. Initial Setup Workflow

```bash
# Step 1: Download historical price data (bulk)
python scripts/ingest_ohlcv_bulk.py --sp500-only --years 10

# Step 2: Download fundamental data (SimFin or Alpha Vantage)
python scripts/ingest_fundamentals_simfin.py --sp500-only

# Step 3: Run full pipeline
python run_core_pipeline.py --skip 1
```

###
