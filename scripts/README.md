# Data Management Scripts

This directory contains utility scripts for managing OHLCV and fundamental data.

## Available Scripts

### 1. `daily_update_data.py`
**Purpose**: Incremental update of both OHLCV and fundamental data
**When to use**: Daily, after market close (automated via cron)

```bash
python daily_update_data.py
```

**What it does**:
- Fetches only the last 7 days of OHLCV data
- Merges with existing data, removing duplicates
- Updates fundamentals with latest quarterly reports
- Takes 1-5 minutes (vs 15-60 min for full re-ingestion)

---

### 2. `update_fundamentals.py`
**Purpose**: Update only fundamental data (skip OHLCV)
**When to use**: When you need to refresh fundamentals without touching price data

```bash
# Update all symbols
python update_fundamentals.py

# Update specific symbols only
python update_fundamentals.py --symbols AAPL MSFT GOOGL TSLA
```

**Options**:
- `--symbols`: Space-separated list of symbols to update
- `--config`: Path to config file (default: config/config.yaml)

---

### 3. `reingest_fundamentals_alphavantage.py`
**Purpose**: Full re-download of ALL fundamental history
**When to use**: Initial setup, switching providers, data corruption recovery

```bash
python reingest_fundamentals_alphavantage.py
```

**What it does**:
- Downloads full 20-year history from Alpha Vantage (81 quarters)
- Replaces existing fundamental data completely
- Takes 15-60 minutes depending on symbol count
- Requires user confirmation before proceeding

**Note**: Free tier Alpha Vantage is limited to ~5 API calls/minute. The script automatically adds delays.

---

## Running Scripts

**All scripts work from any directory**:

```bash
# From project root
python scripts/daily_update_data.py

# From scripts directory
cd scripts
python daily_update_data.py

# From anywhere else
/path/to/project/scripts/daily_update_data.py
```

Scripts automatically detect the project root and change to it before execution.

---

## Configuration

All scripts read settings from `config/config.yaml`:

```yaml
fundamentals:
  provider: "alpha_vantage"  # or "yfinance"
  alpha_vantage:
    api_key: "YOUR_KEY_HERE"
    rate_limit_sleep: 0.3
```

To switch providers:
1. Edit `config/config.yaml`
2. Run full re-ingestion: `python scripts/reingest_fundamentals_alphavantage.py`

---

## Automation (Cron)

Set up daily updates with cron:

```bash
# Edit crontab
crontab -e

# Add this line (runs at 5 PM EST, Monday-Friday)
0 17 * * 1-5 cd /home/bawbawz/Project/MLTrading_WSL && python scripts/daily_update_data.py >> logs/daily_update.log 2>&1
```

---

## Troubleshooting

### Script can't find config file
**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'config/config.yaml'`

This should not happen with the updated scripts, but if it does:
- Ensure you're running from the project root or scripts directory
- Check that `config/config.yaml` exists in the project root

### No existing data to update
**Error**: `No price data found. Run data ingestion first.`

**Solution**: Run the full pipeline first to create initial data:
```bash
python run_core_pipeline.py --steps 1
```

### Alpha Vantage rate limit errors
**Error**: API returns note about call frequency

**Solutions**:
- Free tier: Wait 1 minute, then retry
- Reduce symbol count
- Upgrade to Alpha Vantage premium tier
- Switch to yfinance (edit `config/config.yaml`)

### Import errors (loguru not found)
**Error**: `ModuleNotFoundError: No module named 'loguru'`

**Solution**: Activate the conda environment:
```bash
conda activate us-stock-app
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────┐
│ Initial Setup (First Time Only)                    │
├─────────────────────────────────────────────────────┤
│ 1. python run_core_pipeline.py --steps 1,2         │
│    → Fetches 10 years of OHLCV data                │
│                                                     │
│ 2. python scripts/reingest_fundamentals_alpha...   │
│    → Fetches 20 years of fundamental data          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Daily Operations                                    │
├─────────────────────────────────────────────────────┤
│ 1. python scripts/daily_update_data.py             │
│    → Updates last 7 days of OHLCV                  │
│    → Updates fundamentals (new quarters only)      │
│                                                     │
│ 2. python run_core_pipeline.py --skip 1            │
│    → Runs feature engineering + ML pipeline        │
│    → Uses updated data without re-downloading      │
└─────────────────────────────────────────────────────┘
```

---

## Performance Comparison

| Operation | Method | Time | Data Coverage |
|-----------|--------|------|---------------|
| **Initial OHLCV** | `run_core_pipeline.py --steps 1` | 5-15 min | 10 years |
| **Initial Fundamentals** | `reingest_fundamentals_alphavantage.py` | 15-60 min | 20 years |
| **Daily OHLCV** | `daily_update_data.py` | 1-3 min | Last 7 days |
| **Daily Fundamentals** | `daily_update_data.py` | 1-3 min | New quarters |
| **Specific Symbols** | `update_fundamentals.py --symbols AAPL` | 5-10 sec | Single symbol |

---

For more details, see [PIPELINE_GUIDE.md](../PIPELINE_GUIDE.md) in the project root.
