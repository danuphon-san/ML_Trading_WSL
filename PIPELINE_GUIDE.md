# ML Trading Pipeline - Implementation Guide

**Last Updated**: 2025-10-26

## Overview

This document describes the modular Python pipeline system that implements instruction1.md (Core Pipeline) and instruction2.md (Enhancements) for the ML Trading System.

## Architecture

The system is organized into two main orchestrator scripts:

1. **run_core_pipeline.py** - Steps 1-11 (Core ML workflow)
2. **run_enhancements.py** - Steps 12-15 (Regime detection, allocation, monitoring)

## Quick Start

### Run Core Pipeline (Steps 1-11)

```bash
# Full pipeline with default settings
python run_core_pipeline.py

# Custom configuration
python run_core_pipeline.py --symbols 50 --start-date 2020-01-01

# Run specific steps
python run_core_pipeline.py --steps 1-7

# Skip data ingestion (use cached data)
python run_core_pipeline.py --skip 1,2

# Dry run (show execution plan)
python run_core_pipeline.py --dry-run

# Validate artifacts only
python run_core_pipeline.py --validate-only
```

### Run Enhancements (Steps 12-15)

```bash
# Full enhancements pipeline
python run_enhancements.py

# With email summary
python run_enhancements.py --email-summary

# Validate core artifacts first
python run_enhancements.py --validate-core

# Run specific enhancements
python run_enhancements.py --steps 12,14
```

## Pipeline Steps

### Core Pipeline (Steps 1-11)

| Step | Module | Description | Output |
|------|--------|-------------|--------|
| 1 | Data Ingestion | Fetch OHLCV + fundamentals from yfinance | `data/parquet/1d/*.parquet`, `data/fundamentals/*.parquet` |
| 2 | Preprocessing | Load and align OHLCV data | In-memory DataFrame |
| 3 | Technical Features | RSI, MACD, momentum, volatility, etc. | Enhanced DataFrame |
| 4 | Fundamental Features | P/E, ROE, debt ratios with PIT alignment | Enhanced DataFrame |
| 5 | Label Generation | Forward return labels (5-day default) | DataFrame with labels |
| 6 | Feature Selection | Auto-select features, train/test split | Prepared datasets |
| 7 | Model Training | Train XGBoost/RF with MLflow tracking | Trained model |
| 8 | Signal Generation | Score test period with trained model | ML scores |
| 9 | Portfolio Construction | PyPortfolioOpt or inverse-vol optimizer | Portfolio weights |
| 10 | Backtesting | Simulate performance with realistic costs | Equity curve, metrics |
| 11 | Evaluation | Generate performance report | Summary report |

**Required Output Artifacts (5)**:
1. `data/features/all_features_with_fundamentals.parquet`
2. `data/results/scored_df.parquet`
3. `data/results/weights_df.parquet`
4. `data/results/backtest_results.json`
5. `models/latest/model.pkl`

### Enhancement Pipeline (Steps 12-15)

| Step | Module | Description | Output |
|------|--------|-------------|--------|
| 12 | Regime Detection | Detect risk-on/off via volatility, drawdown | `data/results/regime_state.csv` |
| 13 | Sleeve Allocation | Allocate across equities/crypto/cash (ERC/MVO) | `data/results/sleeve_allocation.json` |
| 14 | Turnover Management | Enforce turnover caps, filter small trades | `data/results/turnover_report.json` |
| 15 | Ops Monitoring | Generate HTML ops report, check kill-switches | `data/results/ops_report_{YYYYMMDD}.html` |

**Required Output Artifacts (4)**:
1. `data/results/regime_state.csv`
2. `data/results/sleeve_allocation.json`
3. `data/results/turnover_report.json`
4. `data/results/ops_report_{YYYYMMDD}.html`

## New Modules Created

### Data & I/O
- `src/io/results_saver.py` - Artifact management with validation

### Portfolio Management
- `src/portfolio/regime_detection.py` - Market regime detection (HMM/rules)
- `src/portfolio/sleeve_allocation.py` - Cross-sleeve allocation (ERC/MVO)
- `src/portfolio/turnover_manager.py` - Turnover caps and cost optimization

### Operations
- `src/live/monitoring.py` - Ops reports, kill-switch monitoring, alerts

### ML Enhancements
- Enhanced `src/ml/dataset.py` with `create_walk_forward_splits()`

### Utilities
- `utils/pipeline_utils.py` - Error handling, tracking, validation
- `utils/cli_parser.py` - CLI argument parsing

## Configuration Updates

New sections added to `config/config.yaml`:

```yaml
# Rebalancing schedules
rebalance:
  equities_frequency: weekly
  crypto_frequency: 3d

# Walk-forward cross-validation
validation:
  scheme: walk_forward
  windows: 6
  embargo_days: 5
  purge_days: 2

# Sleeve allocation
allocation:
  method: erc  # or mvo_reg
  crypto_risk_budget_max: 0.25
  rebalance_threshold: 0.05

# Operations & monitoring
ops:
  alerts: email
  email_recipients:
    - user@example.com
  kill_switch:
    enabled: true
    max_daily_loss_pct: 0.03
    min_live_sharpe_threshold: 0.5
```

## CLI Options

### Core Pipeline Options

```
--steps STEPS          Steps to run (e.g., "1-11", "1,3,5", "1-7")
--skip SKIP            Steps to skip (e.g., "1,2")
--symbols NUM          Number of symbols (default: 100)
--start-date DATE      Start date (YYYY-MM-DD)
--config PATH          Config file path (default: config/config.yaml)
--continue-on-error    Continue if step fails
--dry-run              Print plan without executing
--validate-only        Only validate artifacts
--verbose              Enable verbose logging
```

### Enhancements Pipeline Options

```
--steps STEPS          Steps to run (e.g., "12-15", "12,14")
--skip SKIP            Steps to skip
--config PATH          Config file path
--email-summary        Send email summary
--validate-core        Validate core artifacts before running
--continue-on-error    Continue if step fails
--dry-run              Print plan without executing
--verbose              Enable verbose logging
```

## Example Workflows

### Development Iteration

```bash
# 1. Initial data fetch
python run_core_pipeline.py --steps 1-2

# 2. Iterate on features
python run_core_pipeline.py --steps 3-7 --skip 1,2

# 3. Test portfolio construction
python run_core_pipeline.py --steps 8-11 --skip 1-7

# 4. Run full pipeline
python run_core_pipeline.py

# 5. Add enhancements
python run_enhancements.py --validate-core
```

### Production Daily Run

```bash
# Morning: Run core pipeline
python run_core_pipeline.py --symbols 200 2>&1 | tee logs/daily_run.log

# Afternoon: Run enhancements with monitoring
python run_enhancements.py --email-summary --validate-core
```

### Testing Specific Features

```bash
# Test only regime detection
python run_enhancements.py --steps 12

# Test turnover management
python run_enhancements.py --steps 14

# Test ops monitoring
python run_enhancements.py --steps 15
```

## Error Handling

Both orchestrators support:
- **Continue on error**: `--continue-on-error` flag
- **Step tracking**: Logs success/failure of each step
- **Artifact validation**: Verifies outputs exist
- **Detailed logging**: Saves to `logs/` directory

## Monitoring & Alerts

The ops monitoring module (Step 15) includes:
- Daily/weekly performance summary
- Position change tracking
- Kill-switch monitoring (daily loss, rolling Sharpe)
- Email/Slack alerts (configurable)
- HTML ops report generation

## Point-in-Time (PIT) Alignment

Fundamental features respect publication dates via `src/features/fa_features.py`:
- Minimum lag enforcement (`pit_min_lag_days`)
- Default publication lag (`default_public_lag_days`)
- Earnings blackout periods (`earnings_blackout_days`)

## Walk-Forward Cross-Validation

New function `create_walk_forward_splits()` in `src/ml/dataset.py`:
- Expanding training window (uses all historical data)
- Rolling test windows
- Embargo and purge periods to prevent leakage
- Configured via `validation` section in config.yaml

## Dependencies

All existing dependencies from `environment.yml` plus:
- loguru (for enhanced logging)
- hmmlearn (optional, for HMM regime detection)

## File Structure

```
project_root/
├── run_core_pipeline.py          # Core orchestrator (steps 1-11)
├── run_enhancements.py            # Enhancements orchestrator (steps 12-15)
├── config/
│   └── config.yaml                # Enhanced configuration
├── src/
│   ├── io/
│   │   └── results_saver.py       # New: artifact management
│   ├── portfolio/
│   │   ├── regime_detection.py    # New: step 12
│   │   ├── sleeve_allocation.py   # New: step 13
│   │   └── turnover_manager.py    # New: step 14
│   ├── live/
│   │   └── monitoring.py          # New: step 15
│   └── ml/
│       └── dataset.py             # Enhanced: walk-forward CV
├── utils/
│   ├── pipeline_utils.py          # New: helper functions
│   └── cli_parser.py              # New: CLI parsing
├── data/
│   ├── features/                  # Artifact 1
│   ├── results/                   # Artifacts 2-4, 6-9
│   └── models/                    # Artifact 5
└── logs/                          # Pipeline logs
```

## Troubleshooting

### Missing Core Artifacts
```bash
# Validate what exists
python run_core_pipeline.py --validate-only

# Run full core pipeline
python run_core_pipeline.py
```

### Step Failures
```bash
# Check logs
tail -100 logs/core_pipeline.log

# Run with verbose logging
python run_core_pipeline.py --verbose

# Continue despite errors (for debugging)
python run_core_pipeline.py --continue-on-error
```

### Memory Issues
```bash
# Reduce number of symbols
python run_core_pipeline.py --symbols 30

# Limit rebalance dates (edit MAX_DATES in step 9)
```

## Next Steps

1. **Test Core Pipeline**: Run `python run_core_pipeline.py` and verify 5 artifacts
2. **Test Enhancements**: Run `python run_enhancements.py` and verify 4 artifacts
3. **Configure Email Alerts**: Update `ops.email_recipients` in config.yaml
4. **Customize Regime Detection**: Adjust thresholds in `config.yaml`
5. **Integrate Crypto Data**: Add crypto data sources for sleeve allocation
6. **Set Up Cron Jobs**: Schedule daily/weekly pipeline runs

## Support

For issues or questions:
- Check pipeline logs: `logs/core_pipeline.log`, `logs/enhancements_pipeline.log`
- Review CLAUDE.md for project overview
- Consult instruction1.md and instruction2.md for step details

---

**Generated by Claude Code** | Last Updated: 2025-10-26
