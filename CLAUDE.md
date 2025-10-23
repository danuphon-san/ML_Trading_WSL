# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

US Stock Research, Backtesting & Portfolio Optimization system. Python-based full-stack application for:
- Data ingestion (OHLCV + fundamentals)
- Feature engineering (technical + fundamental)
- ML modeling & scoring
- Portfolio construction (PyPortfolioOpt integration)
- Backtesting with realistic costs
- Daily live operations

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate us-stock-app

# Or update existing environment
conda env update -f environment.yml --prune
```

**Key dependencies**: PyPortfolioOpt 1.5.5, pandas, numpy, scikit-learn, xgboost, mlflow, backtrader, yfinance

## Project Structure

```
├─ config/
│  ├─ config.yaml        # Central config (all parameters)
│  └─ universe.yaml      # Stock universe rules
├─ data/
│  ├─ raw/              # Vendor data (read-only)
│  ├─ parquet/          # OHLCV partitioned by freq/symbol
│  ├─ fundamentals/     # Quarterly snapshots
│  ├─ features/         # Derived features
│  ├─ labels/           # Forward return targets
│  ├─ models/           # Trained models
│  ├─ portfolio/        # Positions, trades, equity curve
│  └─ reports/          # Performance reports
├─ src/
│  ├─ io/              # Data ingestion & storage
│  ├─ features/        # ta_features.py, fa_features.py
│  ├─ labeling/        # labels.py
│  ├─ ml/              # Model training & evaluation
│  ├─ portfolio/       # Ranking, construction (pypfopt), risk
│  ├─ backtest/        # bt_engine.py, backtrader_engine.py
│  ├─ optimize/        # optuna_search.py
│  └─ live/            # Daily operations
└─ notebooks/          # Research notebooks
```

## Core Architecture Principles

### Data Flow
1. **Ingest** → raw vendor data → partitioned parquet
2. **Features** → technical + fundamental (PIT-aligned)
3. **Labels** → forward returns (5d default)
4. **Model** → train/score with time-series CV
5. **Rank** → select Top K by ML score
6. **Optimize** → PyPortfolioOpt (covariance + score→μ mapping)
7. **Backtest** → apply costs/slippage, generate equity curve
8. **Live** → daily rebalancing

### Point-in-Time (PIT) Guardrails
**Critical**: All fundamental features must respect publication dates to prevent look-ahead bias.

- `align_pit_fundamentals()` in `src/features/fa_features.py` enforces:
  - `pit_min_lag_days`: minimum lag between publication and usage
  - `default_public_lag_days`: fallback when public_date missing
  - `earnings_blackout_days`: exclude dates around earnings

### Price Panel Pattern
**Always pass real close prices (`price_panel`) to portfolio construction** — never reconstruct from returns. Derive returns internally when needed for covariance estimation.

Correct:
```python
targets = target_weights(selected_df, price_panel)  # price_panel = close prices
```

Incorrect:
```python
# Don't pass return series as if they were prices
```

### Portfolio Optimization

Two modes (set in `config.yaml`):

**1. PyPortfolioOpt** (`optimizer: "pypfopt"`)
- Maps ML scores → expected returns (μ) via `mu_mapping`: identity, sigmoid, or rank_to_mu
- Covariance from historical returns (`cov_lookback_days`)
- Objectives: `max_sharpe` or `min_volatility`
- Optional L2 regularization for stability
- Constraints: `min_weight`, `max_weight` per position

**2. Inverse Volatility** (`optimizer: "inverse_vol"`)
- Simple fallback: weights ∝ 1/volatility

### Cost Model

**User trading costs** configured in `config.yaml`:
```yaml
portfolio:
  costs_bps: 1.0       # Commission/fee per side (bps of notional)
  slippage_bps: 2.0    # Market impact/bid-ask (bps)
```

Applied during backtesting:
- **Commission**: `notional × costs_bps / 10,000` per trade
- **Slippage**: execution price moved against you by `slippage_bps`

For advanced per-venue/per-symbol overrides, use optional `execution:` config block.

## Configuration Management

**Single source of truth**: `config/config.yaml`

Key sections:
- `data`: paths, stores
- `ingest`: frequencies, date range, provider
- `universe`: filters (min price, volume, exchanges)
- `features`: lookback windows
- `labels`: horizon, target type
- `modeling`: algo, CV scheme
- `portfolio`: Top K, constraints, optimizer settings, costs
- `fundamentals`: PIT parameters
- `reporting`: benchmark, output directory

**Always load config**:
```python
import yaml
cfg = yaml.safe_load(open("config/config.yaml"))
```

## Key Module Responsibilities

### `src/io/`
- Universe selection & refresh
- OHLCV ingestion (yfinance default, extensible to paid vendors)
- Parquet storage with partitioning

### `src/features/`
- `ta_features.py`: momentum, RSI, volatility, moving averages
- `fa_features.py`: fundamental ratios, PIT alignment via `align_pit_fundamentals()`

### `src/ml/`
- Dataset construction with time-aware train/test split
- Model training (RandomForest, XGBoost)
- Time-series cross-validation with embargo
- MLflow logging

### `src/portfolio/`
- `rank.py`: cross-sectional ranking
- `construct.py`: unified entrypoint → calls pypfopt or inverse-vol
- `pypfopt_construct.py`: score→μ mapping, EfficientFrontier setup
- `risk.py`: portfolio analytics

### `src/backtest/`
- `bt_engine.py`: vectorized backtester with cost/slippage application
- `backtrader_engine.py`: event-driven alternative

### `src/live/`
- Daily score generation
- Target weight calculation
- Trade simulation
- Monitoring & reporting

## QA & Stability Checks

Before deploying:
- [ ] Weights sum to 1.0 (long-only: all ≥0)
- [ ] Covariance window has ≥2 symbols with sufficient history
- [ ] No look-ahead: PIT alignment verified
- [ ] Turnover alerts if >35% (configurable)
- [ ] Cost logging: effective costs & slippage per period
- [ ] Reproducibility: pin Python/package versions, log config snapshot

## Development Workflow

1. **Modify config** (`config/config.yaml`) for new parameters
2. **Extend features** in `src/features/` (respect PIT for fundamentals)
3. **Update model** in `src/ml/` if changing algo/hyperparams
4. **Adjust portfolio logic** in `src/portfolio/` for new constraints
5. **Run backtest** to validate changes
6. **Check reports** in `data/reports/` for performance metrics

## Testing

- Unit tests in `tests/` (structure TBD)
- Integration tests should cover:
  - PIT alignment correctness
  - Price panel passthrough (no return reconstruction)
  - Weight constraints (sum=1, bounds respected)
  - Cost application in backtests

## Important Caveats

- **Data quality**: yfinance is starter-grade; production requires premium vendor
- **Survivorship bias**: universe refresh logic must account for delistings
- **Regime shifts**: fixed lookbacks may underperform in volatile markets
- **Optimization instability**: tune `l2_reg` if optimizer produces erratic weights
- **Slippage assumptions**: `slippage_bps` is simplified; real impact varies by size/liquidity
