# US Stock ML Trading System

Full-stack ML-powered trading system for personal portfolio management with web dashboard.

## Features

### Backend System (Phase 1 - Complete)
- **Data Ingestion**: OHLCV + fundamentals from yfinance
- **Feature Engineering**: Technical (momentum, RSI, volatility) + Fundamental (with PIT alignment)
- **ML Modeling**: XGBoost/RandomForest with time-series CV and MLflow tracking
- **Portfolio Construction**: PyPortfolioOpt integration with multiple optimization objectives
- **Backtesting**: Vectorized engine with realistic cost modeling (commission + slippage)
- **Live Operations**: Daily signal generation and portfolio management

### Web Dashboard (Phase 2 - Complete)
- **FastAPI Backend**: REST API for all system operations
- **Streamlit Frontend**: Interactive dashboard with:
  - Real-time portfolio monitoring (positions, P&L, equity curve)
  - System signals and trade execution tracking
  - **Trade Reconciliation**: Compare system signals vs actual executions
  - Manual trade entry with override reasons
  - Backtest execution interface
  - Model training & evaluation monitoring

### Key Technical Safeguards
✅ **PIT Alignment**: Fundamentals respect publication dates (no look-ahead bias)
✅ **Price Panel Pattern**: Real close prices passed to optimization (never reconstructed from returns)
✅ **Cost Realism**: Commission + slippage applied in backtests
✅ **Weight Validation**: Sum=1, bounds respected, turnover monitoring

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate us-stock-app

# Initialize database
python -c "from src.database.models import init_database; init_database()"
```

### 2. Configuration

Edit `config/config.yaml` and `config/universe.yaml` to customize:
- Universe selection rules
- Feature parameters
- ML model settings
- Portfolio constraints
- Cost assumptions

### 3. Data Ingestion

```python
from src.io.ingest_ohlcv import OHLCVIngester
from src.io.universe import load_sp500_constituents
import yaml

config = yaml.safe_load(open('config/config.yaml'))

# Load universe
symbols = load_sp500_constituents()[:100]  # Start with top 100

# Fetch data
ingester = OHLCVIngester(storage_path='data/parquet')
data = ingester.fetch_ohlcv(symbols, '2018-01-01', None)
ingester.save_parquet(data)
```

### 4. Feature Generation

```python
from src.features.ta_features import create_technical_features
from src.labeling.labels import generate_forward_returns

# Load OHLCV data
df = ingester.load_parquet(symbols)

# Generate features
df = create_technical_features(df, config)

# Generate labels
df = generate_forward_returns(df, config)

# Save
from src.io.storage import save_dataframe
save_dataframe(df, 'data/features/all_features.parquet')
```

### 5. Model Training

```python
from src.ml.dataset import MLDataset, create_time_based_split
from src.ml.train import ModelTrainer

# Prepare dataset
dataset = MLDataset(label_col='forward_return_5d')
X, y = dataset.prepare(df)

# Split
train_df, test_df = create_time_based_split(df, test_size=0.2, embargo_days=5)
X_train, y_train = dataset.prepare(train_df, auto_select_features=False)
X_test, y_test = dataset.prepare(test_df, auto_select_features=False)

# Train with MLflow
trainer = ModelTrainer(config)
trainer.train_with_mlflow(X_train, y_train, X_test, y_test, run_name="production_v1")

# Save model
trainer.save_model('data/models/model_v1.pkl')
```

### 6. Backtesting

```python
from src.backtest.bt_engine import VectorizedBacktester
from src.portfolio.construct import construct_portfolio

# Load features and model
trainer.load_model('data/models/model_v1.pkl')

# Generate scores
scores = trainer.predict(X_test)
scored_df = test_df[['date', 'symbol']].copy()
scored_df['ml_score'] = scores

# Construct portfolio weights over time
weights_history = []
for date in scored_df['date'].unique():
    day_scores = scored_df[scored_df['date'] == date]
    weights = construct_portfolio(day_scores, price_panel, config)

    for symbol, weight in weights.items():
        weights_history.append({
            'date': date,
            'symbol': symbol,
            'weight': weight
        })

weights_df = pd.DataFrame(weights_history)

# Run backtest
backtester = VectorizedBacktester(config)
results = backtester.run(weights_df, price_panel)

print(f"Total Return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
```

### 7. Launch Web Dashboard

Terminal 1 - Start API:
```bash
python src/api/main.py
# API runs on http://localhost:8000
```

Terminal 2 - Start Dashboard:
```bash
streamlit run src/frontend/dashboard.py
# Dashboard opens at http://localhost:8501
```

## Dashboard Features

### Portfolio Overview
- Current equity and performance metrics
- Equity curve visualization
- Real-time positions with P&L

### Signals & Trades
- View recent system signals
- Trade execution history
- **Manual trade entry**: Record trades with deviation reasons

### Trade Reconciliation
- Compare system signals vs actual executions
- Track price deviations (bps)
- Monitor manual overrides
- Execution quality analytics

### Backtesting
- Run backtests with custom parameters
- Visualize results
- Compare strategies

### Model Training
- Monitor model performance
- Feature importance
- Trigger retraining

## Project Structure

```
├── config/
│   ├── config.yaml          # Central configuration
│   └── universe.yaml        # Stock selection rules
├── data/
│   ├── parquet/            # OHLCV data
│   ├── fundamentals/       # Fundamental data
│   ├── features/           # Engineered features
│   ├── models/             # Trained models
│   └── portfolio/          # Positions & trades
├── src/
│   ├── io/                 # Data ingestion
│   ├── features/           # Feature engineering
│   ├── labeling/           # Label generation
│   ├── ml/                 # Model training & evaluation
│   ├── portfolio/          # Portfolio construction
│   ├── backtest/           # Backtesting engine
│   ├── live/               # Live operations
│   ├── api/                # FastAPI backend
│   ├── frontend/           # Streamlit dashboard
│   └── database/           # Trade tracking DB models
└── notebooks/              # Research notebooks
```

## Key Configuration Parameters

### Portfolio Construction (`config.yaml`)
```yaml
portfolio:
  top_k: 20                    # Number of positions
  optimizer: "pypfopt"         # pypfopt, inverse_vol, equal_weight

  pypfopt:
    mu_mapping: "rank_to_mu"   # identity, sigmoid, rank_to_mu
    objective: "max_sharpe"    # max_sharpe, min_volatility
    l2_reg: 0.01              # Regularization for stability
    min_weight: 0.01          # Min 1% per position
    max_weight: 0.15          # Max 15% per position

  costs_bps: 1.0              # Commission per side
  slippage_bps: 2.0           # Market impact
```

### PIT Alignment (`config.yaml`)
```yaml
features:
  pit_alignment:
    pit_min_lag_days: 1            # Minimum publication lag
    default_public_lag_days: 45    # Fallback when date missing
    earnings_blackout_days: 2      # Exclude earnings window
```

## Important Notes

1. **Data Quality**: yfinance is for prototyping - use premium vendor for production
2. **Survivorship Bias**: Current implementation doesn't handle delistings
3. **Costs**: Adjust `costs_bps` and `slippage_bps` based on your broker/size
4. **PIT Compliance**: Always verify fundamental features respect publication dates
5. **Trade Tracking**: Use reconciliation feature to monitor execution quality

## Next Steps

- [ ] Integrate premium data provider (Alpha Vantage, Polygon, etc.)
- [ ] Add survivorship bias handling
- [ ] Implement broker integration for live trading
- [ ] Set up alerts (email/Slack)
- [ ] Add more ML models (LightGBM, neural nets)
- [ ] Enhance frontend with React for better UX

## Support

For issues or questions, refer to CLAUDE.md for architecture details and design decisions.
