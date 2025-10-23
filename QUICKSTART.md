# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate us-stock-app

# 2. Initialize database
python -c "from src.database.models import init_database; init_database()"

# 3. Verify installation
python -c "import yaml, pandas, numpy, sklearn, xgboost; print('All packages installed successfully')"
```

## Option A: Run Full Pipeline (30-60 minutes)

```bash
python run_pipeline.py
```

This will:
1. Fetch data for top 100 S&P 500 stocks
2. Generate technical features and labels
3. Train ML model with cross-validation
4. Run backtest and generate performance report
5. Save model to `data/models/latest_model.pkl`

## Option B: Launch Dashboard Only (1 minute)

```bash
./start_dashboard.sh
```

Opens web interface at:
- API: http://localhost:8000
- Dashboard: http://localhost:8501

**Note**: Dashboard requires existing data/models to show meaningful content.

## Option C: Interactive Development

### Step 1: Get Data (5 minutes)
```python
import yaml
from src.io.universe import load_sp500_constituents
from src.io.ingest_ohlcv import OHLCVIngester

config = yaml.safe_load(open('config/config.yaml'))

# Get symbols
symbols = load_sp500_constituents()[:50]  # Start small

# Fetch data
ingester = OHLCVIngester()
data = ingester.fetch_ohlcv(symbols, '2020-01-01', None)
ingester.save_parquet(data)

print(f"Fetched data for {len(data)} symbols")
```

### Step 2: Generate Features (2 minutes)
```python
from src.features.ta_features import create_technical_features
from src.labeling.labels import generate_forward_returns

# Load data
df = ingester.load_parquet(symbols)

# Add features and labels
df = create_technical_features(df, config)
df = generate_forward_returns(df, config)

print(f"Generated {len(df.columns)} features")
```

### Step 3: Train Model (5 minutes)
```python
from src.ml.dataset import MLDataset, create_time_based_split
from src.ml.train import ModelTrainer

# Prepare data
dataset = MLDataset(label_col='forward_return_5d')
train_df, test_df = create_time_based_split(df, test_size=0.2)

X_train, y_train = dataset.prepare(train_df)
X_test, y_test = dataset.prepare(test_df)

# Train
trainer = ModelTrainer(config)
trainer.train(X_train, y_train, X_test, y_test)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
print(f"IC: {metrics['ic']:.4f}, Rank IC: {metrics['rank_ic']:.4f}")

# Save
trainer.save_model('data/models/my_model.pkl')
```

### Step 4: Backtest (2 minutes)
```python
from src.backtest.bt_engine import VectorizedBacktester
from src.portfolio.construct import construct_portfolio
import pandas as pd

# Generate scores
scores = trainer.predict(X_test)
scored_df = test_df[['date', 'symbol']].copy()
scored_df['ml_score'] = scores

# Get prices
price_panel = df[['date', 'symbol', 'close']].copy()

# Build portfolio weights
weights_history = []
for date in scored_df['date'].unique()[:30]:  # First 30 rebalance dates
    day_scores = scored_df[scored_df['date'] == date]
    weights = construct_portfolio(day_scores, price_panel, config)

    for symbol, weight in weights.items():
        weights_history.append({'date': date, 'symbol': symbol, 'weight': weight})

weights_df = pd.DataFrame(weights_history)

# Run backtest
backtester = VectorizedBacktester(config)
results = backtester.run(weights_df, price_panel)

print(f"Total Return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
```

## Key Files to Customize

1. **config/config.yaml** - Adjust parameters:
   - `portfolio.top_k`: Number of positions (default: 20)
   - `portfolio.costs_bps`: Commission rate (default: 1.0 bps)
   - `modeling.algorithm`: ML algorithm (xgboost, random_forest)

2. **config/universe.yaml** - Modify stock filters:
   - `min_price`: Minimum stock price
   - `min_volume`: Minimum daily volume
   - `exchanges`: Exchanges to include

## Common Tasks

### View MLflow Results
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### Access API Documentation
```bash
# Start API, then visit:
http://localhost:8000/docs
```

### Check Data Status
```python
from src.io.storage import list_symbols, get_date_range

symbols = list_symbols('data/parquet/1d')
print(f"Available symbols: {len(symbols)}")

for symbol in symbols[:5]:
    start, end = get_date_range('data/parquet/1d', symbol)
    print(f"{symbol}: {start} to {end}")
```

### Retrain Model with New Data
```bash
# 1. Update data
python -c "from src.io.ingest_ohlcv import OHLCVIngester; ing = OHLCVIngester(); ing.update_data(['AAPL', 'MSFT', 'GOOGL'])"

# 2. Regenerate features (repeat Step 2 above)

# 3. Retrain (repeat Step 3 above)
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the right environment
conda activate us-stock-app

# Reinstall if needed
conda env update -f environment.yml --prune
```

### No Data Showing in Dashboard
- Run `python run_pipeline.py` first to generate data
- Or manually execute Steps 1-3 above

### PyPortfolioOpt Errors
Check that you have sufficient history and at least 2 symbols:
```python
# In config.yaml:
portfolio:
  pypfopt:
    cov_lookback_days: 60  # Reduce if you have limited history
```

### Memory Issues with Large Universe
Start with fewer symbols:
```python
symbols = load_sp500_constituents()[:20]  # Start with 20
```

## Next Steps

1. **Customize**: Modify configs to match your trading style
2. **Backtest**: Run on different time periods/parameters
3. **Monitor**: Track model IC over time using dashboard
4. **Paper Trade**: Use `dry_run=True` in live operations
5. **Scale**: Gradually expand universe as you gain confidence

## Support

- See `README.md` for full documentation
- Check `CLAUDE.md` for architecture details
- Review example notebooks (coming soon) in `notebooks/`
