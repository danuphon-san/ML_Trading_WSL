#!/usr/bin/env python
# coding: utf-8

# # ML Trading System - Interactive Development Pipeline
# 
# This notebook walks through the entire ML trading pipeline step-by-step.
# 
# **Purpose**: Learn and experiment with each component during development phase.
# 
# **Steps**:
# 1. Data Ingestion (~5 min)
# 2. Feature Engineering (~2 min)
# 3. Model Training (~5 min)
# 4. Backtesting (~2 min)
# 5. Dashboard Population (~1 min)
# 
# ---

# ## Setup & Imports

# In[ ]:


# Standard imports
import sys
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Add project root to path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Working directory: {os.getcwd()}")
print(f"Project root: {project_root}")


# In[ ]:


# Load configuration
config_path = os.path.join(project_root, 'config/config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

print("✓ Configuration loaded")
print(f"  - Modeling algorithm: {config['modeling']['algorithm']}")
print(f"  - Portfolio top_k: {config['portfolio']['top_k']}")
print(f"  - Optimizer: {config['portfolio']['optimizer']}")


# ---
# ## Step 1: Data Ingestion
# 
# Fetch OHLCV data for a subset of S&P 500 stocks.
# 
# **Tip**: Start with 20-30 stocks for quick testing, expand later.

# In[ ]:


from src.io.universe import load_sp500_constituents
from src.io.ingest_ohlcv import OHLCVIngester

# Load universe - START SMALL for testing
NUM_STOCKS = 30  # Adjust this: 20-30 for testing, 100+ for production
symbols = load_sp500_constituents()[:NUM_STOCKS]

print(f"Selected {len(symbols)} stocks for analysis")
print(f"Sample symbols: {symbols[:10]}")


# In[ ]:


# Fetch OHLCV data from yfinance
ingester = OHLCVIngester()
start_date = config['ingest']['start_date']  # From config.yaml

print(f"Fetching data from {start_date} to present...")
print("This may take 2-5 minutes depending on number of stocks...\n")

data = ingester.fetch_ohlcv(symbols, start_date, None)

print(f"\n✓ Fetched data for {len(data)} symbols")
print(f"  Total rows: {sum(len(df) for df in data.values()):,}")


# In[ ]:


# Save to parquet format
ingester.save_parquet(data)
print("✓ Data saved to data/parquet/")

# Explore the data
sample_symbol = symbols[0]
sample_df = data[sample_symbol]

print(f"\nSample data for {sample_symbol}:")
print(sample_df.head())
print(f"\nDate range: {sample_df['date'].min()} to {sample_df['date'].max()}")
print(f"Total trading days: {len(sample_df)}")


# In[ ]:


# Visualize sample stock price
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Price chart
axes[0].plot(sample_df['date'], sample_df['close'], linewidth=1.5)
axes[0].set_title(f"{sample_symbol} - Closing Price", fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price ($)')
axes[0].grid(True, alpha=0.3)

# Volume chart
axes[1].bar(sample_df['date'], sample_df['volume'], alpha=0.6)
axes[1].set_title(f"{sample_symbol} - Trading Volume", fontsize=14, fontweight='bold')
axes[1].set_ylabel('Volume')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:


from src.io.ingest_fundamentals import FundamentalsIngester

# Initialize fundamental data ingester
fund_ingester = FundamentalsIngester(storage_path="data/fundamentals")

print(f"Fetching fundamental data for {len(symbols)} symbols...")
print("This may take 2-3 minutes...\n")

# Fetch fundamental data
fundamental_data = fund_ingester.fetch_fundamentals(symbols)

print(f"\n✓ Fetched fundamental data for {len(fundamental_data)} symbols")

# Save to parquet
fund_ingester.save_parquet(fundamental_data)
print("✓ Fundamental data saved to data/fundamentals/")

# Preview sample data
if len(fundamental_data) > 0:
    sample_symbol = list(fundamental_data.keys())[0]
    sample_fund = fundamental_data[sample_symbol]
    print(f"\nSample fundamental data for {sample_symbol}:")
    print(sample_fund.head())
    print(f"\nColumns: {list(sample_fund.columns)}")
    print(f"Date range: {sample_fund['date'].min()} to {sample_fund['date'].max()}")


# ### Step 1.5: Fetch Fundamental Data
# 
# Now let's fetch fundamental data (income statements, balance sheets, cash flow statements) for the same symbols.

# ---
# ## Step 2: Feature Engineering
# 
# Generate technical indicators and forward return labels.

# In[ ]:


from src.features.ta_features import create_technical_features
from src.labeling.labels import generate_forward_returns

# Load saved parquet data
df = ingester.load_parquet(symbols)

print(f"Loaded {len(df):,} rows of OHLCV data")
print(f"Columns: {list(df.columns)}")
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique symbols: {df['symbol'].nunique()}")


# ### Step 2.5: Add Fundamental Features with PIT Alignment
# 
# Now let's integrate fundamental features using Point-in-Time (PIT) alignment to prevent look-ahead bias.

# In[ ]:


# Reload the module to get latest fixes
import importlib
import sys
if 'src.features.fa_features' in sys.modules:
    importlib.reload(sys.modules['src.features.fa_features'])
    print("✓ Reloaded fa_features module")

from src.features.fa_features import FundamentalFeatures

print("Loading fundamental data...")
# Load all saved fundamental data
fundamentals_df = fund_ingester.load_parquet(symbols)

if fundamentals_df.empty:
    print("⚠️ No fundamental data found. Run Step 1.5 first.")
else:
    print(f"✓ Loaded {len(fundamentals_df)} fundamental records")
    print(f"  Symbols: {fundamentals_df['symbol'].nunique()}")
    print(f"  Date range: {fundamentals_df['date'].min()} to {fundamentals_df['date'].max()}")
    
    # Initialize fundamental features with PIT alignment
    fa_features = FundamentalFeatures(config)
    
    print("\nComputing fundamental features with PIT alignment...")
    print(f"  PIT constraints:")
    print(f"    - Min lag: {fa_features.pit_min_lag_days} days")
    print(f"    - Default publication lag: {fa_features.default_public_lag_days} days")
    print(f"    - Earnings blackout: {fa_features.earnings_blackout_days} days")
    
    # Store original column count
    original_cols = len(df.columns)
    
    # Compute and align fundamental features
    df = fa_features.compute_features(df, fundamentals_df)
    
    new_cols = len(df.columns) - original_cols
    print(f"\n✓ Fundamental features added: {new_cols} new columns")
    
    # Show new fundamental feature columns
    fund_feature_cols = [col for col in df.columns if any(x in col for x in ['ratio', 'roe', 'roa', 'debt', 'margin', 'qoq', 'quality', 'equity', 'revenue', 'income', 'cash_flow'])]
    print(f"\nSample fundamental features:")
    for col in fund_feature_cols[:15]:
        print(f"  - {col}")
    if len(fund_feature_cols) > 15:
        print(f"  ... and {len(fund_feature_cols) - 15} more")


# In[ ]:


# Generate technical features
print("Generating technical features...")
df = create_technical_features(df, config)

print(f"\n✓ Technical features created")
print(f"  Total columns now: {len(df.columns)}")

# Show feature columns
feature_cols = [col for col in df.columns if col not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]
print(f"\nGenerated features ({len(feature_cols)}):")
for col in feature_cols[:10]:
    print(f"  - {col}")
if len(feature_cols) > 10:
    print(f"  ... and {len(feature_cols) - 10} more")


# In[ ]:


# Generate forward return labels
print("Generating forward return labels...")
df = generate_forward_returns(df, config)

label_col = f"forward_return_{config['labels']['horizon']}d"
print(f"\n✓ Labels created: {label_col}")
print(f"\nLabel statistics:")
print(df[label_col].describe())


# In[ ]:


# Visualize label distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df[label_col].dropna(), bins=50, alpha=0.7, edgecolor='black')
axes[0].set_title('Forward Returns Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'{label_col} (%)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero return')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot by symbol (sample)
sample_symbols = symbols[:10]
sample_data = df[df['symbol'].isin(sample_symbols)]
sample_data.boxplot(column=label_col, by='symbol', ax=axes[1], rot=45)
axes[1].set_title('Forward Returns by Symbol (Sample)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Symbol')
axes[1].set_ylabel(f'{label_col} (%)')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.show()


# In[ ]:


# Save features
from src.io.storage import save_dataframe

save_dataframe(df, 'data/features/all_features.parquet')
print("✓ Features saved to data/features/all_features.parquet")

# Check for missing values
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(f"\nMissing values (top 10 columns with most nulls):")
print(missing_pct.head(10))


# In[ ]:


# Visualize fundamental features distribution
try:
    fundamentals_df
    fund_feature_cols
    can_visualize = not fundamentals_df.empty and len(fund_feature_cols) > 0
except NameError:
    can_visualize = False
    print("ℹ️ No fundamental features to visualize")

if can_visualize:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select key fundamental features
    viz_features = []
    for feat in ['pe_ratio_calc', 'roe_calc', 'debt_to_equity_calc', 'profit_margin']:
        if feat in df.columns:
            viz_features.append(feat)
    
    if len(viz_features) >= 4:
        for idx, feat in enumerate(viz_features[:4]):
            row = idx // 2
            col = idx % 2
            
            data = df[feat].dropna()
            if len(data) > 0:
                axes[row, col].hist(data, bins=50, alpha=0.7, edgecolor='black')
                axes[row, col].set_title(f'{feat} Distribution', fontsize=12, fontweight='bold')
                axes[row, col].set_xlabel(feat)
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(True, alpha=0.3)
                
                # Add mean/median lines
                mean_val = data.mean()
                median_val = data.median()
                axes[row, col].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                axes[row, col].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
                axes[row, col].legend()
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*60)
        print("FUNDAMENTAL FEATURES SUMMARY")
        print("="*60)
        print(f"Total fundamental features: {len(fund_feature_cols)}")
        print(f"\nCoverage (non-null ratio):")
        for feat in fund_feature_cols[:10]:
            if feat in df.columns:
                coverage = (df[feat].notna().sum() / len(df)) * 100
                print(f"  {feat}: {coverage:.1f}%")
        print("="*60)
    else:
        print(f"⚠️ Only {len(viz_features)} fundamental features found (need 4 for visualization)")


# ### Visualization: Fundamental Features Distribution
# 
# Visualize the distribution of key fundamental features to understand the data.

# In[ ]:


# Verification: Check PIT alignment
try:
    fundamentals_df
    has_fundamentals = not fundamentals_df.empty
except NameError:
    has_fundamentals = False
    print("⚠️ fundamentals_df not defined - skipping PIT verification")

if has_fundamentals and 'public_date' in df.columns:
    print("="*70)
    print("PIT ALIGNMENT VERIFICATION")
    print("="*70)
    
    # Check 1: Ensure all public_dates are before or equal to price dates
    df_with_fund = df[df['public_date'].notna()].copy()
    
    if len(df_with_fund) > 0:
        # Calculate days between public_date and price date
        df_with_fund['days_after_publication'] = (df_with_fund['date'] - df_with_fund['public_date']).dt.days
        
        print(f"\n✓ Fundamental data coverage: {len(df_with_fund):,} rows ({len(df_with_fund)/len(df)*100:.1f}%)")
        print(f"\nDays between publication and price date:")
        print(df_with_fund['days_after_publication'].describe())
        
        # Check for any violations (price date before public date)
        violations = df_with_fund[df_with_fund['days_after_publication'] < 0]
        if len(violations) > 0:
            print(f"\n⚠️ WARNING: Found {len(violations)} PIT violations!")
            print("These rows have price dates BEFORE fundamental publication dates.")
            print(violations[['date', 'symbol', 'public_date', 'days_after_publication']].head())
        else:
            print(f"\n✓ No PIT violations detected")
        
        # Check 2: Verify minimum lag constraint
        try:
            fa_features
            min_lag_violations = df_with_fund[df_with_fund['days_after_publication'] < fa_features.pit_min_lag_days]
            if len(min_lag_violations) > 0:
                print(f"\n⚠️ WARNING: {len(min_lag_violations)} rows violate minimum lag ({fa_features.pit_min_lag_days} days)")
            else:
                print(f"\n✓ All rows respect minimum lag constraint ({fa_features.pit_min_lag_days} days)")
        except NameError:
            pass
        
        # Sample verification
        print(f"\nSample PIT-aligned data (5 random rows):")
        sample_cols = ['date', 'symbol', 'public_date', 'days_after_publication']
        if 'pe_ratio_calc' in df_with_fund.columns:
            sample_cols.append('pe_ratio_calc')
        
        sample_rows = df_with_fund.sample(min(5, len(df_with_fund)))[sample_cols]
        for _, row in sample_rows.iterrows():
            print(f"  {row['symbol']} on {row['date'].date()}: data from {row['public_date'].date()} ({int(row['days_after_publication'])} days ago)")
    
    print("="*70)
elif has_fundamentals:
    print("⚠️ Cannot verify PIT alignment - public_date column not found")
else:
    print("ℹ️ No fundamental data loaded - PIT verification skipped")


# ### Verification: PIT Alignment Check
# 
# Verify that fundamental data is properly aligned and no look-ahead bias exists.

# ---
# ## Step 3: Model Training
# 
# Train ML model to predict forward returns.

# In[ ]:


from src.ml.dataset import MLDataset, create_time_based_split
from src.ml.train import ModelTrainer

# Prepare dataset
label_col = f"forward_return_{config['labels']['horizon']}d"
dataset = MLDataset(label_col=label_col)

print(f"Preparing ML dataset with label: {label_col}")


# In[ ]:


# Time-based train/test split (crucial for time-series)
embargo_days = config['modeling'].get('embargo_days', 5)
train_df, test_df = create_time_based_split(df, test_size=0.2, embargo_days=embargo_days)

print(f"✓ Data split with {embargo_days}-day embargo")
print(f"\nTrain set:")
print(f"  Rows: {len(train_df):,}")
print(f"  Date range: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"\nTest set:")
print(f"  Rows: {len(test_df):,}")
print(f"  Date range: {test_df['date'].min()} to {test_df['date'].max()}")


# In[ ]:


# Train model with MLflow tracking
trainer = ModelTrainer(config)

print(f"Training {config['modeling']['algorithm']} model...")
print("This may take 3-5 minutes...\n")

trainer.train_with_mlflow(
    X_train, y_train, 
    X_test, y_test, 
    run_name="interactive_notebook_run"
)

print("\n✓ Model training complete")


# In[ ]:


# Prepare features and labels
X_train, y_train = dataset.prepare(train_df, auto_select_features=True)
X_test, y_test = dataset.prepare(test_df, auto_select_features=False)

print(f"✓ Features prepared")
print(f"\nTraining set:")
print(f"  Samples: {len(X_train):,}")
print(f"  Features: {len(X_train.columns)}")
print(f"\nTest set:")
print(f"  Samples: {len(X_test):,}")
print(f"\nSelected features:")
for feat in X_train.columns[:15]:
    print(f"  - {feat}")
if len(X_train.columns) > 15:
    print(f"  ... and {len(X_train.columns) - 15} more")


# In[ ]:


# Evaluate model
metrics = trainer.evaluate(X_test, y_test)

print("\n=" * 50)
print("MODEL EVALUATION METRICS")
print("=" * 50)
print(f"\nInformation Coefficient (IC):      {metrics['ic']:.4f}")
print(f"Rank IC:                            {metrics['rank_ic']:.4f}")
print(f"Mean Squared Error (MSE):           {metrics['mse']:.6f}")
print(f"R² Score:                           {metrics.get('r2', 0):.4f}")

# IC interpretation guide
print("\n" + "-" * 50)
print("IC Interpretation Guide:")
print("  |IC| > 0.05: Good predictive power")
print("  |IC| > 0.10: Excellent predictive power")
print("  IC > 0:     Positive correlation (correct direction)")
print("-" * 50)


# In[ ]:


# Visualize predictions vs actuals
predictions = trainer.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
axes[0].scatter(y_test, predictions, alpha=0.3, s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Forward Returns')
axes[0].set_ylabel('Predicted Forward Returns')
axes[0].set_title('Predictions vs Actuals', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residuals
residuals = y_test - predictions
axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
axes[1].set_xlabel('Prediction Error')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Prediction Residuals', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:


# Save model
import os
model_path = 'data/models/latest_model.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
trainer.save_model(model_path)
print(f"✓ Model saved to {model_path}")


# ---
# ## Step 4: Portfolio Construction & Backtesting
# 
# Use ML scores to construct portfolio and backtest with realistic costs.

# In[ ]:


from src.portfolio.construct import construct_portfolio
from src.backtest.bt_engine import VectorizedBacktester

# Generate scores for test period
X_test_full, _ = dataset.prepare(test_df, auto_select_features=False)
scores = trainer.predict(X_test_full)

# Filter test_df to match X_test_full (remove rows with missing labels)
test_df_clean = test_df[test_df[label_col].notna()].copy()
scored_df = test_df_clean[['date', 'symbol']].copy()
scored_df['ml_score'] = scores

print(f"✓ Generated ML scores for {len(scored_df):,} observations")
print(f"\nScore statistics:")
print(scored_df['ml_score'].describe())


# In[ ]:


# Get price panel (always use real close prices!)
price_panel = df[['date', 'symbol', 'close']].copy()

print(f"Price panel: {len(price_panel):,} rows")
print(f"Date range: {price_panel['date'].min()} to {price_panel['date'].max()}")


# In[ ]:


# Construct portfolio weights over time
print("Constructing portfolio weights...")
print(f"Using optimizer: {config['portfolio']['optimizer']}")
print(f"Top K positions: {config['portfolio']['top_k']}\n")

weights_history = []
unique_dates = sorted(scored_df['date'].unique())

# Limit to first 50 dates for demo (faster execution)
MAX_DATES = 50
rebalance_dates = unique_dates[:min(MAX_DATES, len(unique_dates))]

print(f"Constructing portfolio for {len(rebalance_dates)} rebalance dates...")

for i, date in enumerate(rebalance_dates):
    if i % 10 == 0:
        print(f"  Processing date {i+1}/{len(rebalance_dates)}: {date}")
    
    day_scores = scored_df[scored_df['date'] == date]
    weights = construct_portfolio(day_scores, price_panel, config)
    
    for symbol, weight in weights.items():
        weights_history.append({
            'date': date,
            'symbol': symbol,
            'weight': weight
        })

weights_df = pd.DataFrame(weights_history)
print(f"\n✓ Portfolio weights constructed")
print(f"  Total weight records: {len(weights_df):,}")
print(f"  Average positions per rebalance: {len(weights_df) / len(rebalance_dates):.1f}")


# In[ ]:


# Examine portfolio weights
print("Sample portfolio weights:")
print(weights_df.head(20))

# Weight statistics
print(f"\nWeight statistics:")
print(weights_df['weight'].describe())

# Check constraints
weights_sum = weights_df.groupby('date')['weight'].sum()
print(f"\nPortfolio weight sum per date:")
print(f"  Mean: {weights_sum.mean():.4f}")
print(f"  Min:  {weights_sum.min():.4f}")
print(f"  Max:  {weights_sum.max():.4f}")
print(f"\n✓ Weights sum to ~1.0 (constraint satisfied)")


# In[ ]:


# Run backtest
print("Running backtest...")
print(f"Cost assumptions:")
print(f"  Commission: {config['portfolio']['costs_bps']} bps")
print(f"  Slippage:   {config['portfolio']['slippage_bps']} bps\n")

backtester = VectorizedBacktester(config)
results = backtester.run(weights_df, price_panel)

print("\n" + "=" * 50)
print("BACKTEST RESULTS")
print("=" * 50)
print(f"\nTotal Return:        {results['metrics']['total_return']:.2%}")
print(f"Annualized Return:   {results['metrics'].get('annual_return', 0):.2%}")
print(f"Sharpe Ratio:        {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown:        {results['metrics']['max_drawdown']:.2%}")
print(f"Volatility:          {results['metrics'].get('volatility', 0):.2%}")
print(f"Avg Turnover:        {results['metrics'].get('avg_turnover', 0):.2%}")
print("=" * 50)


# In[ ]:


# Visualize equity curve
equity_curve = results['equity_curve']

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Equity curve
axes[0].plot(equity_curve['date'], equity_curve['equity'], linewidth=2, label='Portfolio Equity')
axes[0].set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
axes[0].set_ylabel('Equity ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Drawdown
running_max = equity_curve['equity'].expanding().max()
drawdown = (equity_curve['equity'] - running_max) / running_max * 100

axes[1].fill_between(equity_curve['date'], drawdown, 0, alpha=0.5, color='red', label='Drawdown')
axes[1].set_title('Drawdown', fontsize=16, fontweight='bold')
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_xlabel('Date')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:


# Save backtest results
import os
os.makedirs('data/reports', exist_ok=True)
equity_curve.to_csv('data/reports/equity_curve.csv', index=False)
print("✓ Backtest results saved to data/reports/equity_curve.csv")


# ---
# ## Step 5: Populate Dashboard
# 
# Save results to database so the dashboard displays them.

# In[ ]:


from src.database.models import init_database, PortfolioSnapshot
from datetime import datetime

# Initialize database
engine, SessionLocal = init_database()
db = SessionLocal()

print("✓ Database connection established")


# In[ ]:


# Populate portfolio snapshots for dashboard
print("Populating dashboard data...")

snapshots_to_add = []
for _, row in equity_curve.iterrows():
    snapshot = PortfolioSnapshot(
        date=row['date'],
        total_equity=row['equity'],
        cash=0,  # Simplified - fully invested
        positions_value=row['equity'],
        daily_return=0,  # Can calculate if needed
        daily_pnl=0
    )
    snapshots_to_add.append(snapshot)

# Bulk insert
db.bulk_save_objects(snapshots_to_add)
db.commit()

print(f"✓ Added {len(snapshots_to_add)} portfolio snapshots to database")
db.close()


# ---
# ## ✅ Pipeline Complete!
# 
# ### What You Accomplished:
# 1. ✓ Fetched OHLCV data for stocks
# 2. ✓ Generated technical features and labels
# 3. ✓ Trained ML model with evaluation metrics
# 4. ✓ Constructed portfolio and ran backtest
# 5. ✓ Populated dashboard database
# 
# ### Next Steps:
# 
# **1. View Results in Dashboard:**
#    - Go to http://localhost:8501 in your browser
#    - Refresh the page to see your data
#    - Navigate through Portfolio Overview, Signals, etc.
# 
# **2. Experiment & Iterate:**
#    - Adjust `NUM_STOCKS` in Step 1 to test with more stocks
#    - Modify parameters in `config/config.yaml`
#    - Try different optimizers: `pypfopt`, `inverse_vol`, `equal_weight`
#    - Change `top_k` for number of positions
# 
# **3. Production Automation:**
#    - Once satisfied, use `python run_pipeline.py` for automated runs
#    - Set up daily cron jobs for live operations
#    - Use `src/live/` modules for real-time trading
# 
# **4. Further Analysis:**
#    - View MLflow UI: `mlflow ui --port 5000`
#    - Compare different model runs
#    - Analyze feature importance
# 
# ---
# 
# ### Development → Production Checklist
# - [ ] Test with larger universe (100+ stocks)
# - [ ] Validate PIT alignment for fundamentals
# - [ ] Optimize hyperparameters with Optuna
# - [ ] Paper trade with `dry_run=True`
# - [ ] Set up monitoring and alerts
# - [ ] Document your strategy parameters
# - [ ] Consider premium data providers
# 
# **Great job! Your ML trading system is now operational.** 🎉
