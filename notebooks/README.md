# Interactive Development Notebooks

This directory contains Jupyter notebooks for interactive development and experimentation with the ML Trading System.

## Quick Start

### 1. Launch Jupyter Notebook

```bash
# Make sure you're in the project root and conda environment is activated
cd /home/bawbawz/Project/MLTrading_WSL
conda activate us-stock-app

# Launch Jupyter
jupyter notebook
```

Your browser will open with the Jupyter interface.

### 2. Open the Interactive Pipeline

Navigate to: `notebooks/01_interactive_pipeline.ipynb`

### 3. Run the Cells

Execute cells one by one (Shift+Enter) to:
- Understand what each step does
- Modify parameters and experiment
- See intermediate outputs and visualizations

## Notebook Structure

### `01_interactive_pipeline.ipynb`
**Complete end-to-end ML trading pipeline**

**Sections:**
1. **Data Ingestion** (~5 min) - Fetch stock data
2. **Feature Engineering** (~2 min) - Generate 83+ technical indicators
3. **Model Training** (~5 min) - Train ML model
4. **Backtesting** (~2 min) - Portfolio construction & backtest
5. **Dashboard Population** (~1 min) - Save results to dashboard

**Key Features:**
- Starts with small universe (30 stocks) for quick testing
- Well-documented with explanations
- Visualizations at each step
- Can adjust parameters interactively
- Results populate the dashboard
- **Generates features for notebook 02**

---

### `02_optimizer_testing_matrix.ipynb` **✨ NEW**
**Systematic portfolio optimizer comparison**

**Purpose:** Test all 7 optimizers on same data to find best strategy

**Sections:**
1. **Load Features** - Uses output from notebook 01
2. **Generate ML Scores** - Train/load model
3. **Run Optimizer Comparison** - Test 7 optimizers systematically
4. **Results Analysis** - Metrics table + recommendations
5. **Visualizations** - Performance charts

**Optimizers Tested:**
- Equal Weight (baseline benchmark)
- Score-Weighted (ML confidence-based)
- Inverse Volatility (risk parity)
- MVO (mean-variance optimization)
- MVO_REG (regularized MVO with L2 penalty)
- HRP (hierarchical risk parity)
- Hybrid (score + risk blend) **← Recommended**

**Prerequisites:** ⚠️ Must run notebook 01 first to generate `data/features/all_features.parquet`

**Duration:** ~5-10 minutes

**Output:** Comparison table with Sharpe, returns, drawdown, turnover, concentration (HHI)

## Tips for Development

### Start Small, Scale Up
```python
# In Step 1 of notebook
NUM_STOCKS = 30  # Start here for testing
# Later increase to: 50, 100, 200, etc.
```

### Experiment with Parameters

Edit `config/config.yaml` and re-run notebook:
```yaml
portfolio:
  top_k: 20              # Number of positions
  optimizer: "pypfopt"   # Try: pypfopt, inverse_vol, equal_weight
  costs_bps: 1.0         # Adjust transaction costs

modeling:
  algorithm: "xgboost"   # Try: xgboost, random_forest
```

### Save Your Experiments

Create copies of notebooks for different experiments:
```bash
cp 01_interactive_pipeline.ipynb 03_experiment_large_universe.ipynb
cp 01_interactive_pipeline.ipynb 04_experiment_inverse_vol.ipynb
```

---

## 📋 Complete Development to Production Workflow

### Phase 1: Research & Feature Development
**Goal:** Generate features and understand data

**Steps:**
1. Launch Jupyter: `jupyter notebook`
2. Open `01_interactive_pipeline.ipynb`
3. Run all cells (Kernel → Restart & Run All)
4. Review feature statistics and ML model performance
5. Check outputs:
   - ✓ `data/features/all_features.parquet` created
   - ✓ `data/models/latest_model.pkl` saved
   - ✓ Model metrics (IC, Sharpe) logged

**Duration:** ~10-15 minutes

---

### Phase 2: Optimizer Selection
**Goal:** Find best portfolio construction method

**Steps:**
1. Ensure notebook 01 completed successfully
2. Open `02_optimizer_testing_matrix.ipynb`
3. Configure optimizers to test (edit cell):
   ```python
   optimizers_to_test = [
       'equal', 'score_weighted', 'inv_vol',
       'mvo', 'hybrid', 'hrp'
   ]
   ```
4. Run all cells (Kernel → Restart & Run All)
5. Review comparison table:
   - **Sharpe Ratio** (risk-adjusted return)
   - **Max Drawdown** (downside risk)
   - **Turnover** (transaction costs)
   - **HHI** (concentration/diversification)
6. Check recommendations section for best optimizer

**Duration:** ~5-10 minutes

**Example Output:**
```
OPTIMIZER RECOMMENDATIONS
========================================
🎯 Best Overall:
   → hybrid - Sharpe: 2.45, Return: 18.50%

🎯 Lowest Drawdown:
   → hrp - MaxDD: -8.20%

🎯 Most Diversified:
   → inv_vol - HHI: 0.0520, Effective N: 19.2
```

---

### Phase 3: Configuration Update
**Goal:** Update system with winning optimizer

**Steps:**

#### 1. Identify Best Optimizer from notebook 02 results
   - For **highest returns**: Use best Sharpe ratio
   - For **stability**: Use lowest drawdown
   - For **low costs**: Use lowest turnover
   - For **balanced approach**: Use hybrid

#### 2. Edit `config/config.yaml`:

   ```bash
   # Open config file
   nano config/config.yaml
   # or
   code config/config.yaml
   ```

#### 3. Update Portfolio Section:

   ```yaml
   portfolio:
     # Selection
     top_k: 20  # Keep or adjust based on results

     # ⚠️ CHANGE THIS based on notebook 02 results
     optimizer: "hybrid"  # Options: equal, score_weighted, inv_vol, mvo, hrp, hybrid

     # Configure chosen optimizer settings
     hybrid:  # Example if you chose hybrid
       score_weight: 0.5    # Balance between signal and risk
       risk_weight: 0.5     # Adjust based on preference
       lookback_days: 60

     # Constraints (fine-tune based on results)
     pypfopt:
       min_weight: 0.01   # 1% minimum per position
       max_weight: 0.15   # 15% maximum per position
   ```

#### 4. Optimizer-Specific Configuration Examples:

   **For Score-Weighted:**
   ```yaml
   optimizer: "score_weighted"
   # Uses ML scores directly, no additional config needed
   ```

   **For Inverse Volatility:**
   ```yaml
   optimizer: "inv_vol"
   inverse_vol:
     lookback_days: 60  # Volatility estimation window
   ```

   **For HRP:**
   ```yaml
   optimizer: "hrp"
   hrp:
     lookback_days: 252         # 1 year correlation
     linkage_method: "single"   # Options: single, average, ward
     score_tilt: 0.0           # Optional ML score overlay (0-1)
   ```

   **For Hybrid (Recommended):**
   ```yaml
   optimizer: "hybrid"
   hybrid:
     score_weight: 0.6    # 60% ML signal (if high IC)
     risk_weight: 0.4     # 40% risk balance
     lookback_days: 60
   ```

#### 5. Validate Configuration:

   Re-run notebook 01 with new optimizer:
   ```python
   # In notebook 01, check:
   print(config['portfolio']['optimizer'])  # Should show your chosen optimizer
   ```

#### 6. Save Configuration:
   ```bash
   git add config/config.yaml
   git commit -m "Update optimizer to hybrid based on testing matrix results"
   ```

---

### Phase 4: Validation Backtest
**Goal:** Confirm optimizer performs as expected

**Steps:**
1. Open `01_interactive_pipeline.ipynb`
2. Run full pipeline with updated config
3. Compare results with notebook 02:
   - Sharpe ratio should match
   - Weight distribution looks reasonable
   - Turnover is acceptable
4. Review equity curve and drawdown
5. If satisfied, proceed to production

**Duration:** ~10 minutes

---

### Phase 5: Production Deployment
**Goal:** Move from research to live/paper trading

#### **Option A: Paper Trading** ⚠️ **Recommended First**

1. **Enable Paper Trading:**
   ```yaml
   # config/config.yaml
   live:
     enabled: true
     dry_run: true        # ⚠️ Paper trading mode (no real money)
     execution_time: "09:35"  # After market open (EST)

     # Configure broker (paper account)
     broker: "alpaca"     # or "interactive_brokers"
     # Add broker API keys in .env file
   ```

2. **Create `.env` File** (never commit this):
   ```bash
   # .env
   ALPACA_API_KEY=your_paper_api_key
   ALPACA_SECRET_KEY=your_paper_secret_key
   ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading URL
   ```

3. **Test Live Module:**
   ```bash
   # Dry run - simulates trades without execution
   python -m src.live.daily_runner --dry-run
   ```

4. **Review Simulated Trades:**
   ```bash
   # Check logs
   tail -f logs/app.log

   # Check generated trades
   cat data/portfolio/trades.csv
   ```

5. **Monitor for 1-2 Weeks:**
   - Verify daily execution at scheduled time
   - Review trade signals
   - Check for errors in logs
   - Validate position sizes

#### **Option B: Automated Backtesting** (No Broker Required)

If not ready for live execution, automate daily backtesting:

1. **Create Automation Script:** `scripts/daily_backtest.py`
   ```python
   #!/usr/bin/env python3
   """Daily automated backtest"""
   import yaml
   from src.io.ingest_ohlcv import OHLCVIngester
   from src.features.ta_features import create_technical_features
   # ... import rest of pipeline ...

   # Run full pipeline
   # Save results with timestamp
   ```

2. **Schedule with Cron:**
   ```bash
   # crontab -e
   # Run daily at 6 PM EST (after market close)
   0 18 * * 1-5 cd /home/bawbawz/Project/MLTrading_WSL && /home/bawbawz/miniconda3/envs/us-stock-app/bin/python scripts/daily_backtest.py >> logs/cron.log 2>&1
   ```

#### **Option C: Live Trading** ⚠️ **Use with Extreme Caution**

**Only after successful paper trading for 2+ weeks:**

1. **Switch to Live Account:**
   ```yaml
   # config/config.yaml
   live:
     enabled: true
     dry_run: false      # ⚠️⚠️⚠️ REAL MONEY MODE
     execution_time: "09:35"

     # Risk limits for live trading
     risk_limits:
       max_position_size: 10000      # $10k per position
       max_daily_trades: 50
       max_portfolio_value: 100000   # $100k total
   ```

2. **Update `.env` with Live Credentials:**
   ```bash
   ALPACA_API_KEY=your_live_api_key
   ALPACA_SECRET_KEY=your_live_secret_key
   ALPACA_BASE_URL=https://api.alpaca.markets  # Live trading URL
   ```

3. **Enable Monitoring & Alerts:**
   ```yaml
   # config/config.yaml
   live:
     alerts:
       enabled: true
       email: your_email@example.com
       slack_webhook: your_slack_webhook_url
   ```

4. **Start Small:**
   - Begin with small capital allocation
   - Monitor first week closely
   - Gradually increase position sizes
   - Keep stop-loss limits active

5. **Daily Monitoring Checklist:**
   - [ ] Check execution logs
   - [ ] Verify positions match target weights
   - [ ] Review P&L vs backtest expectations
   - [ ] Monitor for errors or anomalies
   - [ ] Check data freshness

---

### Phase 6: Ongoing Maintenance

**Weekly:**
- Review portfolio performance
- Check model predictions vs actual returns
- Monitor turnover and costs

**Monthly:**
- Re-run optimizer testing matrix (notebook 02)
- Evaluate if optimizer still optimal
- Retrain ML model with recent data
- Update universe if needed

**Quarterly:**
- Full pipeline review
- Feature engineering updates
- Model architecture evaluation
- Risk parameter adjustments

---

## 🚀 Production Quick Start

### Prerequisites Checklist
Before going live, ensure:
- [x] Completed notebook 01 successfully
- [x] Ran notebook 02 and selected optimal optimizer
- [x] Updated config.yaml with chosen optimizer
- [x] Validated results match expectations
- [x] Reviewed Phase 5 deployment options above

### Production Options

**1. Paper Trading** (Recommended - see Phase 5, Option A):
```bash
# Configure paper trading in config.yaml (live.dry_run = true)
python -m src.live.daily_runner --dry-run
```

**2. Automated Backtesting** (No broker - see Phase 5, Option B):
```bash
# Schedule daily pipeline updates
python run_pipeline.py
```

**3. Live Trading** (After paper trading validation - see Phase 5, Option C):
```bash
# Configure live trading (live.dry_run = false)
# ⚠️ Review all safety limits in config.yaml first
python -m src.live.daily_runner
```

### Safety Guidelines
- **Always start with paper trading** for 2+ weeks minimum
- Use small position sizes initially
- Set strict risk limits in config.yaml
- Enable monitoring and alerts
- Never commit API keys (use .env file)
- Keep stop-loss mechanisms active
- Monitor daily for anomalies

## Common Issues & Solutions

### Import Errors
If you get `ModuleNotFoundError: No module named 'src'`:
```python
# Add this at the top of notebook (already included)
import sys
import os
project_root = os.path.abspath('..')
sys.path.insert(0, project_root)
```

### Data Download Slow
```python
# In Step 1, reduce number of stocks
NUM_STOCKS = 10  # Super fast for testing
```

### Memory Issues
```python
# Use fewer stocks or shorter date range
start_date = '2023-01-01'  # More recent data only
```

### Dashboard Not Showing Data
After running notebook, refresh your browser at http://localhost:8501

---

## 🎯 Quick Reference

### Optimizer Selection Guide

| Optimizer | When to Use | Pros | Cons |
|-----------|-------------|------|------|
| **hybrid** 🌟 | **Recommended for most cases** | Balances signal & risk, adaptive | Requires tuning |
| equal | Baseline benchmark testing | Simple, low turnover, stable | Ignores all signals |
| score_weighted | High IC models (IC>0.10) | Follows model confidence | Can be concentrated |
| inv_vol | Risk-focused strategies | Stable, diversified, low drawdown | Ignores return signals |
| mvo | Sharpe optimization | Theoretically optimal | Sensitive to estimation error |
| hrp | Noisy/volatile markets | Robust, stable over regimes | May sacrifice returns |

**IC (Information Coefficient)** = Correlation between predictions and actual returns
- IC > 0.05: Good predictive power
- IC > 0.10: Excellent predictive power

### Configuration Quick Edits

**Change Optimizer:**
```yaml
portfolio:
  optimizer: "hybrid"  # Change this line
```

**Adjust Risk Tolerance:**
```yaml
portfolio:
  pypfopt:
    min_weight: 0.01   # Increase for more concentration: 0.02+
    max_weight: 0.15   # Decrease for more diversification: 0.10
```

**Tune Hybrid Balance:**
```yaml
portfolio:
  hybrid:
    score_weight: 0.7  # Higher = more signal-driven (if high IC)
    risk_weight: 0.3   # Higher = more risk-balanced
```

### Common Config Updates Based on Results

**After High Sharpe → Reduce Risk:**
```yaml
portfolio:
  pypfopt:
    max_weight: 0.10  # Was 0.15, now more diversified
```

**After High Turnover → Reduce Trading Frequency:**
```yaml
backtest:
  rebalance_frequency: "monthly"  # Was "weekly"
```

**After High Concentration → Force Diversification:**
```yaml
portfolio:
  optimizer: "hrp"  # Or "inv_vol" for maximum diversification
```

**After High Drawdown → Lower Risk Exposure:**
```yaml
portfolio:
  pypfopt:
    max_weight: 0.08    # Lower concentration
    objective: "min_volatility"  # Was "max_sharpe"
```

---

## Next Steps

1. **Complete notebook 01** - Run all cells to generate features and initial results
2. **Run notebook 02** - Test all optimizers and identify best strategy
3. **Update config.yaml** - Apply winning optimizer settings
4. **Validate** - Re-run notebook 01 with new config
5. **Paper trade** - Test with broker paper account (Phase 5, Option A)
6. **Monitor & refine** - Track performance for 2+ weeks
7. **Go live** - Transition to real trading when confident (Phase 5, Option C)

## Resources

- [Feature Engineering Guide](../docs/Feature_Engineer_Instruct.md) - Bias-free feature design
- [Portfolio Optimization Guide](../docs/Portfolio_Weight_Optimization.md) - Optimizer testing matrix
- [QUICKSTART.md](../QUICKSTART.md) - Quick reference guide
- [README.md](../README.md) - Full system documentation
- [CLAUDE.md](../CLAUDE.md) - Architecture details
- [MLflow UI](http://localhost:5000) - View model experiments (run `mlflow ui`)

---

**Happy Trading! 📈**

**Remember:** Always start with paper trading. Never risk more than you can afford to lose.
