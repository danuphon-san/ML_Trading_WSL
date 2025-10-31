# Daily Live Trading - Quick Start Guide

**Status**: ✅ Option A Implementation Complete
**Version**: 1.0
**Date**: 2025-10-30

---

## 🚀 Quick Start (5 minutes)

### **1. First Time Setup**

```bash
# Activate environment
conda activate us-stock-app

# Ensure you have trained a model
python run_core_pipeline.py --steps 1-7

# Test dry-run mode
python scripts/daily_live_runner.py --dry-run
```

### **2. Daily Usage**

```bash
# Run daily (default: dry-run mode)
python scripts/daily_live_runner.py

# View results
ls live/$(date +%Y-%m-%d)/
# Output:
#   trades.csv              - Execute these trades
#   portfolio_weights.csv   - Target allocation
#   signals.json            - ML scores
#   report.html             - Daily dashboard
#   monitoring_log.json     - Model health metrics
```

### **3. View Daily Report**

```bash
# Open in browser
open live/$(date +%Y-%m-%d)/report.html

# Or view trades
cat live/$(date +%Y-%m-%d)/trades.csv
```

---

## 📋 What It Does

The daily runner executes this workflow automatically:

```
1. ✓ Check market is open (skip weekends/holidays)
2. ✓ Update data (last 7 days OHLCV + fundamentals)
3. ✓ Load champion model (production/champion_model.pkl)
4. ✓ Score universe (~500 stocks with ML model)
5. ✓ Detect market regime (Risk-Off/Normal/Risk-On)
6. ✓ Construct portfolio (regime-aware, PyPortfolioOpt)
7. ✓ Generate trade orders (buy/sell recommendations)
8. ✓ Run safety checks (kill-switch, position limits)
9. ✓ Save outputs (CSV + HTML + JSON)
10. ✓ Generate dashboard (beautiful HTML report)
```

**Execution time**: ~5 minutes for 500 stocks

---

## 🎛️ Command Line Options

### **Basic Usage**

```bash
# Dry-run (no real trades) - DEFAULT
python scripts/daily_live_runner.py --dry-run

# Production mode (requires approval)
python scripts/daily_live_runner.py --mode production

# Specific date (backfill)
python scripts/daily_live_runner.py --date 2025-10-29

# Skip data update (use existing data)
python scripts/daily_live_runner.py --skip-data-update

# Override capital
python scripts/daily_live_runner.py --capital 50000

# Verbose logging
python scripts/daily_live_runner.py --verbose
```

### **Full Options**

```
--mode              dry-run or production (default: dry-run)
--dry-run           Shorthand for --mode dry-run
--date              Specific date YYYY-MM-DD (default: today)
--skip-data-update  Skip data refresh step
--config            Path to config.yaml (default: config/config.yaml)
--capital           Override portfolio capital in USD
--email             Send email notification (not yet implemented)
--verbose, -v       Debug logging
```

---

## 📊 Understanding the Outputs

### **1. trades.csv** - Execution Instructions

```csv
symbol,side,shares,price,notional,weight_change,commission_est,slippage_est
AAPL,BUY,50,175.23,8761.50,0.0450,0.88,1.75
MSFT,SELL,30,380.45,-11413.50,-0.0523,1.14,2.28
```

**Columns**:
- `symbol`: Stock ticker
- `side`: BUY or SELL
- `shares`: Number of shares to trade
- `price`: Current market price
- `notional`: Dollar value of trade
- `weight_change`: Change in portfolio weight
- `commission_est`: Estimated commission ($)
- `slippage_est`: Estimated slippage ($)

**Actions**:
- **BUY**: Purchase these shares
- **SELL**: Sell these shares

### **2. portfolio_weights.csv** - Target Allocation

```csv
symbol,weight
AAPL,0.0523
MSFT,0.0487
GOOGL,0.0451
...
```

**Sum of weights**: Should equal ~1.0 (100%)

**Cash allocation**: If sum < 1.0, remainder is cash (Risk-Off protection)

### **3. report.html** - Daily Dashboard

Open in browser to see:
- Market regime (Risk-Off/Normal/Risk-On)
- Portfolio summary (positions, trades)
- Trade orders table
- Safety check status

### **4. signals.json** - ML Scores (Audit Trail)

```json
{
  "date": "2025-10-30",
  "scores": [
    {"symbol": "AAPL", "ml_score": 0.8234},
    {"symbol": "MSFT", "ml_score": 0.7891},
    ...
  ]
}
```

**Use for**: Model performance tracking, auditing

### **5. monitoring_log.json** - Model Health

```json
{
  "date": "2025-10-30",
  "regime": {
    "regime_name": "Normal",
    "risk_multiplier": 1.0
  },
  "safety": {
    "overall_valid": true,
    "violations": []
  }
}
```

**Key metrics**: Regime, safety status, portfolio stats

---

## ⚙️ Configuration

All settings are in `config/config.yaml`:

### **Live Trading Settings**

```yaml
live:
  enabled: false              # Set true for production
  dry_run: true               # Set false for real trades
  initial_capital: 100000     # Starting capital
  min_trade_size: 100         # Min $100 trades
  max_position_pct: 0.15      # Max 15% per stock
  enable_regime_adaptation: true
```

### **Safety Settings**

```yaml
ops:
  kill_switch:
    enabled: true
    max_daily_loss_pct: 0.03  # Halt if lose >3%
    min_live_sharpe_threshold: 0.5
```

### **Risk Limits**

```yaml
risk:
  max_sector_weight: 0.30     # Max 30% per sector
```

---

## 🛡️ Safety Checks

The runner automatically validates:

### **Portfolio Constraints**
- ✓ Weights sum to ~1.0 (within 1%)
- ✓ No position > 15% (max_position_pct)
- ✓ All weights >= 0 (long-only)
- ✓ No sector > 30% (max_sector_weight)

### **Trade Validation**
- ✓ Turnover < 35% (max_turnover)
- ✓ No single trade > 10% of portfolio
- ✓ All prices are valid (> 0)
- ✓ Total costs < 0.5% of portfolio

### **Kill-Switch Protection**
- ✓ Daily loss < 3% (max_daily_loss_pct)
- ✓ 6-week Sharpe > 0.5 (min_live_sharpe_threshold)

**If any check fails**: Script exits with error, no trades executed

---

## 📈 Regime Adaptation

The system automatically adjusts based on market conditions:

| Regime | Condition | Exposure | Top-K | Action |
|--------|-----------|----------|-------|--------|
| **Risk-Off** | High vol, drawdown, downtrend | 50% | 60% | Hold cash, concentrate |
| **Normal** | Moderate vol, stable | 100% | 100% | Full investment |
| **Risk-On** | Low vol, uptrend | 120%→100% | 130% | Increase diversification |

**Example**: In Risk-Off regime with 20 positions normally:
- Select top 12 stocks (20 × 0.6)
- Invest only 50% of capital
- Hold 50% cash for safety

---

## 🔄 Daily Workflow

### **Recommended Schedule**

```bash
# Cron job (runs at 4:05 PM EST after market close)
5 16 * * 1-5 cd /path/to/ML_Trading_WSL && conda activate us-stock-app && python scripts/daily_live_runner.py
```

### **Manual Workflow**

```bash
# 1. Run daily (after market close)
python scripts/daily_live_runner.py

# 2. Review report
open live/$(date +%Y-%m-%d)/report.html

# 3. Review trades
cat live/$(date +%Y-%m-%d)/trades.csv

# 4. Check safety
grep "overall_valid" live/$(date +%Y-%m-%d)/monitoring_log.json

# 5. Execute trades (if approved)
#    - Manually via broker
#    - Or automate via broker API (future)
```

---

## 🧪 Testing & Validation

### **Test 1: Dry-Run Mode**

```bash
# Run without real trades
python scripts/daily_live_runner.py --dry-run

# Check outputs
ls live/$(date +%Y-%m-%d)/
```

**Expected**:
- ✓ Script completes without errors
- ✓ Files created in live/ directory
- ✓ HTML report opens in browser
- ✓ trades.csv shows recommendations

### **Test 2: Historical Date**

```bash
# Test on past date (uses historical data)
python scripts/daily_live_runner.py --date 2025-10-15

# Compare to actual market performance
cat live/2025-10-15/trades.csv
```

### **Test 3: Safety Checks**

```bash
# Test with different capital levels
python scripts/daily_live_runner.py --capital 10000   # Small
python scripts/daily_live_runner.py --capital 1000000  # Large

# Verify position sizes scale correctly
```

---

## 🐛 Troubleshooting

### **Issue 1: "Market is closed"**

```
⚠️  Market is closed - exiting
```

**Cause**: Script run on weekend/holiday
**Fix**: Only run Monday-Friday on trading days

### **Issue 2: "Failed to load model"**

```
❌ FileNotFoundError: No model found at production/champion_model.pkl
```

**Cause**: No champion model selected
**Fix**:
```bash
# Train model first
python run_core_pipeline.py --steps 7

# Or copy latest to production
mkdir -p production
cp data/models/latest/model.pkl production/champion_model.pkl
```

### **Issue 3: "Failed to load features"**

```
❌ Features file not found: data/features/all_features_with_fundamentals.parquet
```

**Cause**: Features not generated
**Fix**:
```bash
# Generate features
python run_core_pipeline.py --steps 1-4
```

### **Issue 4: "No trades generated"**

```
✓ No trades needed (portfolio within tolerance)
```

**Cause**: Portfolio already at target weights (normal behavior)
**Action**: No action needed

### **Issue 5: "Safety validation FAILED"**

```
❌ SAFETY VALIDATION FAILED: 3 violations
  - Turnover 45.2% exceeds limit 35.0%
```

**Cause**: Too much portfolio turnover
**Action**: Review trades, consider adjusting turnover limit in config.yaml

---

## 📊 Model Health Monitoring

The system tracks these metrics daily:

### **Information Coefficient (IC)**
- **Target**: IC > 0.05 (strong predictive power)
- **Warning**: IC < 0.02 (review model)
- **Critical**: IC < 0.01 (retrain immediately)

### **Model Drift**
- Tracks avg IC over 20 days
- Alerts if >50% of days below threshold
- **Action**: Retrain model if degraded

### **Regime Performance**
- Validates model works in all regimes
- Alerts if negative Sharpe in any regime
- **Action**: Review strategy if failing in multiple regimes

**View health status**: Check `monitoring_log.json` daily

---

## 🎯 Next Steps

### **Week 1-2: Paper Trading**
```bash
# Run daily in dry-run mode
python scripts/daily_live_runner.py --dry-run

# Track performance vs. actual market
# Validate model predictions
```

### **Week 3: Validate Performance**
```bash
# Compare predictions to actual returns
# Check IC, turnover, safety violations
# Tune parameters if needed
```

### **Week 4+: Go Live (when ready)**
```yaml
# Edit config/config.yaml:
live:
  enabled: true
  dry_run: false  # ⚠️  Real trades!

# Run with human approval
python scripts/daily_live_runner.py --mode production
```

---

## 📞 Support

**Issues**:
- Check logs: `logs/daily_live_runner.log`
- Review outputs: `live/YYYY-MM-DD/monitoring_log.json`
- Test with: `--dry-run --verbose --skip-data-update`

**Documentation**:
- Full details: `MODIFICATIONS_PLAN.md`
- Testing guide: `docs/REGIME_DETECTION_TESTING.md`
- Model selection: `docs/MODEL_SELECTION_GUIDE.md` (Option B)

---

## ✅ Success Checklist

Before going live, verify:

- [ ] Model trained and validated (Sharpe > 1.0, MaxDD < -20%)
- [ ] Dry-run completed for 20+ days
- [ ] Safety checks always pass
- [ ] IC consistently > 0.03
- [ ] Understand all trade recommendations
- [ ] Broker integration ready (manual or API)
- [ ] Emergency stop procedure documented
- [ ] Capital allocated and risk limits set

---

**Ready to start?** Run: `python scripts/daily_live_runner.py --dry-run`

🤖 **Powered by Claude Code**
