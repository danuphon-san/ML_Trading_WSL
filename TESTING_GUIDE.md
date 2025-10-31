# Option A Testing Guide

**Status**: Ready for Testing
**Environment**: Requires `us-stock-app` conda environment
**Duration**: 15-20 minutes for complete test

---

## 🎯 Testing Objectives

1. Verify daily_live_runner.py works end-to-end
2. Validate all outputs are generated correctly
3. Check safety validations trigger properly
4. Confirm regime detection integration
5. Test different execution modes

---

## 📋 Prerequisites Checklist

Before testing, ensure you have:

### 1. **Environment Setup**
```bash
# Activate conda environment
conda activate us-stock-app

# Verify installation
python -c "import pandas, numpy, sklearn, xgboost; print('✓ All packages installed')"
```

### 2. **Data Preparation**

Option A: Use existing data (if you've run pipeline before)
```bash
# Check if you have data
ls data/features/all_features_with_fundamentals.parquet
ls data/models/latest/model.pkl
```

Option B: Run minimal pipeline to generate required data
```bash
# This takes ~10-15 minutes
python run_core_pipeline.py --steps 1-7

# What this does:
# Step 1-2: Ingest OHLCV data
# Step 3: Generate technical features
# Step 4: Generate fundamental features
# Step 5: Generate labels
# Step 6: Feature selection
# Step 7: Train model
```

### 3. **Verify Data Exists**
```bash
# Required files:
ls -lh data/features/all_features_with_fundamentals.parquet  # Features
ls -lh data/models/latest/model.pkl                           # Trained model
ls -lh data/parquet/1d/                                       # Price data

# All should exist before proceeding
```

---

## 🧪 Test Suite

### **Test 1: Help & Version Check** ✅

**Purpose**: Verify script runs without errors

```bash
# Show help
python scripts/daily_live_runner.py --help

# Expected output:
# usage: daily_live_runner.py [-h] [--mode {dry-run,production}] ...
```

**Success Criteria**: Help text displays without errors

---

### **Test 2: Dry-Run Mode (Default)** ✅

**Purpose**: Run complete workflow safely (no real trades)

```bash
# Run with all defaults (dry-run mode)
python scripts/daily_live_runner.py --dry-run

# Or explicitly
python scripts/daily_live_runner.py --mode dry-run
```

**Expected Output**:
```
================================================================================
DAILY LIVE TRADING RUNNER
================================================================================
Mode: DRY-RUN
Date: TODAY
================================================================================
✓ Loaded configuration from config/config.yaml
✓ Market is open on 2025-10-30
✓ Data update completed successfully
✓ Loaded 500 stocks in universe
✓ Loaded features for 450 stocks
✓ Model loaded successfully: XGBRegressor
✓ Generated 450 signals
📊 Current regime: Normal (multiplier=1.00)
✓ Portfolio constructed: 20 positions
✓ Generated 15 trades (turnover=28.5%, costs=$45.23)
✓ Kill-switch check passed: daily_return=0.12%, rolling_sharpe=1.23
✓ SAFETY CHECKS PASSED
✓ Saved 4 output files to live/2025-10-30/
================================================================================
EXECUTION SUMMARY
================================================================================
Date: 2025-10-30
Mode: DRY-RUN
Portfolio: 20 positions
Trades: 15
Safety: ✓ PASSED
Regime: Normal (1.00x)

Outputs saved to: live/2025-10-30/
  - trades: live/2025-10-30/trades.csv
  - weights: live/2025-10-30/portfolio_weights.csv
  - signals: live/2025-10-30/signals.json
  - monitoring: live/2025-10-30/monitoring_log.json

📊 View report: live/2025-10-30/report.html
================================================================================
✅ Daily run completed successfully
```

**Success Criteria**:
- [ ] Script completes without errors
- [ ] Exit code 0 (success)
- [ ] 5 output files created
- [ ] HTML report generated

---

### **Test 3: Verify Outputs** ✅

**Purpose**: Validate all output files

```bash
# Check outputs directory
ls -lh live/$(date +%Y-%m-%d)/

# Expected files:
# trades.csv              - Trade orders
# portfolio_weights.csv   - Target weights
# signals.json            - ML scores
# report.html             - Dashboard
# monitoring_log.json     - Health metrics
```

**Check each file**:

```bash
# 1. Trades
echo "=== TRADES ===="
cat live/$(date +%Y-%m-%d)/trades.csv | head -10

# Expected: CSV with columns: symbol,side,shares,price,notional,...
# Should have both BUY and SELL orders

# 2. Weights
echo "=== WEIGHTS ===="
cat live/$(date +%Y-%m-%d)/portfolio_weights.csv | head -10

# Expected: CSV with columns: symbol,weight
# Sum of weights should be ~1.0 (or less if cash held)

# 3. Signals
echo "=== SIGNALS ===="
cat live/$(date +%Y-%m-%d)/signals.json

# Expected: JSON with date and scores array

# 4. Monitoring
echo "=== MONITORING ===="
cat live/$(date +%Y-%m-%d)/monitoring_log.json

# Expected: JSON with regime, safety, portfolio info

# 5. Report
echo "=== REPORT ===="
ls -lh live/$(date +%Y-%m-%d)/report.html

# Open in browser
open live/$(date +%Y-%m-%d)/report.html
# OR
xdg-open live/$(date +%Y-%m-%d)/report.html
```

**Success Criteria**:
- [ ] All 5 files exist
- [ ] Files are not empty
- [ ] trades.csv has valid trades
- [ ] weights sum to ~1.0
- [ ] HTML report opens in browser
- [ ] Report shows regime, trades, safety status

---

### **Test 4: Validate Trade Recommendations** ✅

**Purpose**: Ensure trades make sense

```bash
# Check trades in detail
python << 'EOF'
import pandas as pd
from datetime import datetime

date = datetime.now().strftime('%Y-%m-%d')
trades = pd.read_csv(f'live/{date}/trades.csv')

print(f"Total trades: {len(trades)}")
print(f"\nBuy trades: {len(trades[trades['side']=='BUY'])}")
print(f"Sell trades: {len(trades[trades['side']=='SELL'])}")
print(f"\nTotal notional: ${trades['notional'].sum():,.0f}")
print(f"Estimated costs: ${trades['total_cost_est'].sum():.2f}")

print("\n=== Top 5 Trades by Size ===")
print(trades.nlargest(5, 'notional')[['symbol', 'side', 'shares', 'price', 'notional']])

print("\n=== Validation Checks ===")
print(f"✓ All prices > 0: {(trades['price'] > 0).all()}")
print(f"✓ All shares != 0: {(trades['shares'] != 0).all()}")
print(f"✓ Buy orders have positive shares: {(trades[trades['side']=='BUY']['shares'] > 0).all()}")
print(f"✓ Sell orders have negative shares: {(trades[trades['side']=='SELL']['shares'] < 0).all()}")
EOF
```

**Success Criteria**:
- [ ] Trades have valid prices (all > 0)
- [ ] Buy orders have positive shares
- [ ] Sell orders have negative shares
- [ ] Total notional is reasonable
- [ ] Costs are < 1% of notional

---

### **Test 5: Validate Portfolio Weights** ✅

**Purpose**: Ensure weights are valid

```bash
# Check weights
python << 'EOF'
import pandas as pd
from datetime import datetime

date = datetime.now().strftime('%Y-%m-%d')
weights = pd.read_csv(f'live/{date}/portfolio_weights.csv')

print(f"Total positions: {len(weights)}")
print(f"Sum of weights: {weights['weight'].sum():.4f}")
print(f"\nWeight range: {weights['weight'].min():.4f} to {weights['weight'].max():.4f}")

print("\n=== Top 10 Positions ===")
print(weights.nlargest(10, 'weight'))

print("\n=== Validation Checks ===")
print(f"✓ All weights >= 0: {(weights['weight'] >= 0).all()}")
print(f"✓ Sum close to 1.0: {abs(weights['weight'].sum() - 1.0) < 0.01}")
print(f"✓ Max weight < 15%: {weights['weight'].max() < 0.15}")
print(f"✓ Min weight > 1%: {weights['weight'].min() > 0.01}")

if weights['weight'].sum() < 1.0:
    cash = 1.0 - weights['weight'].sum()
    print(f"\n💰 Cash allocation: {cash:.2%} (regime protection)")
EOF
```

**Success Criteria**:
- [ ] Sum of weights ≈ 1.0 (within 1%)
- [ ] All weights >= 0 (long-only)
- [ ] No position > 15% (max_weight)
- [ ] All positions > 1% (min_weight)
- [ ] Cash allocation shown if < 100% invested

---

### **Test 6: Check Regime Detection** ✅

**Purpose**: Verify regime detection works

```bash
# Check regime in monitoring log
python << 'EOF'
import json
from datetime import datetime

date = datetime.now().strftime('%Y-%m-%d')
with open(f'live/{date}/monitoring_log.json') as f:
    log = json.load(f)

regime = log.get('regime', {})
print("=== REGIME INFORMATION ===")
print(f"Regime: {regime.get('regime_name', 'Unknown')}")
print(f"Risk Multiplier: {regime.get('risk_multiplier', 'N/A')}")

if 'volatility' in regime:
    print(f"Volatility: {regime['volatility']:.2%}")
if 'drawdown' in regime:
    print(f"Drawdown: {regime['drawdown']:.2%}")

safety = log.get('safety', {})
print("\n=== SAFETY STATUS ===")
print(f"Overall Valid: {safety.get('overall_valid', False)}")
print(f"Violations: {len(safety.get('violations', []))}")

if safety.get('violations'):
    print("\nViolations:")
    for v in safety['violations']:
        print(f"  - {v}")
EOF
```

**Success Criteria**:
- [ ] Regime detected (Risk-Off/Normal/Risk-On)
- [ ] Risk multiplier shows (0.5/1.0/1.2)
- [ ] Volatility and drawdown calculated
- [ ] Safety status shows PASSED

---

### **Test 7: Skip Data Update** ✅

**Purpose**: Test faster execution

```bash
# Run without updating data (uses existing)
python scripts/daily_live_runner.py --skip-data-update

# Should complete in ~2 minutes instead of ~5
```

**Success Criteria**:
- [ ] Completes faster (~2 min vs ~5 min)
- [ ] Still generates all outputs
- [ ] Uses existing data successfully

---

### **Test 8: Historical Date (Backfill)** ✅

**Purpose**: Test running on past dates

```bash
# Run for a specific past date
python scripts/daily_live_runner.py --date 2025-10-15 --skip-data-update

# Check outputs in different directory
ls -lh live/2025-10-15/
```

**Success Criteria**:
- [ ] Creates separate directory for that date
- [ ] Uses historical data correctly
- [ ] All outputs generated

---

### **Test 9: Verbose Logging** ✅

**Purpose**: Test detailed logging

```bash
# Run with verbose output
python scripts/daily_live_runner.py --dry-run --verbose

# Check log file
tail -100 logs/daily_live_runner.log
```

**Success Criteria**:
- [ ] More detailed log output
- [ ] Debug messages visible
- [ ] Log file created

---

### **Test 10: Capital Override** ✅

**Purpose**: Test with different capital

```bash
# Test with small capital
python scripts/daily_live_runner.py --capital 10000 --skip-data-update

# Check position sizes scaled down
cat live/$(date +%Y-%m-%d)/trades.csv
```

**Success Criteria**:
- [ ] Portfolio scales to new capital
- [ ] Position sizes appropriate
- [ ] Minimum trade size respected

---

## 🛡️ Safety Testing

### **Test S1: Kill-Switch Validation**

**Purpose**: Verify kill-switch triggers properly

This is tested automatically during each run. Check monitoring log:

```bash
python << 'EOF'
import json
from datetime import datetime

date = datetime.now().strftime('%Y-%m-%d')
with open(f'live/{date}/monitoring_log.json') as f:
    log = json.load(f)

# Kill-switch should be in safety checks (run by SafetyValidator)
print("Kill-switch checks are run automatically by SafetyValidator")
print("If daily loss > 3% or Sharpe < 0.5, script will exit with error")
print("\nTo test kill-switch, you would need:")
print("1. Historical equity curve with large loss")
print("2. Or low Sharpe ratio over 6 weeks")
EOF
```

---

## 📊 Expected Results Summary

After running all tests, you should have:

### **Directory Structure**
```
live/
├── 2025-10-30/
│   ├── trades.csv             ✓
│   ├── portfolio_weights.csv  ✓
│   ├── signals.json           ✓
│   ├── report.html            ✓
│   └── monitoring_log.json    ✓
└── 2025-10-15/                ✓ (if tested historical date)
    └── ...

logs/
└── daily_live_runner.log      ✓
```

### **Performance Metrics**
- Execution time: 5 minutes (or 2 minutes with --skip-data-update)
- Memory usage: < 2 GB
- CPU usage: < 50% during execution

### **Output Quality**
- Trades: 10-30 orders typical
- Turnover: < 35% (should pass safety check)
- Portfolio: 15-25 positions typical
- Regime: One of Risk-Off/Normal/Risk-On
- Safety: Should PASS all checks

---

## ⚠️ Common Issues & Solutions

### Issue 1: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'pandas'
```

**Solution**:
```bash
# Activate conda environment first
conda activate us-stock-app
```

### Issue 2: FileNotFoundError for model

```
FileNotFoundError: No model found at production/champion_model.pkl
```

**Solution**:
```bash
# Run pipeline to train model
python run_core_pipeline.py --steps 7

# Or copy existing model
cp data/models/latest/model.pkl production/champion_model.pkl
```

### Issue 3: FileNotFoundError for features

```
Features file not found: data/features/all_features_with_fundamentals.parquet
```

**Solution**:
```bash
# Run pipeline to generate features
python run_core_pipeline.py --steps 1-6
```

### Issue 4: Market closed

```
Market is closed - exiting
```

**Solution**: This is normal on weekends/holidays. Either:
- Wait until next trading day
- Or test with historical date: `--date 2025-10-29`

---

## ✅ Testing Checklist

Complete this checklist to validate Option A:

### Prerequisites
- [ ] Conda environment activated
- [ ] All packages installed (pandas, numpy, sklearn, xgboost)
- [ ] Data generated (features, model, prices)

### Basic Tests
- [ ] Help text displays correctly
- [ ] Dry-run mode completes successfully
- [ ] All 5 output files created
- [ ] HTML report opens in browser

### Output Validation
- [ ] Trades CSV has valid orders
- [ ] Weights CSV sums to ~1.0
- [ ] Signals JSON has scores
- [ ] Monitoring log has regime info
- [ ] Report shows all sections

### Functionality Tests
- [ ] Skip data update works
- [ ] Historical date works
- [ ] Verbose logging works
- [ ] Capital override works

### Quality Checks
- [ ] Regime detection working
- [ ] Safety checks pass
- [ ] Trade recommendations make sense
- [ ] Position sizes appropriate
- [ ] No errors in logs

---

## 🎯 Final Validation

Run this comprehensive validation script:

```bash
#!/bin/bash

echo "=== OPTION A VALIDATION SCRIPT ==="
echo ""

# Test 1: Environment
echo "Test 1: Environment"
conda activate us-stock-app || exit 1
python -c "import pandas, numpy, sklearn, xgboost" || exit 1
echo "✓ Environment OK"
echo ""

# Test 2: Prerequisites
echo "Test 2: Prerequisites"
test -f data/features/all_features_with_fundamentals.parquet || echo "⚠️  Features missing"
test -f data/models/latest/model.pkl || echo "⚠️  Model missing"
echo ""

# Test 3: Run daily runner
echo "Test 3: Running daily_live_runner.py"
python scripts/daily_live_runner.py --dry-run --skip-data-update || exit 1
echo "✓ Script executed successfully"
echo ""

# Test 4: Check outputs
DATE=$(date +%Y-%m-%d)
echo "Test 4: Validating outputs for $DATE"
test -f "live/$DATE/trades.csv" || { echo "✗ trades.csv missing"; exit 1; }
test -f "live/$DATE/portfolio_weights.csv" || { echo "✗ weights.csv missing"; exit 1; }
test -f "live/$DATE/signals.json" || { echo "✗ signals.json missing"; exit 1; }
test -f "live/$DATE/report.html" || { echo "✗ report.html missing"; exit 1; }
test -f "live/$DATE/monitoring_log.json" || { echo "✗ monitoring_log.json missing"; exit 1; }
echo "✓ All outputs generated"
echo ""

echo "==================================="
echo "✅ OPTION A VALIDATION COMPLETE"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Review report: open live/$DATE/report.html"
echo "2. Check trades: cat live/$DATE/trades.csv"
echo "3. Start daily paper trading!"
```

Save as `test_option_a.sh` and run:
```bash
chmod +x test_option_a.sh
./test_option_a.sh
```

---

## 📞 Support

If you encounter issues:

1. **Check logs**: `tail -100 logs/daily_live_runner.log`
2. **Verify data**: Ensure pipeline has been run (steps 1-7)
3. **Test environment**: `python -c "import pandas; print('OK')"`
4. **Review config**: Check `config/config.yaml` for correct paths

---

## 🎊 Success!

If all tests pass, you're ready for:

### **Week 1-2: Paper Trading**
```bash
# Run daily
python scripts/daily_live_runner.py --dry-run

# Review outputs
open live/$(date +%Y-%m-%d)/report.html
```

### **Week 3+: Performance Validation**
- Track IC (should be > 0.03)
- Monitor turnover (should be < 35%)
- Check safety passes daily
- Review regime transitions

### **When Ready: Go Live**
```yaml
# config/config.yaml
live:
  dry_run: false  # ⚠️  Real trades!
```

---

**Happy Trading! 🚀**

🤖 Powered by Claude Code
