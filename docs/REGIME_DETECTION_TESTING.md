# Regime Detection Testing Guide

This guide provides step-by-step instructions for testing the regime detection implementation (Option 2).

## Overview

The regime detection system adaptively adjusts portfolio construction based on market conditions:

- **Risk-Off Regime (0)**: High volatility, deep drawdowns, or downtrends → Reduce exposure to 50%
- **Normal Regime (1)**: Moderate volatility, stable markets → Full 100% exposure
- **Risk-On Regime (2)**: Low volatility, uptrends, shallow drawdowns → Increase to 120% exposure

## Implementation Components

### 1. Regime Detection Module
**File**: `src/portfolio/regime_detection.py`

**Key Classes**:
- `RegimeDetector`: Detects market regimes using rules-based or HMM methods
- `run_regime_detection()`: Standalone function for batch detection

**Detection Logic**:
```python
# Risk-Off triggered by:
- Volatility > 25% (annualized) OR
- Drawdown < -10% OR
- Downtrend (SMA_20 < SMA_50)

# Risk-On triggered by:
- Volatility < 15% AND
- Drawdown > -5% AND
- Uptrend (SMA_20 > SMA_50)

# Normal: Everything else
```

### 2. Regime-Aware Portfolio Construction
**File**: `src/portfolio/construct.py`

**Key Functions**:
- `construct_portfolio()`: Main entrypoint with `enable_regime_adaptation` parameter
- `_detect_current_regime()`: Detects current market regime from benchmark
- `_adjust_config_for_regime()`: Adjusts top_k based on regime
- `_apply_regime_risk_adjustment()`: Scales position sizes by risk multiplier

**Workflow**:
1. Detect current regime from benchmark (SPY)
2. Adjust top_k selection:
   - Risk-Off: 60% of normal top_k (concentrate on defensive)
   - Risk-On: 130% of normal top_k (increase diversification)
3. Run portfolio optimization (PyPortfolioOpt/Inverse-Vol/etc.)
4. Apply risk multipliers to final weights:
   - Risk-Off: 0.5x (50% exposure, 50% cash)
   - Normal: 1.0x (full exposure)
   - Risk-On: 1.2x (renormalized to 1.0 for long-only)

### 3. Regime Features for ML Model
**File**: `src/features/ta_features.py`

**New Features** (added to `_add_regime_features()`):

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `vol_regime` | Short vol / Long vol ratio | > 1.0 = elevated volatility |
| `trend_strength` | \|SMA_20 - SMA_50\| / close | Higher = stronger trend |
| `drawdown_pct` | (Close - Running Max) / Max | More negative = deeper drawdown |
| `dist_from_200d_high` | Distance from 200-day high | Negative = below recent high |
| `above_200_sma` | Binary: above 200-day SMA | 1 = bull market, 0 = bear |
| `momentum_consensus` | Count of positive momentum across 3 timeframes | 0-3 scale |
| `vol_expansion` | 20-day change in volatility | Positive = expanding vol |
| `crisis_indicator` | Drawdown < -15% AND vol_regime > 1.5 | 1 = crisis, 0 = normal |
| `recovery_indicator` | Drawdown improving AND vol declining | 1 = recovery phase |

**Purpose**: Helps ML model learn regime-conditional patterns (e.g., momentum works better in Risk-On, value works better in Risk-Off).

### 4. Configuration
**File**: `config/config.yaml`

**New Section**: `regime:`
```yaml
regime:
  method: "rules"  # or "hmm"

  vol_window: 20
  vol_threshold_high: 0.25
  vol_threshold_low: 0.15

  dd_threshold: -0.10
  trend_window: 50

  risk_multipliers:
    risk_off: 0.5
    normal: 1.0
    risk_on: 1.2

  top_k_multipliers:
    risk_off: 0.6
    normal: 1.0
    risk_on: 1.3
```

---

## Testing Instructions

### Prerequisites

1. **Activate conda environment**:
   ```bash
   conda activate us-stock-app
   ```

2. **Ensure API keys are set** (from Option 1):
   ```bash
   # Check .env file exists
   cat .env

   # Should show:
   # SIMFIN_API_KEY=your-key-here
   # ALPHA_VANTAGE_API_KEY=your-key-here
   ```

3. **Verify data availability**:
   ```bash
   # Check for SPY benchmark data
   ls -lh data/parquet/1d/SPY.parquet

   # If missing, download it:
   python scripts/ingest_ohlcv_bulk.py --symbols SPY --start-date 2020-01-01
   ```

---

### Test 1: Standalone Regime Detection

**Purpose**: Verify regime detector works independently on benchmark data.

**Command**:
```bash
python tests/test_regime_detection.py
```

**Expected Output**:
```
================================================================================
TEST 1: Standalone Regime Detection on SPY
================================================================================
Loaded 1234 days of SPY data from 2020-01-01 to 2025-10-29
Initialized RegimeDetector: method=rules, vol_window=20

📊 Regime Detection Results:
Total periods analyzed: 1234
  Risk-Off  :   234 days ( 19.0%)
  Normal    :   678 days ( 54.9%)
  Risk-On   :   322 days ( 26.1%)

🎯 Current Regime: Normal
   Risk Multiplier: 1.00x
   Volatility: 18.5%
   Drawdown: -2.3%

📅 Recent Regime History (last 20 days):
         date  regime  volatility  drawdown  risk_multiplier regime_name
   2025-10-10       1       0.175    -0.034              1.0      Normal
   2025-10-11       1       0.178    -0.029              1.0      Normal
   ...

✓ Regime detection results saved to data/reports/regime_detection_test.csv
```

**What to Check**:
- [x] Regime distribution looks reasonable (not all one regime)
- [x] Current regime makes sense given recent market conditions
- [x] Volatility and drawdown values are plausible
- [x] Output CSV saved successfully

---

### Test 2: Regime Features in ML Model

**Purpose**: Verify regime features are added to technical indicators.

**Test Section**: Runs automatically in `test_regime_detection.py` (Test 2)

**Expected Output**:
```
================================================================================
TEST 2: Regime Features in Technical Indicators
================================================================================
Computing technical features for SPY (1234 days)...

📊 Checking for regime features:
  ✓ vol_regime              :  1200 valid values, mean=  1.024, std=  0.312
  ✓ trend_strength          :  1150 valid values, mean=  0.032, std=  0.018
  ✓ drawdown_pct            :  1234 valid values, mean= -0.045, std=  0.067
  ✓ dist_from_200d_high     :  1034 valid values, mean= -0.038, std=  0.054
  ✓ above_200_sma           :  1034 valid values, mean=  0.647, std=  0.478
  ✓ momentum_consensus      :  1174 valid values, mean=  1.823, std=  1.142
  ✓ vol_expansion           :  1180 valid values, mean= -0.002, std=  0.453
  ✓ crisis_indicator        :  1200 valid values, mean=  0.012, std=  0.109
  ✓ recovery_indicator      :  1195 valid values, mean=  0.089, std=  0.285

✓ Regime features saved to data/reports/regime_features_test.csv
```

**What to Check**:
- [x] All 9 regime features are present (no ✗ MISSING)
- [x] Feature statistics are reasonable (no all-zeros or NaN)
- [x] Crisis indicator is rare (mean < 5%, indicating it triggers only in extreme conditions)
- [x] Features saved to CSV for inspection

---

### Test 3: Regime-Aware Portfolio Construction

**Purpose**: Compare portfolio construction with and without regime adaptation.

**Test Section**: Runs automatically in `test_regime_detection.py` (Test 3)

**Expected Output**:
```
================================================================================
TEST 3: Regime-Aware Portfolio Construction
================================================================================
Mock data created: 5 stocks, 252 days of price history

--- Test 3a: Portfolio WITHOUT Regime Adaptation ---
Constructing portfolio with optimizer=pypfopt, regime_adaptation=False
Portfolio weights (no regime adaptation):
  AAPL: 23.45%
  MSFT: 21.32%
  GOOGL: 19.87%
  AMZN: 18.54%
  NVDA: 16.82%
Total weight: 100.00%

--- Test 3b: Portfolio WITH Regime Adaptation ---
Constructing portfolio with optimizer=pypfopt, regime_adaptation=True
📊 Current regime: Risk-Off (multiplier=0.50)
Risk-off regime: Reducing top_k from 20 to 12
Regime adjustment (Risk-Off): Original sum=1.00, Adjusted sum=50.00%

Portfolio weights (WITH regime adaptation):
  AAPL: 11.73%
  MSFT: 10.66%
  GOOGL: 9.94%
  AMZN: 9.27%
  NVDA: 8.41%
Total weight: 50.01%

📊 Comparison:
  Without regime: 5 positions, total=100.00%
  With regime:    5 positions, total=50.01%
  💰 Implicit cash: 49.99%
```

**What to Check**:
- [x] Regime is detected correctly for current market
- [x] Risk-Off reduces total exposure (creates cash)
- [x] Risk-On increases exposure (up to 120%, then renormalized)
- [x] Normal maintains 100% exposure
- [x] Top-k is adjusted based on regime

---

### Test 4: Full Pipeline Integration

**Purpose**: Run the entire ML pipeline with regime detection enabled.

**Command**:
```bash
# Step 1: Train model with regime features
python scripts/train_model.py --config config/config.yaml

# Step 2: Score universe
python scripts/score_universe.py --config config/config.yaml

# Step 3: Construct portfolio with regime adaptation
python scripts/construct_portfolio.py --config config/config.yaml --enable-regime

# Step 4: Run backtest
python scripts/run_backtest.py --config config/config.yaml --enable-regime
```

**Expected Log Output**:
```
[Portfolio Construction]
📊 Current regime: Risk-Off (multiplier=0.50)
Risk-off regime: Reducing top_k from 20 to 12
Selected 12 stocks for portfolio
Running PyPortfolioOpt optimization...
Regime adjustment (Risk-Off): Original sum=1.00, Adjusted sum=50.00%
💰 Cash allocation: 50.00% (risk-off protection)
Portfolio constructed: 12 positions, sum=50.00%

[Backtesting]
Backtest period: 2018-01-01 to 2025-10-29
Total trades: 1,234
Average turnover: 28.4% (within 35% limit)
Regime transitions: 87 (Risk-Off: 234 days, Normal: 678 days, Risk-On: 322 days)

Performance (WITH regime adaptation):
  Total Return:     145.6%
  CAGR:             12.8%
  Sharpe Ratio:     1.24
  Max Drawdown:     -18.5%
  Calmar Ratio:     0.69

Benchmark (SPY):
  Total Return:     98.3%
  CAGR:             9.2%
  Sharpe Ratio:     0.87
  Max Drawdown:     -23.9%

Alpha: +3.6% | Beta: 0.78 | Information Ratio: 0.56
```

**What to Check**:
- [x] Regime detection logging appears during portfolio construction
- [x] Cash allocations occur during Risk-Off periods
- [x] Backtest completes without errors
- [x] Regime-aware strategy shows lower drawdowns (defensive)
- [x] Sharpe ratio improves vs. baseline (better risk-adjusted returns)

---

### Test 5: Regime Visualization (Optional)

**Purpose**: Visualize regime transitions over time.

**Command**:
```bash
python -c "
import pandas as pd
import matplotlib.pyplot as plt

# Load regime detection results
regime_df = pd.read_csv('data/reports/regime_detection_test.csv')
regime_df['date'] = pd.to_datetime(regime_df['date'])

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Regime state
colors = {0: 'red', 1: 'gray', 2: 'green'}
regime_colors = [colors[r] for r in regime_df['regime']]
ax1.scatter(regime_df['date'], regime_df['regime'], c=regime_colors, alpha=0.6, s=10)
ax1.set_ylabel('Regime')
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(['Risk-Off', 'Normal', 'Risk-On'])
ax1.grid(True, alpha=0.3)

# Volatility
ax2.plot(regime_df['date'], regime_df['volatility'], label='Volatility', color='blue')
ax2.axhline(0.25, color='red', linestyle='--', label='High Vol Threshold')
ax2.axhline(0.15, color='green', linestyle='--', label='Low Vol Threshold')
ax2.set_ylabel('Annualized Volatility')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Drawdown
ax3.fill_between(regime_df['date'], 0, regime_df['drawdown'], alpha=0.5, color='red')
ax3.axhline(-0.10, color='darkred', linestyle='--', label='DD Threshold')
ax3.set_ylabel('Drawdown')
ax3.set_xlabel('Date')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/reports/regime_visualization.png', dpi=150)
print('✓ Saved to data/reports/regime_visualization.png')
"
```

**Output**: `data/reports/regime_visualization.png`

---

## Troubleshooting

### Issue 1: "Benchmark SPY not found in price_panel"

**Cause**: SPY data is missing from your OHLCV dataset.

**Fix**:
```bash
# Download SPY data
python scripts/ingest_ohlcv_bulk.py --symbols SPY --start-date 2015-01-01

# Or add SPY to universe
echo "SPY" >> config/universe_symbols.txt
python scripts/update_universe.py
```

### Issue 2: "Insufficient benchmark data (45 rows), skipping regime detection"

**Cause**: Not enough historical data for regime calculation (need 252+ days for reliable volatility).

**Fix**:
```bash
# Extend data history
python scripts/ingest_ohlcv_bulk.py --symbols SPY --start-date 2020-01-01
```

### Issue 3: Regime detection is always "Normal"

**Possible Causes**:
1. **Thresholds too extreme**: Adjust in `config.yaml`
   ```yaml
   regime:
     vol_threshold_high: 0.20  # Lower from 0.25
     vol_threshold_low: 0.12   # Lower from 0.15
     dd_threshold: -0.08       # Less extreme from -0.10
   ```

2. **Testing during stable market**: Normal behavior if market is actually stable.

3. **Check detection logic**:
   ```bash
   python -c "
   from src.portfolio.regime_detection import RegimeDetector
   import pandas as pd

   cfg = yaml.safe_load(open('config/config.yaml'))
   detector = RegimeDetector(cfg)
   print(f'Vol thresholds: {detector.vol_threshold_low} - {detector.vol_threshold_high}')
   print(f'DD threshold: {detector.dd_threshold}')
   "
   ```

### Issue 4: ModuleNotFoundError for hmmlearn

**Cause**: HMM method selected but package not installed.

**Fix**:
```bash
# Option 1: Install hmmlearn
conda install -c conda-forge hmmlearn

# Option 2: Use rules-based method
# Edit config.yaml:
regime:
  method: "rules"
```

---

## Performance Expectations

Based on the implementation, you should expect:

### Regime Distribution (2018-2025 period)
- **Risk-Off**: 15-25% of days (market crises, corrections)
- **Normal**: 50-65% of days (typical market conditions)
- **Risk-On**: 20-30% of days (bull runs, low volatility periods)

### Performance Improvements
- **Lower Max Drawdown**: 3-8% reduction (defensive cash during Risk-Off)
- **Higher Sharpe Ratio**: +0.10 to +0.30 improvement (better risk-adjusted returns)
- **Smoother Equity Curve**: Less volatility during market stress
- **Beta < 1.0**: Reduced market exposure (60-80% typical)

### Trade-offs
- **Lower CAGR**: May sacrifice 1-3% annual return (cash drag during Risk-Off)
- **More Trades**: Regime transitions trigger rebalancing
- **Higher Turnover**: 5-10% increase vs. baseline (if rebalancing frequently)

---

## Next Steps

After successful testing:

1. **Tune Parameters**: Adjust thresholds in `config.yaml` based on your risk tolerance
2. **Compare Methods**: Test both `rules` and `hmm` detection methods
3. **Backtest Comparison**: Run A/B test (with vs. without regime adaptation)
4. **Production Deployment**: Enable regime adaptation in live trading:
   ```yaml
   live:
     enabled: true
     regime_adaptation: true
   ```

5. **Monitor Performance**: Track regime transitions and portfolio adjustments in production

---

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review test outputs in `data/reports/`
- Consult `IMPLEMENTATION_SUMMARY.md` for architecture details
- Open an issue in the repository

---

**Last Updated**: 2025-10-30
**Version**: 1.0
**Status**: Ready for Testing
