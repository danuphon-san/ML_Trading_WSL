# Option A Testing Walkthrough

**Goal**: Test Option A with manual workflow step-by-step
**Estimated Time**: 30-45 minutes
**Date**: 2025-10-30

---

## 📋 Pre-Test Checklist

Before we begin, let's verify you have everything ready:

```bash
# 1. Activate conda environment
conda activate us-stock-app

# 2. Verify you're in the project directory
pwd
# Should show: /home/user/ML_Trading_WSL

# 3. Check Python version
python --version
# Should be Python 3.8 or higher

# 4. Verify key dependencies
python -c "import pandas, numpy, yaml; print('✓ Dependencies OK')"
```

**Expected Output**: `✓ Dependencies OK`

---

## 🧪 Phase 1: Automated Tests (10 minutes)

### Test 1.1: Run Automated Test Suite

```bash
# Run comprehensive test suite
python tests/test_option_a.py
```

**What to look for:**
- ✅ All prerequisites checked
- ✅ Tests run without errors
- ✅ Final summary shows passing tests

**If tests fail:**
- Missing data: Run `python run_core_pipeline.py --steps 1-7`
- Missing model: Run training pipeline
- Environment issues: Check conda environment

### Test 1.2: Quick Validation Test

```bash
# Run quick validation (skips slow tests)
python tests/test_option_a.py --quick
```

**Expected**: Faster execution, fewer tests

---

## 🎯 Phase 2: Generate First Recommendations (10 minutes)

### Test 2.1: Check Current Setup

```bash
# Check if data directory exists
ls -la data/

# Check if universe exists
ls -la data/universe.csv

# Check if model exists
ls -la data/models/latest/
# OR
ls -la production/
```

### Test 2.2: Generate Trade Recommendations

```bash
# Run daily live runner in dry-run mode
python scripts/daily_live_runner.py --dry-run --verbose
```

**What happens:**
1. Market open check
2. Data update (if enabled)
3. Universe loading
4. Feature computation
5. Model loading
6. Signal generation
7. Regime detection
8. Portfolio construction
9. Trade generation
10. Safety checks
11. Output generation
12. HTML report

**Expected Outputs:**
```
live/2025-10-30/
├── trades.csv              ← Trade recommendations
├── portfolio_weights.csv   ← Target allocation
├── signals.json            ← ML scores
├── report.html             ← Daily dashboard
└── monitoring_log.json     ← Model health
```

### Test 2.3: Inspect Outputs

```bash
# View trades
cat live/$(date +%Y-%m-%d)/trades.csv

# Count trades
wc -l live/$(date +%Y-%m-%d)/trades.csv

# Open report in browser (if on WSL with display)
# open live/$(date +%Y-%m-%d)/report.html
# Or copy to Windows and open
```

**What to check:**
- ✅ Trades file has recommendations
- ✅ Portfolio weights sum to ~1.0
- ✅ Report shows regime, safety checks, IC

---

## 🔧 Phase 3: Test Manual Workflow (15 minutes)

### Test 3.1: Initialize Position Tracking

```bash
# Create empty positions file (first time only)
mkdir -p live
echo "symbol,shares,avg_cost,last_updated" > live/current_positions.csv

# Verify
cat live/current_positions.csv
```

**Expected**: Header row only (empty portfolio)

### Test 3.2: Test Interactive Approval

```bash
# Run approval script
python scripts/approve_trades.py live/$(date +%Y-%m-%d)/trades.csv
```

**Interactive prompts:**
```
================================================================
TRADE APPROVAL - 2025-10-30
================================================================

Summary:
  Total Trades: 28
  Buy Orders: 15
  Sell Orders: 13
  ...

Safety Checks: ✓ ALL PASSED

Review trades.csv? (y/n): n

Do you approve these trades for execution? (yes/no/edit): yes

✓ Trades APPROVED
✓ Approval saved: live/2025-10-30/APPROVED
================================================================
```

**Test scenarios:**
1. **Approve**: Type `yes` → Creates APPROVED marker
2. **Reject**: Type `no` → Creates REJECTED marker
3. **Review**: Type `y` when asked to review → Opens file
4. **Edit**: Type `edit` → Allows manual editing

### Test 3.3: Check Approval Files

```bash
# Check approval marker
ls -la live/$(date +%Y-%m-%d)/APPROVED

# Check approval log
cat live/$(date +%Y-%m-%d)/approval_log.json
```

**Expected**: Approval marker file exists, log shows approval details

### Test 3.4: Test Execution Logger

```bash
# Run interactive execution logger
python scripts/log_execution.py
```

**Interactive test:**
```
Enter execution details (or 'done' to finish):

Symbol (or 'done'): AAPL
Side (buy/sell): buy
Shares: 100
Fill Price: 150.25
Fill Time (HH:MM:SS or leave blank): [Enter]
Commission (default 0.00): 0
Notes (optional): Test execution

  ✓ Logged: AAPL buy 100@$150.25

Symbol (or 'done'): MSFT
Side (buy/sell): buy
Shares: 50
Fill Price: 380.50
Fill Time: [Enter]
Commission: 0
Notes: Test execution

  ✓ Logged: MSFT buy 50@$380.50

Symbol (or 'done'): done

================================================================
✓ Logged 2 executions
✓ Saved to: live/2025-10-30/execution_log.csv
================================================================
```

### Test 3.5: Verify Execution Log

```bash
# View execution log
cat live/$(date +%Y-%m-%d)/execution_log.csv
```

**Expected format:**
```csv
symbol,side,shares,fill_price,fill_time,commission,notes
AAPL,buy,100,150.25,10:30:15,0.0,Test execution
MSFT,buy,50,380.50,10:31:22,0.0,Test execution
```

### Test 3.6: Test Position Update

```bash
# Update positions from execution log
python scripts/update_positions.py \
  --execution-log live/$(date +%Y-%m-%d)/execution_log.csv \
  --show-details
```

**Expected output:**
```
================================================================
POSITION UPDATE - 2025-10-30
================================================================

Starting positions: 0
Executions to process: 2

  Processing: AAPL buy 100@$150.25
  Processing: MSFT buy 50@$380.50

Executions processed:
  Buy orders: 2
  Sell orders: 0

Updated Positions (2 holdings):

  Symbol    Shares    Value      Weight
  AAPL         100   $15,025    50.00%
  MSFT          50   $19,025    50.00%

Total Portfolio Value (estimated): $34,050.00

✓ Positions updated: live/current_positions.csv
================================================================
```

### Test 3.7: Verify Position File

```bash
# View current positions
cat live/current_positions.csv
```

**Expected:**
```csv
symbol,shares,avg_cost,last_updated
AAPL,100,150.25,2025-10-30
MSFT,50,380.50,2025-10-30
```

---

## 🔍 Phase 4: Test Position Import (5 minutes)

### Test 4.1: Create Mock Broker Export

```bash
# Create sample broker CSV
cat > /tmp/broker_positions.csv << EOF
Symbol,Quantity,Avg Cost,Current Price
AAPL,100,150.25,152.00
MSFT,50,380.50,385.00
GOOGL,25,142.80,145.00
EOF
```

### Test 4.2: Test Import with Auto-Detect

```bash
# Import with auto-detection
python scripts/import_positions.py \
  --broker-file /tmp/broker_positions.csv \
  --auto-detect \
  --show-preview
```

**Expected:**
```
Auto-detected broker format: generic

Imported 3 positions:

  Symbol  Shares  Avg_Cost
  AAPL       100    150.25
  MSFT        50    380.50
  GOOGL       25    142.80

Total Portfolio Value (estimated): $48,057.50

Overwrite existing positions? (yes/no): no
```

Type `no` to cancel (since we're just testing)

---

## ✅ Phase 5: Validation Tests (5 minutes)

### Test 5.1: Trade Validation

```bash
# Validate trades before execution
python scripts/validate_trades.py live/$(date +%Y-%m-%d)/trades.csv
```

**Expected output:**
```
TRADE VALIDATION REPORT
================================================================

Total Trades: 28
Buy Orders: 15
Sell Orders: 13
Total Notional: $98,450.00

VALIDATION CHECKS:
------------------------------------------------------------------
FORMAT               ✓ PASS    All required columns present
PRICES               ✓ PASS    Prices valid
QUANTITIES           ✓ PASS    Quantities valid
NOTIONAL             ✓ PASS    Notional values valid
COSTS                ✓ PASS    Costs reasonable
DIVERSIFICATION      ✓ PASS    Diversification acceptable

================================================================
✓ VALIDATION PASSED - Trades are ready for execution
================================================================
```

### Test 5.2: Position Reconciliation

```bash
# Test reconciliation (compares recommended vs executed)
python scripts/reconcile_positions.py \
  --trades live/$(date +%Y-%m-%d)/trades.csv \
  --fills live/$(date +%Y-%m-%d)/execution_log.csv \
  --current-positions live/current_positions.csv
```

**Expected**: Reconciliation report showing execution rate, costs, position accuracy

---

## 📊 Phase 6: Review Complete Workflow (5 minutes)

### Test 6.1: Complete End-to-End Test

Let's simulate a complete day:

```bash
# 1. Generate recommendations (Morning)
echo "=== Step 1: Generate Recommendations ==="
python scripts/daily_live_runner.py --dry-run --skip-data-update

# 2. Approve trades (Morning)
echo ""
echo "=== Step 2: Approve Trades ==="
# Run and type 'yes' when prompted
python scripts/approve_trades.py live/$(date +%Y-%m-%d)/trades.csv

# 3. [Execute manually at broker] - SIMULATED

# 4. Log executions (Throughout day) - SKIP (already tested)

# 5. Update positions (End of day)
echo ""
echo "=== Step 5: Update Positions ==="
python scripts/update_positions.py \
  --execution-log live/$(date +%Y-%m-%d)/execution_log.csv

# 6. View final positions
echo ""
echo "=== Final Positions ==="
cat live/current_positions.csv
```

### Test 6.2: Check All Output Files

```bash
# List all outputs
echo "Outputs generated:"
ls -lh live/$(date +%Y-%m-%d)/

# Expected files:
# - trades.csv
# - portfolio_weights.csv
# - signals.json
# - report.html
# - monitoring_log.json
# - execution_log.csv (if created)
# - APPROVED (if approved)
# - approval_log.json (if approved)
```

---

## 🎯 Success Criteria

**You've successfully tested Option A if:**

### Core Functionality
- ✅ Automated tests pass
- ✅ Daily runner generates recommendations
- ✅ All 5 output files created
- ✅ Safety checks pass
- ✅ Regime detection works

### Manual Workflow
- ✅ Approval script works interactively
- ✅ Execution logger records fills
- ✅ Position updater calculates correctly
- ✅ Broker import handles different formats
- ✅ Trade validation detects issues

### Integration
- ✅ Complete workflow runs end-to-end
- ✅ Position tracking works across days
- ✅ Reconciliation identifies discrepancies

---

## 🚨 Common Issues & Solutions

### Issue 1: "No module named 'pandas'"
**Solution:**
```bash
conda activate us-stock-app
# OR
conda env create -f environment.yml
```

### Issue 2: "No universe.csv found"
**Solution:**
```bash
python run_core_pipeline.py --steps 1
```

### Issue 3: "No model found"
**Solution:**
```bash
python run_core_pipeline.py --steps 1-7
```

### Issue 4: "Trades file not found"
**Solution:**
```bash
# Generate recommendations first
python scripts/daily_live_runner.py --dry-run
```

### Issue 5: Position file corrupted
**Solution:**
```bash
# Recreate from scratch
echo "symbol,shares,avg_cost,last_updated" > live/current_positions.csv
```

---

## 📝 Testing Checklist

Copy this checklist and mark as you complete:

```
Phase 1: Automated Tests
[ ] Run test_option_a.py
[ ] All tests pass
[ ] Quick mode works

Phase 2: Generate Recommendations
[ ] daily_live_runner.py executes
[ ] trades.csv created
[ ] report.html generated
[ ] All 5 output files present

Phase 3: Manual Workflow
[ ] Empty positions file created
[ ] Approval script works
[ ] Approval marker created
[ ] Execution logger works
[ ] Execution log created
[ ] Position updater works
[ ] Positions file updated correctly

Phase 4: Position Import
[ ] Mock broker file created
[ ] Import script works
[ ] Auto-detect works

Phase 5: Validation
[ ] Trade validation passes
[ ] Reconciliation works

Phase 6: End-to-End
[ ] Complete workflow runs
[ ] All files generated
[ ] Positions track correctly
```

---

## 🎓 Next Steps After Testing

**If all tests pass:**
1. ✅ Review `docs/MANUAL_TRADING_WORKFLOW.md` for daily usage
2. ✅ Start paper trading (1-2 weeks)
3. ✅ Monitor IC and model health daily
4. ✅ Consider Option B (Model Comparison) for model selection

**If tests fail:**
1. ⚠️ Check error messages
2. ⚠️ Review prerequisites
3. ⚠️ Consult `TESTING_GUIDE.md` for troubleshooting
4. ⚠️ Ask for help with specific error

---

## 📚 Documentation Quick Reference

- **Daily Workflow**: `docs/MANUAL_TRADING_WORKFLOW.md`
- **Trade Execution**: `docs/TRADE_EXECUTION_GUIDE.md`
- **Testing Guide**: `TESTING_GUIDE.md`
- **Completeness Review**: `OPTION_A_COMPLETENESS_REVIEW.md`

---

**Ready to begin testing? Start with Phase 1!**
