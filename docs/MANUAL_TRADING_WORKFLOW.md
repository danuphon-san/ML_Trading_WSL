# Manual Trading Workflow (No Broker Integration)

**For traders who want to execute manually at their broker but use ML recommendations**

**Version**: 1.0
**Date**: 2025-10-30

---

## 📋 Overview

This workflow allows you to:
- ✅ Get daily ML trade recommendations
- ✅ Review and approve trades interactively
- ✅ Execute trades manually at your broker
- ✅ Track positions without broker API integration
- ✅ Maintain full control and transparency

**No broker API required - 100% manual execution with smart tracking**

---

## 🔄 Daily Workflow

### Step 1: Generate Recommendations (Morning)

```bash
# Run daily workflow to get trade recommendations
python scripts/daily_live_runner.py --dry-run

# Output: live/2025-10-30/trades.csv
```

**What this does:**
- Scores all stocks with ML model
- Detects market regime
- Constructs optimal portfolio
- Generates trade recommendations
- Runs safety checks

### Step 2: Review Recommendations (5 minutes)

```bash
# Open daily report in browser
open live/$(date +%Y-%m-%d)/report.html

# Or view trades in terminal
cat live/$(date +%Y-%m-%d)/trades.csv
```

**Review checklist:**
- [ ] Regime makes sense (Risk-Off/Normal/Risk-On)
- [ ] IC (Information Coefficient) > 0.02
- [ ] Safety checks passed
- [ ] Turnover reasonable (< 35%)
- [ ] No suspicious symbols

### Step 3: Approve or Reject Trades (Interactive)

```bash
# Run interactive approval script
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
  Total Notional: $98,450.00
  Estimated Costs: $295.35
  Portfolio Turnover: 18.5%
  Current Regime: Normal

Safety Checks: ✓ ALL PASSED

================================================================
Review trades.csv? (y/n): y

[Opens trades.csv in your default viewer]

================================================================
Do you approve these trades for execution? (yes/no/edit): yes

✓ Trades APPROVED for execution
✓ Approval saved: live/2025-10-30/APPROVED

Next steps:
1. Execute trades manually at your broker
2. Record fills in: live/2025-10-30/execution_log.csv
3. Update positions: python scripts/update_positions.py
================================================================
```

**Approval creates:**
- `live/2025-10-30/APPROVED` - Approval marker file
- `live/2025-10-30/approval_log.json` - Audit trail

### Step 4: Execute Trades Manually (9:35 AM - 3:00 PM ET)

**Log into your broker and execute trades from trades.csv**

**Execution template:**
```csv
symbol,side,shares,price,notional
AAPL,buy,100,150.25,15025.00
MSFT,sell,50,380.50,19025.00
```

**For each trade:**
1. Open your broker platform (Fidelity, Schwab, etc.)
2. Enter order:
   - Symbol: AAPL
   - Action: BUY
   - Quantity: 100 shares
   - Order Type: LIMIT
   - Limit Price: $150.25 (or current market price)
   - Time in Force: DAY
3. Submit order
4. **Record execution** (see Step 5)

### Step 5: Record Your Executions (As You Trade)

**Keep track in execution log:**

```bash
# Create execution log (CSV format)
nano live/$(date +%Y-%m-%d)/execution_log.csv
```

**Format:**
```csv
symbol,side,shares,fill_price,fill_time,commission,notes
AAPL,buy,100,150.28,09:45:12,0.00,Full fill
MSFT,sell,50,380.45,10:12:34,0.00,Full fill
GOOGL,buy,25,142.80,10:45:00,0.00,Full fill
```

**Or use the helper script:**
```bash
# Interactive execution logger
python scripts/log_execution.py

# Prompts for each trade:
# Symbol: AAPL
# Side: buy
# Shares: 100
# Fill Price: 150.28
# Commission: 0.00
# Notes: Full fill
```

### Step 6: Update Your Positions (End of Day)

```bash
# Update positions based on executions
python scripts/update_positions.py \
  --execution-log live/2025-10-30/execution_log.csv \
  --current-positions live/current_positions.csv
```

**What this does:**
- Reads your execution log
- Updates `live/current_positions.csv`
- Calculates new portfolio weights
- Shows you the updated portfolio

**Output:**
```
================================================================
POSITION UPDATE - 2025-10-30
================================================================

Executions processed: 28
  Buy orders: 15
  Sell orders: 13

Updated Positions (20 holdings):
  Symbol    Shares    Value      Weight
  AAPL         250   $37,562    5.01%
  MSFT          25   $ 9,510    1.27%
  GOOGL        150   $21,420    2.86%
  ...

Total Portfolio Value: $100,000
Cash: $5,430

✓ Positions updated: live/current_positions.csv
================================================================
```

---

## 📁 File Structure

### Files You Maintain Manually:

**1. `live/current_positions.csv`** - Your current portfolio
```csv
symbol,shares,avg_cost,last_updated
AAPL,250,148.20,2025-10-30
MSFT,25,378.40,2025-10-30
GOOGL,150,142.80,2025-10-30
```

**How to initialize** (first time):
```bash
# Start with empty portfolio
echo "symbol,shares,avg_cost,last_updated" > live/current_positions.csv

# Or import from broker
# (Download CSV from broker, convert to format above)
python scripts/import_positions.py \
  --broker-file my_broker_positions.csv \
  --output live/current_positions.csv
```

**2. `live/YYYY-MM-DD/execution_log.csv`** - Daily executions
```csv
symbol,side,shares,fill_price,fill_time,commission,notes
AAPL,buy,100,150.28,09:45:12,0.00,Full fill
```

**Created daily** when you execute trades.

### Files Generated by System:

**3. `live/YYYY-MM-DD/trades.csv`** - Recommendations
- Generated by daily_live_runner.py
- What you should execute

**4. `live/YYYY-MM-DD/APPROVED`** - Approval marker
- Created when you approve trades
- Empty file, presence = approved

**5. `live/YYYY-MM-DD/approval_log.json`** - Audit trail
```json
{
  "date": "2025-10-30",
  "approved_by": "user",
  "approved_at": "2025-10-30 08:15:00",
  "trades_count": 28,
  "total_notional": 98450.00,
  "notes": "All safety checks passed"
}
```

---

## 🔧 Position Management

### How the System Knows Your Positions

**Simple: It reads `live/current_positions.csv`**

```python
# In daily_live_runner.py
current_positions = pd.read_csv('live/current_positions.csv')

# Converts to: {symbol: shares}
positions = dict(zip(current_positions['symbol'], current_positions['shares']))

# Uses this to calculate what trades to make
trades = generate_trades(
    current_positions=positions,  # Your current holdings
    target_weights=target_weights,  # ML recommended allocation
    current_prices=prices,
    portfolio_value=100000
)
```

### Updating Positions (Three Methods)

#### **Method 1: Automatic Update (Recommended)**
```bash
# After executing trades, run this
python scripts/update_positions.py \
  --execution-log live/2025-10-30/execution_log.csv

# Auto-updates current_positions.csv
```

#### **Method 2: Manual CSV Edit**
```bash
# Edit directly
nano live/current_positions.csv

# Update shares for symbols you traded
# Add new symbols (buys)
# Remove symbols (full sells)
```

#### **Method 3: Reimport from Broker**
```bash
# Download positions from broker (CSV export)
# Convert and replace current_positions.csv

python scripts/import_positions.py \
  --broker-file broker_export.csv \
  --output live/current_positions.csv \
  --force
```

---

## 🎯 Complete Example Walkthrough

### Day 1: Starting from Empty Portfolio

**1. Initialize (One-Time Setup)**
```bash
# Create empty positions file
echo "symbol,shares,avg_cost,last_updated" > live/current_positions.csv

# Configure portfolio value in config.yaml
# live:
#   initial_capital: 100000
```

**2. Generate First Recommendations**
```bash
python scripts/daily_live_runner.py --dry-run
```

**Output:**
```
Starting with empty portfolio, capital: $100,000
Regime: Normal
Recommending 20 positions (100% allocation)
Generated trades.csv with 20 BUY orders
```

**3. Review and Approve**
```bash
python scripts/approve_trades.py live/2025-10-30/trades.csv
```

```
Do you approve? yes

✓ Approved 20 trades for execution
```

**4. Execute at Broker**
- Log into broker
- Execute all 20 buy orders from trades.csv
- Note execution prices

**5. Log Executions**
```bash
python scripts/log_execution.py

# For each trade, enter:
# Symbol: AAPL
# Side: buy
# Shares: 100
# Fill Price: 150.28
# ...repeat for all trades...

✓ Logged 20 executions to execution_log.csv
```

**6. Update Positions**
```bash
python scripts/update_positions.py \
  --execution-log live/2025-10-30/execution_log.csv
```

**Result:**
```
✓ Portfolio initialized with 20 positions
✓ Total value: $98,450 (used)
✓ Cash remaining: $1,550
✓ Positions saved to: live/current_positions.csv
```

**Your `current_positions.csv` now contains:**
```csv
symbol,shares,avg_cost,last_updated
AAPL,100,150.28,2025-10-30
MSFT,80,380.45,2025-10-30
GOOGL,25,142.80,2025-10-30
...
```

### Day 2: Rebalancing

**1. Generate Recommendations (Morning)**
```bash
python scripts/daily_live_runner.py --dry-run
```

**Script now:**
- Reads your current positions from `current_positions.csv`
- Calculates new target weights
- Generates trades to rebalance

**Output:**
```
Loaded 20 current positions (value: $101,230)
Regime: Normal
Generating rebalance trades...
Generated trades.csv with 15 trades (8 buys, 7 sells)
Turnover: 12.5%
```

**2. Approve and Execute**
```bash
# Approve
python scripts/approve_trades.py live/2025-10-31/trades.csv

# Execute at broker (manual)
# Log executions
python scripts/log_execution.py

# Update positions
python scripts/update_positions.py \
  --execution-log live/2025-10-31/execution_log.csv
```

**Repeat daily!**

---

## 🛡️ Safety Features

### Built-in Safeguards

**1. Approval Required**
- Trades are NOT executed unless you approve
- Approval script shows full summary
- You can reject and investigate

**2. Position Validation**
```bash
# Verify positions match reality
python scripts/validate_positions.py \
  --system live/current_positions.csv \
  --broker broker_export.csv
```

Shows discrepancies if any.

**3. Reconciliation (Weekly)**
```bash
# Compare system vs broker positions
python scripts/reconcile_positions.py \
  --current-positions live/current_positions.csv \
  --broker-positions broker_export.csv
```

Ensures no drift between system and broker.

---

## 🚨 Common Scenarios

### Scenario 1: Partial Fill

**Problem:** Ordered 100 shares, only filled 50

**Solution:**
```bash
# In execution log, record actual fill
symbol,side,shares,fill_price,fill_time,commission,notes
AAPL,buy,50,150.28,09:45:12,0.00,Partial fill (50/100)

# Update positions will use actual shares (50, not 100)
python scripts/update_positions.py
```

**Next day:**
- System sees you have 50 shares (not 100)
- May recommend buying 50 more if still high-ranked

### Scenario 2: Rejected Trade (Insufficient Funds)

**Problem:** Can't execute trade, insufficient buying power

**Solution:**
```bash
# Don't log the execution
# Leave it out of execution_log.csv

# Update positions without that trade
python scripts/update_positions.py

# System position = broker position (no discrepancy)
```

### Scenario 3: Manual Override Trade

**Problem:** You manually bought TSLA (not in recommendations)

**Solution:**
```bash
# Add to execution_log.csv
echo "TSLA,buy,50,250.00,10:30:00,0.00,Manual purchase" >> \
  live/2025-10-30/execution_log.csv

# Update positions
python scripts/update_positions.py

# TSLA now in current_positions.csv
# Next day, system may recommend selling it (if not in top 20)
```

### Scenario 4: Position Drift (Forgot to Update)

**Problem:** Traded yesterday but forgot to update positions

**Solution:**
```bash
# Export current positions from broker
# [Download CSV from broker website]

# Import to overwrite system positions
python scripts/import_positions.py \
  --broker-file my_broker_export.csv \
  --output live/current_positions.csv \
  --force

✓ Positions synchronized from broker
```

---

## 📊 Monitoring & Reports

### Daily Review

**After updating positions, review:**
```bash
# Generate performance report
python scripts/daily_performance.py

# Shows:
# - Today's P&L
# - Portfolio value
# - Top gainers/losers
# - Position weights
```

### Weekly Reconciliation

```bash
# Every Friday
python scripts/weekly_reconciliation.py

# Compares:
# - System positions vs broker positions
# - Expected portfolio value vs actual
# - Transaction costs this week
# - IC performance
```

---

## 🎓 Best Practices

### 1. **Daily Discipline**
- Run daily_live_runner.py every morning
- Review and approve trades before 9:30 AM
- Execute trades in first hour (9:35-10:30)
- Update positions same day (before 5 PM)

### 2. **Audit Trail**
- Keep all execution_log.csv files (don't delete)
- Keep approval_log.json files
- Useful for performance attribution

### 3. **Weekly Sync**
- Export positions from broker (Friday EOD)
- Run reconciliation script
- Fix any discrepancies immediately

### 4. **Backup**
```bash
# Weekly backup of positions
cp live/current_positions.csv \
   backups/positions_$(date +%Y%m%d).csv
```

### 5. **Cash Management**
```bash
# Track cash separately
# In config.yaml:
live:
  initial_capital: 100000

# After each day, update actual capital if needed
# (deposits/withdrawals)
```

---

## 🔧 Helper Scripts Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `approve_trades.py` | Interactive approval | Daily (morning) |
| `log_execution.py` | Record trade fills | Daily (as you trade) |
| `update_positions.py` | Update current positions | Daily (EOD) |
| `import_positions.py` | Import from broker | Weekly (sync) |
| `validate_positions.py` | Check for discrepancies | Weekly |
| `daily_performance.py` | Daily P&L report | Daily (EOD) |
| `weekly_reconciliation.py` | Full reconciliation | Weekly |

---

## 📋 Quick Reference

### Minimal Daily Workflow

```bash
# Morning (8:00 AM)
python scripts/daily_live_runner.py --dry-run
python scripts/approve_trades.py live/$(date +%Y-%m-%d)/trades.csv

# Execute trades manually at broker (9:35 AM - 3:00 PM)

# Evening (4:00 PM)
python scripts/log_execution.py
python scripts/update_positions.py
```

**That's it! 3 commands + manual execution.**

---

## 🆘 Troubleshooting

### "Script says I have wrong positions"

```bash
# Fix: Import actual positions from broker
python scripts/import_positions.py \
  --broker-file broker_export.csv --force
```

### "I executed some trades but not all"

```bash
# Fix: Only log the trades you actually executed
# Leave others out of execution_log.csv
# System will recommend them again tomorrow if needed
```

### "Positions file is corrupted"

```bash
# Fix: Recreate from broker export
python scripts/import_positions.py \
  --broker-file broker_export.csv \
  --output live/current_positions.csv --force
```

---

## 🎯 Summary

**Manual trading workflow = Simple file-based position tracking**

**The system:**
- Generates recommendations → `trades.csv`
- Reads your positions → `current_positions.csv`
- You approve → interactive script
- You execute → manual at broker
- You record → `execution_log.csv`
- You update → `update_positions.py`

**No broker API needed. Full control. 100% transparent.**

---

**Ready to start?**
1. Initialize: `echo "symbol,shares,avg_cost,last_updated" > live/current_positions.csv`
2. Generate: `python scripts/daily_live_runner.py --dry-run`
3. Trade manually!
