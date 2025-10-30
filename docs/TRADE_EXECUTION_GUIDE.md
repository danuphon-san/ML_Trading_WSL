# Trade Execution Guide

**Version**: 1.0
**Date**: 2025-10-30
**Status**: Production Ready

---

## 📋 Table of Contents

1. [Understanding Trade Recommendations](#understanding-trade-recommendations)
2. [Manual Trade Approval Workflow](#manual-trade-approval-workflow)
3. [Manual Execution Process](#manual-execution-process)
4. [Position Reconciliation](#position-reconciliation)
5. [Cost Tracking & Attribution](#cost-tracking--attribution)
6. [Broker Integration (Future)](#broker-integration-future)
7. [Troubleshooting](#troubleshooting)

---

## 📊 Understanding Trade Recommendations

### trades.csv Format

After running the daily workflow, you'll find `live/YYYY-MM-DD/trades.csv` with this structure:

```csv
symbol,side,shares,price,notional,weight_change,commission_est,slippage_est,total_cost_est
AAPL,buy,100,150.25,15025.00,0.0500,1.50,3.01,4.51
MSFT,sell,50,380.50,19025.00,-0.0300,1.90,3.81,5.71
GOOGL,buy,25,142.80,3570.00,0.0350,0.36,0.71,1.07
```

### Column Definitions

| Column | Description | Example | Notes |
|--------|-------------|---------|-------|
| **symbol** | Stock ticker | AAPL | Standard ticker symbol |
| **side** | Trade direction | buy / sell | BUY to increase position, SELL to decrease |
| **shares** | Number of shares | 100 | Always positive, direction indicated by 'side' |
| **price** | Current market price | 150.25 | Price at signal generation time |
| **notional** | Dollar value of trade | 15025.00 | = shares × price |
| **weight_change** | Portfolio weight change | 0.0500 | Target weight - current weight |
| **commission_est** | Estimated commission | 1.50 | Based on `costs_bps` in config (default 1 bps) |
| **slippage_est** | Estimated slippage | 3.01 | Based on `slippage_bps` in config (default 2 bps) |
| **total_cost_est** | Total transaction cost | 4.51 | Commission + slippage |

### Key Metrics to Review

**In the trades.csv file header (commented rows):**
```csv
# Date: 2025-10-30
# Total Trades: 28
# Buy Orders: 15
# Sell Orders: 13
# Total Notional: $98,450.00
# Estimated Costs: $295.35 (0.30% of notional)
# Portfolio Turnover: 18.5%
# Regime: Normal (Risk Multiplier: 1.0)
```

**What to Check:**
- ✅ **Turnover < 35%**: High turnover = high costs
- ✅ **Total cost < 0.5%**: Transaction costs eating returns
- ✅ **Trade count reasonable**: Too many small trades = inefficient
- ✅ **Regime makes sense**: Does risk multiplier match market conditions?

---

## ✅ Manual Trade Approval Workflow

### Step-by-Step Process

#### **1. Review Daily Report (5 minutes)**

```bash
# Open HTML report
open live/$(date +%Y-%m-%d)/report.html

# Or view in terminal
cat live/$(date +%Y-%m-%d)/trades.csv
```

**Check these sections:**

**A. Regime Detection**
- Current regime: Risk-Off / Normal / Risk-On
- Volatility level
- Market conditions
- Does the regime make sense given recent news/events?

**B. Portfolio Summary**
- Number of positions (should be ~20 for default top_k)
- Total portfolio value
- Largest positions (should be < 15%)
- Sector concentration (should be < 30% per sector)

**C. Safety Checks**
- All safety checks PASSED? ✅
- Any warnings or violations? ⚠️
- Kill-switch status: ACTIVE or TRIGGERED

**D. Model Health**
- IC (Information Coefficient): Should be > 0.02
- Recent IC trend: Improving or degrading?
- Quantile performance: Top quantile outperforming bottom?

#### **2. Validate Trade Recommendations (10 minutes)**

**Run validation script:**
```bash
python scripts/validate_trades.py live/$(date +%Y-%m-%d)/trades.csv
```

**Manual checks:**

**A. Price Sanity**
```bash
# Compare recommended prices to current market
# Flag any price differences > 2%
```

- Are prices reasonable vs. current market?
- Any stale prices (check timestamp)?
- Large price movements since signal generation?

**B. Position Sizes**
- No single trade > 10% of portfolio value
- Share quantities are reasonable (no fractional shares)
- Notional values make sense (shares × price)

**C. Trade Direction**
- Do BUY orders align with positive scores?
- Do SELL orders align with reduced/negative scores?
- Any unexpected reversals?

**D. Cost Analysis**
- Total transaction costs < 0.5% of portfolio
- High-cost trades flagged (> $50 per trade)?
- Turnover reasonable for strategy?

#### **3. Cross-Reference with External Data (5 minutes)**

**Check for:**

**A. Corporate Events (Avoid Trading)**
- Earnings announcements (within 1 day)
- Ex-dividend dates
- Stock splits
- M&A announcements

**B. Market Conditions**
- Market holidays
- Half-day sessions
- High volatility events (FOMC, jobs report)

**C. Liquidity Concerns**
- Average daily volume for each ticker
- Bid-ask spreads reasonable?
- Any halted stocks?

#### **4. Make Approval Decision**

**Decision Matrix:**

| Condition | Action |
|-----------|--------|
| ✅ All safety checks passed<br>✅ IC > 0.02<br>✅ Regime reasonable<br>✅ No corporate events | **APPROVE** - Proceed with execution |
| ⚠️ IC between 0.01-0.02<br>⚠️ Turnover 30-35%<br>⚠️ Minor price staleness | **CONDITIONAL APPROVE** - Execute with caution, adjust prices |
| 🚨 Kill-switch triggered<br>🚨 IC < 0.01<br>🚨 Safety checks failed | **REJECT** - Do not trade, investigate issues |

**Document decision:**
```bash
echo "2025-10-30,APPROVED,All checks passed,John Doe" >> live/trade_approvals.csv
```

---

## 🔧 Manual Execution Process

### Preparation

#### **1. Update Current Positions File**

Before executing trades, ensure current positions are accurate:

```bash
# Edit current positions
nano live/current_positions.csv
```

Format:
```csv
symbol,shares,avg_cost
AAPL,150,145.50
MSFT,75,365.20
```

#### **2. Calculate Available Cash**

```python
# Quick calculation
portfolio_value = 100000  # Your current portfolio value
cash = portfolio_value - sum(shares × current_price for all positions)
```

### Execution Methods

#### **Method 1: Manual Broker Entry (Most Common)**

**For each trade in trades.csv:**

1. **Log into broker platform** (Fidelity, Schwab, IBKR, etc.)

2. **For BUY orders:**
   ```
   Ticker: AAPL
   Action: BUY
   Quantity: 100
   Order Type: LIMIT
   Limit Price: $150.25 (or current bid/ask midpoint)
   Time in Force: DAY
   ```

3. **For SELL orders:**
   ```
   Ticker: MSFT
   Action: SELL
   Quantity: 50
   Order Type: LIMIT
   Limit Price: $380.50 (or current bid/ask midpoint)
   Time in Force: DAY
   ```

4. **Submit and track:**
   - Note order ID
   - Monitor fill status
   - Record actual execution price
   - Update execution log

#### **Method 2: Batch CSV Upload (Some Brokers)**

Some brokers (Interactive Brokers, TD Ameritrade) support CSV uploads:

1. **Convert to broker format:**
   ```bash
   python scripts/convert_to_broker_format.py \
     --input live/2025-10-30/trades.csv \
     --broker ibkr \
     --output trades_ibkr.csv
   ```

2. **Review converted file:**
   ```csv
   Symbol,Action,Quantity,OrderType,LimitPrice,TIF
   AAPL,BUY,100,LMT,150.25,DAY
   MSFT,SELL,50,LMT,380.50,DAY
   ```

3. **Upload to broker platform**
   - Log into broker web interface
   - Navigate to "Batch Orders" or "Import Orders"
   - Upload CSV file
   - Review orders before submission
   - Submit batch

4. **Monitor execution**

### Execution Best Practices

#### **Timing**

**Optimal execution windows:**
- **9:35-10:00 AM ET**: After opening volatility settles
- **2:00-3:00 PM ET**: Afternoon liquidity
- **Avoid**: First 5 minutes (9:30-9:35), last 10 minutes (3:50-4:00)

#### **Order Types**

**Recommended approach:**
```
1. Start with LIMIT orders at midpoint (bid + ask) / 2
2. If no fill after 15 minutes, adjust price by $0.05
3. If urgency, use MARKET order (accept slippage)
4. For large orders, consider VWAP or TWAP algos
```

#### **Order Sizing**

For large orders (> 5% of daily volume):
```python
# Split large order into chunks
total_shares = 1000
daily_volume = 500000
max_pct = 0.05  # Trade max 5% of volume

if total_shares / daily_volume > max_pct:
    # Split across multiple days
    daily_chunks = total_shares / (total_shares / (daily_volume * max_pct))
    print(f"Execute over {math.ceil(daily_chunks)} days")
```

### Execution Log

**Track every trade:**
```csv
date,symbol,side,shares,limit_price,fill_price,fill_time,order_id,commission,slippage,notes
2025-10-30,AAPL,buy,100,150.25,150.28,09:45:12,ORD123,1.50,3.00,Partial fill
2025-10-30,MSFT,sell,50,380.50,380.45,10:12:34,ORD124,1.90,2.50,Full fill
```

Save as: `live/YYYY-MM-DD/execution_log.csv`

---

## 🔄 Position Reconciliation

### Daily Reconciliation Process

#### **1. Collect Execution Data (End of Day)**

```bash
# After market close (4:00 PM ET)
# Download fills from broker
# Save as: live/2025-10-30/broker_fills.csv
```

#### **2. Reconcile Positions**

```python
# Run reconciliation script
python scripts/reconcile_positions.py \
  --trades live/2025-10-30/trades.csv \
  --fills live/2025-10-30/broker_fills.csv \
  --current-positions live/current_positions.csv
```

**Script checks:**
- ✅ All recommended trades executed?
- ✅ Execution prices vs. recommended prices (slippage)
- ✅ Actual commission vs. estimated
- ✅ Current positions match broker account

**Output:**
```
Reconciliation Report - 2025-10-30
===================================
Trades Recommended: 28
Trades Executed: 27 (96.4%)
Not Executed: 1 (TSLA - insufficient liquidity)

Price Slippage:
  Average: 0.08% (target: 0.02%)
  Worst: NVDA +0.25%

Commission:
  Estimated: $295.35
  Actual: $312.50
  Variance: +$17.15 (+5.8%)

Position Matches: ✅
Cash Balance: ✅
```

#### **3. Update Current Positions**

After reconciliation:
```bash
# Script automatically updates
cp live/2025-10-30/positions_after_recon.csv live/current_positions.csv
```

**Verify manually:**
```csv
symbol,shares,avg_cost,current_value,weight,last_updated
AAPL,250,148.20,37562.50,0.0501,2025-10-30
MSFT,25,378.40,9510.00,0.0127,2025-10-30
```

#### **4. Document Discrepancies**

**If positions don't match:**

```bash
# Create discrepancy report
echo "2025-10-30,TSLA,Recommended: 50,Executed: 0,Reason: Insufficient liquidity" \
  >> live/discrepancies.log
```

**Common reasons:**
- Insufficient buying power
- Stock halted
- Limit order not filled
- Manual override decision

---

## 📈 Cost Tracking & Attribution

### Transaction Cost Analysis (TCA)

#### **Daily TCA Report**

```python
python scripts/calculate_tca.py live/2025-10-30/
```

**Output:**
```
Transaction Cost Analysis - 2025-10-30
========================================

1. Commission Costs:
   Estimated: $295.35
   Actual:    $312.50
   Variance:  +$17.15 (+5.8%)

2. Slippage Costs:
   Estimated:    $590.70 (2 bps)
   Actual:       $785.40 (2.67 bps)
   Variance:     +$194.70 (+33%)
   Breakdown:
     - Favorable:   15 trades, saved $125.50
     - Unfavorable: 12 trades, cost $320.20

3. Opportunity Costs:
   Unfilled: 1 trade (TSLA)
   Missed PnL: -$85.00 (stock moved away)

4. Total Costs:
   Total Transaction Costs: $1,097.90
   As % of Notional: 0.37%
   As % of Portfolio: 0.11%

5. Cost Attribution by Symbol:
   Worst: NVDA ($125.50, 0.85% of trade)
   Best: AAPL (-$12.30, -0.08% of trade)
```

### Monthly Cost Summary

```python
# Generate monthly report
python scripts/monthly_cost_report.py --month 2025-10
```

**Tracks:**
- Total commission paid
- Average slippage per trade
- Cost as % of portfolio value
- Turnover vs. costs correlation
- Broker comparison (if using multiple)

### Cost Attribution to Strategy

```python
# Attribute costs to performance
# Gross Return: +2.5%
# Transaction Costs: -0.3%
# Net Return: +2.2%
```

**Track in spreadsheet:**
```csv
month,gross_return,commission_costs,slippage_costs,net_return
2025-10,0.0250,0.0015,0.0015,0.0220
2025-11,0.0180,0.0012,0.0018,0.0150
```

---

## 🤖 Broker Integration (Future)

### Automated Execution Architecture

```
┌─────────────────────────────────────┐
│  Daily Live Runner                  │
│  (daily_live_runner.py)            │
└──────────────┬──────────────────────┘
               │ generates
               ▼
        ┌──────────────┐
        │  trades.csv  │
        └──────┬───────┘
               │ consumed by
               ▼
┌─────────────────────────────────────┐
│  Trade Executor                     │
│  (src/execution/executor.py)       │
└──────────────┬──────────────────────┘
               │ calls
               ▼
    ┌─────────────────────┐
    │  Broker API Client  │
    │  (IBKR/Alpaca/TD)  │
    └──────────┬──────────┘
               │ executes
               ▼
        ┌─────────────┐
        │   Broker    │
        └─────────────┘
```

### Supported Brokers (Future Implementation)

#### **Option 1: Interactive Brokers (IBKR)**

**Pros:**
- ✅ Institutional-grade execution
- ✅ Low commissions ($0.005/share, $1 min)
- ✅ Excellent API (IB Gateway)
- ✅ Direct market access

**Cons:**
- ❌ Complex API setup
- ❌ Minimum account size ($10k)
- ❌ API requires always-on server

**Implementation:**
```python
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

class IBKRExecutor(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def execute_trades(self, trades_df):
        """Execute trades via IBKR API"""
        for _, trade in trades_df.iterrows():
            contract = self._create_contract(trade['symbol'])
            order = self._create_order(
                action=trade['side'].upper(),
                quantity=trade['shares'],
                limit_price=trade['price']
            )
            self.placeOrder(self.next_order_id, contract, order)

    def _create_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def _create_order(self, action, quantity, limit_price):
        order = Order()
        order.action = action  # "BUY" or "SELL"
        order.totalQuantity = quantity
        order.orderType = "LMT"
        order.lmtPrice = limit_price
        order.tif = "DAY"
        return order
```

**Setup steps:**
1. Install IB Gateway or TWS
2. Enable API access in settings
3. Install `ibapi` package: `pip install ibapi`
4. Configure connection (localhost:7497)

---

#### **Option 2: Alpaca (Recommended for Beginners)**

**Pros:**
- ✅ Commission-free trading
- ✅ Simple REST API
- ✅ Paper trading environment
- ✅ No minimum account size
- ✅ Easy to integrate

**Cons:**
- ❌ Retail execution quality
- ❌ Limited order types
- ❌ US stocks only

**Implementation:**
```python
import alpaca_trade_api as tradeapi

class AlpacaExecutor:
    def __init__(self, api_key, api_secret, paper=True):
        """
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca secret key
            paper: Use paper trading (default: True)
        """
        base_url = 'https://paper-api.alpaca.markets' if paper \
                   else 'https://api.alpaca.markets'
        self.api = tradeapi.REST(api_key, api_secret, base_url)

    def execute_trades(self, trades_df):
        """Execute trades via Alpaca API"""
        orders = []

        for _, trade in trades_df.iterrows():
            try:
                order = self.api.submit_order(
                    symbol=trade['symbol'],
                    qty=trade['shares'],
                    side=trade['side'],  # 'buy' or 'sell'
                    type='limit',
                    time_in_force='day',
                    limit_price=round(trade['price'], 2)
                )
                orders.append(order)
                print(f"✓ Order submitted: {trade['symbol']} {trade['side']} {trade['shares']}@${trade['price']}")
            except Exception as e:
                print(f"✗ Order failed: {trade['symbol']} - {e}")

        return orders

    def get_positions(self):
        """Get current positions"""
        positions = self.api.list_positions()
        return {p.symbol: int(p.qty) for p in positions}

    def get_account(self):
        """Get account info"""
        account = self.api.get_account()
        return {
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'buying_power': float(account.buying_power)
        }
```

**Setup steps:**
1. Sign up: https://alpaca.markets
2. Get API keys (paper trading)
3. Install: `pip install alpaca-trade-api`
4. Set environment variables:
   ```bash
   export ALPACA_API_KEY="your_key"
   export ALPACA_SECRET_KEY="your_secret"
   ```

**Usage:**
```python
# In daily_live_runner.py
from src.execution.alpaca_executor import AlpacaExecutor

executor = AlpacaExecutor(
    api_key=os.getenv('ALPACA_API_KEY'),
    api_secret=os.getenv('ALPACA_SECRET_KEY'),
    paper=True  # Start with paper trading
)

# Execute trades
trades_df = pd.read_csv('live/2025-10-30/trades.csv')
orders = executor.execute_trades(trades_df)

# Monitor fills
for order in orders:
    filled_order = executor.api.get_order(order.id)
    print(f"{order.symbol}: {filled_order.status}")
```

---

#### **Option 3: TD Ameritrade**

**Pros:**
- ✅ No commissions (stocks/ETFs)
- ✅ Good API documentation
- ✅ Paper trading via thinkOnDemand
- ✅ Large broker, stable

**Cons:**
- ❌ API throttling limits
- ❌ OAuth authentication complexity
- ❌ Account approval process

**Implementation:**
```python
import requests
from requests_oauthlib import OAuth2Session

class TDAmeritradeExecutor:
    def __init__(self, client_id, refresh_token):
        self.client_id = client_id
        self.access_token = self._refresh_access_token(refresh_token)
        self.base_url = 'https://api.tdameritrade.com/v1'

    def execute_trades(self, trades_df, account_id):
        """Execute trades via TD Ameritrade API"""
        for _, trade in trades_df.iterrows():
            order_spec = {
                "orderType": "LIMIT",
                "session": "NORMAL",
                "duration": "DAY",
                "orderStrategyType": "SINGLE",
                "price": round(trade['price'], 2),
                "orderLegCollection": [{
                    "instruction": "BUY" if trade['side'] == 'buy' else "SELL",
                    "quantity": trade['shares'],
                    "instrument": {
                        "symbol": trade['symbol'],
                        "assetType": "EQUITY"
                    }
                }]
            }

            response = requests.post(
                f"{self.base_url}/accounts/{account_id}/orders",
                headers={"Authorization": f"Bearer {self.access_token}"},
                json=order_spec
            )

            if response.status_code == 201:
                print(f"✓ Order placed: {trade['symbol']}")
            else:
                print(f"✗ Order failed: {response.text}")
```

---

### Generic Executor Interface

**Create unified interface for all brokers:**

```python
# src/execution/base_executor.py
from abc import ABC, abstractmethod

class BaseExecutor(ABC):
    """Base class for all broker executors"""

    @abstractmethod
    def execute_trades(self, trades_df):
        """Execute trades from DataFrame"""
        pass

    @abstractmethod
    def get_positions(self):
        """Get current positions"""
        pass

    @abstractmethod
    def get_account_info(self):
        """Get account information"""
        pass

    @abstractmethod
    def cancel_order(self, order_id):
        """Cancel pending order"""
        pass

    @abstractmethod
    def get_order_status(self, order_id):
        """Get order fill status"""
        pass
```

**Usage in daily runner:**
```python
# src/execution/executor_factory.py
def get_executor(broker_name, config):
    """Factory to create broker-specific executor"""
    if broker_name == 'alpaca':
        return AlpacaExecutor(
            api_key=config['alpaca']['api_key'],
            api_secret=config['alpaca']['api_secret'],
            paper=config['alpaca']['paper']
        )
    elif broker_name == 'ibkr':
        return IBKRExecutor(
            host=config['ibkr']['host'],
            port=config['ibkr']['port'],
            client_id=config['ibkr']['client_id']
        )
    elif broker_name == 'td':
        return TDAmeritradeExecutor(
            client_id=config['td']['client_id'],
            refresh_token=config['td']['refresh_token']
        )
    else:
        raise ValueError(f"Unknown broker: {broker_name}")

# In daily_live_runner.py
executor = get_executor(
    broker_name=config['execution']['broker'],
    config=config['execution']
)
orders = executor.execute_trades(trades_df)
```

---

### Configuration for Automated Execution

**Add to config.yaml:**
```yaml
# Execution Configuration (for automated broker integration)
execution:
  enabled: false  # Set true when ready for automation
  broker: "alpaca"  # alpaca, ibkr, td
  paper_trading: true  # Always start with paper trading

  # Risk controls
  max_order_size_usd: 10000  # Max $10k per order
  max_daily_trades: 50
  require_human_approval: true  # Manual approval before execution

  # Order parameters
  default_order_type: "limit"  # limit, market, stop_limit
  order_timeout_minutes: 30  # Cancel unfilled orders after 30 min
  retry_failed_orders: false

  # Broker-specific configs
  alpaca:
    api_key: ${ALPACA_API_KEY}
    api_secret: ${ALPACA_SECRET_KEY}
    paper: true
    base_url: "https://paper-api.alpaca.markets"

  ibkr:
    host: "localhost"
    port: 7497  # Paper: 7497, Live: 7496
    client_id: 1
    account_id: ${IBKR_ACCOUNT_ID}

  td:
    client_id: ${TD_CLIENT_ID}
    refresh_token: ${TD_REFRESH_TOKEN}
    account_id: ${TD_ACCOUNT_ID}
```

---

## 🔍 Troubleshooting

### Common Issues

#### **1. Trades Not Executing**

**Symptom**: Limit orders not filling

**Solutions:**
```
- Check bid-ask spread (wide spread = use market order)
- Adjust limit price ($0.05 increments)
- Check stock volatility (use wider limits)
- Verify sufficient buying power
- Check for stock halts (trading suspended)
```

#### **2. Position Mismatch After Reconciliation**

**Symptom**: Broker positions ≠ system positions

**Debug steps:**
```bash
# 1. Export broker positions
# Download from broker platform

# 2. Compare with system
python scripts/compare_positions.py \
  --system live/current_positions.csv \
  --broker broker_positions.csv

# 3. Identify discrepancies
# Output shows differences
```

**Common causes:**
- Manual trades not recorded in system
- Corporate actions (splits, dividends)
- Previous day trades not reconciled
- Fractional shares from DRIP

**Fix:**
```bash
# Manual override
python scripts/sync_positions.py \
  --source broker \
  --force-update
```

#### **3. High Slippage**

**Symptom**: Actual execution price far from recommended price

**Analysis:**
```python
# Calculate slippage
slippage_bps = (fill_price - signal_price) / signal_price * 10000

# If slippage > 10 bps consistently:
# - Use limit orders instead of market
# - Trade during liquid hours (10 AM - 3 PM)
# - Split large orders across time
# - Avoid low-volume stocks
```

#### **4. Insufficient Buying Power**

**Symptom**: Broker rejects orders due to insufficient cash

**Solutions:**
```python
# 1. Check available cash before trading
cash = executor.get_account()['buying_power']

# 2. Filter trades by available cash
affordable_trades = trades_df[
    trades_df['notional'].cumsum() <= cash * 0.95  # 5% buffer
]

# 3. Prioritize by conviction
affordable_trades = trades_df.nlargest(20, 'ml_score')
```

#### **5. API Rate Limiting**

**Symptom**: Broker API returns "429 Too Many Requests"

**Solutions:**
```python
import time

# Add delays between orders
for trade in trades_df.iterrows():
    executor.execute_trade(trade)
    time.sleep(0.5)  # 500ms delay

# Or batch orders
executor.execute_batch(trades_df)  # Single API call
```

### Emergency Procedures

#### **Kill-Switch Triggered**

**If kill-switch activates during trading day:**

1. **STOP** all trading immediately
2. **CANCEL** all pending orders
3. **REVIEW** positions and P&L
4. **INVESTIGATE** cause:
   - Daily loss > 3%?
   - Sharpe ratio degraded?
   - Market crash?
   - Model malfunction?
5. **DECISION**:
   - If market crash: Keep positions (don't panic sell)
   - If model issue: Close positions gradually
   - If temporary: Monitor, resume next day

**Script:**
```bash
# Cancel all pending orders
python scripts/emergency_cancel_all.py

# Get current P&L
python scripts/calculate_pnl.py --today

# Generate incident report
python scripts/kill_switch_report.py > incident_$(date +%Y%m%d).txt
```

#### **Broker API Down**

**If broker API is unavailable:**

1. **SWITCH** to manual execution
2. **EXPORT** trades.csv to Excel
3. **LOG IN** to broker web interface
4. **ENTER** orders manually
5. **DOCUMENT** fills in execution_log.csv

---

## 📚 Additional Resources

### Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/validate_trades.py` | Validate trade recommendations |
| `scripts/reconcile_positions.py` | Daily position reconciliation |
| `scripts/calculate_tca.py` | Transaction cost analysis |
| `scripts/convert_to_broker_format.py` | Convert CSV to broker format |
| `scripts/compare_positions.py` | Compare system vs broker positions |
| `scripts/emergency_cancel_all.py` | Cancel all pending orders |

### Best Practices

1. **Always start with paper trading** (2-4 weeks minimum)
2. **Reconcile positions daily** (never skip this)
3. **Track slippage** (if consistently high, adjust execution)
4. **Review monthly costs** (optimize if > 0.5% of returns)
5. **Document all manual overrides** (audit trail)
6. **Test broker integration thoroughly** before live
7. **Have backup plan** (manual execution if API fails)

### Configuration Checklist

Before going live:
- [ ] Broker account funded
- [ ] API keys generated and tested
- [ ] Paper trading completed (2+ weeks)
- [ ] Transaction costs calibrated
- [ ] Position reconciliation tested
- [ ] Emergency procedures documented
- [ ] Manual execution process tested
- [ ] Kill-switch tested and working

---

## 📞 Support

**Questions?**
- Review `docs/DAILY_LIVE_QUICKSTART.md` for daily operations
- Review `TESTING_GUIDE.md` for testing procedures
- Check `CLAUDE.md` for architecture details

**Ready to execute?** Start with manual execution, then transition to automated broker integration when comfortable.
