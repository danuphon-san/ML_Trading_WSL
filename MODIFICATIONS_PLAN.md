# Production ML Trading System - Modifications Plan

**Date**: 2025-10-30
**Status**: Ready for Implementation
**Approach**: Enhance existing pipeline (not rebuild)

---

## 📋 Table of Contents
1. [Option A: Daily Live Runner](#option-a-daily-live-runner) (Week 1-2)
2. [Option B: Model Comparison Framework](#option-b-model-comparison-framework) (Week 3-4)
3. [Implementation Order](#implementation-order)
4. [Testing Strategy](#testing-strategy)
5. [Success Criteria](#success-criteria)

---

# Option A: Daily Live Runner

**Goal**: Create end-to-end daily workflow for generating trade recommendations

**Timeline**: 1-2 days implementation + 1 day testing

---

## A1. New Files to Create

### 1. `scripts/daily_live_runner.py` ⭐ **MAIN SCRIPT**

**Purpose**: Orchestrates complete daily live trading workflow

**Workflow**:
```
1. Pre-flight checks (market open, data available)
2. Update data (OHLCV + fundamentals)
3. Load champion model
4. Generate ML scores for universe
5. Detect current market regime
6. Construct portfolio (regime-aware)
7. Generate trade recommendations
8. Run safety checks (kill-switch, limits)
9. Generate reports (CSV + HTML + JSON)
10. Send notifications (optional email)
```

**Key Functions**:
```python
def main():
    # Orchestrator

def check_market_open() -> bool:
    # Skip weekends/holidays

def update_data(config):
    # Call daily_update_data.py

def load_champion_model(model_path):
    # Load from production/champion_model.pkl

def generate_signals(model, universe, config):
    # Score all stocks with ML model

def construct_live_portfolio(scores, prices, config):
    # Call construct_portfolio with regime adaptation

def generate_trades(current_positions, target_weights, prices, capital):
    # Calculate shares to buy/sell

def run_safety_checks(trades, portfolio, config):
    # Kill-switch, position limits, turnover

def generate_outputs(trades, portfolio, regime, performance):
    # CSV + HTML + JSON

def send_notifications(report_html, config):
    # Email/Slack alerts
```

**Outputs**:
- `live/YYYY-MM-DD/trades.csv` - Trade recommendations
- `live/YYYY-MM-DD/portfolio_weights.csv` - Target allocation
- `live/YYYY-MM-DD/signals.json` - Raw ML scores (audit trail)
- `live/YYYY-MM-DD/report.html` - Performance dashboard
- `live/YYYY-MM-DD/monitoring_log.json` - Model health metrics

**Dependencies**:
- Uses: `src/live/operations.py` (LivePortfolioManager)
- Uses: `src/portfolio/construct.py` (construct_portfolio)
- Uses: `src/live/monitoring.py` (OpsMonitor)
- Uses: `scripts/daily_update_data.py` (data refresh)

---

### 2. `src/live/trade_generator.py` ⭐ **NEW MODULE**

**Purpose**: Generate actual trade orders from target weights

**Key Class**:
```python
class TradeGenerator:
    def __init__(self, config):
        self.costs_bps = config['portfolio']['costs_bps']
        self.slippage_bps = config['portfolio']['slippage_bps']
        self.min_trade_size = config.get('live', {}).get('min_trade_size', 100)

    def generate_trades(
        self,
        current_positions: Dict[str, float],  # {symbol: shares}
        target_weights: Dict[str, float],     # {symbol: weight}
        current_prices: Dict[str, float],     # {symbol: price}
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        Calculate trade orders

        Returns:
            DataFrame with columns:
            - symbol: Stock ticker
            - side: 'BUY' or 'SELL'
            - shares: Number of shares to trade
            - price: Current market price
            - notional: Dollar value (shares × price)
            - weight_change: Target - Current weight
            - commission_est: Estimated commission cost
            - slippage_est: Estimated slippage cost
        """

    def calculate_turnover(self, trades: pd.DataFrame, portfolio_value: float) -> float:
        """Calculate portfolio turnover %"""

    def estimate_costs(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Estimate total transaction costs"""

    def filter_small_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Remove trades below minimum size"""

    def validate_liquidity(self, trades: pd.DataFrame, volume_data: pd.DataFrame) -> pd.DataFrame:
        """Flag trades that exceed daily volume limits"""
```

**Outputs**: Trade DataFrame with all necessary execution details

---

### 3. `src/live/safety_checks.py` ⭐ **NEW MODULE**

**Purpose**: Validate portfolio and trades before execution

**Key Class**:
```python
class SafetyValidator:
    def __init__(self, config):
        self.max_position_weight = config['portfolio']['pypfopt']['max_weight']
        self.max_turnover = config['portfolio']['max_turnover']
        self.max_sector_weight = config.get('risk', {}).get('max_sector_weight', 0.30)
        self.kill_switch_config = config.get('ops', {}).get('kill_switch', {})

    def validate_portfolio(
        self,
        target_weights: Dict[str, float],
        universe_data: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Validate portfolio constraints

        Checks:
        - Weights sum to ~1.0 (within 1%)
        - No position > max_weight
        - No sector > max_sector_weight
        - All weights >= 0 (long-only)

        Returns:
            (is_valid, list_of_violations)
        """

    def validate_trades(
        self,
        trades: pd.DataFrame,
        portfolio_value: float
    ) -> Tuple[bool, List[str]]:
        """
        Validate trade orders

        Checks:
        - Turnover < max_turnover
        - All symbols have valid prices
        - No single trade > 10% of portfolio
        - Total cost < 0.5% of portfolio value

        Returns:
            (is_valid, list_of_violations)
        """

    def check_kill_switch(
        self,
        equity_curve: pd.DataFrame,
        breaches: List[Dict]
    ) -> Dict[str, Any]:
        """
        Check if kill-switch should trigger

        Conditions:
        - Daily loss > 3%
        - 6-week Sharpe < 0.5

        Returns:
            {
                'triggered': bool,
                'reason': str or None,
                'daily_return': float,
                'rolling_sharpe': float
            }
        """

    def validate_liquidity(
        self,
        trades: pd.DataFrame,
        volume_data: pd.DataFrame,
        max_volume_pct: float = 0.05
    ) -> pd.DataFrame:
        """
        Flag illiquid trades

        Adds column 'liquidity_warning' to trades if:
        - Trade size > 5% of daily volume

        Returns:
            Enhanced trades DataFrame
        """
```

**Outputs**: Validation results with violations

---

## A2. Files to Modify

### 4. `src/live/operations.py` 🔧 **ENHANCE**

**Add these methods to `LivePortfolioManager` class**:

```python
def load_latest_model(self, model_dir: str = "production"):
    """Load champion model from production folder"""
    model_path = Path(model_dir) / "champion_model.pkl"
    if not model_path.exists():
        # Fallback to latest training
        model_path = Path("data/models/latest/model.pkl")

    import joblib
    self.model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return self.model

def get_current_positions(self, positions_file: str = None) -> Dict[str, float]:
    """
    Load current portfolio positions

    Args:
        positions_file: Path to positions CSV (default: live/current_positions.csv)

    Returns:
        Dict of {symbol: shares}
    """
    if positions_file is None:
        positions_file = "live/current_positions.csv"

    if not Path(positions_file).exists():
        logger.warning("No current positions file found, assuming empty portfolio")
        return {}

    df = pd.read_csv(positions_file)
    return dict(zip(df['symbol'], df['shares']))

def save_positions(self, positions: Dict[str, float], filepath: str = "live/current_positions.csv"):
    """Save current positions to file"""
    df = pd.DataFrame([
        {'symbol': symbol, 'shares': shares}
        for symbol, shares in positions.items()
    ])
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(positions)} positions to {filepath}")

def get_portfolio_value(self, positions: Dict[str, float], prices: Dict[str, float]) -> float:
    """Calculate total portfolio value"""
    return sum(shares * prices.get(symbol, 0) for symbol, shares in positions.items())
```

**Why**: These utilities are needed by daily runner

---

### 5. `src/live/monitoring.py` 🔧 **ENHANCE**

**Add these methods to `OpsMonitor` class**:

```python
def calculate_information_coefficient(
    self,
    scores: pd.Series,
    forward_returns: pd.Series
) -> Dict[str, float]:
    """
    Calculate Information Coefficient (IC)

    IC measures correlation between ML scores and actual forward returns

    Returns:
        {
            'ic': Pearson correlation,
            'rank_ic': Spearman correlation (robust to outliers),
            'p_value': Statistical significance
        }
    """
    from scipy.stats import pearsonr, spearmanr

    # Align and drop NaN
    aligned = pd.DataFrame({'score': scores, 'return': forward_returns}).dropna()

    ic, ic_pval = pearsonr(aligned['score'], aligned['return'])
    rank_ic, rank_ic_pval = spearmanr(aligned['score'], aligned['return'])

    return {
        'ic': float(ic),
        'rank_ic': float(rank_ic),
        'ic_p_value': float(ic_pval),
        'rank_ic_p_value': float(rank_ic_pval),
        'n_samples': len(aligned)
    }

def calculate_top_quantile_performance(
    self,
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    Calculate performance by score quantile

    Args:
        scores: DataFrame with [date, symbol, ml_score]
        returns: DataFrame with [date, symbol, forward_return]
        n_quantiles: Number of quantiles (default: 5 = quintiles)

    Returns:
        DataFrame with average return per quantile
        Expected: Top quantile (5) > Bottom quantile (1)
    """
    merged = pd.merge(scores, returns, on=['date', 'symbol'])
    merged['quantile'] = pd.qcut(merged['ml_score'], q=n_quantiles, labels=False) + 1

    quantile_perf = merged.groupby('quantile')['forward_return'].agg(['mean', 'std', 'count'])
    quantile_perf.columns = ['avg_return', 'std_return', 'n_stocks']

    return quantile_perf

def detect_model_drift(
    self,
    recent_ic: List[float],
    lookback: int = 20,
    threshold: float = 0.02
) -> Dict[str, Any]:
    """
    Detect model performance degradation

    Args:
        recent_ic: List of recent IC values (last N days)
        lookback: Window to calculate average IC
        threshold: Minimum acceptable IC

    Returns:
        {
            'degraded': bool,
            'avg_ic': float,
            'days_below_threshold': int,
            'action': 'continue' or 'review_model'
        }
    """
    if len(recent_ic) < lookback:
        return {'degraded': False, 'reason': 'insufficient_history'}

    avg_ic = np.mean(recent_ic[-lookback:])
    days_below = sum(1 for ic in recent_ic[-lookback:] if ic < threshold)

    degraded = avg_ic < threshold and days_below >= lookback * 0.5

    return {
        'degraded': degraded,
        'avg_ic': float(avg_ic),
        'days_below_threshold': int(days_below),
        'lookback': lookback,
        'threshold': threshold,
        'action': 'review_model' if degraded else 'continue'
    }

def check_regime_performance(
    self,
    equity_curve: pd.DataFrame,
    regime_history: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate performance breakdown by regime

    Returns:
        DataFrame with columns:
        - regime: 0 (Risk-Off), 1 (Normal), 2 (Risk-On)
        - avg_return: Average daily return
        - sharpe: Sharpe ratio
        - win_rate: % positive days
        - n_days: Number of days in regime
    """
    merged = pd.merge(equity_curve, regime_history, on='date')
    merged['return'] = merged['equity'].pct_change()

    regime_stats = []
    for regime_id in [0, 1, 2]:
        regime_data = merged[merged['regime'] == regime_id]
        returns = regime_data['return'].dropna()

        if len(returns) > 0:
            avg_return = returns.mean()
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            win_rate = (returns > 0).sum() / len(returns)
        else:
            avg_return = sharpe = win_rate = 0

        regime_stats.append({
            'regime': regime_id,
            'regime_name': {0: 'Risk-Off', 1: 'Normal', 2: 'Risk-On'}[regime_id],
            'avg_return': avg_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'n_days': len(regime_data)
        })

    return pd.DataFrame(regime_stats)
```

**Why**: Critical for daily model health monitoring

---

## A3. Configuration Updates

### 6. `config/config.yaml` 🔧 **ADD SECTION**

**Add new `live` configuration section**:

```yaml
# Live Operations (Production Trading)
live:
  enabled: false  # Set to true when ready for production
  dry_run: true   # Default to dry-run (no real trades)

  # Model
  champion_model_path: "production/champion_model.pkl"
  fallback_model_path: "data/models/latest/model.pkl"

  # Portfolio
  initial_capital: 100000  # Starting capital in USD
  positions_file: "live/current_positions.csv"

  # Trade execution
  min_trade_size: 100      # Minimum trade in USD
  max_position_pct: 0.15   # Max 15% per position
  max_volume_pct: 0.05     # Trade max 5% of daily volume

  # Regime adaptation
  enable_regime_adaptation: true

  # Outputs
  output_dir: "live"
  save_trade_history: true

  # Safety checks
  safety:
    validate_before_trade: true
    require_manual_approval: true  # Human must approve trades
    max_daily_trades: 50

  # Notifications
  notifications:
    enabled: false
    method: "email"  # email, slack, webhook
    email_recipients: []
    send_on_kill_switch: true
    send_daily_summary: false

# Operations Monitoring
ops:
  # Performance tracking
  track_ic: true
  ic_lookback_days: 20
  ic_alert_threshold: 0.02

  # Kill-switch
  kill_switch:
    enabled: true
    max_daily_loss_pct: 0.03      # Halt if lose > 3% in one day
    min_live_sharpe_threshold: 0.5
    min_live_sharpe_lookback_weeks: 6

  # Alerts
  alerts: "email"  # email, slack, none
  email_recipients: []

  # Reporting
  generate_daily_report: true
  report_output_dir: "live"

# Risk Management
risk:
  max_sector_weight: 0.30    # Max 30% in any sector
  max_market_cap_weight:
    large_cap: 0.70          # Min 30% in large cap
    mid_cap: 0.25            # Max 25% in mid cap
    small_cap: 0.05          # Max 5% in small cap

  rebalance_threshold: 0.05  # Rebalance if drift > 5%
```

**Why**: Centralized configuration for live operations

---

## A4. Documentation

### 7. `docs/DAILY_LIVE_OPERATIONS.md` 📄 **NEW**

**Content**:
- Daily workflow overview
- How to run daily_live_runner.py
- Understanding the outputs
- Safety checks explained
- Troubleshooting guide
- Emergency procedures (kill-switch triggered)

### 8. `docs/TRADE_EXECUTION_GUIDE.md` 📄 **NEW**

**Content**:
- How to read trades.csv
- Manual trade approval process
- Broker integration guide (for future)
- Position reconciliation
- Cost tracking

---

## A5. Testing Files

### 9. `tests/test_daily_live_runner.py` 🧪 **NEW**

**Test coverage**:
```python
def test_market_open_check():
    # Skip weekends/holidays

def test_load_champion_model():
    # Load from production folder

def test_generate_signals():
    # Score universe with model

def test_construct_portfolio_with_regime():
    # Regime-aware portfolio construction

def test_generate_trades():
    # Calculate buy/sell orders

def test_safety_checks():
    # Validate constraints

def test_output_generation():
    # CSV + HTML + JSON

def test_dry_run_mode():
    # No actual trades executed
```

### 10. `tests/test_trade_generator.py` 🧪 **NEW**

**Test coverage**:
```python
def test_calculate_trade_sizes():
    # Shares to buy/sell

def test_cost_estimation():
    # Commission + slippage

def test_turnover_calculation():
    # Portfolio turnover %

def test_liquidity_filtering():
    # Flag illiquid trades
```

### 11. `tests/test_safety_checks.py` 🧪 **NEW**

**Test coverage**:
```python
def test_portfolio_constraints():
    # Weights sum to 1, max position, etc.

def test_kill_switch_trigger():
    # Daily loss threshold

def test_sector_limits():
    # Max sector exposure

def test_liquidity_validation():
    # Volume checks
```

---

## A6. Summary: Option A Files

| File | Type | Lines | Purpose |
|------|------|-------|---------|
| `scripts/daily_live_runner.py` | NEW | ~500 | Main daily workflow orchestrator |
| `src/live/trade_generator.py` | NEW | ~200 | Generate trade orders |
| `src/live/safety_checks.py` | NEW | ~250 | Validate portfolio & trades |
| `src/live/operations.py` | MODIFY | +100 | Add model loading, position management |
| `src/live/monitoring.py` | MODIFY | +200 | Add IC, drift detection, regime perf |
| `config/config.yaml` | MODIFY | +80 | Add live & ops config sections |
| `docs/DAILY_LIVE_OPERATIONS.md` | NEW | ~400 | User guide |
| `docs/TRADE_EXECUTION_GUIDE.md` | NEW | ~300 | Trade execution manual |
| `tests/test_daily_live_runner.py` | NEW | ~150 | Test main runner |
| `tests/test_trade_generator.py` | NEW | ~100 | Test trade generation |
| `tests/test_safety_checks.py` | NEW | ~100 | Test validations |

**Total**: ~2,480 lines of code

---

# Option B: Model Comparison Framework

**Goal**: Train multiple models, compare them objectively, select champion

**Timeline**: 3-5 days implementation + 2 days testing

---

## B1. New Files to Create

### 1. `scripts/train_model_candidates.py` ⭐ **NEW**

**Purpose**: Train multiple ML models with different algorithms/hyperparameters

**Workflow**:
```
1. Load training data (features + labels)
2. Define candidate models:
   - XGBoost (default params)
   - XGBoost (Optuna tuned)
   - Random Forest
   - LightGBM
   - Ensemble (blend top 3)
3. For each candidate:
   - Train on training set
   - Validate with time-series CV
   - Track metrics: IC, ICIR, Sharpe (on returns from scores)
   - Save model to models/candidates/
4. Generate comparison report
```

**Key Functions**:
```python
def define_candidates(config) -> List[Dict]:
    """Define list of model candidates"""
    return [
        {'name': 'xgboost_default', 'algo': 'xgboost', 'params': {}},
        {'name': 'xgboost_tuned', 'algo': 'xgboost', 'params': 'optuna'},
        {'name': 'random_forest', 'algo': 'random_forest', 'params': {}},
        {'name': 'lightgbm', 'algo': 'lightgbm', 'params': {}},
    ]

def train_candidate(candidate: Dict, X_train, y_train, X_val, y_val) -> Dict:
    """Train single model candidate"""
    from src.ml.train import ModelTrainer

    config = create_config_for_candidate(candidate)
    trainer = ModelTrainer(config)
    model = trainer.train_model(X_train, y_train)

    # Validate
    metrics = trainer.evaluate_model(model, X_val, y_val)

    # Save
    model_path = f"models/candidates/{candidate['name']}.pkl"
    joblib.dump(model, model_path)

    return {
        'name': candidate['name'],
        'model_path': model_path,
        'metrics': metrics
    }

def run_time_series_cv(candidate: Dict, dataset: MLDataset, n_folds: int = 5):
    """Walk-forward cross-validation"""
    from src.ml.dataset import create_time_based_split

    fold_results = []
    for fold_idx in range(n_folds):
        train_data, val_data = create_time_based_split(dataset, fold_idx, n_folds)
        result = train_candidate(candidate, train_data, val_data)
        fold_results.append(result)

    return fold_results
```

**Outputs**:
- `models/candidates/xgboost_default.pkl`
- `models/candidates/xgboost_tuned.pkl`
- `models/candidates/random_forest.pkl`
- `models/candidates/lightgbm.pkl`
- `reports/model_training_comparison.csv` - CV metrics for all candidates

---

### 2. `scripts/compare_backtest_results.py` ⭐ **NEW**

**Purpose**: Backtest all model×optimizer combinations

**Workflow**:
```
1. Load all candidate models
2. Define optimizer variants:
   - PyPortfolioOpt (max_sharpe)
   - PyPortfolioOpt (min_volatility)
   - HRP
   - Inverse Volatility
   - Equal Weight (baseline)
3. For each model × optimizer:
   - Generate scores
   - Construct portfolio (with regime adaptation)
   - Run backtest
   - Calculate metrics
   - Save results
4. Generate comparison matrix (models as rows, optimizers as columns)
```

**Key Functions**:
```python
def load_candidate_models(models_dir: str = "models/candidates") -> Dict[str, Any]:
    """Load all trained candidate models"""
    models = {}
    for model_file in Path(models_dir).glob("*.pkl"):
        model_name = model_file.stem
        models[model_name] = joblib.load(model_file)
    return models

def run_backtest_for_combination(
    model,
    model_name: str,
    optimizer: str,
    config: Dict,
    features: pd.DataFrame,
    prices: pd.DataFrame
) -> Dict:
    """
    Backtest single model+optimizer combination

    Returns:
        {
            'model': model_name,
            'optimizer': optimizer,
            'cagr': float,
            'sharpe': float,
            'max_drawdown': float,
            'calmar': float,
            'turnover': float,
            'win_rate': float,
            'total_return': float,
            'volatility': float,
            'regime_performance': {...}
        }
    """
    from src.backtest.bt_engine import VectorizedBacktester
    from src.portfolio.construct import construct_portfolio

    # Generate scores
    scores = model.predict(features)

    # Configure optimizer
    config['portfolio']['optimizer'] = optimizer

    # Construct portfolio (with regime)
    weights = construct_portfolio(scores, prices, config, enable_regime_adaptation=True)

    # Run backtest
    backtester = VectorizedBacktester(config)
    results = backtester.run(weights, prices)

    return results

def create_comparison_matrix(all_results: List[Dict]) -> pd.DataFrame:
    """
    Create comparison matrix

    Rows: Models
    Columns: Optimizers
    Cells: Key metric (e.g., Sharpe Ratio)
    """
    df = pd.DataFrame(all_results)
    pivot = df.pivot_table(
        index='model',
        columns='optimizer',
        values='sharpe',
        aggfunc='mean'
    )
    return pivot
```

**Outputs**:
- `reports/backtest_comparison_matrix.csv` - Models × Optimizers grid
- `reports/backtest_detailed_results.json` - Full metrics for all combinations
- `reports/equity_curves.png` - Visual comparison of top 5
- `reports/regime_performance_breakdown.csv` - Performance in each regime

---

### 3. `scripts/select_champion_model.py` ⭐ **NEW**

**Purpose**: Select best model using weighted scoring

**Workflow**:
```
1. Load comparison results
2. Define selection criteria:
   - Sharpe Ratio (30%)
   - Max Drawdown (25%)
   - Calmar Ratio (20%)
   - Turnover (15%)
   - Regime Consistency (10%)
3. Calculate weighted score for each combination
4. Filter by thresholds (Sharpe > 1.0, MaxDD < -20%, etc.)
5. Rank remaining candidates
6. Select champion (highest score)
7. Save champion to production/
8. Generate selection report
```

**Key Functions**:
```python
def calculate_weighted_score(result: Dict, weights: Dict) -> float:
    """
    Calculate composite score

    Args:
        result: Backtest results dict
        weights: Criterion weights (must sum to 1.0)

    Returns:
        Weighted score (0-1 scale)
    """
    # Normalize metrics to 0-1 scale
    sharpe_norm = normalize_sharpe(result['sharpe'], min_val=0, max_val=2.0)
    dd_norm = normalize_drawdown(result['max_drawdown'], min_val=-0.30, max_val=0.0)
    calmar_norm = normalize_calmar(result['calmar'], min_val=0, max_val=1.5)
    turnover_norm = normalize_turnover(result['turnover'], min_val=0, max_val=0.50)
    regime_norm = calculate_regime_consistency(result['regime_performance'])

    score = (
        weights['sharpe'] * sharpe_norm +
        weights['max_drawdown'] * dd_norm +
        weights['calmar'] * calmar_norm +
        weights['turnover'] * turnover_norm +
        weights['regime_consistency'] * regime_norm
    )

    return score

def filter_by_thresholds(results: pd.DataFrame, thresholds: Dict) -> pd.DataFrame:
    """
    Filter candidates that don't meet minimum thresholds

    Thresholds:
        - sharpe_min: 1.0
        - max_drawdown_max: -0.20 (must be better than -20%)
        - calmar_min: 0.5
        - turnover_max: 0.40
    """
    filtered = results[
        (results['sharpe'] >= thresholds['sharpe_min']) &
        (results['max_drawdown'] >= thresholds['max_drawdown_max']) &
        (results['calmar'] >= thresholds['calmar_min']) &
        (results['turnover'] <= thresholds['turnover_max'])
    ]
    return filtered

def select_champion(
    results: pd.DataFrame,
    weights: Dict,
    thresholds: Dict
) -> Dict:
    """
    Select champion model

    Returns:
        {
            'model': model_name,
            'optimizer': optimizer_name,
            'score': weighted_score,
            'metrics': {...},
            'reason': selection_justification
        }
    """
    # Filter by thresholds
    candidates = filter_by_thresholds(results, thresholds)

    if len(candidates) == 0:
        raise ValueError("No candidates meet minimum thresholds!")

    # Calculate scores
    candidates['weighted_score'] = candidates.apply(
        lambda row: calculate_weighted_score(row, weights),
        axis=1
    )

    # Select top
    champion = candidates.nlargest(1, 'weighted_score').iloc[0]

    return champion.to_dict()

def save_champion_to_production(champion: Dict):
    """Copy champion model to production folder"""
    src = Path(champion['model_path'])
    dst = Path("production/champion_model.pkl")
    dst.parent.mkdir(exist_ok=True)

    shutil.copy(src, dst)

    # Save metadata
    metadata = {
        'model_name': champion['model'],
        'optimizer': champion['optimizer'],
        'weighted_score': champion['score'],
        'metrics': champion['metrics'],
        'selection_date': datetime.now().isoformat(),
        'selection_reason': champion['reason']
    }

    with open("production/champion_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Champion model saved to {dst}")
```

**Outputs**:
- `production/champion_model.pkl` - Selected model
- `production/champion_metadata.json` - Selection details
- `reports/model_selection_report.md` - Decision justification
- `reports/champion_vs_benchmark.png` - Performance comparison

---

## B2. Files to Modify

### 4. `src/ml/train.py` 🔧 **ENHANCE**

**Add support for LightGBM and ensemble methods**:

```python
def train_model(self, X_train, y_train, X_val=None, y_val=None):
    """Enhanced to support multiple algorithms"""

    algo = self.config['modeling'].get('algorithm', 'xgboost')

    if algo == 'xgboost':
        model = self._train_xgboost(X_train, y_train, X_val, y_val)
    elif algo == 'random_forest':
        model = self._train_random_forest(X_train, y_train)
    elif algo == 'lightgbm':
        model = self._train_lightgbm(X_train, y_train, X_val, y_val)
    elif algo == 'ensemble':
        model = self._train_ensemble(X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    return model

def _train_lightgbm(self, X_train, y_train, X_val, y_val):
    """Train LightGBM model"""
    import lightgbm as lgb

    params = self.config['modeling'].get('lightgbm_params', {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8
    })

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50)]
    )

    return model

def _train_ensemble(self, X_train, y_train, X_val, y_val):
    """Train ensemble of models"""
    from sklearn.ensemble import VotingRegressor

    # Train base models
    xgb_model = self._train_xgboost(X_train, y_train, X_val, y_val)
    rf_model = self._train_random_forest(X_train, y_train)
    lgb_model = self._train_lightgbm(X_train, y_train, X_val, y_val)

    # Ensemble with equal weights
    ensemble = VotingRegressor(
        estimators=[
            ('xgboost', xgb_model),
            ('random_forest', rf_model),
            ('lightgbm', lgb_model)
        ],
        weights=[1, 1, 1]
    )

    return ensemble
```

---

### 5. `src/ml/optuna_search.py` 🔧 **ENHANCE**

**Add hyperparameter tuning support**:

```python
def tune_xgboost_params(X_train, y_train, X_val, y_val, n_trials: int = 100):
    """
    Use Optuna to find optimal XGBoost hyperparameters

    Returns:
        Best parameters dict
    """
    import optuna

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        }

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)

        return mse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
```

---

## B3. Configuration Updates

### 6. `config/config.yaml` 🔧 **ADD SECTION**

**Add model selection configuration**:

```yaml
# Model Selection (Phase 3)
model_selection:
  # Candidates to train
  candidates:
    - name: "xgboost_default"
      algorithm: "xgboost"
      tune_hyperparams: false

    - name: "xgboost_tuned"
      algorithm: "xgboost"
      tune_hyperparams: true
      optuna_trials: 100

    - name: "random_forest"
      algorithm: "random_forest"

    - name: "lightgbm"
      algorithm: "lightgbm"

    - name: "ensemble"
      algorithm: "ensemble"
      base_models: ["xgboost", "random_forest", "lightgbm"]

  # Optimizer variants to test
  optimizers:
    - "pypfopt"       # Mean-Variance
    - "hrp"           # Hierarchical Risk Parity
    - "inverse_vol"   # Inverse Volatility
    - "equal_weight"  # Baseline

  # Selection criteria weights (must sum to 1.0)
  selection_weights:
    sharpe: 0.30
    max_drawdown: 0.25
    calmar: 0.20
    turnover: 0.15
    regime_consistency: 0.10

  # Minimum thresholds (candidates must pass ALL)
  thresholds:
    sharpe_min: 1.0
    max_drawdown_max: -0.20  # Must be better than -20%
    calmar_min: 0.5
    turnover_max: 0.40       # Max 40% turnover

  # Output paths
  candidates_dir: "models/candidates"
  production_dir: "production"
  reports_dir: "reports/model_selection"

# LightGBM parameters (if using LightGBM)
modeling:
  lightgbm_params:
    objective: "regression"
    metric: "rmse"
    boosting_type: "gbdt"
    num_leaves: 31
    learning_rate: 0.05
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    verbose: -1
```

---

## B4. Documentation

### 7. `docs/MODEL_SELECTION_GUIDE.md` 📄 **NEW**

**Content**:
- Model comparison workflow
- Selection criteria explanation
- How to interpret backtest results
- Adding new model candidates
- Tuning selection weights

### 8. `docs/HYPERPARAMETER_TUNING.md` 📄 **NEW**

**Content**:
- Optuna integration guide
- Hyperparameter search spaces
- Avoiding overfitting during tuning
- Cross-validation best practices

---

## B5. Testing Files

### 9. `tests/test_model_training.py` 🧪 **NEW**

**Test coverage**:
```python
def test_train_xgboost():
def test_train_random_forest():
def test_train_lightgbm():
def test_train_ensemble():
def test_optuna_tuning():
```

### 10. `tests/test_model_comparison.py` 🧪 **NEW**

**Test coverage**:
```python
def test_backtest_all_combinations():
def test_comparison_matrix_generation():
def test_regime_performance_breakdown():
```

### 11. `tests/test_model_selection.py` 🧪 **NEW**

**Test coverage**:
```python
def test_weighted_scoring():
def test_threshold_filtering():
def test_champion_selection():
def test_save_to_production():
```

---

## B6. Summary: Option B Files

| File | Type | Lines | Purpose |
|------|------|-------|---------|
| `scripts/train_model_candidates.py` | NEW | ~400 | Train multiple models |
| `scripts/compare_backtest_results.py` | NEW | ~500 | Backtest all combinations |
| `scripts/select_champion_model.py` | NEW | ~350 | Select best model |
| `src/ml/train.py` | MODIFY | +150 | Add LightGBM, ensemble |
| `src/ml/optuna_search.py` | MODIFY | +100 | Hyperparameter tuning |
| `config/config.yaml` | MODIFY | +60 | Model selection config |
| `docs/MODEL_SELECTION_GUIDE.md` | NEW | ~500 | User guide |
| `docs/HYPERPARAMETER_TUNING.md` | NEW | ~300 | Tuning guide |
| `tests/test_model_training.py` | NEW | ~120 | Test training |
| `tests/test_model_comparison.py` | NEW | ~100 | Test comparison |
| `tests/test_model_selection.py` | NEW | ~100 | Test selection |

**Total**: ~2,680 lines of code

---

# Implementation Order

## Week 1: Option A (Daily Live Runner)

### Day 1-2: Core Implementation
```
✅ Create src/live/trade_generator.py
✅ Create src/live/safety_checks.py
✅ Enhance src/live/operations.py
✅ Enhance src/live/monitoring.py (IC, drift, regime perf)
```

### Day 3: Main Script
```
✅ Create scripts/daily_live_runner.py
✅ Add config/config.yaml sections (live, ops, risk)
```

### Day 4: Testing & Documentation
```
✅ Create all test files
✅ Run tests
✅ Create docs/DAILY_LIVE_OPERATIONS.md
✅ Create docs/TRADE_EXECUTION_GUIDE.md
```

### Day 5: Validation
```
✅ Run daily_live_runner.py in dry-run mode
✅ Validate outputs (CSV, HTML, JSON)
✅ Test email notifications (if enabled)
✅ Verify kill-switch triggers correctly
```

---

## Week 2: Option B (Model Comparison)

### Day 1-2: Model Training
```
✅ Enhance src/ml/train.py (LightGBM, ensemble)
✅ Enhance src/ml/optuna_search.py (tuning)
✅ Create scripts/train_model_candidates.py
✅ Test training all candidates
```

### Day 3: Backtesting
```
✅ Create scripts/compare_backtest_results.py
✅ Run backtests for all model×optimizer combinations
✅ Generate comparison matrix
```

### Day 4: Model Selection
```
✅ Create scripts/select_champion_model.py
✅ Implement weighted scoring
✅ Test champion selection
✅ Validate production folder structure
```

### Day 5: Testing & Documentation
```
✅ Create all test files
✅ Run tests
✅ Create docs/MODEL_SELECTION_GUIDE.md
✅ Create docs/HYPERPARAMETER_TUNING.md
```

---

# Testing Strategy

## Unit Tests (Per Component)
```bash
# Option A
pytest tests/test_trade_generator.py -v
pytest tests/test_safety_checks.py -v
pytest tests/test_daily_live_runner.py -v

# Option B
pytest tests/test_model_training.py -v
pytest tests/test_model_comparison.py -v
pytest tests/test_model_selection.py -v
```

## Integration Tests (End-to-End)
```bash
# Option A: Full daily workflow
python scripts/daily_live_runner.py --dry-run --date 2025-10-29

# Option B: Full model selection workflow
python scripts/train_model_candidates.py
python scripts/compare_backtest_results.py
python scripts/select_champion_model.py
```

## Validation Tests (Real Data)
```bash
# Option A: Run for 5 consecutive days
for i in {1..5}; do
    python scripts/daily_live_runner.py --dry-run
    sleep 1d
done

# Option B: Compare with existing model
python scripts/compare_backtest_results.py --baseline data/models/latest/model.pkl
```

---

# Success Criteria

## Option A Success Criteria ✅

1. **Functionality**:
   - ✅ Script runs without errors
   - ✅ Generates all outputs (CSV, HTML, JSON)
   - ✅ Regime detection works correctly
   - ✅ Trade recommendations are valid (positive shares for BUY, negative for SELL)
   - ✅ Safety checks catch violations

2. **Performance**:
   - ✅ Completes in < 5 minutes (for 500-stock universe)
   - ✅ Memory usage < 4GB
   - ✅ No memory leaks

3. **Quality**:
   - ✅ All unit tests pass (>90% coverage)
   - ✅ HTML report renders correctly
   - ✅ Emails send successfully (if enabled)
   - ✅ Kill-switch triggers when it should

4. **Validation**:
   - ✅ Portfolio weights sum to ~1.0 (within 1%)
   - ✅ All positions within limits (max 15%)
   - ✅ Turnover calculated correctly
   - ✅ Cost estimation matches backtest

---

## Option B Success Criteria ✅

1. **Functionality**:
   - ✅ All candidate models train successfully
   - ✅ Backtests run for all combinations
   - ✅ Champion model selected correctly
   - ✅ Production folder populated

2. **Performance**:
   - ✅ Training completes in reasonable time (< 2 hours for all candidates)
   - ✅ Backtesting completes in < 30 minutes
   - ✅ Selection script runs in < 1 minute

3. **Quality**:
   - ✅ All unit tests pass
   - ✅ Comparison matrix is complete (no NaNs)
   - ✅ Selection report is comprehensive
   - ✅ Champion metadata saved correctly

4. **Validation**:
   - ✅ Champion model beats baseline (equal-weight)
   - ✅ Champion Sharpe > 1.0
   - ✅ Champion max drawdown < -20%
   - ✅ Weighted scoring formula is correct

---

# Expected Outputs

## Option A Outputs (Daily)

```
live/
├── 2025-10-30/
│   ├── trades.csv                 # Buy/sell recommendations
│   ├── portfolio_weights.csv      # Target allocation
│   ├── signals.json               # Raw ML scores
│   ├── report.html                # Performance dashboard
│   └── monitoring_log.json        # Model health metrics
├── current_positions.csv          # Updated after trades
└── performance_history.csv        # Daily tracking
```

## Option B Outputs (Once)

```
models/
├── candidates/
│   ├── xgboost_default.pkl
│   ├── xgboost_tuned.pkl
│   ├── random_forest.pkl
│   ├── lightgbm.pkl
│   └── ensemble.pkl
│
production/
├── champion_model.pkl             # Selected best model
└── champion_metadata.json         # Selection details

reports/
└── model_selection/
    ├── training_comparison.csv
    ├── backtest_comparison_matrix.csv
    ├── backtest_detailed_results.json
    ├── model_selection_report.md
    ├── equity_curves.png
    └── regime_performance_breakdown.csv
```

---

# Risk Mitigation

## Potential Issues & Solutions

| Risk | Mitigation |
|------|------------|
| **Kill-switch false positives** | Test with historical data; tune thresholds |
| **Model file not found** | Fallback to latest trained model; alert user |
| **Data missing/corrupted** | Validate data before scoring; fail-safe |
| **Portfolio constraints violated** | Strict validation; halt if invalid |
| **Email delivery fails** | Log locally; continue execution |
| **Long runtime** | Optimize feature computation; parallelize if needed |
| **Memory issues** | Process in batches; clear cache |
| **Backtest discrepancy** | Ensure identical cost model; same regime settings |

---

# Next Steps

1. **Review this plan** ✓
2. **Approve to proceed** (your decision)
3. **Start Option A implementation** (I'll begin coding)
4. **Test & validate**
5. **Deploy for dry-run testing**
6. **Proceed to Option B**

---

**Ready to begin? Should I start implementing Option A now?**
