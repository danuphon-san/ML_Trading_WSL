"""
Vectorized backtest engine with realistic costs
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger

from src.portfolio.risk import calculate_portfolio_metrics, calculate_turnover


class VectorizedBacktester:
    """Vectorized backtesting engine"""

    def __init__(self, config: Dict):
        """
        Initialize backtester

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backtest_config = config.get('backtest', {})
        self.portfolio_config = config.get('portfolio', {})

        self.initial_capital = self.backtest_config.get('initial_capital', 100000)
        self.costs_bps = self.portfolio_config.get('costs_bps', 1.0)
        self.slippage_bps = self.portfolio_config.get('slippage_bps', 2.0)
        self.apply_costs = self.backtest_config.get('apply_costs', True)

        # Execution timing (to avoid look-ahead bias)
        # Options: 'close' (same day close) or 'next_open' (next day open - more realistic)
        self.execution_timing = self.backtest_config.get('execution_timing', 'close')

        logger.info(f"Initialized VectorizedBacktester: capital=${self.initial_capital:,.0f}, "
                   f"costs={self.costs_bps}bps, slippage={self.slippage_bps}bps, "
                   f"execution={self.execution_timing}")

        if self.execution_timing == 'close':
            logger.warning(
                "Using 'close' execution timing may introduce look-ahead bias "
                "if signals are generated using same-day data. "
                "Consider using 'next_open' for more realistic backtests."
            )

        self.equity_curve = []
        self.trades = []
        self.positions = []
        self._close_price_matrix: Optional[pd.DataFrame] = None
        self._open_price_matrix: Optional[pd.DataFrame] = None

    def run(
        self,
        weights_history: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> Dict:
        """
        Run backtest with proper execution timing and cash handling.

        Args:
            weights_history: DataFrame with columns [date, symbol, weight]
            price_data: DataFrame with price history. Requires 'close' and, when
                execution_timing == 'next_open', also 'open'.

        Returns:
            Dictionary with backtest results
        """
        logger.info("Running vectorized backtest")

        # Reset run state
        self.equity_curve = []
        self.trades = []
        self.positions = []

        if weights_history.empty:
            logger.warning("Weights history is empty; returning empty results")
            self.equity_curve = pd.DataFrame(columns=['date', 'equity'])
            return self._calculate_results(empty_ok=True)

        weights_history = weights_history.copy()
        price_data = price_data.copy()

        weights_history['date'] = pd.to_datetime(weights_history['date'])
        price_data['date'] = pd.to_datetime(price_data['date'])

        weights_history = weights_history.sort_values('date')
        price_data = price_data.sort_values('date')

        # Build price matrices for fast lookup
        self._close_price_matrix = price_data.pivot_table(
            index='date', columns='symbol', values='close', aggfunc='last'
        ).sort_index()

        if self._close_price_matrix.empty:
            raise ValueError("Price data must contain 'close' prices for backtesting")

        if self.execution_timing == 'next_open':
            if 'open' not in price_data.columns:
                raise ValueError(
                    "Price data must contain 'open' when execution_timing is 'next_open'"
                )
            self._open_price_matrix = price_data.pivot_table(
                index='date', columns='symbol', values='open', aggfunc='first'
            ).sort_index()
        else:
            self._open_price_matrix = None

        price_calendar = self._close_price_matrix.index
        weight_groups = {date: df for date, df in weights_history.groupby('date')}
        rebalance_dates = sorted(weight_groups.keys())

        capital = float(self.initial_capital)
        cash = float(self.initial_capital)
        current_positions: Dict[str, float] = {}
        equity_records: List[Dict[str, float]] = []

        for signal_date in rebalance_dates:
            execution_date = self._determine_execution_date(signal_date, price_calendar)

            if self.execution_timing == 'next_open':
                trade_prices_series = self._get_price_series(self._open_price_matrix, execution_date)
                valuation_date = execution_date
            else:
                trade_prices_series = self._get_price_series(self._close_price_matrix, execution_date)
                valuation_date = execution_date

            trade_prices = trade_prices_series.dropna().to_dict()
            if not trade_prices:
                raise ValueError(f"No trade prices available for execution date {execution_date}")

            # Mark-to-market existing holdings at execution price to capture gap moves
            capital = self._mark_to_market(current_positions, cash, trade_prices)

            target_weights = weight_groups[signal_date].set_index('symbol')['weight'].to_dict()

            trades = self._calculate_trades(current_positions, target_weights, capital, trade_prices)

            if self.apply_costs and trades:
                cost = self._calculate_trading_costs(trades, trade_prices)
                cash -= cost
                capital -= cost
            else:
                cost = 0.0

            current_positions, cash = self._execute_trades(current_positions, trades, trade_prices, cash)

            if cash < -1e-6:
                logger.warning(
                    f"Cash balance negative after trades on {execution_date}: ${cash:,.2f}"
                )

            if trades:
                trade_date = execution_date
                for trade in trades:
                    trade['date'] = trade_date
            self.trades.extend(trades)

            close_prices_series = self._get_price_series(self._close_price_matrix, valuation_date)
            close_prices = close_prices_series.dropna().to_dict()
            invested_value = self._value_positions(current_positions, close_prices, valuation_date)
            capital = invested_value + cash

            equity_records.append({'date': valuation_date, 'equity': capital})
            self.positions.append({
                'date': valuation_date,
                'positions': current_positions.copy(),
                'capital': capital,
                'cash': cash
            })

        self.equity_curve = (
            pd.DataFrame(equity_records)
            .sort_values('date')
            .drop_duplicates(subset='date', keep='last')
            .reset_index(drop=True)
        )

        results = self._calculate_results()

        logger.info(
            f"Backtest complete: Return={results['metrics']['total_return']:.2%}, "
            f"Sharpe={results['metrics']['sharpe_ratio']:.2f}"
        )

        return results

    def _calculate_trades(
        self,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        capital: float,
        prices: Dict[str, float]
    ) -> List[Dict]:
        """Calculate required trades to reach target weights"""
        trades = []

        all_symbols = set(current_positions.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current_shares = current_positions.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)

            if symbol not in prices:
                continue

            price = prices[symbol]

            # Calculate target shares
            target_value = capital * target_weight
            target_shares = target_value / price if price > 0 else 0

            # Trade size
            trade_shares = target_shares - current_shares

            if abs(trade_shares) > 0.01:  # Minimum trade threshold
                trades.append({
                    'symbol': symbol,
                    'shares': trade_shares,
                    'price': price
                })

        return trades

    def _calculate_trading_costs(
        self,
        trades: List[Dict],
        prices: Dict[str, float]
    ) -> float:
        """
        Calculate trading costs (commission + slippage)

        Args:
            trades: List of trades
            prices: Current prices

        Returns:
            Total cost
        """
        total_cost = 0

        for trade in trades:
            notional = abs(trade['shares']) * trade['price']

            # Commission
            commission = notional * (self.costs_bps / 10000)

            # Slippage (unfavorable execution)
            slippage = notional * (self.slippage_bps / 10000)

            total_cost += commission + slippage

        return total_cost

    def _determine_execution_date(
        self,
        signal_date: pd.Timestamp,
        calendar: pd.Index
    ) -> pd.Timestamp:
        """Resolve the actual execution date based on configuration."""
        if self.execution_timing == 'close':
            if signal_date in calendar:
                return signal_date
            pos = calendar.searchsorted(signal_date, side='left')
            if pos >= len(calendar):
                raise ValueError(f"No price data on or after {signal_date}")
            fallback = calendar[pos]
            logger.warning(
                f"No close price for {signal_date}; using next available date {fallback}"
            )
            return fallback

        # next_open: use the next available trading day
        pos = calendar.searchsorted(signal_date, side='right')
        if pos >= len(calendar):
            if signal_date in calendar:
                logger.warning(
                    f"No future open available after {signal_date}; executing at same-day close"
                )
                return signal_date
            raise ValueError(f"No open price available after {signal_date}")

        return calendar[pos]

    def _get_price_series(
        self,
        price_matrix: Optional[pd.DataFrame],
        date: pd.Timestamp
    ) -> pd.Series:
        """Fetch a row of prices for the given date, returning an empty series if missing."""
        if price_matrix is None or date not in price_matrix.index:
            return pd.Series(dtype=float)
        return price_matrix.loc[date].astype(float)

    def _mark_to_market(
        self,
        positions: Dict[str, float],
        cash: float,
        prices: Dict[str, float]
    ) -> float:
        """Return portfolio equity given current positions, cash, and prices."""
        equity = cash
        missing_symbols = []

        for symbol, shares in positions.items():
            price = prices.get(symbol)
            if price is None or pd.isna(price):
                missing_symbols.append(symbol)
                continue
            equity += shares * price

        if missing_symbols:
            logger.warning(
                f"Missing execution prices for symbols {missing_symbols}; treating as zero value"
            )

        return float(equity)

    def _value_positions(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        date: pd.Timestamp
    ) -> float:
        """Value the current holdings using the provided price dictionary."""
        total = 0.0
        missing = []

        for symbol, shares in positions.items():
            price = prices.get(symbol)
            if price is None or pd.isna(price):
                missing.append(symbol)
                continue
            total += shares * price

        if missing:
            logger.warning(
                f"Missing close prices for symbols {missing} on {date}; assuming zero"
            )

        return float(total)

    def _execute_trades(
        self,
        current_positions: Dict[str, float],
        trades: List[Dict],
        prices: Dict[str, float],
        cash: float
    ) -> Tuple[Dict[str, float], float]:
        """Execute trades, update positions, and return updated cash."""
        new_positions = current_positions.copy()
        new_cash = cash

        for trade in trades:
            symbol = trade['symbol']
            shares = trade['shares']
            price = prices.get(symbol)

            if price is None or pd.isna(price):
                logger.warning(f"Skipping trade for {symbol}: missing execution price")
                continue

            current = new_positions.get(symbol, 0.0)
            new_positions[symbol] = current + shares

            # Cash impact: positive shares consume cash, negative shares release cash
            new_cash -= shares * price

            if abs(new_positions[symbol]) < 1e-6:
                new_positions.pop(symbol, None)

        return new_positions, float(new_cash)

    def _calculate_results(self, empty_ok: bool = False) -> Dict:
        """Calculate backtest results"""
        equity_curve = self.equity_curve.copy()

        if equity_curve.empty:
            if empty_ok:
                metrics = {
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'calmar_ratio': 0.0,
                    'avg_turnover': 0.0
                }
                return {
                    'equity_curve': equity_curve,
                    'metrics': metrics,
                    'trades': pd.DataFrame(self.trades),
                    'final_equity': self.initial_capital
                }
            raise ValueError("Equity curve is empty; cannot compute results")

        equity_curve['returns'] = equity_curve['equity'].pct_change()

        returns = equity_curve['returns'].dropna()

        if returns.empty:
            metrics = {
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0
            }
        else:
            metrics = calculate_portfolio_metrics(returns)

        # Turnover
        if len(self.positions) > 1:
            weights_df = self._positions_to_weights()
            turnover = calculate_turnover(weights_df)
            avg_turnover = turnover.mean()
        else:
            avg_turnover = 0

        metrics['avg_turnover'] = avg_turnover

        results = {
            'equity_curve': equity_curve,
            'metrics': metrics,
            'trades': pd.DataFrame(self.trades),
            'final_equity': equity_curve['equity'].iloc[-1]
        }

        return results

    def _positions_to_weights(self) -> pd.DataFrame:
        """Convert positions history to weights DataFrame"""
        if not self.positions or self._close_price_matrix is None:
            return pd.DataFrame(columns=['date', 'symbol', 'weight'])

        records = []

        for pos_record in self.positions:
            date = pos_record['date']
            positions = pos_record['positions']
            capital = pos_record['capital']

            if capital <= 0:
                continue

            price_series = self._get_price_series(self._close_price_matrix, date)
            prices = price_series.dropna().to_dict()

            for symbol, shares in positions.items():
                price = prices.get(symbol)
                if price is None:
                    continue
                weight = (shares * price) / capital
                records.append({
                    'date': date,
                    'symbol': symbol,
                    'weight': weight
                })

        return pd.DataFrame(records, columns=['date', 'symbol', 'weight'])
