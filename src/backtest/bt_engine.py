"""
Vectorized backtest engine with realistic costs
"""
import numpy as np
import pandas as pd
from typing import Dict, List
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

    def run(
        self,
        weights_history: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> Dict:
        """
        Run backtest

        Args:
            weights_history: DataFrame with [date, symbol, weight]
            price_data: DataFrame with [date, symbol, close]

        Returns:
            Dictionary with backtest results
        """
        logger.info("Running vectorized backtest")

        # Get unique rebalance dates
        rebalance_dates = sorted(weights_history['date'].unique())

        # Initialize
        capital = self.initial_capital
        current_positions = {}
        equity = [capital]
        dates = [rebalance_dates[0]]

        for i, rebal_date in enumerate(rebalance_dates):
            # Get target weights
            target_weights = weights_history[weights_history['date'] == rebal_date].set_index('symbol')['weight'].to_dict()

            # Get prices at rebalance
            prices = price_data[price_data['date'] == rebal_date].set_index('symbol')['close'].to_dict()

            # Calculate trades
            trades = self._calculate_trades(current_positions, target_weights, capital, prices)

            # Apply costs
            if self.apply_costs:
                cost = self._calculate_trading_costs(trades, prices)
                capital -= cost
            else:
                cost = 0

            # Update positions
            current_positions = self._execute_trades(current_positions, trades, prices)

            # Store
            self.trades.extend(trades)
            self.positions.append({
                'date': rebal_date,
                'positions': current_positions.copy(),
                'capital': capital
            })

            # Update capital for next period (if not last date)
            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]

                # Get returns between rebalances
                returns = self._get_returns_between_dates(
                    price_data,
                    current_positions,
                    rebal_date,
                    next_date
                )

                capital = capital * (1 + returns)

            equity.append(capital)
            dates.append(rebal_date)

        # Build equity curve
        self.equity_curve = pd.DataFrame({'date': dates, 'equity': equity})

        # Calculate metrics
        results = self._calculate_results()

        logger.info(f"Backtest complete: Return={results['metrics']['total_return']:.2%}, "
                   f"Sharpe={results['metrics']['sharpe_ratio']:.2f}")

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

    def _execute_trades(
        self,
        current_positions: Dict[str, float],
        trades: List[Dict],
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Execute trades and update positions"""
        new_positions = current_positions.copy()

        for trade in trades:
            symbol = trade['symbol']
            current = new_positions.get(symbol, 0)
            new_positions[symbol] = current + trade['shares']

            # Remove zero positions
            if abs(new_positions[symbol]) < 0.01:
                new_positions.pop(symbol, None)

        return new_positions

    def _get_returns_between_dates(
        self,
        price_data: pd.DataFrame,
        positions: Dict[str, float],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> float:
        """Calculate portfolio return between dates"""
        if not positions:
            return 0.0

        # Get prices at start and end
        start_prices = price_data[price_data['date'] == start_date].set_index('symbol')['close']
        end_prices = price_data[price_data['date'] == end_date].set_index('symbol')['close']

        # Calculate portfolio value change
        start_value = sum(positions[s] * start_prices.get(s, 0) for s in positions.keys())
        end_value = sum(positions[s] * end_prices.get(s, 0) for s in positions.keys())

        if start_value == 0:
            return 0.0

        return (end_value - start_value) / start_value

    def _calculate_results(self) -> Dict:
        """Calculate backtest results"""
        # Calculate returns
        equity_curve = self.equity_curve.copy()
        equity_curve['returns'] = equity_curve['equity'].pct_change()

        # Metrics
        returns = equity_curve['returns'].dropna()
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
        records = []

        for pos_record in self.positions:
            date = pos_record['date']
            positions = pos_record['positions']
            capital = pos_record['capital']

            # Assuming we have prices (simplified)
            for symbol, shares in positions.items():
                weight = shares / capital if capital > 0 else 0
                records.append({
                    'date': date,
                    'symbol': symbol,
                    'weight': weight
                })

        return pd.DataFrame(records)
