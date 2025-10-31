"""
Trade Generation Module

Converts target portfolio weights into executable trade orders
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger


class TradeGenerator:
    """
    Generate trade orders from target portfolio weights

    Responsibilities:
    - Calculate shares to buy/sell from target weights
    - Estimate transaction costs (commission + slippage)
    - Validate liquidity
    - Filter small trades
    """

    def __init__(self, config: Dict):
        """
        Initialize trade generator

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Cost parameters
        portfolio_config = config.get('portfolio', {})
        self.costs_bps = portfolio_config.get('costs_bps', 1.0)
        self.slippage_bps = portfolio_config.get('slippage_bps', 2.0)

        # Trade parameters
        live_config = config.get('live', {})
        self.min_trade_size = live_config.get('min_trade_size', 100)  # Min $100 trades
        self.max_volume_pct = live_config.get('max_volume_pct', 0.05)  # Max 5% of daily volume

        logger.info(f"TradeGenerator initialized: costs={self.costs_bps}bps, slippage={self.slippage_bps}bps")

    def generate_trades(
        self,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float,
        volume_data: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate trade orders from current and target portfolios

        Args:
            current_positions: Current holdings {symbol: shares}
            target_weights: Target portfolio {symbol: weight}
            current_prices: Current market prices {symbol: price}
            portfolio_value: Total portfolio value in USD
            volume_data: Optional daily volumes {symbol: volume}

        Returns:
            DataFrame with columns:
            - symbol: Stock ticker
            - side: 'BUY' or 'SELL'
            - shares: Number of shares to trade (positive for buy, negative for sell)
            - price: Current market price
            - notional: Dollar value (abs(shares) × price)
            - weight_change: Target weight - Current weight
            - current_shares: Current position size
            - target_shares: Target position size
            - commission_est: Estimated commission cost
            - slippage_est: Estimated slippage cost
            - total_cost_est: Total transaction cost
        """
        logger.info(f"Generating trades: portfolio_value=${portfolio_value:,.0f}, "
                   f"current_positions={len(current_positions)}, target_weights={len(target_weights)}")

        # Get all symbols (union of current and target)
        all_symbols = set(current_positions.keys()) | set(target_weights.keys())

        trades_list = []

        for symbol in all_symbols:
            # Current state
            current_shares = current_positions.get(symbol, 0.0)
            current_price = current_prices.get(symbol, 0.0)

            if current_price <= 0:
                logger.warning(f"Skipping {symbol}: invalid price {current_price}")
                continue

            # Target state
            target_weight = target_weights.get(symbol, 0.0)
            target_value = portfolio_value * target_weight
            target_shares = target_value / current_price

            # Calculate trade
            trade_shares = target_shares - current_shares

            # Skip very small trades
            notional = abs(trade_shares) * current_price
            if notional < self.min_trade_size:
                continue

            # Determine side
            side = 'BUY' if trade_shares > 0 else 'SELL'

            # Calculate current weight
            current_value = current_shares * current_price
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0.0
            weight_change = target_weight - current_weight

            # Estimate costs
            commission = (notional * self.costs_bps) / 10000
            slippage = (notional * self.slippage_bps) / 10000
            total_cost = commission + slippage

            trade_dict = {
                'symbol': symbol,
                'side': side,
                'shares': trade_shares,
                'price': current_price,
                'notional': notional,
                'weight_change': weight_change,
                'current_shares': current_shares,
                'target_shares': target_shares,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'commission_est': commission,
                'slippage_est': slippage,
                'total_cost_est': total_cost
            }

            # Add volume data if available
            if volume_data and symbol in volume_data:
                trade_dict['daily_volume'] = volume_data[symbol]
                trade_dict['volume_pct'] = abs(trade_shares) / volume_data[symbol] if volume_data[symbol] > 0 else 0

            trades_list.append(trade_dict)

        # Create DataFrame
        if not trades_list:
            logger.info("No trades generated (all positions within tolerance)")
            return pd.DataFrame()

        trades_df = pd.DataFrame(trades_list)

        # Sort by notional (largest trades first)
        trades_df = trades_df.sort_values('notional', ascending=False)

        logger.info(f"Generated {len(trades_df)} trades: "
                   f"{len(trades_df[trades_df['side']=='BUY'])} BUY, "
                   f"{len(trades_df[trades_df['side']=='SELL'])} SELL")

        return trades_df

    def calculate_turnover(
        self,
        trades: pd.DataFrame,
        portfolio_value: float
    ) -> float:
        """
        Calculate portfolio turnover percentage

        Turnover = Sum(abs(notional)) / (2 * portfolio_value)

        Args:
            trades: Trade orders DataFrame
            portfolio_value: Total portfolio value

        Returns:
            Turnover as decimal (e.g., 0.25 = 25% turnover)
        """
        if len(trades) == 0 or portfolio_value <= 0:
            return 0.0

        total_traded = trades['notional'].sum()
        turnover = total_traded / (2 * portfolio_value)

        logger.info(f"Portfolio turnover: {turnover:.2%} (${total_traded:,.0f} traded)")

        return turnover

    def estimate_total_costs(
        self,
        trades: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Estimate total transaction costs

        Args:
            trades: Trade orders DataFrame

        Returns:
            Dict with:
            - total_commission: Total commission in USD
            - total_slippage: Total slippage in USD
            - total_cost: Total cost in USD
            - cost_bps: Total cost in basis points of traded notional
        """
        if len(trades) == 0:
            return {
                'total_commission': 0.0,
                'total_slippage': 0.0,
                'total_cost': 0.0,
                'cost_bps': 0.0
            }

        total_commission = trades['commission_est'].sum()
        total_slippage = trades['slippage_est'].sum()
        total_cost = trades['total_cost_est'].sum()

        total_notional = trades['notional'].sum()
        cost_bps = (total_cost / total_notional * 10000) if total_notional > 0 else 0.0

        costs = {
            'total_commission': float(total_commission),
            'total_slippage': float(total_slippage),
            'total_cost': float(total_cost),
            'cost_bps': float(cost_bps),
            'total_notional': float(total_notional)
        }

        logger.info(f"Estimated costs: ${total_cost:.2f} ({cost_bps:.1f} bps of ${total_notional:,.0f})")

        return costs

    def filter_small_trades(
        self,
        trades: pd.DataFrame,
        min_notional: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Remove trades below minimum size

        Args:
            trades: Trade orders DataFrame
            min_notional: Minimum trade size (default: from config)

        Returns:
            Filtered DataFrame
        """
        if len(trades) == 0:
            return trades

        if min_notional is None:
            min_notional = self.min_trade_size

        filtered = trades[trades['notional'] >= min_notional].copy()

        removed = len(trades) - len(filtered)
        if removed > 0:
            logger.info(f"Filtered out {removed} trades below ${min_notional:.0f}")

        return filtered

    def validate_liquidity(
        self,
        trades: pd.DataFrame,
        max_volume_pct: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Flag trades that may have liquidity issues

        Args:
            trades: Trade orders DataFrame with 'daily_volume' column
            max_volume_pct: Maximum % of daily volume (default: from config)

        Returns:
            DataFrame with 'liquidity_warning' column added
        """
        if len(trades) == 0:
            return trades

        if max_volume_pct is None:
            max_volume_pct = self.max_volume_pct

        # Check if volume data is available
        if 'daily_volume' not in trades.columns:
            logger.warning("No volume data available for liquidity validation")
            trades['liquidity_warning'] = False
            return trades

        # Calculate volume percentage
        if 'volume_pct' not in trades.columns:
            trades['volume_pct'] = (
                trades['shares'].abs() / trades['daily_volume']
            ).replace([np.inf, -np.inf], 0).fillna(0)

        # Flag illiquid trades
        trades['liquidity_warning'] = trades['volume_pct'] > max_volume_pct

        illiquid_count = trades['liquidity_warning'].sum()
        if illiquid_count > 0:
            logger.warning(f"⚠️  {illiquid_count} trades exceed {max_volume_pct:.1%} of daily volume:")
            for _, trade in trades[trades['liquidity_warning']].iterrows():
                logger.warning(f"  {trade['symbol']}: {trade['volume_pct']:.2%} of volume")

        return trades

    def round_shares(
        self,
        trades: pd.DataFrame,
        round_lots: bool = True
    ) -> pd.DataFrame:
        """
        Round shares to whole numbers or round lots

        Args:
            trades: Trade orders DataFrame
            round_lots: If True, round to nearest 100 shares

        Returns:
            DataFrame with rounded shares
        """
        if len(trades) == 0:
            return trades

        trades = trades.copy()

        if round_lots:
            # Round to nearest 100 shares (round lot)
            trades['shares_rounded'] = (trades['shares'] / 100).round() * 100
        else:
            # Round to whole shares
            trades['shares_rounded'] = trades['shares'].round()

        # Remove trades that round to zero
        trades = trades[trades['shares_rounded'] != 0].copy()

        # Recalculate notional with rounded shares
        trades['notional_rounded'] = abs(trades['shares_rounded']) * trades['price']

        logger.info(f"Rounded {len(trades)} trades to {'round lots (100s)' if round_lots else 'whole shares'}")

        return trades

    def create_execution_summary(
        self,
        trades: pd.DataFrame,
        portfolio_value: float
    ) -> Dict:
        """
        Create summary statistics for trade execution

        Args:
            trades: Trade orders DataFrame
            portfolio_value: Total portfolio value

        Returns:
            Summary dict
        """
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_notional': 0.0,
                'turnover': 0.0,
                'estimated_costs': 0.0,
                'cost_pct_portfolio': 0.0
            }

        costs = self.estimate_total_costs(trades)
        turnover = self.calculate_turnover(trades, portfolio_value)

        summary = {
            'total_trades': len(trades),
            'buy_trades': len(trades[trades['side'] == 'BUY']),
            'sell_trades': len(trades[trades['side'] == 'SELL']),
            'total_notional': float(trades['notional'].sum()),
            'turnover': float(turnover),
            'estimated_costs': float(costs['total_cost']),
            'cost_bps': float(costs['cost_bps']),
            'cost_pct_portfolio': float(costs['total_cost'] / portfolio_value * 100) if portfolio_value > 0 else 0.0,
            'largest_trade': {
                'symbol': trades.iloc[0]['symbol'],
                'side': trades.iloc[0]['side'],
                'notional': float(trades.iloc[0]['notional'])
            } if len(trades) > 0 else None
        }

        return summary
