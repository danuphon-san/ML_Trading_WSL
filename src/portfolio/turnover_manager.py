"""
Turnover Manager (Step 14)

Enforce turnover caps, minimum trade thresholds, and cost-aware position adjustments
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger


class TurnoverManager:
    """
    Portfolio turnover management and cost optimization

    Functions:
    - Calculate portfolio turnover
    - Enforce turnover caps
    - Apply minimum trade thresholds
    - Round lot sizes
    - Generate turnover reports
    """

    def __init__(self, config: Dict):
        """
        Initialize turnover manager

        Args:
            config: Configuration dict
        """
        self.config = config

        # Turnover constraints
        self.turnover_cap_annual = config.get('portfolio', {}).get('turnover_cap_annual', 0.80)  # 80% annual
        self.max_turnover = config.get('portfolio', {}).get('max_turnover', 0.35)  # 35% per rebalance
        self.min_trade_threshold = config.get('portfolio', {}).get('min_trade_threshold', 0.005)  # 0.5%

        # Costs
        self.costs_bps = config.get('portfolio', {}).get('costs_bps', 1.0)
        self.slippage_bps = config.get('portfolio', {}).get('slippage_bps', 2.0)

        logger.info(f"Initialized TurnoverManager: cap={self.turnover_cap_annual:.2%} annual, "
                   f"max_per_rebalance={self.max_turnover:.2%}")

    def calculate_turnover(
        self,
        weights_prev: pd.DataFrame,
        weights_target: pd.DataFrame
    ) -> Tuple[float, pd.DataFrame]:
        """
        Calculate portfolio turnover

        Turnover = Σ |w_target - w_prev| / 2

        Args:
            weights_prev: Previous weights DataFrame [symbol, weight]
            weights_target: Target weights DataFrame [symbol, weight]

        Returns:
            (turnover_pct, trades_df)
        """
        # Merge previous and target weights
        merged = pd.merge(
            weights_prev[['symbol', 'weight']].rename(columns={'weight': 'weight_prev'}),
            weights_target[['symbol', 'weight']].rename(columns={'weight': 'weight_target'}),
            on='symbol',
            how='outer'
        ).fillna(0.0)

        # Calculate changes
        merged['weight_change'] = merged['weight_target'] - merged['weight_prev']
        merged['abs_weight_change'] = merged['weight_change'].abs()

        # Total turnover (sum of absolute changes / 2)
        turnover = merged['abs_weight_change'].sum() / 2.0

        logger.info(f"Portfolio turnover: {turnover:.2%}")

        return turnover, merged

    def enforce_turnover_cap(
        self,
        weights_prev: pd.DataFrame,
        weights_target: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Enforce maximum turnover cap by scaling down proposed changes

        If turnover > max_turnover:
            - Scale down all changes proportionally
            - Keep existing positions more stable

        Args:
            weights_prev: Previous weights
            weights_target: Target weights
            date: Rebalancing date

        Returns:
            Adjusted weights DataFrame
        """
        turnover, trades = self.calculate_turnover(weights_prev, weights_target)

        if turnover <= self.max_turnover:
            logger.info(f"Turnover {turnover:.2%} within limit {self.max_turnover:.2%}, no adjustment needed")
            weights_adjusted = weights_target.copy()
            weights_adjusted['date'] = date
            return weights_adjusted

        # Turnover exceeds cap, scale down changes
        scale_factor = self.max_turnover / turnover
        logger.info(f"Turnover {turnover:.2%} exceeds cap {self.max_turnover:.2%}, "
                   f"scaling changes by {scale_factor:.2f}")

        # Compute adjusted weights
        trades['weight_adjusted'] = (
            trades['weight_prev'] +
            trades['weight_change'] * scale_factor
        )

        # Ensure non-negative and normalize
        trades['weight_adjusted'] = trades['weight_adjusted'].clip(lower=0.0)
        total_weight = trades['weight_adjusted'].sum()

        if total_weight > 0:
            trades['weight_adjusted'] = trades['weight_adjusted'] / total_weight
        else:
            logger.warning("All adjusted weights are zero, using target weights")
            trades['weight_adjusted'] = trades['weight_target']

        # Build result DataFrame
        weights_adjusted = trades[trades['weight_adjusted'] > 0][['symbol', 'weight_adjusted']].copy()
        weights_adjusted.rename(columns={'weight_adjusted': 'weight'}, inplace=True)
        weights_adjusted['date'] = date

        logger.info(f"Adjusted turnover: {self.calculate_turnover(weights_prev, weights_adjusted)[0]:.2%}")

        return weights_adjusted

    def apply_min_trade_threshold(
        self,
        weights_prev: pd.DataFrame,
        weights_target: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Filter out trades below minimum threshold to reduce transaction costs

        Args:
            weights_prev: Previous weights
            weights_target: Target weights
            date: Rebalancing date

        Returns:
            Adjusted weights DataFrame
        """
        _, trades = self.calculate_turnover(weights_prev, weights_target)

        # Flag small trades
        trades['is_small_trade'] = trades['abs_weight_change'] < self.min_trade_threshold

        # Keep previous weight for small trades
        trades['weight_final'] = np.where(
            trades['is_small_trade'],
            trades['weight_prev'],
            trades['weight_target']
        )

        num_filtered = trades['is_small_trade'].sum()
        if num_filtered > 0:
            logger.info(f"Filtered {num_filtered} trades below {self.min_trade_threshold:.2%} threshold")

        # Normalize
        trades['weight_final'] = trades['weight_final'].clip(lower=0.0)
        total_weight = trades['weight_final'].sum()

        if total_weight > 0:
            trades['weight_final'] = trades['weight_final'] / total_weight

        weights_adjusted = trades[trades['weight_final'] > 0][['symbol', 'weight_final']].copy()
        weights_adjusted.rename(columns={'weight_final': 'weight'}, inplace=True)
        weights_adjusted['date'] = date

        return weights_adjusted

    def calculate_trading_costs(
        self,
        weights_prev: pd.DataFrame,
        weights_target: pd.DataFrame,
        portfolio_value: float = 10000.0
    ) -> Dict[str, float]:
        """
        Calculate estimated trading costs

        Args:
            weights_prev: Previous weights
            weights_target: Target weights
            portfolio_value: Current portfolio value ($)

        Returns:
            Dict with cost breakdown
        """
        turnover, trades = self.calculate_turnover(weights_prev, weights_target)

        # Total notional traded
        notional_traded = turnover * 2.0 * portfolio_value  # Both buys and sells

        # Commission costs
        commission = notional_traded * (self.costs_bps / 10000.0)

        # Slippage costs
        slippage = notional_traded * (self.slippage_bps / 10000.0)

        # Total costs
        total_cost = commission + slippage

        costs = {
            'turnover': turnover,
            'notional_traded': notional_traded,
            'commission': commission,
            'slippage': slippage,
            'total_cost': total_cost,
            'cost_bps': (total_cost / portfolio_value) * 10000.0 if portfolio_value > 0 else 0.0
        }

        logger.info(f"Trading costs: {costs['total_cost']:.2f} ({costs['cost_bps']:.1f} bps of portfolio)")

        return costs

    def generate_turnover_report(
        self,
        weights_history: pd.DataFrame,
        portfolio_value: float = 10000.0
    ) -> Dict[str, any]:
        """
        Generate comprehensive turnover report

        Args:
            weights_history: DataFrame with all historical weights [date, symbol, weight]
            portfolio_value: Portfolio value for cost calculation

        Returns:
            Dict with turnover analytics
        """
        logger.info("Generating turnover report...")

        dates = sorted(weights_history['date'].unique())

        turnovers = []
        costs_list = []

        for i in range(1, len(dates)):
            date_prev = dates[i-1]
            date_curr = dates[i]

            weights_prev = weights_history[weights_history['date'] == date_prev]
            weights_curr = weights_history[weights_history['date'] == date_curr]

            turnover, _ = self.calculate_turnover(weights_prev, weights_curr)
            costs = self.calculate_trading_costs(weights_prev, weights_curr, portfolio_value)

            turnovers.append({
                'date': date_curr,
                'turnover': turnover,
                'cost_bps': costs['cost_bps']
            })

            costs_list.append(costs['total_cost'])

        turnover_df = pd.DataFrame(turnovers)

        report = {
            'avg_turnover': float(turnover_df['turnover'].mean()),
            'max_turnover': float(turnover_df['turnover'].max()),
            'min_turnover': float(turnover_df['turnover'].min()),
            'avg_cost_bps': float(turnover_df['cost_bps'].mean()),
            'total_costs': float(sum(costs_list)),
            'num_rebalances': len(dates) - 1,
            'turnover_cap': self.turnover_cap_annual,
            'cap_breaches': int((turnover_df['turnover'] > self.max_turnover).sum()),
            'turnover_history': turnover_df.to_dict('records')
        }

        logger.info(f"✓ Turnover report generated:")
        logger.info(f"  Avg turnover: {report['avg_turnover']:.2%}")
        logger.info(f"  Max turnover: {report['max_turnover']:.2%}")
        logger.info(f"  Cap breaches: {report['cap_breaches']}")
        logger.info(f"  Total costs: ${report['total_costs']:.2f}")

        return report


def run_turnover_management(
    config: Dict,
    weights_df: pd.DataFrame,
    enforce_cap: bool = True,
    apply_min_threshold: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Standalone function to run turnover management

    Args:
        config: Configuration dictionary
        weights_df: DataFrame with portfolio weights history [date, symbol, weight]
        enforce_cap: Whether to enforce turnover cap
        apply_min_threshold: Whether to filter small trades

    Returns:
        (adjusted_weights_df, turnover_report)
    """
    logger.info("Running turnover management...")

    manager = TurnoverManager(config)

    dates = sorted(weights_df['date'].unique())
    adjusted_weights_list = []

    # First date: no adjustments
    first_weights = weights_df[weights_df['date'] == dates[0]].copy()
    adjusted_weights_list.append(first_weights)

    # Subsequent dates: apply constraints
    for i in range(1, len(dates)):
        date_prev = dates[i-1]
        date_curr = dates[i]

        weights_prev = weights_df[weights_df['date'] == date_prev]
        weights_target = weights_df[weights_df['date'] == date_curr]

        # Apply min threshold filter
        if apply_min_threshold:
            weights_target = manager.apply_min_trade_threshold(weights_prev, weights_target, date_curr)

        # Enforce turnover cap
        if enforce_cap:
            weights_adjusted = manager.enforce_turnover_cap(weights_prev, weights_target, date_curr)
        else:
            weights_adjusted = weights_target.copy()
            weights_adjusted['date'] = date_curr

        adjusted_weights_list.append(weights_adjusted)

    adjusted_weights_df = pd.concat(adjusted_weights_list, ignore_index=True)

    # Generate report
    report = manager.generate_turnover_report(adjusted_weights_df)

    logger.info(f"✓ Turnover management complete")

    return adjusted_weights_df, report
