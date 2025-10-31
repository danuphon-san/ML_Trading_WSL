"""
Safety Validation Module

Validates portfolio and trades before execution
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger


class SafetyValidator:
    """
    Validate portfolio construction and trade orders

    Safety checks:
    - Portfolio constraints (weights, position limits)
    - Trade validations (turnover, liquidity)
    - Kill-switch conditions (daily loss, Sharpe degradation)
    - Risk limits (sector exposure, market cap distribution)
    """

    def __init__(self, config: Dict):
        """
        Initialize safety validator

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Portfolio constraints
        portfolio_config = config.get('portfolio', {})
        pypfopt_config = portfolio_config.get('pypfopt', {})

        self.max_position_weight = pypfopt_config.get('max_weight', 0.15)
        self.min_position_weight = pypfopt_config.get('min_weight', 0.01)
        self.max_turnover = portfolio_config.get('max_turnover', 0.35)

        # Risk limits
        risk_config = config.get('risk', {})
        self.max_sector_weight = risk_config.get('max_sector_weight', 0.30)

        # Kill-switch parameters
        ops_config = config.get('ops', {})
        kill_switch_config = ops_config.get('kill_switch', {})

        self.kill_switch_enabled = kill_switch_config.get('enabled', True)
        self.max_daily_loss_pct = kill_switch_config.get('max_daily_loss_pct', 0.03)
        self.min_live_sharpe = kill_switch_config.get('min_live_sharpe_threshold', 0.5)
        self.sharpe_lookback_weeks = kill_switch_config.get('min_live_sharpe_lookback_weeks', 6)

        # Live safety settings
        live_config = config.get('live', {})
        safety_config = live_config.get('safety', {})

        self.max_daily_trades = safety_config.get('max_daily_trades', 50)
        self.max_volume_pct = live_config.get('max_volume_pct', 0.05)

        logger.info(f"SafetyValidator initialized: max_position={self.max_position_weight:.1%}, "
                   f"max_turnover={self.max_turnover:.1%}, kill_switch={self.kill_switch_enabled}")

    def validate_portfolio(
        self,
        target_weights: Dict[str, float],
        universe_data: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate portfolio constraints

        Checks:
        1. Weights sum to approximately 1.0 (within 1%)
        2. All weights >= 0 (long-only constraint)
        3. No position exceeds max_weight
        4. All positions above min_weight (if non-zero)
        5. Sector limits (if universe_data provided)

        Args:
            target_weights: Portfolio weights {symbol: weight}
            universe_data: Optional DataFrame with sector info

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        if not target_weights:
            violations.append("Portfolio is empty")
            return False, violations

        weights_series = pd.Series(target_weights)

        # Check 1: Weights sum to ~1.0
        total_weight = weights_series.sum()
        if abs(total_weight - 1.0) > 0.01:
            violations.append(f"Weights sum to {total_weight:.4f} (expected 1.0)")

        # Check 2: Long-only (all weights >= 0)
        negative_weights = weights_series[weights_series < 0]
        if len(negative_weights) > 0:
            violations.append(f"Found {len(negative_weights)} negative weights (long-only violated)")
            for symbol, weight in negative_weights.items():
                violations.append(f"  {symbol}: {weight:.4f}")

        # Check 3: Max position size
        oversized_positions = weights_series[weights_series > self.max_position_weight]
        if len(oversized_positions) > 0:
            violations.append(f"Found {len(oversized_positions)} positions exceeding {self.max_position_weight:.1%}")
            for symbol, weight in oversized_positions.items():
                violations.append(f"  {symbol}: {weight:.2%} (max: {self.max_position_weight:.2%})")

        # Check 4: Min position size (for non-zero weights)
        non_zero_weights = weights_series[weights_series > 0]
        undersized_positions = non_zero_weights[non_zero_weights < self.min_position_weight]
        if len(undersized_positions) > 0:
            violations.append(f"Found {len(undersized_positions)} positions below {self.min_position_weight:.1%}")
            for symbol, weight in undersized_positions.items():
                violations.append(f"  {symbol}: {weight:.2%} (min: {self.min_position_weight:.2%})")

        # Check 5: Sector limits (if data available)
        if universe_data is not None and 'sector' in universe_data.columns:
            sector_violations = self._check_sector_limits(target_weights, universe_data)
            violations.extend(sector_violations)

        is_valid = len(violations) == 0

        if is_valid:
            logger.info(f"✓ Portfolio validation passed: {len(target_weights)} positions, sum={total_weight:.4f}")
        else:
            logger.error(f"✗ Portfolio validation failed with {len(violations)} violations")
            for violation in violations:
                logger.error(f"  {violation}")

        return is_valid, violations

    def _check_sector_limits(
        self,
        target_weights: Dict[str, float],
        universe_data: pd.DataFrame
    ) -> List[str]:
        """
        Check sector concentration limits

        Args:
            target_weights: Portfolio weights
            universe_data: DataFrame with 'symbol' and 'sector' columns

        Returns:
            List of sector violations
        """
        violations = []

        # Merge weights with sector data
        weights_df = pd.DataFrame([
            {'symbol': symbol, 'weight': weight}
            for symbol, weight in target_weights.items()
        ])

        merged = weights_df.merge(
            universe_data[['symbol', 'sector']],
            on='symbol',
            how='left'
        )

        # Calculate sector exposures
        sector_weights = merged.groupby('sector')['weight'].sum()

        # Check limits
        oversized_sectors = sector_weights[sector_weights > self.max_sector_weight]
        if len(oversized_sectors) > 0:
            violations.append(f"Found {len(oversized_sectors)} sectors exceeding {self.max_sector_weight:.1%}")
            for sector, weight in oversized_sectors.items():
                violations.append(f"  {sector}: {weight:.2%} (max: {self.max_sector_weight:.2%})")

        return violations

    def validate_trades(
        self,
        trades: pd.DataFrame,
        portfolio_value: float,
        turnover: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate trade orders

        Checks:
        1. All symbols have valid prices (> 0)
        2. No single trade exceeds 10% of portfolio
        3. Turnover within limit
        4. Total trades within daily limit
        5. Total estimated costs reasonable (< 0.5% of portfolio)

        Args:
            trades: Trade orders DataFrame
            portfolio_value: Total portfolio value
            turnover: Optional pre-calculated turnover

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        if len(trades) == 0:
            logger.info("✓ No trades to validate (empty trade list)")
            return True, []

        # Check 1: Valid prices
        invalid_prices = trades[trades['price'] <= 0]
        if len(invalid_prices) > 0:
            violations.append(f"Found {len(invalid_prices)} trades with invalid prices")
            for _, trade in invalid_prices.iterrows():
                violations.append(f"  {trade['symbol']}: price={trade['price']}")

        # Check 2: Single trade size
        max_trade_pct = 0.10  # 10% of portfolio
        large_trades = trades[trades['notional'] > portfolio_value * max_trade_pct]
        if len(large_trades) > 0:
            violations.append(f"Found {len(large_trades)} trades exceeding {max_trade_pct:.1%} of portfolio")
            for _, trade in large_trades.iterrows():
                trade_pct = trade['notional'] / portfolio_value
                violations.append(f"  {trade['symbol']}: ${trade['notional']:,.0f} ({trade_pct:.2%})")

        # Check 3: Turnover limit
        if turnover is not None:
            if turnover > self.max_turnover:
                violations.append(f"Turnover {turnover:.2%} exceeds limit {self.max_turnover:.2%}")

        # Check 4: Daily trade count
        if len(trades) > self.max_daily_trades:
            violations.append(f"Trade count {len(trades)} exceeds daily limit {self.max_daily_trades}")

        # Check 5: Total costs
        if 'total_cost_est' in trades.columns:
            total_costs = trades['total_cost_est'].sum()
            cost_pct = total_costs / portfolio_value
            max_cost_pct = 0.005  # 0.5% of portfolio

            if cost_pct > max_cost_pct:
                violations.append(f"Estimated costs ${total_costs:.2f} ({cost_pct:.3%}) exceed {max_cost_pct:.3%} of portfolio")

        # Check 6: Liquidity warnings
        if 'liquidity_warning' in trades.columns:
            illiquid_trades = trades[trades['liquidity_warning'] == True]
            if len(illiquid_trades) > 0:
                violations.append(f"⚠️  {len(illiquid_trades)} trades may have liquidity issues (>{self.max_volume_pct:.1%} of volume)")
                for _, trade in illiquid_trades.head(5).iterrows():
                    violations.append(f"  {trade['symbol']}: {trade.get('volume_pct', 0):.2%} of daily volume")

        is_valid = len(violations) == 0

        if is_valid:
            logger.info(f"✓ Trade validation passed: {len(trades)} trades, turnover={turnover:.2%}")
        else:
            logger.warning(f"⚠️  Trade validation found {len(violations)} issues")
            for violation in violations:
                logger.warning(f"  {violation}")

        return is_valid, violations

    def check_kill_switch(
        self,
        equity_curve: pd.DataFrame,
        breaches: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Check if kill-switch should trigger

        Conditions:
        1. Daily loss exceeds max_daily_loss_pct
        2. Rolling Sharpe < min_live_sharpe over lookback window

        Args:
            equity_curve: DataFrame with columns [date, equity]
            breaches: Optional list to append breaches to

        Returns:
            {
                'triggered': bool,
                'reason': str or None,
                'daily_return': float,
                'daily_loss_breach': bool,
                'rolling_sharpe': float or None,
                'sharpe_breach': bool
            }
        """
        if not self.kill_switch_enabled:
            return {
                'triggered': False,
                'reason': 'Kill-switch disabled',
                'daily_return': None,
                'daily_loss_breach': False,
                'rolling_sharpe': None,
                'sharpe_breach': False
            }

        if len(equity_curve) < 2:
            return {
                'triggered': False,
                'reason': 'Insufficient history',
                'daily_return': None,
                'daily_loss_breach': False,
                'rolling_sharpe': None,
                'sharpe_breach': False
            }

        equity_curve = equity_curve.sort_values('date')
        returns = equity_curve['equity'].pct_change().dropna()

        # Check 1: Daily loss
        daily_return = returns.iloc[-1] if len(returns) > 0 else 0.0
        daily_loss_breach = daily_return < -self.max_daily_loss_pct

        # Check 2: Rolling Sharpe
        lookback_days = self.sharpe_lookback_weeks * 5  # Trading days (5 per week)
        rolling_sharpe = None
        sharpe_breach = False

        if len(returns) >= lookback_days:
            recent_returns = returns.tail(lookback_days)
            risk_free_rate = self.config.get('portfolio', {}).get('pypfopt', {}).get('risk_free_rate', 0.02)

            mean_return = recent_returns.mean()
            std_return = recent_returns.std()

            if std_return > 0:
                rolling_sharpe = (mean_return * 252 - risk_free_rate) / (std_return * np.sqrt(252))
                sharpe_breach = rolling_sharpe < self.min_live_sharpe

        # Determine if kill-switch triggered
        triggered = daily_loss_breach or sharpe_breach

        result = {
            'triggered': triggered,
            'daily_return': float(daily_return) if daily_return is not None else None,
            'daily_loss_breach': daily_loss_breach,
            'rolling_sharpe': float(rolling_sharpe) if rolling_sharpe is not None else None,
            'sharpe_breach': sharpe_breach,
            'reason': None
        }

        if triggered:
            if daily_loss_breach:
                result['reason'] = f"Daily loss {daily_return:.2%} exceeds threshold {-self.max_daily_loss_pct:.2%}"
                logger.error(f"🚨 KILL-SWITCH TRIGGERED: {result['reason']}")
            elif sharpe_breach:
                result['reason'] = f"Rolling Sharpe {rolling_sharpe:.2f} below threshold {self.min_live_sharpe:.2f}"
                logger.error(f"🚨 KILL-SWITCH TRIGGERED: {result['reason']}")

            if breaches is not None:
                breaches.append({
                    'type': 'kill_switch',
                    'date': str(equity_curve['date'].iloc[-1]),
                    'reason': result['reason']
                })

        else:
            logger.info(f"✓ Kill-switch check passed: daily_return={daily_return:.2%}, "
                       f"rolling_sharpe={rolling_sharpe:.2f if rolling_sharpe else 'N/A'}")

        return result

    def validate_liquidity(
        self,
        trades: pd.DataFrame,
        volume_data: pd.DataFrame,
        max_volume_pct: Optional[float] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate trade liquidity against daily volumes

        Args:
            trades: Trade orders DataFrame
            volume_data: DataFrame with [symbol, daily_volume]
            max_volume_pct: Maximum % of daily volume (default: from config)

        Returns:
            (enhanced_trades_df, list_of_warnings)
        """
        if len(trades) == 0:
            return trades, []

        if max_volume_pct is None:
            max_volume_pct = self.max_volume_pct

        warnings = []

        # Merge with volume data
        trades_enhanced = trades.merge(
            volume_data[['symbol', 'volume']].rename(columns={'volume': 'daily_volume'}),
            on='symbol',
            how='left'
        )

        # Calculate volume percentage
        trades_enhanced['volume_pct'] = (
            trades_enhanced['shares'].abs() / trades_enhanced['daily_volume']
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # Flag illiquid trades
        trades_enhanced['liquidity_warning'] = trades_enhanced['volume_pct'] > max_volume_pct

        illiquid_trades = trades_enhanced[trades_enhanced['liquidity_warning']]
        if len(illiquid_trades) > 0:
            warnings.append(f"⚠️  {len(illiquid_trades)} trades exceed {max_volume_pct:.1%} of daily volume")
            for _, trade in illiquid_trades.iterrows():
                warnings.append(f"  {trade['symbol']}: {trade['shares']:.0f} shares "
                               f"({trade['volume_pct']:.2%} of {trade['daily_volume']:.0f} volume)")

        return trades_enhanced, warnings

    def validate_all(
        self,
        target_weights: Dict[str, float],
        trades: pd.DataFrame,
        portfolio_value: float,
        universe_data: Optional[pd.DataFrame] = None,
        equity_curve: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run all safety validations

        Args:
            target_weights: Portfolio weights
            trades: Trade orders
            portfolio_value: Portfolio value
            universe_data: Optional universe data with sectors
            equity_curve: Optional equity curve for kill-switch

        Returns:
            {
                'overall_valid': bool,
                'portfolio_valid': bool,
                'portfolio_violations': [...]
                'trades_valid': bool,
                'trades_violations': [...]
                'kill_switch': {...}
                'all_violations': [...]
            }
        """
        logger.info("="*70)
        logger.info("Running comprehensive safety validation")
        logger.info("="*70)

        # Validate portfolio
        portfolio_valid, portfolio_violations = self.validate_portfolio(target_weights, universe_data)

        # Validate trades
        from src.live.trade_generator import TradeGenerator
        trade_gen = TradeGenerator(self.config)
        turnover = trade_gen.calculate_turnover(trades, portfolio_value)

        trades_valid, trades_violations = self.validate_trades(trades, portfolio_value, turnover)

        # Check kill-switch
        kill_switch_result = {'triggered': False, 'reason': None}
        if equity_curve is not None:
            kill_switch_result = self.check_kill_switch(equity_curve)

        # Compile results
        all_violations = portfolio_violations + trades_violations
        if kill_switch_result['triggered']:
            all_violations.append(f"🚨 KILL-SWITCH: {kill_switch_result['reason']}")

        overall_valid = (
            portfolio_valid and
            trades_valid and
            not kill_switch_result['triggered']
        )

        result = {
            'overall_valid': overall_valid,
            'portfolio_valid': portfolio_valid,
            'portfolio_violations': portfolio_violations,
            'trades_valid': trades_valid,
            'trades_violations': trades_violations,
            'kill_switch': kill_switch_result,
            'all_violations': all_violations
        }

        logger.info("="*70)
        if overall_valid:
            logger.info("✅ ALL SAFETY CHECKS PASSED")
        else:
            logger.error(f"❌ SAFETY VALIDATION FAILED: {len(all_violations)} violations")
            for violation in all_violations:
                logger.error(f"  {violation}")
        logger.info("="*70)

        return result
