"""
Portfolio construction orchestration with regime-awareness
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger

from src.portfolio.rank import PortfolioRanker
from src.portfolio.pypfopt_construct import PyPortfolioOptimizer
from src.portfolio.score_weighted import score_weighted_weights
from src.portfolio.hybrid import hybrid_score_risk_weights
from src.portfolio.hrp import hrp_weights
from src.portfolio.regime_detection import RegimeDetector


def construct_portfolio(
    scored_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    config: Dict,
    score_col: str = 'ml_score',
    enable_regime_adaptation: bool = True
) -> Dict[str, float]:
    """
    Construct portfolio from ML scores with regime adaptation

    Args:
        scored_df: DataFrame with ML scores [date, symbol, ml_score, ...]
        price_panel: Historical prices [date, symbol, close]
        config: Configuration
        score_col: Score column name
        enable_regime_adaptation: Whether to apply regime-based adjustments

    Returns:
        Dictionary of {symbol: weight}
    """
    portfolio_config = config.get('portfolio', {})
    optimizer_type = portfolio_config.get('optimizer', 'pypfopt')

    logger.info(f"Constructing portfolio with optimizer={optimizer_type}, regime_adaptation={enable_regime_adaptation}")

    # Step 0: Detect current regime (if enabled)
    current_regime = None
    if enable_regime_adaptation:
        current_regime = _detect_current_regime(price_panel, config)
        if current_regime:
            logger.info(f"📊 Current regime: {current_regime['regime_name']} (multiplier={current_regime['risk_multiplier']:.2f})")

    # Step 1: Rank and select top K (regime-adjusted)
    adjusted_config = config
    if current_regime:
        adjusted_config = _adjust_config_for_regime(config, current_regime)

    ranker = PortfolioRanker(adjusted_config)
    selected_df = ranker.select_portfolio(scored_df, score_col)

    if len(selected_df) == 0:
        logger.warning("No stocks selected")
        return {}

    # Step 2: Optimize weights
    if optimizer_type == 'pypfopt' or optimizer_type == 'mvo' or optimizer_type == 'mvo_reg':
        optimizer = PyPortfolioOptimizer(config)
        weights = optimizer.optimize(selected_df, price_panel, score_col)

    elif optimizer_type == 'inverse_vol' or optimizer_type == 'inv_vol':
        weights = inverse_volatility_weights(selected_df, price_panel, config)

    elif optimizer_type == 'equal_weight' or optimizer_type == 'equal':
        symbols = selected_df['symbol'].tolist()
        weights = {s: 1.0 / len(symbols) for s in symbols}

    elif optimizer_type == 'score_weighted':
        weights = score_weighted_weights(selected_df, config, score_col)

    elif optimizer_type == 'hybrid':
        weights = hybrid_score_risk_weights(selected_df, price_panel, config, score_col)

    elif optimizer_type == 'hrp':
        weights = hrp_weights(selected_df, price_panel, config)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}. "
                        f"Available: pypfopt, inverse_vol, equal_weight, score_weighted, hybrid, hrp")

    # Step 3: Apply regime-based risk adjustment
    if current_regime and enable_regime_adaptation:
        weights = _apply_regime_risk_adjustment(weights, current_regime)

    logger.info(f"Portfolio constructed: {len(weights)} positions, sum={sum(weights.values()):.2%}")

    return weights


def inverse_volatility_weights(
    selected_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    config: Dict
) -> Dict[str, float]:
    """
    Inverse volatility weighting

    Args:
        selected_df: Selected stocks
        price_panel: Historical prices
        config: Configuration

    Returns:
        Dictionary of weights
    """
    inv_vol_config = config.get('portfolio', {}).get('inverse_vol', {})
    lookback_days = inv_vol_config.get('lookback_days', 60)

    symbols = selected_df['symbol'].tolist()

    # Get prices
    prices = price_panel[price_panel['symbol'].isin(symbols)].copy()
    price_matrix = prices.pivot(index='date', columns='symbol', values='close')
    price_matrix = price_matrix.tail(lookback_days)

    # Calculate volatility
    returns = price_matrix.pct_change().dropna()
    volatility = returns.std()

    # Inverse volatility weights
    inv_vol = 1 / volatility
    weights = inv_vol / inv_vol.sum()

    weights_dict = weights.to_dict()

    logger.info(f"Inverse volatility weights: {len(weights_dict)} positions")

    return weights_dict


def _detect_current_regime(
    price_panel: pd.DataFrame,
    config: Dict
) -> Optional[Dict]:
    """
    Detect current market regime using benchmark

    Args:
        price_panel: Historical prices
        config: Configuration

    Returns:
        Dict with regime info or None if detection fails
    """
    try:
        # Get benchmark symbol (e.g., SPY)
        benchmark_symbol = config.get('reporting', {}).get('benchmark', 'SPY')

        # Extract benchmark prices
        benchmark_data = price_panel[price_panel['symbol'] == benchmark_symbol].copy()

        if len(benchmark_data) < 60:
            logger.warning(f"Insufficient benchmark data ({len(benchmark_data)} rows), skipping regime detection")
            return None

        # Initialize regime detector
        detector = RegimeDetector(config)

        # Detect regimes
        regime_df = detector.detect(
            prices=benchmark_data['close'],
            dates=benchmark_data['date']
        )

        # Get current regime
        current_regime = detector.get_current_regime(regime_df)

        return current_regime

    except Exception as e:
        logger.warning(f"Regime detection failed: {e}, proceeding without regime adjustment")
        return None


def _adjust_config_for_regime(
    config: Dict,
    regime: Dict
) -> Dict:
    """
    Adjust portfolio configuration based on market regime

    Args:
        config: Original configuration
        regime: Regime information

    Returns:
        Adjusted configuration
    """
    adjusted_config = config.copy()
    portfolio_config = adjusted_config.get('portfolio', {})

    original_top_k = portfolio_config.get('top_k', 20)
    regime_id = regime['regime']

    # Adjust top_k based on regime
    if regime_id == 0:  # Risk-off
        # Fewer positions, concentrate on defensive stocks
        adjusted_top_k = max(int(original_top_k * 0.6), 10)
        logger.info(f"Risk-off regime: Reducing top_k from {original_top_k} to {adjusted_top_k}")

    elif regime_id == 2:  # Risk-on
        # More positions, increase diversification
        adjusted_top_k = min(int(original_top_k * 1.3), 50)
        logger.info(f"Risk-on regime: Increasing top_k from {original_top_k} to {adjusted_top_k}")

    else:  # Normal
        adjusted_top_k = original_top_k

    adjusted_config['portfolio']['top_k'] = adjusted_top_k

    return adjusted_config


def _apply_regime_risk_adjustment(
    weights: Dict[str, float],
    regime: Dict
) -> Dict[str, float]:
    """
    Apply regime-based risk adjustment to portfolio weights

    In risk-off regimes: Reduce overall exposure (create implicit cash position)
    In risk-on regimes: May increase exposure slightly (lever up within limits)

    Args:
        weights: Original portfolio weights
        regime: Regime information

    Returns:
        Adjusted weights
    """
    risk_multiplier = regime['risk_multiplier']
    regime_name = regime['regime_name']

    # Apply risk multiplier
    adjusted_weights = {symbol: weight * risk_multiplier for symbol, weight in weights.items()}

    total_weight = sum(adjusted_weights.values())

    logger.info(f"Regime adjustment ({regime_name}): Original sum=1.00, Adjusted sum={total_weight:.2%}")

    # Handle different cases
    if total_weight < 1.0:
        # Create implicit cash position
        cash_allocation = 1.0 - total_weight
        logger.info(f"💰 Cash allocation: {cash_allocation:.2%} (risk-off protection)")
        # Keep weights as-is (not renormalized) - represents partial investment

    elif total_weight > 1.0:
        # Renormalize to 1.0 (no leverage allowed in long-only portfolio)
        adjusted_weights = {symbol: weight / total_weight for symbol, weight in adjusted_weights.items()}
        logger.info(f"⚖️  Renormalized to 1.0 (long-only constraint)")

    return adjusted_weights
