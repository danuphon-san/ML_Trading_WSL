"""
Hybrid portfolio construction combining ML scores and risk metrics

Balances predictive power (ML scores) with diversification (inverse volatility)
"""
import numpy as np
import pandas as pd
from typing import Dict
from loguru import logger


def hybrid_score_risk_weights(
    selected_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    config: Dict,
    score_col: str = 'ml_score'
) -> Dict[str, float]:
    """
    Compute hybrid weights combining ML scores and inverse volatility

    Weight formula:
    w_i ∝ (score_weight × normalized_score_i) × (risk_weight × inverse_vol_i)

    Args:
        selected_df: DataFrame with selected stocks and ML scores [symbol, ml_score, ...]
        price_panel: Historical prices [date, symbol, close]
        config: Configuration dictionary
        score_col: Column name for ML scores

    Returns:
        Dictionary of {symbol: weight}
    """
    portfolio_config = config.get('portfolio', {})
    hybrid_config = portfolio_config.get('hybrid', {})

    score_weight = hybrid_config.get('score_weight', 0.5)
    risk_weight = hybrid_config.get('risk_weight', 0.5)
    lookback_days = hybrid_config.get('lookback_days', 60)
    min_weight = portfolio_config.get('pypfopt', {}).get('min_weight', 0.01)
    max_weight = portfolio_config.get('pypfopt', {}).get('max_weight', 0.15)

    logger.info(f"Computing hybrid portfolio for {len(selected_df)} stocks "
               f"(score_weight={score_weight}, risk_weight={risk_weight})")

    df = selected_df.copy()
    symbols = df['symbol'].tolist()

    # 1. Get normalized ML scores
    scores = df.set_index('symbol')[score_col]

    # Handle negative scores by shifting to positive range
    min_score = scores.min()
    if min_score < 0:
        scores_positive = scores - min_score + 1e-6
    else:
        scores_positive = scores + 1e-6

    # Normalize scores to [0, 1]
    score_range = scores_positive.max() - scores_positive.min()
    if score_range > 0:
        normalized_scores = (scores_positive - scores_positive.min()) / score_range
    else:
        normalized_scores = pd.Series(1.0, index=scores.index)

    # 2. Calculate inverse volatility
    prices = price_panel[price_panel['symbol'].isin(symbols)].copy()
    price_matrix = prices.pivot(index='date', columns='symbol', values='close')
    price_matrix = price_matrix.tail(lookback_days)

    # Calculate volatility
    returns = price_matrix.pct_change().dropna()

    if len(returns) < 2:
        logger.warning("Insufficient price history for volatility calculation, using equal weights")
        return {s: 1.0 / len(symbols) for s in symbols}

    volatility = returns.std()

    # Inverse volatility (handle zero volatility)
    inv_vol = 1 / (volatility + 1e-6)

    # Normalize inverse volatility to [0, 1]
    inv_vol_range = inv_vol.max() - inv_vol.min()
    if inv_vol_range > 0:
        normalized_inv_vol = (inv_vol - inv_vol.min()) / inv_vol_range
    else:
        normalized_inv_vol = pd.Series(1.0, index=inv_vol.index)

    # 3. Combine score and risk components
    # Ensure same symbols in both series
    common_symbols = list(set(normalized_scores.index) & set(normalized_inv_vol.index))

    if len(common_symbols) == 0:
        logger.error("No common symbols between scores and volatility, falling back to equal weights")
        return {s: 1.0 / len(symbols) for s in symbols}

    normalized_scores = normalized_scores.loc[common_symbols]
    normalized_inv_vol = normalized_inv_vol.loc[common_symbols]

    # Hybrid score: weighted combination
    hybrid_scores = (score_weight * normalized_scores + risk_weight * normalized_inv_vol)

    # Compute raw weights proportional to hybrid scores
    raw_weights = hybrid_scores / hybrid_scores.sum()

    # Apply min/max constraints
    weights = raw_weights.clip(lower=min_weight, upper=max_weight)

    # Renormalize to sum to 1.0
    weights = weights / weights.sum()

    # Convert to dictionary
    weights_dict = weights.to_dict()

    # Log weight statistics
    weights_array = np.array(list(weights_dict.values()))
    logger.info(f"Hybrid portfolio: {len(weights_dict)} positions, "
               f"mean={weights_array.mean():.4f}, std={weights_array.std():.4f}, "
               f"min={weights_array.min():.4f}, max={weights_array.max():.4f}")

    # Validate
    total = sum(weights_dict.values())
    if abs(total - 1.0) > 0.01:
        logger.warning(f"Weights sum to {total:.4f}, renormalizing")
        weights_dict = {k: v / total for k, v in weights_dict.items()}

    return weights_dict


def adaptive_hybrid_weights(
    selected_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    config: Dict,
    score_col: str = 'ml_score',
    regime: str = 'normal'
) -> Dict[str, float]:
    """
    Adaptive hybrid weighting that adjusts score/risk balance based on market regime

    Args:
        selected_df: DataFrame with selected stocks and ML scores
        price_panel: Historical prices
        config: Configuration dictionary
        score_col: Column name for ML scores
        regime: Market regime ('normal', 'high_volatility', 'low_volatility')

    Returns:
        Dictionary of {symbol: weight}
    """
    # Adjust weights based on regime
    regime_adjustments = {
        'normal': {'score_weight': 0.5, 'risk_weight': 0.5},
        'high_volatility': {'score_weight': 0.3, 'risk_weight': 0.7},  # More risk-focused
        'low_volatility': {'score_weight': 0.7, 'risk_weight': 0.3},   # More signal-focused
    }

    if regime in regime_adjustments:
        logger.info(f"Adaptive hybrid: Using {regime} regime adjustments")

        # Temporarily override config
        config_copy = config.copy()
        if 'portfolio' not in config_copy:
            config_copy['portfolio'] = {}
        if 'hybrid' not in config_copy['portfolio']:
            config_copy['portfolio']['hybrid'] = {}

        config_copy['portfolio']['hybrid'].update(regime_adjustments[regime])

        return hybrid_score_risk_weights(selected_df, price_panel, config_copy, score_col)
    else:
        logger.warning(f"Unknown regime '{regime}', using normal hybrid weights")
        return hybrid_score_risk_weights(selected_df, price_panel, config, score_col)
