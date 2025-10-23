"""
Hierarchical Risk Parity (HRP) portfolio construction

HRP uses hierarchical clustering to group similar assets and allocates weights
based on risk parity principles within and across clusters.
"""
import numpy as np
import pandas as pd
from typing import Dict
from loguru import logger

try:
    from pypfopt import HRPOpt, risk_models
    from pypfopt.exceptions import OptimizationError
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    logger.error("PyPortfolioOpt not installed - install with: pip install PyPortfolioOpt==1.5.5")


def hrp_weights(
    selected_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    config: Dict
) -> Dict[str, float]:
    """
    Compute Hierarchical Risk Parity (HRP) weights

    HRP provides diversified, stable allocations by:
    1. Clustering assets based on correlation
    2. Allocating weights inversely to cluster variance
    3. Robust to estimation error and regime changes

    Args:
        selected_df: DataFrame with selected stocks [symbol, ...]
        price_panel: Historical prices [date, symbol, close]
        config: Configuration dictionary

    Returns:
        Dictionary of {symbol: weight}
    """
    if not PYPFOPT_AVAILABLE:
        raise ImportError("PyPortfolioOpt not available for HRP")

    portfolio_config = config.get('portfolio', {})
    hrp_config = portfolio_config.get('hrp', {})

    lookback_days = hrp_config.get('lookback_days', 252)
    linkage_method = hrp_config.get('linkage_method', 'single')
    min_weight = portfolio_config.get('pypfopt', {}).get('min_weight', 0.01)
    max_weight = portfolio_config.get('pypfopt', {}).get('max_weight', 0.15)

    logger.info(f"Computing HRP portfolio for {len(selected_df)} stocks "
               f"(lookback={lookback_days}, linkage={linkage_method})")

    symbols = selected_df['symbol'].tolist()

    # Get price history
    prices = price_panel[price_panel['symbol'].isin(symbols)].copy()
    price_matrix = prices.pivot(index='date', columns='symbol', values='close')

    # Sort by date and take last N days
    price_matrix = price_matrix.sort_index()
    if len(price_matrix) > lookback_days:
        price_matrix = price_matrix.tail(lookback_days)

    # Check for sufficient history
    if len(price_matrix) < 20:
        logger.warning(f"Insufficient price history: {len(price_matrix)} days, using equal weights")
        return {s: 1.0 / len(symbols) for s in symbols}

    # Check for sufficient symbols
    if len(symbols) < 2:
        logger.warning("Less than 2 symbols, using equal weights")
        return {s: 1.0 / len(symbols) for s in symbols}

    # Calculate returns
    returns = price_matrix.pct_change().dropna()

    try:
        # Initialize HRP optimizer
        hrp = HRPOpt(returns, returns_data=True)

        # Optimize with specified linkage method
        raw_weights = hrp.optimize(linkage_method=linkage_method)

        # Clean weights (remove tiny positions)
        cleaned_weights = hrp.clean_weights()

        # Apply min/max constraints (HRP doesn't natively support bounds)
        weights = pd.Series(cleaned_weights)
        weights = weights.clip(lower=min_weight, upper=max_weight)

        # Renormalize to sum to 1.0
        weights = weights / weights.sum()

        weights_dict = weights.to_dict()

        # Log weight statistics
        weights_array = np.array(list(weights_dict.values()))
        logger.info(f"HRP portfolio: {len(weights_dict)} positions, "
                   f"mean={weights_array.mean():.4f}, std={weights_array.std():.4f}, "
                   f"min={weights_array.min():.4f}, max={weights_array.max():.4f}")

    except (OptimizationError, ValueError, Exception) as e:
        logger.error(f"HRP optimization failed: {e}, falling back to equal weights")
        weights_dict = {s: 1.0 / len(symbols) for s in symbols}

    # Validate
    total = sum(weights_dict.values())
    if abs(total - 1.0) > 0.01:
        logger.warning(f"Weights sum to {total:.4f}, renormalizing")
        weights_dict = {k: v / total for k, v in weights_dict.items()}

    return weights_dict


def hrp_with_scores_weights(
    selected_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    config: Dict,
    score_col: str = 'ml_score'
) -> Dict[str, float]:
    """
    HRP with ML score overlay

    Compute HRP base weights, then tilt towards higher-scoring assets

    Args:
        selected_df: DataFrame with selected stocks and ML scores
        price_panel: Historical prices
        config: Configuration dictionary
        score_col: Column name for ML scores

    Returns:
        Dictionary of {symbol: weight}
    """
    portfolio_config = config.get('portfolio', {})
    hrp_config = portfolio_config.get('hrp', {})
    score_tilt = hrp_config.get('score_tilt', 0.3)  # How much to tilt towards scores

    logger.info(f"Computing HRP with score tilt={score_tilt}")

    # Get base HRP weights
    hrp_weights_dict = hrp_weights(selected_df, price_panel, config)

    # Get normalized scores
    df = selected_df.copy()
    scores = df.set_index('symbol')[score_col]

    # Normalize scores to [0, 1]
    min_score = scores.min()
    if min_score < 0:
        scores_positive = scores - min_score + 1e-6
    else:
        scores_positive = scores + 1e-6

    score_range = scores_positive.max() - scores_positive.min()
    if score_range > 0:
        normalized_scores = (scores_positive - scores_positive.min()) / score_range
    else:
        normalized_scores = pd.Series(1.0, index=scores.index)

    # Tilt HRP weights towards scores
    # w_tilted = (1 - tilt) * w_hrp + tilt * normalized_score
    hrp_series = pd.Series(hrp_weights_dict)

    # Align indices
    common_symbols = list(set(hrp_series.index) & set(normalized_scores.index))
    hrp_series = hrp_series.loc[common_symbols]
    normalized_scores = normalized_scores.loc[common_symbols]

    # Apply tilt
    tilted_weights = (1 - score_tilt) * hrp_series + score_tilt * (normalized_scores / normalized_scores.sum())

    # Renormalize
    tilted_weights = tilted_weights / tilted_weights.sum()

    weights_dict = tilted_weights.to_dict()

    # Log weight statistics
    weights_array = np.array(list(weights_dict.values()))
    logger.info(f"HRP+Score portfolio: {len(weights_dict)} positions, "
               f"mean={weights_array.mean():.4f}, std={weights_array.std():.4f}")

    return weights_dict
