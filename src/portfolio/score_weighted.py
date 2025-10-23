"""
Score-weighted portfolio construction

Weights proportional to ML signal scores - higher scores get more weight
"""
import numpy as np
import pandas as pd
from typing import Dict
from loguru import logger


def score_weighted_weights(
    selected_df: pd.DataFrame,
    config: Dict,
    score_col: str = 'ml_score'
) -> Dict[str, float]:
    """
    Compute weights proportional to ML scores

    Higher ML scores receive higher weights, respecting min/max constraints.

    Args:
        selected_df: DataFrame with selected stocks and ML scores [symbol, ml_score, ...]
        config: Configuration dictionary
        score_col: Column name for ML scores

    Returns:
        Dictionary of {symbol: weight}
    """
    portfolio_config = config.get('portfolio', {})
    min_weight = portfolio_config.get('pypfopt', {}).get('min_weight', 0.01)
    max_weight = portfolio_config.get('pypfopt', {}).get('max_weight', 0.15)

    logger.info(f"Computing score-weighted portfolio for {len(selected_df)} stocks")

    df = selected_df.copy()

    # Get scores
    scores = df.set_index('symbol')[score_col]
    symbols = scores.index.tolist()

    # Handle negative scores by shifting to positive range
    min_score = scores.min()
    if min_score < 0:
        # Shift scores to be positive
        scores_positive = scores - min_score + 1e-6
    else:
        scores_positive = scores + 1e-6  # Add small epsilon to avoid division by zero

    # Compute raw weights proportional to scores
    raw_weights = scores_positive / scores_positive.sum()

    # Apply min/max constraints
    weights = raw_weights.clip(lower=min_weight, upper=max_weight)

    # Renormalize to sum to 1.0
    weights = weights / weights.sum()

    # Convert to dictionary
    weights_dict = weights.to_dict()

    # Log weight statistics
    weights_array = np.array(list(weights_dict.values()))
    logger.info(f"Score-weighted portfolio: mean={weights_array.mean():.4f}, "
               f"std={weights_array.std():.4f}, min={weights_array.min():.4f}, "
               f"max={weights_array.max():.4f}")

    # Validate
    total = sum(weights_dict.values())
    if abs(total - 1.0) > 0.01:
        logger.warning(f"Weights sum to {total:.4f}, renormalizing")
        weights_dict = {k: v / total for k, v in weights_dict.items()}

    return weights_dict


def softmax_weighted_weights(
    selected_df: pd.DataFrame,
    config: Dict,
    score_col: str = 'ml_score',
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Compute weights using softmax transformation of ML scores

    Softmax provides smoother distribution than linear proportional weighting.
    Temperature controls distribution sharpness:
    - temperature < 1: More concentrated (winner-take-more)
    - temperature > 1: More uniform

    Args:
        selected_df: DataFrame with selected stocks and ML scores
        config: Configuration dictionary
        score_col: Column name for ML scores
        temperature: Softmax temperature parameter

    Returns:
        Dictionary of {symbol: weight}
    """
    portfolio_config = config.get('portfolio', {})
    min_weight = portfolio_config.get('pypfopt', {}).get('min_weight', 0.01)
    max_weight = portfolio_config.get('pypfopt', {}).get('max_weight', 0.15)

    logger.info(f"Computing softmax-weighted portfolio for {len(selected_df)} stocks "
               f"(temperature={temperature})")

    df = selected_df.copy()

    # Get scores
    scores = df.set_index('symbol')[score_col]

    # Apply softmax transformation
    # w_i = exp(score_i / T) / sum(exp(score_j / T))
    exp_scores = np.exp(scores / temperature)
    weights = exp_scores / exp_scores.sum()

    # Apply min/max constraints
    weights = weights.clip(lower=min_weight, upper=max_weight)

    # Renormalize
    weights = weights / weights.sum()

    # Convert to dictionary
    weights_dict = weights.to_dict()

    # Log weight statistics
    weights_array = np.array(list(weights_dict.values()))
    logger.info(f"Softmax-weighted portfolio: mean={weights_array.mean():.4f}, "
               f"std={weights_array.std():.4f}, min={weights_array.min():.4f}, "
               f"max={weights_array.max():.4f}")

    return weights_dict
