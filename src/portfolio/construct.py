"""
Portfolio construction orchestration
"""
import pandas as pd
from typing import Dict
from loguru import logger

from src.portfolio.rank import PortfolioRanker
from src.portfolio.pypfopt_construct import PyPortfolioOptimizer
from src.portfolio.score_weighted import score_weighted_weights
from src.portfolio.hybrid import hybrid_score_risk_weights
from src.portfolio.hrp import hrp_weights


def construct_portfolio(
    scored_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    config: Dict,
    score_col: str = 'ml_score'
) -> Dict[str, float]:
    """
    Construct portfolio from ML scores

    Args:
        scored_df: DataFrame with ML scores [date, symbol, ml_score, ...]
        price_panel: Historical prices [date, symbol, close]
        config: Configuration
        score_col: Score column name

    Returns:
        Dictionary of {symbol: weight}
    """
    portfolio_config = config.get('portfolio', {})
    optimizer_type = portfolio_config.get('optimizer', 'pypfopt')

    logger.info(f"Constructing portfolio with optimizer={optimizer_type}")

    # Step 1: Rank and select top K
    ranker = PortfolioRanker(config)
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

    logger.info(f"Portfolio constructed: {len(weights)} positions")

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
