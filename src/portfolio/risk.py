"""
Portfolio risk analytics
"""
import numpy as np
import pandas as pd
from typing import Dict
from loguru import logger


def calculate_portfolio_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics

    Args:
        returns: Series of portfolio returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of metrics
    """
    # Annualized return
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (252 / n_periods) - 1

    # Volatility
    volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio
    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / volatility if volatility > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
    sortino = excess_return / downside_std if downside_std > 0 else 0

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar
    }

    return metrics


def calculate_turnover(
    weights_history: pd.DataFrame
) -> pd.Series:
    """
    Calculate portfolio turnover

    Args:
        weights_history: DataFrame with [date, symbol, weight]

    Returns:
        Series of turnover by date
    """
    weights_matrix = weights_history.pivot(index='date', columns='symbol', values='weight').fillna(0)

    # Calculate absolute weight changes
    weight_changes = weights_matrix.diff().abs()
    turnover = weight_changes.sum(axis=1) / 2  # Divide by 2 for one-way turnover

    return turnover


def calculate_herfindahl_index(
    weights: Dict[str, float]
) -> float:
    """
    Calculate Herfindahl-Hirschman Index (HHI) for portfolio concentration

    HHI = Σ(w_i²) where w_i are portfolio weights

    Interpretation:
    - HHI = 1.0: Fully concentrated (one asset)
    - HHI = 1/N: Equally weighted (N assets)
    - Lower HHI: More diversified

    Args:
        weights: Dictionary of {symbol: weight}

    Returns:
        Herfindahl Index (concentration measure)
    """
    if not weights:
        return 0.0

    weights_array = np.array(list(weights.values()))
    hhi = np.sum(weights_array ** 2)

    return hhi


def calculate_concentration_metrics(
    weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate multiple concentration metrics for a portfolio

    Args:
        weights: Dictionary of {symbol: weight}

    Returns:
        Dictionary with concentration metrics
    """
    if not weights:
        return {
            'hhi': 0.0,
            'effective_n': 0.0,
            'top_3_concentration': 0.0,
            'top_5_concentration': 0.0
        }

    weights_array = np.array(list(weights.values()))
    sorted_weights = np.sort(weights_array)[::-1]  # Descending order

    # Herfindahl Index
    hhi = np.sum(weights_array ** 2)

    # Effective number of stocks (inverse of HHI)
    effective_n = 1 / hhi if hhi > 0 else 0

    # Top-K concentration
    top_3_concentration = sorted_weights[:3].sum() if len(sorted_weights) >= 3 else sorted_weights.sum()
    top_5_concentration = sorted_weights[:5].sum() if len(sorted_weights) >= 5 else sorted_weights.sum()

    metrics = {
        'hhi': hhi,
        'effective_n': effective_n,
        'top_3_concentration': top_3_concentration,
        'top_5_concentration': top_5_concentration
    }

    return metrics
