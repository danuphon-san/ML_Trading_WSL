"""
Optimizer comparison and testing matrix framework

Systematically compare different portfolio optimizers under consistent conditions
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger

from src.portfolio.construct import construct_portfolio
from src.portfolio.risk import (
    calculate_portfolio_metrics,
    calculate_turnover,
    calculate_herfindahl_index,
    calculate_concentration_metrics
)
from src.backtest.bt_engine import VectorizedBacktester


def run_optimizer_comparison(
    scored_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    config: Dict,
    optimizers: List[str] = None,
    score_col: str = 'ml_score'
) -> pd.DataFrame:
    """
    Run systematic comparison of portfolio optimizers

    Tests all specified optimizers on the same data and generates
    comparative performance metrics.

    Args:
        scored_df: DataFrame with ML scores [date, symbol, ml_score, ...]
        price_panel: Historical prices [date, symbol, close]
        config: Configuration dictionary
        optimizers: List of optimizer names to test (None = all available)
        score_col: Score column name

    Returns:
        DataFrame with comparison metrics for each optimizer
    """
    if optimizers is None:
        optimizers = [
            'equal',
            'score_weighted',
            'inv_vol',
            'mvo',
            'mvo_reg',
            'hrp',
            'hybrid'
        ]

    logger.info(f"Running optimizer comparison for {len(optimizers)} methods")

    results_summary = []
    weights_history_all = {}

    # Test each optimizer
    for optimizer_name in optimizers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing optimizer: {optimizer_name}")
        logger.info(f"{'='*60}")

        try:
            # Update config with current optimizer
            config_copy = config.copy()
            config_copy['portfolio'] = config.get('portfolio', {}).copy()
            config_copy['portfolio']['optimizer'] = optimizer_name

            # Build portfolio weights over time
            weights_history = build_portfolio_weights_history(
                scored_df,
                price_panel,
                config_copy,
                score_col
            )

            if len(weights_history) == 0:
                logger.error(f"No weights generated for {optimizer_name}, skipping")
                continue

            weights_history_all[optimizer_name] = weights_history

            # Run backtest
            backtester = VectorizedBacktester(config_copy)
            backtest_results = backtester.run(weights_history, price_panel)

            metrics = backtest_results['metrics']

            # Calculate concentration metrics for final portfolio
            final_weights = weights_history.groupby('symbol')['weight'].last().to_dict()
            concentration = calculate_concentration_metrics(final_weights)

            # Compile results
            result_row = {
                'optimizer': optimizer_name,
                'total_return': metrics.get('total_return', 0),
                'annual_return': metrics.get('annual_return', 0),
                'volatility': metrics.get('volatility', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'avg_turnover': metrics.get('avg_turnover', 0),
                'hhi': concentration.get('hhi', 0),
                'effective_n': concentration.get('effective_n', 0),
                'top_3_concentration': concentration.get('top_3_concentration', 0)
            }

            results_summary.append(result_row)

            logger.info(f"✓ {optimizer_name}: Sharpe={result_row['sharpe_ratio']:.2f}, "
                       f"Return={result_row['annual_return']:.2%}, "
                       f"MaxDD={result_row['max_drawdown']:.2%}")

        except Exception as e:
            logger.error(f"Error testing {optimizer_name}: {e}")
            continue

    # Create summary DataFrame
    summary_df = pd.DataFrame(results_summary)

    if len(summary_df) > 0:
        # Sort by Sharpe ratio (descending)
        summary_df = summary_df.sort_values('sharpe_ratio', ascending=False)

        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZER COMPARISON SUMMARY")
        logger.info(f"{'='*60}\n")
        logger.info(summary_df.to_string(index=False))

    return summary_df


def build_portfolio_weights_history(
    scored_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    config: Dict,
    score_col: str = 'ml_score'
) -> pd.DataFrame:
    """
    Build portfolio weights over time for given scores

    Args:
        scored_df: DataFrame with ML scores [date, symbol, ml_score, ...]
        price_panel: Historical prices [date, symbol, close]
        config: Configuration dictionary
        score_col: Score column name

    Returns:
        DataFrame with [date, symbol, weight]
    """
    weights_history = []
    unique_dates = sorted(scored_df['date'].unique())

    logger.info(f"Building portfolio weights for {len(unique_dates)} rebalance dates")

    for i, date in enumerate(unique_dates):
        if i % 10 == 0:
            logger.debug(f"  Processing date {i+1}/{len(unique_dates)}: {date}")

        # Get scores for this date
        day_scores = scored_df[scored_df['date'] == date].copy()

        if len(day_scores) == 0:
            continue

        # Construct portfolio
        try:
            weights = construct_portfolio(day_scores, price_panel, config, score_col)

            # Record weights
            for symbol, weight in weights.items():
                weights_history.append({
                    'date': date,
                    'symbol': symbol,
                    'weight': weight
                })

        except Exception as e:
            logger.warning(f"Failed to construct portfolio for {date}: {e}")
            continue

    weights_df = pd.DataFrame(weights_history)

    logger.info(f"Generated {len(weights_df)} weight records over {len(unique_dates)} dates")

    return weights_df


def analyze_optimizer_characteristics(
    weights_history: pd.DataFrame
) -> Dict[str, float]:
    """
    Analyze characteristics of a portfolio optimizer

    Args:
        weights_history: DataFrame with [date, symbol, weight]

    Returns:
        Dictionary with optimizer characteristics
    """
    # Weight stability (std of weights over time for each symbol)
    weight_matrix = weights_history.pivot(index='date', columns='symbol', values='weight').fillna(0)

    stability = {
        'weight_std_mean': weight_matrix.std(axis=0).mean(),
        'weight_std_max': weight_matrix.std(axis=0).max(),
        'active_stocks_mean': (weight_matrix > 0.001).sum(axis=1).mean(),
        'active_stocks_std': (weight_matrix > 0.001).sum(axis=1).std()
    }

    # Turnover
    turnover_series = calculate_turnover(weights_history)
    stability['turnover_mean'] = turnover_series.mean()
    stability['turnover_std'] = turnover_series.std()
    stability['turnover_max'] = turnover_series.max()

    # Concentration over time
    hhi_over_time = []
    for date in weight_matrix.index:
        date_weights = weight_matrix.loc[date]
        date_weights = date_weights[date_weights > 0]
        hhi = calculate_herfindahl_index(date_weights.to_dict())
        hhi_over_time.append(hhi)

    stability['hhi_mean'] = np.mean(hhi_over_time)
    stability['hhi_std'] = np.std(hhi_over_time)

    return stability


def generate_optimizer_recommendations(
    comparison_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Generate recommendations based on optimizer comparison results

    Args:
        comparison_df: DataFrame from run_optimizer_comparison

    Returns:
        Dictionary with recommendations for different objectives
    """
    recommendations = {}

    # Best overall (Sharpe ratio)
    best_sharpe = comparison_df.loc[comparison_df['sharpe_ratio'].idxmax()]
    recommendations['best_overall'] = (
        f"{best_sharpe['optimizer']} - "
        f"Sharpe: {best_sharpe['sharpe_ratio']:.2f}, "
        f"Return: {best_sharpe['annual_return']:.2%}"
    )

    # Best return
    best_return = comparison_df.loc[comparison_df['annual_return'].idxmax()]
    recommendations['best_return'] = (
        f"{best_return['optimizer']} - "
        f"Return: {best_return['annual_return']:.2%}"
    )

    # Lowest drawdown
    min_drawdown = comparison_df.loc[comparison_df['max_drawdown'].idxmax()]  # Max because drawdowns are negative
    recommendations['lowest_drawdown'] = (
        f"{min_drawdown['optimizer']} - "
        f"MaxDD: {min_drawdown['max_drawdown']:.2%}"
    )

    # Lowest turnover
    min_turnover = comparison_df.loc[comparison_df['avg_turnover'].idxmin()]
    recommendations['lowest_turnover'] = (
        f"{min_turnover['optimizer']} - "
        f"Turnover: {min_turnover['avg_turnover']:.2%}"
    )

    # Most diversified (lowest HHI)
    most_diversified = comparison_df.loc[comparison_df['hhi'].idxmin()]
    recommendations['most_diversified'] = (
        f"{most_diversified['optimizer']} - "
        f"HHI: {most_diversified['hhi']:.4f}, "
        f"Effective N: {most_diversified['effective_n']:.1f}"
    )

    return recommendations
