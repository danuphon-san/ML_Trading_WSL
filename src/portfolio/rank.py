"""
Portfolio ranking and selection
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger


class PortfolioRanker:
    """Rank and select stocks for portfolio"""

    def __init__(self, config: Dict):
        """
        Initialize ranker

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.portfolio_config = config.get('portfolio', {})

        self.top_k = self.portfolio_config.get('top_k', 20)
        self.selection_method = self.portfolio_config.get('selection_method', 'top_score')
        self.score_threshold = self.portfolio_config.get('score_threshold', 0.0)

        logger.info(f"Initialized PortfolioRanker: top_k={self.top_k}, method={self.selection_method}")

    def select_portfolio(
        self,
        df: pd.DataFrame,
        score_col: str = 'ml_score',
        date: pd.Timestamp = None
    ) -> pd.DataFrame:
        """
        Select stocks for portfolio based on ML scores

        Args:
            df: DataFrame with [symbol, ml_score, ...]
            score_col: Column name for ML score
            date: Date for selection (for logging)

        Returns:
            DataFrame with selected stocks
        """
        if date is None:
            date = df['date'].iloc[0] if 'date' in df.columns else pd.Timestamp.now()

        logger.info(f"Selecting portfolio for {date} from {len(df)} candidates")

        # Remove invalid scores
        df = df[df[score_col].notna()].copy()

        # Rank by score
        df['rank'] = df[score_col].rank(ascending=False)

        # Select based on method
        if self.selection_method == 'top_score':
            selected = df.nsmallest(self.top_k, 'rank')

        elif self.selection_method == 'threshold':
            selected = df[df[score_col] >= self.score_threshold]
            selected = selected.nsmallest(self.top_k, 'rank')  # Still cap at top_k

        else:
            raise ValueError(f"Unknown selection_method: {self.selection_method}")

        logger.info(f"Selected {len(selected)} stocks")

        return selected

    def rank_cross_sectional(
        self,
        df: pd.DataFrame,
        score_col: str = 'ml_score'
    ) -> pd.DataFrame:
        """
        Rank stocks cross-sectionally by date

        Args:
            df: DataFrame with [date, symbol, ml_score, ...]
            score_col: Column name for ML score

        Returns:
            DataFrame with rank column added
        """
        df = df.copy()

        # Rank within each date
        df['rank'] = df.groupby('date')[score_col].rank(ascending=False)
        df['rank_pct'] = df.groupby('date')[score_col].rank(pct=True)

        return df


def select_top_k_by_score(
    df: pd.DataFrame,
    k: int,
    score_col: str = 'ml_score',
    filters: Dict = None
) -> pd.DataFrame:
    """
    Select top K stocks by score with optional filters

    Args:
        df: DataFrame with scores
        k: Number of stocks to select
        score_col: Score column name
        filters: Optional filters (e.g., {'sector': 'Technology'})

    Returns:
        DataFrame with top K stocks
    """
    # Apply filters
    if filters:
        for col, value in filters.items():
            if col in df.columns:
                if isinstance(value, list):
                    df = df[df[col].isin(value)]
                else:
                    df = df[df[col] == value]

    # Select top K
    df = df.nlargest(k, score_col)

    return df


def diversify_selection(
    df: pd.DataFrame,
    k: int,
    score_col: str = 'ml_score',
    sector_col: str = 'sector',
    max_sector_weight: float = 0.3
) -> pd.DataFrame:
    """
    Select diversified portfolio respecting sector constraints

    Args:
        df: DataFrame with scores and sectors
        k: Total number of stocks
        score_col: Score column name
        sector_col: Sector column name
        max_sector_weight: Maximum weight per sector

    Returns:
        Diversified selection
    """
    df = df.sort_values(score_col, ascending=False)

    max_per_sector = int(k * max_sector_weight)

    selected = []
    sector_counts = {}

    for _, row in df.iterrows():
        sector = row[sector_col]

        # Check sector constraint
        if sector_counts.get(sector, 0) < max_per_sector:
            selected.append(row)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        if len(selected) >= k:
            break

    result = pd.DataFrame(selected)

    logger.info(f"Diversified selection: {len(result)} stocks across {len(sector_counts)} sectors")

    return result


def create_long_short_portfolio(
    df: pd.DataFrame,
    score_col: str = 'ml_score',
    long_pct: float = 0.2,
    short_pct: float = 0.2
) -> tuple:
    """
    Create long-short portfolio

    Args:
        df: DataFrame with scores
        score_col: Score column name
        long_pct: Percentile for long positions (top)
        short_pct: Percentile for short positions (bottom)

    Returns:
        Tuple of (long_df, short_df)
    """
    df = df.sort_values(score_col, ascending=False)

    n = len(df)
    n_long = int(n * long_pct)
    n_short = int(n * short_pct)

    long_df = df.head(n_long).copy()
    short_df = df.tail(n_short).copy()

    logger.info(f"Long-short portfolio: {len(long_df)} long, {len(short_df)} short")

    return long_df, short_df
