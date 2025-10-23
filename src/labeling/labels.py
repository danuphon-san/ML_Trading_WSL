"""
Label generation for ML models (forward returns, etc.)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from loguru import logger


class LabelGenerator:
    """Generate labels for supervised learning"""

    def __init__(self, config: Dict):
        """
        Initialize label generator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.label_config = config.get('labels', {})

        self.horizon = self.label_config.get('horizon', 5)
        self.target_type = self.label_config.get('target_type', 'return')
        self.min_periods = self.label_config.get('min_periods', 3)

        logger.info(f"Initialized LabelGenerator: horizon={self.horizon}, type={self.target_type}")

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels for dataset

        Args:
            df: DataFrame with [date, symbol, close, ...]

        Returns:
            DataFrame with label columns added
        """
        logger.info(f"Generating labels for {len(df)} rows")

        df = df.sort_values(['symbol', 'date'])

        # Generate forward returns by symbol
        df = df.groupby('symbol', group_keys=False).apply(self._generate_symbol_labels)

        logger.info(f"Generated labels: {len(df)} rows")

        return df

    def _generate_symbol_labels(self, group: pd.DataFrame) -> pd.DataFrame:
        """Generate labels for single symbol"""
        df = group.copy()

        # Calculate forward return
        future_price = df['close'].shift(-self.horizon)

        if self.target_type == 'return':
            # Simple return
            df[f'forward_return_{self.horizon}d'] = (future_price - df['close']) / df['close']

        elif self.target_type == 'log_return':
            # Log return
            df[f'forward_return_{self.horizon}d'] = np.log(future_price / df['close'])

        elif self.target_type == 'binary':
            # Binary: positive or negative return
            raw_return = (future_price - df['close']) / df['close']
            df[f'forward_return_{self.horizon}d'] = (raw_return > 0).astype(int)

        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

        # Additional label types
        df = self._add_multi_horizon_labels(df)
        df = self._add_risk_adjusted_labels(df)

        return df

    def _add_multi_horizon_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add labels for multiple horizons"""
        # Common horizons: 1d, 5d, 10d, 20d
        for horizon in [1, 5, 10, 20]:
            if horizon == self.horizon:
                continue  # Already computed

            future_price = df['close'].shift(-horizon)

            if self.target_type == 'return':
                df[f'forward_return_{horizon}d'] = (future_price - df['close']) / df['close']
            elif self.target_type == 'log_return':
                df[f'forward_return_{horizon}d'] = np.log(future_price / df['close'])

        return df

    def _add_risk_adjusted_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add risk-adjusted return labels

        WARNING: This creates a LABEL (not a feature)!
        risk_adjusted_return contains future information (forward returns).

        DO NOT use this as a feature for ML training - it will cause data leakage!
        These columns are excluded automatically in src/ml/dataset.py
        """
        # Risk-adjusted return = return / volatility
        if f'forward_return_{self.horizon}d' in df.columns:
            # Calculate trailing volatility
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=20, min_periods=self.min_periods).std()

            # Avoid division by zero
            volatility = volatility.replace(0, np.nan)

            df[f'risk_adjusted_return_{self.horizon}d'] = df[f'forward_return_{self.horizon}d'] / volatility

        return df

    def create_classification_labels(
        self,
        df: pd.DataFrame,
        n_classes: int = 3
    ) -> pd.DataFrame:
        """
        Create classification labels from continuous returns

        Args:
            df: DataFrame with forward returns
            n_classes: Number of classes (3 = bearish/neutral/bullish, 5 = more granular)

        Returns:
            DataFrame with classification labels
        """
        df = df.copy()

        label_col = f'forward_return_{self.horizon}d'

        if label_col not in df.columns:
            logger.error(f"Label column {label_col} not found")
            return df

        # Create quantile-based classes
        df[f'forward_class_{n_classes}'] = pd.qcut(
            df[label_col],
            q=n_classes,
            labels=range(n_classes),
            duplicates='drop'
        )

        return df


def generate_forward_returns(
    df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Convenience function to generate forward returns

    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary

    Returns:
        DataFrame with forward return labels
    """
    lg = LabelGenerator(config)
    return lg.generate_labels(df)


def create_train_test_split(
    df: pd.DataFrame,
    test_start_date: str,
    label_col: str = 'forward_return_5d'
) -> tuple:
    """
    Create time-based train/test split

    Args:
        df: DataFrame with features and labels
        test_start_date: Start date for test set (YYYY-MM-DD)
        label_col: Label column name

    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values('date')

    # Remove rows with missing labels
    df = df[df[label_col].notna()]

    # Split by date
    test_start = pd.to_datetime(test_start_date)

    train_df = df[df['date'] < test_start].copy()
    test_df = df[df['date'] >= test_start].copy()

    logger.info(f"Train: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    logger.info(f"Test: {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

    return train_df, test_df


def create_cross_sectional_labels(
    df: pd.DataFrame,
    label_col: str = 'forward_return_5d',
    method: str = 'rank'
) -> pd.DataFrame:
    """
    Create cross-sectional labels (rank or quintile within date)

    Args:
        df: DataFrame with forward returns
        label_col: Label column
        method: 'rank' or 'quintile'

    Returns:
        DataFrame with cross-sectional labels
    """
    df = df.copy()

    if method == 'rank':
        # Rank within each date (0 = worst, 1 = best)
        df[f'{label_col}_rank'] = df.groupby('date')[label_col].rank(pct=True)

    elif method == 'quintile':
        # Quintile within each date
        df[f'{label_col}_quintile'] = df.groupby('date')[label_col].transform(
            lambda x: pd.qcut(x, q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
        )

    return df
