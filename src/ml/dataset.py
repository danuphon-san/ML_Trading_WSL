"""
Dataset preparation for ML models
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger


class MLDataset:
    """Prepare datasets for ML training"""

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        label_col: str = 'forward_return_5d',
        exclude_cols: Optional[List[str]] = None
    ):
        """
        Initialize dataset

        Args:
            feature_cols: List of feature columns (None = auto-detect)
            label_col: Label column name
            exclude_cols: Columns to exclude from features
        """
        self.feature_cols = feature_cols
        self.label_col = label_col

        if exclude_cols is None:
            exclude_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']

        self.exclude_cols = exclude_cols

        logger.info(f"Initialized MLDataset with label={label_col}")

    def prepare(
        self,
        df: pd.DataFrame,
        auto_select_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and labels

        Args:
            df: DataFrame with features and labels
            auto_select_features: Automatically select feature columns

        Returns:
            Tuple of (X, y) DataFrames
        """
        logger.info(f"Preparing dataset from {len(df)} rows")

        # Remove rows with missing labels
        df = df[df[self.label_col].notna()].copy()

        # Select feature columns
        if self.feature_cols is None and auto_select_features:
            self.feature_cols = self._auto_select_features(df)

        # Extract features and labels
        X = df[self.feature_cols].copy()
        y = df[self.label_col].copy()

        # Handle missing values in features
        X = X.fillna(X.median())

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        logger.info(f"Dataset prepared: {len(X)} rows, {len(self.feature_cols)} features")

        return X, y

    def _auto_select_features(self, df: pd.DataFrame) -> List[str]:
        """Automatically select feature columns"""
        # Get all columns except excluded ones and label
        feature_cols = []

        for col in df.columns:
            # Skip excluded columns
            if col in self.exclude_cols:
                continue

            # Skip label columns (any column with 'forward' or 'risk_adjusted_return')
            # CRITICAL: risk_adjusted_return contains future information (labels)!
            if col.startswith('forward_') or 'risk_adjusted_return' in col:
                continue

            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            feature_cols.append(col)

        logger.info(f"Auto-selected {len(feature_cols)} features")

        return feature_cols

    def add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add metadata columns (date, symbol) for tracking

        Args:
            df: Original DataFrame

        Returns:
            DataFrame with metadata
        """
        metadata_cols = ['date', 'symbol']
        return df[[c for c in metadata_cols if c in df.columns]].copy()


def create_time_based_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    embargo_days: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create time-based train/test split with embargo

    Args:
        df: DataFrame with date column
        test_size: Fraction for test set
        embargo_days: Gap between train and test to prevent leakage

    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values('date')

    # Calculate split point
    n = len(df['date'].unique())
    split_idx = int(n * (1 - test_size))

    dates = sorted(df['date'].unique())
    train_end_date = dates[split_idx]
    embargo_end_date = train_end_date + pd.Timedelta(days=embargo_days)

    # Split with embargo
    train_df = df[df['date'] <= train_end_date].copy()
    test_df = df[df['date'] > embargo_end_date].copy()

    logger.info(f"Time split: Train={len(train_df)} rows (to {train_end_date}), "
               f"Test={len(test_df)} rows (from {embargo_end_date}), "
               f"Embargo={embargo_days} days")

    return train_df, test_df


def create_three_way_split(
    df: pd.DataFrame,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    embargo_days: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create three-way train/validation/test split with embargo

    This is the RECOMMENDED split for production models to avoid data snooping.

    Args:
        df: DataFrame with date column
        train_size: Fraction for training set
        val_size: Fraction for validation set (hyperparameter tuning)
        test_size: Fraction for final test set (NEVER tune on this!)
        embargo_days: Gap between splits to prevent leakage

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 0.01, \
        "Sizes must sum to 1.0"

    df = df.sort_values('date')
    dates = sorted(df['date'].unique())
    n_dates = len(dates)

    # Calculate split points
    train_end_idx = int(n_dates * train_size)
    val_end_idx = int(n_dates * (train_size + val_size))

    # Add embargo
    train_end_date = dates[train_end_idx]
    train_embargo_end = train_end_date + pd.Timedelta(days=embargo_days)

    val_end_date = dates[val_end_idx]
    val_embargo_end = val_end_date + pd.Timedelta(days=embargo_days)

    # Create splits
    train_df = df[df['date'] <= train_end_date].copy()
    val_df = df[(df['date'] > train_embargo_end) & (df['date'] <= val_end_date)].copy()
    test_df = df[df['date'] > val_embargo_end].copy()

    logger.info(f"Three-way split: "
               f"Train={len(train_df)} rows (to {train_end_date}), "
               f"Val={len(val_df)} rows ({train_embargo_end} to {val_end_date}), "
               f"Test={len(test_df)} rows (from {val_embargo_end}), "
               f"Embargo={embargo_days} days")

    return train_df, val_df, test_df


def create_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.2,
    embargo_days: int = 5,
    purge_days: int = 2
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create time-series cross-validation folds with embargo and purge

    Args:
        df: DataFrame with date column
        n_splits: Number of CV splits
        test_size: Fraction for test set in each split
        embargo_days: Gap between train and test
        purge_days: Additional purge period

    Returns:
        List of (train_df, test_df) tuples
    """
    df = df.sort_values('date')
    dates = sorted(df['date'].unique())

    n_dates = len(dates)
    test_window = int(n_dates * test_size)

    folds = []

    for i in range(n_splits):
        # Calculate window positions
        test_end_idx = n_dates - i * test_window // n_splits
        test_start_idx = test_end_idx - test_window

        if test_start_idx < 0:
            break

        # Define split dates
        test_start_date = dates[test_start_idx]
        test_end_date = dates[test_end_idx - 1]

        # Apply embargo and purge
        train_end_date = test_start_date - pd.Timedelta(days=embargo_days + purge_days)

        # Create splits
        train_df = df[df['date'] <= train_end_date].copy()
        test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)].copy()

        if len(train_df) > 0 and len(test_df) > 0:
            folds.append((train_df, test_df))

            logger.debug(f"Fold {i+1}: Train={len(train_df)} rows (to {train_end_date}), "
                        f"Test={len(test_df)} rows ({test_start_date} to {test_end_date})")

    logger.info(f"Created {len(folds)} CV folds")

    return folds
