"""
Storage utilities for reading/writing data
"""
from pathlib import Path
from typing import Optional, List
import pandas as pd
from loguru import logger


def save_dataframe(
    df: pd.DataFrame,
    path: str,
    format: str = "parquet",
    **kwargs
):
    """
    Save dataframe to file

    Args:
        df: DataFrame to save
        path: Output path
        format: File format (parquet, csv, pickle)
        **kwargs: Additional arguments for save method
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.to_parquet(path, index=False, compression='snappy', **kwargs)
    elif format == "csv":
        df.to_csv(path, index=False, **kwargs)
    elif format == "pickle":
        df.to_pickle(path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.debug(f"Saved {len(df)} rows to {path}")


def load_dataframe(
    path: str,
    format: str = "parquet",
    **kwargs
) -> pd.DataFrame:
    """
    Load dataframe from file

    Args:
        path: Input path
        format: File format (parquet, csv, pickle)
        **kwargs: Additional arguments for load method

    Returns:
        Loaded DataFrame
    """
    path = Path(path)

    if not path.exists():
        logger.error(f"File {path} does not exist")
        return pd.DataFrame()

    if format == "parquet":
        df = pd.read_parquet(path, **kwargs)
    elif format == "csv":
        df = pd.read_csv(path, **kwargs)
    elif format == "pickle":
        df = pd.read_pickle(path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.debug(f"Loaded {len(df)} rows from {path}")
    return df


def list_symbols(storage_path: str, pattern: str = "*.parquet") -> List[str]:
    """
    List available symbols in storage directory

    Args:
        storage_path: Path to storage directory
        pattern: File pattern

    Returns:
        List of symbol strings (without extension)
    """
    path = Path(storage_path)

    if not path.exists():
        logger.warning(f"Path {path} does not exist")
        return []

    files = list(path.glob(pattern))
    symbols = [f.stem for f in files]

    return sorted(symbols)


def get_date_range(
    storage_path: str,
    symbol: str,
    date_column: str = "date"
) -> tuple:
    """
    Get date range for a symbol

    Args:
        storage_path: Path to storage directory
        symbol: Symbol to check
        date_column: Name of date column

    Returns:
        Tuple of (start_date, end_date) as timestamps
    """
    path = Path(storage_path) / f"{symbol}.parquet"

    if not path.exists():
        logger.warning(f"File {path} does not exist")
        return None, None

    df = pd.read_parquet(path, columns=[date_column])

    if df.empty:
        return None, None

    return df[date_column].min(), df[date_column].max()


def merge_price_data(
    ohlcv_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    how: str = "left"
) -> pd.DataFrame:
    """
    Merge price and fundamental data

    Args:
        ohlcv_df: OHLCV DataFrame with [date, symbol, ...]
        fundamentals_df: Fundamentals DataFrame with [date, symbol, ...]
        how: Merge method (left, inner, outer)

    Returns:
        Merged DataFrame
    """
    # Merge on date and symbol
    merged = pd.merge(
        ohlcv_df,
        fundamentals_df,
        on=['date', 'symbol'],
        how=how,
        suffixes=('', '_fund')
    )

    logger.debug(f"Merged {len(ohlcv_df)} price rows with {len(fundamentals_df)} fundamental rows -> {len(merged)} rows")

    return merged
