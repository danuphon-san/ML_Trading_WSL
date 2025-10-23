"""
Fundamental analysis features with Point-in-Time (PIT) alignment

CRITICAL: All fundamental features must respect publication dates to prevent look-ahead bias
"""
import numpy as np
import pandas as pd
from typing import Dict
from loguru import logger


class FundamentalFeatures:
    """Calculate fundamental features with PIT alignment"""

    def __init__(self, config: Dict):
        """
        Initialize with configuration

        Args:
            config: Configuration dictionary with PIT parameters
        """
        self.config = config
        self.pit_config = config.get('features', {}).get('pit_alignment', {})

        self.pit_min_lag_days = self.pit_config.get('pit_min_lag_days', 1)
        self.default_public_lag_days = self.pit_config.get('default_public_lag_days', 45)
        self.earnings_blackout_days = self.pit_config.get('earnings_blackout_days', 2)

        logger.info(f"Initialized FundamentalFeatures with PIT constraints: "
                   f"min_lag={self.pit_min_lag_days}, "
                   f"default_lag={self.default_public_lag_days}, "
                   f"blackout={self.earnings_blackout_days}")

    def compute_features(
        self,
        price_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute fundamental features with PIT alignment

        Args:
            price_df: DataFrame with [date, symbol, close, ...]
            fundamentals_df: DataFrame with [date, symbol, public_date, ...]

        Returns:
            DataFrame with price data and PIT-aligned fundamental features
        """
        logger.info(f"Computing fundamental features with PIT alignment")

        # Ensure we have public_date
        fundamentals_df = self._ensure_public_date(fundamentals_df)

        # Align fundamentals to price data with PIT constraints
        aligned_df = self._align_pit_fundamentals(price_df, fundamentals_df)

        # Compute derived ratios
        aligned_df = self._compute_ratios(aligned_df)

        logger.info(f"Computed fundamental features: {len(aligned_df)} rows")

        return aligned_df

    def _ensure_public_date(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure public_date exists; if missing, estimate from quarter end date

        Args:
            fundamentals_df: Fundamentals DataFrame

        Returns:
            DataFrame with public_date column
        """
        df = fundamentals_df.copy()

        if 'public_date' not in df.columns or df['public_date'].isna().any():
            logger.warning(f"public_date missing for some rows, using default lag of {self.default_public_lag_days} days")

            # If public_date is completely missing, create it
            if 'public_date' not in df.columns:
                df['public_date'] = df['date'] + pd.Timedelta(days=self.default_public_lag_days)
            else:
                # Fill missing values
                mask = df['public_date'].isna()
                df.loc[mask, 'public_date'] = df.loc[mask, 'date'] + pd.Timedelta(days=self.default_public_lag_days)

        return df

    def _align_pit_fundamentals(
        self,
        price_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align fundamentals to price data respecting PIT constraints

        For each price date:
        - Use most recent fundamental data where:
          1. public_date + pit_min_lag_days <= price_date
          2. Not in earnings blackout window

        Args:
            price_df: Price DataFrame
            fundamentals_df: Fundamentals DataFrame with public_date

        Returns:
            Merged DataFrame with PIT-aligned fundamentals
        """
        logger.info("Applying PIT alignment to fundamentals")

        # Add minimum lag to public_date
        fundamentals_df = fundamentals_df.copy()
        price_df = price_df.copy()

        fundamentals_df['available_date'] = fundamentals_df['public_date'] + pd.Timedelta(days=self.pit_min_lag_days)

        # Normalize timezones for merge compatibility
        # Convert both to timezone-naive (strip timezone from price data if present)
        if pd.api.types.is_datetime64tz_dtype(price_df['date']):
            price_df['date'] = price_df['date'].dt.tz_localize(None)
            logger.debug("Stripped timezone from price_df['date'] for merge compatibility")

        if pd.api.types.is_datetime64tz_dtype(fundamentals_df['available_date']):
            fundamentals_df['available_date'] = fundamentals_df['available_date'].dt.tz_localize(None)

        if pd.api.types.is_datetime64tz_dtype(fundamentals_df['public_date']):
            fundamentals_df['public_date'] = fundamentals_df['public_date'].dt.tz_localize(None)

        # Sort both dataframes
        price_df = price_df.sort_values(['symbol', 'date'])
        fundamentals_df = fundamentals_df.sort_values(['symbol', 'available_date'])

        # Merge as-of join (using available_date as key)
        aligned = pd.merge_asof(
            price_df,
            fundamentals_df,
            left_on='date',
            right_on='available_date',
            by='symbol',
            direction='backward',  # Use most recent available data
            suffixes=('', '_fund')
        )

        # Apply earnings blackout filter
        aligned = self._apply_earnings_blackout(aligned)

        # Drop temporary columns
        cols_to_drop = [c for c in aligned.columns if c.endswith('_fund') or c in ['available_date']]
        aligned = aligned.drop(columns=cols_to_drop, errors='ignore')

        logger.info(f"PIT alignment complete: {len(aligned)} rows")

        return aligned

    def _apply_earnings_blackout(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply earnings blackout period

        Exclude using fundamentals within blackout window around public_date

        Args:
            df: DataFrame with price date and public_date

        Returns:
            DataFrame with fundamentals nulled during blackout
        """
        if self.earnings_blackout_days == 0:
            return df

        df = df.copy()

        # Calculate days since publication
        if 'public_date' in df.columns:
            days_since_pub = (df['date'] - df['public_date']).dt.days

            # Mask rows within blackout window
            blackout_mask = days_since_pub < self.earnings_blackout_days

            # Set fundamental columns to NaN during blackout
            fund_cols = [c for c in df.columns if c not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

            for col in fund_cols:
                if col in df.columns:
                    df.loc[blackout_mask, col] = np.nan

            logger.debug(f"Applied earnings blackout to {blackout_mask.sum()} rows")

        return df

    def _compute_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived fundamental ratios

        Args:
            df: DataFrame with fundamental data

        Returns:
            DataFrame with additional ratio features
        """
        df = df.copy()

        # Valuation ratios (already present from data provider, but can recalculate)
        if 'market_cap' in df.columns and 'revenue' in df.columns:
            df['ps_ratio_calc'] = df['market_cap'] / df['revenue'].replace(0, np.nan)

        if 'market_cap' in df.columns and 'net_income' in df.columns:
            df['pe_ratio_calc'] = df['market_cap'] / df['net_income'].replace(0, np.nan)

        if 'market_cap' in df.columns and 'shareholders_equity' in df.columns:
            df['pb_ratio_calc'] = df['market_cap'] / df['shareholders_equity'].replace(0, np.nan)

        # Profitability ratios
        if 'net_income' in df.columns and 'revenue' in df.columns:
            df['profit_margin'] = df['net_income'] / df['revenue'].replace(0, np.nan)

        if 'net_income' in df.columns and 'shareholders_equity' in df.columns:
            df['roe_calc'] = df['net_income'] / df['shareholders_equity'].replace(0, np.nan)

        if 'net_income' in df.columns and 'total_assets' in df.columns:
            df['roa_calc'] = df['net_income'] / df['total_assets'].replace(0, np.nan)

        # Leverage ratios
        if 'total_liabilities' in df.columns and 'shareholders_equity' in df.columns:
            df['debt_to_equity_calc'] = df['total_liabilities'] / df['shareholders_equity'].replace(0, np.nan)

        if 'total_liabilities' in df.columns and 'total_assets' in df.columns:
            df['debt_ratio'] = df['total_liabilities'] / df['total_assets'].replace(0, np.nan)

        # Growth metrics (quarter-over-quarter)
        df = df.sort_values(['symbol', 'date'])

        for metric in ['revenue', 'net_income', 'operating_cash_flow']:
            if metric in df.columns:
                df[f'{metric}_qoq_growth'] = df.groupby('symbol')[metric].pct_change()

        # Quality scores
        if 'operating_cash_flow' in df.columns and 'net_income' in df.columns:
            df['cash_quality'] = df['operating_cash_flow'] / df['net_income'].replace(0, np.nan)

        return df


def align_pit_fundamentals(
    price_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Convenience function to align fundamentals with PIT constraints

    Args:
        price_df: Price DataFrame
        fundamentals_df: Fundamentals DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with PIT-aligned fundamentals
    """
    fa = FundamentalFeatures(config)
    return fa.compute_features(price_df, fundamentals_df)
