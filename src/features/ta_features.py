"""
Technical analysis features
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from loguru import logger


class TechnicalFeatures:
    """Calculate technical indicators"""

    def __init__(self, config: Dict):
        """
        Initialize with configuration

        Args:
            config: Configuration dictionary with lookback windows
        """
        self.config = config
        self.windows = config.get('technical', {}).get('lookback_windows', {})

        logger.info("Initialized TechnicalFeatures")

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical features

        Args:
            df: DataFrame with OHLCV data [date, symbol, open, high, low, close, volume]

        Returns:
            DataFrame with added technical features
        """
        logger.info(f"Computing technical features for {len(df)} rows")

        # Group by symbol to calculate per-symbol features
        df = df.sort_values(['symbol', 'date'])

        # Compute features by group
        df = df.groupby('symbol', group_keys=False).apply(self._compute_symbol_features)

        logger.info(f"Computed technical features: {len(df)} rows")

        return df

    def _compute_symbol_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute features for single symbol"""
        df = group.copy()

        # Momentum features
        df = self._add_momentum(df)

        # RSI
        df = self._add_rsi(df)

        # Volatility
        df = self._add_volatility(df)

        # Moving averages
        df = self._add_moving_averages(df)

        # Volume indicators
        df = self._add_volume_indicators(df)

        # Statistical features
        df = self._add_statistical_features(df)

        # Composite features
        df = self._add_composite_features(df)

        # Range features
        df = self._add_range_features(df)

        # Regime indicators
        df = self._add_regime_features(df)

        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        windows = self.windows.get('short', [5, 10, 20]) + self.windows.get('medium', [50, 100])

        for window in windows:
            # Price momentum (return over window)
            df[f'momentum_{window}d'] = df['close'].pct_change(window)

            # Log return momentum
            df[f'log_momentum_{window}d'] = np.log(df['close'] / df['close'].shift(window))

        return df

    def _add_rsi(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Add RSI (Relative Strength Index)

        Args:
            df: DataFrame
            periods: RSI periods (default: [14, 28])

        Returns:
            DataFrame with RSI columns
        """
        if periods is None:
            periods = [14, 28]

        for period in periods:
            delta = df['close'].diff()

            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            df[f'rsi_{period}d'] = rsi

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        windows = self.windows.get('short', [5, 10, 20]) + self.windows.get('medium', [50])

        for window in windows:
            # Historical volatility (annualized)
            returns = df['close'].pct_change()
            df[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)

            # Parkinson volatility (uses high-low)
            hl = np.log(df['high'] / df['low'])
            df[f'parkinson_vol_{window}d'] = np.sqrt((1 / (4 * np.log(2))) * hl.rolling(window).mean()) * np.sqrt(252)

        # ATR (Average True Range)
        df = self._add_atr(df, periods=[14, 28])

        return df

    def _add_atr(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add ATR (Average True Range)"""
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        for period in periods:
            df[f'atr_{period}d'] = tr.rolling(window=period).mean()

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features"""
        windows = (
            self.windows.get('short', [5, 10, 20]) +
            self.windows.get('medium', [50, 100]) +
            self.windows.get('long', [200])
        )

        for window in windows:
            # Simple moving average
            df[f'sma_{window}d'] = df['close'].rolling(window).mean()

            # Distance from SMA (as percentage)
            df[f'dist_sma_{window}d'] = (df['close'] - df[f'sma_{window}d']) / df[f'sma_{window}d']

        # EMA (Exponential Moving Average)
        for window in [12, 26, 50]:
            df[f'ema_{window}d'] = df['close'].ewm(span=window, adjust=False).mean()

        # MACD
        df['macd'] = df['ema_12d'] - df['ema_26d']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        windows = self.windows.get('short', [5, 10, 20]) + self.windows.get('medium', [50])

        for window in windows:
            # Average volume
            df[f'avg_volume_{window}d'] = df['volume'].rolling(window).mean()

            # Volume ratio (current / average)
            df[f'volume_ratio_{window}d'] = df['volume'] / df[f'avg_volume_{window}d']

        # Dollar volume
        df['dollar_volume'] = df['close'] * df['volume']
        df['avg_dollar_volume_20d'] = df['dollar_volume'].rolling(20).mean()

        # On-Balance Volume (OBV)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        df['obv_ema_20d'] = obv.ewm(span=20, adjust=False).mean()

        # Volume-weighted average price (VWAP) - daily reset needed for proper calc
        # This is simplified version
        df['vwap_approx'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical features for return distribution analysis

        All features use only past data to avoid look-ahead bias
        """
        windows = [20, 60]  # Standard statistical windows

        # Calculate returns for statistical analysis
        returns = df['close'].pct_change()
        log_returns = np.log(df['close'] / df['close'].shift(1))

        for window in windows:
            # Skewness - measures asymmetry of return distribution
            df[f'return_skew_{window}d'] = returns.rolling(window).skew()

            # Kurtosis - measures tail heaviness of return distribution
            df[f'return_kurt_{window}d'] = returns.rolling(window).kurt()

            # Autocorrelation - lag-1 correlation of returns
            df[f'return_autocorr_{window}d'] = returns.rolling(window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan,
                raw=False
            )

            # Return Z-score - standardized returns
            mean_return = returns.rolling(window).mean()
            std_return = returns.rolling(window).std()
            df[f'return_zscore_{window}d'] = (returns - mean_return) / std_return

        # Overall return statistics
        df['return_mean_20d'] = returns.rolling(20).mean()
        df['return_std_20d'] = returns.rolling(20).std()

        return df

    def _add_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add composite/engineered features combining multiple indicators

        These features provide normalized signals for cross-sectional comparison
        """
        # Volatility ratio (short/long) - detects regime changes
        if 'volatility_10d' in df.columns and 'volatility_50d' in df.columns:
            df['vol_ratio_10_50'] = df['volatility_10d'] / (df['volatility_50d'] + 1e-10)

        if 'volatility_20d' in df.columns and 'volatility_100d' in df.columns:
            df['vol_ratio_20_100'] = df['volatility_20d'] / (df['volatility_100d'] + 1e-10)

        # Rolling Sharpe ratio - risk-adjusted returns
        windows = [20, 60]
        for window in windows:
            returns = df['close'].pct_change()
            mean_return = returns.rolling(window).mean()
            std_return = returns.rolling(window).std()
            # Annualized Sharpe (assuming 252 trading days)
            df[f'sharpe_{window}d'] = (mean_return / (std_return + 1e-10)) * np.sqrt(252)

        # Price Z-score - normalized distance from moving average
        # Uses SMA and volatility for normalization
        for window in [20, 50]:
            if f'sma_{window}d' in df.columns and f'volatility_{window}d' in df.columns:
                df[f'price_zscore_{window}d'] = (
                    (df['close'] - df[f'sma_{window}d']) /
                    (df['close'].rolling(window).std() + 1e-10)
                )

        # Momentum-to-volatility ratio (signal strength)
        for window in [10, 20]:
            if f'momentum_{window}d' in df.columns and f'volatility_{window}d' in df.columns:
                df[f'momentum_vol_ratio_{window}d'] = (
                    df[f'momentum_{window}d'] / (df[f'volatility_{window}d'] + 1e-10)
                )

        return df

    def _add_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add range-based features (High-Low patterns)
        """
        # Daily high-low range as % of close
        df['hl_range_pct'] = (df['high'] - df['low']) / df['close']

        # Rolling average of daily range
        windows = [5, 10, 20]
        for window in windows:
            df[f'avg_hl_range_{window}d'] = df['hl_range_pct'].rolling(window).mean()

            # Range expansion/contraction
            df[f'range_ratio_{window}d'] = (
                df['hl_range_pct'] / (df[f'avg_hl_range_{window}d'] + 1e-10)
            )

        # Stochastic Oscillator
        periods = [14, 28]
        for period in periods:
            lowest_low = df['low'].rolling(window=period).min()
            highest_high = df['high'].rolling(window=period).max()

            # %K line
            df[f'stochastic_k_{period}d'] = 100 * (
                (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
            )

            # %D line (3-period SMA of %K)
            df[f'stochastic_d_{period}d'] = df[f'stochastic_k_{period}d'].rolling(3).mean()

        # Direct volume % change (not just ratio)
        df['volume_pct_change'] = df['volume'].pct_change()
        df['volume_pct_change_5d'] = df['volume'].pct_change(5)

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime indicators

        These features help the ML model learn regime-conditional patterns:
        - Volatility regime (high vs low volatility periods)
        - Trend regime (strong trend vs ranging)
        - Drawdown intensity (stressed vs normal markets)
        - Market breadth proxies
        """
        # 1. Volatility Regime
        # Compare short-term vol to long-term vol
        returns = df['close'].pct_change()
        short_vol = returns.rolling(20).std() * np.sqrt(252)
        long_vol = returns.rolling(252).std() * np.sqrt(252)

        df['vol_regime'] = short_vol / (long_vol + 1e-10)
        # Values > 1.0 indicate elevated volatility

        # 2. Trend Strength (ADX-style)
        # Measure of trend vs ranging
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()

        df['trend_strength'] = abs(sma_20 - sma_50) / (df['close'] + 1e-10)
        # Higher values = stronger trend

        # 3. Drawdown Intensity
        # How far from recent high
        running_max = df['close'].expanding().max()
        df['drawdown_pct'] = (df['close'] - running_max) / (running_max + 1e-10)
        # Negative values, more negative = deeper drawdown

        # Distance from 200-day high
        rolling_max_200 = df['close'].rolling(200).max()
        df['dist_from_200d_high'] = (df['close'] - rolling_max_200) / (rolling_max_200 + 1e-10)

        # 4. Bull/Bear Signal
        # Simple binary indicator: above/below 200-day SMA
        sma_200 = df['close'].rolling(200).mean()
        df['above_200_sma'] = (df['close'] > sma_200).astype(int)

        # 5. Momentum Regime
        # Multiple timeframe momentum consensus
        mom_5d = df['close'].pct_change(5)
        mom_20d = df['close'].pct_change(20)
        mom_60d = df['close'].pct_change(60)

        # Count how many timeframes are positive
        df['momentum_consensus'] = (
            (mom_5d > 0).astype(int) +
            (mom_20d > 0).astype(int) +
            (mom_60d > 0).astype(int)
        )
        # 0 = all negative, 3 = all positive

        # 6. Volatility Expansion/Contraction
        # Rate of change of volatility
        df['vol_expansion'] = short_vol.pct_change(20)
        # Positive = volatility expanding, negative = contracting

        # 7. Crisis Indicator
        # Extreme drawdown + high volatility
        df['crisis_indicator'] = (
            (df['drawdown_pct'] < -0.15) &  # > 15% drawdown
            (df['vol_regime'] > 1.5)  # Volatility 50% above normal
        ).astype(int)

        # 8. Recovery Indicator
        # Bouncing from drawdown with declining volatility
        drawdown_improving = df['drawdown_pct'].diff(5) > 0
        vol_declining = df['vol_regime'].diff(5) < 0
        df['recovery_indicator'] = (drawdown_improving & vol_declining).astype(int)

        logger.debug(f"Added regime features: {[col for col in df.columns if 'regime' in col or 'crisis' in col or 'recovery' in col]}")

        return df


def create_technical_features(
    ohlcv_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Convenience function to create technical features

    Args:
        ohlcv_df: OHLCV DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with technical features
    """
    ta = TechnicalFeatures(config)
    return ta.compute_features(ohlcv_df)
