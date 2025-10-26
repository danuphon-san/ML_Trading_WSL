"""
Regime Detection (Step 12)

Detect market regimes (risk-on/risk-off) using volatility, drawdown, and macro indicators.
Methods: Rules-based or HMM (Hidden Markov Model)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger


class RegimeDetector:
    """
    Market regime detection using volatility, drawdown, and trend analysis

    Regimes:
        0 = Risk-Off (high vol, drawdown, negative trend)
        1 = Normal (moderate vol, stable)
        2 = Risk-On (low vol, uptrend)
    """

    def __init__(self, config: Dict):
        """
        Initialize regime detector

        Args:
            config: Configuration dict with regime detection parameters
        """
        self.config = config

        # Detection parameters
        self.vol_window = config.get('regime', {}).get('vol_window', 20)
        self.vol_threshold_high = config.get('regime', {}).get('vol_threshold_high', 0.25)  # >25% annualized = high vol
        self.vol_threshold_low = config.get('regime', {}).get('vol_threshold_low', 0.15)   # <15% annualized = low vol

        self.dd_threshold = config.get('regime', {}).get('dd_threshold', -0.10)  # -10% drawdown = risk-off
        self.trend_window = config.get('regime', {}).get('trend_window', 50)

        self.method = config.get('regime', {}).get('method', 'rules')  # 'rules' or 'hmm'

        logger.info(f"Initialized RegimeDetector: method={self.method}, vol_window={self.vol_window}")

    def detect_regime_rules(self, prices: pd.Series, dates: pd.Series) -> pd.DataFrame:
        """
        Rules-based regime detection

        Rules:
        - Risk-Off (0): High volatility OR deep drawdown OR negative trend
        - Risk-On (2): Low volatility AND positive trend AND no significant drawdown
        - Normal (1): Everything else

        Args:
            prices: Series of prices (e.g., SPY close prices)
            dates: Corresponding dates

        Returns:
            DataFrame with columns [date, regime, volatility, drawdown, trend_signal, risk_multiplier]
        """
        df = pd.DataFrame({
            'date': dates,
            'price': prices
        })

        # Calculate volatility (annualized)
        returns = df['price'].pct_change()
        df['volatility'] = returns.rolling(self.vol_window).std() * np.sqrt(252)

        # Calculate drawdown
        running_max = df['price'].expanding().max()
        df['drawdown'] = (df['price'] - running_max) / running_max

        # Calculate trend (SMA crossover)
        sma_short = df['price'].rolling(20).mean()
        sma_long = df['price'].rolling(self.trend_window).mean()
        df['trend_signal'] = (sma_short > sma_long).astype(int)  # 1 = uptrend, 0 = downtrend

        # Assign regimes
        df['regime'] = 1  # Default: Normal

        # Risk-Off conditions (regime 0)
        risk_off_conditions = (
            (df['volatility'] > self.vol_threshold_high) |  # High volatility
            (df['drawdown'] < self.dd_threshold) |          # Deep drawdown
            (df['trend_signal'] == 0)                        # Downtrend
        )
        df.loc[risk_off_conditions, 'regime'] = 0

        # Risk-On conditions (regime 2)
        risk_on_conditions = (
            (df['volatility'] < self.vol_threshold_low) &   # Low volatility
            (df['drawdown'] > -0.05) &                       # Shallow or no drawdown
            (df['trend_signal'] == 1)                        # Uptrend
        )
        df.loc[risk_on_conditions, 'regime'] = 2

        # Risk multiplier (used to scale position sizes)
        df['risk_multiplier'] = df['regime'].map({
            0: 0.5,   # Risk-Off: reduce positions to 50%
            1: 1.0,   # Normal: full positions
            2: 1.2    # Risk-On: slightly increase positions (capped at 1.2x)
        })

        logger.info(f"Detected regimes: Risk-Off={np.sum(df['regime']==0)}, "
                   f"Normal={np.sum(df['regime']==1)}, Risk-On={np.sum(df['regime']==2)}")

        return df[['date', 'regime', 'volatility', 'drawdown', 'trend_signal', 'risk_multiplier']]

    def detect_regime_hmm(self, prices: pd.Series, dates: pd.Series, n_states: int = 3) -> pd.DataFrame:
        """
        HMM-based regime detection (requires hmmlearn package)

        Args:
            prices: Series of prices
            dates: Corresponding dates
            n_states: Number of hidden states (default: 3)

        Returns:
            DataFrame with columns [date, regime, regime_prob]
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.warning("hmmlearn not installed, falling back to rules-based detection")
            return self.detect_regime_rules(prices, dates)

        # Calculate features for HMM
        returns = prices.pct_change().dropna()

        # Feature: log returns
        features = returns.values.reshape(-1, 1)

        # Fit Gaussian HMM
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
        model.fit(features)

        # Predict regimes
        hidden_states = model.predict(features)

        # Map states to regime labels (0=risk-off, 1=normal, 2=risk-on)
        # Heuristic: sort states by mean return
        state_means = [features[hidden_states == i].mean() for i in range(n_states)]
        sorted_states = np.argsort(state_means)  # Lowest to highest

        state_mapping = {sorted_states[0]: 0, sorted_states[1]: 1, sorted_states[2]: 2}
        regimes = np.array([state_mapping[s] for s in hidden_states])

        # Get regime probabilities
        regime_probs = model.predict_proba(features)

        df = pd.DataFrame({
            'date': dates.iloc[1:].values,  # Skip first date (NaN return)
            'regime': regimes,
            'regime_prob_risk_off': regime_probs[:, state_mapping[0]],
            'regime_prob_normal': regime_probs[:, state_mapping[1]],
            'regime_prob_risk_on': regime_probs[:, state_mapping[2]]
        })

        # Add risk multiplier
        df['risk_multiplier'] = df['regime'].map({0: 0.5, 1: 1.0, 2: 1.2})

        logger.info(f"HMM detected regimes: Risk-Off={np.sum(regimes==0)}, "
                   f"Normal={np.sum(regimes==1)}, Risk-On={np.sum(regimes==2)}")

        return df

    def detect(self, prices: pd.Series, dates: pd.Series) -> pd.DataFrame:
        """
        Main detection method that dispatches to rules or HMM

        Args:
            prices: Series of benchmark prices (e.g., SPY)
            dates: Corresponding dates

        Returns:
            DataFrame with regime information
        """
        if self.method == 'hmm':
            return self.detect_regime_hmm(prices, dates)
        else:
            return self.detect_regime_rules(prices, dates)

    def get_current_regime(self, regime_df: pd.DataFrame) -> Dict:
        """
        Get the most recent regime state

        Args:
            regime_df: DataFrame from detect() method

        Returns:
            Dict with current regime info
        """
        latest = regime_df.iloc[-1]

        regime_names = {0: 'Risk-Off', 1: 'Normal', 2: 'Risk-On'}

        info = {
            'date': str(latest['date']),
            'regime': int(latest['regime']),
            'regime_name': regime_names[int(latest['regime'])],
            'risk_multiplier': float(latest['risk_multiplier'])
        }

        if 'volatility' in latest.index:
            info['volatility'] = float(latest['volatility'])
            info['drawdown'] = float(latest['drawdown'])

        logger.info(f"Current regime: {info['regime_name']} (multiplier={info['risk_multiplier']:.2f})")

        return info

    def apply_regime_adjustment(self, weights: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply regime-based adjustments to portfolio weights

        Args:
            weights: DataFrame with columns [date, symbol, weight]
            regime_df: DataFrame from detect() method

        Returns:
            Adjusted weights DataFrame
        """
        # Merge weights with regime data
        regime_lookup = regime_df[['date', 'risk_multiplier']].set_index('date')

        weights_adj = weights.copy()
        weights_adj['date'] = pd.to_datetime(weights_adj['date'])

        # Map risk multipliers
        weights_adj = weights_adj.merge(
            regime_lookup,
            left_on='date',
            right_index=True,
            how='left'
        )

        # Apply multiplier
        weights_adj['weight_original'] = weights_adj['weight']
        weights_adj['weight'] = weights_adj['weight'] * weights_adj['risk_multiplier']

        # Renormalize to sum to 1.0 per date
        weights_adj['weight'] = weights_adj.groupby('date')['weight'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else x
        )

        logger.info(f"Applied regime adjustments to {len(weights_adj)} weight records")

        return weights_adj[['date', 'symbol', 'weight', 'weight_original', 'risk_multiplier']]


def run_regime_detection(config: Dict, price_panel: pd.DataFrame, benchmark: str = 'SPY') -> pd.DataFrame:
    """
    Standalone function to run regime detection

    Args:
        config: Configuration dictionary
        price_panel: DataFrame with columns [date, symbol, close]
        benchmark: Symbol to use for regime detection (default: SPY)

    Returns:
        DataFrame with regime state data
    """
    logger.info(f"Running regime detection using benchmark: {benchmark}")

    # Extract benchmark prices
    benchmark_data = price_panel[price_panel['symbol'] == benchmark].copy()

    if len(benchmark_data) == 0:
        logger.warning(f"Benchmark {benchmark} not found in price_panel, using first available symbol")
        benchmark = price_panel['symbol'].iloc[0]
        benchmark_data = price_panel[price_panel['symbol'] == benchmark].copy()

    benchmark_data = benchmark_data.sort_values('date')

    # Run detection
    detector = RegimeDetector(config)
    regime_df = detector.detect(
        prices=benchmark_data['close'],
        dates=benchmark_data['date']
    )

    # Add metadata
    regime_df['benchmark'] = benchmark
    regime_df['detection_method'] = detector.method

    logger.info(f"✓ Regime detection complete: {len(regime_df)} periods analyzed")

    return regime_df
