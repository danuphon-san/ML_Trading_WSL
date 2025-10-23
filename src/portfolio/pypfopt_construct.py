"""
Portfolio construction using PyPortfolioOpt

CRITICAL: Always pass real close prices (price_panel) - never reconstruct from returns
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from loguru import logger

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.exceptions import OptimizationError
    import cvxpy
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    logger.error("PyPortfolioOpt not installed - install with: pip install PyPortfolioOpt==1.5.5")


class PyPortfolioOptimizer:
    """Portfolio optimization using PyPortfolioOpt"""

    def __init__(self, config: Dict):
        """
        Initialize optimizer

        Args:
            config: Configuration dictionary
        """
        if not PYPFOPT_AVAILABLE:
            raise ImportError("PyPortfolioOpt not available")

        self.config = config
        self.pypfopt_config = config.get('portfolio', {}).get('pypfopt', {})

        self.mu_mapping = self.pypfopt_config.get('mu_mapping', 'rank_to_mu')
        self.mu_range = self.pypfopt_config.get('mu_range', [0.0, 0.20])
        self.cov_lookback_days = self.pypfopt_config.get('cov_lookback_days', 252)
        self.cov_method = self.pypfopt_config.get('cov_method', 'sample')
        self.objective = self.pypfopt_config.get('objective', 'max_sharpe')
        self.risk_free_rate = self.pypfopt_config.get('risk_free_rate', 0.02)
        self.l2_reg = self.pypfopt_config.get('l2_reg', 0.01)
        self.min_weight = self.pypfopt_config.get('min_weight', 0.01)
        self.max_weight = self.pypfopt_config.get('max_weight', 0.15)

        logger.info(f"Initialized PyPortfolioOptimizer: objective={self.objective}, "
                   f"mu_mapping={self.mu_mapping}")

    def optimize(
        self,
        selected_df: pd.DataFrame,
        price_panel: pd.DataFrame,
        score_col: str = 'ml_score'
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights

        Args:
            selected_df: DataFrame with selected stocks and ML scores [symbol, ml_score, ...]
            price_panel: DataFrame with historical prices [date, symbol, close]
                        CRITICAL: Must be real close prices, not returns!
            score_col: Column name for ML scores

        Returns:
            Dictionary of {symbol: weight}
        """
        logger.info(f"Optimizing portfolio for {len(selected_df)} stocks")

        # Get symbols
        symbols = selected_df['symbol'].tolist()

        # Map ML scores to expected returns (mu)
        mu = self._map_scores_to_mu(selected_df, score_col)

        # Estimate covariance from price panel
        S = self._estimate_covariance(price_panel, symbols)

        # Check validity
        if len(symbols) < 2:
            logger.warning("Less than 2 symbols, using equal weights")
            return {s: 1.0 / len(symbols) for s in symbols}

        # Run optimization
        try:
            weights = self._run_optimization(mu, S, symbols)
        except (OptimizationError, ValueError, cvxpy.error.SolverError) as e:
            logger.warning(f"Optimization failed: {e}, trying with relaxed constraints")
            # Try again with relaxed weight bounds
            try:
                weights = self._run_optimization_relaxed(mu, S, symbols)
            except Exception as e2:
                logger.error(f"Relaxed optimization also failed: {e2}, falling back to equal weights")
                weights = {s: 1.0 / len(symbols) for s in symbols}

        # Validate weights
        weights = self._validate_weights(weights)

        logger.info(f"Optimization complete: {len(weights)} positions")

        return weights

    def _map_scores_to_mu(
        self,
        selected_df: pd.DataFrame,
        score_col: str
    ) -> pd.Series:
        """
        Map ML scores to expected returns (mu)

        Args:
            selected_df: DataFrame with scores
            score_col: Score column name

        Returns:
            Series of expected returns indexed by symbol
        """
        df = selected_df.copy()

        if self.mu_mapping == 'identity':
            # Use scores directly as mu
            mu = df.set_index('symbol')[score_col]

        elif self.mu_mapping == 'sigmoid':
            # Apply sigmoid transformation
            scores = df[score_col].values
            mu_values = 1 / (1 + np.exp(-scores))
            mu_values = self.mu_range[0] + (self.mu_range[1] - self.mu_range[0]) * mu_values
            mu = pd.Series(mu_values, index=df['symbol'])

        elif self.mu_mapping == 'rank_to_mu':
            # Map ranks to mu range linearly
            df['rank'] = df[score_col].rank(ascending=False)
            n = len(df)

            # Higher rank (better score) -> higher mu
            df['mu'] = self.mu_range[1] - (df['rank'] - 1) / (n - 1) * (self.mu_range[1] - self.mu_range[0])
            mu = df.set_index('symbol')['mu']

        else:
            raise ValueError(f"Unknown mu_mapping: {self.mu_mapping}")

        logger.debug(f"Mapped scores to mu: range [{mu.min():.4f}, {mu.max():.4f}]")

        return mu

    def _estimate_covariance(
        self,
        price_panel: pd.DataFrame,
        symbols: list
    ) -> pd.DataFrame:
        """
        Estimate covariance matrix from price panel

        CRITICAL: price_panel must contain real close prices

        Args:
            price_panel: DataFrame with [date, symbol, close]
            symbols: List of symbols

        Returns:
            Covariance matrix
        """
        # Filter price panel to selected symbols
        prices = price_panel[price_panel['symbol'].isin(symbols)].copy()

        # Pivot to wide format (date x symbols)
        price_matrix = prices.pivot(index='date', columns='symbol', values='close')

        # Sort by date
        price_matrix = price_matrix.sort_index()

        # Take last N days
        if len(price_matrix) > self.cov_lookback_days:
            price_matrix = price_matrix.tail(self.cov_lookback_days)

        # Check for sufficient history
        if len(price_matrix) < 20:
            logger.warning(f"Insufficient price history: {len(price_matrix)} days")

        # Estimate covariance
        if self.cov_method == 'sample':
            S = risk_models.sample_cov(price_matrix, returns_data=False)
        elif self.cov_method == 'ledoit_wolf':
            S = risk_models.CovarianceShrinkage(price_matrix, returns_data=False).ledoit_wolf()
        elif self.cov_method == 'shrunk':
            S = risk_models.CovarianceShrinkage(price_matrix, returns_data=False).shrunk_covariance()
        else:
            raise ValueError(f"Unknown cov_method: {self.cov_method}")

        logger.debug(f"Estimated covariance from {len(price_matrix)} days of prices")

        return S

    def _run_optimization(
        self,
        mu: pd.Series,
        S: pd.DataFrame,
        symbols: list
    ) -> Dict[str, float]:
        """
        Run portfolio optimization

        Args:
            mu: Expected returns
            S: Covariance matrix
            symbols: List of symbols

        Returns:
            Dictionary of weights
        """
        # Align mu and S
        common_symbols = list(set(mu.index) & set(S.index))
        mu = mu.loc[common_symbols]
        S = S.loc[common_symbols, common_symbols]

        # Initialize efficient frontier
        ef = EfficientFrontier(mu, S, weight_bounds=(self.min_weight, self.max_weight))

        # Optimize based on objective
        if self.objective == 'max_sharpe':
            # max_sharpe doesn't work well with additional objectives, use gamma_regularization instead
            raw_weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        elif self.objective == 'min_volatility':
            # Add L2 regularization for min_volatility
            if self.l2_reg > 0:
                ef.add_objective(lambda w: self.l2_reg * (w @ w))
            raw_weights = ef.min_volatility()
        elif self.objective == 'efficient_risk':
            # Add L2 regularization for efficient_risk
            if self.l2_reg > 0:
                ef.add_objective(lambda w: self.l2_reg * (w @ w))
            # Target specific risk level
            target_volatility = 0.15  # 15% annualized
            raw_weights = ef.efficient_risk(target_volatility)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # Clean weights (remove tiny positions)
        cleaned_weights = ef.clean_weights()

        return cleaned_weights

    def _run_optimization_relaxed(
        self,
        mu: pd.Series,
        S: pd.DataFrame,
        symbols: list
    ) -> Dict[str, float]:
        """
        Run portfolio optimization with relaxed constraints

        Args:
            mu: Expected returns
            S: Covariance matrix
            symbols: List of symbols

        Returns:
            Dictionary of weights
        """
        # Align mu and S
        common_symbols = list(set(mu.index) & set(S.index))
        mu = mu.loc[common_symbols]
        S = S.loc[common_symbols, common_symbols]

        # Use wider weight bounds
        relaxed_min = max(0.0, self.min_weight - 0.005)
        relaxed_max = min(1.0, self.max_weight + 0.05)

        logger.debug(f"Trying relaxed bounds: [{relaxed_min:.3f}, {relaxed_max:.3f}]")

        # Initialize efficient frontier with relaxed bounds
        ef = EfficientFrontier(mu, S, weight_bounds=(relaxed_min, relaxed_max))

        # Use simpler objective without regularization
        if self.objective == 'max_sharpe':
            raw_weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        else:
            raw_weights = ef.min_volatility()

        cleaned_weights = ef.clean_weights()

        return cleaned_weights

    def _validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and adjust weights

        Args:
            weights: Raw weights

        Returns:
            Validated weights
        """
        # Remove zero weights
        weights = {k: v for k, v in weights.items() if v > 1e-6}

        # Check sum
        total = sum(weights.values())

        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total:.4f}, normalizing to 1.0")
            weights = {k: v / total for k, v in weights.items()}

        # Check bounds
        for symbol, weight in weights.items():
            if weight < 0:
                logger.warning(f"Negative weight for {symbol}: {weight}, setting to 0")
                weights[symbol] = 0
            if weight > 1:
                logger.warning(f"Weight > 1 for {symbol}: {weight}, capping at 1.0")
                weights[symbol] = 1.0

        # Renormalize after bounds check
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


def target_weights(
    selected_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    config: Dict,
    score_col: str = 'ml_score'
) -> Dict[str, float]:
    """
    Convenience function to compute target weights

    CRITICAL: price_panel must be real close prices

    Args:
        selected_df: Selected stocks with scores
        price_panel: Historical prices
        config: Configuration
        score_col: Score column name

    Returns:
        Dictionary of target weights
    """
    optimizer = PyPortfolioOptimizer(config)
    return optimizer.optimize(selected_df, price_panel, score_col)
