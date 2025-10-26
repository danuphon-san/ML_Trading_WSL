"""
Sleeve Allocation (Step 13)

Allocate capital across asset sleeves (equities, crypto, cash) using:
- Equal Risk Contribution (ERC) / Risk Parity
- Regularized Mean-Variance Optimization (MVO)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger


class SleeveAllocator:
    """
    Cross-sleeve portfolio allocation

    Sleeves:
        - equities: US stocks
        - crypto: Top liquid cryptos (BTC, ETH, etc.)
        - cash: Risk-free sleeve
    """

    def __init__(self, config: Dict):
        """
        Initialize sleeve allocator

        Args:
            config: Configuration dict
        """
        self.config = config
        self.method = config.get('allocation', {}).get('method', 'erc')
        self.crypto_risk_budget_max = config.get('allocation', {}).get('crypto_risk_budget_max', 0.25)
        self.rebalance_threshold = config.get('allocation', {}).get('rebalance_threshold', 0.05)

        logger.info(f"Initialized SleeveAllocator: method={self.method}, "
                   f"crypto_risk_budget_max={self.crypto_risk_budget_max}")

    def calculate_sleeve_metrics(
        self,
        equity_returns: pd.Series,
        crypto_returns: Optional[pd.Series] = None,
        lookback_days: int = 60
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate risk/return metrics for each sleeve

        Args:
            equity_returns: Series of equity portfolio returns
            crypto_returns: Optional series of crypto portfolio returns
            lookback_days: Window for metric calculation

        Returns:
            Dict of {sleeve_name: {mu, sigma, sharpe}}
        """
        metrics = {}

        # Equities
        eq_recent = equity_returns.tail(lookback_days)
        metrics['equities'] = {
            'mu': eq_recent.mean() * 252,  # Annualized return
            'sigma': eq_recent.std() * np.sqrt(252),  # Annualized volatility
            'sharpe': (eq_recent.mean() / eq_recent.std() * np.sqrt(252)) if eq_recent.std() > 0 else 0
        }

        # Crypto
        if crypto_returns is not None and len(crypto_returns) > 0:
            cr_recent = crypto_returns.tail(lookback_days)
            metrics['crypto'] = {
                'mu': cr_recent.mean() * 252,
                'sigma': cr_recent.std() * np.sqrt(252),
                'sharpe': (cr_recent.mean() / cr_recent.std() * np.sqrt(252)) if cr_recent.std() > 0 else 0
            }
        else:
            # Default crypto metrics if not available
            metrics['crypto'] = {
                'mu': 0.20,  # Assume 20% annual return
                'sigma': 0.60,  # Assume 60% annual volatility
                'sharpe': 0.33
            }

        # Cash (risk-free)
        risk_free_rate = self.config.get('portfolio', {}).get('pypfopt', {}).get('risk_free_rate', 0.02)
        metrics['cash'] = {
            'mu': risk_free_rate,
            'sigma': 0.0,  # No volatility
            'sharpe': 0.0  # Not applicable for cash
        }

        logger.info(f"Sleeve metrics calculated:")
        for sleeve, m in metrics.items():
            logger.info(f"  {sleeve}: μ={m['mu']:.2%}, σ={m['sigma']:.2%}, Sharpe={m['sharpe']:.2f}")

        return metrics

    def allocate_erc(
        self,
        metrics: Dict[str, Dict[str, float]],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Equal Risk Contribution (Risk Parity) allocation

        Each sleeve contributes equally to portfolio risk

        Args:
            metrics: Dict of sleeve metrics from calculate_sleeve_metrics()
            correlation_matrix: Optional 3x3 correlation matrix [equities, crypto, cash]
                               If None, assumes low correlation (0.3 between equities/crypto)

        Returns:
            Dict of {sleeve_name: weight}
        """
        sleeves = ['equities', 'crypto', 'cash']
        sigmas = np.array([metrics[s]['sigma'] for s in sleeves])

        # Default correlation matrix if not provided
        if correlation_matrix is None:
            correlation_matrix = np.array([
                [1.0, 0.3, 0.0],  # Equities: uncorrelated with crypto (0.3), no corr with cash
                [0.3, 1.0, 0.0],  # Crypto: uncorrelated with equities (0.3), no corr with cash
                [0.0, 0.0, 1.0]   # Cash: no correlation
            ])

        # Covariance matrix
        D = np.diag(sigmas)
        cov_matrix = D @ correlation_matrix @ D

        # ERC weights: inverse volatility contribution
        # w_i ∝ 1 / (Σ_j Cov(i,j))
        inv_risk_contrib = 1.0 / (cov_matrix.sum(axis=1) + 1e-8)
        weights_raw = inv_risk_contrib / inv_risk_contrib.sum()

        # Apply crypto risk budget constraint
        if weights_raw[1] > self.crypto_risk_budget_max:
            logger.info(f"Crypto weight {weights_raw[1]:.2%} exceeds max {self.crypto_risk_budget_max:.2%}, capping")
            weights_raw[1] = self.crypto_risk_budget_max

            # Redistribute excess to equities and cash proportionally
            excess = weights_raw[1] - self.crypto_risk_budget_max
            weights_raw[0] += excess * 0.7  # 70% to equities
            weights_raw[2] += excess * 0.3  # 30% to cash

            # Renormalize
            weights_raw = weights_raw / weights_raw.sum()

        allocation = {sleeve: float(w) for sleeve, w in zip(sleeves, weights_raw)}

        logger.info(f"ERC allocation: Equities={allocation['equities']:.2%}, "
                   f"Crypto={allocation['crypto']:.2%}, Cash={allocation['cash']:.2%}")

        return allocation

    def allocate_mvo_reg(
        self,
        metrics: Dict[str, Dict[str, float]],
        correlation_matrix: Optional[np.ndarray] = None,
        l2_reg: float = 0.01
    ) -> Dict[str, float]:
        """
        Regularized Mean-Variance Optimization

        Maximize: μ'w - λ * w'Σw - γ * ||w||²

        Args:
            metrics: Dict of sleeve metrics
            correlation_matrix: Optional correlation matrix
            l2_reg: L2 regularization parameter (prevents extreme weights)

        Returns:
            Dict of {sleeve_name: weight}
        """
        sleeves = ['equities', 'crypto', 'cash']
        mus = np.array([metrics[s]['mu'] for s in sleeves])
        sigmas = np.array([metrics[s]['sigma'] for s in sleeves])

        # Default correlation matrix
        if correlation_matrix is None:
            correlation_matrix = np.array([
                [1.0, 0.3, 0.0],
                [0.3, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])

        # Covariance matrix
        D = np.diag(sigmas)
        cov_matrix = D @ correlation_matrix @ D

        # Add L2 regularization to covariance
        cov_matrix_reg = cov_matrix + l2_reg * np.eye(len(sleeves))

        # Solve for optimal weights: w = Σ^-1 μ / (1' Σ^-1 μ)
        try:
            cov_inv = np.linalg.inv(cov_matrix_reg)
            weights_raw = cov_inv @ mus
            weights_raw = weights_raw / weights_raw.sum()

            # Clip negative weights (long-only constraint)
            weights_raw = np.maximum(weights_raw, 0)
            weights_raw = weights_raw / weights_raw.sum()

        except np.linalg.LinAlgError:
            logger.warning("MVO optimization failed, falling back to ERC")
            return self.allocate_erc(metrics, correlation_matrix)

        # Apply crypto risk budget constraint
        if weights_raw[1] > self.crypto_risk_budget_max:
            weights_raw[1] = self.crypto_risk_budget_max
            excess = 1.0 - weights_raw.sum()
            weights_raw[0] += excess * 0.7
            weights_raw[2] += excess * 0.3
            weights_raw = weights_raw / weights_raw.sum()

        allocation = {sleeve: float(w) for sleeve, w in zip(sleeves, weights_raw)}

        logger.info(f"MVO allocation: Equities={allocation['equities']:.2%}, "
                   f"Crypto={allocation['crypto']:.2%}, Cash={allocation['cash']:.2%}")

        return allocation

    def allocate(
        self,
        equity_returns: pd.Series,
        crypto_returns: Optional[pd.Series] = None,
        lookback_days: int = 60
    ) -> Dict[str, float]:
        """
        Main allocation method

        Args:
            equity_returns: Series of equity portfolio returns
            crypto_returns: Optional series of crypto returns
            lookback_days: Lookback window for metrics

        Returns:
            Dict of {sleeve_name: weight}
        """
        # Calculate sleeve metrics
        metrics = self.calculate_sleeve_metrics(equity_returns, crypto_returns, lookback_days)

        # Dispatch to allocation method
        if self.method == 'mvo_reg':
            allocation = self.allocate_mvo_reg(metrics)
        else:  # Default to ERC
            allocation = self.allocate_erc(metrics)

        # Validate allocation
        total = sum(allocation.values())
        assert abs(total - 1.0) < 1e-6, f"Allocation does not sum to 1.0: {total}"

        return allocation

    def should_rebalance(self, current_allocation: Dict[str, float], target_allocation: Dict[str, float]) -> bool:
        """
        Determine if rebalancing is needed based on drift threshold

        Args:
            current_allocation: Current sleeve weights
            target_allocation: Target sleeve weights

        Returns:
            True if rebalancing needed
        """
        max_drift = 0.0
        for sleeve in target_allocation.keys():
            current = current_allocation.get(sleeve, 0.0)
            target = target_allocation[sleeve]
            drift = abs(current - target)
            max_drift = max(max_drift, drift)

        needs_rebalance = max_drift > self.rebalance_threshold

        if needs_rebalance:
            logger.info(f"Rebalancing triggered: max drift {max_drift:.2%} > threshold {self.rebalance_threshold:.2%}")
        else:
            logger.info(f"No rebalancing needed: max drift {max_drift:.2%} ≤ threshold {self.rebalance_threshold:.2%}")

        return needs_rebalance


def run_sleeve_allocation(
    config: Dict,
    equity_curve: pd.DataFrame,
    crypto_curve: Optional[pd.DataFrame] = None
) -> Dict[str, any]:
    """
    Standalone function to run sleeve allocation

    Args:
        config: Configuration dictionary
        equity_curve: DataFrame with columns [date, equity] (equity portfolio performance)
        crypto_curve: Optional DataFrame with crypto portfolio performance

    Returns:
        Dict with allocation results and metadata
    """
    logger.info("Running sleeve allocation...")

    # Calculate returns
    equity_returns = equity_curve['equity'].pct_change().dropna()

    crypto_returns = None
    if crypto_curve is not None and len(crypto_curve) > 0:
        crypto_returns = crypto_curve['equity'].pct_change().dropna()

    # Run allocation
    allocator = SleeveAllocator(config)
    allocation = allocator.allocate(equity_returns, crypto_returns)

    # Calculate portfolio-level metrics
    lookback = 60
    eq_vol = equity_returns.tail(lookback).std() * np.sqrt(252)
    cr_vol = crypto_returns.tail(lookback).std() * np.sqrt(252) if crypto_returns is not None else 0.0

    # Estimated portfolio volatility
    portfolio_vol = allocation['equities'] * eq_vol + allocation['crypto'] * cr_vol

    result = {
        'allocation': allocation,
        'method': allocator.method,
        'date': str(equity_curve['date'].iloc[-1]),
        'metrics': {
            'equity_volatility': float(eq_vol),
            'crypto_volatility': float(cr_vol),
            'estimated_portfolio_volatility': float(portfolio_vol)
        },
        'rebalance_threshold': allocator.rebalance_threshold
    }

    logger.info(f"✓ Sleeve allocation complete: Equities={allocation['equities']:.2%}, "
               f"Crypto={allocation['crypto']:.2%}, Cash={allocation['cash']:.2%}")

    return result
