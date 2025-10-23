"""
Live operations - daily portfolio management
"""
import pandas as pd
from typing import Dict
from datetime import datetime
from loguru import logger


class LivePortfolioManager:
    """Manage live portfolio operations"""

    def __init__(self, config: Dict):
        """Initialize live manager"""
        self.config = config
        self.live_config = config.get('live', {})
        self.enabled = self.live_config.get('enabled', False)
        self.dry_run = self.live_config.get('dry_run', True)

        logger.info(f"LivePortfolioManager: enabled={self.enabled}, dry_run={self.dry_run}")

    def generate_daily_signals(
        self,
        model,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate ML scores for today

        Args:
            model: Trained ML model
            features_df: DataFrame with today's features

        Returns:
            DataFrame with scores
        """
        logger.info(f"Generating signals for {len(features_df)} symbols")

        # Predict scores
        X = features_df.drop(columns=['date', 'symbol'], errors='ignore')
        scores = model.predict(X)

        result = features_df[['date', 'symbol']].copy()
        result['ml_score'] = scores

        return result

    def calculate_target_portfolio(
        self,
        scores_df: pd.DataFrame,
        price_panel: pd.DataFrame,
        config: Dict
    ) -> Dict[str, float]:
        """
        Calculate target portfolio weights

        Args:
            scores_df: ML scores
            price_panel: Historical prices
            config: Configuration

        Returns:
            Target weights
        """
        from src.portfolio.construct import construct_portfolio

        weights = construct_portfolio(scores_df, price_panel, config)

        logger.info(f"Target portfolio: {len(weights)} positions")

        return weights

    def generate_trades(
        self,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        capital: float
    ) -> List[Dict]:
        """
        Generate trade orders

        Args:
            current_positions: Current holdings {symbol: shares}
            target_weights: Target weights {symbol: weight}
            current_prices: Current prices {symbol: price}
            capital: Portfolio capital

        Returns:
            List of trade orders
        """
        trades = []

        all_symbols = set(current_positions.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current_shares = current_positions.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            price = current_prices.get(symbol, 0)

            if price == 0:
                continue

            target_value = capital * target_weight
            target_shares = target_value / price

            trade_shares = target_shares - current_shares

            if abs(trade_shares) > 0.01:
                trades.append({
                    'symbol': symbol,
                    'shares': trade_shares,
                    'side': 'buy' if trade_shares > 0 else 'sell',
                    'price': price,
                    'timestamp': datetime.now()
                })

        logger.info(f"Generated {len(trades)} trade orders")

        return trades

    def execute_trades(self, trades: List[Dict]):
        """
        Execute trades (dry run or live)

        Args:
            trades: List of trade orders
        """
        if self.dry_run:
            logger.info(f"DRY RUN: Would execute {len(trades)} trades")
            for trade in trades:
                logger.info(f"  {trade['side'].upper()} {abs(trade['shares']):.2f} {trade['symbol']} @ ${trade['price']:.2f}")
        else:
            logger.info(f"LIVE: Executing {len(trades)} trades")
            # Implement broker integration here
            pass
