"""
Live operations - daily portfolio management
"""
import pandas as pd
from typing import Dict, List
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

    def load_latest_model(self, model_dir: str = "production"):
        """
        Load champion model from production folder

        Args:
            model_dir: Directory containing model (default: production)

        Returns:
            Loaded model object
        """
        from pathlib import Path
        import joblib

        # Try production folder first
        model_path = Path(model_dir) / "champion_model.pkl"

        if not model_path.exists():
            # Fallback to latest training
            logger.warning(f"Champion model not found at {model_path}, using latest training")
            model_path = Path("data/models/latest/model.pkl")

        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")

        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # Store model for later use
        self.model = model

        logger.info(f"✓ Model loaded successfully: {type(model).__name__}")
        return model

    def get_current_positions(self, positions_file: str = None) -> Dict[str, float]:
        """
        Load current portfolio positions from file

        Args:
            positions_file: Path to positions CSV (default: live/current_positions.csv)

        Returns:
            Dict of {symbol: shares}
        """
        from pathlib import Path

        if positions_file is None:
            positions_file = "live/current_positions.csv"

        positions_path = Path(positions_file)

        if not positions_path.exists():
            logger.warning(f"No current positions file found at {positions_file}, assuming empty portfolio")
            return {}

        import pandas as pd
        df = pd.read_csv(positions_path)

        if 'symbol' not in df.columns or 'shares' not in df.columns:
            logger.error(f"Invalid positions file format (need 'symbol' and 'shares' columns)")
            return {}

        positions = dict(zip(df['symbol'], df['shares']))
        logger.info(f"Loaded {len(positions)} positions from {positions_file}")

        return positions

    def save_positions(
        self,
        positions: Dict[str, float],
        filepath: str = "live/current_positions.csv"
    ):
        """
        Save current positions to file

        Args:
            positions: Dict of {symbol: shares}
            filepath: Output file path
        """
        import pandas as pd
        from pathlib import Path

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([
            {'symbol': symbol, 'shares': shares}
            for symbol, shares in positions.items()
            if shares != 0  # Only save non-zero positions
        ])

        df = df.sort_values('symbol')
        df.to_csv(filepath, index=False)

        logger.info(f"✓ Saved {len(positions)} positions to {filepath}")

    def get_portfolio_value(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> float:
        """
        Calculate total portfolio value

        Args:
            positions: Dict of {symbol: shares}
            prices: Dict of {symbol: price}

        Returns:
            Total portfolio value in USD
        """
        total_value = 0.0
        missing_prices = []

        for symbol, shares in positions.items():
            price = prices.get(symbol, 0)

            if price <= 0:
                missing_prices.append(symbol)
                logger.warning(f"Missing or invalid price for {symbol}")
                continue

            value = shares * price
            total_value += value

        if missing_prices:
            logger.warning(f"⚠️  {len(missing_prices)} symbols missing prices: {', '.join(missing_prices)}")

        logger.info(f"Portfolio value: ${total_value:,.2f} ({len(positions)} positions)")

        return total_value

    def get_current_prices(self, symbols: List[str], date: str = None) -> Dict[str, float]:
        """
        Get current market prices for symbols

        Args:
            symbols: List of symbols
            date: Optional date (default: latest)

        Returns:
            Dict of {symbol: price}
        """
        import pandas as pd
        from pathlib import Path

        prices = {}
        parquet_dir = Path(self.config['data']['parquet']) / "1d"

        for symbol in symbols:
            symbol_file = parquet_dir / f"{symbol}.parquet"

            if not symbol_file.exists():
                logger.warning(f"No price data for {symbol}")
                continue

            try:
                df = pd.read_parquet(symbol_file)

                if date:
                    df = df[df['date'] <= date]

                if len(df) == 0:
                    continue

                latest_price = df.iloc[-1]['close']
                prices[symbol] = float(latest_price)

            except Exception as e:
                logger.error(f"Error loading price for {symbol}: {e}")

        logger.info(f"Loaded prices for {len(prices)}/{len(symbols)} symbols")

        return prices
