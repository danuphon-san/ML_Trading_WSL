"""
Universe selection and filtering
"""
import re
from pathlib import Path
from typing import List, Dict, Optional
from io import StringIO
import pandas as pd
import requests
import yaml
from loguru import logger


class UniverseSelector:
    """Select and filter stock universe based on rules"""

    def __init__(self, config_path: str = "config/universe.yaml"):
        """
        Initialize universe selector

        Args:
            config_path: Path to universe configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loaded universe config from {config_path}")

    def select_universe(
        self,
        candidates: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None
    ) -> List[str]:
        """
        Select universe from candidate symbols

        Args:
            candidates: DataFrame with columns [symbol, price, volume, market_cap, exchange, sector, industry, etc.]
            reference_date: Date for universe selection (default: today)

        Returns:
            List of selected symbols
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now()

        logger.info(f"Selecting universe from {len(candidates)} candidates as of {reference_date}")

        # Apply filters
        df = candidates.copy()
        df = self._apply_basic_filters(df)
        df = self._apply_exchange_filters(df)
        df = self._apply_sector_filters(df)
        df = self._apply_security_type_filters(df)
        df = self._apply_quality_filters(df)
        df = self._apply_exclusions(df)

        # Rank and select top N
        symbols = self._rank_and_select(df)

        logger.info(f"Selected {len(symbols)} symbols for universe")
        return symbols

    def _apply_basic_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply price, volume, and market cap filters"""
        cfg = self.config['filters']

        initial_count = len(df)

        if 'min_price' in cfg and cfg['min_price']:
            df = df[df['price'] >= cfg['min_price']]

        if 'max_price' in cfg and cfg['max_price']:
            df = df[df['price'] <= cfg['max_price']]

        if 'min_volume' in cfg and cfg['min_volume']:
            df = df[df['volume'] >= cfg['min_volume']]

        if 'min_market_cap' in cfg and cfg['min_market_cap']:
            df = df[df['market_cap'] >= cfg['min_market_cap']]

        logger.debug(f"Basic filters: {initial_count} -> {len(df)} symbols")
        return df

    def _apply_exchange_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply exchange filters"""
        cfg = self.config['exchanges']

        if cfg.get('include'):
            df = df[df['exchange'].isin(cfg['include'])]

        if cfg.get('exclude'):
            df = df[~df['exchange'].isin(cfg['exclude'])]

        return df

    def _apply_sector_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sector filters"""
        cfg = self.config['sectors']

        if cfg.get('exclude'):
            df = df[~df['sector'].isin(cfg['exclude'])]

        if cfg.get('include'):
            df = df[df['sector'].isin(cfg['include'])]

        return df

    def _apply_security_type_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply security type filters"""
        cfg = self.config['security_types']

        if 'security_type' not in df.columns:
            logger.warning("security_type column not found, skipping security type filters")
            return df

        if cfg.get('include'):
            df = df[df['security_type'].isin(cfg['include'])]

        if cfg.get('exclude'):
            df = df[~df['security_type'].isin(cfg['exclude'])]

        return df

    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters"""
        cfg = self.config['quality']

        # Exclude penny stocks
        if cfg.get('exclude_penny_stocks'):
            df = df[df['price'] >= 5.0]

        # Require fundamentals
        if cfg.get('require_fundamentals'):
            if 'has_fundamentals' in df.columns:
                df = df[df['has_fundamentals'] == True]

        return df

    def _apply_exclusions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply manual exclusions"""
        cfg = self.config['exclusions']

        # Exclude specific symbols
        if cfg.get('symbols'):
            df = df[~df['symbol'].isin(cfg['symbols'])]

        # Exclude by regex patterns
        if cfg.get('patterns'):
            for pattern in cfg['patterns']:
                mask = df['symbol'].str.match(pattern, na=False)
                df = df[~mask]

        return df

    def _rank_and_select(self, df: pd.DataFrame) -> List[str]:
        """Rank symbols and select top N"""
        cfg = self.config['size']

        method = cfg.get('method', 'market_cap')
        max_symbols = cfg.get('max_symbols', 500)

        if method == 'market_cap':
            df = df.sort_values('market_cap', ascending=False)
        elif method == 'liquidity':
            # Use average dollar volume as proxy
            if 'dollar_volume' in df.columns:
                df = df.sort_values('dollar_volume', ascending=False)
            else:
                df = df.sort_values('volume', ascending=False)
        elif method == 'equal_weight':
            # Random selection (or could be alphabetical)
            df = df.sample(frac=1.0, random_state=42)

        # Select top N
        symbols = df.head(max_symbols)['symbol'].tolist()

        return symbols


def load_sp500_constituents() -> List[str]:
    """
    Load S&P 500 constituent symbols from Wikipedia

    WARNING - SURVIVORSHIP BIAS:
    This function loads CURRENT S&P 500 constituents only.
    It does NOT include:
    - Companies that were delisted
    - Companies that were removed from the index
    - Companies that went bankrupt

    This creates SURVIVORSHIP BIAS in backtesting - your model will
    appear to perform better than it would in reality because failed
    companies are excluded.

    MITIGATION STRATEGIES:
    1. Use historical constituent data (if available)
    2. Include delisted stocks in your universe
    3. Use a broader universe (e.g., Russell 3000)
    4. Clearly document this limitation in results
    5. Apply a "reality discount" to backtest returns (e.g., reduce by 1-2% annually)

    Returns:
        List of CURRENT S&P 500 symbols (survivorship-biased)
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        symbols = df['Symbol'].str.replace('.', '-').tolist()

        logger.warning(
            f"Loaded {len(symbols)} CURRENT S&P 500 symbols - "
            "WARNING: This creates SURVIVORSHIP BIAS! "
            "Consider using historical constituent data or broader universe."
        )

        return symbols
    except Exception as e:
        logger.error(f"Failed to load S&P 500 constituents: {e}")
        return []


def create_candidate_dataframe(symbols: List[str]) -> pd.DataFrame:
    """
    Create candidate dataframe with basic info
    This is a placeholder - in production, fetch from data provider

    Args:
        symbols: List of symbols

    Returns:
        DataFrame with candidate information
    """
    # Placeholder implementation
    # In production, fetch real data from yfinance, alpha_vantage, etc.

    data = []
    for symbol in symbols:
        data.append({
            'symbol': symbol,
            'price': 100.0,  # Placeholder
            'volume': 1000000,  # Placeholder
            'market_cap': 1e9,  # Placeholder
            'exchange': 'NASDAQ',  # Placeholder
            'sector': 'Technology',  # Placeholder
            'industry': 'Software',  # Placeholder
            'security_type': 'Common Stock',
            'has_fundamentals': True
        })

    return pd.DataFrame(data)
