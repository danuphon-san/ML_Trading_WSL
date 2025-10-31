"""
OHLCV data ingestion from various providers
"""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm

try:
    import simfin as sf
    from simfin.names import TICKER

    _SIMFIN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _SIMFIN_AVAILABLE = False

    # Lazy import warning handled when provider is used

SIMFIN_TICKER_MAPPING = {
    # Class B shares: hyphen to dot
    'BRK-B': 'BRK.B',
    'BF-B': 'BF.B',
    # Multiple share classes (try alternate class)
    'GOOGL': 'GOOG',
    'FOXA': 'FOX',
    'NWSA': 'NWS',
    # Other known mappings
    'GEV': 'GE',
}


def _simfin_ticker_variants(symbol: str) -> List[str]:
    """
    Generate ticker variants for SimFin compatibility.

    Args:
        symbol: Original ticker symbol (typically S&P500 format)

    Returns:
        Ordered list of candidate tickers to query in SimFin data.
    """
    variants = [symbol]

    mapped = SIMFIN_TICKER_MAPPING.get(symbol)
    if mapped and mapped not in variants:
        variants.append(mapped)

    if '-' in symbol:
        dot_variant = symbol.replace('-', '.')
        if dot_variant not in variants:
            variants.append(dot_variant)

    return variants


def build_provider_kwargs(provider: str, config: Dict) -> Dict:
    """
    Extract provider-specific configuration for OHLCV ingestion from config.

    Currently supports SimFin by reusing credentials defined under fundamentals.
    """
    provider_lower = provider.lower()

    if provider_lower == "simfin":
        fundamentals_cfg = config.get("fundamentals", {})
        simfin_cfg = fundamentals_cfg.get("simfin", {})

        api_key = simfin_cfg.get("api_key")
        data_dir = simfin_cfg.get("data_dir", "data/simfin_data")
        market = simfin_cfg.get("market", "us")
        variant = simfin_cfg.get("shareprice_variant", "daily")

        if not api_key or str(api_key).startswith("${") or "your-simfin-api-key" in str(api_key):
            raise ValueError(
                "SimFin provider selected but API key is not configured. "
                "Set fundamentals.simfin.api_key in config/config.yaml or via environment variable."
            )

        return {
            "api_key": api_key,
            "data_dir": data_dir,
            "market": market,
            "variant": variant or "daily",
        }

    return {}


def _normalize_date_column(series: pd.Series) -> pd.Series:
    """
    Ensure datetime series is timezone naive in UTC for consistent downstream logic.
    """
    if series.empty:
        return series

    series = pd.to_datetime(series, utc=True, errors="coerce")
    return series.dt.tz_localize(None)


class OHLCVIngester:
    """Ingest OHLCV data from providers"""

    def __init__(
        self,
        provider: str = "yfinance",
        storage_path: str = "data/parquet",
        **provider_kwargs,
    ):
        """
        Initialize OHLCV ingester

        Args:
            provider: Data provider (yfinance, alpha_vantage, polygon, etc.)
            storage_path: Path to store parquet files
            provider_kwargs: Optional provider-specific configuration parameters
                (e.g., SimFin API key, data directory, market)
        """
        self.provider = provider
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.provider_kwargs = provider_kwargs

        logger.info(f"Initialized OHLCVIngester with provider={provider}")

    def fetch_ohlcv(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        frequency: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for symbols

        Args:
            symbols: List of symbols to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), None = today
            frequency: Data frequency (1d, 1h, etc.)

        Returns:
            Dictionary of {symbol: DataFrame} with OHLCV data
        """
        provider = self.provider.lower()

        if provider == "yfinance":
            return self._fetch_yfinance(symbols, start_date, end_date, frequency)
        if provider == "simfin":
            return self._fetch_simfin(symbols, start_date, end_date, frequency)

        raise ValueError(f"Provider {self.provider} not implemented")

    def _fetch_yfinance(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str],
        frequency: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from yfinance"""
        data_dict = {}

        logger.info(f"Fetching {len(symbols)} symbols from yfinance: {start_date} to {end_date}")

        for symbol in tqdm(symbols, desc="Fetching OHLCV"):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=frequency,
                    auto_adjust=False  # Keep unadjusted prices
                )

                if df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue

                # Standardize column names
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Dividends': 'dividends',
                    'Stock Splits': 'splits'
                })

                # Add symbol column
                df['symbol'] = symbol

                # Reset index to make date a column
                df = df.reset_index()
                df = df.rename(columns={'Date': 'date', 'Datetime': 'date'})

                data_dict[symbol] = df

                logger.debug(f"Fetched {len(df)} rows for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

        logger.info(f"Successfully fetched {len(data_dict)}/{len(symbols)} symbols")
        return data_dict

    def _fetch_simfin(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str],
        frequency: str,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data from SimFin bulk share price dataset"""
        if not _SIMFIN_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError(
                "SimFin provider requested but 'simfin' package is not installed. "
                "Install with `pip install simfin`."
            )

        if frequency not in ("1d", "1D"):
            raise ValueError("SimFin provider currently supports only daily ('1d') frequency.")

        api_key = self.provider_kwargs.get("api_key") or self.provider_kwargs.get("simfin_api_key")
        data_dir = self.provider_kwargs.get("data_dir") or self.provider_kwargs.get("simfin_data_dir")
        market = self.provider_kwargs.get("market", "us")
        variant = self.provider_kwargs.get("variant", "daily")

        if not api_key:
            raise ValueError("SimFin provider requires 'api_key' in provider kwargs or configuration.")

        if not data_dir:
            data_dir = "data/simfin_data"

        data_dir_path = Path(data_dir)
        data_dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Fetching SimFin share prices for {len(symbols)} symbols ({market}, {variant})")

        sf.set_api_key(api_key)
        sf.set_data_dir(str(data_dir_path))

        df_prices = sf.load_shareprices(variant=variant, market=market)

        available_tickers = df_prices.index.get_level_values(TICKER)
        available_unique = available_tickers.unique()

        logger.info(
            "Loaded SimFin share price dataset with "
            f"{len(available_unique)} tickers"
        )

        available_set = set(available_unique.tolist())

        data_dict: Dict[str, pd.DataFrame] = {}
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date) if end_date else None

        for symbol in symbols:
            ticker_variants = _simfin_ticker_variants(symbol)
            selected_variant = None

            for candidate in ticker_variants:
                if candidate in available_set:
                    selected_variant = candidate
                    if candidate != symbol:
                        logger.debug(f"Ticker mapping applied: {symbol} -> {candidate}")
                    break

            if not selected_variant:
                logger.warning(f"No SimFin data found for {symbol} (tried {ticker_variants})")
                continue

            symbol_df = df_prices.loc[selected_variant].reset_index()

            column_mapping = {
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj. Close": "adj_close",
                "Volume": "volume",
                "Dividend": "dividends",
                "Common Shares Outstanding": "shares_outstanding",
                "Market-Cap": "market_cap",
            }

            symbol_df = symbol_df.rename(columns=column_mapping)

            symbol_df["symbol"] = symbol

            if "date" not in symbol_df.columns:
                logger.warning(f"SimFin data for {symbol} lacks 'Date' column, skipping.")
                continue

            symbol_df["date"] = _normalize_date_column(symbol_df["date"])

            if start_ts is not None:
                symbol_df = symbol_df[symbol_df["date"] >= start_ts]
            if end_ts is not None:
                symbol_df = symbol_df[symbol_df["date"] <= end_ts]

            symbol_df = symbol_df.sort_values("date")

            if "dividends" in symbol_df.columns:
                symbol_df["dividends"] = symbol_df["dividends"].fillna(0)

            data_dict[symbol] = symbol_df.reset_index(drop=True)

        logger.info(f"Successfully fetched {len(data_dict)}/{len(symbols)} symbols from SimFin")
        return data_dict

    def save_parquet(
        self,
        data_dict: Dict[str, pd.DataFrame],
        frequency: str = "1d",
        partitioned: bool = True
    ):
        """
        Save OHLCV data to parquet files

        Args:
            data_dict: Dictionary of {symbol: DataFrame}
            frequency: Data frequency for organizing storage
            partitioned: If True, partition by symbol
        """
        freq_path = self.storage_path / frequency
        freq_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(data_dict)} symbols to {freq_path}")

        for symbol, df in tqdm(data_dict.items(), desc="Saving parquet"):
            try:
                if partitioned:
                    # Save each symbol to its own file
                    symbol_path = freq_path / f"{symbol}.parquet"
                    df.to_parquet(symbol_path, index=False, compression='snappy')
                else:
                    # Append to single file (less efficient for large universes)
                    all_path = freq_path / "all_symbols.parquet"
                    if all_path.exists():
                        existing = pd.read_parquet(all_path)
                        df = pd.concat([existing, df], ignore_index=True)
                    df.to_parquet(all_path, index=False, compression='snappy')

            except Exception as e:
                logger.error(f"Failed to save {symbol}: {e}")

        logger.info(f"Saved data to {freq_path}")

    def save_symbol_parquet(
        self,
        symbol: str,
        df: pd.DataFrame,
        frequency: str = "1d"
    ):
        """
        Save single symbol OHLCV data to parquet file

        Args:
            symbol: Symbol ticker
            df: DataFrame with OHLCV data
            frequency: Data frequency for organizing storage
        """
        freq_path = self.storage_path / frequency
        freq_path.mkdir(parents=True, exist_ok=True)

        symbol_path = freq_path / f"{symbol}.parquet"
        df.to_parquet(symbol_path, index=False, compression='snappy')
        logger.debug(f"Saved {symbol} to {symbol_path}")

    def load_parquet(
        self,
        symbols: Optional[List[str]] = None,
        frequency: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from parquet files

        Args:
            symbols: List of symbols to load (None = all)
            frequency: Data frequency
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            Combined DataFrame with all symbols
        """
        freq_path = self.storage_path / frequency

        if not freq_path.exists():
            logger.error(f"Path {freq_path} does not exist")
            return pd.DataFrame()

        # Load all parquet files in frequency directory
        parquet_files = list(freq_path.glob("*.parquet"))

        if not parquet_files:
            logger.warning(f"No parquet files found in {freq_path}")
            return pd.DataFrame()

        # Filter by symbols if provided
        if symbols:
            parquet_files = [f for f in parquet_files if f.stem in symbols]

        logger.info(f"Loading {len(parquet_files)} symbols from {freq_path}")

        dfs = []
        for file in tqdm(parquet_files, desc="Loading parquet"):
            try:
                df = pd.read_parquet(file)

                if 'date' in df.columns:
                    df['date'] = _normalize_date_column(df['date'])

                # Apply date filters
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]

                dfs.append(df)

            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} rows from {len(dfs)} symbols")

        return combined

    def update_data(
        self,
        symbols: List[str],
        frequency: str = "1d"
    ):
        """
        Update existing data with latest bars

        Args:
            symbols: Symbols to update
            frequency: Data frequency
        """
        logger.info(f"Updating {len(symbols)} symbols")

        for symbol in tqdm(symbols, desc="Updating"):
            try:
                # Load existing data
                symbol_path = self.storage_path / frequency / f"{symbol}.parquet"

                if symbol_path.exists():
                    existing = pd.read_parquet(symbol_path)
                    if 'date' in existing.columns:
                        existing['date'] = _normalize_date_column(existing['date'])
                    last_date = existing['date'].max()

                    # Fetch new data from last date + 1 day
                    start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    # No existing data, fetch from default start
                    start_date = "2015-01-01"

                # Fetch new data
                new_data = self.fetch_ohlcv([symbol], start_date, None, frequency)

                if symbol in new_data and not new_data[symbol].empty:
                    if symbol_path.exists():
                        # Append to existing
                        combined = pd.concat([existing, new_data[symbol]], ignore_index=True)
                        combined = combined.drop_duplicates(subset=['date'], keep='last')
                        combined = combined.sort_values('date')
                    else:
                        combined = new_data[symbol]

                    # Save
                    self.save_parquet({symbol: combined}, frequency, partitioned=True)
                    logger.debug(f"Updated {symbol}")

            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")

        logger.info("Update complete")
