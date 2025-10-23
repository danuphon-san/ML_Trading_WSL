"""
OHLCV data ingestion from various providers
"""
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm


class OHLCVIngester:
    """Ingest OHLCV data from providers"""

    def __init__(
        self,
        provider: str = "yfinance",
        storage_path: str = "data/parquet"
    ):
        """
        Initialize OHLCV ingester

        Args:
            provider: Data provider (yfinance, alpha_vantage, polygon, etc.)
            storage_path: Path to store parquet files
        """
        self.provider = provider
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

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
        if self.provider == "yfinance":
            return self._fetch_yfinance(symbols, start_date, end_date, frequency)
        else:
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
