"""
Fundamental data ingestion and storage
"""
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm


class FundamentalsIngester:
    """Ingest fundamental data from providers"""

    def __init__(self, storage_path: str = "data/fundamentals"):
        """
        Initialize fundamentals ingester

        Args:
            storage_path: Path to store fundamental data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized FundamentalsIngester with storage_path={storage_path}")

    def fetch_fundamentals(
        self,
        symbols: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch fundamental data for symbols

        Args:
            symbols: List of symbols
            metrics: List of metrics to fetch (None = all available)

        Returns:
            Dictionary of {symbol: DataFrame} with fundamental data
        """
        if metrics is None:
            metrics = [
                'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda',
                'roe', 'roa', 'debt_to_equity', 'current_ratio',
                'quick_ratio', 'revenue', 'net_income', 'total_assets',
                'total_liabilities', 'shareholders_equity', 'operating_cash_flow'
            ]

        data_dict = {}

        logger.info(f"Fetching fundamentals for {len(symbols)} symbols")

        for symbol in tqdm(symbols, desc="Fetching fundamentals"):
            try:
                df = self._fetch_symbol_fundamentals(symbol, metrics)
                if df is not None and not df.empty:
                    data_dict[symbol] = df
                    logger.debug(f"Fetched {len(df)} quarters for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")

        logger.info(f"Successfully fetched {len(data_dict)}/{len(symbols)} symbols")
        return data_dict

    def _fetch_symbol_fundamentals(
        self,
        symbol: str,
        metrics: List[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch fundamentals for single symbol from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get quarterly financials
            quarterly_income = ticker.quarterly_income_stmt
            quarterly_balance = ticker.quarterly_balance_sheet
            quarterly_cashflow = ticker.quarterly_cashflow

            if quarterly_income.empty:
                logger.warning(f"No fundamental data for {symbol}")
                return None

            # Construct dataframe with quarterly data
            dates = quarterly_income.columns
            records = []

            for date in dates:
                record = {
                    'symbol': symbol,
                    'date': pd.to_datetime(date),
                    'period_type': 'quarterly'
                }

                # Extract metrics from info dict
                record['market_cap'] = info.get('marketCap', None)
                record['pe_ratio'] = info.get('trailingPE', None)
                record['pb_ratio'] = info.get('priceToBook', None)
                record['ps_ratio'] = info.get('priceToSalesTrailing12Months', None)
                record['ev_ebitda'] = info.get('enterpriseToEbitda', None)
                record['roe'] = info.get('returnOnEquity', None)
                record['roa'] = info.get('returnOnAssets', None)
                record['current_ratio'] = info.get('currentRatio', None)
                record['quick_ratio'] = info.get('quickRatio', None)
                record['debt_to_equity'] = info.get('debtToEquity', None)

                # Extract from financials (with safe access)
                if 'Total Revenue' in quarterly_income.index and date in quarterly_income.columns:
                    try:
                        record['revenue'] = quarterly_income.loc['Total Revenue', date]
                    except Exception:
                        record['revenue'] = None
                else:
                    record['revenue'] = None

                if 'Net Income' in quarterly_income.index and date in quarterly_income.columns:
                    try:
                        record['net_income'] = quarterly_income.loc['Net Income', date]
                    except Exception:
                        record['net_income'] = None
                else:
                    record['net_income'] = None

                if 'Total Assets' in quarterly_balance.index and date in quarterly_balance.columns:
                    try:
                        record['total_assets'] = quarterly_balance.loc['Total Assets', date]
                    except Exception:
                        record['total_assets'] = None
                else:
                    record['total_assets'] = None

                if 'Total Liabilities Net Minority Interest' in quarterly_balance.index and date in quarterly_balance.columns:
                    try:
                        record['total_liabilities'] = quarterly_balance.loc['Total Liabilities Net Minority Interest', date]
                    except Exception:
                        record['total_liabilities'] = None
                else:
                    record['total_liabilities'] = None

                if 'Stockholders Equity' in quarterly_balance.index and date in quarterly_balance.columns:
                    try:
                        record['shareholders_equity'] = quarterly_balance.loc['Stockholders Equity', date]
                    except Exception:
                        record['shareholders_equity'] = None
                else:
                    record['shareholders_equity'] = None

                if 'Operating Cash Flow' in quarterly_cashflow.index and date in quarterly_cashflow.columns:
                    try:
                        record['operating_cash_flow'] = quarterly_cashflow.loc['Operating Cash Flow', date]
                    except Exception:
                        record['operating_cash_flow'] = None
                else:
                    record['operating_cash_flow'] = None

                # Add publication date (estimate: 45 days after quarter end)
                record['public_date'] = record['date'] + pd.Timedelta(days=45)

                records.append(record)

            df = pd.DataFrame(records)
            df = df.sort_values('date')

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {type(e).__name__}: {str(e)}")
            return None

    def save_parquet(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Save fundamental data to parquet files

        Args:
            data_dict: Dictionary of {symbol: DataFrame}
        """
        logger.info(f"Saving {len(data_dict)} symbols to {self.storage_path}")

        for symbol, df in tqdm(data_dict.items(), desc="Saving fundamentals"):
            try:
                symbol_path = self.storage_path / f"{symbol}.parquet"
                df.to_parquet(symbol_path, index=False, compression='snappy')
            except Exception as e:
                logger.error(f"Failed to save {symbol}: {e}")

        logger.info(f"Saved fundamental data to {self.storage_path}")

    def load_parquet(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load fundamental data from parquet files

        Args:
            symbols: List of symbols (None = all)
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            Combined DataFrame
        """
        if not self.storage_path.exists():
            logger.error(f"Path {self.storage_path} does not exist")
            return pd.DataFrame()

        parquet_files = list(self.storage_path.glob("*.parquet"))

        if not parquet_files:
            logger.warning(f"No parquet files found in {self.storage_path}")
            return pd.DataFrame()

        # Filter by symbols
        if symbols:
            parquet_files = [f for f in parquet_files if f.stem in symbols]

        logger.info(f"Loading {len(parquet_files)} symbols from {self.storage_path}")

        dfs = []
        for file in tqdm(parquet_files, desc="Loading fundamentals"):
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

    def update_data(self, symbols: List[str]):
        """
        Update fundamental data with latest quarters

        Args:
            symbols: Symbols to update
        """
        logger.info(f"Updating fundamentals for {len(symbols)} symbols")

        for symbol in tqdm(symbols, desc="Updating fundamentals"):
            try:
                # Fetch new data
                new_data = self._fetch_symbol_fundamentals(symbol, [])

                if new_data is not None and not new_data.empty:
                    symbol_path = self.storage_path / f"{symbol}.parquet"

                    if symbol_path.exists():
                        # Load existing and merge
                        existing = pd.read_parquet(symbol_path)
                        combined = pd.concat([existing, new_data], ignore_index=True)
                        combined = combined.drop_duplicates(subset=['date'], keep='last')
                        combined = combined.sort_values('date')
                    else:
                        combined = new_data

                    # Save
                    combined.to_parquet(symbol_path, index=False, compression='snappy')
                    logger.debug(f"Updated {symbol}")

            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")

        logger.info("Fundamental update complete")
