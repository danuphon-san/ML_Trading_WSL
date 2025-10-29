"""
Fundamental data ingestion and storage
"""
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import yfinance as yf
import requests
import time
from loguru import logger
from tqdm import tqdm

# SimFin imports (optional)
try:
    import simfin as sf
    from simfin.names import TICKER, REPORT_DATE, PUBLISH_DATE
    SIMFIN_AVAILABLE = True
except ImportError:
    SIMFIN_AVAILABLE = False
    logger.warning("SimFin package not available. Install with: pip install simfin")


class FundamentalsIngester:
    """Ingest fundamental data from providers"""
    
    # SimFin column mapping to standard schema
    SIMFIN_COLUMN_MAP = {
        # Income Statement
        'Revenue': 'revenue',
        'Net Income': 'net_income',
        'Gross Profit': 'gross_profit',
        'Operating Income (Loss)': 'operating_income',
        'Pretax Income (Loss)': 'pretax_income',
        
        # Balance Sheet
        'Total Assets': 'total_assets',
        'Total Liabilities': 'total_liabilities',
        'Total Equity': 'shareholders_equity',
        'Total Current Assets': 'current_assets',
        'Total Current Liabilities': 'current_liabilities',
        'Total Noncurrent Assets': 'noncurrent_assets',
        'Total Noncurrent Liabilities': 'noncurrent_liabilities',
        
        # Cash Flow
        'Net Cash from Operating Activities': 'operating_cash_flow',
        'Net Cash from Investing Activities': 'investing_cash_flow',
        'Net Cash from Financing Activities': 'financing_cash_flow',
        'Net Change in Cash': 'net_change_in_cash',
    }

    def __init__(
        self,
        storage_path: str = "data/fundamentals",
        provider: str = "yfinance",
        api_key: Optional[str] = None,
        simfin_data_dir: Optional[str] = None,
        simfin_market: str = "us",
        simfin_variant: str = "quarterly"
    ):
        """
        Initialize fundamentals ingester

        Args:
            storage_path: Path to store fundamental data
            provider: Data provider ('yfinance', 'alpha_vantage', or 'simfin')
            api_key: API key for provider (required for alpha_vantage and simfin)
            simfin_data_dir: Data directory for SimFin cache
            simfin_market: Market for SimFin ('us', 'de', etc.)
            simfin_variant: Variant for SimFin ('quarterly' or 'annual')
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.provider = provider.lower()
        self.api_key = api_key
        self.simfin_market = simfin_market
        self.simfin_variant = simfin_variant

        if self.provider == "alpha_vantage" and not self.api_key:
            raise ValueError("API key required for Alpha Vantage provider")
        
        if self.provider == "simfin":
            if not SIMFIN_AVAILABLE:
                raise ImportError(
                    "SimFin package not installed. Install with: pip install simfin"
                )
            if not self.api_key:
                raise ValueError("API key required for SimFin provider")
            
            # Initialize SimFin
            self._initialize_simfin(simfin_data_dir)

        logger.info(f"Initialized FundamentalsIngester with storage_path={storage_path}, provider={provider}")
    
    def _initialize_simfin(self, data_dir: Optional[str] = None):
        """Initialize SimFin API configuration"""
        try:
            # Set API key
            sf.set_api_key(self.api_key)
            
            # Set data directory for caching
            if data_dir is None:
                data_dir = "data/simfin_data"
            
            data_dir_path = Path(data_dir)
            data_dir_path.mkdir(parents=True, exist_ok=True)
            sf.set_data_dir(str(data_dir_path))
            
            logger.info(f"SimFin initialized with data_dir={data_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize SimFin: {e}")
            raise

    def _fetch_simfin_fundamentals(
        self,
        symbols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch fundamentals from SimFin (bulk loading for efficiency)
        
        Args:
            symbols: List of symbols to fetch
            
        Returns:
            Dictionary of {symbol: DataFrame} with fundamental data
        """
        try:
            logger.info(f"Loading SimFin data for {len(symbols)} symbols...")
            
            # Load all datasets at once (SimFin caches these locally)
            logger.info("Loading income statements...")
            df_income = sf.load_income(variant=self.simfin_variant, market=self.simfin_market)
            
            logger.info("Loading balance sheets...")
            df_balance = sf.load_balance(variant=self.simfin_variant, market=self.simfin_market)
            
            logger.info("Loading cash flow statements...")
            df_cashflow = sf.load_cashflow(variant=self.simfin_variant, market=self.simfin_market)
            
            # Try to load derived metrics (optional)
            try:
                logger.info("Loading derived metrics...")
                df_derived = sf.load_derived(variant=self.simfin_variant, market=self.simfin_market)
            except Exception as e:
                logger.warning(f"Could not load derived metrics: {e}")
                df_derived = None
            
            # Filter to requested symbols
            df_income = df_income[df_income.index.get_level_values(TICKER).isin(symbols)]
            df_balance = df_balance[df_balance.index.get_level_values(TICKER).isin(symbols)]
            df_cashflow = df_cashflow[df_cashflow.index.get_level_values(TICKER).isin(symbols)]
            if df_derived is not None:
                df_derived = df_derived[df_derived.index.get_level_values(TICKER).isin(symbols)]
            
            # Process each symbol
            data_dict = {}
            for symbol in tqdm(symbols, desc="Processing SimFin data"):
                try:
                    df = self._merge_simfin_datasets(
                        symbol, df_income, df_balance, df_cashflow, df_derived
                    )
                    if df is not None and not df.empty:
                        data_dict[symbol] = df
                        logger.debug(f"Processed {len(df)} periods for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Failed to fetch SimFin data: {e}")
            raise
    
    def _merge_simfin_datasets(
        self,
        symbol: str,
        df_income: pd.DataFrame,
        df_balance: pd.DataFrame,
        df_cashflow: pd.DataFrame,
        df_derived: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """
        Merge SimFin datasets for a single symbol into standardized format
        
        Args:
            symbol: Stock ticker
            df_income: Income statement data
            df_balance: Balance sheet data
            df_cashflow: Cash flow data
            df_derived: Derived metrics (optional)
            
        Returns:
            Merged DataFrame in standard schema
        """
        try:
            # Extract data for this symbol
            if symbol not in df_income.index.get_level_values(TICKER):
                logger.warning(f"No income data for {symbol}")
                return None
            
            income = df_income.loc[symbol].copy()
            balance = df_balance.loc[symbol].copy() if symbol in df_balance.index.get_level_values(TICKER) else pd.DataFrame()
            cashflow = df_cashflow.loc[symbol].copy() if symbol in df_cashflow.index.get_level_values(TICKER) else pd.DataFrame()
            derived = df_derived.loc[symbol].copy() if df_derived is not None and symbol in df_derived.index.get_level_values(TICKER) else pd.DataFrame()
            
            # Reset index to get dates as column
            income = income.reset_index()
            balance = balance.reset_index() if not balance.empty else balance
            cashflow = cashflow.reset_index() if not cashflow.empty else cashflow
            derived = derived.reset_index() if not derived.empty else derived
            
            # Merge on Report Date
            merged = income.copy()
            if not balance.empty:
                merged = merged.merge(balance, on=REPORT_DATE, how='outer', suffixes=('', '_balance'))
            if not cashflow.empty:
                merged = merged.merge(cashflow, on=REPORT_DATE, how='outer', suffixes=('', '_cashflow'))
            if not derived.empty:
                merged = merged.merge(derived, on=REPORT_DATE, how='outer', suffixes=('', '_derived'))
            
            # Standardize column names
            merged = self._map_simfin_columns(merged)
            
            # Add symbol column
            merged['symbol'] = symbol
            
            # Rename date columns
            if REPORT_DATE in merged.columns:
                merged['date'] = pd.to_datetime(merged[REPORT_DATE])
            
            # Handle publish date
            if PUBLISH_DATE in merged.columns:
                merged['public_date'] = pd.to_datetime(merged[PUBLISH_DATE])
            else:
                # Estimate: 45 days after report date
                merged['public_date'] = merged['date'] + pd.Timedelta(days=45)
            
            # Add period type
            merged['period_type'] = self.simfin_variant
            
            # Calculate derived metrics if not present
            merged = self._calculate_derived_metrics(merged)
            
            # Select and order columns
            standard_cols = [
                'symbol', 'date', 'public_date', 'period_type',
                'revenue', 'net_income', 'gross_profit', 'operating_income',
                'total_assets', 'total_liabilities', 'shareholders_equity',
                'current_assets', 'current_liabilities',
                'operating_cash_flow', 'investing_cash_flow', 'financing_cash_flow',
                'current_ratio', 'debt_to_equity', 'roe', 'roa',
                'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'quick_ratio'
            ]
            
            # Keep only columns that exist
            available_cols = [col for col in standard_cols if col in merged.columns]
            merged = merged[available_cols]
            
            # Sort by date
            merged = merged.sort_values('date')
            
            return merged
            
        except Exception as e:
            logger.error(f"Failed to merge datasets for {symbol}: {e}")
            return None
    
    def _map_simfin_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map SimFin column names to standard schema
        
        Args:
            df: DataFrame with SimFin columns
            
        Returns:
            DataFrame with standardized column names
        """
        # Rename columns based on mapping
        df = df.rename(columns=self.SIMFIN_COLUMN_MAP)
        return df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived financial metrics
        
        Args:
            df: DataFrame with fundamental data
            
        Returns:
            DataFrame with calculated metrics
        """
        # Current ratio
        if 'current_ratio' not in df.columns or df['current_ratio'].isna().all():
            if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
                df['current_ratio'] = df['current_assets'] / df['current_liabilities']
                df['current_ratio'] = df['current_ratio'].replace([float('inf'), -float('inf')], None)
        
        # Debt to equity
        if 'debt_to_equity' not in df.columns or df['debt_to_equity'].isna().all():
            if 'total_liabilities' in df.columns and 'shareholders_equity' in df.columns:
                df['debt_to_equity'] = df['total_liabilities'] / df['shareholders_equity']
                df['debt_to_equity'] = df['debt_to_equity'].replace([float('inf'), -float('inf')], None)
        
        # ROE (Return on Equity)
        if 'roe' not in df.columns or df['roe'].isna().all():
            if 'net_income' in df.columns and 'shareholders_equity' in df.columns:
                df['roe'] = df['net_income'] / df['shareholders_equity']
                df['roe'] = df['roe'].replace([float('inf'), -float('inf')], None)
        
        # ROA (Return on Assets)
        if 'roa' not in df.columns or df['roa'].isna().all():
            if 'net_income' in df.columns and 'total_assets' in df.columns:
                df['roa'] = df['net_income'] / df['total_assets']
                df['roa'] = df['roa'].replace([float('inf'), -float('inf')], None)
        
        return df

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

        # Use SimFin bulk loading if provider is simfin
        if self.provider == "simfin":
            return self._fetch_simfin_fundamentals(symbols)
        
        # For other providers, fetch symbol by symbol
        data_dict = {}
        logger.info(f"Fetching fundamentals for {len(symbols)} symbols")

        for symbol in tqdm(symbols, desc="Fetching fundamentals"):
            try:
                if self.provider == "alpha_vantage":
                    df = self._fetch_alpha_vantage_fundamentals(symbol, metrics)
                else:
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

    def _fetch_alpha_vantage_fundamentals(
        self,
        symbol: str,
        metrics: List[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch fundamentals for single symbol from Alpha Vantage"""
        try:
            base_url = "https://www.alphavantage.co/query"

            # Fetch Income Statement
            params = {
                'function': 'INCOME_STATEMENT',
                'symbol': symbol,
                'apikey': self.api_key
            }
            response = requests.get(base_url, params=params)
            income_data = response.json()

            if 'Note' in income_data or 'Error Message' in income_data:
                logger.warning(f"API limit or error for {symbol}: {income_data}")
                return None

            time.sleep(0.3)  # Rate limit: ~5 calls per minute for free tier

            # Fetch Balance Sheet
            params['function'] = 'BALANCE_SHEET'
            response = requests.get(base_url, params=params)
            balance_data = response.json()

            time.sleep(0.3)

            # Fetch Cash Flow
            params['function'] = 'CASH_FLOW'
            response = requests.get(base_url, params=params)
            cashflow_data = response.json()

            time.sleep(0.3)

            # Fetch Overview for ratios
            params['function'] = 'OVERVIEW'
            response = requests.get(base_url, params=params)
            overview_data = response.json()

            if 'quarterlyReports' not in income_data:
                logger.warning(f"No quarterly reports for {symbol}")
                return None

            # Build dataframe from quarterly reports
            records = []
            quarterly_income = income_data.get('quarterlyReports', [])
            quarterly_balance = {r['fiscalDateEnding']: r for r in balance_data.get('quarterlyReports', [])}
            quarterly_cashflow = {r['fiscalDateEnding']: r for r in cashflow_data.get('quarterlyReports', [])}

            for income_report in quarterly_income:
                date_str = income_report['fiscalDateEnding']

                record = {
                    'symbol': symbol,
                    'date': pd.to_datetime(date_str),
                    'period_type': 'quarterly'
                }

                # Extract from overview (current ratios)
                record['market_cap'] = self._safe_float(overview_data.get('MarketCapitalization'))
                record['pe_ratio'] = self._safe_float(overview_data.get('PERatio'))
                record['pb_ratio'] = self._safe_float(overview_data.get('PriceToBookRatio'))
                record['ps_ratio'] = self._safe_float(overview_data.get('PriceToSalesRatioTTM'))
                record['ev_ebitda'] = self._safe_float(overview_data.get('EVToEBITDA'))
                record['roe'] = self._safe_float(overview_data.get('ReturnOnEquityTTM'))
                record['roa'] = self._safe_float(overview_data.get('ReturnOnAssetsTTM'))

                # Extract from income statement
                record['revenue'] = self._safe_float(income_report.get('totalRevenue'))
                record['net_income'] = self._safe_float(income_report.get('netIncome'))
                record['ebitda'] = self._safe_float(income_report.get('ebitda'))
                record['gross_profit'] = self._safe_float(income_report.get('grossProfit'))

                # Extract from balance sheet
                if date_str in quarterly_balance:
                    balance = quarterly_balance[date_str]
                    record['total_assets'] = self._safe_float(balance.get('totalAssets'))
                    record['total_liabilities'] = self._safe_float(balance.get('totalLiabilities'))
                    record['shareholders_equity'] = self._safe_float(balance.get('totalShareholderEquity'))
                    record['current_assets'] = self._safe_float(balance.get('totalCurrentAssets'))
                    record['current_liabilities'] = self._safe_float(balance.get('totalCurrentLiabilities'))
                    record['long_term_debt'] = self._safe_float(balance.get('longTermDebt'))
                    record['short_term_debt'] = self._safe_float(balance.get('shortTermDebt'))

                    # Calculate ratios from balance sheet
                    if record['current_liabilities'] and record['current_liabilities'] != 0:
                        record['current_ratio'] = record['current_assets'] / record['current_liabilities']
                    else:
                        record['current_ratio'] = None

                    if record['shareholders_equity'] and record['shareholders_equity'] != 0:
                        total_debt = (record.get('long_term_debt') or 0) + (record.get('short_term_debt') or 0)
                        record['debt_to_equity'] = total_debt / record['shareholders_equity']
                    else:
                        record['debt_to_equity'] = None
                else:
                    record['total_assets'] = None
                    record['total_liabilities'] = None
                    record['shareholders_equity'] = None
                    record['current_ratio'] = None
                    record['debt_to_equity'] = None

                # Extract from cash flow
                if date_str in quarterly_cashflow:
                    cashflow = quarterly_cashflow[date_str]
                    record['operating_cash_flow'] = self._safe_float(cashflow.get('operatingCashflow'))
                    record['capital_expenditure'] = self._safe_float(cashflow.get('capitalExpenditures'))
                else:
                    record['operating_cash_flow'] = None
                    record['capital_expenditure'] = None

                # Quick ratio (if available)
                record['quick_ratio'] = None  # Alpha Vantage doesn't provide this directly

                # Add publication date (estimate: 45 days after quarter end)
                record['public_date'] = record['date'] + pd.Timedelta(days=45)

                records.append(record)

            df = pd.DataFrame(records)
            df = df.sort_values('date')

            logger.info(f"Fetched {len(df)} quarters for {symbol} (from {df['date'].min()} to {df['date'].max()})")

            return df

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {type(e).__name__}: {str(e)}")
            return None

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or value == 'None' or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
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
                if self.provider == "alpha_vantage":
                    new_data = self._fetch_alpha_vantage_fundamentals(symbol, [])
                else:
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
