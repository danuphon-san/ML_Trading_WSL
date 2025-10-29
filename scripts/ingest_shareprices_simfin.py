#!/usr/bin/env python3
"""
SimFin Share Price (OHLCV) Bulk Ingestion Script

This script downloads daily share price data from SimFin for multiple symbols.
Leverages SimFin's bulk download and caching for fast data access.

Usage:
    python scripts/ingest_shareprices_simfin.py --sp500-only
    python scripts/ingest_shareprices_simfin.py --symbols AAPL MSFT GOOGL
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import argparse
import pandas as pd
from loguru import logger
from tqdm import tqdm

try:
    import simfin as sf
    from simfin.names import TICKER, DATE
    SIMFIN_AVAILABLE = True
except ImportError:
    SIMFIN_AVAILABLE = False
    logger.error("SimFin package not available. Install with: pip install simfin")
    sys.exit(1)

from src.io.universe import load_sp500_constituents


# Ticker mapping: S&P 500 format -> SimFin format
TICKER_MAPPING = {
    # Class B shares: hyphen to dot
    'BRK-B': 'BRK.B',
    'BF-B': 'BF.B',
    
    # Multiple share classes (try alternate class)
    'GOOGL': 'GOOG',  # Try Class C if Class A not found
    'FOXA': 'FOX',    # Try FOX if FOXA not found
    'NWSA': 'NWS',    # Try NWS if NWSA not found
    
    # Other known mappings
    'GEV': 'GE',      # General Electric spinoff
}


def translate_ticker_for_simfin(symbol: str) -> list:
    """
    Translate S&P 500 ticker to SimFin format(s)
    
    Returns list of potential ticker variants to try in order:
    1. Original symbol
    2. Explicit mapping (if exists)
    3. Common transformations (hyphen to dot)
    
    Args:
        symbol: Original ticker symbol
        
    Returns:
        List of ticker variants to try
    """
    variants = [symbol]  # Always try original first
    
    # Add explicit mapping if exists
    if symbol in TICKER_MAPPING:
        mapped = TICKER_MAPPING[symbol]
        if mapped not in variants:
            variants.append(mapped)
    
    # Try common transformations
    if '-' in symbol:
        dot_version = symbol.replace('-', '.')
        if dot_version not in variants:
            variants.append(dot_version)
    
    return variants


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_sp500_symbols() -> list:
    """Get S&P 500 symbols"""
    try:
        symbols = load_sp500_constituents()
        if symbols:
            logger.info(f"Loaded {len(symbols)} S&P 500 symbols")
            return symbols
        else:
            logger.warning("Could not load S&P 500, using fallback symbols")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    except Exception as e:
        logger.error(f"Error loading S&P 500: {e}")
        logger.info("Using fallback symbols")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']


def initialize_simfin(api_key: str, data_dir: str):
    """Initialize SimFin configuration"""
    try:
        sf.set_api_key(api_key)
        
        data_dir_path = Path(data_dir)
        data_dir_path.mkdir(parents=True, exist_ok=True)
        sf.set_data_dir(str(data_dir_path))
        
        logger.info(f"SimFin initialized with data_dir={data_dir}")
    except Exception as e:
        logger.error(f"Failed to initialize SimFin: {e}")
        raise


def standardize_simfin_prices(df: pd.DataFrame, symbol: str) -> tuple:
    """
    Convert SimFin price data to standard pipeline format
    
    Tries multiple ticker variants to handle format differences between
    S&P 500 listings and SimFin's ticker format.
    
    Args:
        df: SimFin DataFrame with multi-index (Ticker, Date)
        symbol: Symbol to extract (S&P 500 format)
        
    Returns:
        Tuple of (standardized DataFrame, actual_ticker_used)
    """
    try:
        # Get all ticker variants to try
        ticker_variants = translate_ticker_for_simfin(symbol)
        
        # Try each variant until we find data
        found_ticker = None
        for variant in ticker_variants:
            if variant in df.index.get_level_values(TICKER):
                found_ticker = variant
                if variant != symbol:
                    logger.info(f"✓ Ticker mapping: {symbol} → {variant}")
                break
        
        # No data found for any variant
        if found_ticker is None:
            logger.warning(f"No price data for {symbol} (tried: {ticker_variants})")
            return pd.DataFrame(), None
        
        # Get data for the found ticker
        symbol_df = df.loc[found_ticker].copy()
        
        # Reset index to make date a column
        symbol_df = symbol_df.reset_index()
        
        # Map SimFin columns to standard names
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj. Close': 'adj_close',
            'Volume': 'volume',
            'Dividend': 'dividends',
            'Common Shares Outstanding': 'shares_outstanding',
            'Market-Cap': 'market_cap'
        }
        
        # Rename columns
        symbol_df = symbol_df.rename(columns=column_mapping)
        
        # Add symbol column
        symbol_df['symbol'] = symbol
        
        # Select standard columns (in order)
        standard_cols = [
            'symbol', 'date', 'open', 'high', 'low', 'close', 
            'adj_close', 'volume', 'dividends', 
            'shares_outstanding', 'market_cap'
        ]
        
        # Keep only columns that exist
        available_cols = [col for col in standard_cols if col in symbol_df.columns]
        symbol_df = symbol_df[available_cols]
        
        # Ensure date is datetime
        symbol_df['date'] = pd.to_datetime(symbol_df['date'])
        
        # Sort by date
        symbol_df = symbol_df.sort_values('date')
        
        # Fill NaN dividends with 0 (standard for non-dividend days)
        if 'dividends' in symbol_df.columns:
            symbol_df['dividends'] = symbol_df['dividends'].fillna(0)
        
        return symbol_df, found_ticker
        
    except Exception as e:
        logger.error(f"Error standardizing data for {symbol}: {e}")
        return pd.DataFrame(), None


def save_symbol_parquet(symbol: str, df: pd.DataFrame, storage_path: Path, frequency: str = "1d"):
    """Save single symbol price data to parquet"""
    try:
        freq_path = storage_path / frequency
        freq_path.mkdir(parents=True, exist_ok=True)
        
        symbol_path = freq_path / f"{symbol}.parquet"
        df.to_parquet(symbol_path, index=False, compression='snappy')
        logger.debug(f"Saved {symbol} to {symbol_path}")
    except Exception as e:
        logger.error(f"Failed to save {symbol}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Bulk share price ingestion from SimFin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch S&P 500 daily prices
  python scripts/ingest_shareprices_simfin.py --sp500-only
  
  # Fetch specific symbols
  python scripts/ingest_shareprices_simfin.py --symbols AAPL MSFT GOOGL
  
  # With custom config
  python scripts/ingest_shareprices_simfin.py --sp500-only --config config/config.yaml
        """
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to fetch'
    )
    parser.add_argument(
        '--sp500-only',
        action='store_true',
        help='Fetch all S&P 500 companies'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--frequency',
        default='1d',
        help='Data frequency for storage organization (default: 1d)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info("Loading configuration...")
    cfg = load_config(args.config)
    
    # Get SimFin settings
    simfin_cfg = cfg['fundamentals']['simfin']
    api_key = simfin_cfg['api_key']
    data_dir = simfin_cfg['data_dir']
    market = simfin_cfg['market']
    
    if api_key == "your-simfin-api-key-here":
        logger.error("Please set your SimFin API key in config/config.yaml")
        sys.exit(1)
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
        logger.info(f"Using {len(symbols)} symbols from command line")
    elif args.sp500_only:
        symbols = get_sp500_symbols()
        logger.info(f"Using {len(symbols)} S&P 500 symbols")
    else:
        logger.error("Please specify --symbols or --sp500-only")
        sys.exit(1)
    
    # Print header
    print("\n" + "=" * 70)
    print("SimFin Share Price Bulk Ingestion")
    print("=" * 70)
    print(f"Provider: SimFin")
    print(f"Market: {market}")
    print(f"Symbols: {len(symbols)}")
    print(f"Data type: Daily share prices (OHLCV + fundamentals)")
    print("=" * 70)
    
    # Initialize SimFin
    logger.info("Initializing SimFin...")
    initialize_simfin(api_key, data_dir)
    
    # Load daily share prices (bulk download from SimFin)
    logger.info("Loading daily share prices from SimFin...")
    logger.info("This may take a few minutes on first run (downloading and caching)...")
    logger.info("Subsequent runs will be instant (using cache)")
    
    try:
        # Load daily share prices from SimFin (bulk download)
        df_prices = sf.load_shareprices(variant='daily', market=market)
        
        logger.info(f"Loaded SimFin share prices dataset")
        logger.info(f"Available columns: {list(df_prices.columns)}")
        
        # Filter to requested symbols
        df_prices = df_prices[df_prices.index.get_level_values(TICKER).isin(symbols)]
        
        available_symbols = df_prices.index.get_level_values(TICKER).unique().tolist()
        logger.info(f"Found {len(available_symbols)} symbols in SimFin data")
        
        # Process and save each symbol
        print("\nProcessing and saving share prices...\n")
        
        processed = []
        failed = []
        total_bars = 0
        
        ticker_mappings_used = {}
        
        for symbol in tqdm(symbols, desc="Processing symbols"):
            try:
                # Standardize format (returns tuple: df, actual_ticker)
                symbol_df, actual_ticker = standardize_simfin_prices(df_prices, symbol)
                
                # Track successful mappings
                if actual_ticker and actual_ticker != symbol:
                    ticker_mappings_used[symbol] = actual_ticker
                
                if not symbol_df.empty:
                    # Save to parquet
                    save_symbol_parquet(
                        symbol=symbol,
                        df=symbol_df,
                        storage_path=Path(cfg['data']['parquet']),
                        frequency=args.frequency
                    )
                    
                    processed.append(symbol)
                    total_bars += len(symbol_df)
                else:
                    logger.warning(f"No data for {symbol}")
                    failed.append(symbol)
                    
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                failed.append(symbol)
        
        # Show sample data info
        if processed:
            sample_symbol = processed[0]
            sample_path = Path(cfg['data']['parquet']) / args.frequency / f"{sample_symbol}.parquet"
            if sample_path.exists():
                sample_df = pd.read_parquet(sample_path)
                print(f"\nSample data for {sample_symbol}:")
                print(f"  Periods: {len(sample_df)}")
                print(f"  Date range: {sample_df['date'].min()} to {sample_df['date'].max()}")
                print(f"  Columns: {list(sample_df.columns)}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("✓ Share Price Ingestion Complete!")
        print("=" * 70)
        print(f"  Total symbols: {len(symbols)}")
        print(f"  Processed: {len(processed)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Total bars: {total_bars:,}")
        print(f"  Storage: {cfg['data']['parquet']}/{args.frequency}/")
        print(f"  SimFin cache: {data_dir}")
        print("=" * 70)
        
        if ticker_mappings_used:
            print(f"\n✓ Ticker Mappings Used ({len(ticker_mappings_used)}):")
            for original, mapped in sorted(ticker_mappings_used.items())[:10]:
                print(f"  {original} → {mapped}")
            if len(ticker_mappings_used) > 10:
                print(f"  ... and {len(ticker_mappings_used) - 10} more")
        
        if failed:
            print(f"\nFailed symbols ({len(failed)}):")
            for symbol in failed[:10]:  # Show first 10
                print(f"  - {symbol}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")
        
    except Exception as e:
        logger.error(f"Failed to load share prices: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
