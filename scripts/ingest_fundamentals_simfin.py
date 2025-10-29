#!/usr/bin/env python3
"""
SimFin Fundamental Data Ingestion Script

This script performs full bulk ingestion of fundamental data from SimFin.
Use this for initial setup or when you need to completely refresh the data.

Usage:
    python scripts/ingest_fundamentals_simfin.py
    python scripts/ingest_fundamentals_simfin.py --symbols AAPL MSFT GOOGL
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import argparse
from loguru import logger
from src.io.ingest_fundamentals import FundamentalsIngester
from src.io.universe import load_sp500_constituents


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
        # Fallback to a few sample symbols
        logger.info("Using fallback symbols")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']


def main():
    parser = argparse.ArgumentParser(
        description='Ingest fundamental data from SimFin (bulk download)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to ingest (default: use universe from config)'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--sp500-only',
        action='store_true',
        help='Only fetch S&P 500 companies'
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
    variant = simfin_cfg['variant']
    
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
        # Load from universe
        symbols = get_sp500_symbols()
        logger.info(f"Using {len(symbols)} symbols from universe")
    
    # Initialize ingester
    logger.info("Initializing SimFin ingester...")
    ingester = FundamentalsIngester(
        storage_path=cfg['data']['fundamentals'],
        provider='simfin',
        api_key=api_key,
        simfin_data_dir=data_dir,
        simfin_market=market,
        simfin_variant=variant
    )
    
    # Fetch data (bulk loading from SimFin)
    logger.info(f"Fetching fundamental data for {len(symbols)} symbols from SimFin...")
    logger.info("This may take a few minutes on first run (downloading and caching data)...")
    
    try:
        data_dict = ingester.fetch_fundamentals(symbols)
        
        if not data_dict:
            logger.error("No data fetched!")
            sys.exit(1)
        
        logger.info(f"Successfully fetched data for {len(data_dict)} symbols")
        
        # Show sample statistics
        sample_symbol = list(data_dict.keys())[0]
        sample_df = data_dict[sample_symbol]
        logger.info(f"\nSample data for {sample_symbol}:")
        logger.info(f"  Periods: {len(sample_df)}")
        logger.info(f"  Date range: {sample_df['date'].min()} to {sample_df['date'].max()}")
        logger.info(f"  Columns: {list(sample_df.columns)}")
        
        # Save to parquet
        logger.info("\nSaving data to parquet files...")
        ingester.save_parquet(data_dict)
        
        logger.info("\n✓ SimFin data ingestion complete!")
        logger.info(f"  Total symbols: {len(data_dict)}")
        logger.info(f"  Storage path: {cfg['data']['fundamentals']}")
        logger.info(f"  SimFin cache: {data_dir}")
        
    except Exception as e:
        logger.error(f"Failed to ingest data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
