#!/usr/bin/env python3
"""
Incremental update of fundamental data (adds new quarters, keeps existing)
Usage:
    python scripts/update_fundamentals.py
    python scripts/update_fundamentals.py --symbols AAPL MSFT GOOGL
"""

import sys
import os
from pathlib import Path

# Get project root directory and change to it
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import yaml
from loguru import logger
from src.io.ingest_fundamentals import FundamentalsIngester

def main():
    parser = argparse.ArgumentParser(description='Update fundamental data incrementally')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to update (default: all)')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    args = parser.parse_args()

    # Load configuration (now relative to project root)
    config_path = PROJECT_ROOT / args.config
    cfg = yaml.safe_load(open(config_path))

    # Determine symbols to update
    if args.symbols:
        symbols = args.symbols
        logger.info(f"Updating {len(symbols)} specified symbols")
    else:
        # Get all symbols from existing parquet files
        parquet_path = Path(cfg['data']['parquet']) / '1d'
        if parquet_path.exists():
            symbols = [f.stem for f in parquet_path.glob('*.parquet')]
            logger.info(f"Updating {len(symbols)} symbols from price data")
        else:
            logger.error("No price data found. Run data ingestion first.")
            return

    # Initialize ingester with configured provider
    fund_config = cfg.get('fundamentals', {})
    provider = fund_config.get('provider', 'yfinance')
    api_key = None
    if provider == 'alpha_vantage':
        api_key = fund_config.get('alpha_vantage', {}).get('api_key')

    ingester = FundamentalsIngester(
        storage_path=cfg['data']['fundamentals'],
        provider=provider,
        api_key=api_key
    )

    logger.info(f"Starting incremental update using {provider}...")
    logger.info("Note: This merges new data with existing data (no full re-download)")

    # Use update_data method (incremental merge)
    ingester.update_data(symbols)

    logger.info("✓ Fundamental data update complete")

if __name__ == "__main__":
    main()
