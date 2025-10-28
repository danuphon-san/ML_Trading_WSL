#!/usr/bin/env python3
"""
Re-ingest fundamental data using Alpha Vantage for extended history
Usage: python scripts/reingest_fundamentals_alphavantage.py
"""

import sys
import os
from pathlib import Path

# Get project root directory and change to it
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from loguru import logger
from src.io.ingest_fundamentals import FundamentalsIngester

def main():
    # Load configuration (now relative to project root)
    config_path = PROJECT_ROOT / 'config' / 'config.yaml'
    cfg = yaml.safe_load(open(config_path))

    # Get list of symbols from existing universe or data
    symbols = []
    parquet_path = Path(cfg['data']['parquet']) / '1d'

    if parquet_path.exists():
        # Get symbols from existing price data
        symbols = [f.stem for f in parquet_path.glob('*.parquet')]
        logger.info(f"Found {len(symbols)} symbols from price data")
    else:
        logger.error("No price data found. Please run data ingestion first.")
        return

    # Ask for confirmation
    print(f"\n{'='*70}")
    print(f"Re-ingesting fundamentals for {len(symbols)} symbols using Alpha Vantage")
    print(f"Provider: {cfg['fundamentals']['provider']}")
    print(f"API Key: {cfg['fundamentals']['alpha_vantage']['api_key'][:10]}...")
    print(f"Storage: {cfg['data']['fundamentals']}")
    print(f"\nNote: Free tier allows ~5 calls/minute (4 APIs × symbol)")
    print(f"Estimated time: ~{len(symbols) * 0.8:.1f} minutes for {len(symbols)} symbols")
    print(f"{'='*70}\n")

    response = input("Continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Aborted.")
        return

    # Initialize ingester with Alpha Vantage
    ingester = FundamentalsIngester(
        storage_path=cfg['data']['fundamentals'],
        provider=cfg['fundamentals']['provider'],
        api_key=cfg['fundamentals']['alpha_vantage']['api_key']
    )

    # Fetch and save fundamentals
    logger.info(f"Starting ingestion for {len(symbols)} symbols...")
    data_dict = ingester.fetch_fundamentals(symbols)

    if data_dict:
        logger.info(f"Saving {len(data_dict)} symbols to parquet...")
        ingester.save_parquet(data_dict)

        # Print summary
        print(f"\n{'='*70}")
        print(f"Successfully ingested {len(data_dict)}/{len(symbols)} symbols")

        # Show sample data range
        if data_dict:
            sample_symbol = list(data_dict.keys())[0]
            sample_df = data_dict[sample_symbol]
            print(f"\nSample ({sample_symbol}):")
            print(f"  Quarters: {len(sample_df)}")
            print(f"  Date range: {sample_df['date'].min()} to {sample_df['date'].max()}")

        print(f"{'='*70}\n")
    else:
        logger.error("No data fetched. Check API key and rate limits.")

if __name__ == "__main__":
    main()
