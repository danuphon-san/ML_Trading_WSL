#!/usr/bin/env python3
"""
Daily data update script - Updates both OHLCV and fundamentals incrementally
This is much faster than full re-ingestion as it only fetches new data.

Usage: python scripts/daily_update_data.py
"""

import sys
import os
from pathlib import Path

# Get project root directory and change to it
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from datetime import datetime, timedelta
from loguru import logger
from src.io.ingest_ohlcv import OHLCVIngester
from src.io.ingest_fundamentals import FundamentalsIngester

def main():
    start_time = datetime.now()
    logger.info("="*70)
    logger.info("Daily Data Update - Incremental Mode")
    logger.info(f"Started at: {start_time}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("="*70)

    # Load configuration (now relative to project root)
    config_path = PROJECT_ROOT / 'config' / 'config.yaml'
    cfg = yaml.safe_load(open(config_path))

    # Get symbols from existing data
    parquet_path = Path(cfg['data']['parquet']) / '1d'
    if not parquet_path.exists():
        logger.error("No existing price data found. Run full ingestion first:")
        logger.error("  python run_core_pipeline.py --steps 1")
        return

    symbols = [f.stem for f in parquet_path.glob('*.parquet')]
    logger.info(f"Found {len(symbols)} symbols to update")

    # =========================================================================
    # 1. Update OHLCV data (only fetch last 7 days to append)
    # =========================================================================
    logger.info("\n[1/2] Updating OHLCV data...")

    # Fetch only recent data (last 7 days to be safe with weekends/holidays)
    recent_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    ohlcv_ingester = OHLCVIngester(
        storage_path=cfg['data']['parquet'],
        provider=cfg['ingest']['provider']
    )

    try:
        logger.info(f"Fetching OHLCV from {recent_start} onwards...")
        new_data = ohlcv_ingester.fetch_ohlcv(symbols, recent_start, None)

        if new_data:
            # Merge with existing data
            for symbol, new_df in new_data.items():
                symbol_file = parquet_path / f"{symbol}.parquet"
                if symbol_file.exists():
                    import pandas as pd
                    existing_df = pd.read_parquet(symbol_file)

                    # Combine and remove duplicates
                    combined = pd.concat([existing_df, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['date'], keep='last')
                    combined = combined.sort_values('date')

                    # Save back
                    ohlcv_ingester.save_symbol_parquet(symbol, combined)
                else:
                    # No existing file, just save new data
                    ohlcv_ingester.save_symbol_parquet(symbol, new_df)

            logger.info(f"✓ Updated OHLCV for {len(new_data)} symbols")
        else:
            logger.warning("No new OHLCV data fetched")

    except Exception as e:
        logger.error(f"OHLCV update failed: {e}")

    # =========================================================================
    # 2. Update Fundamental data (merges new quarters)
    # =========================================================================
    logger.info("\n[2/2] Updating fundamental data...")

    fund_config = cfg.get('fundamentals', {})
    provider = fund_config.get('provider', 'yfinance')
    api_key = None
    if provider == 'alpha_vantage':
        api_key = fund_config.get('alpha_vantage', {}).get('api_key')

    fund_ingester = FundamentalsIngester(
        storage_path=cfg['data']['fundamentals'],
        provider=provider,
        api_key=api_key
    )

    try:
        logger.info(f"Updating fundamentals using {provider}...")
        fund_ingester.update_data(symbols)
        logger.info("✓ Fundamental data updated")
    except Exception as e:
        logger.error(f"Fundamental update failed: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "="*70)
    logger.info("Daily Update Complete")
    logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
