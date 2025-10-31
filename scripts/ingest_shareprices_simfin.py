#!/usr/bin/env python3
"""
SimFin Share Price Ingestion Wrapper

Provides a convenience CLI for fetching OHLCV data from SimFin using the bulk
ingestion workflow shared with the general purpose ingest_ohlcv_bulk script.

Usage examples:
    python scripts/ingest_shareprices_simfin.py --sp500-only --years 10
    python scripts/ingest_shareprices_simfin.py --symbols AAPL MSFT --start-date 2015-01-01
"""

import sys
from pathlib import Path

# Add project root to path for module imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from loguru import logger

from ingest_ohlcv_bulk import (
    BulkOHLCVFetcher,
    calculate_date_range,
)

from src.io.ingest_ohlcv import build_provider_kwargs
from src.io.universe import load_sp500_constituents
from utils.config_loader import load_config


def resolve_symbols(args) -> list:
    """Determine symbol universe from CLI args."""
    if args.symbols:
        logger.info(f"Using {len(args.symbols)} symbols from command line")
        return args.symbols

    if args.sp500_only:
        try:
            symbols = load_sp500_constituents()
            logger.info(f"Loaded {len(symbols)} S&P 500 symbols")
            return symbols
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.error(f"Failed to load S&P 500 universe: {exc}")
            logger.info("Using fallback symbols")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    raise ValueError("Please specify --symbols or --sp500-only")


def main():
    parser = argparse.ArgumentParser(
        description='Bulk share price ingestion from SimFin (daily OHLCV)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest_shareprices_simfin.py --sp500-only --years 10
  python scripts/ingest_shareprices_simfin.py --symbols AAPL MSFT --start-date 2018-01-01
        """
    )

    # Symbol selection
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to fetch'
    )
    parser.add_argument(
        '--sp500-only',
        action='store_true',
        help='Fetch all S&P 500 constituents'
    )

    # Date range
    parser.add_argument(
        '--years',
        type=int,
        help='Number of years of history to fetch (overrides start-date)'
    )
    parser.add_argument(
        '--start-date',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        help='End date (YYYY-MM-DD), default: today'
    )

    # Operational options
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if interrupted'
    )
    parser.add_argument(
        '--max-age-days',
        type=int,
        help='Skip files newer than this many days'
    )
    parser.add_argument(
        '--frequency',
        default='1d',
        help='Data frequency (SimFin supports daily / 1d)'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Resolve universe
    try:
        symbols = resolve_symbols(args)
    except ValueError as exc:
        logger.error(exc)
        sys.exit(1)

    # Determine date range
    start_date, end_date = calculate_date_range(args.years, args.start_date, args.end_date)

    # Build provider configuration
    try:
        provider_kwargs = build_provider_kwargs("simfin", cfg)
    except ValueError as exc:
        logger.error(exc)
        sys.exit(1)

    # Header
    print("\n" + "=" * 70)
    print("SimFin Share Price Bulk Ingestion")
    print("=" * 70)
    print(f"Provider: SimFin")
    print(f"Symbols: {len(symbols)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Frequency: {args.frequency}")
    if args.max_age_days:
        print(f"Max age: {args.max_age_days} days")
    print("=" * 70)

    fetcher = BulkOHLCVFetcher(
        provider="simfin",
        storage_path=cfg['data']['parquet'],
        frequency=args.frequency,
        provider_kwargs=provider_kwargs
    )

    try:
        stats = fetcher.fetch_bulk(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            max_age_days=args.max_age_days,
            resume=args.resume
        )

        # Summary
        print("\n" + "=" * 70)
        print("✓ Share Price Ingestion Complete!")
        print("=" * 70)
        print(f"  Total symbols: {stats['total']}")
        print(f"  Downloaded: {stats['downloaded']}")
        print(f"  Skipped (fresh): {stats['skipped']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Total bars: {stats['total_bars']:,}")
        print(f"  Storage: {cfg['data']['parquet']}/{args.frequency}/")
        print(f"  Duration: {stats['duration_seconds'] / 60:.1f} minutes")
        if stats['downloaded'] > 0:
            print(f"  Avg speed: {stats['symbols_per_minute']:.1f} symbols/min")
        print("=" * 70)

        if stats['failed'] == 0:
            fetcher.tracker.clear()
            print("\nCheckpoint cleared (all symbols completed)")
        else:
            print(f"\nCheckpoint saved: {fetcher.tracker.checkpoint_path}")
            print("Run with --resume to retry failed symbols")

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        print(f"Progress saved to: {fetcher.tracker.checkpoint_path}")
        print("Run with --resume to continue")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"SimFin ingestion failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
