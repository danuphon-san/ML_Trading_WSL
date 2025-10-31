#!/usr/bin/env python3
"""
Bulk OHLCV Data Ingestion Script

This script performs bulk download of historical OHLCV data for multiple symbols.
Inspired by SimFin's download patterns with progress tracking and resume capability.

Usage:
    python scripts/ingest_ohlcv_bulk.py --sp500-only --years 10
    python scripts/ingest_ohlcv_bulk.py --symbols AAPL MSFT GOOGL --years 5
    python scripts/ingest_ohlcv_bulk.py --start-date 2015-01-01 --end-date 2024-12-31
    python scripts/ingest_ohlcv_bulk.py --resume
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import argparse
from loguru import logger
from tqdm import tqdm

from src.io.ingest_ohlcv import OHLCVIngester
from src.io.universe import load_sp500_constituents


class ProgressTracker:
    """Track download progress and enable resume capability"""
    
    def __init__(self, checkpoint_path: str = "data/.ohlcv_progress.json"):
        self.checkpoint_path = Path(checkpoint_path)
        self.completed: List[str] = []
        self.failed: List[Dict] = []
        self.in_progress: Optional[str] = None
        self.start_time: Optional[datetime] = None
        
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                    self.completed = data.get('completed', [])
                    self.failed = data.get('failed', [])
                    self.start_time = datetime.fromisoformat(data.get('start_time', datetime.now().isoformat()))
                logger.info(f"Loaded checkpoint: {len(self.completed)} completed, {len(self.failed)} failed")
                return True
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
                return False
        return False
    
    def save_checkpoint(self):
        """Save current progress"""
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'completed': self.completed,
                'failed': self.failed,
                'in_progress': self.in_progress,
                'start_time': self.start_time.isoformat() if self.start_time else datetime.now().isoformat(),
                'last_update': datetime.now().isoformat()
            }
            with open(self.checkpoint_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")
    
    def mark_completed(self, symbol: str):
        """Mark symbol as completed"""
        self.completed.append(symbol)
        self.in_progress = None
        self.save_checkpoint()
    
    def mark_failed(self, symbol: str, error: str):
        """Mark symbol as failed"""
        self.failed.append({'symbol': symbol, 'error': error})
        self.in_progress = None
        self.save_checkpoint()
    
    def mark_in_progress(self, symbol: str):
        """Mark symbol as in progress"""
        self.in_progress = symbol
    
    def get_remaining(self, all_symbols: List[str]) -> List[str]:
        """Get list of symbols not yet completed"""
        completed_set = set(self.completed)
        return [s for s in all_symbols if s not in completed_set]
    
    def clear(self):
        """Clear checkpoint file"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


class BulkOHLCVFetcher:
    """Orchestrate bulk OHLCV downloads"""
    
    def __init__(
        self,
        provider: str = "yfinance",
        storage_path: str = "data/parquet",
        frequency: str = "1d"
    ):
        self.ingester = OHLCVIngester(provider=provider, storage_path=storage_path)
        self.frequency = frequency
        self.tracker = ProgressTracker()
        
    def check_file_age(self, symbol: str) -> Optional[int]:
        """
        Check age of existing data file in days
        
        Returns:
            Age in days if file exists, None otherwise
        """
        file_path = Path(self.ingester.storage_path) / self.frequency / f"{symbol}.parquet"
        if file_path.exists():
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            age = datetime.now() - mtime
            return age.days
        return None
    
    def fetch_bulk(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        max_age_days: Optional[int] = None,
        resume: bool = False
    ) -> Dict:
        """
        Bulk fetch OHLCV data for symbols
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_age_days: Skip files newer than this many days
            resume: Resume from checkpoint if True
            
        Returns:
            Dictionary with statistics
        """
        # Initialize tracker
        if resume and self.tracker.load_checkpoint():
            symbols_to_fetch = self.tracker.get_remaining(symbols)
            logger.info(f"Resuming: {len(symbols_to_fetch)} symbols remaining")
        else:
            symbols_to_fetch = symbols
            self.tracker.start_time = datetime.now()
        
        # Check file ages if max_age_days specified
        skipped_fresh = []
        stale_symbols = []
        missing_symbols = []
        
        if max_age_days is not None:
            print("\nChecking existing data...")
            for symbol in symbols_to_fetch:
                age = self.check_file_age(symbol)
                if age is None:
                    missing_symbols.append(symbol)
                elif age < max_age_days:
                    skipped_fresh.append(symbol)
                else:
                    stale_symbols.append(symbol)
            
            symbols_to_download = stale_symbols + missing_symbols
            
            print(f"- Found {len(symbols_to_fetch)} symbols to check")
            print(f"- {len(skipped_fresh)} fresh (< {max_age_days} days), will skip")
            print(f"- {len(stale_symbols)} stale (>= {max_age_days} days), will refresh")
            print(f"- {len(missing_symbols)} missing, will download")
        else:
            symbols_to_download = symbols_to_fetch
            skipped_fresh = []
        
        if not symbols_to_download:
            print("\nAll data is up to date!")
            return {
                'total': len(symbols),
                'downloaded': 0,
                'skipped': len(skipped_fresh),
                'failed': 0
            }
        
        # Download data
        print(f"\nDownloading OHLCV data for {len(symbols_to_download)} symbols...")
        print(f"Date range: {start_date} to {end_date or 'today'}")
        print(f"Frequency: {self.frequency}\n")
        
        downloaded = []
        failed = []
        total_bars = 0
        
        for symbol in tqdm(symbols_to_download, desc="Fetching OHLCV"):
            self.tracker.mark_in_progress(symbol)
            
            try:
                # Fetch data for single symbol
                data_dict = self.ingester.fetch_ohlcv(
                    symbols=[symbol],
                    start_date=start_date,
                    end_date=end_date,
                    frequency=self.frequency
                )
                
                if symbol in data_dict and not data_dict[symbol].empty:
                    # Save to parquet
                    self.ingester.save_parquet(
                        data_dict={symbol: data_dict[symbol]},
                        frequency=self.frequency,
                        partitioned=True
                    )
                    
                    downloaded.append(symbol)
                    total_bars += len(data_dict[symbol])
                    self.tracker.mark_completed(symbol)
                else:
                    logger.warning(f"No data returned for {symbol}")
                    failed.append(symbol)
                    self.tracker.mark_failed(symbol, "No data returned")
                    
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                failed.append(symbol)
                self.tracker.mark_failed(symbol, str(e))
        
        # Calculate statistics
        duration = datetime.now() - self.tracker.start_time
        
        stats = {
            'total': len(symbols),
            'downloaded': len(downloaded),
            'skipped': len(skipped_fresh),
            'failed': len(failed),
            'total_bars': total_bars,
            'duration_seconds': duration.total_seconds(),
            'symbols_per_minute': len(downloaded) / (duration.total_seconds() / 60) if duration.total_seconds() > 0 else 0
        }
        
        return stats


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_date_range(years: Optional[int], start_date: Optional[str], end_date: Optional[str]) -> tuple:
    """
    Calculate start and end dates
    
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    if start_date and end_date:
        return start_date, end_date
    
    end = datetime.now().strftime('%Y-%m-%d') if not end_date else end_date
    
    if years:
        start = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    elif start_date:
        start = start_date
    else:
        start = "2015-01-01"  # Default: 10 years
    
    return start, end


def main():
    parser = argparse.ArgumentParser(
        description='Bulk OHLCV data ingestion with progress tracking and resume capability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 10 years for S&P 500
  python scripts/ingest_ohlcv_bulk.py --sp500-only --years 10
  
  # Fetch specific symbols
  python scripts/ingest_ohlcv_bulk.py --symbols AAPL MSFT GOOGL --years 5
  
  # Custom date range
  python scripts/ingest_ohlcv_bulk.py --start-date 2015-01-01 --end-date 2024-12-31
  
  # Resume interrupted download
  python scripts/ingest_ohlcv_bulk.py --resume
  
  # Skip fresh data (< 7 days old)
  python scripts/ingest_ohlcv_bulk.py --sp500-only --max-age-days 7
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
        help='Fetch all S&P 500 companies'
    )
    
    # Date range options
    parser.add_argument(
        '--years',
        type=int,
        help='Number of years of history to fetch'
    )
    parser.add_argument(
        '--start-date',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        help='End date (YYYY-MM-DD), default: today'
    )
    
    # Provider and frequency
    parser.add_argument(
        '--provider',
        default='yfinance',
        help='Data provider (default: yfinance)'
    )
    parser.add_argument(
        '--frequency',
        default='1d',
        help='Data frequency (default: 1d)'
    )
    
    # Resume and optimization
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
        '--config',
        default='config/config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
        logger.info(f"Using {len(symbols)} symbols from command line")
    elif args.sp500_only:
        try:
            symbols = load_sp500_constituents()
            logger.info(f"Loaded {len(symbols)} S&P 500 symbols")
        except Exception as e:
            logger.error(f"Could not load S&P 500: {e}")
            logger.info("Using fallback symbols")
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    else:
        logger.error("Please specify --symbols or --sp500-only")
        sys.exit(1)
    
    # Calculate date range
    start_date, end_date = calculate_date_range(args.years, args.start_date, args.end_date)
    
    # Print header
    print("\n" + "=" * 70)
    print("Bulk OHLCV Data Ingestion")
    print("=" * 70)
    print(f"Provider: {args.provider}")
    print(f"Symbols: {len(symbols)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Frequency: {args.frequency}")
    if args.max_age_days:
        print(f"Max age: {args.max_age_days} days")
    print("=" * 70)
    
    # Initialize fetcher
    fetcher = BulkOHLCVFetcher(
        provider=args.provider,
        storage_path=cfg['data']['parquet'],
        frequency=args.frequency
    )
    
    # Fetch data
    try:
        stats = fetcher.fetch_bulk(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            max_age_days=args.max_age_days,
            resume=args.resume
        )
        
        # Print summary
        print("\n" + "=" * 70)
        print("✓ Download Complete!")
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
        
        # Clear checkpoint if successful
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
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
