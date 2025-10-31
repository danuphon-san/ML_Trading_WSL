"""
Interactive execution logger

Records manual trade executions for position tracking and reconciliation.

Usage:
    python scripts/log_execution.py                    # Interactive mode
    python scripts/log_execution.py --date 2025-10-30  # Specific date
    python scripts/log_execution.py --output custom.csv # Custom output file
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(title)
    print("="*70 + "\n")


def get_execution_details():
    """Interactive prompts for execution details"""
    print("Enter execution details (or 'done' to finish):")
    print()

    executions = []

    while True:
        symbol = input("Symbol (or 'done'): ").strip().upper()
        if symbol.lower() == 'done':
            break

        if not symbol:
            print("  ✗ Symbol required")
            continue

        side = input("Side (buy/sell): ").strip().lower()
        if side not in ['buy', 'sell']:
            print("  ✗ Invalid side, must be 'buy' or 'sell'")
            continue

        try:
            shares = int(input("Shares: ").strip())
            if shares <= 0:
                print("  ✗ Shares must be positive")
                continue
        except ValueError:
            print("  ✗ Invalid shares, must be a number")
            continue

        try:
            fill_price = float(input("Fill Price: ").strip())
            if fill_price <= 0:
                print("  ✗ Price must be positive")
                continue
        except ValueError:
            print("  ✗ Invalid price, must be a number")
            continue

        fill_time = input("Fill Time (HH:MM:SS or leave blank for now): ").strip()
        if not fill_time:
            fill_time = datetime.now().strftime("%H:%M:%S")

        try:
            commission = float(input("Commission (default 0.00): ").strip() or "0.00")
        except ValueError:
            commission = 0.00

        notes = input("Notes (optional): ").strip()

        # Record execution
        execution = {
            'symbol': symbol,
            'side': side,
            'shares': shares,
            'fill_price': fill_price,
            'fill_time': fill_time,
            'commission': commission,
            'notes': notes or ''
        }

        executions.append(execution)

        print(f"  ✓ Logged: {symbol} {side} {shares}@${fill_price:.2f}")
        print()

    return executions


def load_existing_log(log_file):
    """Load existing execution log if it exists"""
    if log_file.exists():
        try:
            return pd.read_csv(log_file)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_executions(executions, log_file, append=False):
    """Save executions to CSV"""
    new_df = pd.DataFrame(executions)

    if append and log_file.exists():
        # Append to existing log
        existing_df = load_existing_log(log_file)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(log_file, index=False)
    else:
        # Create new log
        new_df.to_csv(log_file, index=False)

    return len(executions)


def quick_import_mode(trades_file, log_file):
    """Quick mode: Mark all trades as executed at recommended price"""
    if not trades_file.exists():
        print(f"Error: Trades file not found: {trades_file}")
        return False

    trades_df = pd.read_csv(trades_file)

    print(f"Found {len(trades_df)} trades in {trades_file.name}")
    print()
    response = input("Mark all as executed at recommended price? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("Cancelled")
        return False

    # Create execution records
    executions = []
    for _, trade in trades_df.iterrows():
        execution = {
            'symbol': trade['symbol'],
            'side': trade['side'],
            'shares': trade['shares'],
            'fill_price': trade['price'],  # Use recommended price
            'fill_time': datetime.now().strftime("%H:%M:%S"),
            'commission': trade.get('commission_est', 0.00),
            'notes': 'Quick import from trades.csv'
        }
        executions.append(execution)

    # Save
    save_executions(executions, log_file, append=False)
    print(f"\n✓ Imported {len(executions)} executions to {log_file}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Log manual trade executions')
    parser.add_argument('--date', type=str, help='Trading date (YYYY-MM-DD, default: today)')
    parser.add_argument('--output', type=str, help='Output file (default: live/DATE/execution_log.csv)')
    parser.add_argument('--quick-import', action='store_true',
                       help='Quick import: mark all trades.csv as executed')
    parser.add_argument('--append', action='store_true', help='Append to existing log')
    args = parser.parse_args()

    # Determine date
    if args.date:
        date = args.date
    else:
        date = datetime.now().strftime("%Y-%m-%d")

    # Determine output file
    if args.output:
        log_file = Path(args.output)
    else:
        log_file = PROJECT_ROOT / "live" / date / "execution_log.csv"

    # Create directory if needed
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Print header
    print_header(f"EXECUTION LOGGER - {date}")

    # Quick import mode
    if args.quick_import:
        trades_file = log_file.parent / "trades.csv"
        if quick_import_mode(trades_file, log_file):
            return 0
        else:
            return 1

    # Interactive mode
    print("Log your trade executions interactively")
    print()

    # Check if log already exists
    if log_file.exists() and not args.append:
        print(f"⚠ Warning: Log file already exists: {log_file}")
        response = input("Append to existing log? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled. Use --append to append automatically")
            return 1
        args.append = True

    # Get executions
    executions = get_execution_details()

    if not executions:
        print("No executions logged")
        return 0

    # Save
    count = save_executions(executions, log_file, append=args.append)

    # Summary
    print("="*70)
    print(f"✓ Logged {count} executions")
    print(f"✓ Saved to: {log_file}")
    print()
    print("Next steps:")
    print(f"  python scripts/update_positions.py --execution-log {log_file}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
