"""
Import positions from broker CSV export

Converts broker position exports to system format and updates current_positions.csv

Supports common broker formats:
- Fidelity
- Charles Schwab
- TD Ameritrade
- Interactive Brokers
- Generic CSV

Usage:
    python scripts/import_positions.py --broker-file positions.csv --broker fidelity
    python scripts/import_positions.py --broker-file positions.csv --auto-detect
    python scripts/import_positions.py --broker-file positions.csv --force
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


def detect_broker_format(df):
    """Auto-detect broker format from column names"""
    columns = [c.lower() for c in df.columns]

    # Fidelity
    if any('account number' in c for c in columns):
        return 'fidelity'

    # Schwab
    if any('account #' in c for c in columns):
        return 'schwab'

    # TD Ameritrade
    if any('account id' in c for c in columns):
        return 'tdameritrade'

    # Interactive Brokers
    if any('conid' in c for c in columns):
        return 'ibkr'

    # Generic (has symbol, quantity, price columns)
    has_symbol = any('symbol' in c for c in columns)
    has_quantity = any('quantity' in c or 'shares' in c for c in columns)
    has_price = any('price' in c or 'cost' in c for c in columns)

    if has_symbol and has_quantity:
        return 'generic'

    return None


def parse_fidelity(df):
    """Parse Fidelity positions export"""
    # Fidelity format: Symbol, Description, Quantity, Last Price, Current Value, etc.
    column_map = {
        'Symbol': 'symbol',
        'Quantity': 'shares',
        'Last Price': 'current_price',
        'Cost Basis Per Share': 'avg_cost',
        'Last Price Change': None,
        'Current Value': None
    }

    # Rename columns
    df = df.rename(columns=column_map)

    # Keep only needed columns
    df = df[['symbol', 'shares', 'avg_cost']]

    # Clean data
    df = df.dropna(subset=['symbol'])
    df['shares'] = df['shares'].astype(float).astype(int)
    df['avg_cost'] = df['avg_cost'].astype(float)

    # Remove cash rows
    df = df[df['symbol'] != 'SPAXX']  # Fidelity money market
    df = df[df['symbol'].str.len() <= 5]  # Filter out funds

    return df


def parse_schwab(df):
    """Parse Charles Schwab positions export"""
    # Schwab format: Symbol, Description, Qty, Price, Price Change, Value, etc.
    column_map = {
        'Symbol': 'symbol',
        'Qty': 'shares',
        'Price': 'current_price',
        'Cost Basis': 'avg_cost'
    }

    df = df.rename(columns=column_map)
    df = df[['symbol', 'shares', 'avg_cost']]

    df = df.dropna(subset=['symbol'])
    df['shares'] = df['shares'].astype(float).astype(int)
    df['avg_cost'] = df['avg_cost'].astype(float)

    # Remove cash/funds
    df = df[~df['symbol'].str.contains('SWVXX', na=False)]

    return df


def parse_tdameritrade(df):
    """Parse TD Ameritrade positions export"""
    column_map = {
        'Symbol': 'symbol',
        'Quantity': 'shares',
        'Trade Price': 'avg_cost',
        'Mark': 'current_price'
    }

    df = df.rename(columns=column_map)
    df = df[['symbol', 'shares', 'avg_cost']]

    df = df.dropna(subset=['symbol'])
    df['shares'] = df['shares'].astype(float).astype(int)
    df['avg_cost'] = df['avg_cost'].astype(float)

    return df


def parse_ibkr(df):
    """Parse Interactive Brokers positions export"""
    column_map = {
        'Symbol': 'symbol',
        'Quantity': 'shares',
        'Avg Cost': 'avg_cost',
        'Mark Price': 'current_price'
    }

    df = df.rename(columns=column_map)
    df = df[['symbol', 'shares', 'avg_cost']]

    df = df.dropna(subset=['symbol'])
    df['shares'] = df['shares'].astype(float).astype(int)
    df['avg_cost'] = df['avg_cost'].astype(float)

    return df


def parse_generic(df):
    """Parse generic CSV format"""
    # Try to find columns
    columns_lower = {c.lower(): c for c in df.columns}

    # Find symbol column
    symbol_col = None
    for key in ['symbol', 'ticker', 'security']:
        if key in columns_lower:
            symbol_col = columns_lower[key]
            break

    # Find shares column
    shares_col = None
    for key in ['shares', 'quantity', 'qty', 'position']:
        if key in columns_lower:
            shares_col = columns_lower[key]
            break

    # Find cost column
    cost_col = None
    for key in ['avg_cost', 'cost_basis', 'average_cost', 'cost', 'price']:
        if key in columns_lower:
            cost_col = columns_lower[key]
            break

    if not symbol_col or not shares_col:
        raise ValueError("Could not find required columns (symbol and shares)")

    # Rename columns
    rename_map = {
        symbol_col: 'symbol',
        shares_col: 'shares'
    }
    if cost_col:
        rename_map[cost_col] = 'avg_cost'

    df = df.rename(columns=rename_map)

    # Keep only needed columns
    cols = ['symbol', 'shares']
    if 'avg_cost' in df.columns:
        cols.append('avg_cost')
    else:
        # Default avg_cost to 0 (user should update manually)
        df['avg_cost'] = 0.0

    df = df[cols]

    # Clean data
    df = df.dropna(subset=['symbol'])
    df['shares'] = df['shares'].astype(float).astype(int)
    df['avg_cost'] = df['avg_cost'].astype(float)

    return df


def import_positions(broker_file, broker=None, auto_detect=False):
    """Import positions from broker file"""
    # Load file
    try:
        df = pd.read_csv(broker_file)
    except Exception as e:
        print(f"Error loading broker file: {e}")
        return None

    # Auto-detect broker format
    if auto_detect or broker is None:
        detected = detect_broker_format(df)
        if detected:
            print(f"Auto-detected broker format: {detected}")
            broker = detected
        else:
            print("Could not auto-detect broker format")
            print("Available formats: fidelity, schwab, tdameritrade, ibkr, generic")
            print("Please specify with --broker flag")
            return None

    # Parse based on broker
    parsers = {
        'fidelity': parse_fidelity,
        'schwab': parse_schwab,
        'tdameritrade': parse_tdameritrade,
        'ibkr': parse_ibkr,
        'generic': parse_generic
    }

    if broker.lower() not in parsers:
        print(f"Unknown broker: {broker}")
        print(f"Available: {', '.join(parsers.keys())}")
        return None

    try:
        positions_df = parsers[broker.lower()](df)
    except Exception as e:
        print(f"Error parsing broker file: {e}")
        return None

    # Add last_updated
    positions_df['last_updated'] = datetime.now().strftime('%Y-%m-%d')

    return positions_df


def main():
    parser = argparse.ArgumentParser(description='Import positions from broker CSV')
    parser.add_argument('--broker-file', type=str, required=True, help='Path to broker positions CSV')
    parser.add_argument('--broker', type=str, help='Broker name (fidelity, schwab, tdameritrade, ibkr, generic)')
    parser.add_argument('--auto-detect', action='store_true', help='Auto-detect broker format')
    parser.add_argument('--output', type=str, default='live/current_positions.csv',
                       help='Output file (default: live/current_positions.csv)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing positions without confirmation')
    parser.add_argument('--show-preview', action='store_true', help='Preview imported positions before saving')
    args = parser.parse_args()

    broker_file = Path(args.broker_file)
    output_file = Path(args.output)

    # Check broker file exists
    if not broker_file.exists():
        print(f"Error: Broker file not found: {broker_file}")
        return 1

    # Print header
    print_header("IMPORT POSITIONS FROM BROKER")

    print(f"Broker file: {broker_file}")
    print(f"Output file: {output_file}")
    print()

    # Import positions
    positions_df = import_positions(broker_file, args.broker, args.auto_detect)

    if positions_df is None:
        return 1

    # Show preview
    print(f"Imported {len(positions_df)} positions:")
    print()
    print(positions_df[['symbol', 'shares', 'avg_cost']].head(20).to_string(index=False))
    if len(positions_df) > 20:
        print(f"\n... and {len(positions_df) - 20} more positions")
    print()

    # Calculate total value
    if 'avg_cost' in positions_df.columns:
        total_value = (positions_df['shares'] * positions_df['avg_cost']).sum()
        print(f"Total Portfolio Value (estimated): ${total_value:,.2f}")
        print()

    # Confirm before saving
    if output_file.exists() and not args.force:
        print(f"⚠ Warning: {output_file} already exists")
        response = input("Overwrite existing positions? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled")
            return 1

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    positions_df[['symbol', 'shares', 'avg_cost', 'last_updated']].to_csv(output_file, index=False)

    print("="*70)
    print(f"✓ Imported {len(positions_df)} positions")
    print(f"✓ Saved to: {output_file}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
