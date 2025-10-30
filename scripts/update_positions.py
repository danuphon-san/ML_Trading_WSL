"""
Update current positions from execution log

Reads execution log and updates current_positions.csv to reflect
manual trade executions.

Usage:
    python scripts/update_positions.py --execution-log live/2025-10-30/execution_log.csv
    python scripts/update_positions.py --execution-log live/2025-10-30/execution_log.csv --current-positions live/current_positions.csv
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


def load_current_positions(positions_file):
    """Load current positions, create empty if doesn't exist"""
    if positions_file.exists():
        try:
            df = pd.read_csv(positions_file)
            if 'symbol' not in df.columns:
                print(f"Warning: Invalid positions file format, starting fresh")
                return pd.DataFrame(columns=['symbol', 'shares', 'avg_cost', 'last_updated'])
            return df
        except Exception as e:
            print(f"Warning: Could not load positions file: {e}")
            print("Starting with empty portfolio")
            return pd.DataFrame(columns=['symbol', 'shares', 'avg_cost', 'last_updated'])
    else:
        print("No existing positions file, starting with empty portfolio")
        return pd.DataFrame(columns=['symbol', 'shares', 'avg_cost', 'last_updated'])


def update_position(positions_df, symbol, side, shares, fill_price, date):
    """Update a single position"""
    # Find existing position
    existing = positions_df[positions_df['symbol'] == symbol]

    if len(existing) > 0:
        # Update existing position
        idx = existing.index[0]
        current_shares = positions_df.loc[idx, 'shares']
        current_avg_cost = positions_df.loc[idx, 'avg_cost']

        if side == 'buy':
            # Add shares, update avg cost
            new_shares = current_shares + shares
            new_avg_cost = ((current_shares * current_avg_cost) + (shares * fill_price)) / new_shares
            positions_df.loc[idx, 'shares'] = new_shares
            positions_df.loc[idx, 'avg_cost'] = new_avg_cost
            positions_df.loc[idx, 'last_updated'] = date
        else:  # sell
            # Reduce shares
            new_shares = current_shares - shares
            if new_shares <= 0:
                # Position closed
                positions_df = positions_df.drop(idx)
            else:
                positions_df.loc[idx, 'shares'] = new_shares
                positions_df.loc[idx, 'last_updated'] = date
                # Keep same avg_cost
    else:
        # New position
        if side == 'buy':
            new_row = pd.DataFrame([{
                'symbol': symbol,
                'shares': shares,
                'avg_cost': fill_price,
                'last_updated': date
            }])
            positions_df = pd.concat([positions_df, new_row], ignore_index=True)
        else:
            # Selling something we don't own (short?)
            print(f"  ⚠ Warning: Selling {symbol} but no existing position")
            # For now, ignore (could create negative position if shorting allowed)

    return positions_df


def calculate_portfolio_value(positions_df, current_prices):
    """Calculate total portfolio value"""
    if len(positions_df) == 0:
        return 0.0

    total_value = 0.0
    for _, pos in positions_df.iterrows():
        symbol = pos['symbol']
        shares = pos['shares']
        price = current_prices.get(symbol, pos['avg_cost'])  # Fallback to avg_cost if no price
        total_value += shares * price

    return total_value


def main():
    parser = argparse.ArgumentParser(description='Update positions from execution log')
    parser.add_argument('--execution-log', type=str, required=True, help='Path to execution_log.csv')
    parser.add_argument('--current-positions', type=str, default='live/current_positions.csv',
                       help='Path to current_positions.csv')
    parser.add_argument('--show-details', action='store_true', help='Show detailed position updates')
    args = parser.parse_args()

    execution_log_file = Path(args.execution_log)
    positions_file = Path(args.current_positions)

    # Check execution log exists
    if not execution_log_file.exists():
        print(f"Error: Execution log not found: {execution_log_file}")
        return 1

    # Load execution log
    try:
        executions_df = pd.read_csv(execution_log_file)
    except Exception as e:
        print(f"Error loading execution log: {e}")
        return 1

    if len(executions_df) == 0:
        print("No executions in log, nothing to update")
        return 0

    # Get date from execution log path
    date = execution_log_file.parent.name

    # Print header
    print_header(f"POSITION UPDATE - {date}")

    print(f"Execution log: {execution_log_file}")
    print(f"Positions file: {positions_file}")
    print()

    # Load current positions
    positions_df = load_current_positions(positions_file)
    original_count = len(positions_df)

    print(f"Starting positions: {original_count}")
    print(f"Executions to process: {len(executions_df)}")
    print()

    # Process each execution
    buy_count = 0
    sell_count = 0

    for _, execution in executions_df.iterrows():
        symbol = execution['symbol']
        side = execution['side']
        shares = execution['shares']
        fill_price = execution['fill_price']

        if args.show_details:
            print(f"  Processing: {symbol} {side} {shares}@${fill_price:.2f}")

        positions_df = update_position(positions_df, symbol, side, shares, fill_price, date)

        if side == 'buy':
            buy_count += 1
        else:
            sell_count += 1

    # Print summary
    print("="*70)
    print("Executions processed:")
    print(f"  Buy orders: {buy_count}")
    print(f"  Sell orders: {sell_count}")
    print()

    # Show updated positions
    final_count = len(positions_df)
    print(f"Updated Positions ({final_count} holdings):")
    print()

    if final_count > 0:
        # Calculate current value (using avg_cost as proxy for current price)
        positions_df['current_value'] = positions_df['shares'] * positions_df['avg_cost']
        total_value = positions_df['current_value'].sum()
        positions_df['weight'] = positions_df['current_value'] / total_value if total_value > 0 else 0

        # Format display
        display_df = positions_df[['symbol', 'shares', 'current_value', 'weight']].copy()
        display_df['shares'] = display_df['shares'].astype(int)
        display_df['current_value'] = display_df['current_value'].apply(lambda x: f"${x:,.0f}")
        display_df['weight'] = display_df['weight'].apply(lambda x: f"{x:.2%}")

        # Sort by value
        positions_df_sorted = positions_df.sort_values('current_value', ascending=False)
        display_df = positions_df_sorted[['symbol', 'shares', 'current_value', 'weight']].copy()
        display_df['shares'] = display_df['shares'].astype(int)
        display_df['current_value'] = display_df['current_value'].apply(lambda x: f"${x:,.0f}")
        display_df['weight'] = display_df['weight'].apply(lambda x: f"{x:.2%}")

        print(display_df.head(20).to_string(index=False))

        if final_count > 20:
            print(f"\n... and {final_count - 20} more positions")

        print()
        print(f"Total Portfolio Value (estimated): ${total_value:,.2f}")
        print()

    # Drop temporary columns
    positions_df = positions_df[['symbol', 'shares', 'avg_cost', 'last_updated']]

    # Save updated positions
    positions_file.parent.mkdir(parents=True, exist_ok=True)
    positions_df.to_csv(positions_file, index=False)

    print(f"✓ Positions updated: {positions_file}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
