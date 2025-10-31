"""
Interactive trade approval script

Allows manual review and approval of ML-generated trade recommendations
before execution. Creates audit trail of approvals.

Usage:
    python scripts/approve_trades.py live/2025-10-30/trades.csv
    python scripts/approve_trades.py live/2025-10-30/trades.csv --auto-approve
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import subprocess
import platform

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(title)
    print("="*70 + "\n")


def print_trade_summary(trades_df, date):
    """Print summary of trades"""
    total_trades = len(trades_df)
    buy_orders = (trades_df['side'] == 'buy').sum()
    sell_orders = (trades_df['side'] == 'sell').sum()
    total_notional = trades_df['notional'].abs().sum()

    estimated_costs = 0
    if 'commission_est' in trades_df.columns and 'slippage_est' in trades_df.columns:
        estimated_costs = (trades_df['commission_est'] + trades_df['slippage_est']).sum()

    print(f"Date: {date}")
    print(f"Total Trades: {total_trades}")
    print(f"  Buy Orders: {buy_orders}")
    print(f"  Sell Orders: {sell_orders}")
    print(f"Total Notional: ${total_notional:,.2f}")
    if estimated_costs > 0:
        print(f"Estimated Costs: ${estimated_costs:,.2f}")
    print()


def load_monitoring_log(trades_dir):
    """Load monitoring log if available"""
    monitoring_file = trades_dir / "monitoring_log.json"
    if monitoring_file.exists():
        with open(monitoring_file, 'r') as f:
            return json.load(f)
    return None


def print_safety_status(trades_dir):
    """Print safety check status from monitoring log"""
    monitoring = load_monitoring_log(trades_dir)

    if monitoring and 'safety' in monitoring:
        safety = monitoring['safety']
        if safety.get('all_passed', False):
            print("Safety Checks: ✓ ALL PASSED")
        else:
            print("Safety Checks: ✗ SOME FAILED")
            if 'portfolio_issues' in safety:
                for issue in safety['portfolio_issues']:
                    print(f"  ✗ {issue}")
            if 'trade_issues' in safety:
                for issue in safety['trade_issues']:
                    print(f"  ✗ {issue}")
    else:
        print("Safety Checks: ⚠ Not available")

    # Print regime info
    if monitoring and 'regime' in monitoring:
        regime = monitoring['regime']
        regime_name = regime.get('current_regime', 'unknown')
        risk_mult = regime.get('risk_multiplier', 1.0)
        print(f"Current Regime: {regime_name} (Risk Multiplier: {risk_mult})")

    # Print IC
    if monitoring and 'ic_metrics' in monitoring:
        ic = monitoring['ic_metrics'].get('ic', 0)
        print(f"Information Coefficient: {ic:.4f}")

    print()


def open_file_in_viewer(filepath):
    """Open file in default viewer"""
    try:
        system = platform.system()
        if system == 'Darwin':  # macOS
            subprocess.run(['open', filepath], check=True)
        elif system == 'Windows':
            subprocess.run(['start', filepath], shell=True, check=True)
        else:  # Linux
            subprocess.run(['xdg-open', filepath], check=True)
        return True
    except Exception as e:
        print(f"Could not open file automatically: {e}")
        print(f"Please open manually: {filepath}")
        return False


def get_user_approval(trades_df, trades_file):
    """Interactive approval prompt"""
    trades_dir = trades_file.parent

    # Ask if user wants to review file
    response = input("Review trades.csv in viewer? (y/n): ").strip().lower()
    if response == 'y':
        open_file_in_viewer(str(trades_file))
        input("\nPress Enter when you've finished reviewing...")

    # Show sample of trades
    print("\nSample of trades (first 10):")
    print(trades_df[['symbol', 'side', 'shares', 'price', 'notional']].head(10).to_string(index=False))
    if len(trades_df) > 10:
        print(f"\n... and {len(trades_df) - 10} more trades")
    print()

    # Approval decision
    print("="*70)
    response = input("Do you approve these trades for execution? (yes/no/edit): ").strip().lower()

    if response in ['yes', 'y']:
        return 'approved', None
    elif response in ['edit', 'e']:
        return 'edit', "User requested edits"
    else:
        reason = input("Reason for rejection (optional): ").strip()
        return 'rejected', reason or "User rejected"


def create_approval_record(trades_file, status, reason=None, notes=None):
    """Create approval record"""
    trades_dir = trades_file.parent
    date = trades_dir.name

    # Create approval log
    approval_data = {
        "date": date,
        "timestamp": datetime.now().isoformat(),
        "approved_by": "manual",
        "status": status,
        "trades_file": str(trades_file),
        "trades_count": len(pd.read_csv(trades_file)),
        "reason": reason,
        "notes": notes
    }

    approval_log = trades_dir / "approval_log.json"
    with open(approval_log, 'w') as f:
        json.dump(approval_data, f, indent=2)

    # Create marker file if approved
    if status == 'approved':
        approval_marker = trades_dir / "APPROVED"
        approval_marker.touch()

        return True, approval_log
    elif status == 'rejected':
        rejection_marker = trades_dir / "REJECTED"
        rejection_marker.touch()

        return False, approval_log

    return None, approval_log


def main():
    parser = argparse.ArgumentParser(description='Interactive trade approval')
    parser.add_argument('trades_file', type=str, help='Path to trades.csv')
    parser.add_argument('--auto-approve', action='store_true', help='Auto-approve without prompting (use with caution)')
    parser.add_argument('--notes', type=str, help='Additional notes for approval log')
    args = parser.parse_args()

    trades_file = Path(args.trades_file)

    # Check file exists
    if not trades_file.exists():
        print(f"Error: Trades file not found: {trades_file}")
        return 1

    # Load trades
    try:
        trades_df = pd.read_csv(trades_file)
    except Exception as e:
        print(f"Error loading trades file: {e}")
        return 1

    # Extract date from path
    date = trades_file.parent.name

    # Print header
    print_header(f"TRADE APPROVAL - {date}")

    # Print summary
    print("Summary:")
    print_trade_summary(trades_df, date)

    # Print safety status
    print_safety_status(trades_file.parent)

    # Get approval
    if args.auto_approve:
        print("⚠ AUTO-APPROVE mode enabled")
        status = 'approved'
        reason = "Auto-approved via --auto-approve flag"
    else:
        status, reason = get_user_approval(trades_df, trades_file)

    print("="*70)

    # Create approval record
    if status == 'approved':
        approved, log_file = create_approval_record(trades_file, status, notes=args.notes)

        print("✓ Trades APPROVED for execution")
        print(f"✓ Approval saved: {trades_file.parent / 'APPROVED'}")
        print(f"✓ Log saved: {log_file}")
        print()
        print("Next steps:")
        print("1. Execute trades manually at your broker")
        print(f"2. Record fills in: {trades_file.parent}/execution_log.csv")
        print("3. Update positions: python scripts/update_positions.py")
        print("="*70)

        return 0

    elif status == 'rejected':
        approved, log_file = create_approval_record(trades_file, status, reason=reason, notes=args.notes)

        print("✗ Trades REJECTED")
        print(f"✗ Rejection saved: {trades_file.parent / 'REJECTED'}")
        print(f"✗ Log saved: {log_file}")
        if reason:
            print(f"Reason: {reason}")
        print()
        print("No trades will be executed today.")
        print("="*70)

        return 1

    elif status == 'edit':
        print("⚠ Edit mode requested")
        print()
        print("To edit trades:")
        print(f"1. Manually edit: {trades_file}")
        print(f"2. Re-run this script: python scripts/approve_trades.py {trades_file}")
        print("="*70)

        return 2

    return 0


if __name__ == '__main__':
    sys.exit(main())
