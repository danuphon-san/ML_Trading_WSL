"""
Validate trade recommendations before execution

This script performs comprehensive validation of trade recommendations
to ensure they are safe and reasonable before manual or automated execution.

Usage:
    python scripts/validate_trades.py live/2025-10-30/trades.csv
    python scripts/validate_trades.py live/2025-10-30/trades.csv --strict
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TradeValidator:
    """Validates trade recommendations"""

    def __init__(self, strict_mode=False):
        self.strict_mode = strict_mode
        self.warnings = []
        self.errors = []

    def validate(self, trades_df: pd.DataFrame) -> dict:
        """Run all validation checks"""
        results = {
            'valid': True,
            'checks': {}
        }

        # Run validation checks
        results['checks']['format'] = self._check_format(trades_df)
        results['checks']['prices'] = self._check_prices(trades_df)
        results['checks']['quantities'] = self._check_quantities(trades_df)
        results['checks']['notional'] = self._check_notional(trades_df)
        results['checks']['costs'] = self._check_costs(trades_df)
        results['checks']['diversification'] = self._check_diversification(trades_df)

        # Determine overall validity
        results['valid'] = all(check['passed'] for check in results['checks'].values())
        results['warnings'] = self.warnings
        results['errors'] = self.errors

        return results

    def _check_format(self, trades_df: pd.DataFrame) -> dict:
        """Check DataFrame has required columns"""
        required_cols = ['symbol', 'side', 'shares', 'price', 'notional']
        missing_cols = [col for col in required_cols if col not in trades_df.columns]

        if missing_cols:
            self.errors.append(f"Missing required columns: {missing_cols}")
            return {'passed': False, 'message': f"Missing columns: {missing_cols}"}

        return {'passed': True, 'message': 'All required columns present'}

    def _check_prices(self, trades_df: pd.DataFrame) -> dict:
        """Validate prices are reasonable"""
        issues = []

        # Check for non-positive prices
        invalid_prices = trades_df[trades_df['price'] <= 0]
        if len(invalid_prices) > 0:
            self.errors.append(f"{len(invalid_prices)} trades have price <= 0")
            issues.append(f"{len(invalid_prices)} invalid prices")

        # Check for extremely high prices (> $1000)
        high_prices = trades_df[trades_df['price'] > 1000]
        if len(high_prices) > 0:
            self.warnings.append(f"{len(high_prices)} trades have price > $1000")
            issues.append(f"{len(high_prices)} very high prices")

        # Check for penny stocks (< $5)
        penny_stocks = trades_df[trades_df['price'] < 5]
        if len(penny_stocks) > 0:
            msg = f"{len(penny_stocks)} penny stocks (price < $5)"
            if self.strict_mode:
                self.errors.append(msg)
                issues.append(msg)
            else:
                self.warnings.append(msg)

        if issues and self.strict_mode:
            return {'passed': False, 'message': '; '.join(issues)}

        return {'passed': len(issues) == 0 or not self.strict_mode,
                'message': 'Prices valid' if not issues else f"Issues: {'; '.join(issues)}"}

    def _check_quantities(self, trades_df: pd.DataFrame) -> dict:
        """Validate share quantities"""
        issues = []

        # Check for fractional shares
        fractional = trades_df[trades_df['shares'] != trades_df['shares'].astype(int)]
        if len(fractional) > 0:
            self.warnings.append(f"{len(fractional)} trades have fractional shares")
            issues.append(f"{len(fractional)} fractional shares")

        # Check for zero shares
        zero_shares = trades_df[trades_df['shares'] == 0]
        if len(zero_shares) > 0:
            self.errors.append(f"{len(zero_shares)} trades have zero shares")
            issues.append(f"{len(zero_shares)} zero shares")

        # Check for negative shares
        negative_shares = trades_df[trades_df['shares'] < 0]
        if len(negative_shares) > 0:
            self.errors.append(f"{len(negative_shares)} trades have negative shares")
            issues.append(f"{len(negative_shares)} negative shares")

        if any('zero' in i or 'negative' in i for i in issues):
            return {'passed': False, 'message': '; '.join(issues)}

        return {'passed': True, 'message': 'Quantities valid' if not issues else f"Warnings: {'; '.join(issues)}"}

    def _check_notional(self, trades_df: pd.DataFrame) -> dict:
        """Validate notional values"""
        issues = []

        # Check notional consistency
        calculated_notional = (trades_df['shares'] * trades_df['price']).round(2)
        actual_notional = trades_df['notional'].round(2)
        mismatch = np.abs(calculated_notional - actual_notional) > 0.50

        if mismatch.any():
            n_mismatch = mismatch.sum()
            self.warnings.append(f"{n_mismatch} trades have notional mismatch > $0.50")
            issues.append(f"{n_mismatch} notional mismatches")

        # Check for very small trades (< $100)
        if 'min_trade_size' in trades_df.columns:
            min_size = 100  # Default minimum
        else:
            min_size = 100

        small_trades = trades_df[trades_df['notional'] < min_size]
        if len(small_trades) > 0:
            self.warnings.append(f"{len(small_trades)} trades below ${min_size} minimum")
            issues.append(f"{len(small_trades)} small trades")

        return {'passed': True, 'message': 'Notional values valid' if not issues else f"Warnings: {'; '.join(issues)}"}

    def _check_costs(self, trades_df: pd.DataFrame) -> dict:
        """Validate transaction costs are reasonable"""
        issues = []

        # Check if cost columns exist
        if 'commission_est' in trades_df.columns and 'slippage_est' in trades_df.columns:
            total_costs = trades_df['commission_est'] + trades_df['slippage_est']
            total_notional = trades_df['notional'].sum()

            cost_pct = (total_costs.sum() / total_notional * 100) if total_notional > 0 else 0

            # Costs should be < 0.5% of notional
            if cost_pct > 0.5:
                self.warnings.append(f"Total costs ({cost_pct:.2f}%) exceed 0.5% of notional")
                issues.append(f"High costs: {cost_pct:.2f}%")

            # Check for individual expensive trades (> 1% cost)
            if 'total_cost_est' in trades_df.columns:
                expensive = trades_df[
                    (trades_df['total_cost_est'] / trades_df['notional']) > 0.01
                ]
                if len(expensive) > 0:
                    self.warnings.append(f"{len(expensive)} trades have cost > 1% of notional")
                    issues.append(f"{len(expensive)} expensive trades")

        return {'passed': True, 'message': 'Costs reasonable' if not issues else f"Warnings: {'; '.join(issues)}"}

    def _check_diversification(self, trades_df: pd.DataFrame) -> dict:
        """Check trade concentration"""
        issues = []

        if len(trades_df) == 0:
            return {'passed': False, 'message': 'No trades'}

        # Check for single stock dominating
        total_notional = trades_df['notional'].abs().sum()
        if total_notional > 0:
            trades_df['pct_of_total'] = trades_df['notional'].abs() / total_notional

            max_concentration = trades_df['pct_of_total'].max()
            if max_concentration > 0.15:  # 15% limit
                max_symbol = trades_df.loc[trades_df['pct_of_total'].idxmax(), 'symbol']
                msg = f"Single trade ({max_symbol}) is {max_concentration:.1%} of total notional"

                if self.strict_mode:
                    self.errors.append(msg)
                    issues.append(msg)
                else:
                    self.warnings.append(msg)

        # Check for too many trades (potential over-diversification)
        if len(trades_df) > 50:
            self.warnings.append(f"High trade count: {len(trades_df)} trades")
            issues.append(f"Many trades: {len(trades_df)}")

        if issues and self.strict_mode and any('Single trade' in i for i in issues):
            return {'passed': False, 'message': '; '.join(issues)}

        return {'passed': True, 'message': 'Diversification acceptable' if not issues else f"Warnings: {'; '.join(issues)}"}


def print_validation_report(results: dict, trades_df: pd.DataFrame):
    """Print formatted validation report"""
    print("\n" + "="*70)
    print("TRADE VALIDATION REPORT")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Buy Orders: {(trades_df['side'] == 'buy').sum()}")
    print(f"Sell Orders: {(trades_df['side'] == 'sell').sum()}")
    print(f"Total Notional: ${trades_df['notional'].abs().sum():,.2f}")
    print()

    # Print check results
    print("VALIDATION CHECKS:")
    print("-" * 70)
    for check_name, check_result in results['checks'].items():
        status = "✓ PASS" if check_result['passed'] else "✗ FAIL"
        print(f"{check_name.upper():20s} {status:10s} {check_result['message']}")
    print()

    # Print warnings
    if results['warnings']:
        print("WARNINGS:")
        print("-" * 70)
        for warning in results['warnings']:
            print(f"⚠ {warning}")
        print()

    # Print errors
    if results['errors']:
        print("ERRORS:")
        print("-" * 70)
        for error in results['errors']:
            print(f"✗ {error}")
        print()

    # Overall result
    print("="*70)
    if results['valid']:
        print("✓ VALIDATION PASSED - Trades are ready for execution")
    else:
        print("✗ VALIDATION FAILED - Do NOT execute trades, fix errors above")
    print("="*70)
    print()


def main():
    parser = argparse.ArgumentParser(description='Validate trade recommendations')
    parser.add_argument('trades_file', type=str, help='Path to trades.csv file')
    parser.add_argument('--strict', action='store_true', help='Enable strict validation mode')
    parser.add_argument('--output', type=str, help='Save validation report to file')
    args = parser.parse_args()

    # Check file exists
    trades_path = Path(args.trades_file)
    if not trades_path.exists():
        print(f"Error: File not found: {trades_path}")
        return 1

    # Load trades
    try:
        trades_df = pd.read_csv(trades_path)
        print(f"Loaded {len(trades_df)} trades from {trades_path}")
    except Exception as e:
        print(f"Error loading trades file: {e}")
        return 1

    # Run validation
    validator = TradeValidator(strict_mode=args.strict)
    results = validator.validate(trades_df)

    # Print report
    print_validation_report(results, trades_df)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            # Redirect stdout to file
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            print_validation_report(results, trades_df)
            sys.stdout = old_stdout
        print(f"Validation report saved to: {output_path}")

    # Return exit code
    return 0 if results['valid'] else 1


if __name__ == '__main__':
    sys.exit(main())
