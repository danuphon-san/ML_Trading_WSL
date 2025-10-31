"""
Reconcile positions after daily trading

This script compares recommended trades, actual broker fills, and current positions
to ensure everything matches and identifies discrepancies.

Usage:
    python scripts/reconcile_positions.py \
        --trades live/2025-10-30/trades.csv \
        --fills live/2025-10-30/broker_fills.csv \
        --current-positions live/current_positions.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class PositionReconciler:
    """Reconciles positions between system and broker"""

    def __init__(self):
        self.discrepancies = []
        self.metrics = {}

    def reconcile(
        self,
        trades_recommended: pd.DataFrame,
        fills_actual: pd.DataFrame,
        positions_current: pd.DataFrame,
        prices_current: Dict[str, float]
    ) -> dict:
        """
        Perform full position reconciliation

        Returns:
            dict with reconciliation results
        """
        results = {
            'trade_execution': self._reconcile_trade_execution(trades_recommended, fills_actual),
            'position_accuracy': self._reconcile_positions(fills_actual, positions_current),
            'cost_analysis': self._analyze_costs(trades_recommended, fills_actual),
            'updated_positions': self._calculate_updated_positions(positions_current, fills_actual, prices_current),
            'discrepancies': self.discrepancies,
            'metrics': self.metrics
        }

        return results

    def _reconcile_trade_execution(
        self,
        recommended: pd.DataFrame,
        actual: pd.DataFrame
    ) -> dict:
        """Compare recommended trades vs actual fills"""

        # Match trades by symbol and side
        rec_grouped = recommended.groupby(['symbol', 'side']).agg({
            'shares': 'sum',
            'price': 'mean',
            'notional': 'sum'
        }).reset_index()

        actual_grouped = actual.groupby(['symbol', 'side']).agg({
            'shares': 'sum',
            'fill_price': 'mean',
            'notional': 'sum'
        }).reset_index() if 'fill_price' in actual.columns else None

        if actual_grouped is None:
            return {
                'recommended_count': len(recommended),
                'executed_count': 0,
                'execution_rate': 0.0,
                'unfilled': recommended['symbol'].tolist()
            }

        # Find unfilled trades
        rec_set = set(zip(rec_grouped['symbol'], rec_grouped['side']))
        actual_set = set(zip(actual_grouped['symbol'], actual_grouped['side']))
        unfilled = rec_set - actual_set

        # Calculate execution rate
        execution_rate = len(actual_set) / len(rec_set) if len(rec_set) > 0 else 0.0

        # Find partial fills
        partial_fills = []
        for symbol, side in rec_set & actual_set:
            rec_shares = rec_grouped[
                (rec_grouped['symbol'] == symbol) & (rec_grouped['side'] == side)
            ]['shares'].values[0]

            actual_shares = actual_grouped[
                (actual_grouped['symbol'] == symbol) & (actual_grouped['side'] == side)
            ]['shares'].values[0]

            if abs(rec_shares - actual_shares) / rec_shares > 0.05:  # >5% difference
                partial_fills.append({
                    'symbol': symbol,
                    'side': side,
                    'recommended': rec_shares,
                    'actual': actual_shares,
                    'fill_rate': actual_shares / rec_shares
                })

        if partial_fills:
            self.discrepancies.append(f"{len(partial_fills)} partial fills detected")

        return {
            'recommended_count': len(rec_set),
            'executed_count': len(actual_set),
            'execution_rate': execution_rate,
            'unfilled': [f"{s}-{sd}" for s, sd in unfilled],
            'partial_fills': partial_fills
        }

    def _reconcile_positions(
        self,
        fills: pd.DataFrame,
        current_positions: pd.DataFrame
    ) -> dict:
        """Verify current positions match expected after fills"""

        # This is a simplified check
        # In production, you'd integrate with broker API to get actual positions

        position_symbols = set(current_positions['symbol'].tolist())
        expected_symbols = set(fills['symbol'].tolist()) if len(fills) > 0 else set()

        # Symbols that should have positions but don't
        missing = expected_symbols - position_symbols

        # Symbols that have positions but shouldn't
        unexpected = position_symbols - expected_symbols

        if missing:
            self.discrepancies.append(f"Missing positions: {missing}")

        # Note: 'unexpected' is not necessarily an error (could be from previous trades)

        return {
            'matches': len(position_symbols & expected_symbols),
            'missing': list(missing),
            'unexpected': list(unexpected)
        }

    def _analyze_costs(
        self,
        recommended: pd.DataFrame,
        actual: pd.DataFrame
    ) -> dict:
        """Analyze transaction costs vs estimates"""

        # Estimated costs
        estimated_commission = recommended['commission_est'].sum() if 'commission_est' in recommended.columns else 0
        estimated_slippage = recommended['slippage_est'].sum() if 'slippage_est' in recommended.columns else 0
        estimated_total = estimated_commission + estimated_slippage

        # Actual costs (if available in fills)
        actual_commission = actual['commission'].sum() if 'commission' in actual.columns else 0
        actual_slippage = 0.0
        actual_total = actual_commission

        # Calculate slippage if we have both recommended and fill prices
        if 'fill_price' in actual.columns and len(actual) > 0:
            # Merge to compare prices
            comparison = pd.merge(
                recommended[['symbol', 'side', 'shares', 'price']],
                actual[['symbol', 'side', 'shares', 'fill_price']],
                on=['symbol', 'side'],
                how='inner',
                suffixes=('_rec', '_actual')
            )

            if len(comparison) > 0:
                # Calculate slippage per trade
                comparison['slippage'] = np.where(
                    comparison['side'] == 'buy',
                    (comparison['fill_price'] - comparison['price']) * comparison['shares_actual'],
                    (comparison['price'] - comparison['fill_price']) * comparison['shares_actual']
                )
                actual_slippage = comparison['slippage'].sum()
                actual_total = actual_commission + actual_slippage

        # Calculate variances
        commission_var = actual_commission - estimated_commission
        slippage_var = actual_slippage - estimated_slippage
        total_var = actual_total - estimated_total

        self.metrics['cost_variance_pct'] = (total_var / estimated_total * 100) if estimated_total > 0 else 0

        return {
            'estimated': {
                'commission': estimated_commission,
                'slippage': estimated_slippage,
                'total': estimated_total
            },
            'actual': {
                'commission': actual_commission,
                'slippage': actual_slippage,
                'total': actual_total
            },
            'variance': {
                'commission': commission_var,
                'slippage': slippage_var,
                'total': total_var
            }
        }

    def _calculate_updated_positions(
        self,
        current_positions: pd.DataFrame,
        fills: pd.DataFrame,
        prices: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate what positions should be after today's trades"""

        # Start with current positions
        updated = current_positions.copy()

        # Apply fills
        for _, fill in fills.iterrows():
            symbol = fill['symbol']
            shares = fill['shares']
            side = fill['side']

            # Find position in updated df
            pos_idx = updated[updated['symbol'] == symbol].index

            if len(pos_idx) > 0:
                # Update existing position
                idx = pos_idx[0]
                current_shares = updated.loc[idx, 'shares']

                if side == 'buy':
                    new_shares = current_shares + shares
                else:  # sell
                    new_shares = current_shares - shares

                if new_shares > 0:
                    updated.loc[idx, 'shares'] = new_shares
                else:
                    # Position closed
                    updated = updated.drop(idx)
            else:
                # New position
                if side == 'buy':
                    new_row = pd.DataFrame([{
                        'symbol': symbol,
                        'shares': shares,
                        'avg_cost': fill.get('fill_price', prices.get(symbol, 0)),
                        'current_value': 0,
                        'weight': 0,
                        'last_updated': datetime.now().strftime('%Y-%m-%d')
                    }])
                    updated = pd.concat([updated, new_row], ignore_index=True)

        # Update current values and weights
        if len(updated) > 0 and len(prices) > 0:
            updated['current_price'] = updated['symbol'].map(prices)
            updated['current_value'] = updated['shares'] * updated['current_price']
            total_value = updated['current_value'].sum()
            updated['weight'] = updated['current_value'] / total_value if total_value > 0 else 0

        return updated


def print_reconciliation_report(results: dict, date: str):
    """Print formatted reconciliation report"""
    print("\n" + "="*70)
    print(f"POSITION RECONCILIATION REPORT - {date}")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Trade Execution
    print("TRADE EXECUTION:")
    print("-" * 70)
    exec_results = results['trade_execution']
    print(f"Trades Recommended: {exec_results['recommended_count']}")
    print(f"Trades Executed:    {exec_results['executed_count']}")
    print(f"Execution Rate:     {exec_results['execution_rate']:.1%}")

    if exec_results['unfilled']:
        print(f"\nUnfilled Orders ({len(exec_results['unfilled'])}):")
        for trade in exec_results['unfilled'][:10]:  # Show first 10
            print(f"  - {trade}")
        if len(exec_results['unfilled']) > 10:
            print(f"  ... and {len(exec_results['unfilled']) - 10} more")

    if exec_results.get('partial_fills'):
        print(f"\nPartial Fills ({len(exec_results['partial_fills'])}):")
        for pf in exec_results['partial_fills']:
            print(f"  - {pf['symbol']} {pf['side']}: {pf['actual']}/{pf['recommended']} shares ({pf['fill_rate']:.1%})")

    print()

    # Cost Analysis
    print("COST ANALYSIS:")
    print("-" * 70)
    costs = results['cost_analysis']
    print(f"{'':20s} {'Estimated':>12s} {'Actual':>12s} {'Variance':>12s}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'Commission':20s} ${costs['estimated']['commission']:>11,.2f} ${costs['actual']['commission']:>11,.2f} ${costs['variance']['commission']:>11,.2f}")
    print(f"{'Slippage':20s} ${costs['estimated']['slippage']:>11,.2f} ${costs['actual']['slippage']:>11,.2f} ${costs['variance']['slippage']:>11,.2f}")
    print(f"{'Total':20s} ${costs['estimated']['total']:>11,.2f} ${costs['actual']['total']:>11,.2f} ${costs['variance']['total']:>11,.2f}")

    cost_var_pct = results['metrics'].get('cost_variance_pct', 0)
    if abs(cost_var_pct) > 20:
        print(f"\n⚠ Warning: Cost variance of {cost_var_pct:+.1f}% exceeds 20% threshold")
    print()

    # Position Accuracy
    print("POSITION ACCURACY:")
    print("-" * 70)
    pos_results = results['position_accuracy']
    print(f"Matching Positions: {pos_results['matches']}")

    if pos_results['missing']:
        print(f"\n⚠ Missing Positions ({len(pos_results['missing'])}):")
        for symbol in pos_results['missing']:
            print(f"  - {symbol}")

    if pos_results['unexpected']:
        print(f"\nUnexpected Positions ({len(pos_results['unexpected'])}):")
        print(f"  (These may be from previous trades)")
        for symbol in pos_results['unexpected'][:5]:
            print(f"  - {symbol}")

    print()

    # Discrepancies
    if results['discrepancies']:
        print("DISCREPANCIES:")
        print("-" * 70)
        for disc in results['discrepancies']:
            print(f"⚠ {disc}")
        print()

    # Summary
    print("="*70)
    if not results['discrepancies']:
        print("✓ RECONCILIATION COMPLETE - No major discrepancies found")
    else:
        print(f"⚠ RECONCILIATION COMPLETE - {len(results['discrepancies'])} discrepancies found")
        print("  Review discrepancies above and update positions manually if needed")
    print("="*70)
    print()


def main():
    parser = argparse.ArgumentParser(description='Reconcile positions after trading')
    parser.add_argument('--trades', type=str, required=True, help='Path to trades.csv (recommended)')
    parser.add_argument('--fills', type=str, help='Path to broker_fills.csv (actual fills)')
    parser.add_argument('--current-positions', type=str, default='live/current_positions.csv',
                       help='Path to current positions file')
    parser.add_argument('--date', type=str, help='Trading date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Save updated positions to file')
    args = parser.parse_args()

    # Determine date
    date = args.date or datetime.now().strftime('%Y-%m-%d')

    # Load trades
    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"Error: Trades file not found: {trades_path}")
        return 1

    trades_df = pd.read_csv(trades_path)
    print(f"Loaded {len(trades_df)} recommended trades")

    # Load fills (if provided)
    if args.fills:
        fills_path = Path(args.fills)
        if fills_path.exists():
            fills_df = pd.read_csv(fills_path)
            print(f"Loaded {len(fills_df)} actual fills")
        else:
            print(f"Warning: Fills file not found: {fills_path}")
            print("Assuming all recommended trades were executed at recommended prices")
            # Create synthetic fills from trades
            fills_df = trades_df.copy()
            fills_df['fill_price'] = fills_df['price']
            fills_df['commission'] = fills_df.get('commission_est', 0)
    else:
        print("No fills file provided, assuming all recommended trades were executed")
        fills_df = trades_df.copy()
        fills_df['fill_price'] = fills_df['price']
        fills_df['commission'] = fills_df.get('commission_est', 0)

    # Load current positions
    positions_path = Path(args.current_positions)
    if positions_path.exists():
        positions_df = pd.read_csv(positions_path)
        print(f"Loaded {len(positions_df)} current positions")
    else:
        print(f"Warning: Positions file not found: {positions_path}")
        print("Starting with empty portfolio")
        positions_df = pd.DataFrame(columns=['symbol', 'shares', 'avg_cost'])

    # Get current prices (simplified - in production, fetch from data)
    prices = {}
    if 'price' in trades_df.columns:
        prices = dict(zip(trades_df['symbol'], trades_df['price']))

    # Run reconciliation
    reconciler = PositionReconciler()
    results = reconciler.reconcile(trades_df, fills_df, positions_df, prices)

    # Print report
    print_reconciliation_report(results, date)

    # Save updated positions if requested
    if args.output:
        output_path = Path(args.output)
        results['updated_positions'].to_csv(output_path, index=False)
        print(f"\nUpdated positions saved to: {output_path}")
    else:
        # Default: save to current_positions.csv
        output_path = positions_path
        results['updated_positions'].to_csv(output_path, index=False)
        print(f"\nCurrent positions updated: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
