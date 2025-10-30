#!/usr/bin/env python3
"""
Daily Live Trading Runner

Complete workflow for daily portfolio management:
1. Pre-flight checks (market open, data available)
2. Update data (OHLCV + fundamentals)
3. Load champion model
4. Generate ML scores for universe
5. Detect current market regime
6. Construct portfolio (regime-aware)
7. Generate trade recommendations
8. Run safety checks (kill-switch, limits)
9. Generate reports (CSV + HTML + JSON)
10. Send notifications (optional)

Usage:
    python scripts/daily_live_runner.py --dry-run                    # Test mode (default)
    python scripts/daily_live_runner.py --mode production            # Live mode
    python scripts/daily_live_runner.py --date 2025-10-29           # Backfill specific date
    python scripts/daily_live_runner.py --skip-data-update           # Use existing data
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from loguru import logger

from utils.config_loader import load_config
from src.live.operations import LivePortfolioManager
from src.live.trade_generator import TradeGenerator
from src.live.safety_checks import SafetyValidator
from src.live.monitoring import (
    OpsMonitor,
    calculate_information_coefficient,
    track_model_health_daily
)
from src.portfolio.construct import construct_portfolio
from src.portfolio.regime_detection import RegimeDetector


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Daily Live Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['dry-run', 'production'],
        default='dry-run',
        help='Execution mode (default: dry-run)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (no real trades) - same as --mode dry-run'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Specific date to run (YYYY-MM-DD), default: today'
    )

    parser.add_argument(
        '--skip-data-update',
        action='store_true',
        help='Skip data update step (use existing data)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--capital',
        type=float,
        help='Override portfolio capital (USD)'
    )

    parser.add_argument(
        '--email',
        action='store_true',
        help='Send email notification after completion'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    # Normalize mode
    if args.dry_run:
        args.mode = 'dry-run'

    return args


def check_market_open(date: Optional[str] = None) -> bool:
    """
    Check if market is open on given date

    Args:
        date: Date to check (YYYY-MM-DD), default: today

    Returns:
        True if market is open
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    if date:
        check_date = pd.to_datetime(date)
    else:
        check_date = pd.Timestamp.now()

    # Check if weekday
    if check_date.weekday() >= 5:  # Saturday or Sunday
        logger.warning(f"{check_date.date()} is a weekend - market closed")
        return False

    # Check if US holiday
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=check_date, end=check_date)

    if len(holidays) > 0:
        logger.warning(f"{check_date.date()} is a US holiday - market closed")
        return False

    logger.info(f"✓ Market is open on {check_date.date()}")
    return True


def update_data(config: Dict, skip: bool = False) -> bool:
    """
    Update OHLCV and fundamental data

    Args:
        config: Configuration
        skip: Skip update if True

    Returns:
        True if successful
    """
    if skip:
        logger.info("Skipping data update (--skip-data-update flag)")
        return True

    logger.info("Updating data...")

    try:
        # Import daily update script
        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/daily_update_data.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"Data update failed: {result.stderr}")
            return False

        logger.info("✓ Data update completed successfully")
        return True

    except Exception as e:
        logger.error(f"Data update failed: {e}")
        return False


def load_universe(config: Dict) -> pd.DataFrame:
    """
    Load current stock universe

    Args:
        config: Configuration

    Returns:
        DataFrame with universe stocks
    """
    from src.io.universe import load_sp500_constituents

    logger.info("Loading stock universe...")

    try:
        universe = load_sp500_constituents(config)
        logger.info(f"✓ Loaded {len(universe)} stocks in universe")
        return universe

    except Exception as e:
        logger.error(f"Failed to load universe: {e}")
        return pd.DataFrame()


def load_features(config: Dict, universe: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Load or compute features for universe

    Args:
        config: Configuration
        universe: Universe DataFrame
        date: Date to score (YYYY-MM-DD)

    Returns:
        DataFrame with features
    """
    logger.info(f"Loading features for {len(universe)} stocks...")

    features_dir = Path(config['data']['features'])
    features_file = features_dir / "all_features_with_fundamentals.parquet"

    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        logger.info("Run: python run_core_pipeline.py --steps 3,4 to generate features")
        return pd.DataFrame()

    try:
        features_df = pd.read_parquet(features_file)

        # Filter to date and universe
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_filtered = features_df[
            (features_df['date'] <= date) &
            (features_df['symbol'].isin(universe['symbol']))
        ]

        # Get latest features for each symbol
        features_latest = features_filtered.sort_values('date').groupby('symbol').tail(1)

        logger.info(f"✓ Loaded features for {len(features_latest)} stocks")
        return features_latest

    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        return pd.DataFrame()


def generate_signals(
    manager: LivePortfolioManager,
    model: Any,
    features: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate ML scores for universe

    Args:
        manager: LivePortfolioManager instance
        model: Trained ML model
        features: Features DataFrame

    Returns:
        DataFrame with [date, symbol, ml_score]
    """
    logger.info(f"Generating signals for {len(features)} stocks...")

    try:
        scores = manager.generate_daily_signals(model, features)
        logger.info(f"✓ Generated {len(scores)} signals")
        return scores

    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        return pd.DataFrame()


def construct_live_portfolio(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    config: Dict,
    enable_regime: bool = True
) -> Tuple[Dict[str, float], Optional[Dict]]:
    """
    Construct portfolio with regime adaptation

    Args:
        scores: ML scores
        prices: Price panel
        config: Configuration
        enable_regime: Enable regime adaptation

    Returns:
        (target_weights, regime_info)
    """
    logger.info("Constructing portfolio...")

    try:
        weights = construct_portfolio(
            scored_df=scores,
            price_panel=prices,
            config=config,
            score_col='ml_score',
            enable_regime_adaptation=enable_regime
        )

        # Detect regime for reporting
        regime_info = None
        if enable_regime:
            detector = RegimeDetector(config)
            benchmark_symbol = config.get('reporting', {}).get('benchmark', 'SPY')
            benchmark_data = prices[prices['symbol'] == benchmark_symbol].copy()

            if len(benchmark_data) > 0:
                regime_df = detector.detect(
                    prices=benchmark_data['close'],
                    dates=benchmark_data['date']
                )
                regime_info = detector.get_current_regime(regime_df)

        logger.info(f"✓ Portfolio constructed: {len(weights)} positions")
        return weights, regime_info

    except Exception as e:
        logger.error(f"Portfolio construction failed: {e}")
        return {}, None


def generate_trades(
    generator: TradeGenerator,
    manager: LivePortfolioManager,
    target_weights: Dict[str, float],
    config: Dict,
    date: str
) -> pd.DataFrame:
    """
    Generate trade orders

    Args:
        generator: TradeGenerator instance
        manager: LivePortfolioManager instance
        target_weights: Target portfolio weights
        config: Configuration
        date: Trading date

    Returns:
        DataFrame with trade orders
    """
    logger.info("Generating trade orders...")

    try:
        # Get current positions
        current_positions = manager.get_current_positions()

        # Get current prices
        all_symbols = list(set(list(current_positions.keys()) + list(target_weights.keys())))
        current_prices = manager.get_current_prices(all_symbols, date=date)

        # Get portfolio value
        live_config = config.get('live', {})
        if len(current_positions) == 0:
            # First time - use initial capital
            portfolio_value = live_config.get('initial_capital', 100000)
            logger.info(f"Starting with initial capital: ${portfolio_value:,.0f}")
        else:
            portfolio_value = manager.get_portfolio_value(current_positions, current_prices)

        # Generate trades
        trades = generator.generate_trades(
            current_positions=current_positions,
            target_weights=target_weights,
            current_prices=current_prices,
            portfolio_value=portfolio_value
        )

        if len(trades) > 0:
            # Add execution summary
            summary = generator.create_execution_summary(trades, portfolio_value)
            logger.info(f"✓ Generated {summary['total_trades']} trades "
                       f"(turnover={summary['turnover']:.2%}, costs=${summary['estimated_costs']:.2f})")
        else:
            logger.info("✓ No trades needed (portfolio within tolerance)")

        return trades

    except Exception as e:
        logger.error(f"Trade generation failed: {e}")
        return pd.DataFrame()


def run_safety_checks(
    validator: SafetyValidator,
    target_weights: Dict[str, float],
    trades: pd.DataFrame,
    portfolio_value: float,
    universe: pd.DataFrame
) -> Dict[str, Any]:
    """
    Run comprehensive safety validations

    Args:
        validator: SafetyValidator instance
        target_weights: Target weights
        trades: Trade orders
        portfolio_value: Portfolio value
        universe: Universe with sector data

    Returns:
        Validation results
    """
    logger.info("Running safety checks...")

    try:
        results = validator.validate_all(
            target_weights=target_weights,
            trades=trades,
            portfolio_value=portfolio_value,
            universe_data=universe
        )

        return results

    except Exception as e:
        logger.error(f"Safety validation failed: {e}")
        return {'overall_valid': False, 'all_violations': [str(e)]}


def save_outputs(
    date: str,
    trades: pd.DataFrame,
    target_weights: Dict[str, float],
    scores: pd.DataFrame,
    regime_info: Optional[Dict],
    safety_results: Dict,
    config: Dict
) -> Dict[str, str]:
    """
    Save all outputs (CSV + JSON)

    Args:
        date: Trading date
        trades: Trade orders
        target_weights: Portfolio weights
        scores: ML scores
        regime_info: Regime information
        safety_results: Safety validation results
        config: Configuration

    Returns:
        Dict of output file paths
    """
    logger.info("Saving outputs...")

    output_dir = Path(config.get('live', {}).get('output_dir', 'live'))
    date_dir = output_dir / date
    date_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    try:
        # 1. Save trades
        if len(trades) > 0:
            trades_file = date_dir / 'trades.csv'
            trades.to_csv(trades_file, index=False)
            outputs['trades'] = str(trades_file)
            logger.info(f"✓ Saved trades: {trades_file}")

        # 2. Save portfolio weights
        weights_df = pd.DataFrame([
            {'symbol': symbol, 'weight': weight}
            for symbol, weight in target_weights.items()
        ])
        weights_file = date_dir / 'portfolio_weights.csv'
        weights_df.to_csv(weights_file, index=False)
        outputs['weights'] = str(weights_file)
        logger.info(f"✓ Saved weights: {weights_file}")

        # 3. Save signals (ML scores)
        signals_file = date_dir / 'signals.json'
        signals_data = {
            'date': date,
            'scores': scores[['symbol', 'ml_score']].to_dict('records') if len(scores) > 0 else []
        }
        with open(signals_file, 'w') as f:
            json.dump(signals_data, f, indent=2)
        outputs['signals'] = str(signals_file)

        # 4. Save monitoring log
        monitoring_file = date_dir / 'monitoring_log.json'
        monitoring_data = {
            'date': date,
            'timestamp': datetime.now().isoformat(),
            'regime': regime_info,
            'safety': {
                'overall_valid': safety_results.get('overall_valid', False),
                'violations': safety_results.get('all_violations', [])
            },
            'portfolio': {
                'n_positions': len(target_weights),
                'n_trades': len(trades) if len(trades) > 0 else 0
            }
        }
        with open(monitoring_file, 'w') as f:
            json.dump(monitoring_data, f, indent=2)
        outputs['monitoring'] = str(monitoring_file)

        logger.info(f"✓ Saved {len(outputs)} output files to {date_dir}")
        return outputs

    except Exception as e:
        logger.error(f"Failed to save outputs: {e}")
        return outputs


def generate_html_report(
    date: str,
    trades: pd.DataFrame,
    target_weights: Dict[str, float],
    regime_info: Optional[Dict],
    safety_results: Dict,
    config: Dict
) -> str:
    """
    Generate HTML daily report

    Args:
        date: Trading date
        trades: Trade orders
        target_weights: Portfolio weights
        regime_info: Regime information
        safety_results: Safety results
        config: Configuration

    Returns:
        Path to HTML report
    """
    logger.info("Generating HTML report...")

    output_dir = Path(config.get('live', {}).get('output_dir', 'live'))
    date_dir = output_dir / date
    report_file = date_dir / 'report.html'

    try:
        # Generate simple HTML report
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Daily Trading Report - {date}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .value {{ font-size: 24px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        .buy {{ color: #27ae60; }}
        .sell {{ color: #e74c3c; }}
        .status-ok {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-error {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <h1>📊 Daily Trading Report</h1>
    <p><strong>Date:</strong> {date}</p>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

        # Regime section
        if regime_info:
            html += f"""
    <div class="section">
        <h2>🌐 Market Regime</h2>
        <div class="metric">
            <div class="label">Current Regime</div>
            <div class="value">{regime_info['regime_name']}</div>
        </div>
        <div class="metric">
            <div class="label">Risk Multiplier</div>
            <div class="value">{regime_info['risk_multiplier']:.2f}x</div>
        </div>
"""
            if 'volatility' in regime_info:
                html += f"""
        <div class="metric">
            <div class="label">Volatility</div>
            <div class="value">{regime_info['volatility']:.2%}</div>
        </div>
        <div class="metric">
            <div class="label">Drawdown</div>
            <div class="value">{regime_info['drawdown']:.2%}</div>
        </div>
"""
            html += "    </div>\n"

        # Portfolio section
        html += f"""
    <div class="section">
        <h2>💼 Portfolio</h2>
        <div class="metric">
            <div class="label">Positions</div>
            <div class="value">{len(target_weights)}</div>
        </div>
        <div class="metric">
            <div class="label">Trades</div>
            <div class="value">{len(trades) if len(trades) > 0 else 0}</div>
        </div>
    </div>
"""

        # Trades section
        if len(trades) > 0:
            html += """
    <div class="section">
        <h2>📝 Trade Orders</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Side</th>
                <th>Shares</th>
                <th>Price</th>
                <th>Notional</th>
                <th>Weight Change</th>
            </tr>
"""
            for _, trade in trades.head(20).iterrows():
                side_class = 'buy' if trade['side'] == 'BUY' else 'sell'
                html += f"""
            <tr>
                <td><strong>{trade['symbol']}</strong></td>
                <td class="{side_class}">{trade['side']}</td>
                <td>{trade['shares']:.0f}</td>
                <td>${trade['price']:.2f}</td>
                <td>${trade['notional']:,.0f}</td>
                <td>{trade['weight_change']:.2%}</td>
            </tr>
"""
            html += "        </table>\n    </div>\n"

        # Safety section
        status_class = 'status-ok' if safety_results.get('overall_valid') else 'status-error'
        status_text = 'PASSED' if safety_results.get('overall_valid') else 'FAILED'

        html += f"""
    <div class="section">
        <h2>✅ Safety Checks</h2>
        <p class="{status_class}"><strong>Status: {status_text}</strong></p>
"""
        if safety_results.get('all_violations'):
            html += "        <h3>Violations:</h3>\n        <ul>\n"
            for violation in safety_results['all_violations']:
                html += f"            <li>{violation}</li>\n"
            html += "        </ul>\n"

        html += "    </div>\n"

        html += """
    <div style="text-align: center; margin-top: 40px; color: #95a5a6; font-size: 12px;">
        <p>🤖 Generated by ML Trading System</p>
        <p>Powered by Claude Code</p>
    </div>
</body>
</html>
"""

        with open(report_file, 'w') as f:
            f.write(html)

        logger.info(f"✓ HTML report saved: {report_file}")
        return str(report_file)

    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        return ""


def main():
    """Main execution function"""
    args = parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add("logs/daily_live_runner.log", rotation="10 MB", level="DEBUG")

    logger.info("="*80)
    logger.info("DAILY LIVE TRADING RUNNER")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Date: {args.date or 'TODAY'}")
    logger.info("="*80)

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"✓ Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Override config with CLI args
    if args.mode == 'production':
        config.setdefault('live', {})['dry_run'] = False
    else:
        config.setdefault('live', {})['dry_run'] = True

    if args.capital:
        config.setdefault('live', {})['initial_capital'] = args.capital

    # Determine date
    if args.date:
        trade_date = args.date
    else:
        trade_date = datetime.now().strftime('%Y-%m-%d')

    # Step 1: Check market open
    if not check_market_open(trade_date):
        logger.warning("Market is closed - exiting")
        return 0

    # Step 2: Update data
    if not update_data(config, skip=args.skip_data_update):
        logger.error("Data update failed - exiting")
        return 1

    # Step 3: Load universe
    universe = load_universe(config)
    if len(universe) == 0:
        logger.error("Failed to load universe - exiting")
        return 1

    # Step 4: Load features
    features = load_features(config, universe, trade_date)
    if len(features) == 0:
        logger.error("Failed to load features - exiting")
        return 1

    # Step 5: Load model
    manager = LivePortfolioManager(config)
    try:
        model = manager.load_latest_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Step 6: Generate signals
    scores = generate_signals(manager, model, features)
    if len(scores) == 0:
        logger.error("Failed to generate signals - exiting")
        return 1

    # Step 7: Load prices
    logger.info("Loading price data...")
    prices_list = []
    for symbol in universe['symbol'].unique():
        symbol_prices = manager.get_current_prices([symbol], date=trade_date)
        if symbol in symbol_prices:
            prices_list.append({
                'symbol': symbol,
                'close': symbol_prices[symbol],
                'date': pd.to_datetime(trade_date)
            })
    price_panel = pd.DataFrame(prices_list)

    # Step 8: Construct portfolio
    target_weights, regime_info = construct_live_portfolio(
        scores=scores,
        prices=price_panel,
        config=config,
        enable_regime=True
    )

    if len(target_weights) == 0:
        logger.warning("Empty portfolio - no positions")

    # Step 9: Generate trades
    generator = TradeGenerator(config)
    trades = generate_trades(generator, manager, target_weights, config, trade_date)

    # Step 10: Get portfolio value for safety checks
    current_positions = manager.get_current_positions()
    all_symbols = list(set(list(current_positions.keys()) + list(target_weights.keys())))
    current_prices = manager.get_current_prices(all_symbols, date=trade_date)

    if len(current_positions) == 0:
        portfolio_value = config.get('live', {}).get('initial_capital', 100000)
    else:
        portfolio_value = manager.get_portfolio_value(current_positions, current_prices)

    # Step 11: Run safety checks
    validator = SafetyValidator(config)
    safety_results = run_safety_checks(
        validator=validator,
        target_weights=target_weights,
        trades=trades,
        portfolio_value=portfolio_value,
        universe=universe
    )

    # Step 12: Save outputs
    outputs = save_outputs(
        date=trade_date,
        trades=trades,
        target_weights=target_weights,
        scores=scores,
        regime_info=regime_info,
        safety_results=safety_results,
        config=config
    )

    # Step 13: Generate HTML report
    report_file = generate_html_report(
        date=trade_date,
        trades=trades,
        target_weights=target_weights,
        regime_info=regime_info,
        safety_results=safety_results,
        config=config
    )

    # Step 14: Summary
    logger.info("="*80)
    logger.info("EXECUTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Date: {trade_date}")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Portfolio: {len(target_weights)} positions")
    logger.info(f"Trades: {len(trades) if len(trades) > 0 else 0}")
    logger.info(f"Safety: {'✓ PASSED' if safety_results.get('overall_valid') else '✗ FAILED'}")

    if regime_info:
        logger.info(f"Regime: {regime_info['regime_name']} ({regime_info['risk_multiplier']:.2f}x)")

    logger.info(f"\nOutputs saved to: live/{trade_date}/")
    for output_type, output_path in outputs.items():
        logger.info(f"  - {output_type}: {output_path}")

    if report_file:
        logger.info(f"\n📊 View report: {report_file}")

    logger.info("="*80)

    # Step 15: Email notification (if requested)
    if args.email:
        logger.info("Email notification requested but not implemented yet")

    # Return code
    if safety_results.get('overall_valid'):
        logger.info("✅ Daily run completed successfully")
        return 0
    else:
        logger.warning("⚠️  Daily run completed with safety violations")
        return 1


if __name__ == "__main__":
    sys.exit(main())
