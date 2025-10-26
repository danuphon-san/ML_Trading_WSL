#!/usr/bin/env python3
"""
Enhancements & Ops Pipeline Orchestrator (Steps 12-15)

Implements instruction2.md:
- Regime detection (risk-on/off)
- Sleeve allocation (equities/crypto/cash)
- Turnover management
- Operations monitoring & reporting

Requires core pipeline artifacts from run_core_pipeline.py

Produces 4 enhancement artifacts:
1. data/results/regime_state.csv
2. data/results/sleeve_allocation.json
3. data/results/turnover_report.json
4. data/results/ops_report_{YYYYMMDD}.html
"""
import sys
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.cli_parser import create_enhancements_parser, parse_step_ranges, validate_step_ranges, print_execution_plan
from utils.pipeline_utils import PipelineTracker, run_with_error_handling, validate_artifacts, get_step_name, send_email_summary

from src.io.results_saver import ResultsSaver

from src.portfolio.regime_detection import run_regime_detection
from src.portfolio.sleeve_allocation import run_sleeve_allocation
from src.portfolio.turnover_manager import run_turnover_management
from src.live.monitoring import run_monitoring


def main():
    """Run enhancements & ops pipeline"""

    # Parse arguments
    parser = create_enhancements_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add("logs/enhancements_pipeline.log", rotation="10 MB", level="DEBUG")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Parse steps to execute
    steps_to_run = parse_step_ranges(args.steps, args.skip)
    validate_step_ranges(steps_to_run, valid_range=(12, 15))

    # Step names for display
    step_names = {i: get_step_name(i) for i in range(12, 16)}

    if args.dry_run:
        print_execution_plan(steps_to_run, step_names)
        return

    # Initialize result saver
    saver = ResultsSaver()

    # Validate core artifacts if requested
    if args.validate_core:
        logger.info("Validating core pipeline artifacts...")
        core_status = saver.validate_core_artifacts()

        if not all(core_status.values()):
            logger.error("Core artifacts are missing! Run run_core_pipeline.py first.")
            sys.exit(1)

        logger.info("✓ All core artifacts present")

    # Initialize tracking
    tracker = PipelineTracker()
    tracker.start()

    # Load core artifacts
    logger.info("Loading core pipeline artifacts...")

    try:
        weights_df = saver.load_weights_df()
        backtest_results = saver.load_backtest_results()

        # Reconstruct equity curve from backtest results
        if isinstance(backtest_results.get('equity_curve'), list):
            equity_curve = pd.DataFrame(backtest_results['equity_curve'])
        else:
            equity_curve = pd.read_csv('data/reports/equity_curve.csv')
            equity_curve['date'] = pd.to_datetime(equity_curve['date'])

        # Load price panel for regime detection
        # Note: we use the core features file which has price data
        features_df = saver.load_features_with_fundamentals()
        price_panel = features_df[['date', 'symbol', 'close']].drop_duplicates()

        logger.info(f"✓ Loaded core artifacts:")
        logger.info(f"  Weights: {len(weights_df):,} records")
        logger.info(f"  Equity curve: {len(equity_curve)} periods")
        logger.info(f"  Price panel: {len(price_panel):,} records")

    except FileNotFoundError as e:
        logger.error(f"Failed to load core artifacts: {e}")
        logger.error("Run run_core_pipeline.py first to generate required artifacts")
        sys.exit(1)

    # Shared state across steps
    state = {
        'weights_df': weights_df,
        'equity_curve': equity_curve,
        'price_panel': price_panel,
        'backtest_results': backtest_results,
        'regime_df': None,
        'regime_info': None,
        'allocation_result': None,
        'turnover_adjusted_weights': None,
        'turnover_report': None,
        'monitoring_result': None
    }

    # ========================================================================
    # STEP 12: Regime Detection
    # ========================================================================

    def step_12_regime_detection():
        """Detect market regimes (risk-on/off)"""
        logger.info("Running regime detection")

        benchmark = config.get('reporting', {}).get('benchmark', 'SPY')

        # Run regime detection
        state['regime_df'] = run_regime_detection(config, state['price_panel'], benchmark)

        # Get current regime info
        from src.portfolio.regime_detection import RegimeDetector
        detector = RegimeDetector(config)
        state['regime_info'] = detector.get_current_regime(state['regime_df'])

        # Save regime state (artifact #1)
        saver.save_regime_state(state['regime_df'])

        logger.info(f"✓ Regime detection complete: {len(state['regime_df'])} periods analyzed")
        logger.info(f"  Current regime: {state['regime_info']['regime_name']}")

    if 12 in steps_to_run:
        run_with_error_handling(
            step_12_regime_detection, 12, step_names[12], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 13: Sleeve Allocation
    # ========================================================================

    def step_13_sleeve_allocation():
        """Allocate capital across sleeves (equities/crypto/cash)"""
        logger.info("Running sleeve allocation")

        # For now, we only have equity curve
        # Crypto curve would come from a separate crypto pipeline
        crypto_curve = None  # TODO: integrate crypto pipeline when available

        state['allocation_result'] = run_sleeve_allocation(
            config,
            state['equity_curve'],
            crypto_curve
        )

        # Save sleeve allocation (artifact #2)
        saver.save_sleeve_allocation(state['allocation_result'])

        allocation = state['allocation_result']['allocation']
        logger.info(f"✓ Sleeve allocation complete:")
        logger.info(f"  Equities: {allocation['equities']:.2%}")
        logger.info(f"  Crypto:   {allocation['crypto']:.2%}")
        logger.info(f"  Cash:     {allocation['cash']:.2%}")

    if 13 in steps_to_run:
        run_with_error_handling(
            step_13_sleeve_allocation, 13, step_names[13], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 14: Turnover Management
    # ========================================================================

    def step_14_turnover_management():
        """Enforce turnover caps and manage trading costs"""
        logger.info("Running turnover management")

        # Apply turnover constraints
        state['turnover_adjusted_weights'], state['turnover_report'] = run_turnover_management(
            config,
            state['weights_df'],
            enforce_cap=True,
            apply_min_threshold=True
        )

        # Save turnover report (artifact #3)
        saver.save_turnover_report(state['turnover_report'])

        logger.info(f"✓ Turnover management complete:")
        logger.info(f"  Avg turnover: {state['turnover_report']['avg_turnover']:.2%}")
        logger.info(f"  Max turnover: {state['turnover_report']['max_turnover']:.2%}")
        logger.info(f"  Cap breaches: {state['turnover_report']['cap_breaches']}")
        logger.info(f"  Total costs:  ${state['turnover_report']['total_costs']:.2f}")

    if 14 in steps_to_run:
        run_with_error_handling(
            step_14_turnover_management, 14, step_names[14], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 15: Operations Monitoring & Reporting
    # ========================================================================

    def step_15_monitoring():
        """Generate ops report and monitor KPIs"""
        logger.info("Running operations monitoring")

        # Get latest weights for position change tracking
        weights_df = state.get('turnover_adjusted_weights', state['weights_df'])
        dates = sorted(weights_df['date'].unique())

        weights_current = weights_df[weights_df['date'] == dates[-1]]
        weights_prev = weights_df[weights_df['date'] == dates[-2]] if len(dates) > 1 else None

        # Run monitoring
        state['monitoring_result'] = run_monitoring(
            config,
            state['equity_curve'],
            weights_current,
            weights_prev=weights_prev,
            regime_info=state.get('regime_info'),
            allocation_info=state.get('allocation_result', {}).get('allocation'),
            turnover_report=state.get('turnover_report')
        )

        # Save ops report (artifact #4)
        html_report = state['monitoring_result']['html_report']
        saver.save_ops_report(html_report)

        logger.info(f"✓ Operations monitoring complete")

        # Check kill-switch status
        if state['monitoring_result']['kill_switch_triggered']:
            logger.error("⚠️  KILL-SWITCH TRIGGERED!")
            logger.error("Review the ops report for details")

    if 15 in steps_to_run:
        run_with_error_handling(
            step_15_monitoring, 15, step_names[15], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # FINISH
    # ========================================================================

    tracker.finish()

    # Validate enhancement artifacts
    logger.info("\nValidating enhancement artifacts...")
    artifact_status = saver.validate_enhancement_artifacts()

    all_artifacts_exist = all(artifact_status.values())

    if all_artifacts_exist:
        logger.info("\n✓ All 4 enhancement artifacts generated successfully!")
    else:
        logger.warning("\n⚠️  Some enhancement artifacts are missing. Check the logs above.")

    # Generate text summary for email
    if args.email_summary:
        summary_text = generate_summary_text(state, tracker)
        send_email_summary(
            config,
            subject=f"ML Trading System - Enhancements Report {datetime.now().strftime('%Y-%m-%d')}",
            body=summary_text,
            html_body=state.get('monitoring_result', {}).get('html_report')
        )

    # Print summary
    summary = tracker.get_summary()
    logger.info(f"\nPipeline Duration: {summary['duration_seconds']:.1f}s")

    if summary['errors']:
        logger.error(f"\nErrors encountered: {len(summary['errors'])}")
        for error in summary['errors']:
            logger.error(f"  Step {error['step']}: {error['error']}")
        sys.exit(1)


def generate_summary_text(state: dict, tracker: PipelineTracker) -> str:
    """
    Generate plain text summary for email

    Args:
        state: Pipeline state dict
        tracker: PipelineTracker instance

    Returns:
        Summary text string
    """
    lines = [
        "="*60,
        "ML TRADING SYSTEM - ENHANCEMENTS PIPELINE SUMMARY",
        "="*60,
        "",
        f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Duration: {tracker.get_summary()['duration_seconds']:.1f}s",
        ""
    ]

    # Regime info
    if state.get('regime_info'):
        regime = state['regime_info']
        lines.extend([
            "Market Regime:",
            f"  Current: {regime['regime_name']}",
            f"  Risk Multiplier: {regime['risk_multiplier']:.2f}",
            ""
        ])

    # Allocation info
    if state.get('allocation_result'):
        alloc = state['allocation_result']['allocation']
        lines.extend([
            "Sleeve Allocation:",
            f"  Equities: {alloc['equities']:.2%}",
            f"  Crypto:   {alloc['crypto']:.2%}",
            f"  Cash:     {alloc['cash']:.2%}",
            ""
        ])

    # Turnover info
    if state.get('turnover_report'):
        turnover = state['turnover_report']
        lines.extend([
            "Turnover Summary:",
            f"  Average: {turnover['avg_turnover']:.2%}",
            f"  Maximum: {turnover['max_turnover']:.2%}",
            f"  Cap Breaches: {turnover['cap_breaches']}",
            f"  Total Costs: ${turnover['total_costs']:.2f}",
            ""
        ])

    # Kill-switch status
    if state.get('monitoring_result'):
        if state['monitoring_result']['kill_switch_triggered']:
            lines.extend([
                "⚠️  KILL-SWITCH ALERT ⚠️",
                "See HTML report for details",
                ""
            ])

    lines.extend([
        "Artifacts Generated:",
        "  ✓ data/results/regime_state.csv",
        "  ✓ data/results/sleeve_allocation.json",
        "  ✓ data/results/turnover_report.json",
        "  ✓ data/results/ops_report_YYYYMMDD.html",
        "",
        "="*60
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    main()
