#!/usr/bin/env python3
"""
Core ML Trading Pipeline Orchestrator (Steps 1-11)

Implements instruction1.md:
- Data ingestion (OHLCV + fundamentals)
- Feature engineering (technical + PIT fundamentals)
- Label generation
- Model training with MLflow
- Signal generation
- Portfolio construction
- Backtesting
- Evaluation

Produces 5 required artifacts:
1. data/features/all_features_with_fundamentals.parquet
2. data/results/scored_df.parquet
3. data/results/weights_df.parquet
4. data/results/backtest_results.json
5. models/latest/model.pkl
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.cli_parser import create_core_pipeline_parser, parse_step_ranges, validate_step_ranges, print_execution_plan
from utils.pipeline_utils import PipelineTracker, run_with_error_handling, validate_artifacts, ensure_directories, get_step_name
from utils.config_loader import load_config_with_validation

from src.io.universe import load_sp500_constituents
from src.io.ingest_ohlcv import OHLCVIngester, build_provider_kwargs
from src.io.ingest_fundamentals import FundamentalsIngester
from src.io.results_saver import ResultsSaver
from src.io.storage import save_dataframe

from src.features.ta_features import create_technical_features
from src.features.fa_features import FundamentalFeatures

from src.labeling.labels import generate_forward_returns

from src.ml.dataset import MLDataset, create_time_based_split
from src.ml.train import ModelTrainer

from src.portfolio.construct import construct_portfolio
from src.backtest.bt_engine import VectorizedBacktester


def main():
    """Run core ML trading pipeline"""

    # Parse arguments
    parser = create_core_pipeline_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add("logs/core_pipeline.log", rotation="10 MB", level="DEBUG")

    # Load config (includes .env interpolation)
    config = load_config_with_validation(args.config)

    # Override config with CLI args
    if args.start_date:
        config['ingest']['start_date'] = args.start_date

    benchmark_symbol = config.get('reporting', {}).get('benchmark')

    # Fundamental provider settings (shared across steps)
    fund_config = config.get('fundamentals', {})
    fund_provider = fund_config.get('provider', 'yfinance').lower()
    fund_api_key = None
    fund_kwargs = {}

    if fund_provider == 'alpha_vantage':
        alpha_cfg = fund_config.get('alpha_vantage', {})
        fund_api_key = alpha_cfg.get('api_key')
    elif fund_provider == 'simfin':
        simfin_cfg = fund_config.get('simfin', {})
        fund_api_key = simfin_cfg.get('api_key')
        fund_kwargs.update({
            'simfin_data_dir': simfin_cfg.get('data_dir'),
            'simfin_market': simfin_cfg.get('market', 'us'),
            'simfin_variant': simfin_cfg.get('variant', 'quarterly')
        })

    # Parse steps to execute
    steps_to_run = parse_step_ranges(args.steps, args.skip)
    validate_step_ranges(steps_to_run, valid_range=(1, 11))

    # Step names for display
    step_names = {i: get_step_name(i) for i in range(1, 12)}

    if args.dry_run:
        print_execution_plan(steps_to_run, step_names)
        return

    # Validate artifacts only
    if args.validate_only:
        saver = ResultsSaver()
        saver.validate_core_artifacts()
        return

    # Initialize tracking
    tracker = PipelineTracker()
    tracker.start()

    # Ensure directories exist
    ensure_directories([
        'data/parquet/1d',
        'data/fundamentals',
        'data/features',
        'data/results',
        'data/models/latest',
        'data/reports',
        'logs'
    ])

    # Initialize result saver
    saver = ResultsSaver()

    # Shared state across steps
    state = {
        'symbols': None,
        'ohlcv_data': None,
        'fundamentals_data': None,
        'df': None,
        'df_with_features': None,
        'train_df': None,
        'test_df': None,
        'trainer': None,
        'scored_df': None,
        'weights_df': None,
        'backtest_results': None
    }

    # ========================================================================
    # STEP 1: Data Ingestion (OHLCV + Fundamentals)
    # ========================================================================

    def step_1_data_ingestion():
        """Fetch OHLCV and fundamental data"""
        logger.info(f"Loading universe: top {args.symbols} symbols")

        # Load symbols
        state['symbols'] = load_sp500_constituents()[:args.symbols]
        logger.info(f"Selected {len(state['symbols'])} symbols")

        ingest_cfg = config.get('ingest', {})
        price_provider = ingest_cfg.get('provider', 'yfinance')
        frequencies = ingest_cfg.get('frequencies', ['1d'])
        frequency = frequencies[0] if frequencies else '1d'
        start_date = ingest_cfg.get('start_date')
        end_date = ingest_cfg.get('end_date')

        if not start_date:
            start_date = "2015-01-01"

        try:
            provider_kwargs = build_provider_kwargs(price_provider, config)
        except ValueError as exc:
            logger.error(f"OHLCV provider misconfigured: {exc}")
            raise

        storage_path = config.get('data', {}).get('parquet', "data/parquet")
        ingester = OHLCVIngester(
            provider=price_provider,
            storage_path=storage_path,
            **provider_kwargs
        )

        logger.info(f"Fetching OHLCV data from {start_date} via {price_provider}")
        state['ohlcv_data'] = ingester.fetch_ohlcv(
            state['symbols'],
            start_date,
            end_date,
            frequency=frequency
        )
        ingester.save_parquet(state['ohlcv_data'], frequency=frequency)

        if benchmark_symbol and benchmark_symbol not in state['symbols']:
            logger.info(f"Fetching benchmark OHLCV data for {benchmark_symbol}")
            benchmark_data = ingester.fetch_ohlcv(
                [benchmark_symbol],
                start_date,
                end_date,
                frequency=frequency
            )

            if benchmark_data:
                ingester.save_parquet(benchmark_data, frequency=frequency)
            else:
                logger.warning(
                    f"Benchmark {benchmark_symbol} returned no data from provider. "
                    "Verify the ticker symbol for your data source."
                )

        # Fetch fundamentals
        fund_ingester = FundamentalsIngester(
            storage_path="data/fundamentals",
            provider=fund_provider,
            api_key=fund_api_key,
            **fund_kwargs
        )
        logger.info(f"Fetching fundamental data using {fund_provider}")
        state['fundamentals_data'] = fund_ingester.fetch_fundamentals(state['symbols'])
        fund_ingester.save_parquet(state['fundamentals_data'])

        logger.info(f"✓ Fetched data for {len(state['ohlcv_data'])} symbols")

    if 1 in steps_to_run:
        run_with_error_handling(
            step_1_data_ingestion, 1, step_names[1], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 2: Data Preprocessing & Alignment
    # ========================================================================

    def step_2_preprocessing():
        """Load and align data"""
        ingester = OHLCVIngester()

        if state['symbols'] is None:
            state['symbols'] = load_sp500_constituents()[:args.symbols]

        logger.info(f"Loading OHLCV data for {len(state['symbols'])} symbols")
        state['df'] = ingester.load_parquet(state['symbols'])

        logger.info(f"Loaded {len(state['df']):,} rows of OHLCV data")
        logger.info(f"Date range: {state['df']['date'].min()} to {state['df']['date'].max()}")

    if 2 in steps_to_run:
        run_with_error_handling(
            step_2_preprocessing, 2, step_names[2], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 3: Technical Feature Engineering
    # ========================================================================

    def step_3_technical_features():
        """Generate technical indicators"""
        logger.info("Generating technical features")
        state['df'] = create_technical_features(state['df'], config)

        feature_cols = [c for c in state['df'].columns if c not in
                       ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]

        logger.info(f"✓ Generated {len(feature_cols)} technical features")

    if 3 in steps_to_run:
        run_with_error_handling(
            step_3_technical_features, 3, step_names[3], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 4: Fundamental Feature Engineering (PIT Alignment)
    # ========================================================================

    def step_4_fundamental_features():
        """Generate fundamental features with PIT alignment"""
        fund_ingester = FundamentalsIngester(
            storage_path="data/fundamentals",
            provider=fund_provider,
            api_key=fund_api_key,
            **fund_kwargs
        )

        if state['symbols'] is None:
            state['symbols'] = load_sp500_constituents()[:args.symbols]

        logger.info("Loading fundamental data")
        fundamentals_df = fund_ingester.load_parquet(state['symbols'])

        if fundamentals_df.empty:
            logger.warning("No fundamental data found, skipping")
            return

        logger.info(f"Loaded {len(fundamentals_df)} fundamental records")

        # Compute fundamental features with PIT alignment
        fa_features = FundamentalFeatures(config)
        state['df'] = fa_features.compute_features(state['df'], fundamentals_df)

        fund_cols = [c for c in state['df'].columns if
                    any(x in c for x in ['ratio', 'roe', 'roa', 'debt', 'margin', 'quality'])]

        logger.info(f"✓ Generated {len(fund_cols)} fundamental features with PIT alignment")

    if 4 in steps_to_run:
        run_with_error_handling(
            step_4_fundamental_features, 4, step_names[4], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 5: Label Generation
    # ========================================================================

    def step_5_label_generation():
        """Generate forward return labels"""
        logger.info("Generating forward return labels")
        state['df'] = generate_forward_returns(state['df'], config)

        label_col = f"forward_return_{config['labels']['horizon']}d"
        logger.info(f"✓ Generated labels: {label_col}")
        logger.info(f"Label coverage: {state['df'][label_col].notna().sum():,} rows")

    if 5 in steps_to_run:
        run_with_error_handling(
            step_5_label_generation, 5, step_names[5], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 6: Feature Selection & Dataset Preparation
    # ========================================================================

    def step_6_feature_selection():
        """Prepare ML dataset with feature selection"""
        # Save features with fundamentals (artifact #1)
        saver.save_features_with_fundamentals(state['df'])

        # Prepare dataset
        label_col = f"forward_return_{config['labels']['horizon']}d"
        dataset = MLDataset(label_col=label_col)

        # Time-based split with embargo
        embargo_days = config['modeling'].get('embargo_days', 5)
        logger.info(f"Creating time-based train/test split with {embargo_days}-day embargo")

        state['train_df'], state['test_df'] = create_time_based_split(
            state['df'],
            test_size=0.2,
            embargo_days=embargo_days
        )

        logger.info(f"Train: {len(state['train_df']):,} rows, Test: {len(state['test_df']):,} rows")

    if 6 in steps_to_run:
        run_with_error_handling(
            step_6_feature_selection, 6, step_names[6], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 7: Model Training
    # ========================================================================

    def step_7_model_training():
        """Train ML model with MLflow tracking"""
        label_col = f"forward_return_{config['labels']['horizon']}d"
        dataset = MLDataset(label_col=label_col)

        # Prepare features
        X_train, y_train = dataset.prepare(state['train_df'], auto_select_features=True)
        X_test, y_test = dataset.prepare(state['test_df'], auto_select_features=False)

        logger.info(f"Training set: {len(X_train):,} samples, {len(X_train.columns)} features")
        logger.info(f"Test set: {len(X_test):,} samples")

        # Train model
        state['trainer'] = ModelTrainer(config)

        algorithm = config['modeling']['algorithm']
        logger.info(f"Training {algorithm} model...")

        tuning_result = None
        if args.optuna_trials > 0 or args.optuna_timeout:
            logger.info(
                f"Running Optuna tuning: trials={args.optuna_trials}, timeout={args.optuna_timeout}, metric={args.optuna_metric}"
            )
            tuning_result = state['trainer'].tune_with_optuna(
                X_train,
                y_train,
                X_test,
                y_test,
                n_trials=args.optuna_trials,
                timeout=args.optuna_timeout,
                metric=args.optuna_metric,
                study_name=args.optuna_study_name,
                storage=args.optuna_storage,
                run_name="core_pipeline_optuna"
            )

            if tuning_result.get('best_params'):
                logger.info(f"Optuna best params: {tuning_result['best_params']}")
        else:
            state['trainer'].train_with_mlflow(
                X_train,
                y_train,
                X_test,
                y_test,
                run_name="core_pipeline_run"
            )

        # Evaluate (always on hold-out test set)
        metrics = state['trainer'].evaluate(X_test, y_test)
        logger.info(f"Model metrics: IC={metrics['ic']:.4f}, Rank IC={metrics['rank_ic']:.4f}")

        # Save model (artifact #5)
        saver.save_model(state['trainer'].model)

        # Store feature columns for step 8
        state['feature_cols'] = dataset.feature_cols

    if 7 in steps_to_run:
        run_with_error_handling(
            step_7_model_training, 7, step_names[7], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 8: Signal Generation (Scoring)
    # ========================================================================

    def step_8_signal_generation():
        """Generate ML scores for test period"""
        label_col = f"forward_return_{config['labels']['horizon']}d"
        dataset = MLDataset(feature_cols=state['feature_cols'], label_col=label_col)

        # Score test set
        test_df_clean = state['test_df'][state['test_df'][label_col].notna()].copy()
        X_test, _ = dataset.prepare(test_df_clean, auto_select_features=False)

        logger.info(f"Generating scores for {len(X_test):,} observations")
        scores = state['trainer'].predict(X_test)

        # Create scored DataFrame
        state['scored_df'] = test_df_clean[['date', 'symbol']].copy()
        state['scored_df']['ml_score'] = scores

        # Save scored data (artifact #2)
        saver.save_scored_df(state['scored_df'])

        logger.info(f"✓ Generated scores: mean={scores.mean():.4f}, std={scores.std():.4f}")

    if 8 in steps_to_run:
        run_with_error_handling(
            step_8_signal_generation, 8, step_names[8], tracker,
            continue_on_error=args.continue_on_error
        )

    def ensure_price_panel():
        """Ensure downstream steps have price data even when rerunning partial pipelines."""
        ingester = None

        if state['df'] is None:
            logger.info("Reloading OHLCV data for price-dependent steps")

            if state['symbols'] is None:
                state['symbols'] = load_sp500_constituents()[:args.symbols]

            ingester = OHLCVIngester()
            state['df'] = ingester.load_parquet(state['symbols'])

            if state['df'] is None or state['df'].empty:
                raise ValueError("No OHLCV data available for portfolio construction/backtesting")

        need_open_prices = config.get('backtest', {}).get('execution_timing', 'close') == 'next_open'

        price_cols = ['date', 'symbol', 'close']
        if need_open_prices:
            if 'open' in state['df'].columns:
                price_cols.append('open')
            else:
                logger.warning(
                    "Price data missing 'open' column while execution_timing is 'next_open'. "
                    "Attempting to proceed, but backtest will fail without opens."
                )

        price_panel = state['df'][price_cols].copy()

        if benchmark_symbol:
            if price_panel[price_panel['symbol'] == benchmark_symbol].empty:
                if ingester is None:
                    ingester = OHLCVIngester()

                logger.info(f"Benchmark {benchmark_symbol} missing from price data; attempting to load from storage")
                benchmark_prices = ingester.load_parquet([benchmark_symbol])

                if benchmark_prices.empty:
                    logger.warning(
                        f"Benchmark {benchmark_symbol} price data not found in parquet storage; "
                        "regime detection will be skipped"
                    )
                else:
                    benchmark_cols = ['date', 'symbol', 'close']
                    if need_open_prices and 'open' in benchmark_prices.columns:
                        benchmark_cols.append('open')
                    benchmark_prices = benchmark_prices[benchmark_cols]
                    price_panel = pd.concat([price_panel, benchmark_prices], ignore_index=True)

        required_cols = {'date', 'symbol', 'close'}
        if need_open_prices:
            required_cols.add('open')
        missing_cols = required_cols.difference(price_panel.columns)

        if missing_cols:
            missing_str = ", ".join(sorted(missing_cols))
            raise KeyError(f"Price DataFrame missing required columns: {missing_str}")

        return price_panel

    def ensure_scored_df():
        """Ensure scored dataframe exists (load artifact when skipping prior steps)."""
        if state['scored_df'] is None:
            logger.info("Loading scored data artifact for portfolio construction")
            try:
                state['scored_df'] = saver.load_scored_df()
            except FileNotFoundError as exc:
                raise ValueError(
                    "Scored dataset not found. Run step 8 to generate ml_score outputs before step 9."
                ) from exc

        if state['scored_df'] is None or state['scored_df'].empty:
            raise ValueError("Scored dataset is empty; portfolio construction requires ML scores.")

        required_cols = {'date', 'symbol', 'ml_score'}
        missing_cols = required_cols.difference(state['scored_df'].columns)
        if missing_cols:
            missing_str = ", ".join(sorted(missing_cols))
            raise KeyError(f"Scored dataset missing required columns: {missing_str}")

        return state['scored_df']

    # ========================================================================
    # STEP 9: Portfolio Construction
    # ========================================================================

    def step_9_portfolio_construction():
        """Construct portfolio weights using optimizer"""
        price_panel = ensure_price_panel()
        scored_df = ensure_scored_df()

        logger.info(f"Constructing portfolio with optimizer: {config['portfolio']['optimizer']}")
        logger.info(f"Top K positions: {config['portfolio']['top_k']}")

        unique_dates = sorted(scored_df['date'].unique())

        # Limit dates for demo (can remove for production)
        MAX_DATES = 200
        rebalance_dates = unique_dates[:min(MAX_DATES, len(unique_dates))]

        weights_history = []

        for i, date in enumerate(rebalance_dates):
            if i % 20 == 0:
                logger.info(f"Processing rebalance date {i+1}/{len(rebalance_dates)}")

            day_scores = scored_df[scored_df['date'] == date]
            weights = construct_portfolio(day_scores, price_panel, config)

            for symbol, weight in weights.items():
                weights_history.append({
                    'date': date,
                    'symbol': symbol,
                    'weight': weight
                })

        state['weights_df'] = pd.DataFrame(weights_history)

        # Save weights (artifact #3)
        saver.save_weights_df(state['weights_df'])

        logger.info(f"✓ Portfolio weights constructed: {len(state['weights_df']):,} records")

    if 9 in steps_to_run:
        run_with_error_handling(
            step_9_portfolio_construction, 9, step_names[9], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 10: Backtesting
    # ========================================================================

    def step_10_backtesting():
        """Run backtest with realistic costs"""
        price_panel = ensure_price_panel()

        logger.info("Running backtest...")
        logger.info(f"Costs: {config['portfolio']['costs_bps']} bps commission, "
                   f"{config['portfolio']['slippage_bps']} bps slippage")

        backtester = VectorizedBacktester(config)
        state['backtest_results'] = backtester.run(state['weights_df'], price_panel)

        # Save backtest results (artifact #4)
        saver.save_backtest_results(state['backtest_results'])

        # Log metrics
        metrics = state['backtest_results']['metrics']
        logger.info(f"✓ Backtest complete:")
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Avg Turnover: {metrics.get('avg_turnover', 0):.2%}")

    if 10 in steps_to_run:
        run_with_error_handling(
            step_10_backtesting, 10, step_names[10], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # STEP 11: Model Evaluation & Reporting
    # ========================================================================

    def step_11_evaluation():
        """Generate evaluation report"""
        logger.info("Generating evaluation report")

        # Save equity curve
        equity_curve = state['backtest_results']['equity_curve']
        equity_curve.to_csv('data/reports/equity_curve.csv', index=False)

        # Summary report
        metrics = state['backtest_results']['metrics']

        report_lines = [
            "",
            "="*60,
            "CORE PIPELINE COMPLETE - EVALUATION SUMMARY",
            "="*60,
            "",
            "Performance Metrics:",
            f"  Total Return:      {metrics['total_return']:.2%}",
            f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}",
            f"  Max Drawdown:      {metrics['max_drawdown']:.2%}",
            f"  Volatility:        {metrics.get('volatility', 0):.2%}",
            f"  Avg Turnover:      {metrics.get('avg_turnover', 0):.2%}",
            "",
            "Artifacts Generated:",
            "  ✓ data/features/all_features_with_fundamentals.parquet",
            "  ✓ data/results/scored_df.parquet",
            "  ✓ data/results/weights_df.parquet",
            "  ✓ data/results/backtest_results.json",
            "  ✓ models/latest/model.pkl",
            "",
            "="*60
        ]

        for line in report_lines:
            logger.info(line)

    if 11 in steps_to_run:
        run_with_error_handling(
            step_11_evaluation, 11, step_names[11], tracker,
            continue_on_error=args.continue_on_error
        )

    # ========================================================================
    # FINISH
    # ========================================================================

    tracker.finish()

    # Validate all artifacts
    logger.info("\nValidating output artifacts...")
    artifact_status = saver.validate_core_artifacts()

    all_artifacts_exist = all(artifact_status.values())

    if all_artifacts_exist:
        logger.info("\n✓ All 5 required artifacts generated successfully!")
    else:
        logger.warning("\n⚠️  Some artifacts are missing. Check the logs above.")

    # Print summary
    summary = tracker.get_summary()
    logger.info(f"\nPipeline Duration: {summary['duration_seconds']:.1f}s")

    if summary['errors']:
        logger.error(f"\nErrors encountered: {len(summary['errors'])}")
        for error in summary['errors']:
            logger.error(f"  Step {error['step']}: {error['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
