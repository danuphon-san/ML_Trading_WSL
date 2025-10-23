"""
Run full ML trading pipeline
"""
import yaml
from loguru import logger
from pathlib import Path

# Configure logging
logger.add("logs/pipeline.log", rotation="10 MB")

def main():
    """Run complete pipeline"""
    logger.info("="*60)
    logger.info("Starting ML Trading Pipeline")
    logger.info("="*60)

    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded")

    # Step 1: Data Ingestion
    logger.info("Step 1: Data Ingestion")
    from src.io.universe import load_sp500_constituents
    from src.io.ingest_ohlcv import OHLCVIngester

    symbols = load_sp500_constituents()[:100]  # Start with top 100
    logger.info(f"Loaded {len(symbols)} symbols")

    ingester = OHLCVIngester()
    start_date = config['ingest']['start_date']

    logger.info(f"Fetching OHLCV data from {start_date}")
    data = ingester.fetch_ohlcv(symbols, start_date, None)
    ingester.save_parquet(data)

    logger.info("Data ingestion complete")

    # Step 2: Feature Engineering
    logger.info("Step 2: Feature Engineering")
    from src.features.ta_features import create_technical_features
    from src.labeling.labels import generate_forward_returns

    df = ingester.load_parquet(symbols)
    logger.info(f"Loaded {len(df)} rows of price data")

    df = create_technical_features(df, config)
    df = generate_forward_returns(df, config)

    from src.io.storage import save_dataframe
    save_dataframe(df, 'data/features/all_features.parquet')

    logger.info("Feature engineering complete")

    # Step 3: Model Training
    logger.info("Step 3: Model Training")
    from src.ml.dataset import MLDataset, create_time_based_split
    from src.ml.train import ModelTrainer

    dataset = MLDataset(label_col='forward_return_5d')
    train_df, test_df = create_time_based_split(df, test_size=0.2, embargo_days=5)

    X_train, y_train = dataset.prepare(train_df)
    X_test, y_test = dataset.prepare(test_df)

    logger.info(f"Training set: {len(X_train)} samples, {len(X_train.columns)} features")
    logger.info(f"Test set: {len(X_test)} samples")

    trainer = ModelTrainer(config)
    trainer.train_with_mlflow(X_train, y_train, X_test, y_test, run_name="pipeline_run")

    metrics = trainer.evaluate(X_test, y_test)
    logger.info(f"Model evaluation: IC={metrics['ic']:.4f}, Rank IC={metrics['rank_ic']:.4f}")

    trainer.save_model('data/models/latest_model.pkl')
    logger.info("Model training complete")

    # Step 4: Backtesting
    logger.info("Step 4: Backtesting")
    from src.backtest.bt_engine import VectorizedBacktester
    from src.portfolio.construct import construct_portfolio

    # Generate scores
    X_test, _ = dataset.prepare(test_df)
    scores = trainer.predict(X_test)

    scored_df = test_df[['date', 'symbol']].copy()
    scored_df['ml_score'] = scores

    # Get price panel
    price_panel = df[['date', 'symbol', 'close']].copy()

    # Construct portfolio weights over time
    weights_history = []
    for date in scored_df['date'].unique()[:50]:  # Limit to 50 dates for demo
        day_scores = scored_df[scored_df['date'] == date]
        weights = construct_portfolio(day_scores, price_panel, config)

        for symbol, weight in weights.items():
            weights_history.append({
                'date': date,
                'symbol': symbol,
                'weight': weight
            })

    import pandas as pd
    weights_df = pd.DataFrame(weights_history)

    # Run backtest
    backtester = VectorizedBacktester(config)
    results = backtester.run(weights_df, price_panel)

    logger.info("Backtest Results:")
    logger.info(f"  Total Return: {results['metrics']['total_return']:.2%}")
    logger.info(f"  Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    logger.info(f"  Avg Turnover: {results['metrics'].get('avg_turnover', 0):.2%}")

    # Save results
    results['equity_curve'].to_csv('data/reports/equity_curve.csv', index=False)
    logger.info("Backtest complete, results saved to data/reports/")

    logger.info("="*60)
    logger.info("Pipeline Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
