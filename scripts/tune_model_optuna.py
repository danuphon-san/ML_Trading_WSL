#!/usr/bin/env python3
"""Standalone Optuna tuning entrypoint for ML models."""
import sys
from pathlib import Path
import argparse
from loguru import logger

# Add project root to path for module imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config_with_validation
from src.io.results_saver import ResultsSaver
from src.ml.dataset import MLDataset, create_time_based_split
from src.ml.train import ModelTrainer, OPTUNA_AVAILABLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for ML trading models"
    )

    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--metric', type=str, default='rank_ic', help='Metric to optimise')
    parser.add_argument('--study-name', type=str, default=None, help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URI')
    parser.add_argument('--algorithm', type=str, default=None, help='Override algorithm from config')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    return parser.parse_args()


def main():
    args = parse_args()

    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    if not OPTUNA_AVAILABLE:
        logger.error("Optuna is not installed. Install optuna to use this tuner.")
        sys.exit(1)

    config = load_config_with_validation(args.config)

    if args.algorithm:
        config['modeling']['algorithm'] = args.algorithm

    saver = ResultsSaver()

    try:
        df = saver.load_features_with_fundamentals()
    except FileNotFoundError:
        logger.error("Feature artifact not found. Run core pipeline steps 1-6 first to build features.")
        sys.exit(1)

    label_col = f"forward_return_{config['labels']['horizon']}d"
    dataset = MLDataset(label_col=label_col)

    train_df, test_df = create_time_based_split(
        df,
        test_size=config['modeling']['cv'].get('test_size', 0.2),
        embargo_days=config['modeling'].get('embargo_days', 5)
    )

    X_train, y_train = dataset.prepare(train_df, auto_select_features=True)
    X_test, y_test = dataset.prepare(test_df, auto_select_features=False)

    trainer = ModelTrainer(config)

    logger.info(
        f"Starting Optuna tuning for {config['modeling']['algorithm']} with {args.trials} trials"
    )

    tuning_result = trainer.tune_with_optuna(
        X_train,
        y_train,
        X_test,
        y_test,
        n_trials=args.trials,
        timeout=args.timeout,
        metric=args.metric,
        study_name=args.study_name,
        storage=args.storage,
        run_name="optuna_tuning_run"
    )

    if tuning_result.get('validation_metrics'):
        vm = tuning_result['validation_metrics']
        logger.info(
            "Validation metrics | RMSE={rmse:.4f} IC={ic:.4f} RankIC={rank_ic:.4f}".format(**vm)
        )

    final_metrics = trainer.evaluate(X_test, y_test)
    logger.info(
        "Hold-out metrics after tuning | RMSE={rmse:.4f} IC={ic:.4f} RankIC={rank_ic:.4f}".format(**final_metrics)
    )

    saver.save_model(trainer.model)
    logger.info("Saved tuned model to models/latest/model.pkl")


if __name__ == '__main__':
    main()
