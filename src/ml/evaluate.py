"""
Model evaluation with time-series cross-validation
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from loguru import logger

from src.ml.dataset import MLDataset, create_cv_folds
from src.ml.train import ModelTrainer


class CrossValidator:
    """Time-series cross-validation"""

    def __init__(self, config: Dict):
        """
        Initialize cross-validator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cv_config = config.get('modeling', {}).get('cv', {})

        self.n_splits = self.cv_config.get('n_splits', 5)
        self.test_size = self.cv_config.get('test_size', 0.2)
        self.embargo_days = self.cv_config.get('embargo_days', 5)
        self.purge_days = self.cv_config.get('purge_days', 2)

        logger.info(f"Initialized CrossValidator with n_splits={self.n_splits}")

    def run_cv(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'forward_return_5d'
    ) -> Dict:
        """
        Run time-series cross-validation

        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature columns
            label_col: Label column name

        Returns:
            Dictionary with CV results
        """
        logger.info("Running time-series cross-validation")

        # Create CV folds
        folds = create_cv_folds(
            df,
            n_splits=self.n_splits,
            test_size=self.test_size,
            embargo_days=self.embargo_days,
            purge_days=self.purge_days
        )

        # Run CV
        fold_results = []
        predictions = []

        for fold_idx, (train_df, test_df) in enumerate(folds):
            logger.info(f"Running fold {fold_idx + 1}/{len(folds)}")

            # Prepare datasets
            dataset = MLDataset(feature_cols=feature_cols, label_col=label_col)

            X_train, y_train = dataset.prepare(train_df, auto_select_features=False)
            X_test, y_test = dataset.prepare(test_df, auto_select_features=False)

            # Train model
            trainer = ModelTrainer(self.config)
            trainer.train(X_train, y_train)

            # Evaluate
            metrics = trainer.evaluate(X_test, y_test)
            fold_results.append(metrics)

            # Store predictions
            y_pred = trainer.predict(X_test)
            pred_df = test_df[['date', 'symbol']].copy()
            pred_df['y_true'] = y_test.values
            pred_df['y_pred'] = y_pred
            predictions.append(pred_df)

            logger.info(f"Fold {fold_idx + 1} metrics: {metrics}")

        # Aggregate results
        results = self._aggregate_results(fold_results)
        results['fold_results'] = fold_results
        results['predictions'] = pd.concat(predictions, ignore_index=True)

        logger.info(f"CV complete - Average IC: {results['mean_ic']:.4f} (+/- {results['std_ic']:.4f})")

        return results

    def _aggregate_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate results across folds"""
        metrics = {}

        for key in fold_results[0].keys():
            values = [fold[key] for fold in fold_results]
            metrics[f'mean_{key}'] = np.mean(values)
            metrics[f'std_{key}'] = np.std(values)

        return metrics


def evaluate_model_performance(
    predictions_df: pd.DataFrame,
    by_time: bool = True,
    by_sector: bool = False
) -> Dict:
    """
    Evaluate model performance with various analytics

    Args:
        predictions_df: DataFrame with [date, symbol, y_true, y_pred]
        by_time: Analyze performance over time
        by_sector: Analyze performance by sector (requires sector column)

    Returns:
        Dictionary with performance analytics
    """
    results = {}

    # Overall metrics
    ic = predictions_df[['y_true', 'y_pred']].corr().iloc[0, 1]
    rank_ic = predictions_df[['y_true', 'y_pred']].corr(method='spearman').iloc[0, 1]

    results['overall'] = {
        'ic': ic,
        'rank_ic': rank_ic,
        'n_predictions': len(predictions_df)
    }

    # By time
    if by_time:
        time_metrics = predictions_df.groupby('date').apply(
            lambda x: pd.Series({
                'ic': x[['y_true', 'y_pred']].corr().iloc[0, 1] if len(x) > 1 else np.nan,
                'rank_ic': x[['y_true', 'y_pred']].corr(method='spearman').iloc[0, 1] if len(x) > 1 else np.nan,
                'n': len(x)
            })
        )

        results['by_time'] = time_metrics

    # By sector
    if by_sector and 'sector' in predictions_df.columns:
        sector_metrics = predictions_df.groupby('sector').apply(
            lambda x: pd.Series({
                'ic': x[['y_true', 'y_pred']].corr().iloc[0, 1] if len(x) > 1 else np.nan,
                'rank_ic': x[['y_true', 'y_pred']].corr(method='spearman').iloc[0, 1] if len(x) > 1 else np.nan,
                'n': len(x)
            })
        )

        results['by_sector'] = sector_metrics

    logger.info(f"Performance analysis: IC={ic:.4f}, Rank IC={rank_ic:.4f}")

    return results
