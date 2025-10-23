"""
Model training with MLflow tracking
"""
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from loguru import logger

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    from mlflow.models import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available, logging disabled")


class ModelTrainer:
    """Train and evaluate ML models"""

    def __init__(self, config: Dict):
        """
        Initialize trainer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('modeling', {})

        self.algorithm = self.model_config.get('algorithm', 'xgboost')
        self.model = None

        # MLflow setup
        if MLFLOW_AVAILABLE:
            self.mlflow_config = self.model_config.get('mlflow', {})
            tracking_uri = self.mlflow_config.get('tracking_uri', 'mlruns')
            experiment_name = self.mlflow_config.get('experiment_name', 'us_stock_ml')

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

        logger.info(f"Initialized ModelTrainer with algorithm={self.algorithm}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict] = None
    ) -> Any:
        """
        Train model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            params: Model parameters (optional, uses config defaults)

        Returns:
            Trained model
        """
        logger.info(f"Training {self.algorithm} model on {len(X_train)} samples")

        # Get model parameters
        if params is None:
            params = self.model_config.get(self.algorithm, {})

        # Train model
        if self.algorithm == 'xgboost':
            self.model = self._train_xgboost(X_train, y_train, X_val, y_val, params)
        elif self.algorithm == 'random_forest':
            self.model = self._train_random_forest(X_train, y_train, params)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        logger.info("Training complete")

        return self.model

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        params: Dict
    ) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        model = xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('learning_rate', 0.05),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            objective=params.get('objective', 'reg:squarederror'),
            random_state=42
        )

        # Early stopping with validation set
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        return model

    def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Dict
    ) -> RandomForestRegressor:
        """Train Random Forest model"""
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 20),
            min_samples_leaf=params.get('min_samples_leaf', 10),
            max_features=params.get('max_features', 'sqrt'),
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        return model

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test set

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        y_pred = self.model.predict(X_test)

        # Calculate MSE first, then derive RMSE
        mse = mean_squared_error(y_test, y_pred)

        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        # Additional metrics
        metrics['ic'] = np.corrcoef(y_test, y_pred)[0, 1]  # Information Coefficient
        metrics['rank_ic'] = pd.Series(y_test).corr(pd.Series(y_pred), method='spearman')

        logger.info(f"Evaluation metrics: RMSE={metrics['rmse']:.4f}, MSE={metrics['mse']:.6f}, IC={metrics['ic']:.4f}, Rank IC={metrics['rank_ic']:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances

        Returns:
            DataFrame with feature names and importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.model.feature_names_in_

            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })

            df = df.sort_values('importance', ascending=False)

            return df
        else:
            logger.warning("Model does not have feature_importances_")
            return pd.DataFrame()

    def save_model(self, path: str):
        """
        Save model to disk

        Args:
            path: Save path
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"Saved model to {path}")

    def load_model(self, path: str):
        """
        Load model from disk

        Args:
            path: Model path
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

        logger.info(f"Loaded model from {path}")

    def train_with_mlflow(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        run_name: Optional[str] = None
    ):
        """
        Train model with MLflow tracking

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            run_name: MLflow run name
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, training without tracking")
            return self.train(X_train, y_train, X_val, y_val)

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            params = self.model_config.get(self.algorithm, {})
            mlflow.log_params(params)

            # Train model
            self.train(X_train, y_train, X_val, y_val)

            # Evaluate on validation
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})

            # Log feature importance
            feature_importance = self.get_feature_importance()
            if not feature_importance.empty:
                mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")

            # Create model signature and input example
            predictions = self.model.predict(X_train[:100])  # Use subset for efficiency
            signature = infer_signature(X_train[:100], predictions)
            input_example = X_train.head(5)

            # Log model with signature and input example
            if self.algorithm == 'xgboost':
                mlflow.xgboost.log_model(
                    self.model,
                    "model",
                    signature=signature,
                    input_example=input_example
                )
            else:
                mlflow.sklearn.log_model(
                    self.model,
                    "model",
                    signature=signature,
                    input_example=input_example
                )

            logger.info(f"Logged training run to MLflow")

        return self.model
