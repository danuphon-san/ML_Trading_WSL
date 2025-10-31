"""
Model training with MLflow tracking
"""
import io
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from loguru import logger

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

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
        self.best_params: Dict[str, Any] = {}

        # Validate algorithm dependencies early
        if self.algorithm == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM selected but the package is not installed. Install lightgbm or choose another algorithm.")

        if self.algorithm == 'catboost' and not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost selected but the package is not installed. Install catboost or choose another algorithm.")

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
        else:
            params = params.copy()

        self.best_params = params.copy()

        # Train model
        self.model = self._fit_algorithm(X_train, y_train, X_val, y_val, params)

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
            min_child_weight=params.get('min_child_weight', 1.0),
            gamma=params.get('gamma', 0.0),
            reg_lambda=params.get('reg_lambda', 1.0),
            reg_alpha=params.get('reg_alpha', 0.0),
            objective=params.get('objective', 'reg:squarederror'),
            random_state=42
        )

        early_stopping_rounds = params.get('early_stopping_rounds')
        callbacks = []

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

            if early_stopping_rounds:
                try:
                    callbacks.append(
                        xgb.callback.EarlyStopping(
                            rounds=int(early_stopping_rounds),
                            save_best=True
                        )
                    )
                except AttributeError:
                    logger.warning("xgboost callback API unavailable; skipping early stopping")

        fit_kwargs = {
            'X': X_train,
            'y': y_train,
            'eval_set': eval_set,
            'verbose': False
        }

        eval_metric = params.get('eval_metric')
        if eval_metric:
            fit_kwargs['eval_metric'] = eval_metric

        if callbacks:
            try:
                model.fit(**fit_kwargs, callbacks=callbacks)
                return model
            except TypeError:
                logger.warning("xgboost callbacks unsupported; retrying without callbacks")

        allow_early_stop = bool(early_stopping_rounds and eval_set is not None)

        if allow_early_stop:
            fit_kwargs['early_stopping_rounds'] = int(early_stopping_rounds)
            try:
                model.fit(**fit_kwargs)
                return model
            except TypeError:
                logger.warning("xgboost early_stopping_rounds unsupported; training without early stopping")
                fit_kwargs.pop('early_stopping_rounds', None)

        model.fit(**fit_kwargs)

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

    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics shared across training workflows."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        y_true_series = pd.Series(y_true)
        y_pred_series = pd.Series(y_pred)

        ic = np.corrcoef(y_true_series, y_pred_series)[0, 1]
        rank_ic = y_true_series.corr(y_pred_series, method='spearman')

        if not np.isfinite(ic):
            ic = 0.0
        if not np.isfinite(rank_ic):
            rank_ic = 0.0

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'ic': ic,
            'rank_ic': rank_ic
        }

    def _fit_algorithm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        params: Dict
    ):
        if self.algorithm == 'xgboost':
            return self._train_xgboost(X_train, y_train, X_val, y_val, params)
        if self.algorithm == 'random_forest':
            return self._train_random_forest(X_train, y_train, params)
        if self.algorithm == 'lightgbm':
            return self._train_lightgbm(X_train, y_train, X_val, y_val, params)
        if self.algorithm == 'catboost':
            return self._train_catboost(X_train, y_train, X_val, y_val, params)
        raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        params: Dict
    ):
        """Train LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed")

        model = lgb.LGBMRegressor(
            n_estimators=params.get('n_estimators', 400),
            learning_rate=params.get('learning_rate', 0.05),
            max_depth=params.get('max_depth', -1),
            num_leaves=params.get('num_leaves', 31),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            min_child_samples=params.get('min_child_samples', 20),
            reg_lambda=params.get('reg_lambda', 0.0),
            reg_alpha=params.get('reg_alpha', 0.0),
            objective=params.get('objective', 'regression'),
            random_state=42,
            n_jobs=-1
        )

        callbacks = []
        if X_val is not None and y_val is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=params.get('early_stopping_rounds', 30),
                    verbose=False
                )
            )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)] if X_val is not None and y_val is not None else None,
            eval_metric=params.get('eval_metric', 'rmse'),
            callbacks=callbacks
        )

        return model

    def _train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        params: Dict
    ):
        """Train CatBoost model"""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed")

        model = CatBoostRegressor(
            iterations=params.get('iterations', 500),
            depth=params.get('depth', 6),
            learning_rate=params.get('learning_rate', 0.05),
            loss_function=params.get('loss_function', 'RMSE'),
            l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
            subsample=params.get('subsample', 0.8),
            random_state=42,
            verbose=False
        )

        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=False
            )
        else:
            model.fit(X_train, y_train, verbose=False)

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

        metrics = self._compute_metrics(y_test, y_pred)

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
            feature_names = getattr(self.model, 'feature_names_in_', None)
            if feature_names is None and hasattr(self.model, 'feature_names_'):
                feature_names = self.model.feature_names_
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

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

    def tune_with_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        *,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        metric: str = 'rank_ic',
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Hyperparameter tuning using Optuna."""

        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install optuna to enable tuning.")

        if n_trials <= 0 and timeout is None:
            raise ValueError("Optuna tuning requires a positive number of trials or a timeout.")

        maximize_metrics = {'rank_ic', 'ic', 'r2'}
        minimize_metrics = {'rmse', 'mse', 'mae'}
        valid_metrics = maximize_metrics | minimize_metrics

        if metric not in valid_metrics:
            raise ValueError(f"Unsupported Optuna metric '{metric}'. Choose from {sorted(valid_metrics)}")

        direction = 'maximize' if metric in maximize_metrics else 'minimize'

        inner_X_train = X_train.copy()
        inner_y_train = y_train.copy()
        inner_X_val = X_val
        inner_y_val = y_val

        if inner_X_val is None or inner_y_val is None:
            logger.warning("Validation set not provided; using last 20% of training data as validation for tuning")
            inner_X_train, inner_X_val, inner_y_train, inner_y_val = train_test_split(
                inner_X_train,
                inner_y_train,
                test_size=0.2,
                shuffle=False
            )

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=bool(storage)
        )

        logger.info(
            f"Starting Optuna study (direction={direction}, trials={n_trials}, timeout={timeout})"
        )

        def objective(trial: optuna.trial.Trial) -> float:
            params = self._suggest_params(trial)
            try:
                model = self._fit_algorithm(inner_X_train, inner_y_train, inner_X_val, inner_y_val, params)
                predictions = model.predict(inner_X_val)
                metrics = self._compute_metrics(inner_y_val, predictions)
                score = metrics[metric]

                if not np.isfinite(score) or score == 0.0:
                    raise optuna.exceptions.TrialPruned(f"Metric {metric} is invalid ({score})")

                trial.set_user_attr('metrics', metrics)
                return float(score)

            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(f"Optuna trial failed: {exc}")
                raise optuna.exceptions.TrialPruned(str(exc))

        study.optimize(objective, n_trials=n_trials if n_trials > 0 else None, timeout=timeout)

        if not study.trials:
            raise RuntimeError("Optuna study produced no trials; cannot select best parameters")

        best_params = study.best_trial.params
        best_value = study.best_value
        self.best_params = best_params.copy()

        logger.info(f"Optuna best {metric}: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        self.model = self._fit_algorithm(inner_X_train, inner_y_train, inner_X_val, inner_y_val, best_params)

        validation_metrics = {}
        if inner_X_val is not None and inner_y_val is not None and len(inner_X_val) > 0:
            validation_metrics = self.evaluate(inner_X_val, inner_y_val)

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=run_name or f"{self.algorithm}_optuna_tuning"):
                mlflow.log_param('optuna_metric', metric)
                mlflow.log_params(best_params)
                mlflow.log_metric(f"best_{metric}", float(best_value))
                mlflow.log_metrics({f"val_{k}": v for k, v in validation_metrics.items()})

                trials_df = study.trials_dataframe()
                if not trials_df.empty:
                    buffer = io.StringIO()
                    trials_df.to_csv(buffer, index=False)
                    mlflow.log_text(buffer.getvalue(), f"optuna_trials_{self.algorithm}.csv")

                # Log feature importance when available
                feature_importance = self.get_feature_importance()
                if not feature_importance.empty:
                    mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")

                sample_X = inner_X_train.iloc[: min(100, len(inner_X_train))]
                predictions = self.model.predict(sample_X)
                signature = infer_signature(sample_X, predictions)
                input_example = sample_X.head(5)

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

                logger.info("Logged Optuna tuning run to MLflow")

        return {
            'study': study,
            'best_params': best_params,
            'best_value': best_value,
            'metric': metric,
            'direction': direction,
            'validation_metrics': validation_metrics
        }

    def _suggest_params(self, trial: 'optuna.trial.Trial') -> Dict[str, Any]:
        """Suggest algorithm-specific hyperparameters for Optuna."""

        if self.algorithm == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
                'objective': 'reg:squarederror',
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 80)
            }

        if self.algorithm == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                'max_depth': trial.suggest_int('max_depth', 4, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 40),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.8])
            }

        if self.algorithm == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'max_depth': trial.suggest_int('max_depth', -1, 16),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
                'objective': 'regression',
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 80)
            }

        if self.algorithm == 'catboost':
            return {
                'iterations': trial.suggest_int('iterations', 200, 800),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'loss_function': 'RMSE'
            }

        raise ValueError(f"Unknown algorithm: {self.algorithm}")
