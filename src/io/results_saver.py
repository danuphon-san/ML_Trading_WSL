"""
Results artifact management for pipeline outputs

Handles saving and loading artifacts in instruction1.md and instruction2.md formats
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from loguru import logger


class ResultsSaver:
    """Manage pipeline output artifacts"""

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.features_dir = self.data_root / "features"
        self.results_dir = self.data_root / "results"
        self.models_dir = self.data_root / "models"

        # Create directories
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # CORE PIPELINE ARTIFACTS (instruction1.md outputs)
    # ========================================================================

    def save_features_with_fundamentals(self, df: pd.DataFrame) -> Path:
        """
        Save: data/features/all_features_with_fundamentals.parquet

        Args:
            df: DataFrame with technical + fundamental features + labels

        Returns:
            Path to saved file
        """
        path = self.features_dir / "all_features_with_fundamentals.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"✓ Saved features: {path} ({len(df):,} rows)")
        return path

    def load_features_with_fundamentals(self) -> pd.DataFrame:
        """Load features artifact"""
        path = self.features_dir / "all_features_with_fundamentals.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Features not found: {path}")
        df = pd.read_parquet(path)
        logger.info(f"✓ Loaded features: {len(df):,} rows")
        return df

    def save_scored_df(self, df: pd.DataFrame) -> Path:
        """
        Save: data/results/scored_df.parquet

        Args:
            df: DataFrame with columns [date, symbol, ml_score]

        Returns:
            Path to saved file
        """
        path = self.results_dir / "scored_df.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"✓ Saved scored data: {path} ({len(df):,} rows)")
        return path

    def load_scored_df(self) -> pd.DataFrame:
        """Load scored data artifact"""
        path = self.results_dir / "scored_df.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Scored data not found: {path}")
        df = pd.read_parquet(path)
        logger.info(f"✓ Loaded scored data: {len(df):,} rows")
        return df

    def save_weights_df(self, df: pd.DataFrame) -> Path:
        """
        Save: data/results/weights_df.parquet

        Args:
            df: DataFrame with columns [date, symbol, weight]

        Returns:
            Path to saved file
        """
        path = self.results_dir / "weights_df.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"✓ Saved portfolio weights: {path} ({len(df):,} rows)")
        return path

    def load_weights_df(self) -> pd.DataFrame:
        """Load portfolio weights artifact"""
        path = self.results_dir / "weights_df.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Weights not found: {path}")
        df = pd.read_parquet(path)
        logger.info(f"✓ Loaded weights: {len(df):,} rows")
        return df

    def save_backtest_results(self, results: Dict[str, Any]) -> Path:
        """
        Save: data/results/backtest_results.json

        Args:
            results: Dictionary with keys:
                - metrics: Dict of performance metrics
                - equity_curve: DataFrame or dict
                - trades: Optional trades list

        Returns:
            Path to saved file
        """
        path = self.results_dir / "backtest_results.json"

        # Convert DataFrames to JSON-serializable format
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict('records')
            else:
                serializable_results[key] = value

        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"✓ Saved backtest results: {path}")
        return path

    def load_backtest_results(self) -> Dict[str, Any]:
        """Load backtest results artifact"""
        path = self.results_dir / "backtest_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Backtest results not found: {path}")

        with open(path, 'r') as f:
            results = json.load(f)

        logger.info(f"✓ Loaded backtest results")
        return results

    def save_model(self, model: Any, model_name: str = "model.pkl") -> Path:
        """
        Save: models/latest/model.pkl

        Args:
            model: Trained model object
            model_name: Filename (default: model.pkl)

        Returns:
            Path to saved file
        """
        latest_dir = self.models_dir / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)

        path = latest_dir / model_name

        with open(path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"✓ Saved model: {path}")
        return path

    def load_model(self, model_name: str = "model.pkl") -> Any:
        """Load trained model"""
        path = self.models_dir / "latest" / model_name
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        with open(path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"✓ Loaded model: {path}")
        return model

    # ========================================================================
    # ENHANCEMENT ARTIFACTS (instruction2.md outputs)
    # ========================================================================

    def save_regime_state(self, df: pd.DataFrame) -> Path:
        """
        Save: data/results/regime_state.csv

        Args:
            df: DataFrame with regime detection results

        Returns:
            Path to saved file
        """
        path = self.results_dir / "regime_state.csv"
        df.to_csv(path, index=False)
        logger.info(f"✓ Saved regime state: {path}")
        return path

    def load_regime_state(self) -> pd.DataFrame:
        """Load regime state artifact"""
        path = self.results_dir / "regime_state.csv"
        if not path.exists():
            raise FileNotFoundError(f"Regime state not found: {path}")
        df = pd.read_csv(path, parse_dates=['date'] if 'date' in pd.read_csv(path, nrows=1).columns else None)
        logger.info(f"✓ Loaded regime state")
        return df

    def save_sleeve_allocation(self, allocation: Dict[str, Any]) -> Path:
        """
        Save: data/results/sleeve_allocation.json

        Args:
            allocation: Dictionary with sleeve allocations (equities, crypto, cash)

        Returns:
            Path to saved file
        """
        path = self.results_dir / "sleeve_allocation.json"

        with open(path, 'w') as f:
            json.dump(allocation, f, indent=2, default=str)

        logger.info(f"✓ Saved sleeve allocation: {path}")
        return path

    def load_sleeve_allocation(self) -> Dict[str, Any]:
        """Load sleeve allocation artifact"""
        path = self.results_dir / "sleeve_allocation.json"
        if not path.exists():
            raise FileNotFoundError(f"Sleeve allocation not found: {path}")

        with open(path, 'r') as f:
            allocation = json.load(f)

        logger.info(f"✓ Loaded sleeve allocation")
        return allocation

    def save_turnover_report(self, report: Dict[str, Any]) -> Path:
        """
        Save: data/results/turnover_report.json

        Args:
            report: Dictionary with turnover analytics

        Returns:
            Path to saved file
        """
        path = self.results_dir / "turnover_report.json"

        # Convert any DataFrames to records
        serializable_report = {}
        for key, value in report.items():
            if isinstance(value, pd.DataFrame):
                serializable_report[key] = value.to_dict('records')
            else:
                serializable_report[key] = value

        with open(path, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)

        logger.info(f"✓ Saved turnover report: {path}")
        return path

    def load_turnover_report(self) -> Dict[str, Any]:
        """Load turnover report artifact"""
        path = self.results_dir / "turnover_report.json"
        if not path.exists():
            raise FileNotFoundError(f"Turnover report not found: {path}")

        with open(path, 'r') as f:
            report = json.load(f)

        logger.info(f"✓ Loaded turnover report")
        return report

    def save_ops_report(self, report_html: str, date_suffix: Optional[str] = None) -> Path:
        """
        Save: data/results/ops_report_{YYYYMMDD}.html

        Args:
            report_html: HTML content of ops report
            date_suffix: Date string (YYYYMMDD), if None uses current date

        Returns:
            Path to saved file
        """
        if date_suffix is None:
            from datetime import datetime
            date_suffix = datetime.now().strftime("%Y%m%d")

        path = self.results_dir / f"ops_report_{date_suffix}.html"

        with open(path, 'w') as f:
            f.write(report_html)

        logger.info(f"✓ Saved ops report: {path}")
        return path

    # ========================================================================
    # VALIDATION & UTILITIES
    # ========================================================================

    def validate_core_artifacts(self) -> Dict[str, bool]:
        """
        Validate that all 5 required core artifacts exist

        Returns:
            Dict mapping artifact name to existence status
        """
        artifacts = {
            "features": self.features_dir / "all_features_with_fundamentals.parquet",
            "scored_df": self.results_dir / "scored_df.parquet",
            "weights_df": self.results_dir / "weights_df.parquet",
            "backtest_results": self.results_dir / "backtest_results.json",
            "model": self.models_dir / "latest" / "model.pkl"
        }

        status = {name: path.exists() for name, path in artifacts.items()}

        logger.info("Core artifact validation:")
        for name, exists in status.items():
            symbol = "✓" if exists else "✗"
            logger.info(f"  {symbol} {name}")

        return status

    def validate_enhancement_artifacts(self) -> Dict[str, bool]:
        """
        Validate that all 4 required enhancement artifacts exist

        Returns:
            Dict mapping artifact name to existence status
        """
        artifacts = {
            "regime_state": self.results_dir / "regime_state.csv",
            "sleeve_allocation": self.results_dir / "sleeve_allocation.json",
            "turnover_report": self.results_dir / "turnover_report.json",
            "ops_report": list(self.results_dir.glob("ops_report_*.html"))
        }

        status = {}
        for name, path in artifacts.items():
            if isinstance(path, list):  # ops_report pattern match
                status[name] = len(path) > 0
            else:
                status[name] = path.exists()

        logger.info("Enhancement artifact validation:")
        for name, exists in status.items():
            symbol = "✓" if exists else "✗"
            logger.info(f"  {symbol} {name}")

        return status

    def list_artifacts(self) -> Dict[str, list]:
        """List all available artifacts"""
        return {
            "features": list(self.features_dir.glob("*.parquet")),
            "results": list(self.results_dir.glob("*")),
            "models": list(self.models_dir.glob("**/*.pkl"))
        }
