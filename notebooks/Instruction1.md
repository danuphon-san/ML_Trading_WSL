# Core ML Portfolio Pipeline (Steps 1–11)

## Scope
Build features (technical + PIT fundamentals), generate labels, train supervised model,
score assets, construct portfolio, and backtest.

## Prerequisites
- configs/config.yaml
- data/ohlcv/*.parquet
- data/fundamentals/fundamentals.parquet

## Outputs (consumed by instructions2.md)
- data/features/all_features_with_fundamentals.parquet
- data/results/scored_df.parquet
- data/results/weights_df.parquet
- data/results/backtest_results.json
- models/latest/model.pkl

🧠 ML Portfolio Pipeline Guidelines
📘 Overview

This project implements a Feature-Based Alpha Modeling pipeline — a supervised learning workflow that predicts expected future returns (alphas) for portfolio construction.

The pipeline follows 11 key stages, from data fetching to model evaluation.
Each stage is organized into a modular Python script for clean execution and scalability.

🗂️ Folder Structure
project_root/
│
├── data/
│   ├── ohlcv/               # Raw OHLCV data (.parquet)
│   ├── fundamentals/        # Raw fundamental data (.parquet)
│   ├── features/            # Engineered features
│   ├── processed/           # Intermediate cleaned data
│   └── results/             # Backtest and evaluation outputs
│
├── modules/
│   ├── 01_ingest_data.py
│   ├── 02_preprocess_data.py
│   ├── 03_features_technical.py
│   ├── 04_features_fundamental.py
│   ├── 05_label_generation.py
│   ├── 06_feature_selection.py
│   ├── 07_model_training.py
│   ├── 08_signal_generation.py
│   ├── 09_portfolio_construction.py
│   ├── 10_backtesting.py
│   └── 11_evaluation.py
│
└── run_pipeline.py

⚙️ Execution Order (Top → Bottom)
Step	Module	Description	Output
1️⃣ Data Fetching	01_ingest_data.py	Fetch OHLCV (price/volume) and fundamental data from Yahoo Finance using yfinance.	data/ohlcv/*.parquet, data/fundamentals/*.parquet
2️⃣ Data Preprocessing	02_preprocess_data.py	Clean and align OHLCV + fundamentals; fill missing data; unify date indices.	data/processed/df_base.parquet
3️⃣ Technical Feature Engineering	03_features_technical.py	Compute RSI, MACD, Momentum, Volatility, etc.	data/features/df_tech_features.parquet
4️⃣ Fundamental Feature Engineering	04_features_fundamental.py	Compute ratios like P/E, ROE, D/E, growth metrics, and align by public_date.	data/features/df_with_fundamentals.parquet
5️⃣ Label Generation	05_label_generation.py	Create supervised learning labels (e.g., 5-day forward returns).	data/features/df_with_labels.parquet
6️⃣ Feature Selection & Normalization	06_feature_selection.py	Merge all features; normalize and select using correlation or variance filters.	Train/Test split files
7️⃣ Model Training	07_model_training.py	Train ML model (LightGBM / XGBoost / RF); log metrics to MLflow.	Saved model
8️⃣ Scoring & Signal Generation	08_signal_generation.py	Use trained model to predict expected future returns.	data/results/scored_df.parquet
9️⃣ Portfolio Construction	09_portfolio_construction.py	Convert scores to weights using optimizer (equal-weight, MVO, HRP).	data/results/weights_df.parquet
🔟 Backtesting	10_backtesting.py	Simulate portfolio performance historically.	data/results/backtest_results.json
11️⃣ Model Evaluation	11_evaluation.py	Compute Sharpe Ratio, IC, Drawdown, Turnover, and feature importance.	Performance report
🚀 Running the Full Pipeline

Use the orchestrator script to run all modules sequentially:

# run_pipeline.py
import subprocess

modules = [
    "01_ingest_data.py",
    "02_preprocess_data.py",
    "03_features_technical.py",
    "04_features_fundamental.py",
    "05_label_generation.py",
    "06_feature_selection.py",
    "07_model_training.py",
    "08_signal_generation.py",
    "09_portfolio_construction.py",
    "10_backtesting.py",
    "11_evaluation.py",
]

for m in modules:
    print(f"\n🚀 Running {m} ...")
    subprocess.run(["python", f"modules/{m}"], check=True)


Run once:

python run_pipeline.py

🧠 Design Philosophy
Principle	Description
Feature-Based Alpha Modeling	Predicts expected returns (continuous label) using supervised ML
Supervised Learning	Model learns mapping: Features → Future Return
Point-in-Time (PIT)	Fundamental data aligned to publication date to avoid look-ahead bias
Cross-Sectional Training	Model compares all stocks each period to rank likely outperformers
Weekly Rebalancing	Buy/hold top-ranked stocks; rebalance weekly for signal refresh
Bias Control	Strict forward-shifted labels and PIT fundamentals ensure realistic backtest
🧩 Rebalancing Logic

Example (weekly roll-over):

Date	Action
Jan 1	Score → Buy top 30 stocks
Jan 8	Re-score → Sell underperformers, buy new top picks
Jan 15	Repeat cycle

Hold period = 1 week

Rebalance frequency = weekly

Configurable: top-K, costs, slippage, rebalance period

📈 Evaluation Metrics
Category	Metric	Purpose
Model Quality	IC, Rank IC	Measures correlation between predicted vs. actual returns
Portfolio Performance	Sharpe, Total Return, Max Drawdown	Risk-adjusted performance
Trading Cost Control	Turnover, Slippage Impact	Measures trading frequency and cost
Explainability	Feature Importance, SHAP values	Identify key alpha drivers
💡 Development Workflow
Mode	Description
Initial Build	Run all 11 modules sequentially.
Experimenting	Skip ingestion → start from step 3 onward (feature updates).
Model Tuning	Iterate between steps 6–8 only.
Strategy Optimization	Iterate between steps 9–10 with different optimizers.
🧱 Bias Control Checklist

✅ Use public_date alignment for fundamentals.
✅ Generate labels with .shift(-N) to prevent look-ahead bias.
✅ Split data by time (not randomly).
✅ Use train-validation-test chronological order.
✅ Evaluate model out-of-sample before deployment.

🛠️ Optional Enhancements

12_feature_importance.py — Compute SHAP or permutation importance.

13_hyperparameter_optimization.py — Optuna tuning for LightGBM.

14_rolling_retrain.py — Adaptive retraining over time windows.

15_live_inference.py — Deploy for daily signal generation.

✅ Summary

Your ML pipeline is a complete supervised learning system for alpha discovery and portfolio construction:

Data Fetching → Preprocessing → Feature Engineering → Labeling → Model Training
→ Scoring → Portfolio Optimization → Backtesting → Evaluation


Each step is modular, reproducible, and compatible with professional quant research environments.