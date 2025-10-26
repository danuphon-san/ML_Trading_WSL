# Enhancements & Ops (Steps 12–15)

## Scope
Regime detection, sleeve allocation (equities/crypto/cash), turnover management,
and monitoring/alerts. No feature/model retraining here.

## Prerequisites (from instructions1.md)
- models/latest/model.pkl
- data/results/scored_df.parquet
- data/results/weights_df.parquet
- data/results/backtest_results.json
- configs/config.yaml

## New Outputs
- data/results/sleeve_allocation.json
- data/results/turnover_report.json
- data/results/regime_state.csv
- data/results/ops_report_{YYYYMMDD}.html

🧠 Pro-Quant Upgrade Guidance (for AI Assistant)
🎯 Goal

Strengthen our feature-based, supervised alpha modeling pipeline with professional-grade practices for:

US Equities (incl. sector rotation & macro)

Top-Liquid Crypto (BTC, ETH, BNB, XRP, etc.)

Live-ready operations (cron, alerts, guardrails)

✅ What to KEEP

Feature-based alpha: technical + PIT fundamentals

Cross-sectional ranking / regression models

Weekly rebalance (equities); separate crypto sleeve

11-stage modular pipeline and backtester

🔧 What to MODIFY (High impact, low friction)
1) Sleeve-specific cadence & horizons

Equities: rebalance weekly; horizons = [5, 10, 20] days (pick via walk-forward IC/Sharpe)

Crypto: rebalance 2–3×/week; horizons = [3, 7] days

2) Use Ranking objective for equities

Prefer LightGBM/CB Ranker; primary metric: Rank IC

Keep regression variant for diagnostics

3) Walk-forward CV with purge + embargo

Rolling OOS selection of horizons & hyperparams

Prevent label overlap leakage

4) Turnover control

Turnover cap (e.g., ≤ 80% annualized)

Min trade threshold; optional L1/L2 penalty on weight changes

5) Volatility targeting per sleeve

Equities target vol ~ 10%

Crypto target vol ~ 30–40%

Scale portfolio weights to target realized vol

➕ What to ADD
A) Sector rotation track (equities)

Build sector ETF panel (XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLRE, XLU, XLC)

Add macro features: yield curve slope, PMI, CPI YoY, unemployment, credit spreads, DXY

Two-stage allocation: overweight sectors → stock-pick within those sectors

B) Cross-sleeve allocation optimizer

Allocate across Equities / Crypto / Cash

Start with ERC (equal risk contribution) or risk parity

Advanced: regularized MVO with risk budgets (e.g., crypto risk ≤ 25%)

C) Risk overlays & guardrails

Exposure limits: per-name ≤ 5%, per-sector ≤ 25%, net beta bounds

Liquidity limits: equities as %ADV; crypto as % of 30m volume

Kill switches: halt signals if daily loss > X%, or live Sharpe below threshold over N weeks

D) Model governance & diagnostics

MLflow model registry, tags

SHAP / feature importance each retrain; prune dead features quarterly

Report deflated Sharpe / bootstrap CIs

E) Monitoring & Ops

Cron schedules: equities after close, crypto daily or 3×/week

Email/Slack report: positions delta, turnover, sleeve vol, PnL, breaches

Runbook for outages, model failure, kill-switch

🗂️ Module Add-ons (extend 11-step flow)

regime_detection.py

Detect risk-on/off via vol, drawdown, macro (HMM or rules)

Output: risk multiplier, sector tilts

sleeve_allocation.py

Inputs: sleeve μ, vol, corr

Methods: ERC / regularized MVO + risk budgets

Output: sleeve weights

turnover_manager.py

Enforce turnover cap, min trade size, rounding, cost-aware deltas

monitoring.py

Generate daily/weekly ops report & alerts; log KPIs and breaches

⚙️ Config Additions (YAML)
rebalance:
  equities_frequency: weekly
  crypto_frequency: 3d
labels:
  equities_horizons: [5, 10, 20]
  crypto_horizons: [3, 7]
validation:
  scheme: walk_forward
  windows: 6
  embargo_days: 5
portfolio:
  per_name_cap: 0.05
  per_sector_cap: 0.25
  target_vol:
    equities: 0.10
    crypto: 0.35
  turnover_cap_annual: 0.80
allocation:
  method: erc   # erc | mvo_reg
  crypto_risk_budget_max: 0.25
ops:
  schedule:
    equities: "cron: 30 21 * * 1-5"   # after US market close UTC
    crypto:   "cron: 0 12 * * 1,3,5"  # example
  alerts: email
  kill_switch:
    max_daily_loss_pct: 0.03
    min_live_sharpe_lookback_weeks: 6

🧪 Validation & Testing Checklist
Data & Features

 PIT alignment for fundamentals (public_date)

 No forward-looking features; labels use .shift(-N)

 Train/val/test split is chronological

Modeling

 Walk-forward CV with purge & embargo

 Rank IC (primary), IC, MSE (secondary)

 SHAP/feature importance stored per run

Portfolio & Risk

 Vol targeting; turnover cap enforced

 Exposure, liquidity, sector caps validated

 Kill-switch logic unit-tested

Ops

 Cron jobs & retries set; logs persisted

 Email/Slack report includes positions delta, PnL, breaches

 Runbook exists for data/model failures

🧩 Implementation Tasks (AI Assistant)

Ranking model (equities): Implement LightGBM Ranker path; add Rank IC metric.

Walk-forward CV: Add purged K-fold with embargo; grid horizons by sleeve.

Turnover manager: Pre-trade smoothing + cap + min trade threshold.

Vol targeting: Scale weights to hit target sleeve volatility.

Sector rotation: Build sector ETF dataset + macro features; two-stage allocation.

Sleeve allocator: ERC baseline; optional regularized MVO with risk budgets.

Risk overlays: Exposure, liquidity caps; kill-switch module.

Monitoring: Cron configs, email template, run summaries & breach alerts.

Governance: MLflow model registry; SHAP plots; deflated Sharpe report.

Scenario analysis: Equity crash / crypto crash / both; save stress-test results.

📤 Email/Slack Report (Template)

Subject: Daily Strategy Report — {date}
Body:

PnL (day / MTD / YTD) by sleeve

Target vs. realized vol, turnover (vs. cap)

Position changes (top adds/cuts), exposure & sector caps

Breaches/Kill-switch status

Next scheduled run & data health

🧭 TL;DR

Keep the core pipeline.

Add ranking, walk-forward with embargo, vol targeting, turnover control.

Introduce sector rotation + sleeve allocation + risk guardrails.

Operationalize with cron + alerts + monitoring.

This elevates us from a solid research system to a professional, live-ready quant workflow.