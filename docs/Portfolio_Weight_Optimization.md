# 📘 Modern Portfolio Weight Optimization — Instruction + Testing Matrix

This document expands the optimizer design with a **Testing Matrix** for systematic evaluation of portfolio weighting strategies.

---

## 🎯 Objective

To allow testing and comparison between different portfolio optimizers under consistent market conditions, using standardized evaluation metrics.

---

## 🧩 1. Optimizer Methods (Selectable Options)

| Method                               | Description                                | Goal                                               |
| ------------------------------------ | ------------------------------------------ | -------------------------------------------------- |
| **Equal Weight**                     | Assign equal weights to top-K assets       | Simple benchmark                                   |
| **Score-Weighted**                   | Weights proportional to ML signal score    | Utilize model confidence                           |
| **Inverse Volatility**               | Weight = 1 / volatility                    | Risk parity balance                                |
| **Mean-Variance Optimization (MVO)** | Maximize Sharpe Ratio or minimize variance | Optimal tradeoff between risk & return             |
| **Regularized MVO**                  | Add L2 penalty on weights                  | Improves stability                                 |
| **Hierarchical Risk Parity (HRP)**   | Cluster assets by correlation              | Diversified and robust allocation                  |
| **Hybrid Score + Risk**              | Combine ML scores and inverse volatility   | Balance between predictive power & diversification |

Each of these can be selected in the configuration file:

```python
config['portfolio']['optimizer'] = 'hrp'  # options: equal, score_weighted, inv_vol, mvo, mvo_reg, hrp, hybrid
```

---

## 🧠 2. Testing Matrix — Comparative Evaluation Plan

| Test ID | Optimizer           | Key Features               | Expected Behavior                    | Evaluation Focus                    |
| ------- | ------------------- | -------------------------- | ------------------------------------ | ----------------------------------- |
| T1      | Equal Weight        | Baseline allocation        | Stable but low Sharpe                | Benchmark comparison                |
| T2      | Score-Weighted      | Based on ML signals        | Reflects model predictive power      | Correlation between signal & return |
| T3      | Inverse Volatility  | Risk-balanced              | Smooth volatility profile            | Lower drawdown consistency          |
| T4      | MVO                 | Risk-return optimized      | Higher Sharpe (if stable covariance) | Covariance estimation quality       |
| T5      | Regularized MVO     | Penalized large weights    | More stable allocations              | Robustness under small samples      |
| T6      | HRP                 | Hierarchical risk clusters | Diversified and low turnover         | Regime robustness                   |
| T7      | Hybrid Score + Risk | Weighted by signal × risk  | Adaptive tradeoff                    | Best practical compromise           |

---

## 📈 3. Evaluation Metrics

Each optimizer is evaluated on the same backtest period using consistent cost assumptions.

| Category          | Metric                          | Formula / Meaning           |               |       |
| ----------------- | ------------------------------- | --------------------------- | ------------- | ----- |
| **Performance**   | Total Return, Annualized Return | ( \frac{P_T}{P_0} - 1 )     |               |       |
| **Risk**          | Volatility, Max Drawdown        | Rolling std, peak-to-trough |               |       |
| **Efficiency**    | Sharpe Ratio, Sortino Ratio     | Risk-adjusted return        |               |       |
| **Execution**     | Turnover                        | ( \sum                      | w_t - w_{t-1} | ) / n |
| **Concentration** | Herfindahl Index                | ( \sum w_i^2 )              |               |       |

---

## ⚙️ 4. Test Automation Example

```python
methods = ['equal', 'score_weighted', 'inv_vol', 'mvo', 'mvo_reg', 'hrp', 'hybrid']
results_summary = []

for method in methods:
    config['portfolio']['optimizer'] = method
    weights_df = build_portfolio_weights(scored_df, price_panel, config)
    backtester = VectorizedBacktester(config)
    results = backtester.run(weights_df, price_panel)

    results_summary.append({
        'method': method,
        'sharpe': results['metrics']['sharpe_ratio'],
        'drawdown': results['metrics']['max_drawdown'],
        'annual_return': results['metrics']['annual_return']
    })

summary_df = pd.DataFrame(results_summary)
print(summary_df.sort_values('sharpe', ascending=False))
```

---

## 🧩 5. Interpretation Framework

| Observation                 | Interpretation                       | Action                           |
| --------------------------- | ------------------------------------ | -------------------------------- |
| High Sharpe + Low Drawdown  | Efficient allocation                 | Prioritize for production        |
| High Turnover + Same Sharpe | Overfitting or excessive rebalancing | Increase rebalance interval      |
| HRP > MVO Stability         | Covariance estimation too noisy      | Use shrinkage or HRP             |
| Hybrid Outperforms          | ML scores add predictive alpha       | Keep hybrid weighting as default |

---

## ✅ 6. AI Assistant Review Checklist

The assistant should verify:

* [ ] Time alignment — no future returns in features.
* [ ] Normalization — weights sum ≈ 1.0.
* [ ] Constraints — no shorting if not allowed.
* [ ] Risk scaling — vol estimation window consistent.
* [ ] Comparative metrics — Sharpe, drawdown, turnover computed consistently.
* [ ] Summary table — clearly identifies the best-performing optimizer.

---

**Author:** Danuphon Santiwong
**Purpose:** Enable structured, bias-free testing and benchmarking of modern portfolio optimization techniques for ML-driven trading systems.