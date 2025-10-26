rebalance:
  equities_frequency: weekly
  crypto_frequency: 3d
labels:
  equities_horizons: [5,10,20]
  crypto_horizons: [3,7]
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
  method: erc
  crypto_risk_budget_max: 0.25
ops:
  alerts: email