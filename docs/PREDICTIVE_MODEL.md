# Predictive model and multi-supplier orders (2026-10-19)

## What changed

| Before | Now |
|---|---|
| Supplier `available_qty`, `defect_rate`, `fulfillment_rate` were **random draws** (seed 20260926) | No random numbers. Every value is learned or measured from real data. |
| Urgent = inventory below the planner's fixed reorder point | Urgent = below the planner's point **or** the point the demand forecast says the coming demand needs |
| Order size = median of past order sizes | Order size = forecast demand over lead time + order cycle, plus 95% safety stock, minus stock on hand |
| Daily demand = average of the last 28 days | Daily demand = trained forecast for each future day |
| Order split only if the pool could cover it, else "unprocessable" | Split across the fewest suppliers that cover it; if even all together fall short, a **partial order** is planned and the shortfall reported |

## 1. Demand forecaster (`backend/ml/demand_forecaster.py`)

- **Learns from:** 91,250 real daily rows (`Units_Sold`, `Date`, `Promotion_Flag`), 250 SKU-warehouse lanes, 2024.
- **Model:** regression on the yearly season (Fourier terms, count chosen by cross-validation) plus a promotion lift that changes with the season. Each lane gets its own scale, shrunk towards the shared curve (empirical Bayes), so a lane only differs when its data really shows it.
- **Accuracy (honest test: train on the past only, forecast the next 14 days):** about 8% error, against 17% for the previous 28-day average, so the error is roughly halved. A gradient-boosting model was also tried and was not better (WAPE 0.062 vs 0.057), because what is left over is pure noise.
- **Not used:** the source column `Demand_Forecast`. It matches the *same* day's sales (correlation 0.95) far better than the next day's (0.61), so it was made from the answer and is not a real forecast.
- Pure NumPy, trained in about 5 seconds when the database is built. No new dependencies.

How the Perception Agent uses it, per warehouse:

```
demand during lead time   ~ Normal(mu_L, sd_L)          from the model
forecast reorder point    = mu_L + 1.645 * sd_L          (95% service level)
urgent                    = inventory < max(planner reorder point, forecast reorder point)
stock-out risk            = P(demand during lead time > inventory)
order up to               = mu_(L+C) + 1.645 * sd_(L+C)  C = the lane's own order cycle
required quantity         = sum over urgent lanes of (order up to - inventory), rounded up to 10
```

## 2. Supplier estimator (`backend/ml/supplier_estimator.py`)

| Field | How it is estimated |
|---|---|
| `available_qty` | The **largest order this supplier has actually delivered** for this SKU in the ledger (at least 10 orders needed, else unknown). It is a proven minimum, not live stock. |
| `defect_rate` | Empirical Bayes: starts at the real average defect rate of the 5 KPI vendors (7.0%) and moves towards a supplier's own rate as it gets real records. |
| `fulfillment_rate` | Beta-Binomial: starts at the real share of KPI orders fully delivered (80.5%), updated the same way. |

The KPI file's vendors (Alpha_Inc …) share no key with SUP_1–SUP_10. A test showed that the order features both files share cannot predict defects (R² −0.10). So until real records are added (`DataSets/supplier_evidence/`), every supplier sits on the same baseline and is shown as **Unrated**. Ranking is then decided by real cost, real lead time and proven capacity. That still gives different best suppliers for different SKUs (10 different top suppliers across the 50 SKUs).

## 3. Multi-supplier orders (Knowledge Agent Rule 6)

When no single supplier can cover the required quantity:

1. Choose the supplier set: one that covers the order, with the fewest High Risk suppliers, then the fewest suppliers, then the best ranks.
2. Fill it best-ranked first, never asking a supplier for more than its capacity.
3. If even all suppliers together fall short, order everything available (a **partial order**) and report the shortfall.

Nothing is planned only when no supplier has any known capacity at all.
