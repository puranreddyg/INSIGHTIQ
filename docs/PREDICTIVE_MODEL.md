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

---

# Update 2026-10-20: supplier performance learned from order outcomes

## Predictions supported

| Prediction | Original dataset | After the dataset change |
|---|---|---|
| Daily demand per SKU-warehouse | ✅ learned (14-day error 8.4% vs 17.2%) | ✅ same |
| Stock-out risk, reorder point, order size | ✅ | ✅ also accounts for each supplier's predicted delay and its variability |
| Supplier capacity | ⚠️ largest order (floor) | ⚠️ largest delivery (floor) |
| Supplier defect rate | ❌ no data (every supplier on the same baseline, "Unrated") | ✅ learned per SKU-supplier |
| On-time probability, expected delay, real lead time | ❌ all orders instant | ✅ learned |
| Fill rate / short shipments | ❌ all orders complete | ✅ learned |
| Preferred / High Risk / Reliable labels | ❌ never assigned | ✅ from learned values |

## Supplier model (`backend/ml/supplier_model.py`)
Hierarchical empirical Bayes (partial pooling SKU-supplier → supplier → all), with
pooling strength estimated from the data. Backtest: trained on orders received before
1 Oct 2024, tested on 865 later orders:

| | Model | Average-supplier baseline |
|---|---|---|
| Defect rate error (pp) | 1.05 | 3.91 |
| On-time log-loss | 0.489 | 0.516 |
| Delay error (days) | 1.23 | 1.33 |
| Supplier defect ranking vs actual | 0.988 rank correlation | — |

## How each agent uses the models
- **Perception:** reorder point and safety stock use the learned lead time (promised + predicted delay) and its uncertainty.
- **Supplier:** heuristic scores use learned defect rate, on-time-in-full and lead time.
- **Knowledge:** rules fire on learned values; Rule 6 runs the hybrid heuristic search (`backend/agents/procurement_search.py`) whenever no single supplier can cover the order within the quality rule.
- **Planning:** arrival dates from learned lead times; reports expected usable units, on-time probability and a quality flag.

## Hybrid heuristic search (Rule 6)
Best-first branch-and-bound over supplier combinations, guided by the learned heuristic
scores. Hard constraints: capacity per supplier, full quantity, quantity-weighted
predicted defect rate ≤ 5%, at most 3 suppliers. Objective: fewest High Risk suppliers,
then fewest suppliers, then highest score. An exhaustive test confirms it finds the
optimum for every urgent SKU. If no combination can meet the 5% rule, the lowest-defect
option is chosen and flagged; if capacity is short, a partial order reports the shortfall.

---

# Update 2026-10-20 (2): budget constraint, exact allocation, forecast evaluation page

**Supplier selection.** A single supplier is chosen only if it meets capacity, quality (not High Risk) and
**budget** (Rule 7: qty × unit cost ≤ budget; default budget = qty × the SKU's mean lane unit cost × 1.10,
or the `budget` query parameter). Otherwise Rule 6 searches combinations; inside each candidate set the
quantities are solved exactly with an integer linear program (scipy `milp`) under: capacity, ≥ 50 units per
line, full quantity, weighted predicted defect ≤ 5%, total cost ≤ budget, ≤ 3 suppliers. **A split is made
only if every constraint holds.** If none does: a single supplier that can cover the order alone is chosen
with the smallest violation and flagged; otherwise the SKU goes to **needs review** (no order). Partial
orders (capacity short) must still meet quality and budget. An independent brute-force enumeration in the
tests confirms the result is optimal for every urgent SKU.

**Forecast Evaluation page** (`/api/forecast-evaluation`, `backend/ml/forecast_evaluation.py`).
Train 2024-01-01–09-30, test 10-01–12-30 (91 days, unseen). Weekly re-forecast protocol:

| Level | Method | MAE | RMSE | MAPE |
|---|---|---|---|---|
| All-SKU daily total | Learned model | 72.0 | 88.0 | 2.28% |
| | 28-day average (previous) | 426.3 | 496.4 | 11.55% |
| | 7-day average | 209.2 | 247.4 | 5.88% |
| Per lane per day | Learned model | 4.16 | 5.23 | 49.0% |
| | 28-day average | 4.49 | 5.67 | 47.1% |
| | 7-day average | 4.49 | 5.65 | 49.6% |

The model wins MAE/RMSE at both levels; per-lane MAPE is slightly worse than the 28-day average
(MAPE over-weights low-sales days; zero-sales days are excluded).
