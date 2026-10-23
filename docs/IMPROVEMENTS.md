# InsightIQ — Split-Order Allocation, Forecast-Based Quantity and Demand Model

This note covers three changes to the four-agent pipeline. Numbers come from the real
`supply_chain_dataset1.csv` (snapshot 2024-12-30) and the committed model artifact.

## Data checks done first

| Check | Result |
|---|---|
| Rows per (date, SKU, warehouse) | exactly 1 (91,250 rows = 50 SKUs × 5 warehouses × 365 days) |
| Suppliers per (SKU, warehouse) lane | exactly 1 all year (3–5 per SKU across warehouses) |
| Days with zero sales | 0.7% |
| Day-of-week effect | none (daily means 19.99–20.11) |
| Annual pattern | strong (monthly mean 10.2 in Sep, 29.9 in Mar), but only **one** year of data |
| Inventory identity | `inv[t] = inv[t-1] − sold[t] + ordered[t]`: orders arrive the same day, so there is no on-order stock |
| Reorder point vs lead time | unrelated (corr 0.03) |
| Real order cycle per lane | median 15 days (11–17.5) |

Lane-day aggregation still protects against double counting if a future feed repeats a
lane-day. It drops exact duplicates and **sums** Units_Sold. Inventory and reorder point
are state readings, so they are read **once** per lane and never summed.

## 1. Multi-supplier (split-order) allocation

**Flow**

1. The Supplier Agent ranks single suppliers that can cover the full quantity
   (`available_qty ≥ required`), as before.
2. Whenever a quantity is required, it also scores every supplier with stock using the
   same heuristic. This pool is returned as `allocation_pool`.
3. The Knowledge Agent picks the top-ranked eligible supplier that is not High Risk. That
   is the **single-supplier path**, and it is unchanged.
4. If there is no such supplier, the agent switches to **split mode**. It runs Rule 5 with
   an ELSE branch of "Partial Capacity" instead of "Ineligible". The chain then
   continues, so Rules 2–4 still run, **including the High Risk check**.
5. The Planning Agent calls `backend/agents/allocation.py`.

**Allocation policy** (deterministic):

* **High Risk suppliers are never allocated automatically.** They are listed in
  `excluded_high_risk` with their availability.
* Acceptable suppliers are filled in **heuristic-rank order**, each up to its
  `available_qty`, until the requirement is met. Ties break on `supplier_id`.
* The total allocated never exceeds the requirement, and no line exceeds its supplier's
  availability.
* A shortfall is never silent:
  * `is_partial`, `residual_shortage` and `max_feasible_quantity` are always reported,
    plus `max_feasible_quantity_ignoring_risk`.
  * `shortage_reasons` explains why: combined stock too low, High Risk exclusion, or both.

**Trade-offs**

* Rank-greedy reuses the explainable ranking, which already weights cost 25%, delivery
  20%, quality 30% and fulfillment 25%.
* It is **not** a cost optimiser. It can cost more than the cheapest feasible split and
  can use more suppliers than the minimum.
* The plan is complete only when the slowest supplier delivers. Each line therefore keeps
  its own arrival date, and `estimated_arrival_date` is the last one.

**Behaviour change.** Previously, when every supplier able to cover the order alone was
High Risk, the agent picked the top one anyway. Now it splits the order across acceptable
suppliers, or reports a partial plan. Rule 3 is never bypassed.

**Plan statuses**

| Status | Meaning |
|---|---|
| `planned` | Single supplier |
| `planned_split` | Full split |
| `planned_partial` | Explicitly partial |
| `no_eligible_supplier` | No acceptable supplier has stock |
| `no_action` | Nothing to order |

Every planned status carries `allocations[]`. Each line holds `supplier_id`,
`allocated_qty`, `unit_cost`, `line_cost`, `lead_time_days`, `estimated_arrival`,
`risk_status` and provenance. The legacy single-supplier fields are kept:
`primary_supplier_id`, `order_quantity`, `unit_cost` (a quantity-weighted average for a
split), `total_expense`, `estimated_arrival_date` and `backup_supplier_id`.

**Approval** (`POST /api/approve-order` with `allocations[]`)

The server re-derives everything before writing:

* the current requirement
* each supplier's lane and profile
* availability
* risk status (Rule 3)
* unit cost

It rejects the request (HTTP 422) if any line exceeds availability, a supplier repeats,
a supplier is High Risk, the total exceeds the requirement, or the total is below the
requirement without `is_partial: true`.

All lines are written in **one transaction** with a shared `order_group`. A failure on
any line rolls back the whole order; a test forces this with a SQLite trigger. The legacy
single-supplier payload still works.

Real-data examples (forecast quantities):

| SKU | Status | Allocation |
|---|---|---|
| SKU_44 | `planned_split` | 450 = SUP_9 348 + SUP_1 102. Three suppliers are High Risk; arrivals 11 and 12 Nov. |
| SKU_19 | `planned_partial` | 200 of 260. The other 1,031 units sit with High Risk suppliers. |
| SKU_47 | `planned` | Single supplier SUP_1. |

## 2. Forecast-based RequiredQuantity (Perception Agent)

The urgency rule is unchanged: a lane is urgent when `Inventory < ReorderPoint`. For each
**urgent** lane:

```
L  = lane's real supplier lead time
R  = review period = median days between the lane's real historical orders
     (config FORECAST_REVIEW_PERIOD_DAYS = 14 if a lane has < 2 orders)
H  = L + R
D  = forecast demand over the next H days
SS = z(SERVICE_LEVEL) · σ_daily · √H      (σ over the last 28 days; 95% → z = 1.645)
Target   = D + SS
LaneQty  = max(0, Target − Inventory)
Required = ceil(Σ LaneQty over urgent lanes / 10) × 10
```

**Why `L + R` and not `L` alone.** Reorder points in this dataset sit well above
lead-time demand. A pure lead-time target is below current stock on 13 of the 14 urgent
lanes, so it would order nothing for SKUs that are urgent. `L + R` is the standard
periodic-review protection interval. Setting `FORECAST_REVIEW_PERIOD_SOURCE = "fixed"`
and `FORECAST_REVIEW_PERIOD_DAYS = 0` gives the pure lead-time version.

**Safety-stock assumptions.** Daily demand is treated as i.i.d. normal, and lead time as
deterministic (it is constant per lane in the source). The alternative method is
`SAFETY_STOCK_METHOD = "days_of_cover"`, which uses SAFETY_STOCK_DAYS × mean demand.

**Aggregation.** Quantities are summed over urgent lanes, each counted once. Non-urgent
lanes order nothing.

**Fallback.** `REQUIRED_QUANTITY_METHOD = "median"` restores the legacy rule (sum of the
urgent lanes' median historical lot sizes). The forecast method also falls back to it
automatically if forecast inputs are incomplete.

**Output.** `quantity_calculation` holds the method, forecast method and reason, forecast
demand, safety stock, target stock, current inventory, quantity before rounding, final
quantity, the median-method quantity for comparison, and a per-lane explanation. Each
lane fact also carries LeadTimeDays, ReviewPeriodDays, HorizonDays, ForecastDemand,
SafetyStock, TargetStock and LaneQuantity.

On real data the forecast quantities (90–450) are lower than the median-lot quantities
(300–790).

## 3. Lightweight demand model

The model is a pooled **ridge regression in NumPy**, so it adds no new dependency for
Render.

* **Features:** `lag_1`, `lag_2`, `lag_7`, 7- and 28-day trailing means, and day-of-week.
* **Excluded on purpose:**
  * Month or annual seasonality: there is one year, so it cannot be validated.
  * `Promotion_Flag`: unknown for future days.
  * `Demand_Forecast`: correlates 0.95 with same-day sales, which suggests leakage.

**Validation** is chronological. The model trains on 2024-01-29 → 2024-11-04 and is
tested on 2024-11-05 → 2024-12-30. The scaler and coefficients are fitted on training days
only, and a test proves that changing any test-period value leaves the fit unchanged.

| Held-out (250 lanes) | Model | 28-day MA | Seasonal naive (weekly) |
|---|---|---|---|
| Next-day WAPE | 29.1% | 29.1% | 38.3% |
| 21-day total WAPE (6 origins) | 23.0% | 21.0% | **15.9%** |

**The model does not beat the baselines,** so it is not used. `FORECAST_METHOD = "auto"`
picks the method with the lowest held-out 21-day WAPE, which is currently **seasonal
naive**. All methods under-forecast (bias −13% to −23%) because demand was rising in
Nov–Dec.

**Artifact and fallback**

* Run `python -m backend.forecasting.train` to write
  `backend/forecasting/artifacts/demand_model.json` (byte-identical on re-run).
* Nothing is retrained on a web request.
* A missing or invalid artifact falls back to the moving average, with the reason
  reported.

**Endpoints and UI**

* `GET /api/forecast/metrics` and `GET /api/forecast?sku_id=…&horizon_days=…`
* The Pipeline & XAI view has a forecast summary card. It shows the method in use, the
  reason and both metric tables.

## Limitations (slide content)

* **Supplier availability, defect rate and fulfillment rate are SIMULATED.**
  * Split plans are planning aids, not reservations of live stock.
  * Approvals do not decrement `available_qty`, so approving the same plan twice is
    possible.
* **Rank-greedy allocation is explainable, not cost-optimal.** It may cost more, or use
  more suppliers, than the cheapest feasible split.
* **High Risk suppliers are never auto-allocated.** A risk-driven shortage needs a manual
  buyer decision; there is no override flow yet.
* **The forecast quantity assumes normal i.i.d. demand and a fixed lead time.**
  * The review period is inferred from historical order spacing, not from a real policy.
  * Orders arrive the same day in the data, so on-order stock is not modelled.
* **Only one year of history exists.** Annual seasonality cannot be learned or
  validated, and every method under-forecasts the Nov–Dec rise.
* **The ridge model did not beat the baselines on held-out data, so it is not used.**
  Its validation covers only 8 weeks, 6 horizon origins and one season.
* **Arrival dates are estimates.** They add working-day lead time and a 0–2 day
  calendar buffer.
* **No genetic algorithm or weight optimisation is included.** The heuristic weights
  remain hand-set; they can be injected through `weights`.
