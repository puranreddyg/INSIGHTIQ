# Simulated dataset — `supplier_sku_profile.csv`

**Everything in this folder is generated, not observed.** It fills three gaps that
the real datasets genuinely do not cover. All relationships in it are real.

| File | What it is |
|---|---|
| `supplier_sku_profile.csv` | The simulated table (211 rows) loaded into SQLite as `supplier_sku_profile` |
| `supplier_sku_profile.meta.json` | Seed, parameters, real statistics used, input/output SHA-256, generation time |
| `evidence/order_quantity_by_pair.csv` | Real historical order-size statistics for every SKU-supplier pair |

Generator: `backend/simulation/supplier_sku_profile.py` · Seed: **`20260926`**

## Data dictionary

| Field | Type | Status | Granularity | Source | Method |
|---|---|---|---|---|---|
| `sku_id` | text | **REAL** | row key | `supply_chain_dataset1.csv` `SKU_ID` | Existing relationship, copied |
| `supplier_id` | text | **REAL** | row key | `supply_chain_dataset1.csv` `Supplier_ID` | Existing relationship, copied |
| `available_qty` | integer (units) | **SIMULATED** | SKU-supplier | real `Order_Quantity` history of the same pair | One draw from that pair's own past order sizes |
| `defect_rate` | fraction 0–1 | **SIMULATED** | supplier | KPI `Defective_Units / Quantity` | Beta fitted to the 5 real supplier rates, capped at the real maximum |
| `fulfillment_rate` | fraction 0–1 | **SIMULATED** | supplier | KPI `Order_Status` | Binomial sampling around the real pooled rate |

`defect_rate` and `fulfillment_rate` are **supplier-level**: a supplier has the same
value on every SKU row. The KPI dataset has no SKUs, so SKU-specific quality cannot
be supported and is not claimed.

## Relationships
The 211 `(sku_id, supplier_id)` pairs are exactly the distinct pairs in
`supply_chain_dataset1.csv` (50 SKUs, 10 suppliers, 3–5 suppliers per SKU). No pair
was added. Warehouse is **not** a column: each (SKU, warehouse) in the source has
exactly one supplier all year, and procurement requests carry no warehouse, so
candidate suppliers are considered at SKU level. Assumption: a supplier that
delivers an SKU to one warehouse could deliver it to any warehouse.

## `available_qty`
- **Why simulated:** the source has no supplier-side stock. `Inventory_Level` is the
  buyer's own warehouse stock (it satisfies `inv[t] = inv[t-1] − Units_Sold[t] + Order_Quantity[t]`
  on 100% of rows), not what a supplier can ship.
- **Basis:** the 5,027 real orders (`Order_Quantity > 0`). Every pair has 18–62 orders
  (minimum required: 10). Sizes range 200–499 (median 351).
- **Method:** for each pair, one value drawn uniformly at random (with replacement)
  from that pair's own historical order sizes — so it is always a size this supplier
  actually delivered for this SKU.
- **Limitation:** the pairs' order-size distributions are statistically
  indistinguishable (Kruskal-Wallis p = 0.87; sizes are uniform 200–499). Differences
  between suppliers therefore come from the draw, not from real supplier behaviour.

## `defect_rate`
- **Why simulated:** the KPI dataset's suppliers (Alpha_Inc, Beta_Supplies, Delta_Logistics,
  Epsilon_Group, Gamma_Co) share no key, SKU, category or date range with SUP_1–SUP_10.
  The previous round-robin mapping between them had no evidence and was removed.
- **Calculation:** `Defective_Units / Quantity` per PO; 136 POs with blank
  `Defective_Units` are excluded (unknown ≠ zero), leaving 641. Supplier rate =
  Σ defective / Σ quantity: Alpha 2.25%, Epsilon 2.97%, Gamma 4.96%, Beta 9.35%, Delta 14.43%.
  Defect rate is a supplier trait (Kruskal-Wallis by supplier p ≈ 10⁻¹¹⁸; nearly constant across categories).
- **Distribution:** Beta(α = 1.5927, β = 21.8594), method-of-moments fit to those five
  rates (mean 0.0679, sd 0.0509). One draw per simulated supplier, rounded to 4 dp.
- **Cap:** the real maximum is 0.14429860; the cap is that value floored to 4 dp,
  **0.1442** (rounding up would exceed the real maximum). Applied to SUP_3 (0.1634 → 0.1442).
- **Limitation:** the fit rests on only five real suppliers.

## `fulfillment_rate` (not an on-time rate)
- **Why not on-time:** the KPI dataset has no promised/expected delivery date, so
  lateness cannot be measured. Its delivery dates are also unreliable (cancelled and
  pending POs carry delivery dates; one delivery precedes its order).
- **Definition:** `Delivered / (Delivered + Partially Delivered + Cancelled)`; Pending is
  excluded because its outcome is unknown. Real pooled rate **0.804598** (supplier range 0.779–0.823).
- **Method:** real suppliers do not differ (chi-square p = 0.91), so every simulated
  supplier shares the same true rate p = 0.804598 and its observed rate is
  `Binomial(137, p) / 137` (137 = median finished POs per real supplier).
- **Limitation:** differences between simulated suppliers are sampling noise by design;
  the simulated spread (0.75–0.87) is wider than the real one (0.78–0.82).

## Reproduce / verify
```bash
python -m backend.simulation.supplier_sku_profile --check   # regenerate in memory, compare with the committed CSV
python -m backend.simulation.supplier_sku_profile           # rewrite the files
```
Replacing this table with real ERP / supplier-performance data needs no code change:
supply a table with the same five columns (the loader rejects any pair that does not
exist in the supply-chain data).
