# Deletion log

**No raw source records were deleted.** All 6 raw CSVs have identical SHA-256 before and after (verified); `supply_chain` keeps all 91,250 rows and 15 columns.

Every deletion below is of a derived database artefact, code, or behaviour.

## 1. SQLite table supplier_profiles
- **Count:** 10 rows × 13 columns
- **Why:** Built on an unsupported mapping and a mislabelled stock field
- **Evidence:** No shared key between KPI vendors and SUP_x; Inventory_Level is buyer stock
- **Future impact:** None: every real input remains in supply_chain
- **Preserved in backup:** Yes: data/backup/pre_migration_2026-09-27/insightiq.db and full contents in state_before.json
- **Replaced by:** sku_supplier_lanes (real) + supplier_sku_profile (simulated)

## 2. Round-robin supplier mapping
- **Count:** 10 assignments: SUP_1→Alpha_Inc, SUP_2→Beta_Supplies, SUP_3→Delta_Logistics, SUP_4→Epsilon_Group, SUP_5→Gamma_Co, SUP_6→Alpha_Inc, SUP_7→Beta_Supplies, SUP_8→Delta_Logistics, SUP_9→Epsilon_Group, SUP_10→Gamma_Co
- **Why:** Relationship with no evidence
- **Evidence:** Assigned by sorted position only
- **Future impact:** None
- **Preserved in backup:** Yes (state_before.json)
- **Replaced by:** Explicitly simulated supplier-level quality

## 3. Derived products columns target_stock, deficit, urgent
- **Count:** 1000 rows × 3 columns
- **Why:** Only used for product-based procurement
- **Evidence:** 53 urgent flags, 26 on discontinued products
- **Future impact:** None: recomputable from raw product.csv
- **Preserved in backup:** Yes (backup DB); raw CSV untouched
- **Replaced by:** SKU urgency from supply_chain

## 4. Index idx_products_urgent
- **Count:** 1 index
- **Why:** Indexed a removed column
- **Evidence:** -
- **Future impact:** None
- **Preserved in backup:** Yes (backup DB)
- **Replaced by:** -

## 5. Products excluded from procurement (FILTERED, not deleted)
- **Count:** 1000 rows
- **Why:** No product→SKU/supplier relationship
- **Evidence:** SupplierID == ProductID on every row
- **Future impact:** None: table retained for reference
- **Preserved in backup:** Rows are still in the database and raw CSV
- **Replaced by:** SKU-based requests

## 6. Calendar duplicate dates (pre-existing, unchanged)
- **Count:** 65 rows
- **Why:** Loader keeps one row per date
- **Evidence:** Duplicate parsed dates
- **Future impact:** None
- **Preserved in backup:** Raw CSV untouched
- **Replaced by:** -

## 7. Config constants
- **Count:** 3 constants
- **Why:** No remaining consumers
- **Evidence:** -
- **Future impact:** None
- **Preserved in backup:** git history
- **Replaced by:** See change log

## 8. Code: perception demand_signals (orders + customers aggregation)
- **Count:** 1 function
- **Why:** E-commerce orders have no product/SKU reference; unused by the UI
- **Evidence:** order.csv has no product column
- **Future impact:** None: orders/customers tables retained
- **Preserved in backup:** git history
- **Replaced by:** Real Units_Sold demand per SKU

## 9. Behaviour: split-order fallback that ranked the full pool when nobody was eligible
- **Count:** 1 code path
- **Why:** It recommended a supplier that could not fulfil the order
- **Evidence:** Required by the 'no false selection' requirement
- **Future impact:** Split-order optimisation can be added later; combined availability is reported
- **Preserved in backup:** git history
- **Replaced by:** no_eligible_supplier result, no recommendation, no plan
- **Later change (2026-10-19):** a *safe* split order was added (Knowledge Agent Rule 6). It only runs when no single supplier can cover the request but the pool's combined `available_qty` can; each supplier is asked for at most its own `available_qty`, safe suppliers are used first, High Risk suppliers only to finish the order, and the plan's total always equals the required quantity. When even the combined stock is short, the result is still `no_eligible_supplier` with no plan. The old behaviour (ranking suppliers that could not fulfil the order) was not restored.

## 10. API GET /api/products
- **Count:** 1 endpoint
- **Why:** Product table no longer drives procurement
- **Evidence:** -
- **Future impact:** None
- **Preserved in backup:** git history
- **Replaced by:** GET /api/skus (old path returns 410)

## 11. Simulated supplier-SKU profile (2026-10-19)
- **Count:** 1 generator (`backend/simulation/`), 4 files in `DataSets/simulated/`, 14 tests
- **Why:** Its `available_qty`, `defect_rate` and `fulfillment_rate` were random draws
- **Evidence:** Replaced by values learned or measured from real data (docs/PREDICTIVE_MODEL.md)
- **Future impact:** None. Real supplier records can be added in `DataSets/supplier_evidence/`
- **Preserved in backup:** git history (commit e74f7a3)
- **Replaced by:** `backend/ml/supplier_estimator.py` (proven capacity + empirical-Bayes quality) and `backend/ml/demand_forecaster.py`
