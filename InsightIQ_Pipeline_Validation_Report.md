# InsightIQ — Four-Agent Procurement Pipeline Validation Report

## A. Agent Architecture
The system architecture correctly implements the four-stage sequential pipeline:
1. **Perception Agent** (Observer)
2. **Supplier Intelligence Agent** (Evaluator & Heuristic Search)
3. **Knowledge Agent** (Risk Assessor & Forward Chaining)
4. **Planning Agent** (Strategist)

The workflow executes successfully from end-to-end, passing `sku_id` correctly between each agent.

## B. Exact Responsibility of Each Agent
*   **Perception Agent**: Examines the current inventory state by comparing `Inventory_Level` to `Reorder_Point`. It calculates the `RequiredQuantity` (deficit) per SKU based on historical lot sizes of urgent lanes.
*   **Supplier Intelligence Agent**: Finds genuine suppliers for the specified SKU, applies an availability filter (`available_qty >= RequiredQuantity`), and ranks eligible suppliers using a Heuristic Search algorithm.
*   **Knowledge Agent**: Runs Forward Chaining IF-THEN rules on the ranked candidates to assess risk, assigning a `Supplier_Status` (`Preferred`, `High Risk`, `Reliable`, or `Standard`).
*   **Planning Agent**: Takes the recommended (safe) supplier and generates an operational plan, calculating total expense, projecting an estimated delivery date (adjusting for weekends and seasonal friction), and designating a backup supplier.

## C. Exact Input/Output of Each Agent
*   **Perception In/Out**: Reads `supply_chain` ledger. Outputs SKU focus facts, deficit, and urgency (`RequiredQuantity`).
*   **Supplier In/Out**: Receives `RequiredQuantity` and `sku_id`. Reads `sku_supplier_lanes` and `supplier_sku_profile`. Outputs eligible/ineligible suppliers and a ranked candidate matrix.
*   **Knowledge In/Out**: Receives ranked candidates and perception facts. Outputs annotated candidates with assigned risk statuses and a reasoning trace.
*   **Planning In/Out**: Receives annotated candidates, `RequiredQuantity`, and perception facts. Outputs a final `status: planned` with order quantity, expense, and estimated arrival.

## D. Data Lineage & Provenance
The pipeline correctly utilizes both the real historical data and the simulated supplier profile extension. Provenance is strictly maintained in the code (`FIELD_PROVENANCE`).

| Field | Source | Real/Simulated | Agent Using It |
| :--- | :--- | :--- | :--- |
| `sku_id` | `supply_chain` | Real | Perception, Supplier, Knowledge, Planning |
| `supplier_id` | `sku_supplier_lanes` | Real | Supplier Intelligence |
| `available_qty` | `supplier_sku_profile` | Simulated | Supplier (Filter), Knowledge (R5) |
| `defect_rate` | `supplier_sku_profile` | Simulated | Supplier (Quality Score), Knowledge (R2, R3) |
| `fulfillment_rate` | `supplier_sku_profile` | Simulated | Supplier (Fulfill Score), Knowledge (R4) |
| `unit_cost` | `sku_supplier_lanes` | Real | Supplier (Cost Score), Planning (Expense) |
| `lead_time_days` | `sku_supplier_lanes` | Real | Supplier (Delivery Score), Knowledge (R2), Planning |
| `inventory_level` | `supply_chain` | Real | Perception |
| `units_sold` | `supply_chain` | Real | Perception (Demand Logic), Planning |
| `reorder_point` | `supply_chain` | Real | Perception |

## E. Heuristic Search Methodology
The Heuristic Search inside the **Supplier Intelligence Agent** evaluates multiple criteria using a weighted formula. 
It uses Min-Max normalization for metrics where lower is better (Cost, Lead Time), and uses absolute bounds for bounded rates (Defect, Fulfillment).

**Current Validated Configuration:**
*   `QualityScore` = `1.0 - defect_rate` (Weight: **0.30**)
*   `CostScore` = `1.0 - MinMax(unit_cost)` (Weight: **0.25**)
*   `FulfillmentScore` = `fulfillment_rate` (Weight: **0.25**)
*   `DeliveryScore` = `1.0 - MinMax(lead_time_days)` (Weight: **0.20**)

**Formula:**
1.  `RawScore` = $\sum (\text{Component} \times \text{Weight})$
2.  `FinalScore` = `MinMax(RawScore) * 100`

*Only suppliers with `available_qty >= RequiredQuantity` are scored.*

## F. Knowledge Agent Rules
The **Knowledge Agent** runs exact Forward Chaining IF-THEN rules on the absolute values of the candidates:
*   **Rule 1 (Urgency):** IF `Inventory` < `ReorderPoint` THEN `Urgent Procurement` ELSE `Monitoring`
*   **Rule 5 (Capacity):** IF `AvailableQty` >= `RequiredQty` THEN ok ELSE `Ineligible`
*   **Rule 2 (Preferred):** IF `LeadTime` <= 3d AND `DefectRate` <= 0.05 THEN `Preferred`
*   **Rule 3 (High Risk):** IF `DefectRate` > 0.05 THEN `High Risk`
*   **Rule 4 (Reliable):** IF no prior status AND `FulfillmentRate` >= 0.90 THEN `Reliable` ELSE `Standard`

## G. Planning Methodology
*   **Order Quantity:** Replicates historical median replenishment policies of urgent lanes, rounded up to the nearest multiple of 10.
*   **Total Expense:** `Order Quantity` $\times$ `unit_cost`.
*   **Delivery Estimate:** Starts with `Today` + `LeadTime` (working days only), and adds a `seasonality_delay_days` buffer (0-2 days) derived from historical weekend density in the dataset. This is explicitly labeled as an *estimated_arrival_date*, not a guaranteed historical delivery.
*   **Backup Supplier:** Selected from the remaining SKU-specific eligible suppliers, actively avoiding `High Risk` flagged suppliers.

## H. 10-SKU End-to-End Tests
Real-world pipeline executions were tested for 10 urgent SKUs (e.g., `SKU_44`, `SKU_19`, `SKU_37`, `SKU_2`, `SKU_47`, `SKU_42`, `SKU_41`, `SKU_39`, `SKU_50`, `SKU_21`).
*   Each SKU returned a highly specific, restricted pool of suppliers.
*   Heuristic scoring and risk assignment executed perfectly on all candidates.
*   Planning generated full procurement curves and estimated delivery dates.

## I. Procurement-Required Tests
For the urgent SKUs (like `SKU_44`), the Perception Agent calculated a deficit, flagged the state as "Urgent Procurement", and successfully triggered the downstream agents (Supplier $\rightarrow$ Knowledge $\rightarrow$ Planning), outputting a final `status: planned`.

## J. No-Procurement Tests
For non-urgent SKUs (e.g., `SKU_1`, `SKU_3`), the Perception Agent recognized that `Inventory_Level` was safely above the `Reorder_Point`. The pipeline gracefully halted at the Planning Agent with `status: no_action` and `error: No procurement required`, preventing unnecessary logic execution.

## K. SKU-Isolation Tests
A strict isolation test was performed:
1. Run pipeline for `SKU_44`
2. Run pipeline for `SKU_19`
3. Run pipeline for `SKU_44` again

The output payloads for execution 1 and 3 were strictly evaluated and confirmed to be **100% identical** (ignoring timestamps). No supplier lists, availability metrics, or risk statuses leaked between executions.

## L. Updated-Data-Runtime Verification
**Is the running four-agent application actually using the updated SKU-based architecture and supplier_sku_profile.csv?**
**Yes.** Runtime execution proves that `available_qty`, `defect_rate`, and `fulfillment_rate` are pulled directly from `supplier_sku_profile.csv` via the database context. Furthermore, `ProductID` is entirely retired from the agent engine. The `Supplier Intelligence Agent` accurately filters ineligible suppliers dynamically using the simulated `available_qty` specific to the requested `sku_id`.

## M. Legacy-Code Audit
A deep code search was performed across the backend for legacy terms: `ProductID`, `product_id`, `products`, `supplier_profiles`, `Alpha_Inc`, `Beta_Supplies`.
*   **ACTIVE IN AGENTS:** None.
*   **LEGACY / SAFE:** All occurrences are isolated to deprecated database schemas, API compatibility layers explicitly throwing `410 Gone` HTTP errors, or assertions within the test suite ensuring they do not trigger procurement. No obsolete mapping influences current procurement.

## N. Automated Test Results
*   **Status:** PASSED
*   **Execution:** `pytest backend/tests/` ran successfully.
*   **Result:** 82 tests passed, confirming Perception logic, Supplier heuristic calculations, Forward Chaining integrity, and SKU isolation.

## O. Problems Found/Fixed
*   **None.** The existing architecture was rigorously evaluated and was found to be operating exactly as designed. The codebase required no alterations to fulfill the validation criteria.

## O-2. Update (2026-10-23): split orders, forecast quantity, demand model
Sections B, E, F and G above describe the earlier behaviour. Since then:
*   **Perception:** RequiredQuantity uses a forecast-based order-up-to rule over lead time + review period, plus safety stock. The median-lot rule is a configurable fallback.
*   **Supplier / Knowledge / Planning:** when no acceptable sole source exists, the order is split across non-High-Risk suppliers, or marked partial with the residual shortage.
*   **Approval:** validates and saves split orders in a single transaction.
*   **Demand model:** a ridge model was validated chronologically. It did **not** beat the seasonal-naive baseline, so it is not used.
*   **Tests:** `pytest backend/tests/` → **164 passed**.

Details and limitations: `docs/IMPROVEMENTS.md`.

## P. Final Verdict
**PASS**
All final acceptance criteria have been fully verified. The InsightIQ Four-Agent Pipeline correctly executes a pure SKU-based procurement workflow using both real supply-chain facts and the simulated supplier-profile extensions.
