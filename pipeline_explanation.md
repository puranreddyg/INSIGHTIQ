# InsightIQ Agent Pipeline & Heuristic Search Explanation

This document explains the four-stage AI agent pipeline in InsightIQ, detailing the mathematical formulas and logical rules used at each step—especially the heuristic search and forward chaining trace.

## 1. The Agentic Pipeline Overview

InsightIQ operates using a sequential multi-agent pipeline orchestrated by the **Coordinator Agent**. Information flows from one agent to the next:

1. **Perception Agent**: Scans the database to observe inventory states and detect "Urgent" items.
2. **Supplier Intelligence Agent**: Uses a **heuristic search** to evaluate, rank, and score all eligible suppliers based on the required inventory deficit.
3. **Knowledge Agent**: Uses **forward chaining** (an Explainable AI technique) to apply logical rules to the facts gathered by the previous agents, determining supplier risk and status.
4. **Planning Agent**: Computes the final operational strategy, including order quantities, expected delivery dates (adjusting for weekends/seasonality), and designates backup suppliers.

---

## 2. Heuristic Search (Supplier Intelligence Agent)

The Supplier Agent's goal is to rank a pool of suppliers and find the best one for an order. It doesn't just look for the cheapest option; it balances Cost, Quality (Rating), Speed (Delivery Time), and Reliability.

### Step 1: Eligibility Filtering
Candidates are the suppliers that actually serve the SKU in the supply-chain data. A supplier is eligible as a **sole source** when its (simulated) `available_qty` ≥ `RequiredQuantity`. Whenever a quantity is required, every supplier with stock is also scored as an `allocation_pool`. If no acceptable (non-High-Risk) sole source exists, the Planning Agent splits the order across that pool. See `docs/IMPROVEMENTS.md`.

### Step 2: Normalization
Cost and lead time are Min-Max normalized over the suppliers in contention, direction-corrected so that 1 = best. When all values are equal, every supplier scores 1.

$$ \text{CostScore} = 1 - \frac{x - \min(x)}{\max(x) - \min(x)} $$

### Step 3: Heuristic Formula
Bounded rates (defect rate and fulfillment rate) are used absolutely, not Min-Max normalized.

$$ \text{Raw Score} = w_1 \cdot (1 - \text{DefectRate}) + w_2 \cdot \text{CostScore} + w_3 \cdot \text{DeliveryScore} + w_4 \cdot \text{FulfillmentRate} $$

**Weights Used:**
*   $w_1$ (Quality) = 0.30
*   $w_2$ (Cost) = 0.25
*   $w_3$ (Delivery Time) = 0.20
*   $w_4$ (Fulfillment) = 0.25

### Step 4: Final Rescaling
The `Raw Score` is then normalized one last time and multiplied by 100 to give an intuitive `0-100` score shown in the UI:
$$ \text{Final Score} = N(\text{Raw Score}) \times 100 $$

---

## 3. Forward Chaining Trace (Knowledge Agent)

**Forward Chaining** is an Explainable AI (XAI) technique where the system starts with known **facts**, applies **IF-THEN rules**, and iterates to extract new facts until no more rules match (a "fixed point"). This makes the AI's decision-making 100% auditable.

The trace logs three types of steps:
1. **Fact Detected**: The raw data observed.
2. **Rule Evaluated**: The IF-THEN condition tested.
3. **Decision Made**: The final deduced state.

### The Rules Engine Formulas

**Rule 1: Inventory State Evaluation**
> **IF** `Inventory` < `ReorderPoint`  
> **THEN** State = `"Urgent Procurement"`  
> **ELSE** State = `"Monitoring"`

**Rule 5: Capacity Check**
> **IF** `AvailableQty` $\ge$ `RequiredQty` **THEN** the supplier can fulfil the request alone  
> **ELSE** `"Ineligible"`, or `"Partial Capacity"` in split-order mode, where the chain continues so Rules 2–4 still run

**Rule 2: Preferred Supplier Check**
> **IF** `LeadTime` $\le 3$ days **AND** `DefectRate` $\le 0.05$  
> **THEN** Supplier_Status = `"Preferred"`

**Rule 3: Risk Assessment (High Priority)**
> **IF** `DefectRate` $> 0.05$ (5%)  
> **THEN** Supplier_Status = `"High Risk"`

**Rule 4: Standard/Reliable Fallback**
> **IF** (No prior status assigned) **AND** `FulfillmentRate` $\ge 0.90$  
> **THEN** Supplier_Status = `"Reliable"`  
> **ELSE** Supplier_Status = `"Standard"`

*Note: The `High Risk` flag always takes precedence. The system recommends the highest-ranked sole source that does NOT trigger Rule 3. If there is none, it plans a split order across non-High-Risk suppliers. High Risk suppliers are never allocated automatically.*

---

## 4. Other Pipeline Formulas

### Perception Agent (Required Quantity)
For each urgent lane (`Inventory < ReorderPoint`):
*   **Horizon** $H = L + R$: lead time plus review period (the median gap between the lane's historical orders)
*   **Target Stock** = forecast demand over $H$ + safety stock, where $SS = z_{95\%}\,\sigma_{daily}\sqrt{H}$
*   **Lane Quantity** = $\max(0, \text{Target} - \text{Inventory})$
*   **RequiredQuantity** = $\lceil \sum \text{Lane Quantity} / 10 \rceil \times 10$

The demand forecast is whichever of the ridge model, the 28-day moving average or weekly seasonal-naive has the lowest held-out 21-day WAPE (currently seasonal-naive). The legacy median-lot rule remains available as a fallback. Details and validation are in `docs/IMPROVEMENTS.md`.

### Planning Agent (Logistics & Costs)
*   **Order Quantity**: RequiredQuantity, from one supplier or split across several (`allocations[]`). A plan that can only partly cover it is labelled `planned_partial` and shows the residual shortage.
*   **Daily Demand Estimate**: mean real Units_Sold over the last 28 days of the focus lanes.
*   **Delivery Scheduling** (per allocated supplier): `Today` + `LeadTime` in working days, skipping weekends, plus a 0–2 day seasonality buffer. The plan's arrival date is the latest supplier's.
*   **Total Expense**:
    $$ \text{Total Expense} = \sum_{\text{lines}} \text{AllocatedQty} \times \text{UnitCost} $$
