> **Outdated.** This describes an older version of the pipeline. The current design, including the AI demand forecast and multi-supplier orders, is in [docs/PREDICTIVE_MODEL.md](docs/PREDICTIVE_MODEL.md).

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
It first filters out any suppliers whose current `stock_availability` cannot cover the required `Deficit`. If no single supplier can cover the entire deficit, it falls back to evaluating the entire pool for a potential split order.

### Step 2: Normalization
To compare different metrics fairly, it normalizes them using **Min-Max Normalization** so values fall between `0.0` and `1.0`. A small floor guard ($\epsilon = 0.15$) is added to cost and delivery time so that dividing by zero is avoided for the cheapest/fastest suppliers.

$$ N(x) = \frac{x - \min(x)}{\max(x) - \min(x)} $$

### Step 3: Heuristic Formula
The agent calculates a raw heuristic score using predefined weights. Cost and Delivery Time use inverse relationships ($\frac{1}{x}$) because *lower* costs and *lower* delivery times are better.

$$ \text{Raw Score} = w_1 \cdot \left(\frac{\text{Rating}}{5.0}\right) + w_2 \cdot \left(\frac{1}{N(\text{Cost}) + 0.15}\right) + w_3 \cdot \left(\frac{1}{N(\text{Delivery}) + 0.15}\right) + w_4 \cdot \text{Reliability} $$

**Weights Used:**
*   $w_1$ (Rating) = 0.30
*   $w_2$ (Cost) = 0.25
*   $w_3$ (Delivery Time) = 0.20
*   $w_4$ (Reliability) = 0.25

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

**Rule 2: Preferred Supplier Check**
> **IF** `Rating` $\ge 4.5$ **AND** `DeliveryTime` $\le 3$ days  
> **THEN** Supplier_Status = `"Preferred"`

**Rule 3: Risk Assessment (High Priority)**
> **IF** `DefectRate` $> 0.05$ (5%)  
> **THEN** Supplier_Status = `"High Risk"`

**Rule 4: Standard/Reliable Fallback**
> **IF** (No prior status assigned) **AND** `Reliability` $\ge 0.90$  
> **THEN** Supplier_Status = `"Reliable"`  
> **ELSE** Supplier_Status = `"Standard"`

*Note: The `High Risk` flag always takes precedence. The system will recommend the highest-ranked supplier that does NOT trigger the High Risk rule.*

---

## 4. Other Pipeline Formulas

### Perception Agent (Deficit Calculation)
*   **Target Stock** = `ReorderPoint` $\times 3$
*   **Deficit** = `Target Stock` - `Inventory`

### Planning Agent (Logistics & Costs)
*   **Order Quantity**: The deficit rounded up to the nearest multiple of 10.
    $$ \text{OrderQty} = \lceil \frac{\max(\text{Deficit}, 10)}{10} \rceil \times 10 $$
*   **Daily Demand Estimate**:
    $$ \text{DailyDemand} = \max\left(1, \text{round}\left(\frac{\text{ReorderPoint}}{14}\right)\right) $$
*   **Delivery Scheduling**: 
    Starts with a Naive Arrival (`Today` + `LeadTime` in working days). It skips weekends (Saturdays and Sundays). Then, it adds 0-2 extra buffer days depending on **seasonality** (e.g. Winter months might add a delay due to historical shipping friction).
*   **Total Expense**:
    $$ \text{Total Expense} = \text{OrderQty} \times \text{UnitCost} $$
