# InsightIQ Heuristic Search Explanation

Here is a comprehensive, step-by-step breakdown of how the **Heuristic Search** works in the InsightIQ project. 

In your pipeline, the Heuristic Search is executed by the **Supplier Intelligence Agent**. Its goal is to rank a pool of suppliers to find the absolute best one for an order. Instead of just picking the cheapest option, it uses a math-based heuristic function to balance **Cost, Quality (Rating), Speed (Delivery Time), and Reliability**.

Here is exactly how it works, step-by-step, including all math formulas and an example.

---

## Step 1: Eligibility Filtering
Before doing any math, the agent filters the list of available suppliers based on raw constraints. 
* **The Rule:** The supplier's current `stock_availability` must be greater than or equal to the required `Deficit` (the amount of inventory you need to order).
* **Fallback:** If *no acceptable single supplier* can cover the requirement, every supplier with stock is scored as an `allocation_pool`. Acceptable (non-High-Risk) suppliers are then allocated in heuristic-rank order, each up to its availability. Any residual shortage is reported explicitly (see `docs/IMPROVEMENTS.md`).

## Step 2: Normalization
Because Cost (e.g., $10), Delivery Time (e.g., 5 days), and Ratings (e.g., 4.5 stars) are completely different units, the system must put them on an equal playing field. It does this using **Min-Max Normalization** so all values fall between `0.0` and `1.0`.

**The Formula:**
`N(x) = (x - min(x)) / (max(x) - min(x))`
*(Where `x` is the value being normalized, and `min`/`max` are the lowest/highest values among all eligible suppliers).*

## Step 3: Heuristic Scoring Formula
Once the metrics are normalized, the agent calculates a `Raw Score` for each supplier. 

For Rating and Reliability, *higher is better*. However, for Cost and Delivery Time, *lower is better*. To mathematically account for this, the system uses inverse relationships (`1/x`). A floor guard (`0.15`) is added to Cost and Delivery so that dividing by zero is avoided for the cheapest or fastest suppliers.

**The Math Formula:**
`Raw Score = w1 * (Rating / 5.0) + w2 * (1 / (N(Cost) + 0.15)) + w3 * (1 / (N(Delivery) + 0.15)) + w4 * Reliability`

**The Weights (Priority Configuration):**
* `w1` (Rating) = 0.30
* `w2` (Cost) = 0.25
* `w3` (Delivery Time) = 0.20
* `w4` (Reliability) = 0.25

## Step 4: Final Rescaling
A raw score like `3.46` isn't very user-friendly. The agent normalizes the calculated Raw Scores one last time among the group and multiplies by 100 to yield a clean `0-100` score that can be displayed on your frontend UI.

**The Formula:**
`Final Score = ((Raw Score - min(Raw Score)) / (max(Raw Score) - min(Raw Score))) * 100`

---

## A Concrete Example Let's Walk Through It

Let's assume your **Deficit** is **50 units**, and you have 3 potential suppliers in your database.

| Supplier | Stock | Rating (out of 5) | Cost per unit | Delivery Time | Reliability (0-1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Supplier A** | 100 | 4.0 | $10 | 2 days | 0.90 |
| **Supplier B** | 60 | 4.8 | $15 | 4 days | 0.95 |
| **Supplier C** | 20 | 4.5 | $12 | 3 days | 0.85 |

### 1. Apply Filtering
**Supplier C** is immediately eliminated because their Stock (20) is less than the Deficit (50). We will only evaluate A and B.

### 2. Apply Normalization
* **Cost:** min is $10 (A), max is $15 (B)
  * `N(Cost_A)` = `(10 - 10) / (15 - 10)` = `0.0`
  * `N(Cost_B)` = `(15 - 10) / (15 - 10)` = `1.0`
* **Delivery:** min is 2 days (A), max is 4 days (B)
  * `N(Delivery_A)` = `(2 - 2) / (4 - 2)` = `0.0`
  * `N(Delivery_B)` = `(4 - 2) / (4 - 2)` = `1.0`

### 3. Calculate Raw Scores
**Supplier A:**
* Rating component: `0.30 * (4.0 / 5.0)` = `0.24`
* Cost component: `0.25 * (1 / (0.0 + 0.15))` = `0.25 * 6.66` = `1.66`
* Delivery component: `0.20 * (1 / (0.0 + 0.15))` = `0.20 * 6.66` = `1.33`
* Reliability component: `0.25 * 0.90` = `0.225`
* **Supplier A Raw Score:** `0.24 + 1.66 + 1.33 + 0.225` = `3.455`

**Supplier B:**
* Rating component: `0.30 * (4.8 / 5.0)` = `0.288`
* Cost component: `0.25 * (1 / (1.0 + 0.15))` = `0.25 * 0.87` = `0.217`
* Delivery component: `0.20 * (1 / (1.0 + 0.15))` = `0.20 * 0.87` = `0.174`
* Reliability component: `0.25 * 0.95` = `0.2375`
* **Supplier B Raw Score:** `0.288 + 0.217 + 0.174 + 0.2375` = `0.9165`

### 4. Final Rescaling
Now we scale the raw scores out of 100, where `3.455` is the highest score and `0.9165` is the lowest score.
* **Supplier A Final Score:** `100` (The clear winner, primarily due to being significantly faster and cheaper).
* **Supplier B Final Score:** `0` (The loser in this match-up, mainly penalized heavily by the inverse division of being the most expensive and slowest option).

In reality, with many suppliers, you'd get a nice distribution of scores (e.g., 100, 85, 72, 40) rather than just 100 and 0. The top-scoring supplier is then passed on to the **Knowledge Agent** to verify risk before finally being utilized!
