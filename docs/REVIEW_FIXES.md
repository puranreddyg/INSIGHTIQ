# Code review fixes (2026-10-23)

A full read-through of the multi-agent pipeline, the ML layer, the API and the frontend,
followed by an all-50-SKU run. **No ML code was changed**: the demand forecaster, the
supplier performance model, the hybrid search and the Forecast Evaluation page are as before.
The all-SKU run found no crashes and no quantity or capacity violation (a plan never exceeds
a supplier's capacity or the requirement).

## Fixed

| # | Problem | Fix |
|---|---|---|
| 1 | **Approval trusted the browser's unit cost**, so a purchase order could be saved at any price, and the saved total differed from the plan's by rounding. | The server derives unit cost from the ledger (same rounding as the agents) and rejects a client cost that disagrees. PO totals now equal the plan's total. |
| 2 | **An order could be approved for a SKU with no requirement**, or for more than the requirement. | The requirement is recomputed server-side. Total above it, or any order when it is 0, is rejected. |
| 3 | **A partial order was not explicit.** A total below the requirement was accepted silently. | It must be sent with `is_partial: true`. It is stored in `purchase_orders.is_partial` and shown in the history. |
| 4 | **High Risk suppliers (Rule 3) could be ordered with one click**, including inside split plans and flagged single plans. | Approval is rejected unless `acknowledge_risk` is set. The Action Hub shows a "needs your decision" notice and a confirmation naming the suppliers. The optimiser is unchanged: it still prefers plans with the fewest High Risk suppliers. |
| 5 | **Approval was not a single transaction** and was not serialised against concurrent approvals. | `BEGIN IMMEDIATE`, validate, insert all lines, `COMMIT`; any failure rolls everything back (tested with a SQLite trigger). Lines of one approval share an `order_group`. |
| 6 | **Supplier Matrix never showed the split plan.** It said "run the Agent Pipeline". | `/api/suppliers` attaches `allocation_preview` (quantities, cost, per-supplier arrival dates, shortfall). The matrix renders it. |
| 7 | **Knowledge decision said `single` for a SKU that needs nothing.** | It is now `no_action` when the requirement is 0 (the informational supplier pick is unchanged). The no-action text mentions the forecast reorder point. |
| 8 | **The supplier model's inputs were presented as plain "learned".** The purchase-order outcomes are semi-synthetic (`DataSets/augmented/README.md`). | UI labels and tooltips now say "semi-synthetic purchase-order outcomes". |
| 9 | A tooltip printed `${rec.promised_lead_time_days}` literally (a plain string inside a template literal). | Fixed. |
| 10 | The replenishment chart only marked the last delivery of a split order. | Every delivery date is marked. |
| 11 | `purchase_orders` columns added by a newer release were only created on a full rebuild. | Checked on every start (`_ensure_purchase_orders`). |
| 12 | Windows `core.autocrlf` rewrote the checksummed data files, so `test_original_files_are_backed_up_and_unchanged` failed on Windows. | `.gitattributes` marks `DataSets/**` and `*.sha256` as `-text`. |
| 13 | `/api/dashboard` could divide by zero if the baseline error were 0. | Guarded. |

## Tests

`pytest backend/tests` : 118 passed (105 before; 13 new in `test_approval.py`).
Two existing approval tests were rewritten, not removed: they sent an arbitrary unit cost
and ordered a SKU with no requirement, which the old endpoint wrongly accepted.

## Known limitations (unchanged)

* Supplier capacity is **estimated** from the largest historical order, not live stock, and
  approving an order does not reduce it, so the same plan can be approved twice.
* The supplier quality inputs are semi-synthetic; only unit cost, lead time and sales are real.
* The yearly demand cycle is learned from one year of data.
