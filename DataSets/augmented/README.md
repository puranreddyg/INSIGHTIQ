# Purchase-order outcomes (added 2026-10-20)

**Why:** the original ledger has 5,027 replenishments, but every one arrives on the
promised day, in full, with no defects. Supplier quality and reliability therefore
cannot be learned from it.

**What was added:** `purchase_order_outcomes.csv`, one row per existing replenishment
(5,027 rows). No original file or row was changed; originals are backed up in
`DataSets/backup/original_2026-10-20/` with a SHA-256 manifest.

| Column | Meaning |
|---|---|
| received_date, sku_id, warehouse_id, supplier_id | the ledger row this outcome belongs to (stock rose on this day) |
| accepted_qty | = the ledger's Order_Quantity (units that entered stock) |
| rejected_qty | units rejected at inspection (received but never stocked) |
| short_qty | units the supplier failed to ship |
| received_qty, ordered_qty | accepted + rejected; received + short |
| promised_lead_time_days | = the lane's real Supplier_Lead_Time_Days |
| delay_days, actual_lead_time_days | days late; promised + delay |
| order_date, promised_date | received_date − actual lead time; order_date + promised lead time |

**How it was made:** `python -m backend.augmentation.purchase_order_outcomes` (seed 20261020;
`--check` verifies the committed file). Each supplier has a hidden performance profile,
spread over real ranges: defect rate 2.4–14.6% (real KPI vendors), on-time 60–97%
(KPI: 80.5% fully delivered), short shipments 1–10% (AdventureWorks: 4.3% of lines).

**Honest label:** semi-synthetic supplier outcomes calibrated to real procurement data.
`supplier_profiles_HIDDEN.json` is the answer key. The application never reads it
(a test enforces this); tests use it to confirm the model learns the true profiles.
