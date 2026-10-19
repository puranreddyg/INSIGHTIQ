# Supplier evidence (optional)

The supply-chain data has no defect or delivery records for SUP_1 … SUP_10, so every
supplier starts on the real industry baseline from the procurement KPI file and shows
as **Unrated**. Put real records here and the estimates update on the next start, with
no code change (`backend/ml/supplier_estimator.py`).

## supplier_outcomes.csv: real purchase-order outcomes per supplier

```
supplier_id,quantity,defective_units,order_status
SUP_3,500,12,Delivered
SUP_3,400,,Partially Delivered
```

- `supplier_id` must be a supplier in the supply-chain data.
- `defective_units` may be blank (unknown, not zero).
- `order_status` is one of Delivered, Partially Delivered, Cancelled, Pending.

## vendor_links.json: only if you KNOW two names are the same company

```json
{"SUP_3": "Delta_Logistics"}
```

This uses that KPI vendor's real purchase orders as SUP_3's evidence. Each vendor can
be linked to at most one supplier.

A supplier needs at least 5 purchase orders of its own before the Knowledge Agent gives
it a quality label (Preferred, High Risk, Reliable, Standard).
