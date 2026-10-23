# InsightIQ — ProductID → SKU_ID Migration Plan

Written **before** any data or code was modified (2026-09-27). The final PDF report
describes what was actually done; this file records what was planned.

## Goal
Make procurement SKU-based using `supply_chain_dataset1.csv` as the source of truth,
remove the unsupported supplier-quality mapping, and fill the two genuine data gaps
(supplier availability, supplier quality) with a clearly labelled, reproducible
simulated table. No raw file is modified.

## Data layout
| Location | Role | Tracked in git |
|---|---|---|
| `DataSets/` (existing) | Raw source CSVs — **read-only, never modified** | yes |
| `DataSets/CHECKSUMS.sha256` | SHA-256 of every raw file, used to prove they are unchanged | yes |
| `DataSets/simulated/` | Simulated supplier-SKU profile + metadata + data dictionary | yes |
| `data/backup/pre_migration_2026-09-27/` | Byte copy of all raw CSVs + the pre-migration SQLite DB | no (`data/` is ignored; git history also holds the raw CSVs) |
| `docs/migration/` | Plan, before/after state, change log, deletion log, validation, PDF | yes |

## Files
**Added**
- `DataSets/CHECKSUMS.sha256`
- `DataSets/simulated/supplier_sku_profile.csv`, `supplier_sku_profile.meta.json`, `README.md`
- `backend/simulation/__init__.py`, `backend/simulation/supplier_sku_profile.py` (generator)
- `scripts/migration/capture_state.py`, `change_log.py`, `validate.py`, `build_report.py`
- `backend/tests/test_simulation.py`, `backend/tests/test_sku_pipeline.py`
- `docs/migration/*` (this plan, state snapshots, logs, PDF)

**Modified**
- `backend/config.py`, `backend/database.py`, `backend/main.py`
- `backend/agents/perception_agent.py`, `supplier_agent.py`, `knowledge_agent.py`, `planning_agent.py`, `coordinator_agent.py`
- `backend/tests/test_supplier_agent.py` (rewritten against the new schema)
- `frontend/static/js/app.js`, `frontend/static/js/api.js`, `frontend/static/css/styles.css`
- `requirements.txt` (+ `httpx`, needed by FastAPI's TestClient)

**Deleted files:** none.

## Database (SQLite, rebuilt from CSVs; schema version 1 → 2 via `PRAGMA user_version`)
| Table | Action | Detail |
|---|---|---|
| `supply_chain` | kept, all 91,250 rows / 15 columns | + indexes on `(sku_id, date)` and `(sku_id, supplier_id)` |
| `sku_supplier_lanes` | **added** (real, derived) | one row per (SKU, warehouse): supplier, unit cost, lead time — lossless, constant per lane |
| `supplier_sku_profile` | **added** (simulated) | sku_id, supplier_id, available_qty, defect_rate, fulfillment_rate |
| `supplier_profiles` | **dropped** | built on the unsupported round-robin Alpha_Inc…→SUP_x mapping |
| `products` | modified | derived columns `target_stock`, `deficit`, `urgent` removed (only used for product-based procurement); raw columns kept |
| `purchase_orders` | modified | + `sku_id` column (ALTER, existing rows kept; 0 rows at migration time) |
| `orders`, `customers`, `calendar`, `procurement_kpi` | unchanged | kept; orders/customers no longer read by agents |
| index `idx_products_urgent` | dropped | column it indexed is removed |

## APIs
- `POST /api/run-agent-pipeline?product_id=` → `?sku_id=` (product_id now returns 410 Gone with explanation)
- `GET /api/products` → replaced by `GET /api/skus`
- `GET /api/suppliers?required_deficit=` → `?sku_id=&required_quantity=`
- `POST /api/approve-order`: `product_id` → `sku_id`; validates the (SKU, supplier) pair exists and quantity ≤ available_qty
- `GET /api/dashboard`, `/api/system-health`: SKU-based alerts and KPIs; provenance labels for simulated fields

## Frontend
Dashboard alerts/chart, pipeline SKU picker and perceived-state card, supplier matrix
(SKU-scoped, ineligible suppliers listed), action hub payload, "SIMULATED" labels.

## Rules (Knowledge Agent) — to be justified in the report
- R1 unchanged (Inventory < ReorderPoint), evaluated per urgent warehouse lane.
- R2 input `Rating` no longer exists (it came from the invalid mapping). Replace with
  already-existing thresholds only: LeadTime ≤ 3 d AND DefectRate ≤ 0.05.
- R3 unchanged (DefectRate > 0.05).
- R4 threshold 0.90 kept, now applied to fulfillment_rate — not lowered.
- R5 added: Capacity check (AvailableQty ≥ RequiredQty), so exclusions are auditable.

## Deletions (raw data)
None planned. Every deletion is to derived DB artefacts and is recorded in `DELETION_LOG.md`.
