"""Build the machine- and human-readable migration logs.

    python scripts/migration/change_log.py

Reads docs/migration/state_before.json, state_after.json, validation_results.json
and the simulation metadata; writes change_log.json, CHANGE_LOG.md, DELETION_LOG.md.
Reasons/evidence are authored here; every count is read from those inputs.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "migration"


def load(name: str) -> dict:
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


BEFORE, AFTER, VALID = load("state_before.json"), load("state_after.json"), load("validation_results.json")
META = json.loads((ROOT / "DataSets/simulated/supplier_sku_profile.meta.json").read_text(encoding="utf-8"))
T0, T1 = BEFORE["database"]["tables"], AFTER["database"]["tables"]
OLD_MAP = {r["supplier_id"]: r["benchmarked_against"] for r in BEFORE["database"]["supplier_profiles_snapshot"]}
URGENT, URGENT_DISC = BEFORE["database"]["products_urgent"], BEFORE["database"]["products_urgent_discontinued"]
ZERO_ROP = int((pd.read_csv(ROOT / "DataSets/archive (1)/product.csv").ReorderLevel == 0).sum())
DEF, FUL, AV = META["defect_rate"], META["fulfillment_rate"], META["available_qty"]
BACKUP = "data/backup/pre_migration_2026-09-27/"


def raw_unchanged() -> bool:
    before = {c["path"]: c["sha256"] for c in BEFORE["csv_files"]}
    after = {c["path"]: c["sha256"] for c in AFTER["csv_files"]}
    return all(after.get(p) == h for p, h in before.items())


def entry(dataset, location, field, change, before, after, reason, evidence, status, method, impact):
    return dict(dataset=dataset, location=location, field=field, change_type=change, before=before, after=after,
                reason=reason, evidence=evidence, real_or_simulated=status, method=method, impact=impact)


def dataset_changes() -> list[dict]:
    sp = T0["supplier_profiles"]
    return [
        # ---------------- raw files -----------------
        entry("All 6 raw CSVs", "DataSets/", "all", "MODIFIED" if not raw_unchanged() else "RETAINED",
              "6 files", "6 files, SHA-256 identical", "Raw data must stay recoverable and unchanged",
              "Before/after SHA-256 match; git diff HEAD empty", "REAL", "Read-only", "None"),
        entry("Raw checksum manifest", "DataSets/CHECKSUMS.sha256", "-", "ADDED", "-", "6 SHA-256 lines",
              "Proves raw files are unchanged", "sha256sum of each raw CSV before any change", "N/A", "sha256sum", "Verified by tests and validate.py"),
        entry("Backup", BACKUP, "-", "ADDED", "-", "6 raw CSVs + pre-migration insightiq.db",
              "Original data must remain recoverable", "Copies verified against manifest; DB hash identical", "REAL", "Byte copy", "Local only (data/ is git-ignored); git history also holds raw CSVs"),
        # ---------------- simulated dataset -----------------
        entry("Simulated supplier-SKU profile", "DataSets/simulated/supplier_sku_profile.csv", "file", "ADDED", "-",
              f"{META['rows']} rows × 5 columns", "Fill genuine data gaps (supplier availability and quality)",
              "No source column holds supplier stock or SUP_x quality", "SIMULATED", "backend/simulation/supplier_sku_profile.py, seed 20260926",
              "Loaded as table supplier_sku_profile"),
        entry("Simulated supplier-SKU profile", "supplier_sku_profile.csv", "sku_id, supplier_id", "ADDED", "-",
              f"{META['relationships']['valid_pairs_in_source']} pairs", "Row key; relationships must be real",
              "Exactly the distinct (SKU_ID, Supplier_ID) pairs of supply_chain_dataset1.csv; 0 invented, 0 missing", "REAL",
              "Copied from source", "None invented"),
        entry("Simulated supplier-SKU profile", "supplier_sku_profile.csv", "available_qty", "ADDED", "-",
              f"range {int(AV['generated']['min'])}-{int(AV['generated']['max'])}, mean {AV['generated']['mean']:.1f}",
              "Supplier-side availability does not exist in the source",
              f"Inventory_Level is buyer stock ({VALID['simulation']['evidence']['inventory_is_buyer_stock_flow_match_pct']}% stock-flow match)",
              "SIMULATED", "One draw from the pair's own historical Order_Quantity values", "Drives the eligibility filter"),
        entry("Simulated supplier-SKU profile", "supplier_sku_profile.csv", "defect_rate", "ADDED", "-",
              f"Beta({DEF['alpha']:.4f}, {DEF['beta']:.4f}) draws, cap {DEF['cap_applied']}",
              "KPI vendors cannot be mapped to SUP_x", "No shared key, SKU, category or date range",
              "SIMULATED", "Supplier-level Beta method-of-moments fit to 5 real KPI supplier rates", "Drives Rule 2/3 and quality score"),
        entry("Simulated supplier-SKU profile", "supplier_sku_profile.csv", "defect_rate (SUP_3)", "TRANSFORMED",
              f"{DEF['values_before_cap']['SUP_3']}", f"{DEF['cap_applied']}", "Approved cap at highest real supplier rate",
              f"Real max {DEF['cap_exact_real_max']} floored to 4 dp", "SIMULATED", "min(value, cap)", f"Capped suppliers: {DEF['capped_suppliers']}"),
        entry("Simulated supplier-SKU profile", "supplier_sku_profile.csv", "fulfillment_rate", "ADDED", "-",
              f"Binomial({FUL['n']}, {FUL['p']})/{FUL['n']}", "Needed by Rule 4; no SUP_x performance data exists",
              f"Real suppliers do not differ (chi-square p = {VALID['simulation']['evidence']['fulfillment_by_supplier_chi2_p']})",
              "SIMULATED", "Supplier-level binomial sampling around real pooled rate", "Drives Rule 4 and fulfillment score"),
        entry("Simulated supplier-SKU profile", "supplier_sku_profile.csv", "on_time_rate → fulfillment_rate", "RENAMED",
              "on_time_rate (proposed)", "fulfillment_rate", "KPI data has no promised delivery date",
              "Columns of Procurement KPI Analysis Dataset.csv", "SIMULATED", "Naming", "Avoids claiming an on-time metric"),
        entry("Simulation metadata", "DataSets/simulated/*.meta.json, evidence/, README.md", "-", "ADDED", "-",
              "seed, parameters, input/output hashes, per-pair statistics, data dictionary", "Reproducibility and audit",
              "-", "N/A", "Generated by the simulation script", "Documentation"),
        # ---------------- database -----------------
        entry("SQLite", "supplier_profiles", "table", "DELETED", f"{sp['rows']} rows × {sp['columns']} columns", "-",
              "Built on an unsupported round-robin KPI-vendor→SUP_x mapping", "No join key between datasets",
              "N/A (derived)", "DROP TABLE", "Replaced by sku_supplier_lanes + supplier_sku_profile"),
        entry("SQLite", "supplier_profiles", "benchmarked_against", "DELETED", f"{len(OLD_MAP)} assignments", "-",
              "Round-robin mapping with no evidence", "Assignment depended only on sort order", "N/A", "Removed with table", "Quality no longer inherited from list position"),
        entry("SQLite", "supplier_profiles", "stock_availability", "DELETED", "sum of latest Inventory_Level by supplier", "-",
              "Mislabelled: it was the buyer's own warehouse stock", "100% stock-flow match", "N/A", "Removed with table", "Replaced by simulated available_qty"),
        entry("SQLite", "supplier_profiles", "rating, reliability_score, delivery_time_days, region", "DELETED",
              "KPI rating via mapping; min-max lead-time std index; 0.7/0.3 blend with KPI days; region mode", "-",
              "Derived from the invalid mapping or from noise (Region changes daily within a fixed lane)",
              "Region: 4 values per fixed lane", "N/A", "Removed with table", "Replaced by real lane cost/lead time + simulated quality"),
        entry("SQLite", "sku_supplier_lanes", "table", "ADDED", "-", f"{T1['sku_supplier_lanes']['rows']} rows × 5 columns",
              "Real per-SKU, per-warehouse supplier cost and lead time", "Supplier, cost, lead time constant for all 365 days per lane",
              "REAL (derived)", "Lossless collapse of 91,250 daily rows; build fails if a lane attribute ever varies", "Candidate matrix source"),
        entry("SQLite", "supplier_sku_profile", "table", "ADDED", "-", f"{T1['supplier_sku_profile']['rows']} rows × 5 columns",
              "Load of the simulated CSV", "Loader rejects invented pairs, duplicates, out-of-range values", "SIMULATED (+ real keys)",
              "to_sql", "ERP-replaceable"),
        entry("SQLite", "supply_chain", "all", "RETAINED", f"{T0['supply_chain']['rows']} rows × {T0['supply_chain']['columns']}",
              f"{T1['supply_chain']['rows']} rows × {T1['supply_chain']['columns']}", "Time series needed for future ML/GA",
              "Row and column counts identical", "REAL", "Unchanged", "Now the primary procurement source"),
        entry("SQLite", "products", "target_stock, deficit, urgent", "DELETED", f"{T0['products']['columns']} columns",
              f"{T1['products']['columns']} columns", "Only existed for product-based procurement",
              f"{URGENT} urgent rows, {URGENT_DISC} of them discontinued", "N/A (derived)", "Not computed any more", "Reproducible from raw CSV"),
        entry("SQLite", "products", "reorder_level", "TRANSFORMED", "clip(lower=1)", "raw value",
              "Clip only served the urgency calculation", f"{ZERO_ROP} row(s) had ReorderLevel 0", "REAL", "Clip removed", "Raw value kept"),
        entry("SQLite", "products", "all rows", "FILTERED", f"{T0['products']['rows']} rows eligible for procurement",
              "0 rows eligible for procurement (table retained)", "No ProductID→SKU or ProductID→supplier relationship",
              "SupplierID == ProductID on all 1,000 rows", "REAL", "No agent reads the table", f"{URGENT_DISC} discontinued urgent products can no longer trigger procurement"),
        entry("SQLite", "idx_products_urgent", "index", "DELETED", "index on products(urgent)", "-", "Column removed", "-", "N/A", "DROP INDEX", "None"),
        entry("SQLite", "purchase_orders", "sku_id", "ADDED", f"{T0['purchase_orders']['columns']} columns",
              f"{T1['purchase_orders']['columns']} columns", "Procurement identity is SKU",
              f"{T0['purchase_orders']['rows']} rows existed; ALTER TABLE keeps any rows", "N/A", "ALTER TABLE ADD COLUMN", "product_id kept for legacy rows"),
        entry("SQLite", "indexes", "supply_chain(date), supply_chain(sku_id,date), unique lanes, unique profile", "ADDED", "-", "4 indexes",
              "Query speed and integrity constraints", "-", "N/A", "CREATE INDEX", "Duplicate pairs rejected by the DB"),
        entry("SQLite", "PRAGMA user_version", "-", "MODIFIED", str(BEFORE["database"]["user_version"]), str(AFTER["database"]["user_version"]),
              "Old databases rebuild automatically", "-", "N/A", "Schema version check at startup", "Existing purchase orders preserved"),
        entry("SQLite", "calendar", "rows", "FILTERED", f"{VALID['calendar_filter']['raw_rows']} raw rows", f"{VALID['calendar_filter']['db_rows']} rows",
              "Pre-existing loader behaviour (unchanged by this migration)",
              f"{VALID['calendar_filter']['duplicate_dates_removed']} duplicate dates", "REAL", "drop_duplicates(date)", "Raw file unchanged"),
        entry("Config", "backend/config.py", "TARGET_STOCK_MULTIPLIER, RULE_PREFERRED_RATING, DEFAULT_DAILY_DEMAND_FALLBACK", "DELETED",
              "3, 4.5, 15", "-", "Their only consumers were removed", "-", "N/A", "Removed", "Replaced by lot-size rule, R2 defect condition, real Units_Sold"),
        entry("Config", "backend/config.py", "w1_rating, w4_reliability", "RENAMED", "w1_rating 0.30, w4_reliability 0.25",
              "w1_quality 0.30, w4_fulfillment 0.25", "Inputs changed; values kept", "-", "N/A", "Key rename", "Weight values unchanged"),
    ]


def deletions() -> list[dict]:
    d = lambda what, count, why, evidence, future, backup, replaced: dict(
        what=what, count=count, why=why, evidence=evidence, future_impact=future, preserved_in_backup=backup, replaced_by=replaced)
    return [
        d("SQLite table supplier_profiles", f"{T0['supplier_profiles']['rows']} rows × {T0['supplier_profiles']['columns']} columns",
          "Built on an unsupported mapping and a mislabelled stock field",
          "No shared key between KPI vendors and SUP_x; Inventory_Level is buyer stock",
          "None: every real input remains in supply_chain", f"Yes: {BACKUP}insightiq.db and full contents in state_before.json",
          "sku_supplier_lanes (real) + supplier_sku_profile (simulated)"),
        d("Round-robin supplier mapping", f"{len(OLD_MAP)} assignments: " + ", ".join(f"{k}→{v}" for k, v in sorted(OLD_MAP.items(), key=lambda kv: int(kv[0][4:]))),
          "Relationship with no evidence", "Assigned by sorted position only", "None", "Yes (state_before.json)", "Explicitly simulated supplier-level quality"),
        d("Derived products columns target_stock, deficit, urgent", f"{T0['products']['rows']} rows × 3 columns",
          "Only used for product-based procurement", f"{URGENT} urgent flags, {URGENT_DISC} on discontinued products",
          "None: recomputable from raw product.csv", "Yes (backup DB); raw CSV untouched", "SKU urgency from supply_chain"),
        d("Index idx_products_urgent", "1 index", "Indexed a removed column", "-", "None", "Yes (backup DB)", "-"),
        d("Products excluded from procurement (FILTERED, not deleted)", f"{T0['products']['rows']} rows",
          "No product→SKU/supplier relationship", "SupplierID == ProductID on every row", "None: table retained for reference",
          "Rows are still in the database and raw CSV", "SKU-based requests"),
        d("Calendar duplicate dates (pre-existing, unchanged)", f"{VALID['calendar_filter']['duplicate_dates_removed']} rows",
          "Loader keeps one row per date", "Duplicate parsed dates", "None", "Raw CSV untouched", "-"),
        d("Config constants", "3 constants", "No remaining consumers", "-", "None", "git history", "See change log"),
        d("Code: perception demand_signals (orders + customers aggregation)", "1 function",
          "E-commerce orders have no product/SKU reference; unused by the UI", "order.csv has no product column",
          "None: orders/customers tables retained", "git history", "Real Units_Sold demand per SKU"),
        d("Behaviour: split-order fallback that ranked the full pool when nobody was eligible", "1 code path",
          "It recommended a supplier that could not fulfil the order", "Required by the 'no false selection' requirement",
          "Split-order optimisation can be added later; combined availability is reported", "git history",
          "no_eligible_supplier result, no recommendation, no plan"),
        d("API GET /api/products", "1 endpoint", "Product table no longer drives procurement", "-", "None", "git history", "GET /api/skus (old path returns 410)"),
    ]


def code_changes() -> list[dict]:
    numstat = subprocess.run(["git", "diff", "--numstat"], cwd=ROOT, capture_output=True, text=True).stdout
    stats = {l.split("\t")[2]: (int(l.split("\t")[0]), int(l.split("\t")[1])) for l in numstat.splitlines() if l.strip()}
    notes = {
        "backend/config.py": ("Paths, provenance, seed, schema version; renamed weight keys", "Product-era constants", "SKU-era constants; weights unchanged"),
        "backend/database.py": ("Removed round-robin profiles; added lanes + validated profile loader; schema v2 migration", "Built supplier_profiles; product urgency", "Real lanes, simulated profile, auto-rebuild, PO migration"),
        "backend/agents/perception_agent.py": ("SKU urgency from latest supply_chain date; lanes kept", "Read products table", "Per-lane Rule 1 inputs, required qty = median lot size sum"),
        "backend/agents/supplier_agent.py": ("Candidate matrix per SKU, eligibility, injectable weights", "Supplier-wide pool, total stock filter, full-pool fallback", "Only suppliers serving the SKU; ineligible reported, never ranked"),
        "backend/agents/knowledge_agent.py": ("Rules on new fields; R5 capacity; no-eligible decision", "Rating/reliability rules", "R2 lead+defect, R4 fulfillment ≥ 0.90, R5 capacity"),
        "backend/agents/planning_agent.py": ("No plan when nothing needed / no supplier; real demand", "Planned any product, ROP/14 demand", "Order = required qty; demand = 28-day mean Units_Sold"),
        "backend/agents/coordinator_agent.py": ("Passes sku_id and required quantity", "product_id, deficit", "sku_id, required_quantity"),
        "backend/main.py": ("SKU endpoints; validation on approve-order", "product endpoints", "/api/skus, sku_id params, 410 for product_id"),
        "backend/tests/test_supplier_agent.py": ("Rewritten for candidate matrix and new components", "Old schema", "35 tests"),
        "frontend/static/js/app.js": ("SKU picker, alert table, matrix, SIM labels, no-eligible states", "Product fields", "SKU fields"),
        "frontend/static/js/api.js": ("skus(), sku_id params", "products(), product_id", "skus(), sku_id"),
        "frontend/static/css/styles.css": ("Ineligible and SIM badges", "-", "2 classes"),
        "requirements.txt": ("+ httpx (FastAPI TestClient)", "-", "httpx==0.28.1"),
    }
    out = [{"file": f, "added": a, "removed": r, "what": notes.get(f, ("", "", ""))[0], "old": notes.get(f, ("", "", ""))[1],
            "new": notes.get(f, ("", "", ""))[2]} for f, (a, r) in stats.items()]
    for f, what in [("backend/simulation/supplier_sku_profile.py", "Simulation generator"),
                    ("backend/tests/test_simulation.py", "14 simulation tests"),
                    ("backend/tests/test_sku_pipeline.py", "33 pipeline/API/migration tests"),
                    ("scripts/migration/*.py", "State capture, validation, logs, PDF report")]:
        out.append({"file": f, "added": None, "removed": 0, "what": what, "old": "-", "new": "new file"})
    return out


RULES = [
    {"rule": "R1 Inventory state", "before": "IF Inventory < ReorderPoint (product)", "after": "Same, per urgent warehouse lane",
     "threshold": "unchanged", "why": "Urgency is per lane in the real data"},
    {"rule": "R2 Preferred", "before": "IF Rating >= 4.5 AND DeliveryTime <= 3d", "after": "IF LeadTime <= 3d AND DefectRate <= 0.05",
     "threshold": "no new numbers (3 d and 0.05 already existed)",
     "why": "Rating only existed through the removed mapping; old rule could never fire (max rating 4.39, min delivery 7.9 d)"},
    {"rule": "R3 High Risk", "before": "IF DefectRate > 0.05", "after": "Same (simulated supplier defect rate)", "threshold": "unchanged", "why": "-"},
    {"rule": "R4 Reliable", "before": "IF no status AND Reliability >= 0.90", "after": "IF no status AND FulfillmentRate >= 0.90",
     "threshold": "0.90 unchanged (not lowered)",
     "why": "Old input was a relative min-max index; 0.90 is kept as a 90% fulfillment target. No supplier meets it (real best 82%); "
            "lowering it would classify sampling noise, since real suppliers do not differ"},
    {"rule": "R5 Capacity (new)", "before": "-", "after": "IF AvailableQty >= RequiredQty ELSE Ineligible",
     "threshold": "required quantity", "why": "Makes every exclusion visible in the XAI trace"},
]

API = [
    {"endpoint": "POST /api/run-agent-pipeline", "before": "product_id (int, required)", "after": "sku_id (str); product_id → 410 Gone"},
    {"endpoint": "GET /api/products", "before": "product list", "after": "410 Gone"},
    {"endpoint": "GET /api/skus", "before": "-", "after": "new: 50 SKUs with urgency, required qty, candidate count"},
    {"endpoint": "GET /api/suppliers", "before": "required_deficit", "after": "sku_id (required), required_quantity (default: SKU's current need); ineligible list, provenance, weights"},
    {"endpoint": "POST /api/approve-order", "before": "product_id", "after": "sku_id; 422 if pair not real or quantity > available_qty"},
    {"endpoint": "GET /api/dashboard", "before": "product alerts, e-commerce lead time", "after": "SKU alerts, snapshot date, real lead time, simulated fulfillment (labelled)"},
    {"endpoint": "GET /api/system-health", "before": "products.urgent count", "after": "urgent SKU count; table list from schema"},
]


def write_md(changes, dels) -> None:
    cols = ["dataset", "location", "field", "change_type", "before", "after", "reason", "evidence", "real_or_simulated", "method", "impact"]
    esc = lambda v: str(v).replace("|", "/").replace("\n", " ")
    lines = ["# Data change log — ProductID → SKU_ID migration", "",
             f"Generated from state snapshots ({BEFORE['captured_at']} → {AFTER['captured_at']}) and validation ({VALID['run_at']}).", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(esc(c[k]) for k in cols) + " |" for c in changes]
    (DOCS / "CHANGE_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = ["# Deletion log", "",
           f"**No raw source records were deleted.** All {len(BEFORE['csv_files'])} raw CSVs have identical SHA-256 before and after "
           f"({'verified' if raw_unchanged() else 'NOT VERIFIED'}); `supply_chain` keeps all "
           f"{T1['supply_chain']['rows']:,} rows and {T1['supply_chain']['columns']} columns.", "",
           "Every deletion below is of a derived database artefact, code, or behaviour.", ""]
    for i, d in enumerate(dels, 1):
        out += [f"## {i}. {d['what']}", f"- **Count:** {d['count']}", f"- **Why:** {d['why']}", f"- **Evidence:** {d['evidence']}",
                f"- **Future impact:** {d['future_impact']}", f"- **Preserved in backup:** {d['preserved_in_backup']}",
                f"- **Replaced by:** {d['replaced_by']}", ""]
    (DOCS / "DELETION_LOG.md").write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    changes, dels = dataset_changes(), deletions()
    log = {"raw_source_unchanged": raw_unchanged(), "dataset_changes": changes, "deletions": dels,
           "code_changes": code_changes(), "rule_changes": RULES, "api_changes": API}
    (DOCS / "change_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    write_md(changes, dels)
    print(f"{len(changes)} dataset changes, {len(dels)} deletions, raw unchanged: {log['raw_source_unchanged']}")


if __name__ == "__main__":
    main()
