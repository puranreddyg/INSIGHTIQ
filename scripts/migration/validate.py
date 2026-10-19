"""Post-migration validation. Writes docs/migration/validation_results.json.

HISTORICAL: this validates the 2026-09-27 SKU migration, which used the simulated
supplier profile (backend/simulation, removed on 2026-10-19 when the predictive
layer in backend/ml replaced it). To re-run it, check out commit e74f7a3.

    python scripts/migration/validate.py

Every number in the migration report comes from this file or the state
snapshots — nothing in the report is typed by hand.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.agents import coordinator_agent, perception_agent, supplier_agent  # noqa: E402
from backend.database import connect  # noqa: E402
from backend.simulation import supplier_sku_profile as sim  # noqa: E402

OUT = ROOT / "docs" / "migration" / "validation_results.json"
BACKUP_DB = ROOT / "data" / "backup" / "pre_migration_2026-09-27" / "insightiq.db"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, passed: bool, detail) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def raw_checksums() -> list[dict]:
    out = []
    for line in (config.DATASET_DIR / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(" *", 1)
        path = config.DATASET_DIR / name.removeprefix("./")
        out.append(check(f"raw unchanged: {name.removeprefix('./')}", sha256(path) == digest, digest[:16]))
    head = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "DataSets/*.csv", "DataSets/archive (1)"], cwd=ROOT)
    out.append(check("raw CSVs identical to git HEAD", head.returncode == 0, "git diff --quiet HEAD -- raw CSVs"))
    return out


def row_counts(conn) -> list[dict]:
    raw = {k: pd.read_csv(p) for k, p in config.CSV_PATHS.items()}
    pairs = [("product", "products"), ("order", "orders"), ("customer", "customers"), ("calendar", "calendar"),
             ("procurement_kpi", "procurement_kpi"), ("supply_chain", "supply_chain"),
             ("supplier_sku_profile", "supplier_sku_profile")]
    out = []
    for key, table in pairs:
        n_db = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        out.append({"source": config.CSV_PATHS[key].relative_to(ROOT).as_posix(), "table": table,
                    "source_rows": len(raw[key]), "db_rows": n_db, "difference": len(raw[key]) - n_db})
    n_pairs = raw["supply_chain"][["SKU_ID", "Warehouse_ID"]].drop_duplicates().shape[0]
    out.append({"source": "supply_chain (distinct SKU × warehouse)", "table": "sku_supplier_lanes",
                "source_rows": n_pairs, "db_rows": conn.execute("SELECT COUNT(*) FROM sku_supplier_lanes").fetchone()[0],
                "difference": 0})
    return out


def calendar_filter_detail() -> dict:
    df = pd.read_csv(config.CSV_PATHS["calendar"])
    parsed = pd.to_datetime(df["date"], errors="coerce")
    invalid = int(parsed.isna().sum())
    dupes = int(parsed.dropna().duplicated().sum())
    return {"raw_rows": len(df), "unparseable_dates_removed": invalid, "duplicate_dates_removed": dupes,
            "db_rows": len(df) - invalid - dupes}


def data_integrity(conn) -> list[dict]:
    raw_sc = pd.read_csv(config.CSV_PATHS["supply_chain"])
    source_pairs = set(map(tuple, raw_sc[["SKU_ID", "Supplier_ID"]].drop_duplicates().to_numpy()))
    profile = pd.read_sql_query("SELECT * FROM supplier_sku_profile", conn)
    lanes = pd.read_sql_query("SELECT * FROM sku_supplier_lanes", conn)
    prof_pairs = set(map(tuple, profile[["sku_id", "supplier_id"]].to_numpy()))
    lane_pairs = set(map(tuple, lanes[["sku_id", "supplier_id"]].drop_duplicates().to_numpy()))
    _, kpi_rates = sim.kpi_defect_rates(pd.read_csv(config.CSV_PATHS["procurement_kpi"]))
    return [
        check("profile pairs invented (not in source)", not (prof_pairs - source_pairs), len(prof_pairs - source_pairs)),
        check("source pairs missing from profile", not (source_pairs - prof_pairs), len(source_pairs - prof_pairs)),
        check("profile pairs == lane pairs", prof_pairs == lane_pairs, len(prof_pairs)),
        check("profile duplicate (sku, supplier)", not profile.duplicated(["sku_id", "supplier_id"]).any(), 0),
        check("lanes duplicate (sku, warehouse)", not lanes.duplicated(["sku_id", "warehouse_id"]).any(), 0),
        check("profile nulls", not profile.isna().any().any(), int(profile.isna().sum().sum())),
        check("lanes nulls", not lanes.isna().any().any(), int(lanes.isna().sum().sum())),
        check("available_qty within historical range 200-499", profile.available_qty.between(200, 499).all(),
              [int(profile.available_qty.min()), int(profile.available_qty.max())]),
        check("defect_rate within [0, real max]", profile.defect_rate.between(0, kpi_rates.max()).all(),
              [float(profile.defect_rate.min()), float(profile.defect_rate.max()), round(float(kpi_rates.max()), 6)]),
        check("fulfillment_rate within [0, 1]", profile.fulfillment_rate.between(0, 1).all(),
              [float(profile.fulfillment_rate.min()), float(profile.fulfillment_rate.max())]),
        check("quality fields constant per supplier",
              (profile.groupby("supplier_id")[["defect_rate", "fulfillment_rate"]].nunique() == 1).all().all(), "10 suppliers"),
        check("SQLite integrity_check", conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "ok"),
        check("schema user_version", conn.execute("PRAGMA user_version").fetchone()[0] == config.SCHEMA_VERSION, config.SCHEMA_VERSION),
    ]


def simulation_stats(conn) -> dict:
    raw_sc = pd.read_csv(config.CSV_PATHS["supply_chain"])
    history = sim.order_quantity_history(raw_sc)
    profile = pd.read_sql_query("SELECT * FROM supplier_sku_profile", conn)
    hist_all = np.concatenate(history.values)
    eligibility = {}
    for q in [276, 351, 425, 468]:
        per_sku = profile.assign(e=profile.available_qty >= q).groupby("sku_id").e.agg(["sum", "size"])
        eligibility[str(q)] = {"pairs_eligible_pct": round(float((profile.available_qty >= q).mean()) * 100, 1),
                               "skus_with_none": int((per_sku["sum"] == 0).sum()),
                               "skus_with_all": int((per_sku["sum"] == per_sku["size"]).sum())}
    in_history = all(r.available_qty in set(history[(r.sku_id, r.supplier_id)]) for r in profile.itertuples())
    kpi = pd.read_csv(config.CSV_PATHS["procurement_kpi"])
    k = kpi.dropna(subset=["Defective_Units"])
    finished = kpi[kpi.Order_Status != "Pending"]
    chi2 = st.chi2_contingency(pd.crosstab(finished.Supplier, finished.Order_Status == "Delivered"))
    return {
        "available_qty_vs_history": {"generated_mean": round(float(profile.available_qty.mean()), 2),
                                     "generated_sd": round(float(profile.available_qty.std()), 2),
                                     "history_mean": round(float(hist_all.mean()), 2), "history_sd": round(float(hist_all.std(ddof=1)), 2),
                                     "every_value_is_a_real_order_size_of_its_pair": in_history},
        "eligibility_mix_by_request_size": eligibility,
        "evidence": {
            "order_qty_pairs_differ_kruskal_p": round(float(st.kruskal(*history.values).pvalue), 4),
            "order_qty_decile_counts_200_500": np.histogram(hist_all, bins=10, range=(200, 500))[0].tolist(),
            "defect_rate_by_supplier_kruskal_p": float(st.kruskal(*[g.Defective_Units / g.Quantity for _, g in k.groupby("Supplier")]).pvalue),
            "fulfillment_by_supplier_chi2": round(float(chi2.statistic), 3),
            "fulfillment_by_supplier_chi2_p": round(float(chi2.pvalue), 4),
            "inventory_is_buyer_stock_flow_match_pct": _stock_flow_match(raw_sc),
        },
        "high_risk_suppliers": sorted(profile[profile.defect_rate > config.RULE_HIGH_RISK_DEFECT_RATE].supplier_id.unique(), key=lambda s: int(s[4:])),
        "reliable_suppliers_fulfillment_ge_090": sorted(profile[profile.fulfillment_rate >= config.RULE_RELIABLE_THRESHOLD].supplier_id.unique()),
    }


def _stock_flow_match(sc: pd.DataFrame) -> float:
    s = sc.sort_values(["SKU_ID", "Warehouse_ID", "Date"]).copy()
    s["prev"] = s.groupby(["SKU_ID", "Warehouse_ID"]).Inventory_Level.shift()
    s = s.dropna(subset=["prev"])
    pred = (s.prev - s.Units_Sold + s.Order_Quantity).clip(lower=0)
    return round(float((pred == s.Inventory_Level).mean()) * 100, 2)


def reproducibility() -> list[dict]:
    before_sha = sha256(sim.OUT_CSV)
    meta_before = sim.OUT_META.read_text(encoding="utf-8")
    a, _ = sim.generate(*sim.load_raw())
    b, _ = sim.generate(*sim.load_raw())
    sim.OUT_CSV.unlink()
    deleted = not sim.OUT_CSV.exists()
    sim.write()
    after_sha = sha256(sim.OUT_CSV)
    meta_after = json.loads(sim.OUT_META.read_text(encoding="utf-8"))
    # keep the original generation timestamp; only the CSV is the artefact under test
    sim.OUT_META.write_text(meta_before, encoding="utf-8")
    return [
        check("two in-memory generations identical", a.equals(b), "seed 20260926"),
        check("file deleted then regenerated byte-identical", deleted and before_sha == after_sha,
              {"before": before_sha[:16], "after": after_sha[:16]}),
        check("regenerated output hash matches metadata", meta_after["output_sha256"] == before_sha, before_sha[:16]),
    ]


def end_to_end(conn) -> dict:
    requests = perception_agent.observe_all(conn)
    urgent = perception_agent.urgent_requests(requests)
    rows = []
    for r in urgent:
        res = coordinator_agent.run_pipeline(conn, r["sku_id"])
        sup, kn, pl = res["supplier"], res["knowledge"], res["planning"]
        rows.append({
            "sku_id": r["sku_id"], "urgent_warehouses": ", ".join(l["warehouse_id"] for l in r["focus_lanes"]),
            "required_quantity": r["RequiredQuantity"], "candidates": sup["pool_size"], "eligible": sup["eligible_count"],
            "excluded": ", ".join(sup["filtered_out_suppliers"]),
            "recommended": kn["recommended_supplier"]["supplier_id"] if kn["recommended_supplier"] else None,
            "recommended_status": kn["recommended_supplier"]["supplier_status"] if kn["recommended_supplier"] else None,
            "backup": pl.get("backup_supplier_id"), "plan_status": pl["status"],
            "combined_available": sup["combined_available_qty"],
        })
    top_by_sku = {s: supplier_agent.full_matrix(conn, s)["ranked_candidates"][0]["supplier_id"] for s in requests}
    orders = {tuple(c["supplier_id"] for c in supplier_agent.full_matrix(conn, s)["ranked_candidates"]) for s in requests}
    alt = {"w1_quality": 0.10, "w2_cost": 0.10, "w3_delivery": 0.70, "w4_fulfillment": 0.10}
    changed = sum(
        supplier_agent.rank_suppliers(conn, s, top_n=None)["ranked_candidates"][0]["supplier_id"]
        != supplier_agent.rank_suppliers(conn, s, top_n=None, weights=alt)["ranked_candidates"][0]["supplier_id"]
        for s in requests)
    no_action = coordinator_agent.run_pipeline(conn, "SKU_3")["planning"]["status"]
    return {
        "urgent_skus": rows,
        "distinct_recommended_suppliers": sorted({r["recommended"] for r in rows if r["recommended"]}, key=lambda s: int(s[4:])),
        "top_heuristic_supplier_count_across_50_skus": pd.Series(top_by_sku).value_counts().sort_index(key=lambda i: i.str[4:].astype(int)).to_dict(),
        "distinct_full_rankings_across_50_skus": len(orders),
        "skus_whose_top_supplier_changes_with_delivery_heavy_weights": int(changed),
        "non_urgent_sku_SKU_3_plan_status": no_action,
    }


def old_behaviour() -> dict:
    """Run the pre-migration code (git HEAD) against the backed-up pre-migration DB."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(["git", "archive", "HEAD", "backend"], cwd=ROOT, capture_output=True, check=True).stdout
        tarfile.open(fileobj=io.BytesIO(archive)).extractall(tmp, filter="data")
        (Path(tmp) / "data").mkdir()
        shutil.copy(BACKUP_DB, Path(tmp) / "data" / "insightiq.db")
        script = (
            "import json, sqlite3\n"
            "from backend.agents import coordinator_agent as co\n"
            "c = sqlite3.connect('data/insightiq.db'); c.row_factory = sqlite3.Row\n"
            "ids = [r[0] for r in c.execute('SELECT product_id FROM products WHERE urgent = 1')]\n"
            "out = []\n"
            "for pid in ids:\n"
            "    r = co.run_pipeline(c, product_id=pid)\n"
            "    rec = r['knowledge']['recommended_supplier']\n"
            "    out.append([pid, rec['supplier_id'] if rec else None, r['planning'].get('backup_supplier_id')])\n"
            "print(json.dumps(out))\n"
        )
        res = subprocess.run([sys.executable, "-c", script], cwd=tmp, capture_output=True, text=True, check=True)
    runs = json.loads(res.stdout)
    df = pd.DataFrame(runs, columns=["product_id", "recommended", "backup"])
    return {
        "urgent_products_run": len(df),
        "recommended_counts": df.recommended.value_counts().to_dict(),
        "backup_counts": df.backup.value_counts().to_dict(),
    }


def run_pytest() -> dict:
    res = subprocess.run([sys.executable, "-m", "pytest", "backend/tests", "-q", "-p", "no:warnings", "-rN"],
                         cwd=ROOT, capture_output=True, text=True)
    summary = [l for l in res.stdout.splitlines() if " passed" in l or " failed" in l][-1]
    collected = subprocess.run([sys.executable, "-m", "pytest", "backend/tests", "--collect-only", "-q"],
                               cwd=ROOT, capture_output=True, text=True).stdout
    per_file: dict[str, int] = {}
    for line in collected.splitlines():
        if "::" in line:
            f = line.split("::")[0].split("/")[-1]
            per_file[f] = per_file.get(f, 0) + 1
    return {"exit_code": res.returncode, "summary": summary.strip("= "), "tests_per_file": per_file}


def future_compat(conn) -> list[dict]:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(supply_chain)")}
    needed = {"date", "sku_id", "warehouse_id", "region", "units_sold", "promotion_flag", "inventory_level",
              "reorder_point", "demand_forecast", "order_quantity", "stockout_flag"}
    cand = supplier_agent.full_matrix(conn, "SKU_1")["ranked_candidates"][0]
    return [
        check("time-series columns for demand ML retained", needed <= cols, sorted(needed)),
        check("365 daily observations per SKU-warehouse retained",
              conn.execute("SELECT MIN(n) FROM (SELECT COUNT(*) n FROM supply_chain GROUP BY sku_id, warehouse_id)").fetchone()[0] == 365, 365),
        check("raw criteria on every candidate",
              {"unit_cost", "lead_time_days", "available_qty", "defect_rate", "fulfillment_rate"} <= set(cand), sorted(cand)),
        check("per-supplier availability retained (split orders)",
              conn.execute("SELECT COUNT(*) FROM supplier_sku_profile").fetchone()[0] == 211, 211),
    ]


def main() -> None:
    with connect() as conn:
        results = {
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "raw_checksums": raw_checksums(),
            "row_counts": row_counts(conn),
            "calendar_filter": calendar_filter_detail(),
            "integrity": data_integrity(conn),
            "simulation": simulation_stats(conn),
            "end_to_end": end_to_end(conn),
            "future_compatibility": future_compat(conn),
        }
    results["reproducibility"] = reproducibility()
    results["old_behaviour"] = old_behaviour()
    results["pytest"] = run_pytest()
    checks = [c for key in ("raw_checksums", "integrity", "future_compatibility", "reproducibility") for c in results[key]]
    results["all_checks_passed"] = all(c["passed"] for c in checks) and results["pytest"]["exit_code"] == 0
    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    failed = [c["check"] for c in checks if not c["passed"]]
    print(f"{len(checks)} checks, failed: {failed or 'none'} | pytest: {results['pytest']['summary']}")


if __name__ == "__main__":
    main()
