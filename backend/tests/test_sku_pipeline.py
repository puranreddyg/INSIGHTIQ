"""
End-to-end tests of the SKU-based pipeline and API against a database built from
the real CSVs in a temporary directory (the working database is never touched).
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import config, database
from backend.agents import coordinator_agent, knowledge_agent, perception_agent, supplier_agent

AGENT_DIR = Path(__file__).resolve().parents[1] / "agents"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("db")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "DATA_DIR", tmp)
    mp.setattr(config, "DB_PATH", tmp / "insightiq.db")
    database.build_database(force=True)
    yield config.DB_PATH
    mp.undo()


@pytest.fixture(scope="module")
def conn(db_path):
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


@pytest.fixture(scope="module")
def client(db_path):
    from backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def urgent(conn):
    return perception_agent.urgent_requests(perception_agent.observe_all(conn))


# ---------------------------------------------------------------------------
# Raw data untouched and time series retained
# ---------------------------------------------------------------------------

def test_raw_source_files_are_byte_identical_to_the_recorded_checksums():
    manifest = config.DATASET_DIR / "CHECKSUMS.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(" *", 1)
        path = config.DATASET_DIR / name.removeprefix("./")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_full_supply_chain_time_series_is_retained(conn):
    raw = pd.read_csv(config.CSV_PATHS["supply_chain"])
    assert conn.execute("SELECT COUNT(*) FROM supply_chain").fetchone()[0] == len(raw) == 91_250
    columns = [r[1] for r in conn.execute("PRAGMA table_info(supply_chain)")]
    for needed in ["date", "sku_id", "warehouse_id", "supplier_id", "region", "units_sold", "inventory_level",
                   "reorder_point", "order_quantity", "unit_cost", "supplier_lead_time_days",
                   "promotion_flag", "stockout_flag", "demand_forecast"]:
        assert needed in columns
    assert conn.execute("SELECT COUNT(DISTINCT date) FROM supply_chain").fetchone()[0] == 365


def test_schema_version_and_obsolete_table_removed(conn):
    assert conn.execute("PRAGMA user_version").fetchone()[0] == config.SCHEMA_VERSION
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert "supplier_profiles" not in names
    assert {"sku_supplier_lanes", "supplier_sku_profile", "purchase_orders"} <= names


def test_lanes_table_is_a_lossless_view_of_the_source(conn):
    raw = pd.read_csv(config.CSV_PATHS["supply_chain"])
    expected = raw.groupby(["SKU_ID", "Warehouse_ID"])[["Supplier_ID", "Unit_Cost", "Supplier_Lead_Time_Days"]].first()
    lanes = pd.read_sql_query("SELECT * FROM sku_supplier_lanes", conn).set_index(["sku_id", "warehouse_id"])
    assert len(lanes) == 250
    for (sku, wh), row in expected.iterrows():
        lane = lanes.loc[(sku, wh)]
        assert (lane.supplier_id, lane.unit_cost, lane.lead_time_days) == (row.Supplier_ID, row.Unit_Cost, row.Supplier_Lead_Time_Days)


# ---------------------------------------------------------------------------
# Unsupported relationships are gone
# ---------------------------------------------------------------------------

def test_no_agent_reads_the_product_table_or_kpi_vendor_names():
    for path in AGENT_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "FROM products" not in source, path.name
        for vendor in ["Alpha_Inc", "Beta_Supplies", "Gamma_Co", "Delta_Logistics", "Epsilon_Group"]:
            assert vendor not in source, (path.name, vendor)


def test_products_table_has_no_procurement_columns(conn):
    columns = {r[1] for r in conn.execute("PRAGMA table_info(products)")}
    assert not {"urgent", "deficit", "target_stock"} & columns
    assert conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 1000


def test_loader_rejects_an_invented_relationship(tmp_path, monkeypatch):
    bad = pd.read_csv(config.CSV_PATHS["supplier_sku_profile"])
    bad.loc[0, "supplier_id"] = "SUP_1"  # SKU_1 is never supplied by SUP_1
    assert "SUP_1" not in set(bad[bad.sku_id == "SKU_1"].supplier_id.iloc[1:])
    path = tmp_path / "profile.csv"
    bad.to_csv(path, index=False)
    monkeypatch.setitem(config.CSV_PATHS, "supplier_sku_profile", path)
    sc = database._load_supply_chain()
    with pytest.raises(ValueError, match="absent from supply_chain"):
        database._load_supplier_sku_profile(sc)


# ---------------------------------------------------------------------------
# Procurement flow
# ---------------------------------------------------------------------------

def test_perception_detects_urgent_skus_from_the_latest_date(conn, urgent):
    assert perception_agent.snapshot_date(conn).startswith("2024-12-30")
    assert len(urgent) == 11
    for r in urgent:
        assert r["RequiredQuantity"] > 0 and r["RequiredQuantity"] % config.ORDER_QUANTITY_ROUNDING == 0
        assert all(l["Inventory"] < l["ReorderPoint"] for l in r["focus_lanes"])


def test_required_quantity_is_sum_of_urgent_lane_median_lot_sizes(conn, urgent):
    raw = pd.read_csv(config.CSV_PATHS["supply_chain"])
    medians = raw[raw.Order_Quantity > 0].groupby(["SKU_ID", "Warehouse_ID"]).Order_Quantity.median()
    for r in urgent:
        raw_need = sum(medians[(r["sku_id"], l["warehouse_id"])] for l in r["focus_lanes"])
        assert r["required_quantity_raw"] == pytest.approx(raw_need)


def test_warehouse_detail_is_preserved_in_the_request(urgent):
    r = urgent[0]
    assert r["lane_count"] == len(r["lanes"]) == 5
    assert {"warehouse_id", "supplier_id", "Inventory", "ReorderPoint", "DailyDemand"} <= set(r["lanes"][0])


def test_eligibility_is_applied_for_every_urgent_sku(conn, urgent):
    for r in urgent:
        result = supplier_agent.rank_suppliers(conn, r["sku_id"], r["RequiredQuantity"])
        assert all(c["available_qty"] >= r["RequiredQuantity"] for c in result["ranked_candidates"])
        assert all(c["available_qty"] < r["RequiredQuantity"] for c in result["ineligible_suppliers"])
        assert result["eligible_count"] + len(result["ineligible_suppliers"]) == result["pool_size"]


def test_candidates_are_only_suppliers_that_serve_the_sku(conn):
    raw = pd.read_csv(config.CSV_PATHS["supply_chain"])
    for sku in ["SKU_1", "SKU_7", "SKU_50"]:
        served = set(raw[raw.SKU_ID == sku].Supplier_ID)
        result = supplier_agent.full_matrix(conn, sku)
        assert {c["supplier_id"] for c in result["ranked_candidates"]} == served


def test_different_skus_produce_different_rankings(conn):
    orders = {tuple(c["supplier_id"] for c in supplier_agent.full_matrix(conn, f"SKU_{i}")["ranked_candidates"]) for i in range(1, 51)}
    tops = {o[0] for o in orders}
    assert len(orders) > 10 and len(tops) >= 5


def test_recommendations_differ_across_urgent_skus(conn, urgent):
    recommended = set()
    for r in urgent:
        rec = coordinator_agent.run_pipeline(conn, r["sku_id"])["knowledge"]["recommended_supplier"]
        if rec:
            recommended.add(rec["supplier_id"])
    assert len(recommended) >= 3


def test_no_eligible_supplier_selects_nobody_end_to_end(conn, urgent):
    blocked = [r for r in urgent if not supplier_agent.rank_suppliers(conn, r["sku_id"], r["RequiredQuantity"])["eligible_count"]]
    assert blocked, "expected at least one urgent SKU with no single eligible supplier"
    result = coordinator_agent.run_pipeline(conn, blocked[0]["sku_id"])
    assert result["knowledge"]["recommended_supplier"] is None
    assert result["planning"]["status"] == "no_eligible_supplier"
    assert "order_quantity" not in result["planning"]


def test_non_urgent_sku_produces_no_purchase_order(conn):
    result = coordinator_agent.run_pipeline(conn, "SKU_3")
    assert result["knowledge"]["inventory_state"] == "Monitoring"
    assert result["planning"]["status"] == "no_action"


def test_planned_order_never_exceeds_the_primary_suppliers_availability(conn, urgent):
    for r in urgent:
        plan = coordinator_agent.run_pipeline(conn, r["sku_id"])["planning"]
        if plan["status"] == "planned":
            rec = supplier_agent.rank_suppliers(conn, r["sku_id"], r["RequiredQuantity"])
            available = {c["supplier_id"]: c["available_qty"] for c in rec["ranked_candidates"]}
            assert plan["order_quantity"] <= available[plan["primary_supplier_id"]]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _candidate(**kw):
    base = {"supplier_id": "S", "unit_cost": 10.0, "lead_time_days": 5.0, "available_qty": 400,
            "defect_rate": 0.02, "fulfillment_rate": 0.80}
    return {**base, **kw}


def test_rule_thresholds_are_unchanged():
    assert (config.RULE_HIGH_RISK_DEFECT_RATE, config.RULE_RELIABLE_THRESHOLD, config.RULE_PREFERRED_DELIVERY_DAYS) == (0.05, 0.90, 3)


@pytest.mark.parametrize("kw,expected", [
    ({"lead_time_days": 3.0, "defect_rate": 0.05}, "Preferred"),
    ({"lead_time_days": 3.0, "defect_rate": 0.0501}, "High Risk"),
    ({"lead_time_days": 3.5, "defect_rate": 0.02}, "Standard"),
    ({"fulfillment_rate": 0.90}, "Reliable"),
    ({"fulfillment_rate": 0.8999}, "Standard"),
    ({"available_qty": 299}, "Ineligible"),
])
def test_rule_outcomes(kw, expected):
    assert knowledge_agent.evaluate_supplier(_candidate(**kw), 300, [])["supplier_status"] == expected


def test_capacity_rule_is_in_the_trace():
    trace: list[dict] = []
    knowledge_agent.evaluate_supplier(_candidate(available_qty=299), 300, trace)
    r5 = next(t for t in trace if t.get("ruleId") == 5)
    assert r5["matched"] is False and r5["conditions"][0]["expression"] == "299 >= 300"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_dashboard_is_sku_based(client):
    body = client.get("/api/dashboard").json()
    assert body["snapshot_date"] == "2024-12-30"
    assert body["kpis"]["critical_low_stock_skus"] == len(body["alerts"]) == 11
    assert body["kpi_provenance"]["avg_supplier_fulfillment_rate"] == "SIMULATED"
    assert all(a["sku_id"].startswith("SKU_") for a in body["alerts"])


def test_pipeline_by_sku(client):
    body = client.post("/api/run-agent-pipeline?sku_id=SKU_2").json()
    assert body["sku_id"] == "SKU_2"
    assert body["supplier"]["field_provenance"]["available_qty"] == "SIMULATED"


def test_product_id_can_no_longer_trigger_procurement(client):
    r = client.post("/api/run-agent-pipeline?product_id=6")
    assert r.status_code == 410
    assert client.get("/api/products").status_code == 410


def test_unknown_sku_is_404(client):
    assert client.post("/api/run-agent-pipeline?sku_id=SKU_999").status_code == 404
    assert client.get("/api/suppliers?sku_id=SKU_999").status_code == 404


def test_supplier_matrix_defaults_to_the_skus_requirement(client, urgent):
    r = urgent[0]
    body = client.get(f"/api/suppliers?sku_id={r['sku_id']}").json()
    assert body["required_quantity"] == r["RequiredQuantity"]
    assert client.get(f"/api/suppliers?sku_id={r['sku_id']}&required_quantity=0").json()["eligible_count"] == body["pool_size"]


def test_approve_order_validates_relationship_and_availability(client, conn):
    ok = conn.execute("SELECT sku_id, supplier_id, available_qty FROM supplier_sku_profile LIMIT 1").fetchone()
    base = {"sku_id": ok["sku_id"], "supplier_id": ok["supplier_id"], "quantity": ok["available_qty"], "unit_cost": 10.0}
    assert client.post("/api/approve-order", json={**base, "supplier_id": "SUP_1"}).status_code == 422
    assert client.post("/api/approve-order", json={**base, "quantity": ok["available_qty"] + 1}).status_code == 422
    assert client.post("/api/approve-order", json={**base, "product_id": 6, "sku_id": None}).status_code == 422
    created = client.post("/api/approve-order", json=base)
    assert created.status_code == 200
    assert created.json()["purchase_order"]["sku_id"] == ok["sku_id"]


def test_existing_purchase_orders_survive_a_schema_migration(tmp_path, monkeypatch):
    """A v1 database with a PO row is rebuilt to v2 without losing the row."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "insightiq.db")
    old = sqlite3.connect(config.DB_PATH)
    old.execute("""CREATE TABLE purchase_orders (id INTEGER PRIMARY KEY AUTOINCREMENT, product_id INTEGER,
                   product_name TEXT, supplier_id TEXT, supplier_name TEXT, quantity INTEGER, unit_cost REAL,
                   total_cost REAL, order_date TEXT, estimated_arrival TEXT, backup_supplier_id TEXT, status TEXT,
                   created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    old.execute("CREATE TABLE supplier_profiles (supplier_id TEXT)")
    old.execute("INSERT INTO purchase_orders (product_id, product_name, supplier_id, quantity, status) VALUES (6, 'Television', 'SUP_1', 10, 'Approved')")
    old.commit()
    old.close()

    database.build_database(force=False)

    new = sqlite3.connect(config.DB_PATH)
    assert new.execute("PRAGMA user_version").fetchone()[0] == config.SCHEMA_VERSION
    assert new.execute("SELECT product_id, product_name, sku_id FROM purchase_orders").fetchall() == [(6, "Television", None)]
    assert new.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'supplier_profiles'").fetchone()[0] == 0
    new.close()


# ---------------------------------------------------------------------------
# Future compatibility: agents run on any source that fills the same tables
# ---------------------------------------------------------------------------

def test_pipeline_runs_on_non_csv_data_with_the_same_schema():
    """Stand-in for an ERP feed: tables filled from plain records, no CSV involved."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    days = pd.date_range("2025-01-01", periods=30, freq="D").strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for wh, sup, inv in [("WH_A", "ERP_S1", 90), ("WH_B", "ERP_S2", 500)]:
        for i, d in enumerate(days):
            rows.append({"date": d, "sku_id": "ERP_SKU", "warehouse_id": wh, "supplier_id": sup, "region": "X",
                         "units_sold": 10, "inventory_level": inv, "supplier_lead_time_days": 4, "reorder_point": 100,
                         "order_quantity": 120 if i % 10 == 0 else 0, "unit_cost": 7.5, "unit_price": 12.0,
                         "promotion_flag": 0, "stockout_flag": 0, "demand_forecast": 10.0})
    sc = pd.DataFrame(rows)
    sc.to_sql("supply_chain", c, index=False)
    database._build_sku_supplier_lanes(sc).to_sql("sku_supplier_lanes", c, index=False)
    pd.DataFrame([
        {"sku_id": "ERP_SKU", "supplier_id": "ERP_S1", "available_qty": 500, "defect_rate": 0.01, "fulfillment_rate": 0.97},
        {"sku_id": "ERP_SKU", "supplier_id": "ERP_S2", "available_qty": 50, "defect_rate": 0.01, "fulfillment_rate": 0.97},
    ]).to_sql("supplier_sku_profile", c, index=False)
    pd.DataFrame({"season": ["Winter"], "is_weekend": [False]}).to_sql("calendar", c, index=False)

    result = coordinator_agent.run_pipeline(c, "ERP_SKU")
    assert result["focus_fact"]["RequiredQuantity"] == 120
    assert result["supplier"]["filtered_out_suppliers"] == ["ERP_S2"]
    assert result["knowledge"]["recommended_supplier"]["supplier_status"] == "Reliable"
    assert result["planning"]["status"] == "planned"
