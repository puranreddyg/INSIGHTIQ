"""
End-to-end tests of the SKU-based pipeline and API against a database built from
the real CSVs in a temporary directory (the working database is never touched).
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import config, database
from backend.agents import coordinator_agent, knowledge_agent, perception_agent, supplier_agent
from backend.ml import demand_forecaster, supplier_estimator

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
    """The checksums were recorded on Windows (CRLF working copy); git stores the same
    files with LF. Content must match exactly; only the line-ending convention may differ."""
    manifest = config.DATASET_DIR / "CHECKSUMS.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(" *", 1)
        data = (config.DATASET_DIR / name.removeprefix("./")).read_bytes()
        as_is = hashlib.sha256(data).hexdigest()
        as_crlf = hashlib.sha256(data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
        assert digest in (as_is, as_crlf), name


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


def test_loader_rejects_an_invented_relationship():
    sc = database._load_supply_chain()
    profile, _ = supplier_estimator.build_profile(sc, database._load_procurement_kpi())
    bad = profile.copy()
    bad.loc[0, "supplier_id"] = "SUP_1"  # SKU_1 is never supplied by SUP_1
    assert "SUP_1" not in set(bad[bad.sku_id == "SKU_1"].supplier_id.iloc[1:])
    with pytest.raises(ValueError, match="absent from supply_chain"):
        database._validate_supplier_profile(bad, sc)


# ---------------------------------------------------------------------------
# Procurement flow
# ---------------------------------------------------------------------------

PLANNER_URGENT_SKUS = {"SKU_2", "SKU_5", "SKU_19", "SKU_21", "SKU_37", "SKU_39", "SKU_41", "SKU_42", "SKU_44", "SKU_47", "SKU_50"}


def test_perception_detects_urgent_skus_from_the_latest_date(conn, urgent):
    assert perception_agent.snapshot_date(conn).startswith("2024-12-30")
    ids = {r["sku_id"] for r in urgent}
    # every SKU below the planner's static reorder point is still urgent; the forecast
    # (demand x learned lead time) only ever ADDS SKUs the static point under-covers
    assert PLANNER_URGENT_SKUS <= ids and len(ids) == 16
    for r in urgent:
        if r["sku_id"] not in PLANNER_URGENT_SKUS:
            assert any(l["Inventory"] < l["ForecastReorderPoint"] for l in r["focus_lanes"])
    for r in urgent:
        assert r["RequiredQuantity"] > 0 and r["RequiredQuantity"] % config.ORDER_QUANTITY_ROUNDING == 0
        for lane in r["focus_lanes"]:
            assert lane["Inventory"] < lane["EffectiveReorderPoint"]
            assert lane["EffectiveReorderPoint"] == max(lane["ReorderPoint"], lane["ForecastReorderPoint"])
            assert 0.0 <= lane["StockoutProbability"] <= 1.0


def test_required_quantity_is_the_forecast_order_up_to_gap(conn, urgent):
    for r in urgent:
        raw_need = sum(max(0.0, l["OrderUpTo"] - l["Inventory"]) for l in r["focus_lanes"])
        assert r["required_quantity_raw"] == pytest.approx(raw_need, abs=0.2)
        assert r["RequiredQuantity"] - config.ORDER_QUANTITY_ROUNDING < r["required_quantity_raw"] <= r["RequiredQuantity"]


def test_previous_lot_size_rule_is_kept_for_comparison(conn, urgent):
    raw = pd.read_csv(config.CSV_PATHS["supply_chain"])
    medians = raw[raw.Order_Quantity > 0].groupby(["SKU_ID", "Warehouse_ID"]).Order_Quantity.median()
    for r in urgent:
        baseline = sum(medians[(r["sku_id"], l["warehouse_id"])] for l in r["focus_lanes"])
        step = config.ORDER_QUANTITY_ROUNDING
        assert r["required_quantity_baseline"] == math.ceil(baseline / step) * step


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


def _lines(plan):
    if plan.get("split_order"):
        return [(l["supplier_id"], l["quantity"]) for l in plan["allocations"]]
    return [(plan["primary_supplier_id"], plan["order_quantity"])]


def test_partial_order_meets_quality_and_budget_when_capacity_is_short(conn):
    """Not 'unprocessable': buy the most that still meets quality, budget and size, report the gap."""
    from backend.agents import planning_agent
    focus = perception_agent.observe_sku(conn, "SKU_44")
    impossible = 10**6
    sup = supplier_agent.rank_suppliers(conn, "SKU_44", impossible)
    combined = sup["combined_available_qty"]
    assert sup["no_eligible_supplier"] and sup["partial_order_possible"]
    know = knowledge_agent.run({"focus_facts": [focus]}, sup)
    assert know["decision"] == "partial" and know["split_order"] and know["partial_order"]
    pool = {c["supplier_id"]: c for c in sup["pool_candidates"]}
    lines = [(a["supplier_id"], a["allocated_qty"]) for a in know["allocations"]]
    qty = sum(q for _, q in lines)
    assert 0 < qty <= combined and len(lines) <= config.SEARCH_MAX_SUPPLIERS
    assert all(config.MIN_SPLIT_LINE_QTY <= q <= pool[s]["available_qty"] for s, q in lines)
    assert sum(q * pool[s]["defect_rate"] for s, q in lines) <= config.RULE_HIGH_RISK_DEFECT_RATE * qty + 1e-6
    assert know["shortfall_qty"] == impossible - qty
    plan = planning_agent.run(conn, focus, know, impossible, sup)
    assert plan["status"] == "planned" and plan["partial_order"] and plan["order_quantity"] == qty


def test_needs_review_when_no_supplier_has_any_capacity(conn):
    from backend.agents import planning_agent
    focus = perception_agent.observe_sku(conn, "SKU_44")
    sup = {"required_quantity": 500, "ranked_candidates": [], "split_candidates": [], "pool_candidates": [],
           "split_order_possible": False, "partial_order_possible": False, "no_eligible_supplier": True,
           "combined_available_qty": 0, "ineligible_suppliers": [], "budget": None}
    know = knowledge_agent.run({"focus_facts": [focus]}, sup)
    assert know["decision"] == "needs_review" and know["recommended_supplier"] is None and know["allocations"] == []
    plan = planning_agent.run(conn, focus, know, 500, sup)
    assert plan["status"] == "needs_review" and "order_quantity" not in plan


def test_no_split_is_made_when_the_combination_breaks_a_constraint(conn):
    """SKU_5: no single supplier can cover it and no combination meets the 5% defect rule,
    so the order must NOT be split — it goes to manual review."""
    res = coordinator_agent.run_pipeline(conn, "SKU_5")
    assert res["knowledge"]["decision"] == "needs_review" and res["knowledge"]["allocations"] == []
    assert res["planning"]["status"] == "needs_review" and "defect" in res["planning"]["error"]


def test_budget_constraint_is_enforced_and_can_be_set_by_the_caller(conn):
    sku = "SKU_21"
    default = coordinator_agent.run_pipeline(conn, sku)
    assert default["knowledge"]["decision"] == "single"
    assert default["planning"]["total_expense"] <= default["supplier"]["budget"]
    assert default["supplier"]["budget_info"]["source"] == "default"
    # a budget too small for any supplier: no compliant split, single best-effort flagged on budget
    tiny = coordinator_agent.run_pipeline(conn, sku, budget=10.0)
    assert tiny["supplier"]["budget"] == 10.0 and tiny["knowledge"]["budget_constraint_met"] is False
    assert tiny["knowledge"]["decision"] in ("single_flagged", "needs_review")
    assert any(t.get("ruleId") == 7 and t["matched"] is False for t in tiny["knowledge"]["reasoning_trace"])


def test_every_plan_meets_all_constraints_and_is_optimal(conn, urgent):
    """End-to-end over every urgent SKU, checked against an independent exhaustive
    enumeration (backend/tests/_bruteforce.py, no solver): splits/singles meet capacity,
    quantity, quality and budget and are optimal (fewest High Risk, then fewest suppliers);
    a flag or review happens ONLY when no compliant plan exists."""
    from backend.tests._bruteforce import best_key
    cap = config.RULE_HIGH_RISK_DEFECT_RATE
    for r in urgent:
        res = coordinator_agent.run_pipeline(conn, r["sku_id"])
        plan, know, sup = res["planning"], res["knowledge"], res["supplier"]
        R, budget = r["RequiredQuantity"], sup["budget"]
        pool = {c["supplier_id"]: c for c in sup["pool_candidates"]}
        best = best_key(list(pool.values()), R, budget)
        if know["decision"] == "needs_review":
            assert best is None and plan["status"] == "needs_review", r["sku_id"]
            continue
        assert plan["status"] == "planned", r["sku_id"]
        lines = _lines(plan)
        qty = sum(q for _, q in lines)
        cost = sum(q * pool[s]["unit_cost"] for s, q in lines)
        w_def = sum(q * pool[s]["defect_rate"] for s, q in lines) / qty
        assert all(0 < q <= pool[s]["available_qty"] for s, q in lines)            # capacity
        assert qty + plan.get("shortfall_qty", 0) == R                              # quantity
        assert plan["total_expense"] == pytest.approx(cost, abs=0.05)
        assert plan["predictions"]["expected_defect_rate"] == pytest.approx(w_def, abs=1e-4)
        if know["decision"] in ("single", "split"):
            assert w_def <= cap + 1e-9 and cost <= budget + 0.01, r["sku_id"]       # quality + budget
            assert len(lines) <= config.SEARCH_MAX_SUPPLIERS
            risky = sum(pool[s]["defect_rate"] > cap for s, _ in lines)
            assert (risky, len(lines)) == best, (r["sku_id"], (risky, len(lines)), best)  # optimal
            if know["decision"] == "split":
                assert all(q >= config.MIN_SPLIT_LINE_QTY for _, q in lines)
        else:
            assert know["decision"] == "single_flagged" and best is None, r["sku_id"]
            assert not (plan["quality_constraint_met"] and plan["budget_constraint_met"])


def test_split_allocation_unit_behaviour():
    cands = [
        _candidate(supplier_id="A", rank=1, available_qty=300, defect_rate=0.02),   # safe
        _candidate(supplier_id="B", rank=2, available_qty=450, defect_rate=0.10),   # High Risk
        _candidate(supplier_id="C", rank=3, available_qty=200, defect_rate=0.03),   # safe
    ]
    # safe suppliers can cover 450 together -> the High Risk one is not used even though it alone could
    allocs, spare, short = knowledge_agent.allocate_split_order(cands, 450, [])
    assert [(a["supplier_id"], a["allocated_qty"]) for a in allocs] == [("A", 300), ("C", 150)]
    assert [s["supplier_id"] for s in spare] == ["B"] and spare[0]["supplier_status"] == "High Risk" and short == 0
    # 700 needs B. The exact allocation gives B (better score than C) as much as the 5% rule allows:
    # 2%*300 + 10%*qB + 3%*(400-qB) <= 35  ->  qB <= 242
    allocs, spare, short = knowledge_agent.allocate_split_order(cands, 700, [])
    assert [(a["supplier_id"], a["allocated_qty"]) for a in allocs] == [("A", 300), ("B", 242), ("C", 158)] and short == 0
    assert allocs[0]["search"]["weighted_defect_rate"] <= 0.05
    # a budget that rules out the 3-supplier plan -> no split at all
    allocs, spare, short = knowledge_agent.allocate_split_order(cands, 700, [], budget=100.0)
    assert allocs == [] and short == 700
    # 1000 exceeds everyone together (950): partial order, still within 5% defects:
    # A300 + C200 + B260 -> 6 + 6 + 26 = 38 <= 5% of 760
    allocs, spare, short = knowledge_agent.allocate_split_order(cands, 1000, [])
    assert sum(a["allocated_qty"] for a in allocs) == 760 and short == 240


def test_split_respects_the_minimum_line_size():
    cands = [_candidate(supplier_id="A", rank=1, available_qty=520), _candidate(supplier_id="B", rank=2, available_qty=520)]
    allocs, _, _ = knowledge_agent.allocate_split_order(cands, 540, [])
    assert sorted(a["allocated_qty"] for a in allocs) == [50, 490]  # B gets the minimum line, not 20 units


def test_split_prefers_fewer_suppliers_over_more():
    cands = [
        _candidate(supplier_id="A", rank=1, available_qty=250),
        _candidate(supplier_id="B", rank=2, available_qty=250),
        _candidate(supplier_id="C", rank=3, available_qty=480),
        _candidate(supplier_id="D", rank=4, available_qty=100),
    ]
    allocs, _, _ = knowledge_agent.allocate_split_order(cands, 560, [])
    assert [(a["supplier_id"], a["allocated_qty"]) for a in allocs] == [("A", 250), ("C", 310)]


def test_non_urgent_sku_produces_no_purchase_order(conn):
    result = coordinator_agent.run_pipeline(conn, "SKU_3")
    assert result["knowledge"]["inventory_state"] == "Monitoring"
    assert result["planning"]["status"] == "no_action"


def test_planned_order_never_exceeds_the_primary_suppliers_availability(conn, urgent):
    for r in urgent:
        plan = coordinator_agent.run_pipeline(conn, r["sku_id"])["planning"]
        if plan["status"] == "planned" and not plan.get("split_order"):
            rec = supplier_agent.rank_suppliers(conn, r["sku_id"], r["RequiredQuantity"])
            available = {c["supplier_id"]: c["available_qty"] for c in rec["ranked_candidates"]}
            assert plan["order_quantity"] <= available[plan["primary_supplier_id"]]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _candidate(**kw):
    base = {"supplier_id": "S", "unit_cost": 10.0, "lead_time_days": 5.0, "available_qty": 400,
            "defect_rate": 0.02, "fulfillment_rate": 0.80, "evidence_pos": 20}
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
    # quality labels need the supplier's OWN evidence; an estimate that is only the prior is "Unrated"
    ({"defect_rate": 0.20, "evidence_pos": 4}, "Unrated"),
    ({"lead_time_days": 2.0, "defect_rate": 0.01, "fulfillment_rate": 0.99, "evidence_pos": 0}, "Unrated"),
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
    assert body["kpis"]["critical_low_stock_skus"] == len(body["alerts"]) == 16
    assert body["kpi_provenance"]["avg_supplier_fulfillment_rate"] == "LEARNED"
    assert all(a["sku_id"].startswith("SKU_") for a in body["alerts"])
    assert all(0 <= a["stockout_probability"] <= 1 and a["urgent_reasons"] for a in body["alerts"])


def test_pipeline_by_sku(client):
    body = client.post("/api/run-agent-pipeline?sku_id=SKU_2").json()
    assert body["sku_id"] == "SKU_2"
    assert body["supplier"]["field_provenance"]["available_qty"] == "ESTIMATED"
    assert body["supplier"]["field_provenance"]["defect_rate"] == "LEARNED"
    assert "SIMULATED" not in str(body)


def test_model_card_reports_the_backtest(client):
    body = client.get("/api/model").json()
    bt = body["demand_forecaster"]["backtest"]
    k = bt["selected_k"]
    for h in (7, 14):
        assert bt[f"block_h{h}"][f"model_k{k}"] < bt[f"block_h{h}"]["baseline_trailing_28d"]
        assert bt[f"strict_h{h}"][f"model_k{k}"] < bt[f"strict_h{h}"]["baseline_trailing_28d"]


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


def test_split_order_approval_is_all_or_nothing(client):
    plan = client.post("/api/run-agent-pipeline?sku_id=SKU_44").json()["planning"]
    assert plan["split_order"] is True
    lines = [{"supplier_id": l["supplier_id"], "quantity": l["quantity"], "unit_cost": l["unit_cost"],
              "estimated_arrival": l["estimated_arrival_date"]} for l in plan["allocations"]]

    def count():
        return len(client.get("/api/orders?limit=500").json()["purchase_orders"])

    before = count()
    too_big = [dict(lines[0]), {**lines[1], "quantity": 10_000}] + lines[2:]          # 2nd line exceeds availability
    assert client.post("/api/approve-split-order", json={"sku_id": "SKU_44", "allocations": too_big}).status_code == 422
    duplicate = [lines[0], dict(lines[0])]
    assert client.post("/api/approve-split-order", json={"sku_id": "SKU_44", "allocations": duplicate}).status_code == 422
    foreign = [lines[0], {**lines[1], "supplier_id": "SUP_999"}]
    assert client.post("/api/approve-split-order", json={"sku_id": "SKU_44", "allocations": foreign}).status_code == 422
    assert client.post("/api/approve-split-order", json={"sku_id": "SKU_44", "allocations": []}).status_code == 422
    assert count() == before, "a rejected split order must not create any purchase order"

    ok = client.post("/api/approve-split-order", json={"sku_id": "SKU_44", "allocations": lines,
                                                         "backup_supplier_id": plan["backup_supplier_id"]})
    assert ok.status_code == 200
    created = ok.json()["purchase_orders"]
    assert [p["supplier_id"] for p in created] == [l["supplier_id"] for l in lines]
    assert sum(p["quantity"] for p in created) == plan["order_quantity"]
    assert all(p["sku_id"] == "SKU_44" and p["status"] == "Approved" for p in created)
    assert count() == before + len(lines)


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
    model = demand_forecaster.fit(sc, run_backtest=False)
    demand_forecaster.lane_statistics(sc, model, config.ORDER_CYCLE_FALLBACK_DAYS).to_sql("lane_stats", c, index=False)
    pd.DataFrame([{"name": "demand_forecaster", "payload": model.to_json()}]).to_sql("ml_models", c, index=False)
    extra = {"defect_sd": 0.0, "evidence_pos": 40, "capacity_orders": 3, "evidence_source": "ERP"}
    pd.DataFrame([
        {"sku_id": "ERP_SKU", "supplier_id": "ERP_S1", "available_qty": 500, "defect_rate": 0.01, "fulfillment_rate": 0.97, **extra},
        {"sku_id": "ERP_SKU", "supplier_id": "ERP_S2", "available_qty": 40, "defect_rate": 0.01, "fulfillment_rate": 0.97, **extra},
    ]).to_sql("supplier_sku_profile", c, index=False)
    pd.DataFrame({"season": ["Winter"], "is_weekend": [False]}).to_sql("calendar", c, index=False)

    result = coordinator_agent.run_pipeline(c, "ERP_SKU")
    # 10 units/day, lead 4 d + order cycle 10 d -> order up to 140; inventory 90 -> 50
    assert result["focus_fact"]["RequiredQuantity"] == 50
    assert result["supplier"]["filtered_out_suppliers"] == ["ERP_S2"]
    assert result["knowledge"]["recommended_supplier"]["supplier_status"] == "Reliable"
    assert result["planning"]["status"] == "planned"
