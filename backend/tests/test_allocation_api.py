"""
End-to-end: pipeline -> split/partial plan -> approval endpoint, against a
database built from the real CSVs in a temporary directory.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend import config, database
from backend.agents import perception_agent


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
def client(db_path):
    from backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def plans(client, db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    urgent = perception_agent.urgent_requests(perception_agent.observe_all(conn))
    conn.close()
    out = {}
    for r in urgent:
        body = client.post(f"/api/run-agent-pipeline?sku_id={r['sku_id']}").json()
        out.setdefault(body["planning"]["status"], body)
    return out


def _payload(plan, **kw):
    return {
        "sku_id": plan["sku_id"],
        "allocations": [{"supplier_id": l["supplier_id"], "quantity": l["allocated_qty"], "unit_cost": l["unit_cost"],
                         "estimated_arrival": l["estimated_arrival"]} for l in plan["allocations"]],
        "is_partial": plan["is_partial"],
        **kw,
    }


def _po_count(db_path):
    with sqlite3.connect(db_path) as c:
        return c.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0]


def test_real_data_exercises_single_split_and_partial_plans(plans):
    assert {"planned", "planned_split", "planned_partial"} <= set(plans)


def test_pipeline_exposes_the_quantity_calculation(plans):
    calc = plans["planned"]["focus_fact"]["quantity_calculation"]
    assert calc["method"] == "forecast"
    for key in ("forecast_demand", "safety_stock", "target_stock", "current_inventory", "quantity_before_rounding",
                "final_quantity", "explanation", "forecast_method"):
        assert key in calc


def test_split_plan_is_approved_as_one_group(client, plans, db_path):
    plan = plans["planned_split"]["planning"]
    before = _po_count(db_path)
    res = client.post("/api/approve-order", json=_payload(plan))
    assert res.status_code == 200, res.text
    body = res.json()
    assert _po_count(db_path) == before + len(plan["allocations"])
    assert len({r["order_group"] for r in body["purchase_orders"]}) == 1
    assert body["total_quantity"] == plan["order_quantity"]
    assert body["total_cost"] == pytest.approx(plan["total_expense"], abs=0.05)
    for row, line in zip(body["purchase_orders"], plan["allocations"]):
        assert (row["supplier_id"], row["quantity"], row["estimated_arrival"]) == (line["supplier_id"], line["allocated_qty"], line["estimated_arrival"])
        assert row["allocation_count"] == len(plan["allocations"]) and row["is_partial"] == 0


def test_partial_plan_needs_the_explicit_partial_flag(client, plans):
    plan = plans["planned_partial"]["planning"]
    assert client.post("/api/approve-order", json=_payload(plan, is_partial=False)).status_code == 422
    res = client.post("/api/approve-order", json=_payload(plan))
    assert res.status_code == 200 and res.json()["is_partial"] is True
    assert all(r["is_partial"] == 1 and r["required_quantity"] == plan["required_quantity"] for r in res.json()["purchase_orders"])


def test_single_supplier_plan_via_allocations_and_legacy_payload(client, plans):
    plan = plans["planned"]["planning"]
    assert client.post("/api/approve-order", json=_payload(plan)).status_code == 200
    legacy = {"sku_id": plan["sku_id"], "supplier_id": plan["primary_supplier_id"], "quantity": plan["order_quantity"],
              "unit_cost": plan["unit_cost"], "estimated_arrival": plan["estimated_arrival_date"]}
    res = client.post("/api/approve-order", json=legacy)
    assert res.status_code == 200 and res.json()["purchase_order"]["supplier_id"] == plan["primary_supplier_id"]


@pytest.mark.parametrize("mutate,fragment", [
    (lambda p: p["allocations"][0].update(quantity=10_000), "available"),
    (lambda p: p["allocations"][0].update(quantity=0), "positive"),
    (lambda p: p["allocations"].append(dict(p["allocations"][0])), "only once"),
    (lambda p: p["allocations"][0].update(supplier_id="SUP_NOPE"), "does not supply"),
    (lambda p: p["allocations"][0].update(unit_cost=0.01), "unit cost"),
    (lambda p: p.update(allocations=[]), "empty"),
])
def test_invalid_allocations_are_rejected_without_writing(client, plans, db_path, mutate, fragment):
    payload = _payload(plans["planned_split"]["planning"])
    mutate(payload)
    before = _po_count(db_path)
    res = client.post("/api/approve-order", json=payload)
    assert res.status_code == 422 and fragment in res.json()["detail"]
    assert _po_count(db_path) == before


def test_total_above_requirement_is_rejected(client, plans):
    plan = plans["planned_split"]["planning"]
    payload = _payload(plan)
    # Greedy fills earlier lines to capacity; the last line is the one with headroom.
    i = next(i for i, l in enumerate(plan["allocations"]) if l["available_qty"] > l["allocated_qty"])
    payload["allocations"][i]["quantity"] += 1
    res = client.post("/api/approve-order", json=payload)
    assert res.status_code == 422 and "exceeds the requirement" in res.json()["detail"]


def test_high_risk_supplier_is_rejected_at_approval(client, plans, db_path):
    plan = plans["planned_partial"]["planning"]
    risky = plan["allocation"]["excluded_high_risk"]
    assert risky, "the partial plan on real data is caused by High Risk exclusions"
    payload = _payload(plan)
    payload["allocations"].append({"supplier_id": risky[0]["supplier_id"], "quantity": 1})
    res = client.post("/api/approve-order", json=payload)
    assert res.status_code == 422 and "High Risk" in res.json()["detail"]


def test_a_failing_line_rolls_back_the_whole_order(client, plans, db_path):
    plan = plans["planned_split"]["planning"]
    second = plan["allocations"][1]["supplier_id"]
    with sqlite3.connect(db_path) as c:
        c.execute(f"""CREATE TRIGGER fail_second BEFORE INSERT ON purchase_orders
                      WHEN NEW.supplier_id = '{second}' BEGIN SELECT RAISE(ABORT, 'simulated failure'); END""")
    try:
        before = _po_count(db_path)
        res = client.post("/api/approve-order", json=_payload(plan))
        assert res.status_code == 500 and "rolled back" in res.json()["detail"]
        assert _po_count(db_path) == before
    finally:
        with sqlite3.connect(db_path) as c:
            c.execute("DROP TRIGGER fail_second")


def test_supplier_matrix_includes_the_split_allocation(client, plans):
    sku = plans["planned_split"]["sku_id"]
    body = client.get(f"/api/suppliers?sku_id={sku}").json()
    assert body["allocation_mode"] == "split"
    assert body["allocation"]["status"] == "full"
    assert body["allocation"]["total_allocated"] == body["required_quantity"]
    assert body["field_provenance"]["available_qty"] == "SIMULATED"


def test_forecast_endpoints(client):
    metrics = client.get("/api/forecast/metrics").json()
    assert metrics["method"] in ("model", "moving_average", "seasonal_naive")
    assert metrics["validation"]["train_period"][1] < metrics["validation"]["test_period"][0]
    body = client.get("/api/forecast?sku_id=SKU_1&horizon_days=7").json()
    assert len(body["lanes"]) == 5 and all(len(l["daily"]) == 7 for l in body["lanes"])
    assert client.get("/api/forecast?sku_id=SKU_999").status_code == 404
