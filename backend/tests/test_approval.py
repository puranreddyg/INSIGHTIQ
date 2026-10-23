"""
Purchase-order approval: the server re-derives cost, capacity, requirement and risk
(never trusts the browser), refuses silent partial / High Risk orders, and saves every
line of an order in one transaction.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend import config, database


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
def runs(client):
    """Pipeline result per urgent SKU, keyed by the Knowledge Agent's decision."""
    out: dict[str, dict] = {}
    for sku in ("SKU_44", "SKU_21", "SKU_36", "SKU_37", "SKU_50"):
        body = client.post(f"/api/run-agent-pipeline?sku_id={sku}").json()
        out.setdefault(body["knowledge"]["decision"], body)
    return out


def _lines(plan):
    if plan.get("split_order"):
        return [{"supplier_id": l["supplier_id"], "quantity": l["quantity"], "unit_cost": l["unit_cost"],
                 "estimated_arrival": l["estimated_arrival_date"]} for l in plan["allocations"]]
    return [{"supplier_id": plan["primary_supplier_id"], "quantity": plan["order_quantity"],
             "unit_cost": plan["unit_cost"], "estimated_arrival": plan["estimated_arrival_date"]}]


def _post(client, body, **flags):
    return client.post("/api/approve-split-order", json={"sku_id": body["sku_id"], "allocations": _lines(body["planning"]), **flags})


def _po_count(db_path):
    with sqlite3.connect(db_path) as c:
        return c.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0]


def test_real_data_covers_single_split_and_flagged_plans(runs):
    assert {"single", "split", "single_flagged"} <= set(runs)


def test_single_order_is_saved_with_the_plans_total(client, runs):
    body = runs["single"]
    plan = body["planning"]
    res = _post(client, body)
    assert res.status_code == 200, res.text
    po = res.json()["purchase_orders"][0]
    assert po["total_cost"] == pytest.approx(plan["total_expense"], abs=0.011)
    assert po["order_group"] and po["is_partial"] == 0 and po["quantity"] == plan["order_quantity"]


def test_split_order_lines_share_a_group_and_match_the_plan(client, runs):
    body = runs["split"]
    plan = body["planning"]
    res = _post(client, body, acknowledge_risk=True)
    assert res.status_code == 200, res.text
    data = res.json()
    assert len({r["order_group"] for r in data["purchase_orders"]}) == 1
    assert data["total_quantity"] == plan["order_quantity"]
    assert data["total_cost"] == pytest.approx(plan["total_expense"], abs=0.05)


def test_a_client_cannot_change_the_unit_cost(client, runs, db_path):
    body = runs["single"]
    lines = _lines(body["planning"])
    lines[0]["unit_cost"] = 0.01
    before = _po_count(db_path)
    res = client.post("/api/approve-split-order", json={"sku_id": body["sku_id"], "allocations": lines})
    assert res.status_code == 422 and "unit cost" in res.json()["detail"]
    assert _po_count(db_path) == before


def test_order_above_the_requirement_is_rejected(client, runs, db_path):
    body = runs["split"]
    lines = _lines(body["planning"])
    i = next(i for i, l in enumerate(body["planning"]["allocations"]) if l["available_qty"] > l["quantity"])
    lines[i]["quantity"] += 1
    before = _po_count(db_path)
    res = client.post("/api/approve-split-order", json={"sku_id": body["sku_id"], "allocations": lines, "acknowledge_risk": True})
    assert res.status_code == 422 and "exceeds the current requirement" in res.json()["detail"]
    assert _po_count(db_path) == before


def test_partial_order_must_be_marked_partial(client, runs, db_path):
    body = runs["single"]
    lines = _lines(body["planning"])
    lines[0]["quantity"] -= 10
    payload = {"sku_id": body["sku_id"], "allocations": lines}
    before = _po_count(db_path)
    res = client.post("/api/approve-split-order", json=payload)
    assert res.status_code == 422 and "is_partial" in res.json()["detail"]
    assert _po_count(db_path) == before
    ok = client.post("/api/approve-split-order", json={**payload, "is_partial": True})
    assert ok.status_code == 200 and ok.json()["purchase_orders"][0]["is_partial"] == 1


def test_sku_without_a_requirement_cannot_be_ordered(client):
    res = client.post("/api/approve-order", json={"sku_id": "SKU_3", "supplier_id": "SUP_1", "quantity": 10, "unit_cost": 5.0})
    assert res.status_code == 422 and "no current procurement requirement" in res.json()["detail"]


def test_high_risk_supplier_needs_an_explicit_acknowledgement(client, runs, db_path):
    body = runs["split"]
    assert any(l["supplier_status"] == "High Risk" for l in body["planning"]["allocations"])
    before = _po_count(db_path)
    res = _post(client, body)
    assert res.status_code == 422 and "High Risk" in res.json()["detail"] and "acknowledge_risk" in res.json()["detail"]
    assert _po_count(db_path) == before
    assert _post(client, body, acknowledge_risk=True).status_code == 200


def test_flagged_single_plan_also_needs_acknowledgement(client, runs):
    body = runs["single_flagged"]
    assert body["planning"]["quality_constraint_met"] is False
    assert _post(client, body).status_code == 422
    assert _post(client, body, acknowledge_risk=True).status_code == 200


def test_a_failing_line_rolls_back_the_whole_order(client, runs, db_path):
    body = runs["split"]
    second = body["planning"]["allocations"][1]["supplier_id"]
    with sqlite3.connect(db_path) as c:
        c.execute(f"""CREATE TRIGGER fail_second BEFORE INSERT ON purchase_orders
                      WHEN NEW.supplier_id = '{second}' BEGIN SELECT RAISE(ABORT, 'simulated failure'); END""")
    try:
        before = _po_count(db_path)
        res = _post(client, body, acknowledge_risk=True)
        assert res.status_code == 500 and "rolled back" in res.json()["detail"]
        assert _po_count(db_path) == before
    finally:
        with sqlite3.connect(db_path) as c:
            c.execute("DROP TRIGGER fail_second")


def test_matrix_shows_the_split_plan_with_arrival_dates(client, runs):
    body = runs["split"]
    data = client.get(f"/api/suppliers?sku_id={body['sku_id']}").json()
    preview = data["allocation_preview"]
    assert data["decision"] == "split" and preview["order_quantity"] == body["planning"]["order_quantity"]
    assert [l["supplier_id"] for l in preview["allocations"]] == [l["supplier_id"] for l in body["planning"]["allocations"]]
    assert all(l["estimated_arrival_date"] for l in preview["allocations"])
    single = client.get(f"/api/suppliers?sku_id={runs['single']['sku_id']}").json()
    assert "allocation_preview" not in single


def test_sku_that_needs_nothing_is_a_no_action_decision(client):
    body = client.post("/api/run-agent-pipeline?sku_id=SKU_3").json()
    assert body["knowledge"]["decision"] == "no_action" and body["planning"]["status"] == "no_action"
    assert "planner or forecast reorder point" in body["planning"]["error"]


def test_purchase_order_columns_are_added_to_an_older_table(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "insightiq.db")
    database.build_database(force=True)
    with sqlite3.connect(config.DB_PATH) as c:
        c.execute("ALTER TABLE purchase_orders DROP COLUMN order_group")
        c.execute("ALTER TABLE purchase_orders DROP COLUMN is_partial")
    database.build_database(force=False)  # same schema version: must still repair the table
    with sqlite3.connect(config.DB_PATH) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(purchase_orders)")}
    assert {"order_group", "is_partial"} <= cols
