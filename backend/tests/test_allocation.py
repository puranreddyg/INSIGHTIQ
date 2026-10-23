"""
Split-order allocation: single-supplier path kept, split when no acceptable single
supplier exists, High Risk never auto-allocated, partial plans explicit, and the
quantity invariants (≤ requirement, ≤ each supplier's availability) always hold.
"""
from __future__ import annotations

import random
import sqlite3

import pandas as pd
import pytest

from backend.agents import allocation, knowledge_agent, planning_agent, supplier_agent

SKU = "SKU_T"
SAFE = 0.01   # defect rate below the High Risk threshold
RISKY = 0.09  # above it


def supplier(sid, *, cost=10.0, lead=5.0, defect=SAFE, fulfillment=0.80, avail=100):
    return {
        "lane": {"sku_id": SKU, "warehouse_id": "WH_1", "supplier_id": sid, "unit_cost": cost, "lead_time_days": lead},
        "profile": {"sku_id": SKU, "supplier_id": sid, "available_qty": avail, "defect_rate": defect, "fulfillment_rate": fulfillment},
    }


def make_conn(rows, profiles=True):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    pd.DataFrame([r["lane"] for r in rows]).to_sql("sku_supplier_lanes", conn, index=False)
    pd.DataFrame([r["profile"] for r in rows if profiles]
                 or [{"sku_id": SKU, "supplier_id": "_", "available_qty": 0, "defect_rate": 0.0, "fulfillment_rate": 0.0}]
                 ).to_sql("supplier_sku_profile", conn, index=False)
    pd.DataFrame({"season": ["Winter", "Summer"], "is_weekend": [False, True]}).to_sql("calendar", conn, index=False)
    return conn


SKU_FACT = {"sku_id": SKU, "SKU": SKU, "Urgent": True, "Inventory": 50, "DailyDemand": 10, "focus_scope": "urgent lanes",
            "required_quantity_rule": "test"}


def pipeline(rows, required, conn=None):
    conn = conn or make_conn(rows)
    sup = supplier_agent.rank_suppliers(conn, SKU, required_quantity=required)
    know = knowledge_agent.run({"focus_facts": []}, sup)
    plan = planning_agent.run(conn, SKU_FACT, know, required, sup)
    return sup, know, plan


def assert_invariants(plan, sup, required):
    avail = {c["supplier_id"]: c["available_qty"] for c in sup["allocation_pool"]}
    lines = plan.get("allocations", [])
    assert sum(l["allocated_qty"] for l in lines) <= required
    assert all(0 < l["allocated_qty"] <= avail[l["supplier_id"]] for l in lines)
    assert len({l["supplier_id"] for l in lines}) == len(lines)
    if lines:
        assert plan["order_quantity"] == sum(l["allocated_qty"] for l in lines)
        assert plan["total_expense"] == pytest.approx(sum(l["line_cost"] for l in lines), abs=0.011)
        for l in lines:
            assert l["line_cost"] == pytest.approx(l["allocated_qty"] * l["unit_cost"], abs=0.006)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def test_single_supplier_path_is_kept_when_one_supplier_suffices():
    sup, know, plan = pipeline([supplier("BIG", avail=500), supplier("SMALL", avail=100)], 300)
    assert know["allocation_mode"] == "single"
    assert plan["status"] == "planned" and plan["allocation_mode"] == "single"
    assert plan["primary_supplier_id"] == "BIG" and plan["order_quantity"] == 300
    assert [(l["supplier_id"], l["allocated_qty"]) for l in plan["allocations"]] == [("BIG", 300)]
    # legacy fields still present for the existing frontend
    for key in ("unit_cost", "total_expense", "estimated_arrival_date", "backup_supplier_id", "replenishment_curve"):
        assert key in plan
    assert_invariants(plan, sup, 300)


def test_split_order_covers_requirement_when_no_single_supplier_can():
    rows = [supplier("A", avail=200, cost=10, lead=3), supplier("B", avail=150, cost=12, lead=6), supplier("C", avail=120, cost=11, lead=9)]
    sup, know, plan = pipeline(rows, 400)
    assert sup["eligible_count"] == 0 and know["recommended_supplier"] is None
    assert plan["status"] == "planned_split" and plan["is_partial"] is False and plan["residual_shortage"] == 0
    assert plan["order_quantity"] == 400
    assert len(plan["allocations"]) >= 2
    for line in plan["allocations"]:
        for key in ("supplier_id", "allocated_qty", "unit_cost", "estimated_arrival", "risk_status", "line_cost"):
            assert key in line
    # different lead times -> per-line arrival dates; the plan arrives with the last one
    assert plan["estimated_arrival_date"] == max(l["estimated_arrival"] for l in plan["allocations"])
    assert_invariants(plan, sup, 400)


def test_split_order_fills_in_heuristic_rank_order():
    rows = [supplier("A", avail=200, cost=10, lead=3), supplier("B", avail=150, cost=20, lead=9), supplier("C", avail=120, cost=11, lead=4)]
    sup, _, plan = pipeline(rows, 300)
    pool_order = [c["supplier_id"] for c in sup["allocation_pool"]]
    assert pool_order[0] == "A"
    assert [l["supplier_id"] for l in plan["allocations"]] == pool_order[:2]
    assert [l["allocated_qty"] for l in plan["allocations"]] == [200, 100]


def test_insufficient_combined_availability_gives_an_explicit_partial_plan():
    sup, _, plan = pipeline([supplier("A", avail=100), supplier("B", avail=80)], 500)
    assert sup["split_order_possible"] is False
    assert plan["status"] == "planned_partial" and plan["is_partial"] is True
    assert plan["order_quantity"] == 180 and plan["residual_shortage"] == 320
    assert plan["allocation"]["max_feasible_quantity"] == 180
    assert any("below the required" in r for r in plan["allocation"]["shortage_reasons"])
    assert any("PARTIAL PLAN" in r for r in plan["reasoning"])
    assert_invariants(plan, sup, 500)


def test_high_risk_supplier_is_not_used_when_acceptable_suppliers_cover_it():
    rows = [supplier("RISKY", avail=300, cost=1, lead=1, defect=RISKY), supplier("A", avail=200), supplier("B", avail=150)]
    sup, _, plan = pipeline(rows, 320)
    assert plan["status"] == "planned_split"
    assert "RISKY" not in {l["supplier_id"] for l in plan["allocations"]}
    assert {l["risk_status"] for l in plan["allocations"]} <= {"Preferred", "Reliable", "Standard"}
    assert plan["allocation"]["excluded_high_risk"][0]["supplier_id"] == "RISKY"


def test_high_risk_sole_source_is_replaced_by_an_acceptable_split():
    """The only supplier able to cover it alone is High Risk — previously it was
    picked anyway; now acceptable suppliers split the order."""
    rows = [supplier("RISKY", avail=600, defect=RISKY), supplier("A", avail=200), supplier("B", avail=200)]
    sup, know, plan = pipeline(rows, 400)
    assert sup["eligible_count"] == 1 and know["allocation_mode"] == "split"
    assert plan["status"] == "planned_split"
    assert {l["supplier_id"] for l in plan["allocations"]} == {"A", "B"}


def test_risk_constraint_shortage_is_reported_not_bypassed():
    rows = [supplier("RISKY", avail=500, defect=RISKY), supplier("A", avail=100)]
    _, _, plan = pipeline(rows, 300)
    assert plan["status"] == "planned_partial" and plan["residual_shortage"] == 200
    assert plan["allocation"]["max_feasible_quantity_ignoring_risk"] == 300
    assert any("RISKY" in r and "Rule 3" in r for r in plan["allocation"]["shortage_reasons"])


def test_only_high_risk_stock_produces_no_plan():
    _, _, plan = pipeline([supplier("RISKY", avail=100, defect=RISKY)], 300)
    assert plan["status"] == "no_eligible_supplier" and "order_quantity" not in plan
    assert plan["residual_shortage"] == 300


def test_zero_required_quantity_plans_nothing():
    sup, know, plan = pipeline([supplier("A", avail=100)], 0)
    assert sup["allocation_pool"] == [] and know["allocation_mode"] is None
    assert plan["status"] == "no_action"
    assert allocation.allocate([{"supplier_id": "A", "available_qty": 10, "unit_cost": 1, "lead_time_days": 1}], 0)["allocations"] == []


def test_missing_supplier_profiles_are_never_allocated():
    rows = [supplier("A", avail=200), supplier("B", avail=200)]
    conn = make_conn(rows)
    conn.execute("DELETE FROM supplier_sku_profile WHERE supplier_id = 'B'")
    sup, _, plan = pipeline(rows, 300, conn=conn)
    assert sup["unscorable_suppliers"] == ["B"]
    assert {l["supplier_id"] for l in plan["allocations"]} == {"A"}
    assert plan["is_partial"] and plan["residual_shortage"] == 100


def test_no_profiles_at_all_gives_no_plan():
    _, _, plan = pipeline([supplier("A", avail=200)], 300, conn=make_conn([supplier("A")], profiles=False))
    assert plan["status"] == "no_eligible_supplier"


def test_suppliers_with_zero_stock_are_skipped():
    _, _, plan = pipeline([supplier("EMPTY", avail=0, cost=1), supplier("A", avail=200), supplier("B", avail=200)], 300)
    assert "EMPTY" not in {l["supplier_id"] for l in plan["allocations"]}


def test_allocation_is_deterministic():
    rows = [supplier(s, avail=a, cost=c) for s, a, c in [("X", 100, 10), ("Y", 100, 10), ("Z", 100, 10)]]
    first = pipeline(rows, 250)[2]["allocations"]
    for _ in range(3):
        again = pipeline(list(reversed(rows)), 250)[2]["allocations"]
        assert [(l["supplier_id"], l["allocated_qty"]) for l in again] == [(l["supplier_id"], l["allocated_qty"]) for l in first]


@pytest.mark.parametrize("seed", range(25))
def test_allocation_never_exceeds_requirement_or_availability(seed):
    rng = random.Random(seed)
    rows = [supplier(f"S{i}", avail=rng.randint(0, 400), cost=rng.uniform(5, 30), lead=rng.randint(2, 14),
                     defect=rng.choice([SAFE, SAFE, RISKY]), fulfillment=rng.uniform(0.6, 1.0)) for i in range(rng.randint(1, 6))]
    required = rng.randint(1, 1500)
    sup, know, plan = pipeline(rows, required)
    assert_invariants(plan, sup, required)
    risky = {r["profile"]["supplier_id"] for r in rows if r["profile"]["defect_rate"] == RISKY}
    assert not risky & {l["supplier_id"] for l in plan.get("allocations", [])}
    if plan.get("allocations"):
        assert plan["order_quantity"] + plan["residual_shortage"] == required


def test_partial_suppliers_still_get_full_risk_reasoning_in_the_trace():
    _, know, _ = pipeline([supplier("A", avail=100, defect=RISKY), supplier("B", avail=100)], 500)
    rules_for_a = {t["ruleId"] for t in know["reasoning_trace"] if t.get("type") == "rule" and t["entity"] == "A"}
    assert {5, 3} <= rules_for_a
    r5 = next(t for t in know["reasoning_trace"] if t.get("ruleId") == 5 and t["entity"] == "A")
    assert r5["outcome"] == "Partial Capacity"
