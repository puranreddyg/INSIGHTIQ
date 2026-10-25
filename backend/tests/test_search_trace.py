"""The Multi-Supplier Search Trace is recorded INSIDE procurement_search while it runs.
These tests check it is internally consistent (counters = events), honest about what was
never evaluated, in real execution order, and that recording it did not change any result."""
from __future__ import annotations

import random

import pytest

from backend.agents import procurement_search as ps
from backend.config import RULE_HIGH_RISK_DEFECT_RATE, SEARCH_MAX_SUPPLIERS


def cand(i, avail, defect, cost, score, fulfil=0.9):
    return {"supplier_id": f"S{i}", "available_qty": avail, "defect_rate": defect, "unit_cost": cost, "fulfillment_rate": fulfil,
            "evidence_pos": 20, "heuristic_score": score, "rank": i}


POOL = [cand(1, 120, 0.02, 10, 90), cand(2, 150, 0.03, 9, 80), cand(3, 90, 0.04, 8, 70), cand(4, 200, 0.12, 7, 60)]


def types(res):
    return [e["type"] for e in res["trace"]["events"]]


def test_steps_are_consecutive_from_one_and_the_search_starts_and_completes():
    res = ps.search(POOL, 300, {"S4"}, None)
    ev = res["trace"]["events"]
    assert [e["step"] for e in ev] == list(range(1, len(ev) + 1))
    assert ev[0]["type"] == "search_started" and ev[-1]["type"] == "search_completed"
    assert types(res).count("search_completed") == 1


def test_summary_counters_equal_the_event_counts():
    for budget in (None, 2_900.0, 1_000.0):
        res = ps.search(POOL, 300, {"S4"}, budget)
        t, c = types(res), res["trace"]["summary"]
        assert c["nodes_created"] == t.count("node_created")
        assert c["nodes_expanded"] == t.count("node_expanded") == res["nodes_expanded"]
        assert c["branches_pruned"] == t.count("branch_pruned") == res["nodes_pruned"]
        assert c["combinations_evaluated"] == t.count("combination_evaluated") == res["plans_evaluated"]
        assert c["feasible_allocations"] == t.count("feasible_allocation")
        assert c["solver_invocations"] == t.count("solver_invoked")
        assert c["combinations_rejected"] == t.count("combination_rejected")


def test_unevaluated_constraints_are_not_reported_as_passed():
    res = ps.search(POOL, 400, set(), None)
    short = [e for e in res["trace"]["events"] if e["type"] == "combination_rejected" and "capacity" in e["reason"]]
    assert short, "a single supplier cannot cover 400 here, so capacity rejections must exist"
    for e in short:
        assert e["capacity"]["status"] == "FAIL"
        assert e["quality"]["status"] == ps.NOT_EVALUATED and e["budget"]["status"] == ps.NOT_EVALUATED
        assert e["min_allocation"]["status"] == ps.NOT_EVALUATED


def test_combinations_are_canonical_and_unique_per_generation():
    res = ps.search(POOL, 300, set(), None)
    keys = [e["combination"]["key"] for e in res["trace"]["events"] if e["type"] == "node_created" and e["combination"]["size"]]
    assert len(keys) == len(set(keys))
    for k in keys:
        ids = k.split(" + ")
        assert ids == sorted(ids, key=lambda x: int(x[1:]))  # canonical: heuristic order, never permutations


def test_final_allocation_matches_the_returned_plan_and_objective_values_are_recorded():
    res = ps.search(POOL, 300, {"S4"}, None)
    fin = res["trace"]["final"]
    assert {l["supplier_id"]: l["allocated_qty"] for l in fin["allocation"]["lines"]} == {c["supplier_id"]: q for c, q in res["shares"]}
    assert fin["allocation"]["total_quantity"] == 300 and fin["constraints_passed"]["total_equals_required"]
    assert fin["constraints_passed"]["quality"] and fin["constraints_passed"]["max_suppliers"]
    assert fin["objective"]["priority_2_suppliers_used"] == len(res["shares"])
    best_events = [e for e in res["trace"]["events"] if e["type"] == "new_best"]
    assert best_events and best_events[-1]["allocation"] == fin["allocation"]


def test_pruning_events_name_the_rule_and_only_happen_after_a_risk_free_incumbent():
    res = ps.search(POOL, 300, {"S4"}, None)
    ev = res["trace"]["events"]
    first_best = next(i for i, e in enumerate(ev) if e["type"] == "new_best" and e["objective"]["priority_1_high_risk_suppliers"] == 0)
    for i, e in enumerate(ev):
        if e["type"] == "branch_pruned":
            assert i > first_best and e["pruning"]["rule"] in ("larger_than_incumbent", "bound_cannot_beat_incumbent")


def test_infeasible_and_partial_searches_also_carry_a_trace():
    bad = [cand(1, 200, 0.2, 10, 90), cand(2, 200, 0.2, 9, 80)]
    inf = ps.search(bad, 300, set(), None)
    assert inf["status"] == "infeasible" and inf["trace"]["final"] is None
    assert inf["trace"]["summary"]["result_type"] == "no feasible allocation"
    assert any(e["type"] == "combination_rejected" and e["quality"]["status"] == "FAIL" for e in inf["trace"]["events"])
    part = ps.partial(POOL, 1_000, {"S4"}, None)
    assert part["status"] == "partial" and part["trace"]["mode"] == "partial_milp"
    assert part["trace"]["final"]["shortfall"] == part["shortfall"]


def test_budget_rejection_is_attributed_to_the_budget():
    res = ps.search(POOL[:3], 300, set(), 100.0)
    assert res["status"] == "infeasible"
    rej = [e for e in res["trace"]["events"] if e["type"] == "combination_rejected" and e["budget"]["status"] == "FAIL"]
    assert rej and all("budget" in e["reason"] for e in rej)


@pytest.mark.parametrize("seed", range(40))
def test_instrumentation_does_not_change_the_result_vs_exhaustive_enumeration(seed):
    from backend.tests._bruteforce import best_key
    rng = random.Random(seed)
    pool = [cand(i + 1, rng.randint(40, 400), round(rng.uniform(0.005, 0.12), 4), round(rng.uniform(5, 20), 2), round(rng.uniform(0, 100), 2))
            for i in range(rng.randint(2, 6))]
    R = rng.randint(60, 700)
    budget = rng.choice([None, R * rng.uniform(6, 20)])
    risky = {c["supplier_id"] for c in pool if c["defect_rate"] > RULE_HIGH_RISK_DEFECT_RATE}
    if sum(c["available_qty"] for c in pool) < R:
        return
    res = ps.search(pool, R, risky, budget)
    exp = best_key(pool, R, budget)
    if exp is None:
        assert res["status"] == "infeasible"
    else:
        assert res["status"] == "full"
        assert (res["risky"], len(res["shares"])) == (exp[0], exp[1])
        assert res["trace"]["final"]["objective"]["priority_2_suppliers_used"] <= SEARCH_MAX_SUPPLIERS
