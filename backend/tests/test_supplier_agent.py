"""
Tests for the Supplier Intelligence Agent: eligibility filter + weighted heuristic.

Raw = 0.30*Quality + 0.25*Cost + 0.25*Fulfillment + 0.20*Delivery, where Cost and
Delivery are direction-corrected Min-Max (0 = worst, 1 = best) and Quality
(1 - defect_rate) and Fulfillment are bounded rates used absolutely. These tests
pin those directions, the eligibility rule, the degenerate-pool behaviour and
the boundary with the Knowledge Agent (which must see ABSOLUTE values).
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from backend.agents import knowledge_agent, supplier_agent
from backend.config import HEURISTIC_WEIGHTS

SKU = "SKU_T"


def supplier(sid, *, cost=10.0, lead=5.0, defect=0.01, fulfillment=0.80, avail=1_000, lanes=None, evidence=20):
    """One candidate. `lanes` = [(warehouse, cost, lead), ...] overrides a single lane.
    `evidence` = purchase orders of the supplier's own history (rules need >= 5)."""
    lanes = lanes or [("WH_1", cost, lead)]
    return {
        "lanes": [{"sku_id": SKU, "warehouse_id": wh, "supplier_id": sid, "unit_cost": c, "lead_time_days": lt} for wh, c, lt in lanes],
        "profile": {"sku_id": SKU, "supplier_id": sid, "available_qty": avail, "defect_rate": defect, "fulfillment_rate": fulfillment,
                    "defect_sd": 0.01, "evidence_pos": evidence, "capacity_orders": 20, "evidence_source": "test"},
    }


def make_conn(rows):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    pd.DataFrame([lane for r in rows for lane in r["lanes"]]).to_sql("sku_supplier_lanes", conn, index=False)
    pd.DataFrame([r["profile"] for r in rows]).to_sql("supplier_sku_profile", conn, index=False)
    return conn


def rank(rows, required=0, **kw):
    return supplier_agent.rank_suppliers(make_conn(rows), SKU, required_quantity=required, **kw)


def by_id(result):
    return {c["supplier_id"]: c for c in result["ranked_candidates"]}


def components(result, sid):
    return by_id(result)[sid]["scoring"]["components"]


SPREAD = [
    supplier("CHEAP", cost=5.0, lead=2.0, defect=0.01, fulfillment=0.95),
    supplier("MID", cost=10.0, lead=5.0, defect=0.03, fulfillment=0.80),
    supplier("PRICEY", cost=20.0, lead=9.0, defect=0.10, fulfillment=0.60),
]


# ---------------------------------------------------------------------------
# Component direction
# ---------------------------------------------------------------------------

def test_cheapest_supplier_scores_one_on_cost():
    assert components(rank(SPREAD), "CHEAP")["cost"] == pytest.approx(1.0)


def test_most_expensive_supplier_scores_zero_on_cost():
    assert components(rank(SPREAD), "PRICEY")["cost"] == pytest.approx(0.0)


def test_fastest_supplier_scores_one_on_delivery():
    assert components(rank(SPREAD), "CHEAP")["delivery"] == pytest.approx(1.0)


def test_slowest_supplier_scores_zero_on_delivery():
    assert components(rank(SPREAD), "PRICEY")["delivery"] == pytest.approx(0.0)


def test_quality_is_one_minus_defect_rate_not_min_maxed():
    """A bounded rate is used absolutely, so the worst supplier does not collapse to 0."""
    result = rank(SPREAD)
    assert components(result, "CHEAP")["quality"] == pytest.approx(0.99)
    assert components(result, "PRICEY")["quality"] == pytest.approx(0.90)


def test_fulfillment_is_used_absolutely_not_min_maxed():
    result = rank(SPREAD)
    assert components(result, "CHEAP")["fulfillment"] == pytest.approx(0.95)
    assert components(result, "PRICEY")["fulfillment"] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Worked example
# ---------------------------------------------------------------------------

CANONICAL = [
    supplier("A", cost=10.0, lead=2.0, defect=0.02, fulfillment=0.80),
    supplier("B", cost=15.0, lead=4.0, defect=0.03, fulfillment=0.90),
    supplier("C", cost=12.0, lead=3.0, defect=0.01, fulfillment=0.85),
]


def test_canonical_example_component_scores():
    result = rank(CANONICAL)
    assert components(result, "A") == pytest.approx({"quality": 0.98, "cost": 1.0, "fulfillment": 0.80, "delivery": 1.0})
    assert components(result, "B") == pytest.approx({"quality": 0.97, "cost": 0.0, "fulfillment": 0.90, "delivery": 0.0})
    assert components(result, "C") == pytest.approx({"quality": 0.99, "cost": 0.6, "fulfillment": 0.85, "delivery": 0.5})


def test_canonical_example_raw_and_final_scores():
    scored = by_id(rank(CANONICAL))
    assert scored["A"]["scoring"]["raw_score"] == pytest.approx(0.944, abs=1e-6)
    assert scored["B"]["scoring"]["raw_score"] == pytest.approx(0.516, abs=1e-6)
    assert scored["C"]["scoring"]["raw_score"] == pytest.approx(0.7595, abs=1e-6)
    assert scored["A"]["scoring"]["final_score"] == pytest.approx(100.0, abs=0.01)
    assert scored["C"]["scoring"]["final_score"] == pytest.approx(56.89, abs=0.01)
    assert scored["B"]["scoring"]["final_score"] == pytest.approx(0.0, abs=0.01)


def test_canonical_example_ranking_order():
    assert [c["supplier_id"] for c in rank(CANONICAL)["ranked_candidates"]] == ["A", "C", "B"]


# ---------------------------------------------------------------------------
# Degenerate pools
# ---------------------------------------------------------------------------

def test_equal_costs_give_every_supplier_a_full_cost_score():
    result = rank([supplier("X", cost=7.0, defect=0.01), supplier("Y", cost=7.0, defect=0.04)])
    assert components(result, "X")["cost"] == pytest.approx(1.0)
    assert components(result, "Y")["cost"] == pytest.approx(1.0)


def test_equal_lead_times_give_every_supplier_a_full_delivery_score():
    result = rank([supplier("X", lead=4.0, defect=0.01), supplier("Y", lead=4.0, defect=0.04)])
    assert components(result, "X")["delivery"] == pytest.approx(1.0)
    assert components(result, "Y")["delivery"] == pytest.approx(1.0)


def test_single_eligible_supplier_scores_one_hundred():
    result = rank([supplier("BIG", avail=500), supplier("SMALL", avail=10)], required=400)
    assert result["eligible_count"] == 1
    assert result["filtered_out_suppliers"] == ["SMALL"]
    only = result["ranked_candidates"][0]
    assert (only["supplier_id"], only["rank"]) == ("BIG", 1)
    assert only["scoring"]["final_score"] == pytest.approx(100.0)


def test_identical_suppliers_all_score_one_hundred():
    finals = [c["scoring"]["final_score"] for c in rank([supplier("X"), supplier("Y"), supplier("Z")])["ranked_candidates"]]
    assert finals == pytest.approx([100.0, 100.0, 100.0])


# ---------------------------------------------------------------------------
# Missing / invalid data is quarantined, never imputed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [None, float("inf"), float("-inf")])
def test_candidate_with_unusable_input_is_excluded_not_imputed(bad_value):
    result = rank([supplier("GOOD"), supplier("BAD", cost=bad_value)])
    assert result["unscorable_suppliers"] == ["BAD"]
    assert [c["supplier_id"] for c in result["ranked_candidates"]] == ["GOOD"]


def test_supplier_with_lane_but_no_profile_row_is_unscorable():
    conn = make_conn([supplier("GOOD")])
    pd.DataFrame([{"sku_id": SKU, "warehouse_id": "WH_2", "supplier_id": "NOPROFILE", "unit_cost": 5.0, "lead_time_days": 2}]).to_sql(
        "sku_supplier_lanes", conn, index=False, if_exists="append")
    result = supplier_agent.rank_suppliers(conn, SKU)
    assert result["unscorable_suppliers"] == ["NOPROFILE"]


# ---------------------------------------------------------------------------
# Eligibility: available_qty >= required_quantity
# ---------------------------------------------------------------------------

def test_ineligible_suppliers_are_filtered_before_scoring():
    result = rank([supplier("BIG", avail=450), supplier("SMALL", avail=300)], required=351)
    assert result["filtered_out_suppliers"] == ["SMALL"]
    assert [c["supplier_id"] for c in result["ranked_candidates"]] == ["BIG"]
    assert result["ineligible_suppliers"][0]["shortfall"] == 51


def test_available_equal_to_required_is_eligible():
    assert rank([supplier("EXACT", avail=350)], required=350)["eligible_count"] == 1


def test_min_max_uses_the_whole_scorable_pool_not_only_full_order_suppliers():
    """Workflow step 1: EVERY scorable supplier is scored together, so a supplier short of the full
    order still takes part in the Min/Max and appears in the complete ranking (labelled, not hidden)."""
    result = rank(
        [supplier("A", cost=10.0, avail=900), supplier("B", cost=12.0, avail=900),
         supplier("CHEAP_BUT_SHORT", cost=1.0, avail=10)],
        required=500,
    )
    ranking = {c["supplier_id"]: c for c in result["scored_ranking"]}
    assert set(ranking) == {"A", "B", "CHEAP_BUT_SHORT"}
    assert ranking["CHEAP_BUT_SHORT"]["scoring"]["components"]["cost"] == pytest.approx(1.0)
    assert ranking["B"]["scoring"]["components"]["cost"] == pytest.approx(0.0)
    assert ranking["CHEAP_BUT_SHORT"]["eligibility_status"] == "Insufficient capacity for full order"
    assert ranking["A"]["eligibility_status"] == "Eligible for full order"


def test_complete_ranking_is_sorted_by_score_not_by_capacity_and_eligible_keeps_original_rank():
    rows = [supplier("BEST_SHORT", cost=5.0, lead=2.0, defect=0.01, fulfillment=0.95, avail=100),
            supplier("MID_BIG", cost=10.0, lead=5.0, defect=0.03, fulfillment=0.80, avail=900),
            supplier("WORST_BIG", cost=20.0, lead=9.0, defect=0.10, fulfillment=0.60, avail=2_000)]
    r = rank(rows, required=500)
    assert [c["supplier_id"] for c in r["scored_ranking"]] == ["BEST_SHORT", "MID_BIG", "WORST_BIG"]
    assert [c["rank"] for c in r["scored_ranking"]] == [1, 2, 3]
    scores = [c["heuristic_score"] for c in r["scored_ranking"]]
    assert scores == sorted(scores, reverse=True)
    # eligible list: same order, original ranks kept (BEST_SHORT is rank 1 overall but cannot supply 500)
    assert [(c["supplier_id"], c["rank"], c["eligible_position"]) for c in r["eligible_single"]] == [("MID_BIG", 2, 1), ("WORST_BIG", 3, 2)]
    assert r["ranked_candidates"] == r["eligible_single"]
    assert [c["supplier_id"] for c in r["ineligible_suppliers"]] == ["BEST_SHORT"]


def test_non_scorable_suppliers_are_listed_with_a_reason_and_never_scored():
    good = supplier("GOOD")
    bad = supplier("BAD"); bad["profile"]["defect_rate"] = None
    r = rank([good, bad], required=100)
    assert [x["supplier_id"] for x in r["non_scorable"]] == ["BAD"]
    assert "defect_rate" in r["non_scorable"][0]["missing_fields"]
    assert r["non_scorable"][0]["status"] == "Not scorable"
    assert "BAD" not in {c["supplier_id"] for c in r["scored_ranking"]}


def test_every_eligible_supplier_is_evaluated_not_only_the_top_five():
    rows = [supplier(f"S{i}", cost=5.0 + i, lead=2.0 + i * 0.1) for i in range(8)]
    r = rank(rows, required=100)
    assert len(r["ranked_candidates"]) == 8


def test_single_supplier_evaluation_is_chronological_and_stops_at_first_pass():
    """High Risk #1 is rejected with its reason, #2 (over budget) is rejected, #3 is selected;
    #4 is only a labelled backup evaluation."""
    rows = [supplier("RISKY", cost=5.0, lead=2.0, defect=0.09, fulfillment=0.95),
            supplier("PRICEY", cost=30.0, lead=2.5, defect=0.01, fulfillment=0.95),
            supplier("GOOD", cost=8.0, lead=3.0, defect=0.02, fulfillment=0.90),
            supplier("LAST", cost=9.0, lead=6.0, defect=0.02, fulfillment=0.70)]
    sup = rank(rows, required=100, budget=1_500.0)
    ev = knowledge_agent.run({"focus_facts": []}, sup)["single_supplier_evaluation"]
    order = [(e["supplier_id"], e["outcome"]) for e in ev["evaluations"]]
    sel = [e for e in ev["evaluations"] if e["phase"] == "selection"]
    assert sel[-1]["outcome"] == "SELECTED" and ev["selected_supplier_id"] == sel[-1]["supplier_id"]
    assert all(e["outcome"] == "REJECTED" for e in sel[:-1])
    assert [e["step"] for e in ev["evaluations"]] == list(range(1, len(order) + 1))
    # supplier order is the heuristic order
    assert [e["supplier_id"] for e in ev["evaluations"]] == [c["supplier_id"] for c in sup["eligible_single"]]
    risky = next(e for e in ev["evaluations"] if e["supplier_id"] == "RISKY")
    assert risky["status"] == "High Risk" and any("High Risk" in r for r in risky["reasons"])
    pricey = next(e for e in ev["evaluations"] if e["supplier_id"] == "PRICEY")
    assert any("Over budget" in r for r in pricey["reasons"])
    # backups come only after the selection and are labelled
    after = ev["evaluations"][len(sel):]
    assert all(e["phase"] == "backup_evaluation" for e in after)


def test_each_rule_trace_entry_follows_its_own_supplier_fact():
    """R7 must be written next to the supplier it was evaluated for (it used to be appended after all suppliers)."""
    rows = [supplier("A", cost=5.0, defect=0.09), supplier("B", cost=6.0, defect=0.01)]
    k = knowledge_agent.run({"focus_facts": []}, rank(rows, required=100, budget=1_000.0))
    current = None
    for e in k["reasoning_trace"]:
        if e["type"] == "fact" and e["entityKind"] == "supplier":
            current = e["entity"]
        elif e["type"] == "rule" and e["ruleId"] in (2, 3, 4, 5, 7) and e.get("phase") in ("selection", "backup_evaluation"):
            assert e["entity"] == current


def test_case_a_when_no_supplier_has_the_capacity():
    rows = [supplier("A", avail=100), supplier("B", avail=150)]
    k = knowledge_agent.run({"focus_facts": []}, rank(rows, required=200))
    ev = k["single_supplier_evaluation"]
    assert ev["case"] == "A" and ev["eligible_list"] == []
    assert ev["eligible_empty_message"] == "No supplier has sufficient capacity to fulfil the complete order individually."
    assert ev["headline"].startswith("NO SINGLE-SUPPLIER SOLUTION")
    assert ev["transition_message"] == "Starting multi-supplier allocation because no valid single-supplier solution was found."
    assert k["search_trace"]["events"][0]["type"] == "search_started"


def test_case_b_when_capacity_exists_but_every_supplier_is_rejected():
    rows = [supplier("R1", defect=0.09), supplier("R2", defect=0.08, cost=11.0)]
    k = knowledge_agent.run({"focus_facts": []}, rank(rows, required=100))
    ev = k["single_supplier_evaluation"]
    assert ev["case"] == "B" and ev["headline"].startswith("NO ACCEPTABLE SINGLE-SUPPLIER SOLUTION")
    assert len(ev["reasons"]) == 2 and all("High Risk" in r for r in ev["reasons"])
    assert ev["transition_message"].startswith("Starting multi-supplier allocation")


def test_no_eligible_supplier_is_never_named_a_sole_supplier_but_the_order_is_split():
    """No supplier is ranked as a *sole* supplier; the Knowledge Agent splits the order instead."""
    result = rank([supplier("A", avail=300), supplier("B", avail=250)], required=500)
    assert result["no_eligible_supplier"] is True
    assert result["ranked_candidates"] == [] and result["primary_supplier"] is None
    assert result["combined_available_qty"] == 550
    assert result["split_order_possible"] is True
    assert {c["supplier_id"] for c in result["split_candidates"]} == {"A", "B"}
    knowledge = knowledge_agent.run({"focus_facts": []}, result)
    assert knowledge["split_order"] is True
    allocations = {a["supplier_id"]: a["allocated_qty"] for a in knowledge["allocations"]}
    assert sum(allocations.values()) == 500 and all(v <= {"A": 300, "B": 250}[k] for k, v in allocations.items())
    assert knowledge["ineligible_candidates"] == []


def test_full_split_not_possible_when_combined_availability_is_short():
    result = rank([supplier("A", avail=100), supplier("B", avail=100)], required=500)
    assert result["no_eligible_supplier"] is True and result["split_order_possible"] is False


# ---------------------------------------------------------------------------
# Warehouse lanes: mean is used, min/max and lane count are kept
# ---------------------------------------------------------------------------

def test_multi_lane_supplier_uses_mean_and_keeps_range():
    result = rank([supplier("M", lanes=[("WH_1", 11.46, 8), ("WH_3", 19.41, 4)])])
    c = by_id(result)["M"]
    assert c["unit_cost"] == pytest.approx(15.435, abs=1e-3)
    assert (c["unit_cost_min"], c["unit_cost_max"], c["lane_count"]) == (11.46, 19.41, 2)
    assert c["lead_time_days"] == pytest.approx(6.0)
    assert c["warehouses"] == "WH_1, WH_3"


# ---------------------------------------------------------------------------
# Heuristic ranks, the Knowledge Agent vetoes — High Risk wins
# ---------------------------------------------------------------------------

HIGH_RISK_POOL = [
    supplier("TOP", cost=5.0, lead=1.0, defect=0.09, fulfillment=0.95),
    supplier("SAFE", cost=10.0, lead=5.0, defect=0.01, fulfillment=0.70),
]


def test_high_risk_supplier_still_ranks_first_on_the_heuristic():
    assert rank(HIGH_RISK_POOL)["ranked_candidates"][0]["supplier_id"] == "TOP"


def test_recommendation_skips_the_high_risk_leader():
    knowledge = knowledge_agent.run({"focus_facts": []}, rank(HIGH_RISK_POOL))
    statuses = {c["supplier_id"]: c["supplier_status"] for c in knowledge["annotated_candidates"]}
    assert statuses["TOP"] == "High Risk"
    assert knowledge["recommended_supplier"]["supplier_id"] == "SAFE"


# ---------------------------------------------------------------------------
# Weighted-sum invariants and injectable weights
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    assert sum(HEURISTIC_WEIGHTS.values()) == pytest.approx(1.0)


def test_weight_values_are_unchanged_from_the_product_based_version():
    assert HEURISTIC_WEIGHTS == {"w1_quality": 0.30, "w2_cost": 0.25, "w3_delivery": 0.20, "w4_fulfillment": 0.25}


def test_raw_score_is_bounded_and_equals_sum_of_contributions():
    for c in rank(SPREAD)["ranked_candidates"]:
        s = c["scoring"]
        assert 0.0 <= s["raw_score"] <= 1.0
        # each of the 4 contributions is rounded to 6 dp, so allow 4 × 0.5e-6
        assert sum(s["contributions"].values()) == pytest.approx(s["raw_score"], abs=2e-6)
        for name, contribution in s["contributions"].items():
            assert contribution == pytest.approx(s["components"][name] * s["weights"][name], abs=1e-6)


def test_injected_weights_change_the_ranking_without_code_changes():
    """The GA / optimiser hook: weights come in as data."""
    cheap_slow = supplier("CHEAP_SLOW", cost=5.0, lead=14.0)
    fast_pricey = supplier("FAST_PRICEY", cost=20.0, lead=2.0)
    cost_heavy = {"w1_quality": 0.0, "w2_cost": 1.0, "w3_delivery": 0.0, "w4_fulfillment": 0.0}
    delivery_heavy = {"w1_quality": 0.0, "w2_cost": 0.0, "w3_delivery": 1.0, "w4_fulfillment": 0.0}
    assert rank([cheap_slow, fast_pricey], weights=cost_heavy)["ranked_candidates"][0]["supplier_id"] == "CHEAP_SLOW"
    assert rank([cheap_slow, fast_pricey], weights=delivery_heavy)["ranked_candidates"][0]["supplier_id"] == "FAST_PRICEY"


def test_score_candidates_works_on_a_plain_matrix():
    """AHP/TOPSIS/GA can reuse the same raw criteria without a database."""
    matrix = pd.DataFrame([
        {"supplier_id": "P", "unit_cost": 5.0, "lead_time_days": 2.0, "available_qty": 400, "defect_rate": 0.02, "fulfillment_rate": 0.8},
        {"supplier_id": "Q", "unit_cost": 9.0, "lead_time_days": 6.0, "available_qty": 300, "defect_rate": 0.04, "fulfillment_rate": 0.8},
    ])
    scored = supplier_agent.score_candidates(matrix)
    assert list(scored.sort_values("raw_score", ascending=False).supplier_id) == ["P", "Q"]


# ---------------------------------------------------------------------------
# Raw criteria preserved at full precision for the Knowledge Agent / future methods
# ---------------------------------------------------------------------------

def test_every_candidate_carries_all_raw_criteria():
    for c in rank(SPREAD)["ranked_candidates"]:
        assert {"unit_cost", "lead_time_days", "available_qty", "defect_rate", "fulfillment_rate"} <= set(c)


def test_rule_input_fields_are_not_rounded_for_display():
    c = by_id(rank([supplier("X", defect=0.05021, fulfillment=0.87234)]))["X"]
    assert c["defect_rate"] == 0.05021
    assert c["fulfillment_rate"] == 0.87234


def test_borderline_defect_rate_still_triggers_high_risk():
    knowledge = knowledge_agent.run({"focus_facts": []}, rank([supplier("X", defect=0.0502)]))
    assert knowledge["annotated_candidates"][0]["supplier_status"] == "High Risk"


def test_full_matrix_and_required_path_expose_identical_scoring_keys():
    conn = make_conn(SPREAD)
    matrix = supplier_agent.full_matrix(conn, SKU)
    filtered = supplier_agent.rank_suppliers(conn, SKU, required_quantity=100)
    assert set(matrix["ranked_candidates"][0]) == set(filtered["ranked_candidates"][0])
    for c in matrix["ranked_candidates"] + filtered["ranked_candidates"]:
        assert set(c["scoring"]) == {"weights", "components", "contributions", "raw_score", "final_score"}


def test_estimated_fields_are_labelled_in_the_response():
    prov = rank(SPREAD)["field_provenance"]
    assert prov["available_qty"] == "ESTIMATED"
    assert prov["defect_rate"] == prov["fulfillment_rate"] == prov["lead_time_days"] == "LEARNED"
    assert prov["unit_cost"] == prov["promised_lead_time_days"] == "REAL"


def test_a_supplier_without_its_own_evidence_is_never_labelled_high_risk():
    """Quality rules fire on a supplier's own history only, never on the prior alone."""
    knowledge = knowledge_agent.run({"focus_facts": []}, rank([supplier("NEW", defect=0.09, evidence=0)]))
    assert knowledge["annotated_candidates"][0]["supplier_status"] == "Unrated"


def test_partial_order_when_even_the_whole_pool_is_short():
    result = rank([supplier("A", avail=100), supplier("B", avail=150)], required=500)
    assert result["partial_order_possible"] is True and result["shortfall_qty"] == 250
    knowledge = knowledge_agent.run({"focus_facts": []}, result)
    assert knowledge["partial_order"] is True and knowledge["shortfall_qty"] == 250
    assert sorted(a["allocated_qty"] for a in knowledge["allocations"]) == [100, 150]
