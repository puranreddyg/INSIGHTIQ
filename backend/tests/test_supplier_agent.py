"""
Tests for the Supplier Intelligence Agent's weighted Min-Max heuristic.

Every component is direction-corrected so 0 = worst and 1 = best, then combined
as 0.30*Rating + 0.25*Cost + 0.25*Reliability + 0.20*Delivery. These tests pin
the direction of each component, the deterministic degenerate-pool behaviour,
and the boundary with the Knowledge Agent (which must keep seeing ABSOLUTE,
full-precision attribute values).
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from backend.agents import knowledge_agent, supplier_agent
from backend.config import HEURISTIC_WEIGHTS


def supplier(sid, *, rating=4.0, cost=10.0, delivery=5.0, reliability=0.80, defect=0.01, stock=10_000):
    return {
        "supplier_id": sid,
        "rating": rating,
        "avg_cost": cost,
        "delivery_time_days": delivery,
        "reliability_score": reliability,
        "defect_rate": defect,
        "stock_availability": stock,
    }


def make_conn(rows):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    pd.DataFrame(rows).to_sql("supplier_profiles", conn, index=False)
    return conn


def rank(rows, deficit=0):
    return supplier_agent.rank_suppliers(make_conn(rows), required_deficit=deficit)


def by_id(result):
    return {c["supplier_id"]: c for c in result["ranked_candidates"]}


def components(result, sid):
    return by_id(result)[sid]["scoring"]["components"]


# Spread across cost, delivery and reliability so every axis discriminates.
SPREAD = [
    supplier("CHEAP", cost=5.0, delivery=2.0, reliability=0.95, rating=5.0),
    supplier("MID", cost=10.0, delivery=5.0, reliability=0.80, rating=4.0),
    supplier("PRICEY", cost=20.0, delivery=9.0, reliability=0.60, rating=3.0),
]


# ---------------------------------------------------------------------------
# 1-7: component direction — 0 = worst, 1 = best on every axis
# ---------------------------------------------------------------------------

def test_cheapest_supplier_scores_one_on_cost():
    assert components(rank(SPREAD), "CHEAP")["cost"] == pytest.approx(1.0)


def test_most_expensive_supplier_scores_zero_on_cost():
    assert components(rank(SPREAD), "PRICEY")["cost"] == pytest.approx(0.0)


def test_most_reliable_supplier_scores_one():
    assert components(rank(SPREAD), "CHEAP")["reliability"] == pytest.approx(1.0)


def test_least_reliable_supplier_scores_zero():
    assert components(rank(SPREAD), "PRICEY")["reliability"] == pytest.approx(0.0)


def test_fastest_supplier_scores_one_on_delivery():
    assert components(rank(SPREAD), "CHEAP")["delivery"] == pytest.approx(1.0)


def test_slowest_supplier_scores_zero_on_delivery():
    assert components(rank(SPREAD), "PRICEY")["delivery"] == pytest.approx(0.0)


def test_rating_is_absolute_fifths_not_min_maxed():
    """Rating is already 0-5, so it is divided by 5 — never Min-Max normalized.
    The lowest-rated supplier must NOT collapse to 0."""
    result = rank(SPREAD)
    assert components(result, "CHEAP")["rating"] == pytest.approx(1.0)   # 5.0 / 5
    assert components(result, "MID")["rating"] == pytest.approx(0.8)     # 4.0 / 5
    assert components(result, "PRICEY")["rating"] == pytest.approx(0.6)  # 3.0 / 5


# ---------------------------------------------------------------------------
# 8: the canonical worked example (matches the approved plan, section I)
# ---------------------------------------------------------------------------

CANONICAL = [
    supplier("A", rating=4.5, cost=10.0, delivery=2.0, reliability=0.80),
    supplier("B", rating=4.0, cost=15.0, delivery=4.0, reliability=0.90),
    supplier("C", rating=5.0, cost=12.0, delivery=3.0, reliability=0.95),
]


def test_canonical_example_component_scores():
    result = rank(CANONICAL)
    assert components(result, "A") == pytest.approx({"rating": 0.9, "cost": 1.0, "reliability": 0.0, "delivery": 1.0})
    assert components(result, "B") == pytest.approx({"rating": 0.8, "cost": 0.0, "reliability": 2 / 3, "delivery": 0.0})
    assert components(result, "C") == pytest.approx({"rating": 1.0, "cost": 0.6, "reliability": 1.0, "delivery": 0.5})


def test_canonical_example_raw_and_final_scores():
    scored = by_id(rank(CANONICAL))
    assert scored["A"]["scoring"]["raw_score"] == pytest.approx(0.720000, abs=1e-6)
    assert scored["B"]["scoring"]["raw_score"] == pytest.approx(0.406667, abs=1e-6)
    assert scored["C"]["scoring"]["raw_score"] == pytest.approx(0.800000, abs=1e-6)

    assert scored["C"]["scoring"]["final_score"] == pytest.approx(100.0, abs=0.01)
    assert scored["A"]["scoring"]["final_score"] == pytest.approx(79.66, abs=0.01)
    assert scored["B"]["scoring"]["final_score"] == pytest.approx(0.0, abs=0.01)


def test_canonical_example_ranking_order():
    """C wins despite A being both cheapest and fastest — the point of removing
    the old reciprocal formula, which inflated those two terms ~7.7x."""
    assert [c["supplier_id"] for c in rank(CANONICAL)["ranked_candidates"]] == ["C", "A", "B"]


# ---------------------------------------------------------------------------
# 9-11: an axis that cannot discriminate must not penalise anybody
# ---------------------------------------------------------------------------

def test_equal_costs_give_every_supplier_a_full_cost_score():
    result = rank([supplier("X", cost=7.0, rating=5.0), supplier("Y", cost=7.0, rating=3.0)])
    assert components(result, "X")["cost"] == pytest.approx(1.0)
    assert components(result, "Y")["cost"] == pytest.approx(1.0)


def test_equal_reliability_gives_every_supplier_a_full_reliability_score():
    result = rank([supplier("X", reliability=0.7, rating=5.0), supplier("Y", reliability=0.7, rating=3.0)])
    assert components(result, "X")["reliability"] == pytest.approx(1.0)
    assert components(result, "Y")["reliability"] == pytest.approx(1.0)


def test_equal_delivery_gives_every_supplier_a_full_delivery_score():
    result = rank([supplier("X", delivery=4.0, rating=5.0), supplier("Y", delivery=4.0, rating=3.0)])
    assert components(result, "X")["delivery"] == pytest.approx(1.0)
    assert components(result, "Y")["delivery"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 12-13: degenerate pools rescale to 100, never divide by zero
# ---------------------------------------------------------------------------

def test_single_eligible_supplier_scores_one_hundred():
    result = rank(
        [supplier("BIG", stock=50_000, rating=4.0), supplier("SMALL", stock=10)],
        deficit=20_000,
    )
    assert result["eligible_count"] == 1
    assert result["filtered_out_suppliers"] == ["SMALL"]
    only = result["ranked_candidates"][0]
    assert only["supplier_id"] == "BIG"
    assert only["rank"] == 1
    assert only["scoring"]["final_score"] == pytest.approx(100.0)
    # Min/Max axes cannot discriminate a pool of one, so they yield 1.0 — but
    # rating stays absolute (4.0 / 5), because it is never Min-Max normalized.
    assert only["scoring"]["components"] == pytest.approx(
        {"rating": 0.8, "cost": 1.0, "reliability": 1.0, "delivery": 1.0}
    )


def test_identical_suppliers_all_score_one_hundred():
    result = rank([supplier("X"), supplier("Y"), supplier("Z")])
    finals = [c["scoring"]["final_score"] for c in result["ranked_candidates"]]
    assert finals == pytest.approx([100.0, 100.0, 100.0])


# ---------------------------------------------------------------------------
# 14: missing / invalid data is quarantined, never imputed or leaked as NaN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [None, float("inf"), float("-inf")])
def test_supplier_with_unusable_input_is_excluded_not_imputed(bad_value):
    result = rank([supplier("GOOD"), supplier("BAD", cost=bad_value)])
    assert result["unscorable_suppliers"] == ["BAD"]
    assert [c["supplier_id"] for c in result["ranked_candidates"]] == ["GOOD"]


def test_no_nan_reaches_any_raw_score():
    result = rank([supplier("GOOD"), supplier("BAD", rating=None), supplier("ALSO_BAD", reliability=None)])
    assert sorted(result["unscorable_suppliers"]) == ["ALSO_BAD", "BAD"]
    for candidate in result["ranked_candidates"]:
        raw = candidate["scoring"]["raw_score"]
        assert raw == raw  # NaN != NaN
        assert 0.0 <= raw <= 1.0


# ---------------------------------------------------------------------------
# 15-16: heuristic ranks, the Knowledge Agent vetoes — High Risk wins
# ---------------------------------------------------------------------------

HIGH_RISK_POOL = [
    # Best on every heuristic axis, but its defect rate trips Rule 3.
    supplier("TOP", rating=5.0, cost=5.0, delivery=1.0, reliability=1.0, defect=0.09),
    supplier("SAFE", rating=4.0, cost=10.0, delivery=5.0, reliability=0.70, defect=0.01),
]


def test_high_risk_supplier_still_ranks_first_on_the_heuristic():
    """Risk must not leak into the heuristic — ranking is purely the score."""
    assert rank(HIGH_RISK_POOL)["ranked_candidates"][0]["supplier_id"] == "TOP"


def test_recommendation_skips_the_high_risk_leader_for_the_next_candidate():
    supplier_result = rank(HIGH_RISK_POOL)
    result = knowledge_agent.run({"focus_facts": []}, supplier_result)
    statuses = {c["supplier_id"]: c["supplier_status"] for c in result["annotated_candidates"]}
    assert statuses["TOP"] == "High Risk"
    assert result["recommended_supplier"]["supplier_id"] == "SAFE"


# ---------------------------------------------------------------------------
# 17-19: invariants of the weighted sum
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    assert sum(HEURISTIC_WEIGHTS.values()) == pytest.approx(1.0)


def test_cost_and_reliability_are_exactly_half_the_score():
    assert HEURISTIC_WEIGHTS["w2_cost"] + HEURISTIC_WEIGHTS["w4_reliability"] == pytest.approx(0.50)


def test_raw_score_is_bounded_between_zero_and_one():
    for candidate in rank(SPREAD)["ranked_candidates"]:
        assert 0.0 <= candidate["scoring"]["raw_score"] <= 1.0


def test_contributions_sum_to_the_raw_score():
    for candidate in rank(SPREAD)["ranked_candidates"]:
        scoring = candidate["scoring"]
        assert sum(scoring["contributions"].values()) == pytest.approx(scoring["raw_score"], abs=1e-6)


def test_each_contribution_is_component_times_weight():
    for candidate in rank(SPREAD)["ranked_candidates"]:
        scoring = candidate["scoring"]
        for name, contribution in scoring["contributions"].items():
            expected = scoring["components"][name] * scoring["weights"][name]
            assert contribution == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 20: eligibility filtering and the split-order fallback
# ---------------------------------------------------------------------------

def test_ineligible_suppliers_are_filtered_before_scoring():
    result = rank([supplier("BIG", stock=9_000), supplier("SMALL", stock=100)], deficit=5_000)
    assert result["filtered_out_suppliers"] == ["SMALL"]
    assert [c["supplier_id"] for c in result["ranked_candidates"]] == ["BIG"]
    assert result["split_order_fallback"] is False


def test_min_max_ignores_filtered_out_suppliers():
    """An excluded supplier must not stretch the Min/Max of the eligible pool."""
    result = rank(
        [supplier("A", cost=10.0, stock=9_000), supplier("B", cost=12.0, stock=9_000),
         supplier("CHEAP_BUT_TINY", cost=1.0, stock=10)],
        deficit=5_000,
    )
    # A is cheapest among the *eligible* two, so it takes the full cost score
    # despite the excluded supplier being far cheaper.
    assert components(result, "A")["cost"] == pytest.approx(1.0)
    assert components(result, "B")["cost"] == pytest.approx(0.0)


def test_no_eligible_supplier_falls_back_to_the_full_pool():
    """Nobody can cover the deficit alone — rank everyone anyway so a partial /
    split-order plan is still possible downstream."""
    result = rank([supplier("A", stock=100), supplier("B", stock=200)], deficit=999_999)
    assert result["split_order_fallback"] is True
    assert len(result["ranked_candidates"]) == 2


# ---------------------------------------------------------------------------
# Knowledge Agent boundary: absolute, full-precision values must survive
# ---------------------------------------------------------------------------

def test_rule_input_fields_are_not_rounded_for_display():
    """Display rounding must never reach a Rule threshold. A defect rate of
    0.0502 is genuinely above Rule 3's 0.05 bar; rounding it to 0.050 would
    silently flip the verdict."""
    candidate = by_id(rank([supplier("X", defect=0.0502, reliability=0.8723)]))["X"]
    assert candidate["defect_rate"] == 0.0502
    assert candidate["reliability_score"] == 0.8723


def test_borderline_defect_rate_still_triggers_high_risk():
    supplier_result = rank([supplier("X", defect=0.0502)])
    result = knowledge_agent.run({"focus_facts": []}, supplier_result)
    assert result["annotated_candidates"][0]["supplier_status"] == "High Risk"


def test_normalized_scores_never_overwrite_absolute_attributes():
    """Rule 4 compares reliability against an absolute 0.90. If the normalized
    (relative) value overwrote it, the rule would silently break."""
    result = rank([supplier("LOW", reliability=0.60), supplier("HIGH", reliability=0.95)])
    low = by_id(result)["LOW"]
    assert low["reliability_score"] == pytest.approx(0.60)          # absolute, untouched
    assert low["scoring"]["components"]["reliability"] == pytest.approx(0.0)  # relative


# ---------------------------------------------------------------------------
# Regression: both API paths must expose the same scoring payload
# ---------------------------------------------------------------------------

def test_full_matrix_and_ranked_path_expose_identical_scoring_keys():
    """The deficit-filtered path used to omit the component-score fields
    entirely, leaving the breakdown chart empty."""
    conn = make_conn(SPREAD)
    matrix = supplier_agent.full_matrix(conn)
    filtered = supplier_agent.rank_suppliers(conn, required_deficit=1_000)
    assert set(matrix["ranked_candidates"][0]) == set(filtered["ranked_candidates"][0])
    for candidate in matrix["ranked_candidates"] + filtered["ranked_candidates"]:
        assert set(candidate["scoring"]) == {"weights", "components", "contributions", "raw_score", "final_score"}


def test_full_matrix_returns_the_whole_pool_not_just_top_n():
    conn = make_conn([supplier(f"S{i}", cost=float(i + 1)) for i in range(8)])
    assert len(supplier_agent.full_matrix(conn)["ranked_candidates"]) == 8
    assert len(supplier_agent.rank_suppliers(conn)["ranked_candidates"]) == 5
