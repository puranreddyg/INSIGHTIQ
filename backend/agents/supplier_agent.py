"""
Supplier Intelligence Agent — eligibility filter + heuristic ranking for one SKU.

Candidates are the suppliers that actually serve the SKU in the supply-chain
data (`sku_supplier_lanes`, REAL), joined to `supplier_sku_profile` (ESTIMATED
from real history by backend/ml/supplier_estimator.py — no random values). The
raw criteria per candidate form the candidate matrix:

  unit_cost        REAL       mean of the supplier's lane costs for this SKU
  lead_time_days   REAL       mean of the supplier's lane lead times for this SKU
  available_qty    ESTIMATED  demonstrated capacity: largest order delivered for this SKU
  defect_rate      ESTIMATED  empirical-Bayes: real KPI prior, moved by supplier evidence
  fulfillment_rate ESTIMATED  Beta-Binomial: real KPI prior, moved by supplier evidence

A supplier serves an SKU at 1–3 warehouses; when it serves several, cost and
lead time differ per warehouse and procurement requests have no warehouse, so
the arithmetic mean is used and min/max/lane count are kept alongside it.

Eligibility:  available_qty >= required_quantity. Ineligible suppliers are
reported with their shortfall and never scored or recommended.

Heuristic Rank Score over the eligible pool (0 = worst, 1 = best per component):

  QualityScore     = 1 - defect_rate            (bounded rate, used absolutely)
  CostScore        = 1 - MinMax(unit_cost)
  FulfillmentScore = fulfillment_rate           (bounded rate, used absolutely)
  DeliveryScore    = 1 - MinMax(lead_time_days)

  RawScore   = w1*Quality + w2*Cost + w4*Fulfillment + w3*Delivery   -> 0..1
  FinalScore = MinMax(RawScore) * 100                                 -> 0..100

Bounded rates are used absolutely — the precedent set by the former Rating/5
term — because Min-Max would stretch small differences into 0..1. A supplier
with no evidence of its own sits exactly on the prior, so quality and
fulfilment then add the same amount to every candidate and the ranking is
decided by the real cost and lead-time differences.

FinalScore is a *relative ranking* score. Risk is the Knowledge Agent's job,
which reads the ABSOLUTE attribute values carried alongside these scores.

`weights` is injectable so hard-coded weights can later be replaced by
optimised ones (e.g. a GA) without touching this module.
"""
from __future__ import annotations

import sqlite3
from typing import Mapping

import numpy as np
import pandas as pd

from backend.config import BUDGET_TOLERANCE, FIELD_PROVENANCE, HEURISTIC_WEIGHTS, TOP_N_SUPPLIER_CANDIDATES

# Component name -> weight key. Ordered for display.
COMPONENT_WEIGHTS = {
    "quality": "w1_quality",
    "cost": "w2_cost",
    "fulfillment": "w4_fulfillment",
    "delivery": "w3_delivery",
}

# Raw criteria the heuristic needs. A candidate missing any of them cannot be
# scored fairly, so it is excluded rather than silently imputed.
SCORING_FIELDS = ["unit_cost", "lead_time_days", "available_qty", "defect_rate", "fulfillment_rate"]

# Fields the Knowledge Agent compares against fixed thresholds. They must reach
# it at full precision — display rounding could silently flip a rule verdict.
RULE_INPUT_FIELDS = ["lead_time_days", "available_qty", "defect_rate", "fulfillment_rate"]

DISPLAY_DECIMALS = 3

OPTIONAL_PROFILE_DEFAULTS = {
    "fill_rate": 1.0, "on_time_prob": None, "expected_delay_days": 0.0, "delay_sd": 0.0,
    "evidence_pos": 0, "capacity_orders": 0, "evidence_source": None,
}


# ---------------------------------------------------------------------------
# Normalization primitives — collapse to a deterministic value when the pool
# cannot be discriminated on an axis (min == max, or a single candidate).
# ---------------------------------------------------------------------------

def _degenerate(lo: float, hi: float) -> bool:
    return not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-9


def _score_lower_better(series: pd.Series) -> pd.Series:
    """Min-Max normalize where a smaller raw value is better. 0 = worst, 1 = best."""
    lo, hi = series.min(), series.max()
    if _degenerate(lo, hi):
        return pd.Series(1.0, index=series.index, dtype=float)
    return 1.0 - (series - lo) / (hi - lo)


def _rescale_final(raw: pd.Series) -> pd.Series:
    lo, hi = raw.min(), raw.max()
    if _degenerate(lo, hi):
        return pd.Series(100.0, index=raw.index, dtype=float)
    return (raw - lo) / (hi - lo) * 100.0


# ---------------------------------------------------------------------------
# Candidate matrix (raw criteria) — reusable by other ranking methods
# ---------------------------------------------------------------------------

def candidate_matrix(conn: sqlite3.Connection, sku_id: str) -> pd.DataFrame:
    lanes = pd.read_sql_query(
        "SELECT supplier_id, warehouse_id, unit_cost, lead_time_days FROM sku_supplier_lanes WHERE sku_id = ?",
        conn, params=(sku_id,),
    )
    if lanes.empty:
        return pd.DataFrame(columns=["supplier_id", *SCORING_FIELDS])
    per_supplier = lanes.groupby("supplier_id").agg(
        unit_cost=("unit_cost", "mean"),
        unit_cost_min=("unit_cost", "min"),
        unit_cost_max=("unit_cost", "max"),
        lead_time_days=("lead_time_days", "mean"),
        lead_time_min=("lead_time_days", "min"),
        lead_time_max=("lead_time_days", "max"),
        lane_count=("warehouse_id", "nunique"),
        warehouses=("warehouse_id", lambda s: ", ".join(sorted(s))),
    ).reset_index()
    profile = pd.read_sql_query("SELECT * FROM supplier_sku_profile WHERE sku_id = ?", conn, params=(sku_id,))
    profile = profile.drop(columns=["sku_id"], errors="ignore")
    # Optional learned fields: a feed without delivery history means no known delay.
    for col, default in OPTIONAL_PROFILE_DEFAULTS.items():
        if col not in profile.columns:
            profile[col] = default
    matrix = per_supplier.merge(profile, on="supplier_id", how="left")
    # Delivery is judged on the LEARNED lead time: what the supplier promises plus the
    # delay the model predicts for it. The promise itself is kept for display.
    matrix["promised_lead_time_days"] = matrix["lead_time_days"]
    matrix["lead_time_days"] = matrix["promised_lead_time_days"] + matrix["expected_delay_days"].fillna(0.0)
    matrix.insert(0, "sku_id", sku_id)
    return matrix


def _coerce_scoring_inputs(pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Force the scoring columns numeric and quarantine unscorable candidates —
    anything non-numeric, null or infinite. Nothing is imputed."""
    df = pool.copy()
    for field in SCORING_FIELDS:
        if field not in df.columns:
            df[field] = np.nan
        df[field] = pd.to_numeric(df[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
    incomplete = df[SCORING_FIELDS].isna().any(axis=1)
    return df.loc[~incomplete].copy(), df.loc[incomplete, "supplier_id"].tolist()


def score_candidates(pool: pd.DataFrame, weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Attach component scores, Raw Score and Final Score to an eligible pool.
    Min/Max come from `pool` itself, so scores are relative to the suppliers
    actually in contention."""
    weights = dict(weights or HEURISTIC_WEIGHTS)
    df = pool.copy()
    components = {
        "quality": (1.0 - df["defect_rate"]).clip(0.0, 1.0),
        "cost": _score_lower_better(df["unit_cost"]),
        "fulfillment": df["fulfillment_rate"].clip(0.0, 1.0),
        "delivery": _score_lower_better(df["lead_time_days"]),
    }
    raw = pd.Series(0.0, index=df.index, dtype=float)
    for name, series in components.items():
        df[f"_component_{name}"] = series
        raw = raw + series * weights[COMPONENT_WEIGHTS[name]]
    df["raw_score"] = raw
    df["heuristic_score"] = _rescale_final(raw)
    return df


def _record(row: dict) -> dict:
    record: dict = {}
    for key, value in row.items():
        if isinstance(value, np.integer):
            record[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            value = float(value)
            record[key] = value if key in RULE_INPUT_FIELDS else round(value, DISPLAY_DECIMALS)
        else:
            record[key] = value
    if "available_qty" in record and record["available_qty"] is not None:
        record["available_qty"] = int(record["available_qty"])
    return record


def _candidate_records(scored: pd.DataFrame, weights: Mapping[str, float]) -> list[dict]:
    records: list[dict] = []
    for row in scored.to_dict(orient="records"):
        components = {name: float(row.pop(f"_component_{name}")) for name in COMPONENT_WEIGHTS}
        w = {name: weights[key] for name, key in COMPONENT_WEIGHTS.items()}
        contributions = {name: components[name] * w[name] for name in COMPONENT_WEIGHTS}
        raw_score = float(row.pop("raw_score"))
        final_score = float(row.pop("heuristic_score"))
        record = _record(row)
        record["heuristic_score"] = round(final_score, DISPLAY_DECIMALS)
        record["raw_score"] = round(raw_score, 6)
        record["scoring"] = {
            "weights": w,
            "components": {k: round(v, 6) for k, v in components.items()},
            "contributions": {k: round(v, 6) for k, v in contributions.items()},
            "raw_score": round(raw_score, 6),
            "final_score": round(final_score, DISPLAY_DECIMALS),
        }
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def order_budget(conn: sqlite3.Connection, sku_id: str, required_quantity: float, budget: float | None = None) -> dict:
    """Budget for this order: the caller's, or required qty x the SKU's reference unit
    cost (mean of its lane costs in the ledger) x (1 + BUDGET_TOLERANCE)."""
    row = conn.execute("SELECT AVG(unit_cost) FROM sku_supplier_lanes WHERE sku_id = ?", (sku_id,)).fetchone()
    ref = float(row[0]) if row and row[0] is not None else None
    if budget is not None:
        return {"budget": round(float(budget), 2), "source": "user", "reference_unit_cost": ref, "tolerance": None}
    if ref is None or not required_quantity:
        return {"budget": None, "source": "none", "reference_unit_cost": ref, "tolerance": BUDGET_TOLERANCE}
    return {"budget": round(required_quantity * ref * (1 + BUDGET_TOLERANCE), 2), "source": "default",
            "reference_unit_cost": round(ref, 4), "tolerance": BUDGET_TOLERANCE}


def rank_suppliers(
    conn: sqlite3.Connection,
    sku_id: str,
    required_quantity: float = 0,
    top_n: int | None = TOP_N_SUPPLIER_CANDIDATES,
    weights: Mapping[str, float] | None = None,
    budget: float | None = None,
) -> dict:
    weights = dict(weights or HEURISTIC_WEIGHTS)
    matrix = candidate_matrix(conn, sku_id)
    budget_info = order_budget(conn, sku_id, required_quantity, budget)
    scorable, unscorable = _coerce_scoring_inputs(matrix)

    eligible = scorable[scorable["available_qty"] >= required_quantity]
    ineligible = scorable[scorable["available_qty"] < required_quantity].sort_values("available_qty", ascending=False)

    if eligible.empty:
        candidates: list[dict] = []
    else:
        scored = score_candidates(eligible, weights).sort_values("raw_score", ascending=False).reset_index(drop=True)
        scored["rank"] = scored.index + 1
        candidates = _candidate_records(scored if top_n is None else scored.head(top_n), weights)

    ineligible_records = [
        {**_record(r), "shortfall": int(required_quantity - r["available_qty"])}
        for r in ineligible.to_dict(orient="records")
    ]
    combined = int(scorable["available_qty"].sum()) if not scorable.empty else 0
    no_eligible = bool(len(scorable)) and eligible.empty and required_quantity > 0
    split_possible = no_eligible and combined >= required_quantity
    # Even the whole pool together falls short: still plan what CAN be bought
    # (a partial order) instead of giving up, and report the shortfall.
    partial_possible = no_eligible and 0 < combined < required_quantity

    # When no single supplier can cover the request, the whole scorable pool is
    # ranked (scores relative to that pool) so the Knowledge Agent can allocate
    # the order across suppliers. `ranked_candidates` stays empty: nobody is
    # recommended as a *sole* supplier.
    pool_candidates: list[dict] = []
    if not scorable.empty:
        pool_scored = score_candidates(scorable, weights).sort_values("raw_score", ascending=False).reset_index(drop=True)
        pool_scored["rank"] = pool_scored.index + 1
        pool_candidates = _candidate_records(pool_scored, weights)
    split_candidates = pool_candidates if (split_possible or partial_possible) else []

    return {
        "agent": "Supplier Intelligence Agent",
        "sku_id": sku_id,
        "required_quantity": required_quantity,
        "pool_size": int(len(matrix)),
        "eligible_count": int(len(eligible)),
        "filtered_out_suppliers": ineligible["supplier_id"].tolist(),
        "ineligible_suppliers": ineligible_records,
        "unscorable_suppliers": unscorable,
        "no_eligible_supplier": no_eligible,
        "combined_available_qty": combined,
        # True when no single supplier can cover the request but the pool combined can.
        # The Knowledge Agent then allocates the order across `split_candidates`.
        "split_order_possible": split_possible,
        # True when even the combined pool cannot cover it; a partial split is planned.
        "partial_order_possible": partial_possible,
        "shortfall_qty": int(max(0, required_quantity - combined)) if partial_possible else 0,
        "split_candidates": split_candidates,
        # the whole scorable pool, ranked against itself — input to the multi-supplier search
        "pool_candidates": pool_candidates,
        "ranked_candidates": candidates,
        "primary_supplier": candidates[0] if candidates else None,
        "backup_supplier": candidates[1] if len(candidates) > 1 else None,
        "weights": {name: weights[key] for name, key in COMPONENT_WEIGHTS.items()},
        "field_provenance": FIELD_PROVENANCE,
        "budget": budget_info["budget"],
        "budget_info": budget_info,
    }


def full_matrix(conn: sqlite3.Connection, sku_id: str) -> dict:
    """Every candidate for the SKU, ranked with no quantity requirement."""
    return rank_suppliers(conn, sku_id, required_quantity=0, top_n=None)


def run(conn: sqlite3.Connection, sku_id: str, required_quantity: float = 0, budget: float | None = None) -> dict:
    return rank_suppliers(conn, sku_id, required_quantity=required_quantity, budget=budget)
