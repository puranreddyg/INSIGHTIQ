"""
Supplier Intelligence Agent — heuristic search & ranking over the unified
supplier pool (backend/database.py::_build_supplier_profiles), which blends:
  - live logistics telemetry from supply_chain_dataset1.csv (cost, lead time,
    current stock, delivery consistency)
  - historical quality benchmarks from Procurement KPI Analysis Dataset.csv
    (rating, defect rate, cost variance)

Heuristic Rank Score. Every component is direction-corrected Min-Max normalized
over the *eligible* pool so that 0 = worst and 1 = best, then combined with the
configured business weights:

  RatingScore      = Rating / 5                (already bounded 0..1, not re-normalized)
  CostScore        = 1 - MinMax(Cost)          (cheaper       -> higher score)
  ReliabilityScore =     MinMax(Reliability)   (more reliable -> higher score)
  DeliveryScore    = 1 - MinMax(DeliveryTime)  (faster        -> higher score)

  RawScore   = 0.30*Rating + 0.25*Cost + 0.25*Reliability + 0.20*Delivery  -> 0..1
  FinalScore = MinMax(RawScore) * 100                                      -> 0..100

Cost + Reliability therefore contribute exactly 50% of the Raw Score.

FinalScore is a *relative ranking* score — it says how a supplier compares with
the rest of the eligible pool, nothing more. It is not a safety score, a risk
probability, or a success likelihood. Risk is the Knowledge Agent's job
(Rules 2/3/4), which reads the ABSOLUTE attribute values carried alongside
these scores — never the normalized ones.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from backend.config import HEURISTIC_WEIGHTS, TOP_N_SUPPLIER_CANDIDATES

RATING_MAX = 5.0

# Component name -> weight key in config.HEURISTIC_WEIGHTS. Ordered for display.
COMPONENT_WEIGHTS = {
    "rating": "w1_rating",
    "cost": "w2_cost",
    "reliability": "w4_reliability",
    "delivery": "w3_delivery",
}

# Raw attributes the heuristic scores on. A supplier missing any of them cannot
# be scored fairly, so it is excluded rather than silently imputed.
SCORING_FIELDS = ["rating", "avg_cost", "reliability_score", "delivery_time_days"]

# Attributes the Knowledge Agent evaluates as ABSOLUTE values against fixed
# thresholds (Rule 2 rating/delivery, Rule 3 defect rate, Rule 4 reliability).
# These must reach it at full precision — rounding them for display would
# silently flip a rule verdict (e.g. defect 0.0502 -> 0.050 escapes "> 0.05").
RULE_INPUT_FIELDS = ["rating", "delivery_time_days", "defect_rate", "reliability_score"]

DISPLAY_DECIMALS = 3


# ---------------------------------------------------------------------------
# Normalization primitives
#
# All three collapse to a deterministic value when the pool cannot be
# discriminated on that axis (min == max, or a single supplier). Components
# fall back to 1.0 so nobody is penalised on an axis that carries no
# information; the final rescale falls back to 100 because no supplier can be
# ranked above another.
# ---------------------------------------------------------------------------

def _degenerate(lo: float, hi: float) -> bool:
    return not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-9


def _score_higher_better(series: pd.Series) -> pd.Series:
    """Min-Max normalize where a larger raw value is better. 0 = worst, 1 = best."""
    lo, hi = series.min(), series.max()
    if _degenerate(lo, hi):
        return pd.Series(1.0, index=series.index, dtype=float)
    return (series - lo) / (hi - lo)


def _score_lower_better(series: pd.Series) -> pd.Series:
    """Min-Max normalize where a smaller raw value is better. 0 = worst, 1 = best."""
    lo, hi = series.min(), series.max()
    if _degenerate(lo, hi):
        return pd.Series(1.0, index=series.index, dtype=float)
    return 1.0 - (series - lo) / (hi - lo)


def _rescale_final(raw: pd.Series) -> pd.Series:
    """Rescale Raw Scores to the UI-friendly 0-100 relative ranking score."""
    lo, hi = raw.min(), raw.max()
    if _degenerate(lo, hi):
        return pd.Series(100.0, index=raw.index, dtype=float)
    return (raw - lo) / (hi - lo) * 100.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def load_pool(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM supplier_profiles", conn)


def _coerce_scoring_inputs(pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Force the scoring columns numeric and quarantine unscorable suppliers.

    Anything non-numeric, null or infinite becomes NaN; a supplier missing any
    scoring input is split off so NaN/inf can never reach a Raw Score. We do
    not invent replacement values — an unscorable supplier is reported, not
    guessed at."""
    df = pool.copy()
    for field in SCORING_FIELDS:
        if field not in df.columns:
            df[field] = np.nan
        df[field] = pd.to_numeric(df[field], errors="coerce").replace([np.inf, -np.inf], np.nan)

    incomplete = df[SCORING_FIELDS].isna().any(axis=1)
    unscorable = df.loc[incomplete, "supplier_id"].tolist()
    return df.loc[~incomplete].copy(), unscorable


def _score_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Attach component scores, Raw Score and Final Score to an eligible pool.

    Min/Max are taken from `pool` itself, so scores are always relative to the
    suppliers actually in contention — never to ones filtered out earlier."""
    df = pool.copy()

    components = {
        "rating": (df["rating"] / RATING_MAX).clip(0.0, 1.0),
        "cost": _score_lower_better(df["avg_cost"]),
        "reliability": _score_higher_better(df["reliability_score"]),
        "delivery": _score_lower_better(df["delivery_time_days"]),
    }

    raw = pd.Series(0.0, index=df.index, dtype=float)
    for name, series in components.items():
        weight = HEURISTIC_WEIGHTS[COMPONENT_WEIGHTS[name]]
        df[f"_component_{name}"] = series
        raw = raw + series * weight

    df["raw_score"] = raw
    df["heuristic_score"] = _rescale_final(raw)
    return df


def _candidate_records(scored: pd.DataFrame) -> list[dict]:
    """Convert scored rows to API records with a nested `scoring` breakdown.

    Rounding is applied for display only, and never to the fields the
    Knowledge Agent compares against fixed thresholds."""
    records: list[dict] = []
    for row in scored.to_dict(orient="records"):
        components = {name: float(row.pop(f"_component_{name}")) for name in COMPONENT_WEIGHTS}
        weights = {name: HEURISTIC_WEIGHTS[key] for name, key in COMPONENT_WEIGHTS.items()}
        contributions = {name: components[name] * weights[name] for name in COMPONENT_WEIGHTS}
        raw_score = float(row.pop("raw_score"))
        final_score = float(row.pop("heuristic_score"))

        record: dict = {}
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                record[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                value = float(value)
                record[key] = value if key in RULE_INPUT_FIELDS else round(value, DISPLAY_DECIMALS)
            else:
                record[key] = value

        record["heuristic_score"] = round(final_score, DISPLAY_DECIMALS)
        record["raw_score"] = round(raw_score, 6)
        record["scoring"] = {
            "weights": weights,
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

def rank_suppliers(
    conn: sqlite3.Connection,
    required_deficit: float = 0,
    top_n: int | None = TOP_N_SUPPLIER_CANDIDATES,
) -> dict:
    """Filter candidates that cannot cover the required deficit, score the rest
    with the weighted heuristic, and return a sorted candidate list.

    `top_n=None` scores and returns the entire pool (used by the decision
    matrix view) — the scoring path is identical either way, so both views
    always agree."""
    pool = load_pool(conn)
    total_pool = len(pool)

    eligible = pool[pool["stock_availability"] >= required_deficit].copy()
    filtered_out = pool[pool["stock_availability"] < required_deficit]["supplier_id"].tolist()

    # No supplier can single-handedly cover the deficit — fall back to ranking
    # the full pool so the system can still recommend a partial fulfillment /
    # split-order strategy downstream.
    split_order_fallback = False
    if eligible.empty:
        eligible = pool.copy()
        split_order_fallback = True

    scorable, unscorable = _coerce_scoring_inputs(eligible)

    if scorable.empty:
        candidates: list[dict] = []
    else:
        scored = _score_pool(scorable)
        scored = scored.sort_values("raw_score", ascending=False).reset_index(drop=True)
        scored["rank"] = scored.index + 1
        candidates = _candidate_records(scored if top_n is None else scored.head(top_n))

    return {
        "agent": "Supplier Intelligence Agent",
        "required_deficit": required_deficit,
        "pool_size": total_pool,
        "eligible_count": len(scorable),
        "filtered_out_suppliers": filtered_out,
        "unscorable_suppliers": unscorable,
        "split_order_fallback": split_order_fallback,
        "ranked_candidates": candidates,
        "primary_supplier": candidates[0] if candidates else None,
        "backup_supplier": candidates[1] if len(candidates) > 1 else None,
    }


def full_matrix(conn: sqlite3.Connection) -> dict:
    """Baseline ranking (deficit=0, whole pool) for the standalone Supplier
    Decision Matrix view, so every supplier is shown regardless of any specific
    SKU's current deficit."""
    return rank_suppliers(conn, required_deficit=0, top_n=None)


def run(conn: sqlite3.Connection, required_deficit: float = 0) -> dict:
    return rank_suppliers(conn, required_deficit=required_deficit)
