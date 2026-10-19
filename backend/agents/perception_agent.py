"""
Perception Agent — observes SKU stock state and PREDICTS what happens next.

Percepts:  `supply_chain` (Inventory_Level, Reorder_Point per SKU x warehouse on
           the latest date), the lane's real supplier lead time, and the trained
           demand forecaster (backend/ml/demand_forecaster.py).
Actions:   emits one procurement request per SKU for the downstream agents.

Per warehouse lane, with L = the LEARNED lead time of the lane's supplier (promised
lead time + its predicted delay, backend/ml/supplier_model.py) and C = the lane's
own replenishment cycle (median days between its historical orders):

  forecast demand during lead time   D_L  ~ Normal(mu_L, sd_L)       (demand model)
      sd_L also carries lead-time uncertainty: sqrt(sd_demand^2 + (daily demand * delay_sd)^2)
  forecast reorder point             ROP_f = mu_L + z * sd_L         (z = 95% service level)
  effective reorder point            ROP_e = max(planner Reorder_Point, ROP_f)
  urgent                             Inventory < ROP_e
  stock-out probability              P(D_L > Inventory) before a new order could arrive
  order-up-to level                  S = mu_(L+C) + z * sd_(L+C)
  recommended quantity               max(0, S - Inventory)

The planner's static reorder point is still honoured (a lane below it is always
urgent); the forecast can only make the system act EARLIER, when the coming
season or a long lead time means the static point is too low.

RequiredQuantity = sum of the urgent lanes' recommended quantities, rounded up to
ORDER_QUANTITY_ROUNDING. The previous rule (sum of median historical lot sizes) is
kept alongside as `required_quantity_baseline` for comparison.
"""
from __future__ import annotations

import math
import re
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

from backend.config import ORDER_QUANTITY_ROUNDING, SERVICE_LEVEL_Z
from backend.ml.store import load_demand_model


def snapshot_date(conn: sqlite3.Connection) -> str:
    return conn.execute("SELECT MAX(date) FROM supply_chain").fetchone()[0]


def _normal_sf(x: np.ndarray) -> np.ndarray:
    """P(Z > x) for a standard normal, without scipy."""
    return 0.5 * np.vectorize(math.erfc)(np.asarray(x, dtype=float) / math.sqrt(2.0))


def _delay_columns(conn: sqlite3.Connection) -> str:
    """Learned delay columns if the supplier profile has them; a feed without delivery
    history means no known delay (the promised lead time is used as is)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(supplier_sku_profile)")}
    delay = "COALESCE(p.expected_delay_days, 0)" if "expected_delay_days" in cols else "0"
    sd = "COALESCE(p.delay_sd, 0)" if "delay_sd" in cols else "0"
    return f"{delay} AS expected_delay_days, {sd} AS delay_sd"


def _lane_table(conn: sqlite3.Connection, as_of: str) -> pd.DataFrame:
    lanes = pd.read_sql_query(
        """SELECT s.sku_id, s.warehouse_id, s.supplier_id, s.inventory_level, s.reorder_point,
                  l.lead_time_days AS promised_lead_time_days, t.demand_scale, t.order_cadence_days,
                  t.lot_size_median, t.trailing_mean_28d,
                  {delay_cols}
           FROM supply_chain s
           JOIN sku_supplier_lanes l ON l.sku_id = s.sku_id AND l.warehouse_id = s.warehouse_id
           JOIN lane_stats t ON t.sku_id = s.sku_id AND t.warehouse_id = s.warehouse_id
           LEFT JOIN supplier_sku_profile p ON p.sku_id = s.sku_id AND p.supplier_id = s.supplier_id
           WHERE s.date = ?""".format(delay_cols=_delay_columns(conn)),
        conn, params=(as_of,),
    )
    model = load_demand_model(conn)
    z = SERVICE_LEVEL_Z
    lanes["lead_time_days"] = lanes.promised_lead_time_days + lanes.expected_delay_days
    lead = np.maximum(1, np.round(lanes.lead_time_days.to_numpy(float))).astype(int)
    cycle = np.maximum(1, np.round(lanes.order_cadence_days.to_numpy(float))).astype(int)
    scale = lanes.demand_scale.to_numpy(float)
    inv = lanes.inventory_level.to_numpy(float)

    mu_l = model.horizon_means(as_of, lead, scale)
    mu_lc = model.horizon_means(as_of, lead + cycle, scale)
    # delivery-time uncertainty adds (daily demand x delay sd)^2 to the demand variance
    lt_var_l = (mu_l / lead * lanes.delay_sd.to_numpy(float)) ** 2
    lt_var_lc = (mu_lc / (lead + cycle) * lanes.delay_sd.to_numpy(float)) ** 2
    sd_l = np.sqrt(model.horizon_std(lead, scale) ** 2 + lt_var_l)
    sd_lc = np.sqrt(model.horizon_std(lead + cycle, scale) ** 2 + lt_var_lc)

    lanes["lead_days"] = lead
    lanes["cycle_days"] = cycle
    lanes["forecast_lead_demand"] = mu_l
    lanes["forecast_daily_demand"] = mu_lc / (lead + cycle)
    lanes["forecast_reorder_point"] = np.ceil(mu_l + z * sd_l)
    lanes["effective_reorder_point"] = np.maximum(lanes.reorder_point.to_numpy(float), lanes.forecast_reorder_point)
    lanes["stockout_probability"] = _normal_sf((inv - mu_l) / sd_l)
    lanes["order_up_to"] = mu_lc + z * sd_lc
    lanes["recommended_qty"] = np.maximum(0.0, lanes.order_up_to - inv)

    below_planner = inv < lanes.reorder_point.to_numpy(float)
    below_forecast = inv < lanes.forecast_reorder_point.to_numpy(float)
    lanes["urgent"] = below_planner | below_forecast
    lanes["urgent_reason"] = np.select(
        [below_planner & below_forecast, below_planner, below_forecast],
        ["below planner and forecast reorder points", "below planner reorder point", "below forecast reorder point"],
        default="",
    )
    lanes["shortfall"] = (lanes.effective_reorder_point - inv).clip(lower=0)
    return lanes


def _round_up(value: float, base: int) -> int:
    return int(math.ceil(value / base) * base)


def _lane_fact(r) -> dict:
    return {
        "warehouse_id": r.warehouse_id,
        "supplier_id": r.supplier_id,
        "Inventory": int(r.inventory_level),
        "ReorderPoint": int(r.reorder_point),
        "ForecastReorderPoint": int(r.forecast_reorder_point),
        "EffectiveReorderPoint": int(r.effective_reorder_point),
        "Shortfall": int(r.shortfall),
        "Urgent": bool(r.urgent),
        "UrgentReason": r.urgent_reason or None,
        "LeadTimeDays": int(r.lead_days),
        "PromisedLeadTimeDays": int(r.promised_lead_time_days),
        "PredictedDelayDays": round(float(r.expected_delay_days), 2),
        "OrderCycleDays": int(r.cycle_days),
        "ForecastLeadDemand": round(float(r.forecast_lead_demand), 1),
        "StockoutProbability": round(float(r.stockout_probability), 3),
        "OrderUpTo": round(float(r.order_up_to), 1),
        "RecommendedQty": round(float(r.recommended_qty), 1),
        "DemandScale": round(float(r.demand_scale), 4),
        "LotSizeMedian": float(r.lot_size_median) if pd.notna(r.lot_size_median) else None,
        "DailyDemand": round(float(r.forecast_daily_demand), 2),
        "TrailingDailyDemand": round(float(r.trailing_mean_28d), 2) if pd.notna(r.trailing_mean_28d) else None,
    }


def _sku_request(sku_id: str, lanes: pd.DataFrame, as_of: str) -> dict:
    lanes = lanes.sort_values("warehouse_id")
    urgent = lanes[lanes.urgent]
    # Urgent lanes drive the request. With none, the lane closest to its effective
    # reorder point is shown so Rule 1 has a concrete fact to evaluate.
    focus = urgent if not urgent.empty else lanes.assign(
        _m=lanes.inventory_level - lanes.effective_reorder_point).nsmallest(1, "_m")
    raw_need = float(urgent.recommended_qty.sum()) if not urgent.empty else 0.0
    baseline = float(urgent.lot_size_median.sum()) if not urgent.empty else 0.0
    return {
        "SKU": sku_id,
        "sku_id": sku_id,
        "snapshot_date": as_of[:10],
        "Urgent": not urgent.empty,
        "urgent_lane_count": int(len(urgent)),
        "lane_count": int(len(lanes)),
        "lanes": [_lane_fact(r) for r in lanes.itertuples()],
        "focus_lanes": [_lane_fact(r) for r in focus.itertuples()],
        "focus_scope": "urgent lanes" if not urgent.empty else "closest lane (none urgent)",
        "Inventory": int(focus.inventory_level.sum()),
        "ReorderPoint": int(focus.reorder_point.sum()),
        "EffectiveReorderPoint": int(focus.effective_reorder_point.sum()),
        "Shortfall": int(urgent.shortfall.sum()),
        "StockoutProbability": round(float(focus.stockout_probability.max()), 3),
        "DailyDemand": round(float(focus.forecast_daily_demand.sum()), 2),
        "TrailingDailyDemand": round(float(focus.trailing_mean_28d.sum()), 2),
        "required_quantity_raw": round(raw_need, 1),
        "RequiredQuantity": _round_up(raw_need, ORDER_QUANTITY_ROUNDING) if raw_need > 0 else 0,
        "required_quantity_baseline": _round_up(baseline, ORDER_QUANTITY_ROUNDING) if baseline else 0,
        "required_quantity_rule": (
            "sum over urgent lanes of (forecast demand over lead time + order cycle + 95% safety stock − inventory), "
            f"rounded up to the nearest {ORDER_QUANTITY_ROUNDING}"
        ),
    }


def _natural_key(identifier: str) -> list:
    """SKU_2 before SKU_10, without assuming any particular ID format."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", identifier)]


def observe_all(conn: sqlite3.Connection) -> dict[str, dict]:
    as_of = snapshot_date(conn)
    lanes = _lane_table(conn, as_of)
    requests = {sku: _sku_request(sku, g, as_of) for sku, g in lanes.groupby("sku_id")}
    return dict(sorted(requests.items(), key=lambda kv: _natural_key(kv[0])))


def observe_sku(conn: sqlite3.Connection, sku_id: str) -> Optional[dict]:
    return observe_all(conn).get(sku_id)


def urgent_requests(requests: dict[str, dict]) -> list[dict]:
    urgent = [r for r in requests.values() if r["Urgent"]]
    return sorted(urgent, key=lambda r: (-r["StockoutProbability"], -r["RequiredQuantity"], r["sku_id"]))


def run(conn: sqlite3.Connection, sku_id: Optional[str] = None) -> dict:
    """Entry point used by the Coordinator Agent."""
    requests = observe_all(conn)
    urgent = urgent_requests(requests)
    focus = requests.get(sku_id) if sku_id is not None else (urgent[0] if urgent else None)
    model = load_demand_model(conn)
    return {
        "agent": "Perception Agent",
        "snapshot_date": snapshot_date(conn)[:10],
        "focus_facts": [focus] if focus else [],
        "urgent_skus": [{k: v for k, v in r.items() if k not in ("lanes", "focus_lanes")} for r in urgent],
        "urgent_sku_count": len(urgent),
        "forecast_model": {
            "type": f"seasonal regression (yearly Fourier K={model.k} + promotion x season)",
            "trained_on": f"{model.n_rows} real daily rows, {model.trained_from} to {model.trained_through}",
            "r2": round(model.r2, 3),
            "backtest": model.metrics,
            "service_level_z": SERVICE_LEVEL_Z,
        },
    }
