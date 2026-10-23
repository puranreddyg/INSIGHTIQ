"""
Perception Agent — observes SKU stock state in the real supply-chain ledger.

Percepts:  `supply_chain` (daily Inventory_Level, Reorder_Point, Units_Sold,
           Order_Quantity per SKU × warehouse) as of the latest date in the data.
Actions:   emits one procurement request per SKU for the downstream agents.

Urgency is decided per warehouse lane (Inventory < ReorderPoint) — unchanged. The
request is per SKU and keeps its lanes, so warehouse detail is carried, not
aggregated away. Lane state is read once per (SKU, warehouse): if a feed ever
repeats a lane-day across several supplier rows, Inventory_Level and
Reorder_Point are not double-counted (see backend/forecasting/demand.py).

RequiredQuantity (config.REQUIRED_QUANTITY_METHOD):

  "forecast" (preferred) — an order-up-to calculation per URGENT lane:
      L  = the lane's real supplier lead time (days)
      R  = review period: median days between the lane's real historical orders
      H  = L + R                                  (protection interval)
      D  = forecast demand over the next H days   (backend/forecasting)
      SS = safety stock (config.SAFETY_STOCK_METHOD)
      Target = D + SS;   LaneQty = max(0, Target − Inventory)
    The SKU quantity is the sum of LaneQty over urgent lanes (non-urgent lanes
    order nothing), rounded UP to ORDER_QUANTITY_ROUNDING. Inventory never goes
    negative in the sum and each lane is counted once. In the source data orders
    arrive the day they are placed (inv[t] = inv[t-1] − sold + ordered), so there
    is no on-order stock to subtract.
    Why L + R and not L alone: in this dataset Reorder_Point is unrelated to lead
    time (corr 0.03) and sits well above lead-time demand, so a pure lead-time
    target is below current stock on 13 of the 14 urgent lanes and would order
    nothing for an urgent SKU.

  "median" (fallback) — the legacy rule: an order is placed only when a lane drops
    below its reorder point and lot sizes do not depend on the shortfall, so each
    urgent lane needs its median historical lot size; the SKU needs the sum,
    rounded up to ORDER_QUANTITY_ROUNDING.
"""
from __future__ import annotations

import math
import re
import sqlite3
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

from backend import config
from backend.forecasting import service as forecast_service
from backend.forecasting.demand import LANE_KEYS, daily_lane_demand


def snapshot_date(conn: sqlite3.Connection) -> str:
    return conn.execute("SELECT MAX(date) FROM supply_chain").fetchone()[0]


def _history_window_start(as_of: str) -> str:
    days = max(config.DEMAND_LOOKBACK_DAYS, 28) + 7
    return (pd.Timestamp(as_of) - pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d %H:%M:%S")


def _lane_table(conn: sqlite3.Connection, as_of: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_sql_query(
        """SELECT sku_id, warehouse_id, supplier_id, inventory_level, reorder_point, supplier_lead_time_days
           FROM supply_chain WHERE date = ?""",
        conn, params=(as_of,),
    ).drop_duplicates()
    # One state reading per lane: state is lane-level, so never summed.
    lanes = rows.groupby(LANE_KEYS, as_index=False).agg(
        supplier_id=("supplier_id", lambda s: ", ".join(sorted(set(s)))),
        inventory_level=("inventory_level", "first"),
        reorder_point=("reorder_point", "first"),
        lead_time_days=("supplier_lead_time_days", "max"),
    )

    orders = pd.read_sql_query(
        "SELECT sku_id, warehouse_id, date, order_quantity FROM supply_chain WHERE order_quantity > 0", conn
    )
    lot = orders.groupby(LANE_KEYS).order_quantity.median().rename("lot_size_median")
    order_days = orders.assign(date=pd.to_datetime(orders.date)).drop_duplicates(LANE_KEYS + ["date"]).sort_values(LANE_KEYS + ["date"])
    order_days["gap"] = order_days.groupby(LANE_KEYS).date.diff().dt.days
    cycle = order_days.groupby(LANE_KEYS).gap.median().rename("order_cycle_days")

    history = daily_lane_demand(conn, since=_history_window_start(as_of))
    window = history.tail(config.DEMAND_LOOKBACK_DAYS)
    stats = pd.DataFrame({"daily_demand": window.mean(), "demand_std": window.std(ddof=1)})
    stats.index.names = LANE_KEYS

    lanes = lanes.join(lot, on=LANE_KEYS).join(cycle, on=LANE_KEYS).join(stats, on=LANE_KEYS)
    lanes["shortfall"] = (lanes.reorder_point - lanes.inventory_level).clip(lower=0)
    lanes["urgent"] = lanes.inventory_level < lanes.reorder_point
    return lanes, history


# ---------------------------------------------------------------------------
# Forecast-based order-up-to quantity
# ---------------------------------------------------------------------------

def _review_period(order_cycle) -> float:
    if config.FORECAST_REVIEW_PERIOD_SOURCE == "order_cycle" and pd.notna(order_cycle):
        return float(order_cycle)
    return float(config.FORECAST_REVIEW_PERIOD_DAYS)


def _safety_stock(mean: float, std: float, horizon_days: float) -> tuple[float, str]:
    std = 0.0 if not np.isfinite(std) else std
    if config.SAFETY_STOCK_METHOD == "days_of_cover":
        return config.SAFETY_STOCK_DAYS * mean, f"{config.SAFETY_STOCK_DAYS} days × mean demand"
    z = NormalDist().inv_cdf(config.SERVICE_LEVEL)
    return z * std * math.sqrt(horizon_days), f"z({config.SERVICE_LEVEL:.0%})={z:.3f} × σ × √H"


def _forecast_lanes(lanes: pd.DataFrame, history: pd.DataFrame, method: dict) -> pd.DataFrame:
    """Adds horizon, forecast demand, safety stock, target and lane quantity to
    every urgent lane. Non-urgent lanes are left blank: they order nothing."""
    out = lanes.copy()
    for col in ["review_period_days", "horizon_days", "forecast_demand", "safety_stock", "target_stock", "lane_quantity"]:
        out[col] = np.nan
    out["safety_stock_rule"] = None
    urgent = out[out.urgent]
    if urgent.empty or history.empty:
        return out

    review = urgent.order_cycle_days.map(_review_period)
    horizon = (urgent.lead_time_days + review).round().astype(int).clip(lower=1)
    keys = [k for k in zip(urgent.sku_id, urgent.warehouse_id) if k in history.columns]
    fc = forecast_service.forecast_daily(history[keys], int(horizon.max()), method["method"], method["model"]) if keys else None

    for idx, r in urgent.iterrows():
        h = int(horizon[idx])
        key = (r.sku_id, r.warehouse_id)
        demand = float(fc[key].iloc[:h].sum()) if fc is not None and key in fc.columns else float(r.daily_demand or 0) * h
        ss, rule = _safety_stock(float(r.daily_demand or 0), float(r.demand_std) if pd.notna(r.demand_std) else 0.0, h)
        target = demand + ss
        out.loc[idx, ["review_period_days", "horizon_days", "forecast_demand", "safety_stock", "target_stock", "lane_quantity"]] = [
            float(review[idx]), h, demand, ss, target, max(0.0, target - float(r.inventory_level)),
        ]
        out.loc[idx, "safety_stock_rule"] = rule
    return out


def _round_up(value: float, base: int) -> int:
    return int(math.ceil(value / base) * base)


def _opt(value, digits: int = 2):
    return round(float(value), digits) if pd.notna(value) else None


def _lane_fact(r) -> dict:
    return {
        "warehouse_id": r.warehouse_id,
        "supplier_id": r.supplier_id,
        "Inventory": int(r.inventory_level),
        "ReorderPoint": int(r.reorder_point),
        "Shortfall": int(r.shortfall),
        "Urgent": bool(r.urgent),
        "LotSizeMedian": float(r.lot_size_median) if pd.notna(r.lot_size_median) else None,
        "DailyDemand": _opt(r.daily_demand),
        "DemandStd": _opt(r.demand_std),
        "LeadTimeDays": _opt(r.lead_time_days, 1),
        "ReviewPeriodDays": _opt(r.review_period_days, 1),
        "HorizonDays": int(r.horizon_days) if pd.notna(r.horizon_days) else None,
        "ForecastDemand": _opt(r.forecast_demand),
        "SafetyStock": _opt(r.safety_stock),
        "TargetStock": _opt(r.target_stock),
        "LaneQuantity": _opt(r.lane_quantity),
    }


def _quantity(urgent: pd.DataFrame, method: dict) -> dict:
    median_raw = float(urgent.lot_size_median.sum()) if not urgent.empty else 0.0
    forecast_raw = float(urgent.lane_quantity.sum()) if not urgent.empty else 0.0
    base = config.ORDER_QUANTITY_ROUNDING
    use_forecast = config.REQUIRED_QUANTITY_METHOD == "forecast" and not urgent.lane_quantity.isna().any()
    raw = forecast_raw if use_forecast else median_raw
    final = _round_up(raw, base) if raw > 0 else 0

    calc = {
        "method": "forecast" if use_forecast else "median",
        "configured_method": config.REQUIRED_QUANTITY_METHOD,
        "forecast_method": method["method"] if use_forecast else None,
        "forecast_method_reason": method["reason"] if use_forecast else None,
        "rounding": base,
        "quantity_before_rounding": round(raw, 2),
        "final_quantity": final,
        "median_method_quantity": _round_up(median_raw, base) if median_raw else 0,
        "aggregation": "sum over urgent warehouse lanes, each lane counted once; non-urgent lanes order nothing",
    }
    if use_forecast:
        calc.update({
            "forecast_demand": round(float(urgent.forecast_demand.sum()), 2),
            "safety_stock": round(float(urgent.safety_stock.sum()), 2),
            "target_stock": round(float(urgent.target_stock.sum()), 2),
            "current_inventory": int(urgent.inventory_level.sum()),
            "safety_stock_method": config.SAFETY_STOCK_METHOD,
            "service_level": config.SERVICE_LEVEL if config.SAFETY_STOCK_METHOD == "service_level" else None,
        })
        lines = [
            f"{r.warehouse_id}: forecast {r.forecast_demand:.1f} over H={int(r.horizon_days)}d "
            f"(L={r.lead_time_days:g} + R={r.review_period_days:g}) + SS {r.safety_stock:.1f} = target {r.target_stock:.1f}; "
            f"− inventory {int(r.inventory_level)} → {r.lane_quantity:.1f}"
            for r in urgent.itertuples()
        ]
        calc["explanation"] = (
            f"Order-up-to per urgent lane using {method['method'].replace('_', ' ')} demand forecast. "
            + " | ".join(lines)
            + f". Sum {raw:.1f} rounded up to the nearest {base} = {final}."
        )
        calc["rule"] = (f"Σ over urgent lanes of max(0, forecast demand over lead time + review period + safety stock − inventory), "
                        f"rounded up to the nearest {base}")
    else:
        reason = "configured" if config.REQUIRED_QUANTITY_METHOD == "median" else "fallback: forecast inputs incomplete"
        calc["explanation"] = (f"Median historical lot size summed over urgent lanes ({reason}): {median_raw:.1f} "
                               f"rounded up to the nearest {base} = {final}.")
        calc["rule"] = (f"sum over urgent lanes of the lane's median historical Order_Quantity, rounded up to the nearest {base}")
    return calc


def _sku_request(sku_id: str, lanes: pd.DataFrame, as_of: str, method: dict) -> dict:
    lanes = lanes.sort_values("warehouse_id")
    urgent = lanes[lanes.urgent]
    # Urgent lanes drive the request. With none, the lane closest to its reorder
    # point is shown so Rule 1 has a concrete fact to evaluate.
    focus = urgent if not urgent.empty else lanes.assign(_m=lanes.inventory_level - lanes.reorder_point).nsmallest(1, "_m")
    calc = _quantity(urgent, method)
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
        "Shortfall": int(urgent.shortfall.sum()),
        "DailyDemand": round(float(focus.daily_demand.fillna(0).sum()), 2),
        "ForecastDemand": calc.get("forecast_demand"),
        "SafetyStock": calc.get("safety_stock"),
        "TargetStock": calc.get("target_stock"),
        "required_quantity_raw": calc["quantity_before_rounding"],
        "RequiredQuantity": calc["final_quantity"],
        "required_quantity_rule": calc["rule"],
        "quantity_calculation": calc,
    }


def _natural_key(identifier: str) -> list:
    """SKU_2 before SKU_10, without assuming any particular ID format."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", identifier)]


def observe_all(conn: sqlite3.Connection) -> dict[str, dict]:
    as_of = snapshot_date(conn)
    lanes, history = _lane_table(conn, as_of)
    method = forecast_service.resolve_method()
    lanes = _forecast_lanes(lanes, history, method)
    requests = {sku: _sku_request(sku, g, as_of, method) for sku, g in lanes.groupby("sku_id")}
    return dict(sorted(requests.items(), key=lambda kv: _natural_key(kv[0])))


def observe_sku(conn: sqlite3.Connection, sku_id: str) -> Optional[dict]:
    return observe_all(conn).get(sku_id)


def urgent_requests(requests: dict[str, dict]) -> list[dict]:
    urgent = [r for r in requests.values() if r["Urgent"]]
    return sorted(urgent, key=lambda r: (-r["RequiredQuantity"], -r["Shortfall"], r["sku_id"]))


def run(conn: sqlite3.Connection, sku_id: Optional[str] = None) -> dict:
    """Entry point used by the Coordinator Agent."""
    requests = observe_all(conn)
    urgent = urgent_requests(requests)
    if sku_id is not None:
        focus = requests.get(sku_id)
    else:
        focus = urgent[0] if urgent else None

    return {
        "agent": "Perception Agent",
        "snapshot_date": snapshot_date(conn)[:10],
        "focus_facts": [focus] if focus else [],
        "urgent_skus": [{k: v for k, v in r.items() if k not in ("lanes", "focus_lanes", "quantity_calculation")} for r in urgent],
        "urgent_sku_count": len(urgent),
    }
