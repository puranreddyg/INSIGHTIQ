"""
Perception Agent — observes SKU stock state in the real supply-chain ledger.

Percepts:  `supply_chain` (daily Inventory_Level, Reorder_Point, Units_Sold,
           Order_Quantity per SKU × warehouse) as of the latest date in the data.
Actions:   emits one procurement request per SKU for the downstream agents.

Urgency is decided per warehouse lane (Inventory < ReorderPoint). The request is
per SKU and keeps its lanes, so warehouse detail is carried, not aggregated away.

required_quantity replicates the replenishment policy observed in the data: an
order is placed only when a lane drops below its reorder point, and lot sizes do
not depend on the shortfall. Each urgent lane therefore needs its median
historical lot size; the SKU needs the sum, rounded up to ORDER_QUANTITY_ROUNDING.
"""
from __future__ import annotations

import math
import re
import sqlite3
from typing import Optional

import pandas as pd

from backend.config import DEMAND_LOOKBACK_DAYS, ORDER_QUANTITY_ROUNDING


def snapshot_date(conn: sqlite3.Connection) -> str:
    return conn.execute("SELECT MAX(date) FROM supply_chain").fetchone()[0]


def _lane_table(conn: sqlite3.Connection, as_of: str) -> pd.DataFrame:
    lanes = pd.read_sql_query(
        """SELECT sku_id, warehouse_id, supplier_id, inventory_level, reorder_point
           FROM supply_chain WHERE date = ?""",
        conn, params=(as_of,),
    )
    orders = pd.read_sql_query(
        "SELECT sku_id, warehouse_id, order_quantity FROM supply_chain WHERE order_quantity > 0", conn
    )
    lot = orders.groupby(["sku_id", "warehouse_id"]).order_quantity.median().rename("lot_size_median").reset_index()

    cutoff = (pd.Timestamp(as_of) - pd.Timedelta(days=DEMAND_LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    demand = pd.read_sql_query(
        "SELECT sku_id, warehouse_id, units_sold FROM supply_chain WHERE date > ? AND date <= ?",
        conn, params=(cutoff, as_of),
    ).groupby(["sku_id", "warehouse_id"]).units_sold.mean().rename("daily_demand").reset_index()

    lanes = lanes.merge(lot, on=["sku_id", "warehouse_id"], how="left").merge(demand, on=["sku_id", "warehouse_id"], how="left")
    lanes["shortfall"] = (lanes.reorder_point - lanes.inventory_level).clip(lower=0)
    lanes["urgent"] = lanes.inventory_level < lanes.reorder_point
    return lanes


def _round_up(value: float, base: int) -> int:
    return int(math.ceil(value / base) * base)


def _lane_fact(r) -> dict:
    return {
        "warehouse_id": r.warehouse_id,
        "supplier_id": r.supplier_id,
        "Inventory": int(r.inventory_level),
        "ReorderPoint": int(r.reorder_point),
        "Shortfall": int(r.shortfall),
        "Urgent": bool(r.urgent),
        "LotSizeMedian": float(r.lot_size_median) if pd.notna(r.lot_size_median) else None,
        "DailyDemand": round(float(r.daily_demand), 2) if pd.notna(r.daily_demand) else None,
    }


def _sku_request(sku_id: str, lanes: pd.DataFrame, as_of: str) -> dict:
    lanes = lanes.sort_values("warehouse_id")
    urgent = lanes[lanes.urgent]
    # Urgent lanes drive the request. With none, the lane closest to its reorder
    # point is shown so Rule 1 has a concrete fact to evaluate.
    focus = urgent if not urgent.empty else lanes.assign(_m=lanes.inventory_level - lanes.reorder_point).nsmallest(1, "_m")
    raw_need = float(urgent.lot_size_median.sum()) if not urgent.empty else 0.0
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
        "DailyDemand": round(float(focus.daily_demand.sum()), 2),
        "required_quantity_raw": raw_need,
        "RequiredQuantity": _round_up(raw_need, ORDER_QUANTITY_ROUNDING) if raw_need else 0,
        "required_quantity_rule": "sum over urgent lanes of the lane's median historical Order_Quantity, rounded up to the nearest "
                                  f"{ORDER_QUANTITY_ROUNDING}",
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
        "urgent_skus": [{k: v for k, v in r.items() if k not in ("lanes", "focus_lanes")} for r in urgent],
        "urgent_sku_count": len(urgent),
    }
