"""
Planning Agent — operational strategy & contingency generation.

Cross-references the cleaned `calendar` table (working days / weekend
detection, seasonality) to project a realistic arrival date, computes order
economics, a replenishment curve, and designates a backup supplier.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd

from backend.config import DEFAULT_DAILY_DEMAND_FALLBACK, ORDER_QUANTITY_ROUNDING


def _round_up(value: float, base: int) -> int:
    return int(math.ceil(value / base) * base)


def _add_working_days(start: date, working_days: int) -> date:
    current = start
    added = 0
    while added < working_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            added += 1
    return current


def _seasonality_delay_map(conn: sqlite3.Connection) -> dict[str, int]:
    """Derives a data-driven extra-delay-days heuristic per season from the
    cleaned calendar table: seasons with a higher weekend-day density in the
    dataset are treated as higher delivery-friction periods (fewer effective
    business days to route/clear shipments), adding 0-2 buffer days."""
    cal = pd.read_sql_query("SELECT season, is_weekend FROM calendar", conn)
    if cal.empty:
        return {}
    density = cal.groupby("season")["is_weekend"].mean()
    lo, hi = density.min(), density.max()
    span = (hi - lo) if hi > lo else 1.0
    delay = ((density - lo) / span * 2).round().astype(int)
    return delay.to_dict()


def _daily_demand_estimate(product_fact: dict) -> int:
    reorder_point = product_fact.get("ReorderPoint") if product_fact else None
    if not reorder_point:
        return DEFAULT_DAILY_DEMAND_FALLBACK
    # Heuristic: reorder point is sized to cover ~2 weeks of expected demand.
    return max(1, round(reorder_point / 14))


def build_plan(
    conn: sqlite3.Connection,
    product_fact: dict | None,
    primary_supplier: dict | None,
    backup_supplier: dict | None,
    deficit: float,
) -> dict:
    trace: list[str] = []
    today = datetime.now().date()

    if primary_supplier is None:
        return {
            "agent": "Planning Agent",
            "error": "No eligible supplier available to plan against.",
            "reasoning": ["No candidate supplier passed the Supplier Intelligence Agent's filters."],
        }

    order_quantity = _round_up(max(deficit, ORDER_QUANTITY_ROUNDING), ORDER_QUANTITY_ROUNDING)
    trace.append(f"Order quantity set to {order_quantity} units (deficit {deficit:.0f}, rounded up to nearest {ORDER_QUANTITY_ROUNDING}).")

    base_lead_days = max(1, round(primary_supplier.get("delivery_time_days", 5)))
    naive_arrival = _add_working_days(today, base_lead_days)
    trace.append(f"Base supplier lead time: {base_lead_days} working days → naive arrival {naive_arrival.isoformat()}.")

    season_delays = _seasonality_delay_map(conn)
    arrival_month = naive_arrival.month
    season = ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer", "Summer", "Summer", "Fall", "Fall", "Fall", "Winter"][arrival_month - 1]
    extra_delay = season_delays.get(season, 0)
    final_arrival = _add_working_days(naive_arrival, extra_delay) if extra_delay else naive_arrival
    if extra_delay:
        trace.append(f"Calendar seasonality adjustment: arrival falls in {season} (+{extra_delay} working day(s) delivery-friction buffer) → {final_arrival.isoformat()}.")
    else:
        trace.append(f"Calendar seasonality adjustment: arrival falls in {season} (no additional delay).")

    unit_cost = primary_supplier.get("avg_cost", 0)
    total_expense = round(order_quantity * unit_cost, 2)
    trace.append(f"Total expense: {order_quantity} × ₹{unit_cost:.2f} = ₹{total_expense:,.2f} (primary supplier {primary_supplier['supplier_id']}).")

    if backup_supplier:
        trace.append(f"Backup supplier designated: {backup_supplier['supplier_id']} (heuristic rank #{backup_supplier.get('rank', '?')}) in case primary fails.")
    else:
        trace.append("No backup supplier available in the current candidate pool.")

    daily_demand = _daily_demand_estimate(product_fact)
    current_inventory = product_fact.get("Inventory", 0) if product_fact else 0
    horizon_days = max((final_arrival - today).days + 7, 7)

    curve = []
    level = current_inventory
    for i in range(horizon_days + 1):
        day = today + timedelta(days=i)
        if day == final_arrival:
            level += order_quantity
        level = max(0, level - daily_demand) if day != final_arrival else level
        curve.append({"date": day.isoformat(), "projected_inventory": level})

    return {
        "agent": "Planning Agent",
        "order_quantity": order_quantity,
        "primary_supplier_id": primary_supplier["supplier_id"],
        "backup_supplier_id": backup_supplier["supplier_id"] if backup_supplier else None,
        "order_date": today.isoformat(),
        "estimated_arrival_date": final_arrival.isoformat(),
        "base_lead_time_working_days": base_lead_days,
        "seasonality_delay_days": extra_delay,
        "season": season,
        "unit_cost": round(unit_cost, 2),
        "total_expense": total_expense,
        "daily_demand_estimate": daily_demand,
        "replenishment_curve": curve,
        "reasoning": trace,
    }


def run(conn: sqlite3.Connection, product_fact: dict | None, knowledge_result: dict, deficit: float) -> dict:
    primary = knowledge_result.get("recommended_supplier")
    candidates = knowledge_result.get("annotated_candidates", [])
    alternatives = [c for c in candidates if primary and c["supplier_id"] != primary["supplier_id"]]
    # Prefer a contingency the Knowledge Agent hasn't flagged — designating a
    # High Risk supplier as the fallback would contradict the same rule that
    # disqualified it from being primary. Only fall back to a flagged one when
    # the candidate pool offers nothing else.
    backup = next(
        (c for c in alternatives if c.get("supplier_status") != "High Risk"),
        next(iter(alternatives), None),
    )
    return build_plan(conn, product_fact, primary, backup, deficit)
