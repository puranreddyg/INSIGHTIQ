"""
Planning Agent — operational strategy & contingency generation.

Cross-references the cleaned `calendar` table (working days / weekend
detection, seasonality) to project a realistic arrival date, computes order
economics, a replenishment curve, and designates a backup supplier.

The order quantity is the Perception Agent's required_quantity, already rounded
and already checked against the primary supplier's availability. No plan is
produced when nothing is needed or when no single supplier qualifies.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd


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


def build_plan(
    conn: sqlite3.Connection,
    sku_fact: dict | None,
    primary_supplier: dict | None,
    backup_supplier: dict | None,
    required_quantity: float,
    supplier_result: dict | None = None,
) -> dict:
    trace: list[str] = []
    today = datetime.now().date()

    if not required_quantity:
        return {
            "agent": "Planning Agent",
            "status": "no_action",
            "error": "No procurement required — no warehouse lane is below its reorder point.",
            "reasoning": ["Rule 1 did not fire for any lane of this SKU, so no purchase order is planned."],
        }

    if primary_supplier is None:
        combined = (supplier_result or {}).get("combined_available_qty", 0)
        return {
            "agent": "Planning Agent",
            "status": "no_eligible_supplier",
            "error": f"No single supplier can fulfil {required_quantity:.0f} units — no supplier was selected.",
            "combined_available_qty": combined,
            "reasoning": [
                f"Every candidate's available quantity is below the required {required_quantity:.0f} units.",
                f"Combined availability across candidates is {combined} units; "
                + ("a split order across suppliers could cover it (not implemented)." if combined >= required_quantity
                   else "even a split order could not cover it."),
            ],
        }

    order_quantity = int(required_quantity)
    trace.append(
        f"Order quantity {order_quantity} units = {sku_fact.get('required_quantity_rule', 'required quantity')} "
        f"(primary supplier has {primary_supplier.get('available_qty')} available)."
    )

    base_lead_days = max(1, round(primary_supplier.get("lead_time_days", 5)))
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

    unit_cost = primary_supplier.get("unit_cost", 0)
    total_expense = round(order_quantity * unit_cost, 2)
    trace.append(f"Total expense: {order_quantity} × ₹{unit_cost:.2f} = ₹{total_expense:,.2f} (primary supplier {primary_supplier['supplier_id']}).")

    if backup_supplier:
        trace.append(f"Backup supplier designated: {backup_supplier['supplier_id']} (heuristic rank #{backup_supplier.get('rank', '?')}) in case primary fails.")
    else:
        trace.append("No backup supplier available in the current candidate pool.")

    daily_demand = max(1, round(sku_fact.get("DailyDemand") or 0))
    trace.append(f"Daily demand {daily_demand} units = mean real Units_Sold over the last 28 days of the {sku_fact.get('focus_scope', 'lanes')}.")
    current_inventory = sku_fact.get("Inventory", 0)
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
        "status": "planned",
        "sku_id": sku_fact["sku_id"],
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


_SEASON_BY_MONTH = ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer", "Summer", "Summer", "Fall", "Fall", "Fall", "Winter"]


def build_split_plan(
    conn: sqlite3.Connection,
    sku_fact: dict,
    allocations: list[dict],
    backup_candidates: list[dict],
    required_quantity: float,
) -> dict:
    """Plan an order that no single supplier could cover: one purchase line per
    allocated supplier, each with its own cost and arrival date. The order is
    complete when the last supplier delivers."""
    today = datetime.now().date()
    season_delays = _seasonality_delay_map(conn)
    trace: list[str] = [
        f"No single supplier can fulfil {int(required_quantity)} units, so the order is split across "
        f"{len(allocations)} suppliers (safe suppliers first, in heuristic-rank order)."
    ]

    lines: list[dict] = []
    arrivals: dict[date, int] = {}
    for a in allocations:
        qty = int(a["allocated_qty"])
        base_days = max(1, round(a.get("lead_time_days", 5)))
        naive = _add_working_days(today, base_days)
        season = _SEASON_BY_MONTH[naive.month - 1]
        extra = season_delays.get(season, 0)
        arrival = _add_working_days(naive, extra) if extra else naive
        unit_cost = float(a.get("unit_cost", 0))
        expense = round(qty * unit_cost, 2)
        arrivals[arrival] = arrivals.get(arrival, 0) + qty
        lines.append({
            "supplier_id": a["supplier_id"],
            "supplier_status": a.get("supplier_status"),
            "quantity": qty,
            "available_qty": int(a["available_qty"]),
            "unit_cost": round(unit_cost, 2),
            "expense": expense,
            "lead_time_working_days": base_days,
            "seasonality_delay_days": extra,
            "season": season,
            "estimated_arrival_date": arrival.isoformat(),
        })
        trace.append(
            f"{a['supplier_id']}: {qty} of {a['available_qty']} available × ₹{unit_cost:.2f} = ₹{expense:,.2f}; "
            f"{base_days} working days"
            + (f" + {extra} seasonal buffer" if extra else "")
            + f" → arrives {arrival.isoformat()}."
        )

    order_quantity = sum(line["quantity"] for line in lines)
    total_expense = round(sum(line["expense"] for line in lines), 2)
    final_arrival = max(date.fromisoformat(line["estimated_arrival_date"]) for line in lines)
    last_line = max(lines, key=lambda l: l["estimated_arrival_date"])
    trace.append(
        f"Order quantity {order_quantity} units; total expense ₹{total_expense:,.2f}; "
        f"complete when the last supplier delivers on {final_arrival.isoformat()}."
    )

    lead_ids = {line["supplier_id"] for line in lines}
    spare = [c for c in backup_candidates if c["supplier_id"] not in lead_ids]
    backup = next((c for c in spare if c.get("supplier_status") != "High Risk"), None)
    if backup:
        trace.append(f"Backup supplier designated: {backup['supplier_id']} (not used in the split, rank #{backup.get('rank', '?')}).")
    else:
        trace.append("No unused, non-High-Risk supplier is available as a backup.")

    daily_demand = max(1, round(sku_fact.get("DailyDemand") or 0))
    trace.append(f"Daily demand {daily_demand} units = mean real Units_Sold over the last 28 days of the {sku_fact.get('focus_scope', 'lanes')}.")
    level = sku_fact.get("Inventory", 0)
    horizon_days = max((final_arrival - today).days + 7, 7)
    curve = []
    for i in range(horizon_days + 1):
        day = today + timedelta(days=i)
        if day in arrivals:
            level += arrivals[day]
        else:
            level = max(0, level - daily_demand)
        curve.append({"date": day.isoformat(), "projected_inventory": level})

    return {
        "agent": "Planning Agent",
        "status": "planned",
        "split_order": True,
        "sku_id": sku_fact["sku_id"],
        "order_quantity": order_quantity,
        "allocations": lines,
        "primary_supplier_id": lines[0]["supplier_id"],
        "backup_supplier_id": backup["supplier_id"] if backup else None,
        "order_date": today.isoformat(),
        "estimated_arrival_date": final_arrival.isoformat(),
        "base_lead_time_working_days": max(line["lead_time_working_days"] for line in lines),
        "seasonality_delay_days": last_line["seasonality_delay_days"],
        "season": last_line["season"],
        "unit_cost": round(total_expense / order_quantity, 2) if order_quantity else 0,  # blended
        "total_expense": total_expense,
        "daily_demand_estimate": daily_demand,
        "replenishment_curve": curve,
        "reasoning": trace,
    }


def run(conn: sqlite3.Connection, sku_fact: dict | None, knowledge_result: dict, required_quantity: float,
        supplier_result: dict | None = None) -> dict:
    if required_quantity and knowledge_result.get("split_order") and knowledge_result.get("allocations"):
        return build_split_plan(
            conn, sku_fact, knowledge_result["allocations"],
            knowledge_result.get("backup_candidates", []), required_quantity,
        )
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
    return build_plan(conn, sku_fact, primary, backup, required_quantity, supplier_result)
