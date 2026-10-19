"""
Planning Agent — operational strategy & contingency generation.

Cross-references the cleaned `calendar` table (working days / weekend
detection, seasonality) to project a realistic arrival date, computes order
economics, a replenishment curve, and designates a backup supplier.

The order quantity is the Perception Agent's forecast-based required_quantity,
already rounded and checked against supplier capacity. When no single supplier
can cover it the order is split (Knowledge Agent Rule 6); when even all
suppliers together fall short a PARTIAL plan is produced with its shortfall.

The projected-inventory curve subtracts the demand FORECAST for each day (the
trained model, day i after the data snapshot), not a flat trailing average.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from backend.ml.store import load_demand_model


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


def _demand_path(conn: sqlite3.Connection, sku_fact: dict, days: int) -> list[float]:
    """Forecast units/day for the SKU's focus lanes, for days 1..days after the snapshot."""
    model = load_demand_model(conn)
    start = pd.Timestamp(sku_fact["snapshot_date"]) + pd.Timedelta(days=1)
    daily = model.expected_daily(pd.date_range(start, periods=days + 1))
    scale = sum(l.get("DemandScale", 1.0) for l in sku_fact.get("focus_lanes", [])) or 1.0
    return list(np.asarray(daily) * scale)


def _curve(today: date, start_level: float, arrivals: dict[date, int], demand: list[float]) -> list[dict]:
    curve, level = [], float(start_level)
    for i, d in enumerate(demand):
        day = today + timedelta(days=i)
        level = max(0.0, level - d) + arrivals.get(day, 0)
        curve.append({"date": day.isoformat(), "projected_inventory": int(round(level))})
    return curve


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
            "error": f"No supplier has any capacity for this SKU — {required_quantity:.0f} units cannot be planned.",
            "combined_available_qty": combined,
            "reasoning": [
                f"Combined capacity across all candidates is {combined} units, so not even a partial order is possible.",
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

    horizon_days = max((final_arrival - today).days + 7, 7)
    demand = _demand_path(conn, sku_fact, horizon_days)
    daily_demand = round(float(np.mean(demand)), 1)
    trace.append(
        f"Projected inventory uses the demand forecast day by day (avg {daily_demand} units/day over the next "
        f"{horizon_days} days; previous method, the last-28-day average, was {sku_fact.get('TrailingDailyDemand')})."
    )
    curve = _curve(today, sku_fact.get("Inventory", 0), {final_arrival: order_quantity}, demand)

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
    shortfall_qty: int = 0,
) -> dict:
    """Plan an order that no single supplier could cover: one purchase line per
    allocated supplier, each with its own cost and arrival date. The order is
    complete when the last supplier delivers."""
    today = datetime.now().date()
    season_delays = _seasonality_delay_map(conn)
    trace: list[str] = [
        f"No single supplier can fulfil {int(required_quantity)} units, so the order is split across "
        f"{len(allocations)} suppliers (fewest suppliers, safe ones first, filled in heuristic-rank order)."
    ]
    if shortfall_qty:
        trace.append(
            f"PARTIAL ORDER: all suppliers together can supply only {int(required_quantity) - shortfall_qty} units; "
            f"{shortfall_qty} units remain uncovered and must be sourced from a new supplier or a later order."
        )

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
            "defect_rate": a.get("defect_rate"),
            "on_time_prob": a.get("on_time_prob"),
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

    horizon_days = max((final_arrival - today).days + 7, 7)
    demand = _demand_path(conn, sku_fact, horizon_days)
    daily_demand = round(float(np.mean(demand)), 1)
    trace.append(f"Projected inventory uses the demand forecast day by day (avg {daily_demand} units/day).")
    curve = _curve(today, sku_fact.get("Inventory", 0), arrivals, demand)

    return {
        "agent": "Planning Agent",
        "status": "planned",
        "split_order": True,
        "partial_order": bool(shortfall_qty),
        "required_quantity": int(required_quantity),
        "shortfall_qty": int(shortfall_qty),
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


def _add_predictions(plan: dict, lines: list[tuple[dict, int]], knowledge_result: dict) -> dict:
    """Attach what the supplier model expects this plan to deliver."""
    if plan.get("status") != "planned" or not lines:
        return plan
    def f(c, key, default):
        v = c.get(key)
        return default if v is None else float(v)
    good = sum(q * f(c, "fill_rate", 1.0) * (1 - f(c, "defect_rate", 0.0)) for c, q in lines)
    on_time_all = float(np.prod([f(c, "on_time_prob", 1.0) for c, _ in lines]))
    qty = sum(q for _, q in lines)
    w_def = sum(q * f(c, "defect_rate", 0.0) for c, q in lines) / qty
    plan["predictions"] = {
        "expected_good_units": int(round(good)),
        "expected_defect_rate": round(w_def, 4),
        "on_time_probability": round(on_time_all, 3),
        "lines": [{
            "supplier_id": c["supplier_id"],
            "promised_lead_time_days": c.get("promised_lead_time_days"),
            "predicted_delay_days": round(f(c, "expected_delay_days", 0.0), 2),
            "predicted_lead_time_days": round(f(c, "lead_time_days", 0.0), 2),
            "on_time_prob": round(f(c, "on_time_prob", 1.0), 3),
            "defect_rate": round(f(c, "defect_rate", 0.0), 4),
            "fill_rate": round(f(c, "fill_rate", 1.0), 4),
        } for c, _ in lines],
    }
    plan["quality_constraint_met"] = bool(knowledge_result.get("quality_constraint_met", True))
    plan["budget_constraint_met"] = bool(knowledge_result.get("budget_constraint_met", True))
    plan["budget"] = knowledge_result.get("budget")
    plan["decision"] = knowledge_result.get("decision")
    if plan["budget"] is not None:
        plan["reasoning"].append(
            f"Budget: total cost ₹{plan['total_expense']:,.2f} vs budget ₹{plan['budget']:,.2f} "
            f"({'within' if plan['total_expense'] <= plan['budget'] + 0.01 else 'OVER'} budget)."
        )
    plan["reasoning"].append(
        f"Supplier model: of {qty} units ordered, about {int(round(good))} are expected to arrive usable "
        f"(predicted defect rate {w_def:.1%}, short shipments included); probability that "
        f"{'every line' if len(lines) > 1 else 'the order'} arrives on time = {on_time_all:.0%}. "
        "Arrival dates use the learned lead time (promised + predicted delay)."
    )
    if not plan["quality_constraint_met"]:
        plan["reasoning"].append(
            "QUALITY FLAG: no supplier or combination keeps the predicted defect rate within the 5% rule — "
            "the smallest violation was chosen; review before approving."
        )
    if not plan["budget_constraint_met"]:
        plan["reasoning"].append("BUDGET FLAG: no compliant option fits the budget; review before approving.")
    return plan


def run(conn: sqlite3.Connection, sku_fact: dict | None, knowledge_result: dict, required_quantity: float,
        supplier_result: dict | None = None) -> dict:
    if required_quantity and knowledge_result.get("decision") == "needs_review":
        reason = knowledge_result.get("review_reason") or "no supplier plan meets the constraints"
        return {
            "agent": "Planning Agent",
            "status": "needs_review",
            "error": f"No purchase order planned: {reason}. A buyer must review this SKU (e.g. raise the budget, "
                     "accept a quality exception, or add a supplier).",
            "budget": knowledge_result.get("budget"),
            "reasoning": [f"Knowledge Agent could not find any supplier plan meeting all constraints: {reason}."],
        }
    if required_quantity and knowledge_result.get("split_order") and knowledge_result.get("allocations"):
        plan = build_split_plan(
            conn, sku_fact, knowledge_result["allocations"],
            knowledge_result.get("backup_candidates", []), required_quantity,
            knowledge_result.get("shortfall_qty", 0),
        )
        plan["search"] = knowledge_result["allocations"][0].get("search")
        return _add_predictions(plan, [(a, a["allocated_qty"]) for a in knowledge_result["allocations"]], knowledge_result)
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
    plan = build_plan(conn, sku_fact, primary, backup, required_quantity, supplier_result)
    return _add_predictions(plan, [(primary, plan.get("order_quantity", 0))] if primary else [], knowledge_result)
