"""
Planning Agent — operational strategy & contingency generation.

Cross-references the cleaned `calendar` table (working days / weekend
detection, seasonality) to project a realistic arrival date, computes order
economics, a replenishment curve, and designates a backup supplier.

The order quantity is the Perception Agent's required_quantity, already rounded.
Two plan shapes share one response format (`allocations`, one line per supplier):

  single  — the Knowledge Agent's recommended supplier covers it alone
            (status "planned"; the legacy top-level fields describe that supplier).
  split   — no acceptable single supplier: backend/agents/allocation.py splits the
            quantity across acceptable suppliers (status "planned_split"), or
            covers only part of it and says so explicitly (status "planned_partial",
            is_partial = true, residual_shortage > 0).

No plan is produced when nothing is needed, or when no acceptable supplier has
any stock (status "no_eligible_supplier"). Every line gets its own arrival date
because suppliers have different lead times; the plan's estimated_arrival_date is
the LAST line's arrival (when the full allocated quantity is in).
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd

from backend.agents import allocation as allocator
from backend.config import FIELD_PROVENANCE

SEASON_BY_MONTH = ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer", "Summer", "Summer", "Fall", "Fall", "Fall", "Winter"]


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


def _arrival(today: date, lead_time_days: float, season_delays: dict[str, int]) -> dict:
    base = max(1, round(lead_time_days or 5))
    naive = _add_working_days(today, base)
    season = SEASON_BY_MONTH[naive.month - 1]
    extra = season_delays.get(season, 0)
    final = _add_working_days(naive, extra) if extra else naive
    return {"base_lead_time_working_days": base, "naive_arrival": naive, "season": season,
            "seasonality_delay_days": extra, "arrival": final}


def _replenishment_curve(today: date, start_inventory: int, daily_demand: int, arrivals: dict[date, int], last_arrival: date) -> list[dict]:
    horizon_days = max((last_arrival - today).days + 7, 7)
    curve = []
    level = start_inventory
    for i in range(horizon_days + 1):
        day = today + timedelta(days=i)
        if day in arrivals:
            level += arrivals[day]
        else:
            level = max(0, level - daily_demand)
        curve.append({"date": day.isoformat(), "projected_inventory": level})
    return curve


def _no_supplier_plan(required_quantity: float, supplier_result: dict | None, allocation: dict | None) -> dict:
    combined = (supplier_result or {}).get("combined_available_qty", 0)
    reasons = [f"Every candidate's available quantity is below the required {required_quantity:.0f} units, "
               f"or the suppliers that could cover it are High Risk."]
    reasons += (allocation or {}).get("shortage_reasons", [])
    return {
        "agent": "Planning Agent",
        "status": "no_eligible_supplier",
        "error": f"No acceptable supplier can fulfil any of the {required_quantity:.0f} units — no supplier was selected.",
        "combined_available_qty": combined,
        "allocation": allocation,
        "residual_shortage": int(required_quantity),
        "reasoning": reasons,
    }


def build_plan(
    conn: sqlite3.Connection,
    sku_fact: dict | None,
    primary_supplier: dict | None,
    backup_supplier: dict | None,
    required_quantity: float,
    supplier_result: dict | None = None,
    allocation: dict | None = None,
) -> dict:
    trace: list[str] = []
    today = datetime.now().date()

    if not required_quantity:
        urgent = bool(sku_fact and sku_fact.get("Urgent"))
        return {
            "agent": "Planning Agent",
            "status": "no_action",
            "error": ("Below reorder point, but forecast demand plus safety stock is already covered by current stock — no order needed."
                      if urgent else "No procurement required — no warehouse lane is below its reorder point."),
            "reasoning": [((sku_fact or {}).get("quantity_calculation") or {}).get("explanation")
                          if urgent else "Rule 1 did not fire for any lane of this SKU, so no purchase order is planned."],
        }

    if primary_supplier is not None:
        mode = "single"
        lines = [allocator.allocation_line(primary_supplier, int(required_quantity))]
        allocation = {
            "status": "full", "is_partial": False, "required_quantity": int(required_quantity),
            "allocations": lines, "supplier_count": 1, "total_allocated": int(required_quantity),
            "total_cost": lines[0]["line_cost"], "residual_shortage": 0,
            "max_feasible_quantity": int(required_quantity), "excluded_high_risk": [], "shortage_reasons": [],
            "policy": "Single supplier: the top-ranked eligible supplier that is not High Risk.",
            "availability_provenance": FIELD_PROVENANCE["available_qty"],
        }
    elif allocation is not None and allocation["total_allocated"] > 0:
        mode = "split"
    else:
        return _no_supplier_plan(required_quantity, supplier_result, allocation)

    order_quantity = int(allocation["total_allocated"])
    rule = sku_fact.get("required_quantity_rule", "required quantity") if sku_fact else "required quantity"
    trace.append(f"Required quantity {int(required_quantity)} units = {rule}.")
    if mode == "single":
        trace.append(f"Single supplier {primary_supplier['supplier_id']} has {primary_supplier.get('available_qty')} "
                     f"available (SIMULATED) and covers the full quantity.")
    else:
        trace.append("No acceptable single supplier — split order. " + allocation["policy"])

    season_delays = _seasonality_delay_map(conn)
    for line in allocation["allocations"]:
        a = _arrival(today, line["lead_time_days"], season_delays)
        line.update({
            "estimated_arrival": a["arrival"].isoformat(),
            "base_lead_time_working_days": a["base_lead_time_working_days"],
            "seasonality_delay_days": a["seasonality_delay_days"],
            "season": a["season"],
        })
        delay_note = f" + {a['seasonality_delay_days']} {a['season']} buffer day(s)" if a["seasonality_delay_days"] else ""
        trace.append(f"{line['supplier_id']}: {line['allocated_qty']} × ₹{line['unit_cost']:.2f} = ₹{line['line_cost']:,.2f}; "
                     f"{a['base_lead_time_working_days']} working-day lead time{delay_note} → arrives {line['estimated_arrival']} "
                     f"[{line['risk_status']}].")

    arrivals = sorted(date.fromisoformat(line["estimated_arrival"]) for line in allocation["allocations"])
    final_arrival, first_arrival = arrivals[-1], arrivals[0]
    total_expense = round(sum(line["line_cost"] for line in allocation["allocations"]), 2)
    allocation["total_cost"] = total_expense
    trace.append(f"Total expense: ₹{total_expense:,.2f} for {order_quantity} units from {len(arrivals)} supplier(s); "
                 f"full allocated quantity in by {final_arrival.isoformat()}.")

    if allocation["is_partial"]:
        trace.append(f"PARTIAL PLAN: only {order_quantity} of {int(required_quantity)} units can be allocated; "
                     f"residual shortage {allocation['residual_shortage']} units.")
        trace.extend(allocation["shortage_reasons"])

    if mode == "single":
        if backup_supplier:
            trace.append(f"Backup supplier designated: {backup_supplier['supplier_id']} (heuristic rank #{backup_supplier.get('rank', '?')}) in case primary fails.")
        else:
            trace.append("No backup supplier available in the current candidate pool.")

    daily_demand = max(1, round(sku_fact.get("DailyDemand") or 0))
    trace.append(f"Daily demand {daily_demand} units = mean real Units_Sold over the last 28 days of the {sku_fact.get('focus_scope', 'lanes')}.")
    arrivals_by_day: dict[date, int] = {}
    for line in allocation["allocations"]:
        d = date.fromisoformat(line["estimated_arrival"])
        arrivals_by_day[d] = arrivals_by_day.get(d, 0) + line["allocated_qty"]
    curve = _replenishment_curve(today, sku_fact.get("Inventory", 0), daily_demand, arrivals_by_day, final_arrival)

    primary_line = allocation["allocations"][0]
    weighted_unit_cost = total_expense / order_quantity if order_quantity else 0.0
    status = "planned" if mode == "single" else ("planned_partial" if allocation["is_partial"] else "planned_split")
    return {
        "agent": "Planning Agent",
        "status": status,
        "allocation_mode": mode,
        "sku_id": sku_fact["sku_id"],
        "required_quantity": int(required_quantity),
        "order_quantity": order_quantity,
        "is_partial": allocation["is_partial"],
        "residual_shortage": allocation["residual_shortage"],
        "allocation": allocation,
        "allocations": allocation["allocations"],
        "primary_supplier_id": primary_line["supplier_id"],
        "backup_supplier_id": backup_supplier["supplier_id"] if (mode == "single" and backup_supplier) else None,
        "order_date": today.isoformat(),
        "estimated_arrival_date": final_arrival.isoformat(),
        "first_arrival_date": first_arrival.isoformat(),
        "base_lead_time_working_days": primary_line["base_lead_time_working_days"],
        "seasonality_delay_days": primary_line["seasonality_delay_days"],
        "season": primary_line["season"],
        # Single supplier: that supplier's unit cost. Split: quantity-weighted average.
        "unit_cost": round(primary_line["unit_cost"] if mode == "single" else weighted_unit_cost, 4),
        "total_expense": total_expense,
        "daily_demand_estimate": daily_demand,
        "replenishment_curve": curve,
        "reasoning": trace,
    }


def allocation_for(knowledge_result: dict, required_quantity: float) -> dict | None:
    if knowledge_result.get("allocation_mode") != "split":
        return None
    return allocator.allocate(knowledge_result.get("allocation_candidates", []), required_quantity)


def run(conn: sqlite3.Connection, sku_fact: dict | None, knowledge_result: dict, required_quantity: float,
        supplier_result: dict | None = None) -> dict:
    primary = knowledge_result.get("recommended_supplier") if knowledge_result.get("allocation_mode") == "single" else None
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
    return build_plan(conn, sku_fact, primary, backup, required_quantity, supplier_result,
                      allocation_for(knowledge_result, required_quantity))
