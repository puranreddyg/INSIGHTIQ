"""
Multi-supplier (split-order) allocation — deterministic, risk-respecting.

Input: the Supplier Agent's `allocation_pool` after the Knowledge Agent has
annotated each supplier with its risk status. Output: a structured allocation.

Allocation policy
  1. Exclude High Risk suppliers (Knowledge Agent Rule 3). They are never
     allocated automatically, even if that leaves a shortage; they are reported
     with their availability so a buyer can make an explicit decision.
  2. Fill the remaining (acceptable) suppliers in heuristic-rank order — the same
     Heuristic Rank Score that ranks single suppliers, which already weighs cost
     (25%), delivery time (20%), quality (30%) and fulfillment (25%) — each up to
     its available quantity, until the requirement is met. Ties break on
     supplier_id, so the result is fully deterministic.
  3. Never allocate more than a supplier's available quantity, nor more than the
     requirement in total.
  4. If acceptable suppliers cannot cover the requirement, the allocation is
     marked PARTIAL with the residual shortage, the maximum feasible quantity and
     the reason (combined availability too low, and/or High Risk exclusion).

Trade-offs (documented, deliberate)
  * Rank-greedy is explainable and reuses the existing ranking, but it is not a
    cost optimiser: it may pay more per unit than the cheapest feasible split, and
    it may use more suppliers than the minimum. Total cost and the latest arrival
    are reported so the buyer can see both.
  * Delivery is only as fast as the slowest allocated supplier; each line keeps
    its own lead time and arrival date.
  * Availability is SIMULATED (DataSets/simulated) — a split plan is a planning
    aid, not a reservation of live supplier stock.
"""
from __future__ import annotations

from backend.config import FIELD_PROVENANCE

HIGH_RISK = "High Risk"

ALLOCATION_POLICY = (
    "High Risk suppliers are excluded; acceptable suppliers are filled in heuristic-rank order "
    "(cost, delivery, quality and fulfillment already weighted in the score), each up to its available "
    "quantity, until the requirement is met. Ties break on supplier_id."
)


def allocation_line(c: dict, qty: int) -> dict:
    unit_cost = float(c["unit_cost"])
    return {
        "supplier_id": c["supplier_id"],
        "allocated_qty": int(qty),
        "available_qty": int(c["available_qty"]),
        "unit_cost": round(unit_cost, 4),
        "line_cost": round(qty * unit_cost, 2),
        "lead_time_days": float(c["lead_time_days"]),
        "risk_status": c.get("supplier_status"),
        "status_flags": c.get("status_flags", []),
        "heuristic_rank": c.get("rank"),
        "heuristic_score": c.get("heuristic_score"),
        "warehouses": c.get("warehouses"),
        "provenance": {k: FIELD_PROVENANCE[k] for k in ("available_qty", "unit_cost", "lead_time_days")},
    }


def allocate(candidates: list[dict], required_quantity: float) -> dict:
    required = int(required_quantity or 0)
    usable = [c for c in candidates if c.get("available_qty") is not None and int(c["available_qty"]) > 0]
    order = sorted(usable, key=lambda c: (c.get("rank") if c.get("rank") is not None else 10**9, c["supplier_id"]))
    acceptable = [c for c in order if c.get("supplier_status") != HIGH_RISK]
    high_risk = [c for c in order if c.get("supplier_status") == HIGH_RISK]

    lines: list[dict] = []
    remaining = max(required, 0)
    for c in acceptable:
        if remaining <= 0:
            break
        qty = min(int(c["available_qty"]), remaining)
        lines.append(allocation_line(c, qty))
        remaining -= qty

    acceptable_total = sum(int(c["available_qty"]) for c in acceptable)
    high_risk_total = sum(int(c["available_qty"]) for c in high_risk)
    combined_total = acceptable_total + high_risk_total
    allocated = sum(line["allocated_qty"] for line in lines)

    reasons: list[str] = []
    if required <= 0:
        status = "nothing_required"
    elif remaining == 0:
        status = "full"
    else:
        status = "partial" if allocated else "none"
        if combined_total < required:
            reasons.append(f"Combined availability of all {len(order)} candidate supplier(s) is {combined_total} units, "
                           f"below the required {required}.")
        if high_risk:
            ids = ", ".join(c["supplier_id"] for c in high_risk)
            reasons.append(f"{high_risk_total} unit(s) at High Risk supplier(s) {ids} were excluded by Rule 3 "
                           f"(defect rate above threshold); allocating them needs an explicit buyer decision.")
        if not order:
            reasons.append("No candidate supplier has any available quantity.")

    return {
        "status": status,
        "is_partial": status == "partial",
        "required_quantity": required,
        "allocations": lines,
        "supplier_count": len(lines),
        "total_allocated": allocated,
        "total_cost": round(sum(line["line_cost"] for line in lines), 2),
        "residual_shortage": max(required - allocated, 0),
        "max_feasible_quantity": min(required, acceptable_total),
        "max_feasible_quantity_ignoring_risk": min(required, combined_total),
        "excluded_high_risk": [
            {"supplier_id": c["supplier_id"], "available_qty": int(c["available_qty"]), "supplier_status": HIGH_RISK}
            for c in high_risk
        ],
        "shortage_reasons": reasons,
        "policy": ALLOCATION_POLICY,
        "availability_provenance": FIELD_PROVENANCE["available_qty"],
    }
