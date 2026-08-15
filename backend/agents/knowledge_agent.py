"""
Knowledge Agent — explainable, rule-based reasoning via forward chaining.

Maintains a working-memory `facts` dict seeded by the Perception & Supplier
agents, then repeatedly applies an IF-THEN rule set until no new fact fires
(fixed point), logging every firing into a structured reasoning_trace so the
decision is fully auditable (XAI). Each trace entry carries both a
human-readable `text` line (legacy/log form) and structured fields — actual
values, per-condition comparisons, AND/OR logic, matched flags — so the
frontend can animate the forward-chaining process without re-deriving or
guessing any of the underlying reasoning.
"""
from __future__ import annotations

from backend.config import (
    RULE_HIGH_RISK_DEFECT_RATE,
    RULE_PREFERRED_DELIVERY_DAYS,
    RULE_PREFERRED_RATING,
    RULE_RELIABLE_THRESHOLD,
)


def _fact_entry(entity: str, entity_kind: str, values: dict, text: str) -> dict:
    return {"type": "fact", "entity": entity, "entityKind": entity_kind, "values": values, "text": text}


def _condition(label: str, value, operator: str, threshold, result: bool, unit: str = "") -> dict:
    return {
        "label": label,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "result": bool(result),
        "expression": f"{value}{unit} {operator} {threshold}{unit}",
    }


def _rule_entry(
    rule_id: int,
    name: str,
    entity: str,
    if_text: str,
    then_text: str,
    conditions: list[dict],
    logic: str | None,
    matched: bool,
    outcome: str | None,
    text: str,
    else_text: str | None = None,
) -> dict:
    return {
        "type": "rule",
        "ruleId": rule_id,
        "name": name,
        "entity": entity,
        "ifText": if_text,
        "thenText": then_text,
        "elseText": else_text,
        "conditions": conditions,
        "logic": logic,
        "matched": matched,
        "outcome": outcome,
        "text": text,
    }


def _decision_entry(entity: str | None, field: str | None, value: str, text: str) -> dict:
    return {"type": "decision", "entity": entity, "field": field, "value": value, "text": text}


def evaluate_inventory_state(perception_fact: dict, trace: list[dict]) -> str:
    inventory = perception_fact["Inventory"]
    reorder_point = perception_fact["ReorderPoint"]
    sku = perception_fact["SKU"]

    trace.append(_fact_entry(
        sku, "sku",
        {"Inventory": inventory, "ReorderPoint": reorder_point},
        f"✓ Fact Detected → {sku} Inventory={inventory}, ReorderPoint={reorder_point}",
    ))

    matched = inventory < reorder_point
    state = "Urgent Procurement" if matched else "Monitoring"
    label = "MATCHED" if matched else "not matched"
    condition = _condition("Inventory", inventory, "<", reorder_point, matched)
    trace.append(_rule_entry(
        rule_id=1,
        name="Inventory State Evaluation",
        entity=sku,
        if_text="IF Inventory < ReorderPoint",
        then_text="THEN State = \"Urgent Procurement\"",
        else_text="ELSE State = \"Monitoring\"",
        conditions=[condition],
        logic=None,
        matched=matched,
        outcome=state,
        text=(
            f"✓ Rule Evaluated → IF Inventory({inventory}) < ReorderPoint({reorder_point}) "
            f"THEN State = 'Urgent Procurement'  [Rule 1: {label}]"
        ),
    ))
    trace.append(_decision_entry(sku, "procurement_state", state, f"✓ Decision Made → {sku} procurement state = '{state}'"))
    return state


def evaluate_supplier(candidate: dict, trace: list[dict]) -> dict:
    """Runs forward chaining over a single supplier candidate's facts,
    iterating rules 2 and 3 to a fixed point (a Supplier_Status label plus
    any secondary status), returning the annotated candidate."""
    supplier_id = candidate["supplier_id"]
    rating = candidate.get("rating")
    delivery_time = candidate.get("delivery_time_days")
    defect_rate = candidate.get("defect_rate")
    reliability = candidate.get("reliability_score")

    trace.append(_fact_entry(
        supplier_id, "supplier",
        {"Rating": rating, "DeliveryTime": delivery_time, "DefectRate": defect_rate, "Reliability": reliability},
        f"✓ Fact Detected → {supplier_id} Rating={rating}, DeliveryTime={delivery_time}d, "
        f"DefectRate={defect_rate}, Reliability={reliability}",
    ))

    statuses: list[str] = []
    fired = True
    seen_rules = set()
    # Fixed-point loop: keep sweeping the rule set until nothing new fires.
    while fired:
        fired = False

        if "R2" not in seen_rules and rating is not None and delivery_time is not None:
            cond_rating = _condition("Rating", rating, ">=", RULE_PREFERRED_RATING, rating >= RULE_PREFERRED_RATING)
            cond_delivery = _condition(
                "DeliveryTime", delivery_time, "<=", RULE_PREFERRED_DELIVERY_DAYS,
                delivery_time <= RULE_PREFERRED_DELIVERY_DAYS, unit="d",
            )
            matched = cond_rating["result"] and cond_delivery["result"]
            label = "MATCHED" if matched else "not matched"
            trace.append(_rule_entry(
                rule_id=2,
                name="Preferred Supplier",
                entity=supplier_id,
                if_text=f"IF Rating >= {RULE_PREFERRED_RATING} AND DeliveryTime <= {RULE_PREFERRED_DELIVERY_DAYS}d",
                then_text="THEN Supplier_Status = \"Preferred\"",
                conditions=[cond_rating, cond_delivery],
                logic="AND",
                matched=matched,
                outcome="Preferred" if matched else None,
                text=(
                    f"✓ Rule Evaluated → IF Rating({rating}) >= {RULE_PREFERRED_RATING} AND "
                    f"DeliveryTime({delivery_time}) <= {RULE_PREFERRED_DELIVERY_DAYS} "
                    f"THEN Supplier_Status = 'Preferred'  [Rule 2: {label}]"
                ),
            ))
            if matched:
                statuses.append("Preferred")
            seen_rules.add("R2")
            fired = True

        if "R3" not in seen_rules and defect_rate is not None:
            cond_defect = _condition("DefectRate", defect_rate, ">", RULE_HIGH_RISK_DEFECT_RATE, defect_rate > RULE_HIGH_RISK_DEFECT_RATE)
            matched = cond_defect["result"]
            label = "MATCHED" if matched else "not matched"
            trace.append(_rule_entry(
                rule_id=3,
                name="Risk Assessment",
                entity=supplier_id,
                if_text=f"IF DefectRate > {RULE_HIGH_RISK_DEFECT_RATE}",
                then_text="THEN Supplier_Status = \"High Risk\"",
                conditions=[cond_defect],
                logic=None,
                matched=matched,
                outcome="High Risk" if matched else None,
                text=(
                    f"✓ Rule Evaluated → IF DefectRate({defect_rate}) > {RULE_HIGH_RISK_DEFECT_RATE} "
                    f"THEN Supplier_Status = 'High Risk'  [Rule 3: {label}]"
                ),
            ))
            if matched:
                statuses.append("High Risk")
            seen_rules.add("R3")
            fired = True

        if "R4" not in seen_rules and reliability is not None and not statuses:
            # Gated on `not statuses`, so "no prior status" is always true whenever
            # this branch runs — surfaced explicitly as a condition for the trace.
            cond_prior = _condition("Prior Status", "None", "==", "None", True)
            cond_reliability = _condition("Reliability", reliability, ">=", RULE_RELIABLE_THRESHOLD, reliability >= RULE_RELIABLE_THRESHOLD)
            matched = cond_reliability["result"]
            label = "Reliable" if matched else "Standard"
            trace.append(_rule_entry(
                rule_id=4,
                name="Reliability / Fallback",
                entity=supplier_id,
                if_text=f"IF no prior status AND Reliability >= {RULE_RELIABLE_THRESHOLD}",
                then_text="THEN Supplier_Status = \"Reliable\"",
                else_text="ELSE Supplier_Status = \"Standard\"",
                conditions=[cond_prior, cond_reliability],
                logic="AND",
                matched=matched,
                outcome=label,
                text=(
                    f"✓ Rule Evaluated → IF no prior status AND Reliability({reliability}) >= {RULE_RELIABLE_THRESHOLD} "
                    f"THEN Supplier_Status = 'Reliable' ELSE 'Standard'  [Rule 4: {label}]"
                ),
            ))
            statuses.append(label)
            seen_rules.add("R4")
            fired = True

    final_status = statuses[0] if statuses else "Standard"
    if "High Risk" in statuses:
        final_status = "High Risk"  # risk flag always takes precedence for display
    elif "Preferred" in statuses:
        final_status = "Preferred"

    trace.append(_decision_entry(
        supplier_id, "Supplier_Status", final_status,
        f"✓ Decision Made → {supplier_id} Supplier_Status = '{final_status}'",
    ))

    return {**candidate, "supplier_status": final_status, "status_flags": statuses}


def run(perception_result: dict, supplier_result: dict) -> dict:
    trace: list[dict] = []

    inventory_state = None
    focus_facts = perception_result.get("focus_facts") or []
    if focus_facts:
        inventory_state = evaluate_inventory_state(focus_facts[0], trace)

    annotated_candidates = [
        evaluate_supplier(c, trace) for c in supplier_result.get("ranked_candidates", [])
    ]

    preferred = [c for c in annotated_candidates if c["supplier_status"] == "Preferred"]
    high_risk = [c for c in annotated_candidates if c["supplier_status"] == "High Risk"]

    recommended = None
    for c in annotated_candidates:
        if c["supplier_status"] != "High Risk":
            recommended = c
            break
    if recommended is None and annotated_candidates:
        recommended = annotated_candidates[0]
        trace.append(_decision_entry(
            None, None,
            f"No risk-free candidate available — defaulting to top-ranked {recommended['supplier_id']} despite High Risk flag",
            f"✓ Decision Made → No risk-free candidate available — defaulting to top-ranked "
            f"{recommended['supplier_id']} despite High Risk flag",
        ))

    return {
        "agent": "Knowledge Agent",
        "inventory_state": inventory_state,
        "annotated_candidates": annotated_candidates,
        "preferred_suppliers": preferred,
        "high_risk_suppliers": high_risk,
        "recommended_supplier": recommended,
        "reasoning_trace": trace,
    }
