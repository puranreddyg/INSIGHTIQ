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

Rules
  R1  IF Inventory < ReorderPoint (per warehouse lane) THEN "Urgent Procurement"
  R5  IF AvailableQty >= RequiredQty                   ELSE "Ineligible" (stop)
  R2  IF LeadTime <= 3d AND DefectRate <= 0.05          THEN "Preferred"
  R3  IF DefectRate > 0.05                              THEN "High Risk"
  R4  IF no prior status AND FulfillmentRate >= 0.90    THEN "Reliable" ELSE "Standard"

R2 used a KPI "Rating" that only existed through the removed KPI-vendor→SUP_x
mapping; its quality condition now reuses R3's existing defect threshold, so no
new number is introduced. R4's 0.90 is unchanged and is now read as a 90%
fulfillment business target (docs/migration: rule changes).
"""
from __future__ import annotations

from backend.config import (
    RULE_HIGH_RISK_DEFECT_RATE,
    RULE_PREFERRED_DELIVERY_DAYS,
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


def _evaluate_lane(sku: str, lane: dict, trace: list[dict]) -> bool:
    entity = f"{sku} @ {lane['warehouse_id']}"
    inventory, reorder_point = lane["Inventory"], lane["ReorderPoint"]
    trace.append(_fact_entry(
        entity, "sku",
        {"Inventory": inventory, "ReorderPoint": reorder_point, "Supplier": lane["supplier_id"]},
        f"✓ Fact Detected → {entity} Inventory={inventory}, ReorderPoint={reorder_point}",
    ))

    matched = inventory < reorder_point
    state = "Urgent Procurement" if matched else "Monitoring"
    label = "MATCHED" if matched else "not matched"
    trace.append(_rule_entry(
        rule_id=1,
        name="Inventory State Evaluation",
        entity=entity,
        if_text="IF Inventory < ReorderPoint",
        then_text="THEN State = \"Urgent Procurement\"",
        else_text="ELSE State = \"Monitoring\"",
        conditions=[_condition("Inventory", inventory, "<", reorder_point, matched)],
        logic=None,
        matched=matched,
        outcome=state,
        text=(
            f"✓ Rule Evaluated → IF Inventory({inventory}) < ReorderPoint({reorder_point}) "
            f"THEN State = 'Urgent Procurement'  [Rule 1: {label}]"
        ),
    ))
    trace.append(_decision_entry(entity, "procurement_state", state, f"✓ Decision Made → {entity} procurement state = '{state}'"))
    return matched


def evaluate_inventory_state(perception_fact: dict, trace: list[dict]) -> str:
    """Rule 1 per focus lane; the SKU is urgent if any of its lanes is."""
    urgent = [_evaluate_lane(perception_fact["SKU"], lane, trace) for lane in perception_fact["focus_lanes"]]
    return "Urgent Procurement" if any(urgent) else "Monitoring"


def _evaluate_capacity(supplier_id: str, available, required, trace: list[dict]) -> bool:
    cond = _condition("AvailableQty", available, ">=", required, available >= required)
    matched = cond["result"]
    trace.append(_rule_entry(
        rule_id=5,
        name="Capacity Check",
        entity=supplier_id,
        if_text="IF AvailableQty >= RequiredQty",
        then_text="THEN supplier can fulfil the request alone",
        else_text="ELSE Supplier_Status = \"Ineligible\"",
        conditions=[cond],
        logic=None,
        matched=matched,
        outcome=None if matched else "Ineligible",
        text=(
            f"✓ Rule Evaluated → IF AvailableQty({available}) >= RequiredQty({required}) "
            f"ELSE Supplier_Status = 'Ineligible'  [Rule 5: {'MATCHED' if matched else 'not matched'}]"
        ),
    ))
    return matched


def evaluate_supplier(candidate: dict, required_quantity: float, trace: list[dict]) -> dict:
    """Runs forward chaining over a single supplier candidate's facts,
    iterating rules to a fixed point (a Supplier_Status label plus any
    secondary status), returning the annotated candidate."""
    supplier_id = candidate["supplier_id"]
    lead_time = candidate.get("lead_time_days")
    defect_rate = candidate.get("defect_rate")
    fulfillment = candidate.get("fulfillment_rate")
    available = candidate.get("available_qty")

    trace.append(_fact_entry(
        supplier_id, "supplier",
        {"UnitCost": candidate.get("unit_cost"), "LeadTime": lead_time, "AvailableQty": available,
         "RequiredQty": required_quantity, "DefectRate": defect_rate, "FulfillmentRate": fulfillment},
        f"✓ Fact Detected → {supplier_id} LeadTime={lead_time}d, AvailableQty={available}, "
        f"DefectRate={defect_rate}, FulfillmentRate={fulfillment}",
    ))

    if not _evaluate_capacity(supplier_id, available, required_quantity, trace):
        trace.append(_decision_entry(
            supplier_id, "Supplier_Status", "Ineligible",
            f"✓ Decision Made → {supplier_id} Supplier_Status = 'Ineligible'",
        ))
        return {**candidate, "supplier_status": "Ineligible", "status_flags": ["Ineligible"]}

    statuses: list[str] = []
    fired = True
    seen_rules = set()
    # Fixed-point loop: keep sweeping the rule set until nothing new fires.
    while fired:
        fired = False

        if "R2" not in seen_rules and defect_rate is not None and lead_time is not None:
            cond_delivery = _condition(
                "LeadTime", lead_time, "<=", RULE_PREFERRED_DELIVERY_DAYS,
                lead_time <= RULE_PREFERRED_DELIVERY_DAYS, unit="d",
            )
            cond_quality = _condition("DefectRate", defect_rate, "<=", RULE_HIGH_RISK_DEFECT_RATE, defect_rate <= RULE_HIGH_RISK_DEFECT_RATE)
            matched = cond_delivery["result"] and cond_quality["result"]
            label = "MATCHED" if matched else "not matched"
            trace.append(_rule_entry(
                rule_id=2,
                name="Preferred Supplier",
                entity=supplier_id,
                if_text=f"IF LeadTime <= {RULE_PREFERRED_DELIVERY_DAYS}d AND DefectRate <= {RULE_HIGH_RISK_DEFECT_RATE}",
                then_text="THEN Supplier_Status = \"Preferred\"",
                conditions=[cond_delivery, cond_quality],
                logic="AND",
                matched=matched,
                outcome="Preferred" if matched else None,
                text=(
                    f"✓ Rule Evaluated → IF LeadTime({lead_time}) <= {RULE_PREFERRED_DELIVERY_DAYS} AND "
                    f"DefectRate({defect_rate}) <= {RULE_HIGH_RISK_DEFECT_RATE} "
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

        if "R4" not in seen_rules and fulfillment is not None and not statuses:
            # Gated on `not statuses`, so "no prior status" is always true whenever
            # this branch runs — surfaced explicitly as a condition for the trace.
            cond_prior = _condition("Prior Status", "None", "==", "None", True)
            cond_fulfillment = _condition("FulfillmentRate", fulfillment, ">=", RULE_RELIABLE_THRESHOLD, fulfillment >= RULE_RELIABLE_THRESHOLD)
            matched = cond_fulfillment["result"]
            label = "Reliable" if matched else "Standard"
            trace.append(_rule_entry(
                rule_id=4,
                name="Reliability / Fallback",
                entity=supplier_id,
                if_text=f"IF no prior status AND FulfillmentRate >= {RULE_RELIABLE_THRESHOLD}",
                then_text="THEN Supplier_Status = \"Reliable\"",
                else_text="ELSE Supplier_Status = \"Standard\"",
                conditions=[cond_prior, cond_fulfillment],
                logic="AND",
                matched=matched,
                outcome=label,
                text=(
                    f"✓ Rule Evaluated → IF no prior status AND FulfillmentRate({fulfillment}) >= {RULE_RELIABLE_THRESHOLD} "
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

    required = supplier_result.get("required_quantity", 0)
    annotated_candidates = [
        evaluate_supplier(c, required, trace) for c in supplier_result.get("ranked_candidates", [])
    ]
    ineligible_candidates = [
        evaluate_supplier(c, required, trace) for c in supplier_result.get("ineligible_suppliers", [])
    ]

    preferred = [c for c in annotated_candidates if c["supplier_status"] == "Preferred"]
    high_risk = [c for c in annotated_candidates if c["supplier_status"] == "High Risk"]

    recommended = next((c for c in annotated_candidates if c["supplier_status"] != "High Risk"), None)
    if recommended is None and annotated_candidates:
        recommended = annotated_candidates[0]
        trace.append(_decision_entry(
            None, None,
            f"No risk-free candidate available — defaulting to top-ranked {recommended['supplier_id']} despite High Risk flag",
            f"✓ Decision Made → No risk-free candidate available — defaulting to top-ranked "
            f"{recommended['supplier_id']} despite High Risk flag",
        ))
    elif not annotated_candidates and supplier_result.get("no_eligible_supplier"):
        combined = supplier_result.get("combined_available_qty", 0)
        text = (
            f"No supplier can fulfil RequiredQty({required}) alone — no supplier selected. "
            f"Combined availability across candidates = {combined}; "
            + ("a split order could cover it (not implemented)." if supplier_result.get("split_order_possible")
               else "even a split order could not cover it.")
        )
        trace.append(_decision_entry(None, None, text, f"✓ Decision Made → {text}"))

    return {
        "agent": "Knowledge Agent",
        "inventory_state": inventory_state,
        "annotated_candidates": annotated_candidates,
        "ineligible_candidates": ineligible_candidates,
        "preferred_suppliers": preferred,
        "high_risk_suppliers": high_risk,
        "recommended_supplier": recommended,
        "reasoning_trace": trace,
    }
