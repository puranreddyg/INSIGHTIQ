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
  R1  IF Inventory < EffectiveReorderPoint (per warehouse lane) THEN "Urgent Procurement"
      EffectiveReorderPoint = max(planner Reorder_Point, forecast reorder point)
  R5  IF AvailableQty >= RequiredQty                   ELSE "Ineligible" (stop)
  R2  IF LeadTime <= 3d AND DefectRate <= 0.05 AND Evidence >= 5 POs  THEN "Preferred"
  R3  IF DefectRate > 0.05 AND Evidence >= 5 POs                      THEN "High Risk"
  R4  IF no prior status AND FulfillmentRate >= 0.90 AND Evidence >= 5 POs THEN "Reliable"
      ELSE "Standard" (supplier has evidence) / "Unrated" (supplier has none)
  R6  IF no single supplier can cover RequiredQty
      THEN split the order across suppliers in rank order, safe suppliers first;
      if even the whole pool falls short, plan a PARTIAL order and report the shortfall

Quality rules only fire on a supplier's OWN evidence. DefectRate / FulfillmentRate
are estimates (backend/ml/supplier_estimator.py); a supplier with no purchase
history of its own sits on the industry prior, and labelling it Preferred or High
Risk from the prior alone would be a claim the data cannot support.

R2 used a KPI "Rating" that only existed through the removed KPI-vendor→SUP_x
mapping; its quality condition now reuses R3's existing defect threshold, so no
new number is introduced. R4's 0.90 is unchanged and is now read as a 90%
fulfillment business target (docs/migration: rule changes).
"""
from __future__ import annotations

from backend.config import (
    RULE_HIGH_RISK_DEFECT_RATE,
    RULE_MIN_EVIDENCE_POS,
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
    inventory = lane["Inventory"]
    planner_rop = lane["ReorderPoint"]
    forecast_rop = lane.get("ForecastReorderPoint", planner_rop)
    reorder_point = lane.get("EffectiveReorderPoint", planner_rop)
    trace.append(_fact_entry(
        entity, "sku",
        {"Inventory": inventory, "ReorderPoint": planner_rop, "ForecastReorderPoint": forecast_rop,
         "EffectiveReorderPoint": reorder_point, "StockoutProbability": lane.get("StockoutProbability"),
         "Supplier": lane["supplier_id"]},
        f"✓ Fact Detected → {entity} Inventory={inventory}, ReorderPoint={planner_rop}, "
        f"ForecastReorderPoint={forecast_rop} → EffectiveReorderPoint={reorder_point}",
    ))

    matched = inventory < reorder_point
    state = "Urgent Procurement" if matched else "Monitoring"
    label = "MATCHED" if matched else "not matched"
    trace.append(_rule_entry(
        rule_id=1,
        name="Inventory State Evaluation",
        entity=entity,
        if_text="IF Inventory < EffectiveReorderPoint",
        then_text="THEN State = \"Urgent Procurement\"",
        else_text="ELSE State = \"Monitoring\"",
        conditions=[_condition("Inventory", inventory, "<", reorder_point, matched)],
        logic=None,
        matched=matched,
        outcome=state,
        text=(
            f"✓ Rule Evaluated → IF Inventory({inventory}) < EffectiveReorderPoint({reorder_point}) "
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
    # No capacity estimate (too few historical orders) means capacity is unknown, never assumed.
    has_estimate = available is not None and available == available  # rejects None and NaN
    cond = _condition("AvailableQty", available if has_estimate else "unknown", ">=", required,
                      has_estimate and available >= required)
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
    evidence = int(candidate.get("evidence_pos") or 0)
    rated = evidence >= RULE_MIN_EVIDENCE_POS

    trace.append(_fact_entry(
        supplier_id, "supplier",
        {"UnitCost": candidate.get("unit_cost"), "LeadTime": lead_time, "AvailableQty": available,
         "RequiredQty": required_quantity, "DefectRate": defect_rate, "FulfillmentRate": fulfillment,
         "EvidencePOs": evidence},
        f"✓ Fact Detected → {supplier_id} LeadTime={lead_time}d, AvailableQty={available}, "
        f"DefectRate={defect_rate}, FulfillmentRate={fulfillment}, EvidencePOs={evidence}",
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
            cond_evidence = _condition("EvidencePOs", evidence, ">=", RULE_MIN_EVIDENCE_POS, rated)
            matched = cond_delivery["result"] and cond_quality["result"] and rated
            label = "MATCHED" if matched else "not matched"
            trace.append(_rule_entry(
                rule_id=2,
                name="Preferred Supplier",
                entity=supplier_id,
                if_text=(f"IF LeadTime <= {RULE_PREFERRED_DELIVERY_DAYS}d AND DefectRate <= {RULE_HIGH_RISK_DEFECT_RATE} "
                         f"AND EvidencePOs >= {RULE_MIN_EVIDENCE_POS}"),
                then_text="THEN Supplier_Status = \"Preferred\"",
                conditions=[cond_delivery, cond_quality, cond_evidence],
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
            cond_evidence = _condition("EvidencePOs", evidence, ">=", RULE_MIN_EVIDENCE_POS, rated)
            matched = cond_defect["result"] and rated
            label = "MATCHED" if matched else "not matched"
            trace.append(_rule_entry(
                rule_id=3,
                name="Risk Assessment",
                entity=supplier_id,
                if_text=f"IF DefectRate > {RULE_HIGH_RISK_DEFECT_RATE} AND EvidencePOs >= {RULE_MIN_EVIDENCE_POS}",
                then_text="THEN Supplier_Status = \"High Risk\"",
                conditions=[cond_defect, cond_evidence],
                logic="AND",
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
            cond_evidence = _condition("EvidencePOs", evidence, ">=", RULE_MIN_EVIDENCE_POS, rated)
            matched = cond_fulfillment["result"] and rated
            label = "Reliable" if matched else ("Standard" if rated else "Unrated")
            trace.append(_rule_entry(
                rule_id=4,
                name="Reliability / Fallback",
                entity=supplier_id,
                if_text=f"IF no prior status AND FulfillmentRate >= {RULE_RELIABLE_THRESHOLD} AND EvidencePOs >= {RULE_MIN_EVIDENCE_POS}",
                then_text="THEN Supplier_Status = \"Reliable\"",
                else_text="ELSE Supplier_Status = \"Standard\" (has evidence) or \"Unrated\" (no evidence yet)",
                conditions=[cond_prior, cond_fulfillment, cond_evidence],
                logic="AND",
                matched=matched,
                outcome=label,
                text=(
                    f"✓ Rule Evaluated → IF no prior status AND FulfillmentRate({fulfillment}) >= {RULE_RELIABLE_THRESHOLD} "
                    f"AND EvidencePOs({evidence}) >= {RULE_MIN_EVIDENCE_POS} "
                    f"THEN Supplier_Status = 'Reliable' ELSE 'Standard'/'Unrated'  [Rule 4: {label}]"
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


def choose_supplier_set(candidates: list[dict], required: int, risky_ids: set[str]) -> list[dict]:
    """Pick WHICH suppliers share an order no single supplier can cover.

    Exhaustive over subsets (a SKU has at most a handful of suppliers), choosing
    lexicographically:
      1. a set that covers the requirement (if none can, the whole pool — partial order);
      2. the fewest High Risk suppliers;
      3. the fewest suppliers (fewer purchase orders to manage);
      4. the best heuristic ranks (sum of rank numbers, lower is better).
    Returned in heuristic-rank order.
    """
    from itertools import combinations

    pool = sorted(candidates, key=lambda c: c.get("rank", 0))
    if sum(int(c["available_qty"]) for c in pool) < required:
        return pool
    best_key, best = None, None
    for k in range(1, len(pool) + 1):
        for subset in combinations(pool, k):
            if sum(int(c["available_qty"]) for c in subset) < required:
                continue
            key = (sum(c["supplier_id"] in risky_ids for c in subset), k, sum(c.get("rank", 0) for c in subset))
            if best_key is None or key < best_key:
                best_key, best = key, list(subset)
    return best


def allocate_split_order(candidates: list[dict], required_quantity: float, trace: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Rule 6 (Split Order) — used only when no single supplier can cover the request.

      R6  IF no supplier has AvailableQty >= RequiredQty
          THEN choose the supplier set (choose_supplier_set) and fill it in
          heuristic-rank order, taking min(AvailableQty, remaining) from each.
          IF even the combined pool falls short, every supplier is used to its
          capacity (a PARTIAL order) and the shortfall is reported.

    Returns (allocations, unallocated, shortfall). Each allocation is the supplier's
    fully annotated record (R5, R2-R4 evaluated for the quantity it is asked to
    supply) plus `allocated_qty` / `allocation_order`. Unallocated suppliers keep
    their status so the Planning Agent can pick a backup from them.
    """
    required = int(required_quantity)
    # Status depends only on defect rate / lead time / fulfilment / evidence, not on
    # the quantity, so classify quietly first (scratch trace) to choose the set.
    scratch: list[dict] = []
    status_by_id = {c["supplier_id"]: evaluate_supplier(c, 0, scratch)["supplier_status"] for c in candidates}
    risky = {sid for sid, st in status_by_id.items() if st == "High Risk"}
    chosen = choose_supplier_set(candidates, required, risky)
    chosen_ids = {c["supplier_id"] for c in chosen}

    plan_shares: list[tuple[dict, int]] = []
    remaining = required
    for c in chosen:
        share = int(min(int(c["available_qty"]), remaining))
        if share > 0:
            plan_shares.append((c, share))
            remaining -= share
    unallocated = [{**c, "supplier_status": status_by_id[c["supplier_id"]]}
                   for c in candidates if c["supplier_id"] not in {x["supplier_id"] for x, _ in plan_shares}]

    covered = sum(share for _, share in plan_shares)
    combined = sum(int(c["available_qty"]) for c in candidates)
    shortfall = max(0, required - covered)
    full = shortfall == 0
    text = (
        f"No supplier can fulfil RequiredQty({required}) alone — "
        + (f"splitting it across {len(plan_shares)} suppliers (fewest suppliers and fewest High Risk ones, "
           f"then best rank), filled in rank order."
           if full else
           f"even all {len(candidates)} suppliers together have only {combined}. Planning a PARTIAL order of "
           f"{covered} units; shortfall {shortfall} units must be sourced elsewhere.")
    )
    trace.append({
        "type": "rule", "ruleId": 6, "name": "Split Order" if full else "Partial Split Order",
        "entity": "Order",
        "ifText": "IF no single supplier has AvailableQty >= RequiredQty",
        "thenText": ("THEN split across the smallest safe set of suppliers that covers it; "
                     "if the whole pool falls short, order everything available and report the shortfall"),
        "elseText": None,
        "conditions": [
            _condition("Combined AvailableQty", combined, ">=", required, combined >= required),
        ],
        "logic": None, "matched": True, "outcome": "Split Order" if full else "Partial Order",
        "text": f"✓ Rule Evaluated → {text}  [Rule 6: MATCHED]",
    })

    allocations: list[dict] = []
    for i, (c, share) in enumerate(plan_shares, start=1):
        annotated = evaluate_supplier(c, share, trace)
        annotated["allocated_qty"] = share
        annotated["allocation_order"] = i
        allocations.append(annotated)

    summary = " + ".join(f"{a['supplier_id']} {a['allocated_qty']}" for a in allocations) + f" = {covered} units"
    if not full:
        summary += f" (shortfall {shortfall})"
    trace.append(_decision_entry(None, None, f"Split order: {summary}", f"✓ Decision Made → Split order: {summary}"))
    return allocations, unallocated, shortfall


def run(perception_result: dict, supplier_result: dict) -> dict:
    trace: list[dict] = []

    inventory_state = None
    focus_facts = perception_result.get("focus_facts") or []
    if focus_facts:
        inventory_state = evaluate_inventory_state(focus_facts[0], trace)

    required = supplier_result.get("required_quantity", 0)
    split_mode = bool(
        not supplier_result.get("ranked_candidates")
        and (supplier_result.get("split_order_possible") or supplier_result.get("partial_order_possible"))
        and supplier_result.get("split_candidates")
    )

    allocations: list[dict] = []
    backup_candidates: list[dict] = []
    shortfall = 0
    if split_mode:
        # No single supplier is eligible, so the single-supplier capacity rule (R5) is
        # not applied to the pool as a whole; each allocated share is checked instead.
        allocations, backup_candidates, shortfall = allocate_split_order(supplier_result["split_candidates"], required, trace)
        annotated_candidates = allocations
        ineligible_candidates: list[dict] = []
    else:
        annotated_candidates = [
            evaluate_supplier(c, required, trace) for c in supplier_result.get("ranked_candidates", [])
        ]
        ineligible_candidates = [
            evaluate_supplier(c, required, trace) for c in supplier_result.get("ineligible_suppliers", [])
        ]

    preferred = [c for c in annotated_candidates if c["supplier_status"] == "Preferred"]
    high_risk = [c for c in annotated_candidates if c["supplier_status"] == "High Risk"]

    if split_mode:
        # The lead supplier of a split order is the first allocation (largest priority).
        recommended = allocations[0] if allocations else None
    else:
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
            f"No supplier can fulfil RequiredQty({required}) and no candidate has any capacity "
            f"(combined availability = {combined}) — no supplier selected."
        )
        trace.append(_decision_entry(None, None, text, f"✓ Decision Made → {text}"))

    return {
        "agent": "Knowledge Agent",
        "inventory_state": inventory_state,
        "split_order": split_mode,
        "partial_order": bool(split_mode and shortfall > 0),
        "shortfall_qty": shortfall,
        "allocations": allocations,
        "backup_candidates": backup_candidates,
        "annotated_candidates": annotated_candidates,
        "ineligible_candidates": ineligible_candidates,
        "preferred_suppliers": preferred,
        "high_risk_suppliers": high_risk,
        "recommended_supplier": recommended,
        "reasoning_trace": trace,
    }
