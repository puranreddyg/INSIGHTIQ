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

from backend.agents import procurement_search
from backend.config import (
    MIN_SPLIT_LINE_QTY,
    RULE_HIGH_RISK_DEFECT_RATE,
    SEARCH_MAX_SUPPLIERS,
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


def _risky_ids(candidates: list[dict]) -> tuple[set[str], dict[str, str]]:
    scratch: list[dict] = []
    status_by_id = {c["supplier_id"]: evaluate_supplier(c, 0, scratch)["supplier_status"] for c in candidates}
    return {sid for sid, st in status_by_id.items() if st == "High Risk"}, status_by_id


def allocate_split_order(candidates: list[dict], required_quantity: float, trace: list[dict],
                         budget: float | None = None, out: dict | None = None) -> tuple[list[dict], list[dict], int]:
    """Rule 6 (Multi-supplier order) — hybrid heuristic search (procurement_search.py).

      R6  IF no single supplier meets capacity, quality AND budget
          THEN search supplier combinations for a plan meeting ALL hard constraints:
          capacity (>= MIN_SPLIT_LINE_QTY per line), full quantity, weighted predicted
          defect rate <= 5%, total cost <= budget, <= SEARCH_MAX_SUPPLIERS suppliers.
          IF even all suppliers together lack capacity: buy the most that still meets
          quality, budget and size (PARTIAL) and report the shortfall.
          IF no plan meets the constraints: no split (status "infeasible").

    Returns (allocations, unallocated, shortfall). allocations is empty when infeasible;
    the search summary (or failure reason) is attached under allocations[0]["search"] /
    the trace.
    """
    required = int(required_quantity)
    risky, status_by_id = _risky_ids(candidates)
    combined = sum(int(c["available_qty"]) for c in candidates)
    if combined >= required:
        result = procurement_search.search(candidates, required, risky, budget)
    else:
        result = procurement_search.partial(candidates, required, risky, budget)
    if out is not None:
        out["search_result"] = result
    status = result["status"]
    plan_shares = result["shares"]
    chosen_ids = {c["supplier_id"] for c, _ in plan_shares}
    unallocated = [{**c, "supplier_status": status_by_id[c["supplier_id"]]} for c in candidates if c["supplier_id"] not in chosen_ids]
    covered = sum(q for _, q in plan_shares)
    shortfall = int(result["shortfall"])
    budget_txt = f"cost <= ₹{budget:,.2f}" if budget is not None else "no budget limit"
    if status == "full":
        text = (f"No single supplier meets every constraint for RequiredQty({required}) — the search found "
                f"{len(plan_shares)} suppliers that together do: defect rate {result['weighted_defect_rate']:.2%} <= "
                f"{RULE_HIGH_RISK_DEFECT_RATE:.0%}, cost ₹{result['total_cost']:,.2f} ({budget_txt}).")
    elif status == "partial":
        text = (f"Even all {len(candidates)} suppliers together have only {combined} units. The most that can be bought "
                f"within quality and budget is {covered} units (PARTIAL); shortfall {shortfall} units.")
    else:
        text = f"No supplier combination can be used: {result['reason']}. The order is NOT split."
    trace.append({
        "type": "rule", "ruleId": 6,
        "name": {"full": "Multi-Supplier Order", "partial": "Partial Multi-Supplier Order",
                 "infeasible": "Multi-Supplier Order (constraints cannot be met)"}[status],
        "entity": "Order",
        "ifText": "IF no single supplier meets capacity, quality and budget",
        "thenText": (f"THEN search supplier combinations meeting ALL of: capacity (>= {MIN_SPLIT_LINE_QTY} per line), full quantity, "
                     f"weighted defect rate <= {RULE_HIGH_RISK_DEFECT_RATE:.0%}, {budget_txt}, <= {SEARCH_MAX_SUPPLIERS} suppliers"),
        "elseText": None,
        "conditions": [_condition("Combined AvailableQty", combined, ">=", required, combined >= required)],
        "logic": None, "matched": status != "infeasible",
        "outcome": {"full": "Split Order", "partial": "Partial Order", "infeasible": "No Split"}[status],
        "text": (f"✓ Rule Evaluated → {text} Search expanded {result['nodes_expanded']} nodes, evaluated "
                 f"{result['plans_evaluated']} supplier sets, pruned {result['nodes_pruned']}.  "
                 f"[Rule 6: {'MATCHED' if status != 'infeasible' else 'not matched'}]"),
    })
    allocations: list[dict] = []
    for order, (c, share) in enumerate(plan_shares, start=1):
        annotated = evaluate_supplier(c, share, trace)
        annotated["allocated_qty"] = share
        annotated["allocation_order"] = order
        allocations.append(annotated)
    summary = {k: result.get(k) for k in ("status", "reason", "nodes_expanded", "nodes_pruned", "plans_evaluated", "constraints")}
    if allocations:
        summary.update({"weighted_defect_rate": round(result["weighted_defect_rate"], 4),
                        "weighted_otif": round(result["weighted_otif"], 4), "total_cost": result["total_cost"],
                        "meets_quality": result["weighted_defect_rate"] <= RULE_HIGH_RISK_DEFECT_RATE + 1e-9,
                        "meets_budget": budget is None or result["total_cost"] <= budget + 0.01})
        allocations[0]["search"] = summary
        line = " + ".join(f"{a['supplier_id']} {a['allocated_qty']}" for a in allocations) + f" = {covered} units"
        if shortfall:
            line += f" (shortfall {shortfall})"
        trace.append(_decision_entry(None, None, f"Split order: {line}", f"✓ Decision Made → Split order: {line}"))
    else:
        unallocated = [{**c, "supplier_status": status_by_id[c["supplier_id"]]} for c in candidates]
        unallocated and unallocated[0].setdefault("_search_failure", summary)
    return allocations, unallocated, shortfall


def _budget_rule(c: dict, required: float, budget: float | None, trace: list[dict]) -> bool:
    """R7 Budget: the supplier's cost for the whole order must fit the budget."""
    if budget is None:
        return True
    cost = round(required * float(c.get("unit_cost") or 0), 2)
    ok = cost <= budget + 0.01
    trace.append(_rule_entry(
        rule_id=7, name="Budget Check", entity=c["supplier_id"],
        if_text="IF RequiredQty x UnitCost <= Budget", then_text="THEN within budget",
        else_text="ELSE over budget (cannot be the sole supplier)",
        conditions=[_condition("OrderCost", cost, "<=", budget, ok)], logic=None, matched=ok,
        outcome=None if ok else "Over Budget",
        text=f"✓ Rule Evaluated → IF OrderCost(₹{cost:,.2f}) <= Budget(₹{budget:,.2f})  [Rule 7: {'MATCHED' if ok else 'not matched'}]",
    ))
    return ok


def _rule_step(entry: dict) -> dict:
    """Compact, UI-ready view of one rule firing (from the trace entry the rule itself wrote)."""
    return {"rule_id": f"R{entry['ruleId']}", "rule_name": entry["name"], "if": entry.get("ifText"), "then": entry.get("thenText"),
            "inputs": {c["label"]: c["value"] for c in entry["conditions"]},
            "conditions": [{k: c[k] for k in ("label", "value", "operator", "threshold", "result", "expression")} for c in entry["conditions"]],
            "logic": entry.get("logic"), "result": "MATCHED" if entry["matched"] else "NOT MATCHED", "outcome": entry.get("outcome")}


def _evaluate_eligible_in_order(eligible: list[dict], required: float, budget: float | None, trace: list[dict]) -> tuple[list[dict], dict | None, list[dict]]:
    """Evaluate the suppliers able to supply the full order ONE BY ONE in heuristic order and stop
    selecting at the first that passes the risk gate (not High Risk) and the budget gate (R7).
    Each supplier's facts, R5/R2/R3/R4 firings, status and R7 are written to the trace together,
    so every rule is attached to the supplier it was evaluated for. Suppliers ranked after the
    selected one are still annotated (phase `backup_evaluation`) because the plan lists backups,
    but they play no part in the selection."""
    annotated: list[dict] = []
    evaluations: list[dict] = []
    selected: dict | None = None
    for step, cand in enumerate(eligible, start=1):
        phase = "selection" if selected is None else "backup_evaluation"
        start = len(trace)
        ann = evaluate_supplier(cand, required, trace)
        budget_ok = _budget_rule(ann, required, budget, trace) if required else True
        for e in trace[start:]:
            e["phase"], e["evalStep"] = phase, step
        rules = [_rule_step(e) for e in trace[start:] if e["type"] == "rule"]
        status = ann["supplier_status"]
        risk_ok = status != "High Risk"
        cost = round(float(required) * float(ann.get("unit_cost") or 0), 2) if required else 0.0
        budget_on = budget is not None and bool(required)
        reasons = []
        if not risk_ok:
            reasons.append(f"High Risk: predicted defect rate {float(ann.get('defect_rate')):.4f} > {RULE_HIGH_RISK_DEFECT_RATE} "
                           f"with {int(ann.get('evidence_pos') or 0)} purchase orders of evidence (>= {RULE_MIN_EVIDENCE_POS})")
        if not budget_ok:
            reasons.append(f"Over budget: order cost {cost:,.2f} > budget {budget:,.2f}")
        accepted = risk_ok and budget_ok
        budget_detail = (f"order cost {cost:,.2f} {'<=' if budget_ok else '>'} budget {budget:,.2f}" if budget_on else "no budget limit set")
        gates = [
            {"gate": "Capacity (R5)", "result": "PASS", "detail": f"available {ann['available_qty']} >= required {int(required)}"},
            {"gate": "Risk (R3)", "result": "PASS" if risk_ok else "FAIL",
             "detail": f"status {status}" + ("" if risk_ok else " — High Risk is never selected as a sole supplier")},
            {"gate": "Budget (R7)", "result": ("PASS" if budget_ok else "FAIL") if budget_on else "NOT APPLICABLE", "detail": budget_detail},
        ]
        rec = {"step": step, "phase": phase, "supplier_id": ann["supplier_id"], "rank": ann.get("rank"), "heuristic_score": ann.get("heuristic_score"),
               "available_qty": ann.get("available_qty"), "unit_cost": ann.get("unit_cost"), "order_cost": cost, "defect_rate": ann.get("defect_rate"),
               "evidence_pos": int(ann.get("evidence_pos") or 0), "status": status, "status_flags": ann.get("status_flags"),
               "rules": rules, "gates": gates, "accepted": accepted, "reasons": reasons}
        if phase == "selection":
            if accepted:
                selected = ann
                rec["outcome"], rec["flow"] = "SELECTED", "STOP — first eligible supplier passing the risk and budget checks"
                tail = f" and costs {cost:,.2f} within the budget of {budget:,.2f}." if budget_on else "."
                rec["decision_reason"] = f"{ann['supplier_id']} (rank {ann.get('rank')}) can supply the full order, is {status} (not High Risk){tail}"
            else:
                rec["outcome"], rec["flow"] = "REJECTED", "CONTINUE — evaluate the next eligible supplier"
                rec["decision_reason"] = f"{ann['supplier_id']} rejected: " + "; ".join(reasons) + "."
        else:
            rec["outcome"], rec["flow"] = "NOT NEEDED FOR SELECTION", "Selection already made; evaluated only to list backup suppliers"
            rec["decision_reason"] = "Would also pass the checks." if accepted else "Would not pass the checks: " + "; ".join(reasons) + "."
        evaluations.append(rec)
        annotated.append(ann)
    return annotated, selected, evaluations


def run(perception_result: dict, supplier_result: dict) -> dict:
    trace: list[dict] = []

    inventory_state = None
    focus_facts = perception_result.get("focus_facts") or []
    if focus_facts:
        inventory_state = evaluate_inventory_state(focus_facts[0], trace)

    required = supplier_result.get("required_quantity", 0)
    budget = supplier_result.get("budget")
    pool = supplier_result.get("pool_candidates") or supplier_result.get("split_candidates") or []
    eligible = supplier_result.get("ranked_candidates", [])

    # Steps 5-8: chronological evaluation of the eligible single suppliers, heuristic order, stop at the first pass.
    annotated_candidates, recommended, evaluations = _evaluate_eligible_in_order(eligible, required, budget, trace)
    ineligible_start = len(trace)
    ineligible_candidates = [evaluate_supplier(c, required, trace) for c in supplier_result.get("ineligible_suppliers", [])]
    for e in trace[ineligible_start:]:
        e["phase"] = "capacity_screen"

    allocations: list[dict] = []
    backup_candidates: list[dict] = []
    shortfall = 0
    split_mode = False
    decision = "no_action"
    review_reason = None
    quality_ok = budget_ok = True
    search_out: dict = {}

    ranking = supplier_result.get("scored_ranking") or pool
    eligible_ids = {c["supplier_id"] for c in eligible}
    eval_block = {
        "required_quantity": required, "budget": budget,
        "complete_ranking": [{"rank": c.get("rank"), "supplier_id": c["supplier_id"], "heuristic_score": c.get("heuristic_score"),
                              "available_qty": c.get("available_qty"), "eligible": c["supplier_id"] in eligible_ids} for c in ranking],
        "eligible_list": [{"rank": c.get("rank"), "supplier_id": c["supplier_id"], "heuristic_score": c.get("heuristic_score"),
                           "available_qty": c.get("available_qty")} for c in eligible],
        "capacity_ineligible": [c["supplier_id"] for c in supplier_result.get("ineligible_suppliers", [])],
        "eligible_empty_message": None if eligible else "No supplier has sufficient capacity to fulfil the complete order individually.",
        "evaluations": evaluations, "selected_supplier_id": recommended["supplier_id"] if recommended else None,
        "result_legend": "Rules R2-R4 report MATCHED / NOT MATCHED (did the rule fire); the capacity, risk and budget gates report PASS / FAIL.",
        "case": None, "headline": None, "reasons": [], "transition_message": None,
    }

    if recommended is not None:
        decision = "single" if required else "no_action"  # nothing to buy: the pick is informational only
        eval_block["case"] = "SELECTED" if required else "NO_REQUIREMENT"
        eval_block["headline"] = f"Single supplier selected: {recommended['supplier_id']}." if required else "No quantity is required."
    elif required and eligible:
        eval_block["case"] = "B"
        eval_block["headline"] = "NO ACCEPTABLE SINGLE-SUPPLIER SOLUTION: Suppliers were available, but none satisfied all required selection constraints."
        eval_block["reasons"] = [f"{e['supplier_id']} (rank {e['rank']}): " + "; ".join(e["reasons"]) for e in evaluations if not e["accepted"]]
    elif required and pool:
        eval_block["case"] = "A"
        eval_block["headline"] = "NO SINGLE-SUPPLIER SOLUTION: None of the suppliers has enough available capacity to fulfil the complete required quantity individually."
        eval_block["reasons"] = [f"{c['supplier_id']} (rank {c.get('rank')}): available {c['available_qty']} < required {int(required)} "
                                 f"(shortfall {int(required - c['available_qty'])})" for c in supplier_result.get("ineligible_suppliers", [])]
    if eval_block["case"] in ("A", "B"):
        trace.append({**_decision_entry(None, None, eval_block["headline"], f"✓ Decision Made → {eval_block['headline']}"), "phase": "single_supplier_outcome"})

    if required and recommended is None and pool:
        eval_block["transition_message"] = "Starting multi-supplier allocation because no valid single-supplier solution was found."
        trace.append({**_decision_entry(None, None, eval_block["transition_message"], f"✓ Decision Made → {eval_block['transition_message']}"),
                      "phase": "single_supplier_outcome"})
        allocations, backup_candidates, shortfall = allocate_split_order(pool, required, trace, budget, out=search_out)
        if allocations:
            split_mode = True
            decision = "partial" if shortfall else "split"
            recommended = allocations[0]
            annotated_candidates = allocations
            ineligible_candidates = []
        elif annotated_candidates:
            # No split meets every constraint, but some supplier can cover the order alone:
            # choose the smallest violation (quality first, then budget) and flag it for review.
            def violation(c):
                d = max(0.0, float(c.get("defect_rate") or 1.0) - RULE_HIGH_RISK_DEFECT_RATE)
                over = max(0.0, required * float(c.get("unit_cost") or 0) - (budget if budget is not None else float("inf")))
                return (round(d, 6), over, c.get("rank", 0))
            recommended = min(annotated_candidates, key=violation)
            quality_ok = recommended["supplier_status"] != "High Risk"
            budget_ok = budget is None or required * float(recommended["unit_cost"]) <= budget + 0.01
            decision = "single_flagged"
            failed = [n for n, ok in (("quality", quality_ok), ("budget", budget_ok)) if not ok]
            text = (f"No supplier or combination meets every constraint — choosing {recommended['supplier_id']}, the smallest "
                    f"violation ({', '.join(failed)} not met); flagged for review")
            trace.append(_decision_entry(None, None, text, f"✓ Decision Made → {text}"))
        else:
            failure = (backup_candidates[0].get("_search_failure") if backup_candidates else None) or {}
            review_reason = failure.get("reason") or "no supplier plan meets the constraints"
            decision = "needs_review"
            quality_ok = budget_ok = False
            text = f"No supplier plan meets the constraints ({review_reason}) — order needs manual review."
            trace.append(_decision_entry(None, None, text, f"✓ Decision Made → {text}"))
    elif required and not annotated_candidates and not pool:
        decision = "needs_review"
        review_reason = "no supplier has known capacity for this SKU"
        trace.append(_decision_entry(None, None, review_reason, f"✓ Decision Made → {review_reason}"))

    for c in backup_candidates:
        c.pop("_search_failure", None)
    if split_mode and allocations:
        srch = allocations[0].get("search", {})
        quality_ok, budget_ok = bool(srch.get("meets_quality", True)), bool(srch.get("meets_budget", True))

    search_trace = (search_out.get("search_result") or {}).get("trace")
    preferred = [c for c in annotated_candidates if c["supplier_status"] == "Preferred"]
    high_risk = [c for c in annotated_candidates if c["supplier_status"] == "High Risk"]

    return {
        "agent": "Knowledge Agent",
        "inventory_state": inventory_state,
        "decision": decision,  # single | split | partial | single_flagged | needs_review | no_action
        "review_reason": review_reason,
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
        "single_supplier_evaluation": eval_block,
        "search_trace": search_trace,
        "budget": budget,
        "quality_constraint_met": quality_ok,
        "budget_constraint_met": budget_ok,
        "reasoning_trace": trace,
    }
