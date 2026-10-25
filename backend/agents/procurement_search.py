"""
Hybrid heuristic search for multi-supplier orders (Knowledge Agent Rule 6).

"Hybrid" = the LEARNED supplier predictions (defect rate, on-time-in-full, lead time)
feed each supplier's weighted HEURISTIC score; the score guides a best-first
branch-and-bound SEARCH over supplier sets; inside each set the quantities are
solved EXACTLY by a small integer linear program.

Hard constraints (a plan is only returned if it meets ALL of them)
  C1 capacity   min_line <= q_i <= capacity_i for every supplier used
  C2 quantity   sum q_i = required quantity
  C3 quality    sum q_i * defect_i <= 5% * sum q_i       (quantity-weighted predicted defect rate)
  C4 budget     sum q_i * unit_cost_i <= budget
  C5 size       at most SEARCH_MAX_SUPPLIERS suppliers
Objective (lexicographic)
  1. fewest High Risk suppliers  2. fewest suppliers  3. highest quantity-weighted score
Search     frontier ordered by (set size, optimistic bound); bound = best score
           reachable from a node, which no allocation can exceed (admissible). Once a
           risk-free incumbent exists, larger sets and same-size sets whose bound
           cannot beat it are pruned.

If no set satisfies C1-C5 the result is "infeasible" with the reason; the caller
must NOT split the order. `partial()` handles the case where even all suppliers
together lack capacity: it maximises the quantity that CAN be bought while still
meeting C1, C3, C4 and C5.

TRACE. Both entry points return a `trace` recorded INSIDE the algorithm as it runs:
every event (node created / queued / expanded, combination evaluated, solver
invoked, combination rejected, branch pruned, feasible allocation, new best,
search completed) is appended at the moment it happens, with a monotonically
increasing step number. Nothing is reconstructed afterwards, and the summary
counters are incremented by the same operations that emit the events. A
constraint that was never evaluated for a combination is reported as
"NOT YET EVALUATED", never as passed or failed. The integer solver is a black
box: its inputs and outputs are recorded, not an invented internal sequence.
"""
from __future__ import annotations

import heapq
from itertools import count

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from backend.config import MIN_SPLIT_LINE_QTY, RULE_HIGH_RISK_DEFECT_RATE, SEARCH_MAX_SUPPLIERS

NOT_EVALUATED = "NOT YET EVALUATED"


def _score(c: dict, n: int) -> float:
    if c.get("heuristic_score") is not None:
        return float(c["heuristic_score"]) / 100.0
    return 1.0 - (float(c.get("rank", n)) - 1.0) / max(n, 1)  # rank fallback, 1.0 for rank 1


def _arrays(group: list[dict]):
    cap = np.array([float(c["available_qty"]) for c in group])
    d = np.array([float(c["defect_rate"]) for c in group])
    cost = np.array([float(c["unit_cost"]) for c in group])
    return cap, d, cost


def _min_line(group_size: int, required: int) -> float:
    return float(np.minimum(float(MIN_SPLIT_LINE_QTY), float(required))) if group_size == 1 else float(MIN_SPLIT_LINE_QTY)


def allocate_detail(group: list[dict], required: int, budget: float | None, scores: dict) -> dict:
    """Exact integer allocation of `required` units over ALL suppliers in `group`
    (each gets >= the minimum line) meeting C1-C4, maximising the weighted score.

    Returns {"shares": [(candidate, qty)] | None, "reason": None | code, "detail": text}.
    Reason codes: min_line (a capacity below the minimum line), capacity (combined
    capacity below the requirement), min_line_total (suppliers x minimum line exceeds
    the requirement), infeasible (the solver found no integer allocation)."""
    cap, d, cost = _arrays(group)
    lo = _min_line(len(group), required)
    if (cap < lo).any():
        low = [c["supplier_id"] for c, k in zip(group, cap) if k < lo]
        return {"shares": None, "reason": "min_line", "detail": f"{', '.join(low)} has capacity below the minimum line of {int(lo)} units"}
    if cap.sum() < required:
        return {"shares": None, "reason": "capacity", "detail": f"combined capacity {int(cap.sum())} is below the required {required}"}
    if lo * len(group) > required:
        return {"shares": None, "reason": "min_line_total", "detail": f"{len(group)} suppliers x minimum line {int(lo)} = {int(lo) * len(group)} exceeds the required {required}"}
    rows = [np.ones(len(group)), d - RULE_HIGH_RISK_DEFECT_RATE]
    lb, ub = [required, -np.inf], [required, 1e-9]
    if budget is not None:
        rows.append(cost); lb.append(-np.inf); ub.append(budget + 1e-6)
    s = np.array([scores[c["supplier_id"]] for c in group])
    res = milp(c=-s, constraints=LinearConstraint(np.vstack(rows), lb, ub),
               integrality=np.ones(len(group)), bounds=Bounds(np.full(len(group), lo), cap))
    if not res.success:
        return {"shares": None, "reason": "infeasible", "detail": "the integer solver found no allocation meeting all included constraints"}
    q = np.round(res.x).astype(int)
    return {"shares": [(c, int(x)) for c, x in zip(group, q)], "reason": None, "detail": "optimal integer allocation found"}


def allocate(group: list[dict], required: int, budget: float | None, scores: dict) -> list[tuple[dict, int]] | None:
    """Shares-only view of `allocate_detail` (None if infeasible)."""
    return allocate_detail(group, required, budget, scores)["shares"]


def _min_blended_defect(group: list[dict], required: int) -> float | None:
    """Lowest blended defect rate any allocation can reach within capacity and the
    minimum line (continuous relaxation, filled cleanest-first). A lower bound."""
    cap, d, _ = _arrays(group)
    lo = _min_line(len(group), required)
    if (cap < lo).any() or cap.sum() < required or lo * len(group) > required:
        return None
    q = np.full(len(group), lo)
    left = required - q.sum()
    for i in np.argsort(d, kind="stable"):
        add = min(left, cap[i] - q[i]); q[i] += add; left -= add
    return float((q * d).sum() / required)


def _min_cost(group: list[dict], required: int) -> float | None:
    """Cheapest spend any allocation can reach within capacity and the minimum line
    (continuous relaxation, filled cheapest-first), ignoring quality."""
    cap, _, cost = _arrays(group)
    lo = _min_line(len(group), required)
    if (cap < lo).any() or cap.sum() < required or lo * len(group) > required:
        return None
    q = np.full(len(group), lo)
    left = required - q.sum()
    for i in np.argsort(cost, kind="stable"):
        add = min(left, cap[i] - q[i]); q[i] += add; left -= add
    return float((q * cost).sum())


def _metrics(shares, required, scores, risky) -> dict:
    qty = sum(q for _, q in shares)
    spend = sum(q * float(c["unit_cost"]) for c, q in shares)
    return {
        "quantity": qty,
        "weighted_defect_rate": sum(q * float(c["defect_rate"]) for c, q in shares) / qty if qty else 0.0,
        "weighted_otif": sum(q * float(c.get("fulfillment_rate") or 0) for c, q in shares) / qty if qty else 0.0,
        "total_cost": round(spend, 2),
        "value": sum(q * scores[c["supplier_id"]] for c, q in shares) / required if required else 0.0,
        "risky": sum(c["supplier_id"] in risky for c, _ in shares),
    }


def _constraints(budget):
    return {"max_defect_rate": RULE_HIGH_RISK_DEFECT_RATE, "max_suppliers": SEARCH_MAX_SUPPLIERS,
            "min_line_qty": MIN_SPLIT_LINE_QTY, "budget": None if budget is None else round(float(budget), 2)}


# ---------------------------------------------------------------------------
# Trace recorder — events are appended by the algorithm at the moment they occur
# ---------------------------------------------------------------------------

class _Trace:
    def __init__(self, mode: str, pool: list[dict], required: int, budget, risky_ids: set, scores: dict):
        self.mode, self.required, self.budget, self.risky, self.scores = mode, required, budget, risky_ids, scores
        self.events: list[dict] = []
        self.c = dict(nodes_created=0, nodes_expanded=0, branches_pruned=0, combinations_evaluated=0, combinations_rejected=0,
                      feasible_allocations=0, solver_invocations=0)
        self.feasible: list[dict] = []
        self.pool_info = {x["supplier_id"]: self._member(x) for x in pool}

    # -- helpers -------------------------------------------------------
    def _member(self, c: dict) -> dict:
        return {"supplier_id": c["supplier_id"], "heuristic_rank": c.get("rank"), "heuristic_score": c.get("heuristic_score"),
                "available_qty": int(c["available_qty"]), "defect_rate": round(float(c["defect_rate"]), 4),
                "unit_cost": round(float(c["unit_cost"]), 3), "high_risk": c["supplier_id"] in self.risky}

    def combo(self, group: list[dict]) -> dict:
        ids = [c["supplier_id"] for c in group]
        return {"key": " + ".join(ids) if ids else "(empty set)", "ids": ids, "size": len(ids), "members": [self.pool_info[i] for i in ids]}

    def capacity(self, group: list[dict]) -> dict:
        combined = int(sum(c["available_qty"] for c in group))
        return {"required": self.required, "per_supplier": {c["supplier_id"]: int(c["available_qty"]) for c in group},
                "combined": combined, "sufficient": combined >= self.required, "status": "PASS" if combined >= self.required else "FAIL",
                "reason": None if combined >= self.required else f"combined capacity {combined} < required {self.required}"}

    def add(self, etype: str, message: str, group: list[dict] | None = None, **fields) -> dict:
        ev = {"step": len(self.events) + 1, "type": etype, "phase": "multi_supplier_search", "message": message}
        if group is not None:
            ev["combination"] = self.combo(group)
        ev.update(fields)
        self.events.append(ev)
        return ev

    @staticmethod
    def not_evaluated(name: str, extra: dict | None = None) -> dict:
        return {"status": NOT_EVALUATED, "constraint": name, **(extra or {})}

    def allocation_block(self, shares, metrics) -> dict:
        lines = [{"supplier_id": c["supplier_id"], "heuristic_rank": c.get("rank"), "heuristic_score": c.get("heuristic_score"),
                  "allocated_qty": q, "available_qty": int(c["available_qty"]), "unit_cost": round(float(c["unit_cost"]), 3),
                  "line_cost": round(q * float(c["unit_cost"]), 2), "defect_rate": round(float(c["defect_rate"]), 4),
                  "high_risk": c["supplier_id"] in self.risky} for c, q in shares]
        return {"lines": lines, "total_quantity": metrics["quantity"], "total_cost": metrics["total_cost"],
                "weighted_defect_rate": round(metrics["weighted_defect_rate"], 4), "weighted_score": round(metrics["value"], 4),
                "high_risk_suppliers": metrics["risky"], "suppliers_used": len(shares)}

    def objective(self, metrics, n_suppliers) -> dict:
        return {"priority_1_high_risk_suppliers": metrics["risky"], "priority_2_suppliers_used": n_suppliers,
                "priority_3_weighted_score": round(metrics["value"], 4), "comparison_key": [metrics["risky"], n_suppliers, round(-metrics["value"], 4)]}

    def finish(self, status: str, termination: str, best) -> dict:
        shares = best[1] if best else []
        final = None
        if best:
            m = best[2]
            alternatives = [f for f in self.feasible if f["combination_key"] != self.combo([c for c, _ in shares])["key"]]
            why = []
            for alt in alternatives:
                a, b = alt["objective"]["comparison_key"], self.objective(m, len(shares))["comparison_key"]
                names = ["number of High Risk suppliers", "number of suppliers", "quantity-weighted heuristic score (higher is better)"]
                for i in range(3):
                    if a[i] != b[i]:
                        why.append({"alternative": alt["combination_key"], "decided_by": names[i],
                                    "selected": b[i] if i < 2 else round(-b[i], 4), "alternative_value": a[i] if i < 2 else round(-a[i], 4)})
                        break
            final = {"allocation": self.allocation_block(shares, m), "objective": self.objective(m, len(shares)),
                     "constraints_passed": {"capacity": True, "total_equals_required": m["quantity"] == self.required,
                                            "quality": m["weighted_defect_rate"] <= RULE_HIGH_RISK_DEFECT_RATE + 1e-9,
                                            "budget": self.budget is None or m["total_cost"] <= self.budget + 0.01,
                                            "max_suppliers": len(shares) <= SEARCH_MAX_SUPPLIERS},
                     "shortfall": self.required - m["quantity"], "alternatives_evaluated": len(alternatives),
                     "preferred_over": why}
        summary = {**self.c, "termination_reason": termination, "result_type": status,
                   "search_limits": {"max_suppliers_per_combination": SEARCH_MAX_SUPPLIERS, "min_line_qty": MIN_SPLIT_LINE_QTY,
                                     "node_limit": None, "time_limit": None, "limit_reached": False}}
        self.add("search_completed", f"Search completed: {status}. {termination}", summary=summary)
        return {"mode": self.mode, "required_quantity": self.required, "events": self.events, "summary": summary, "final": final,
                "event_types": ["search_started", "node_created", "node_queued", "node_expanded", "combination_evaluated", "solver_invoked",
                                "combination_rejected", "branch_pruned", "feasible_allocation", "new_best", "search_completed"]}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(candidates: list[dict], required_quantity: float, risky_ids: set[str], budget: float | None = None) -> dict:
    """Best plan meeting C1-C5, or status 'infeasible' (with diagnostics) if none exists."""
    required = int(required_quantity)
    pool = sorted(candidates, key=lambda c: (-_score(c, len(candidates)), c.get("rank", 0)))
    n = len(pool)
    scores = {c["supplier_id"]: _score(c, n) for c in pool}
    best = None  # (key, shares, metrics); key = (risky, suppliers, -value)
    expanded = pruned = evaluated = 0
    capacity_ok = quality_ok = False  # diagnostics: did any set pass capacity / capacity+quality?
    tie = count()
    tr = _Trace("branch_and_bound", pool, required, budget, risky_ids, scores)
    tr.add("search_started",
           f"Searching combinations of up to {SEARCH_MAX_SUPPLIERS} suppliers from a pool of {n} scored suppliers, in heuristic order, "
           f"to cover {required} units.", None, pool=[tr.pool_info[c["supplier_id"]] for c in pool],
           constraints=_constraints(budget), required_quantity=required,
           search_strategy="best-first branch and bound: smaller sets first, then the higher optimistic score bound")

    def beaten(size: int, bound: float) -> dict | None:
        """Pruning test (unchanged logic). Returns the explanation, or None if not pruned."""
        if best is None or best[0][0] != 0:
            return None
        if size > best[0][1]:
            return {"rule": "larger_than_incumbent", "reason": f"a risk-free plan with {best[0][1]} supplier(s) is already known; "
                    f"any set of {size} suppliers would use more suppliers", "incumbent_key": list(best[0]), "node_size": size, "bound": round(bound, 4)}
        if size == best[0][1] and bound <= -best[0][2] + 1e-12:
            return {"rule": "bound_cannot_beat_incumbent", "reason": f"best score reachable from this branch ({bound:.4f}) cannot beat the "
                    f"incumbent's weighted score ({-best[0][2]:.4f}) at the same number of suppliers",
                    "incumbent_key": list(best[0]), "node_size": size, "bound": round(bound, 4)}
        return None

    root_bound = max(scores.values(), default=0.0)
    frontier = [(0, -root_bound, next(tie), ())]
    tr.c["nodes_created"] += 1
    tr.add("node_created", "Root node (no suppliers chosen yet) created and queued.", [], parent=None,
           priority={"set_size": 0, "optimistic_bound": round(root_bound, 4)})
    while frontier and n:
        size, neg_bound, _, members = heapq.heappop(frontier)
        group_here = [pool[i] for i in members]
        pr = beaten(size + 1, -neg_bound)
        if pr:
            pruned += 1
            tr.c["branches_pruned"] += 1
            tr.add("branch_pruned", f"Queued node {tr.combo(group_here)['key']} popped and pruned without expanding: {pr['reason']}.", group_here,
                   pruning={"pruned": True, "stage": "when popped from the queue", **pr}, outcome="Pruned")
            continue
        expanded += 1
        tr.c["nodes_expanded"] += 1
        tr.add("node_expanded", f"Popped {tr.combo(group_here)['key']} from the queue (set size {size}, optimistic bound {-neg_bound:.4f}) "
               f"and expanded: adding each remaining supplier in heuristic order.", group_here,
               priority={"set_size": size, "optimistic_bound": round(-neg_bound, 4)}, queue_size_after_pop=len(frontier), outcome="Expanded")
        start = members[-1] + 1 if members else 0
        for j in range(start, n):
            new = members + (j,)
            group = [pool[i] for i in new]
            tr.c["nodes_created"] += 1
            tr.add("node_created", f"Generated {tr.combo(group)['key']} by adding {pool[j]['supplier_id']} to {tr.combo(group_here)['key']}.", group,
                   parent=tr.combo(group_here)["key"], duplicate_check="each set is generated once: supplier indices are strictly increasing")
            pr = beaten(len(new), max(scores[c["supplier_id"]] for c in group))
            if pr:
                pruned += 1
                tr.c["branches_pruned"] += 1
                tr.add("branch_pruned", f"{tr.combo(group)['key']} pruned before evaluation: {pr['reason']}.", group,
                       pruning={"pruned": True, "stage": "when generated", **pr}, outcome="Pruned")
                continue
            cap = tr.capacity(group)
            if sum(float(c["available_qty"]) for c in group) >= required:
                evaluated += 1
                capacity_ok = True
                tr.c["combinations_evaluated"] += 1
                tr.add("combination_evaluated", f"Evaluating {tr.combo(group)['key']}: combined capacity is sufficient, so the allocation solver will be run.",
                       group, capacity=cap, quality=tr.not_evaluated("blended defect rate"), budget=tr.not_evaluated("total cost"),
                       min_allocation={"status": NOT_EVALUATED, "min_line_qty": MIN_SPLIT_LINE_QTY, "max_suppliers": SEARCH_MAX_SUPPLIERS},
                       outcome="Evaluated")
                # first solve: capacity + minimum line + quality (no budget) — shows whether quality alone is achievable
                r0 = allocate_detail(group, required, None, scores)
                tr.c["solver_invocations"] += 1
                tr.add("solver_invoked", "Integer solver run with capacity, minimum line, total quantity and the 5% quality limit (budget not included).", group,
                       solver={"kind": "capacity + minimum line + quality (no budget)", "status": "feasible" if r0["shares"] else "infeasible", "detail": r0["detail"],
                               "inputs": {"required": required, "capacities": cap["per_supplier"], "min_line_qty": MIN_SPLIT_LINE_QTY,
                                          "max_defect_rate": RULE_HIGH_RISK_DEFECT_RATE, "budget": None}})
                if r0["shares"] is not None:
                    quality_ok = True
                if budget is None:
                    r1 = r0
                else:
                    r1 = allocate_detail(group, required, budget, scores)
                    tr.c["solver_invocations"] += 1
                    tr.add("solver_invoked", "Integer solver run again with the budget limit added.", group,
                           solver={"kind": "all constraints (with budget)", "status": "feasible" if r1["shares"] else "infeasible", "detail": r1["detail"],
                                   "inputs": {"required": required, "capacities": cap["per_supplier"], "min_line_qty": MIN_SPLIT_LINE_QTY,
                                              "max_defect_rate": RULE_HIGH_RISK_DEFECT_RATE, "budget": round(float(budget), 2)}})
                shares = r1["shares"]
                if shares is not None:
                    m = _metrics(shares, required, scores, risky_ids)
                    key = (m["risky"], len(shares), -m["value"])
                    alloc = tr.allocation_block(shares, m)
                    tr.c["feasible_allocations"] += 1
                    obj = tr.objective(m, len(shares))
                    q_status = {"status": "PASS", "constraint": "blended defect rate", "weighted_defect_rate": alloc["weighted_defect_rate"],
                                "limit": RULE_HIGH_RISK_DEFECT_RATE, "per_supplier_defect": {c["supplier_id"]: round(float(c["defect_rate"]), 4) for c in group}}
                    b_status = ({"status": "PASS", "constraint": "total cost", "total_spend": m["total_cost"], "budget": round(float(budget), 2)}
                                if budget is not None else {"status": "PASS", "constraint": "total cost", "total_spend": m["total_cost"], "budget": None, "note": "no budget limit set"})
                    tr.feasible.append({"combination_key": tr.combo(group)["key"], "objective": obj, "allocation": alloc})
                    tr.add("feasible_allocation", f"{tr.combo(group)['key']} has a feasible allocation of {m['quantity']} units costing {m['total_cost']:,.2f}; "
                           f"blended defect {m['weighted_defect_rate']:.2%}.", group, capacity=cap, quality=q_status, budget=b_status,
                           min_allocation={"status": "PASS", "min_line_qty": MIN_SPLIT_LINE_QTY, "max_suppliers": SEARCH_MAX_SUPPLIERS},
                           allocation=alloc, objective=obj, outcome="Feasible candidate")
                    if best is None or key < best[0]:
                        prev = None if best is None else {"combination_key": tr.combo([c for c, _ in best[1]])["key"], "comparison_key": [best[0][0], best[0][1], round(best[0][2], 4)]}
                        best = (key, shares, m)
                        tr.add("new_best", f"{tr.combo(group)['key']} is the best plan so far under the lexicographic objective "
                               f"({m['risky']} High Risk, {len(shares)} supplier(s), weighted score {m['value']:.4f}).", group, allocation=alloc, objective=obj,
                               replaced=prev, outcome="Best solution so far")
                else:
                    tr.c["combinations_rejected"] += 1
                    cls = r0["reason"] if r0["shares"] is None else r1["reason"]
                    if r0["shares"] is None:
                        lb_def = _min_blended_defect(group, required)
                        lb_txt = f"; even filling the cleanest suppliers first the lowest reachable blended defect rate is {lb_def:.2%}" if lb_def is not None else ""
                        if cls in ("min_line", "min_line_total"):
                            reason = f"minimum-allocation constraint: {r0['detail']}"
                            q_s, mn = tr.not_evaluated("blended defect rate"), {"status": "FAIL", "min_line_qty": MIN_SPLIT_LINE_QTY, "max_suppliers": SEARCH_MAX_SUPPLIERS, "reason": r0["detail"]}
                            b_s = tr.not_evaluated("total cost")
                        else:
                            reason = (f"quality constraint: no integer allocation within capacities and the minimum line keeps the blended defect rate "
                                      f"at or below {RULE_HIGH_RISK_DEFECT_RATE:.0%}{lb_txt}")
                            q_s = {"status": "FAIL", "constraint": "blended defect rate", "limit": RULE_HIGH_RISK_DEFECT_RATE, "lowest_reachable_blended_defect": None if lb_def is None else round(lb_def, 4),
                                   "per_supplier_defect": {c["supplier_id"]: round(float(c["defect_rate"]), 4) for c in group}, "reason": reason}
                            mn = {"status": "PASS", "min_line_qty": MIN_SPLIT_LINE_QTY, "max_suppliers": SEARCH_MAX_SUPPLIERS}
                            b_s = tr.not_evaluated("total cost", {"note": "not reached: no allocation passes the quality limit, so there is no spend to compare"})
                    else:
                        mc = _min_cost(group, required)
                        only_budget = mc is not None and budget is not None and mc > budget + 1e-6
                        reason = (f"budget constraint: even the cheapest allocation within capacities and the minimum line costs {mc:,.2f}, above the budget of {budget:,.2f}"
                                  if only_budget else
                                  f"quality and budget together: an allocation within the 5% defect limit exists, and the budget alone is reachable "
                                  f"(cheapest spend {mc:,.2f} against {budget:,.2f}), but no allocation satisfies both")
                        q_s = {"status": "PASS", "constraint": "blended defect rate (achievable without the budget limit)", "limit": RULE_HIGH_RISK_DEFECT_RATE,
                               "weighted_defect_rate": round(_metrics(r0["shares"], required, scores, risky_ids)["weighted_defect_rate"], 4)}
                        b_s = {"status": "FAIL", "constraint": "total cost", "budget": round(float(budget), 2), "cheapest_possible_spend": None if mc is None else round(mc, 2), "reason": reason}
                        mn = {"status": "PASS", "min_line_qty": MIN_SPLIT_LINE_QTY, "max_suppliers": SEARCH_MAX_SUPPLIERS}
                    tr.add("combination_rejected", f"{tr.combo(group)['key']} rejected: {reason}.", group, capacity=cap, quality=q_s, budget=b_s,
                           min_allocation=mn, reason=reason, outcome="Rejected")
            else:
                tr.c["combinations_rejected"] += 1
                reason = cap["reason"]
                tr.add("combination_rejected", f"{tr.combo(group)['key']} rejected: {reason}. Quality, budget and allocation were not evaluated; "
                       f"a larger set may still add enough capacity.", group, capacity=cap, quality=tr.not_evaluated("blended defect rate"),
                       budget=tr.not_evaluated("total cost"), min_allocation={"status": NOT_EVALUATED, "min_line_qty": MIN_SPLIT_LINE_QTY, "max_suppliers": SEARCH_MAX_SUPPLIERS},
                       reason=reason, outcome="Rejected (capacity)")
            if len(new) < SEARCH_MAX_SUPPLIERS and j + 1 < n:
                reach = max(scores[pool[i]["supplier_id"]] for i in list(new) + list(range(j + 1, n)))
                heapq.heappush(frontier, (len(new), -reach, next(tie), new))
                tr.add("node_queued", f"{tr.combo(group)['key']} queued for later expansion with priority (set size {len(new)}, optimistic bound {reach:.4f}).", group,
                       priority={"set_size": len(new), "optimistic_bound": round(reach, 4)}, queue_size_after_push=len(frontier))
            elif len(new) >= SEARCH_MAX_SUPPLIERS:
                tr.add("node_queued", f"{tr.combo(group)['key']} not queued: it already has the maximum of {SEARCH_MAX_SUPPLIERS} suppliers.", group,
                       queued=False, reason="maximum supplier count reached")

    stats = {"nodes_expanded": expanded, "nodes_pruned": pruned, "plans_evaluated": evaluated,
             "constraints": _constraints(budget)}
    if best is None:
        reason = ("no combination of up to {} suppliers has enough capacity".format(SEARCH_MAX_SUPPLIERS) if not capacity_ok
                  else "no combination keeps the predicted defect rate within {:.0%}".format(RULE_HIGH_RISK_DEFECT_RATE) if not quality_ok
                  else "every combination that meets capacity and quality exceeds the budget")
        trace = tr.finish("no feasible allocation", f"The queue was exhausted and no combination met every constraint: {reason}.", None)
        return {"status": "infeasible", "reason": reason, "shares": [], "shortfall": required, "trace": trace, **stats}
    shares = sorted(best[1], key=lambda cq: -scores[cq[0]["supplier_id"]])
    trace = tr.finish("full allocation", "The queue was exhausted: every queued node was expanded or pruned, and the best feasible plan was kept.", best)
    return {"status": "full", "shares": shares, "shortfall": 0, "trace": trace, **best[2], **stats}


def partial(candidates: list[dict], required_quantity: float, risky_ids: set[str], budget: float | None = None) -> dict:
    """Even all suppliers together lack capacity: buy as much as possible (then best
    score) while meeting C1 (with min line), C3 quality, C4 budget, C5 size."""
    required = int(required_quantity)
    pool = [c for c in candidates if c.get("available_qty") is not None]
    n = len(pool)
    scores = {c["supplier_id"]: _score(c, n) for c in pool}
    tr = _Trace("partial_milp", pool, required, budget, risky_ids, scores)
    combined = int(sum(c["available_qty"] for c in pool))
    tr.add("search_started", f"Combined capacity of all {n} suppliers is {combined} units, below the required {required}, so a complete order is impossible. "
           f"Looking for the largest purchase that still meets quality, budget and size limits.", None, pool=[tr.pool_info[c["supplier_id"]] for c in pool],
           constraints=_constraints(budget), required_quantity=required, search_strategy="single integer program maximising quantity, then heuristic score")
    if not n:
        trace = tr.finish("no feasible allocation", "No supplier has a known capacity.", None)
        return {"status": "infeasible", "reason": "no supplier has known capacity", "shares": [], "shortfall": required,
                "constraints": _constraints(budget), "nodes_expanded": 0, "nodes_pruned": 0, "plans_evaluated": 0, "trace": trace}
    cap, d, cost = _arrays(pool)
    s = np.array([scores[c["supplier_id"]] for c in pool])
    # variables: q (n integers), y (n binaries: supplier used)
    eye = np.eye(n)
    rows = [np.concatenate([np.ones(n), np.zeros(n)]),            # sum q <= required
            np.concatenate([d - RULE_HIGH_RISK_DEFECT_RATE, np.zeros(n)]),  # quality
            np.concatenate([np.zeros(n), np.ones(n)])]            # suppliers <= max
    lb, ub = [-np.inf, -np.inf, -np.inf], [required, 1e-9, SEARCH_MAX_SUPPLIERS]
    if budget is not None:
        rows.append(np.concatenate([cost, np.zeros(n)])); lb.append(-np.inf); ub.append(budget + 1e-6)
    A = np.vstack(rows + [np.hstack([eye, -np.diag(cap)]),                # q_i <= cap_i * y_i
                          np.hstack([-eye, np.diag(np.full(n, float(MIN_SPLIT_LINE_QTY)))])])  # q_i >= min*y_i
    lb += [-np.inf] * (2 * n); ub += [0.0] * (2 * n)
    objective = -np.concatenate([1000.0 + s, np.zeros(n)])  # quantity first, then score
    res = milp(c=objective, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(2 * n),
               bounds=Bounds(np.zeros(2 * n), np.concatenate([cap, np.ones(n)])))
    tr.c["solver_invocations"] += 1
    stats = {"nodes_expanded": 1, "nodes_pruned": 0, "plans_evaluated": 1, "constraints": _constraints(budget)}
    q = np.round(res.x[:n]).astype(int) if res.success else np.zeros(n, dtype=int)
    shares = sorted([(c, int(x)) for c, x in zip(pool, q) if x > 0], key=lambda cq: -scores[cq[0]["supplier_id"]])
    tr.add("solver_invoked", "Integer solver run once over ALL suppliers: maximise the quantity bought, then the heuristic score, subject to "
           "capacity, the 50-unit minimum line, the 5% blended defect limit, the budget and at most 3 suppliers.", None,
           solver={"kind": "maximise quantity under all constraints", "status": "feasible" if shares else "infeasible",
                   "detail": "optimal integer purchase found" if shares else "no purchase meets the quality and budget constraints",
                   "inputs": {"required": required, "capacities": {c["supplier_id"]: int(c["available_qty"]) for c in pool}, "min_line_qty": MIN_SPLIT_LINE_QTY,
                              "max_defect_rate": RULE_HIGH_RISK_DEFECT_RATE, "budget": None if budget is None else round(float(budget), 2), "max_suppliers": SEARCH_MAX_SUPPLIERS}})
    if not shares:
        trace = tr.finish("no feasible allocation", "The solver found no purchase that meets the quality and budget constraints.", None)
        return {"status": "infeasible", "reason": "no purchase meets the quality and budget constraints",
                "shares": [], "shortfall": required, "trace": trace, **stats}
    m = _metrics(shares, required, scores, risky_ids)
    alloc = tr.allocation_block(shares, m)
    tr.c["feasible_allocations"] += 1
    obj = tr.objective(m, len(shares))
    tr.feasible.append({"combination_key": tr.combo([c for c, _ in shares])["key"], "objective": obj, "allocation": alloc})
    tr.add("feasible_allocation", f"Partial purchase of {m['quantity']} of the {required} units found (shortfall {required - m['quantity']}).", [c for c, _ in shares],
           allocation=alloc, objective=obj, outcome="Feasible candidate (partial)")
    trace = tr.finish("partial allocation", f"A single solve gave the largest feasible purchase; {required - m['quantity']} units cannot be covered.",
                      (None, shares, m))
    return {"status": "partial", "shares": shares, "shortfall": required - m["quantity"], **m, "trace": trace, **stats}
