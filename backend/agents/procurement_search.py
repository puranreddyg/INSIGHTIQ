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
"""
from __future__ import annotations

import heapq
from itertools import count

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from backend.config import MIN_SPLIT_LINE_QTY, RULE_HIGH_RISK_DEFECT_RATE, SEARCH_MAX_SUPPLIERS


def _score(c: dict, n: int) -> float:
    if c.get("heuristic_score") is not None:
        return float(c["heuristic_score"]) / 100.0
    return 1.0 - (float(c.get("rank", n)) - 1.0) / max(n, 1)  # rank fallback, 1.0 for rank 1


def _arrays(group: list[dict]):
    cap = np.array([float(c["available_qty"]) for c in group])
    d = np.array([float(c["defect_rate"]) for c in group])
    cost = np.array([float(c["unit_cost"]) for c in group])
    return cap, d, cost


def allocate(group: list[dict], required: int, budget: float | None, scores: dict) -> list[tuple[dict, int]] | None:
    """Exact integer allocation of `required` units over ALL suppliers in `group`
    (each gets >= MIN_SPLIT_LINE_QTY) meeting C1-C4, maximising the weighted score.
    None if infeasible."""
    cap, d, cost = _arrays(group)
    lo = np.minimum(float(MIN_SPLIT_LINE_QTY), float(required)) if len(group) == 1 else float(MIN_SPLIT_LINE_QTY)
    if (cap < lo).any() or cap.sum() < required or lo * len(group) > required:
        return None
    rows = [np.ones(len(group)), d - RULE_HIGH_RISK_DEFECT_RATE]
    lb, ub = [required, -np.inf], [required, 1e-9]
    if budget is not None:
        rows.append(cost); lb.append(-np.inf); ub.append(budget + 1e-6)
    s = np.array([scores[c["supplier_id"]] for c in group])
    res = milp(c=-s, constraints=LinearConstraint(np.vstack(rows), lb, ub),
               integrality=np.ones(len(group)), bounds=Bounds(np.full(len(group), lo), cap))
    if not res.success:
        return None
    q = np.round(res.x).astype(int)
    return [(c, int(x)) for c, x in zip(group, q)]


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

    def beaten(size: int, bound: float) -> bool:
        if best is None or best[0][0] != 0:
            return False
        return size > best[0][1] or (size == best[0][1] and bound <= -best[0][2] + 1e-12)

    frontier = [(0, -max(scores.values(), default=0.0), next(tie), ())]
    while frontier and n:
        size, neg_bound, _, members = heapq.heappop(frontier)
        if beaten(size + 1, -neg_bound):
            pruned += 1
            continue
        expanded += 1
        start = members[-1] + 1 if members else 0
        for j in range(start, n):
            new = members + (j,)
            group = [pool[i] for i in new]
            if beaten(len(new), max(scores[c["supplier_id"]] for c in group)):
                pruned += 1
                continue
            if sum(float(c["available_qty"]) for c in group) >= required:
                evaluated += 1
                capacity_ok = True
                if allocate(group, required, None, scores) is not None:
                    quality_ok = True
                shares = allocate(group, required, budget, scores)
                if shares is not None:
                    m = _metrics(shares, required, scores, risky_ids)
                    key = (m["risky"], len(shares), -m["value"])
                    if best is None or key < best[0]:
                        best = (key, shares, m)
            if len(new) < SEARCH_MAX_SUPPLIERS and j + 1 < n:
                reach = max(scores[pool[i]["supplier_id"]] for i in list(new) + list(range(j + 1, n)))
                heapq.heappush(frontier, (len(new), -reach, next(tie), new))

    stats = {"nodes_expanded": expanded, "nodes_pruned": pruned, "plans_evaluated": evaluated,
             "constraints": _constraints(budget)}
    if best is None:
        reason = ("no combination of up to {} suppliers has enough capacity".format(SEARCH_MAX_SUPPLIERS) if not capacity_ok
                  else "no combination keeps the predicted defect rate within {:.0%}".format(RULE_HIGH_RISK_DEFECT_RATE) if not quality_ok
                  else "every combination that meets capacity and quality exceeds the budget")
        return {"status": "infeasible", "reason": reason, "shares": [], "shortfall": required, **stats}
    shares = sorted(best[1], key=lambda cq: -scores[cq[0]["supplier_id"]])
    return {"status": "full", "shares": shares, "shortfall": 0, **best[2], **stats}


def partial(candidates: list[dict], required_quantity: float, risky_ids: set[str], budget: float | None = None) -> dict:
    """Even all suppliers together lack capacity: buy as much as possible (then best
    score) while meeting C1 (with min line), C3 quality, C4 budget, C5 size."""
    required = int(required_quantity)
    pool = [c for c in candidates if c.get("available_qty") is not None]
    n = len(pool)
    scores = {c["supplier_id"]: _score(c, n) for c in pool}
    if not n:
        return {"status": "infeasible", "reason": "no supplier has known capacity", "shares": [], "shortfall": required,
                "constraints": _constraints(budget), "nodes_expanded": 0, "nodes_pruned": 0, "plans_evaluated": 0}
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
    stats = {"nodes_expanded": 1, "nodes_pruned": 0, "plans_evaluated": 1, "constraints": _constraints(budget)}
    q = np.round(res.x[:n]).astype(int) if res.success else np.zeros(n, dtype=int)
    shares = sorted([(c, int(x)) for c, x in zip(pool, q) if x > 0], key=lambda cq: -scores[cq[0]["supplier_id"]])
    if not shares:
        return {"status": "infeasible", "reason": "no purchase meets the quality and budget constraints",
                "shares": [], "shortfall": required, **stats}
    m = _metrics(shares, required, scores, risky_ids)
    return {"status": "partial", "shares": shares, "shortfall": required - m["quantity"], **m, **stats}
