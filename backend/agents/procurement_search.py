"""
Hybrid heuristic search for multi-supplier orders (used by Knowledge Agent Rule 6).

"Hybrid" = the LEARNED supplier predictions (defect rate, on-time-in-full, lead
time) feed the weighted HEURISTIC score of every supplier, and that score guides a
best-first branch-and-bound SEARCH over supplier combinations under hard constraints.

State      a set of suppliers (built by adding suppliers in heuristic-rank order)
Hard constraints for a complete plan
  C1 capacity   each supplier gets at most its demonstrated capacity
  C2 quantity   the shares add up to the required quantity
  C3 quality    quantity-weighted predicted defect rate <= RULE_HIGH_RISK_DEFECT_RATE
  C4 size       at most SEARCH_MAX_SUPPLIERS suppliers
Objective (lexicographic)
  1. fewest High Risk suppliers
  2. fewest suppliers (each extra supplier is another purchase order to manage)
  3. highest value = quantity-weighted heuristic score
Search     best-first over (set size, optimistic bound); bound = the best score
           reachable from a node, which no allocation can exceed (admissible). Once a
           risk-free incumbent exists, larger sets and same-size sets whose bound cannot
           beat it are pruned.
Allocation inside a set: two fills are tried — best score first, and lowest
predicted defect first (repairs a C3 violation) — and the better feasible one kept.

If no combination meets C3, the capacity-feasible plan with the lowest defect rate
is returned and flagged (`quality_relaxed`). If even all suppliers together lack
capacity, every supplier is used to capacity (`partial`).
"""
from __future__ import annotations

import heapq
from itertools import count

from backend.config import RULE_HIGH_RISK_DEFECT_RATE, SEARCH_MAX_SUPPLIERS


def _score(c: dict, n: int) -> float:
    if c.get("heuristic_score") is not None:
        return float(c["heuristic_score"]) / 100.0
    return 1.0 - (float(c.get("rank", n)) - 1.0) / max(n, 1)  # rank fallback, 1.0 for rank 1


def _fill(members: list[dict], required: int, order_key) -> list[tuple[dict, int]]:
    remaining, shares = required, []
    for c in sorted(members, key=order_key):
        q = int(min(int(c["available_qty"]), remaining))
        if q > 0:
            shares.append((c, q))
            remaining -= q
    return shares


def _metrics(shares, required, scores, risky) -> dict:
    qty = sum(q for _, q in shares)
    w_def = sum(q * float(c["defect_rate"]) for c, q in shares) / qty if qty else 0.0
    w_otif = sum(q * float(c.get("fulfillment_rate") or 0) for c, q in shares) / qty if qty else 0.0
    value = sum(q * scores[c["supplier_id"]] for c, q in shares) / required if required else 0.0
    return {
        "quantity": qty, "weighted_defect_rate": w_def, "weighted_otif": w_otif, "value": value,
        "risky": sum(c["supplier_id"] in risky for c, _ in shares),
        "meets_quality": w_def <= RULE_HIGH_RISK_DEFECT_RATE + 1e-12,
    }


def search(candidates: list[dict], required_quantity: float, risky_ids: set[str]) -> dict:
    required = int(required_quantity)
    pool = sorted(candidates, key=lambda c: (-_score(c, len(candidates)), c.get("rank", 0)))
    n = len(pool)
    scores = {c["supplier_id"]: _score(c, n) for c in pool}
    total_capacity = sum(int(c["available_qty"]) for c in pool)
    constraints = {"max_defect_rate": RULE_HIGH_RISK_DEFECT_RATE, "max_suppliers": SEARCH_MAX_SUPPLIERS}

    if total_capacity < required:  # nothing can cover it: order everything available
        shares = _fill(pool, required, lambda c: -scores[c["supplier_id"]])
        m = _metrics(shares, required, scores, risky_ids)
        return {"status": "partial", "shares": shares, "shortfall": required - m["quantity"], **m,
                "nodes_expanded": 0, "nodes_pruned": 0, "plans_evaluated": 0, "constraints": constraints}

    best = None          # (key, shares, metrics); key = (risky, suppliers, -value), smaller is better
    best_relaxed = None  # lowest-defect plan meeting C1, C2, C4 (fallback if C3 is impossible)
    expanded = pruned = evaluated = 0
    tie = count()

    def beaten(size: int, bound: float) -> bool:
        """True if no plan of `size` suppliers with this bound can beat a risk-free incumbent."""
        if best is None or best[0][0] != 0:
            return False
        return size > best[0][1] or (size == best[0][1] and bound <= -best[0][2] + 1e-12)

    frontier = [(0, -max(scores.values()), next(tie), ())]  # (set size, -bound, tiebreak, members)
    while frontier:
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
            if sum(int(c["available_qty"]) for c in group) >= required:
                evaluated += 1
                options = [_fill(group, required, lambda c: -scores[c["supplier_id"]]),
                           _fill(group, required, lambda c: (float(c["defect_rate"]), -scores[c["supplier_id"]]))]
                for shares in options:
                    if sum(q for _, q in shares) < required:
                        continue
                    m = _metrics(shares, required, scores, risky_ids)
                    if m["meets_quality"]:
                        key = (m["risky"], len(shares), -m["value"])
                        if best is None or key < best[0]:
                            best = (key, shares, m)
                    if best_relaxed is None or m["weighted_defect_rate"] < best_relaxed[1]["weighted_defect_rate"]:
                        best_relaxed = (shares, m)
            if len(new) < SEARCH_MAX_SUPPLIERS and j + 1 < n:
                reach = max(scores[pool[i]["supplier_id"]] for i in list(new) + list(range(j + 1, n)))
                heapq.heappush(frontier, (len(new), -reach, next(tie), new))

    stats = {"nodes_expanded": expanded, "nodes_pruned": pruned, "plans_evaluated": evaluated, "constraints": constraints}
    if best is not None:
        shares, m = best[1], best[2]
        shares = sorted(shares, key=lambda cq: -scores[cq[0]["supplier_id"]])
        return {"status": "full", "shares": shares, "shortfall": 0, **m, **stats}
    if best_relaxed is not None:
        shares, m = best_relaxed
        shares = sorted(shares, key=lambda cq: -scores[cq[0]["supplier_id"]])
        return {"status": "quality_relaxed", "shares": shares, "shortfall": 0, **m, **stats}
    # capacity suffices only with more than SEARCH_MAX_SUPPLIERS suppliers: use everyone needed
    shares = _fill(pool, required, lambda c: -scores[c["supplier_id"]])
    m = _metrics(shares, required, scores, risky_ids)
    return {"status": "quality_relaxed" if not m["meets_quality"] else "full", "shares": shares, "shortfall": 0, **m, **stats}
