"""
AI Coordinator Agent — system controller & API gateway.

Orchestrates the sequential pipeline Perception -> Supplier -> Knowledge ->
Planning, tracks per-agent execution status/timing, and bundles everything
into a single structured payload for the frontend (pipeline view + XAI
inspector + action hub all read from this one response).
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from typing import Optional

from backend.agents import knowledge_agent, perception_agent, planning_agent, supplier_agent

PIPELINE_STAGES = ["Perception Agent", "Supplier Intelligence Agent", "Knowledge Agent", "Planning Agent"]


def run_pipeline(conn: sqlite3.Connection, sku_id: Optional[str] = None) -> dict:
    started_at = datetime.now().isoformat()
    stage_log: list[dict] = []

    def _log(name: str, status: str, started: float, extra: dict | None = None):
        stage_log.append({
            "agent": name,
            "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            **(extra or {}),
        })

    t0 = time.perf_counter()
    perception_result = perception_agent.run(conn, sku_id=sku_id)
    focus = perception_result["focus_facts"][0] if perception_result["focus_facts"] else None
    required = focus["RequiredQuantity"] if focus else 0
    focus_sku = focus["sku_id"] if focus else sku_id
    calc = (focus or {}).get("quantity_calculation") or {}
    _log("Perception Agent", "completed", t0, {
        "summary": f"Scanned supply-chain ledger ({perception_result['snapshot_date']}) — "
                   f"{perception_result['urgent_sku_count']} urgent SKU(s) detected; {focus_sku} needs {required} units "
                   f"({calc.get('method', 'n/a')} method"
                   + (f", {calc['forecast_method'].replace('_', ' ')} forecast" if calc.get("forecast_method") else "") + ").",
    })

    t1 = time.perf_counter()
    supplier_result = supplier_agent.run(conn, sku_id=focus_sku, required_quantity=required)
    _log("Supplier Intelligence Agent", "completed", t1, {
        "summary": f"{supplier_result['eligible_count']}/{supplier_result['pool_size']} candidate suppliers can supply "
                   f"{required} units alone; ranked via heuristic search."
                   + (f" Combined availability {supplier_result['combined_available_qty']} (SIM)." if supplier_result["no_eligible_supplier"] else ""),
    })

    t2 = time.perf_counter()
    knowledge_result = knowledge_agent.run(perception_result, supplier_result)
    rec = knowledge_result["recommended_supplier"]
    decided = (f"recommended {rec['supplier_id']}" if knowledge_result["allocation_mode"] == "single"
               else f"split order across {len(knowledge_result['allocation_candidates'])} risk-evaluated supplier(s)"
               if knowledge_result["allocation_mode"] == "split" else "no supplier needed")
    _log("Knowledge Agent", "completed", t2, {
        "summary": f"Forward-chained {len(knowledge_result['reasoning_trace'])} inference steps; {decided}.",
    })

    t3 = time.perf_counter()
    planning_result = planning_agent.run(conn, focus, knowledge_result, required, supplier_result)
    if "error" in planning_result:
        plan_summary = planning_result["error"]
    else:
        n = len(planning_result.get("allocations", []))
        plan_summary = (f"{'Partial plan' if planning_result.get('is_partial') else 'Plan ready'} — "
                        f"{planning_result.get('order_quantity', 0)} units from {n} supplier(s), "
                        f"all in by {planning_result.get('estimated_arrival_date', 'N/A')}."
                        + (f" Shortage {planning_result['residual_shortage']}." if planning_result.get("residual_shortage") else ""))
    _log("Planning Agent", "completed", t3, {"summary": plan_summary})

    return {
        "agent": "AI Coordinator Agent",
        "sku_id": focus_sku,
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
        "pipeline_stages": PIPELINE_STAGES,
        "stage_log": stage_log,
        "perception": perception_result,
        "supplier": supplier_result,
        "knowledge": knowledge_result,
        "planning": planning_result,
        "focus_fact": focus,
    }
