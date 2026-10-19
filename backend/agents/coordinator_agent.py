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
    _log("Perception Agent", "completed", t0, {
        "summary": f"Scanned supply-chain ledger ({perception_result['snapshot_date']}) — "
                   f"{perception_result['urgent_sku_count']} urgent SKU(s) detected.",
    })

    focus = perception_result["focus_facts"][0] if perception_result["focus_facts"] else None
    required = focus["RequiredQuantity"] if focus else 0
    focus_sku = focus["sku_id"] if focus else sku_id

    t1 = time.perf_counter()
    supplier_result = supplier_agent.run(conn, sku_id=focus_sku, required_quantity=required)
    _log("Supplier Intelligence Agent", "completed", t1, {
        "summary": f"{supplier_result['eligible_count']}/{supplier_result['pool_size']} candidate suppliers can supply "
                   f"{required} units; ranked via heuristic search.",
    })

    t2 = time.perf_counter()
    knowledge_result = knowledge_agent.run(perception_result, supplier_result)
    _log("Knowledge Agent", "completed", t2, {
        "summary": f"Forward-chained {len(knowledge_result['reasoning_trace'])} inference steps; "
                    f"recommended {knowledge_result['recommended_supplier']['supplier_id'] if knowledge_result['recommended_supplier'] else 'N/A'}.",
    })

    t3 = time.perf_counter()
    planning_result = planning_agent.run(conn, focus, knowledge_result, required, supplier_result)
    _log("Planning Agent", "completed", t3, {
        "summary": (
            f"Plan ready — {planning_result.get('order_quantity', 0)} units, "
            f"arriving {planning_result.get('estimated_arrival_date', 'N/A')}."
        ) if "error" not in planning_result else planning_result["error"],
    })

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
