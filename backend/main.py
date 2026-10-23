"""InsightIQ FastAPI application: REST gateway over the multi-agent engine."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import config
from backend.agents import coordinator_agent, knowledge_agent, perception_agent, planning_agent, supplier_agent
from backend.database import TABLES_TO_BUILD, build_database, connect
from backend.forecasting import service as forecast_service
from backend.forecasting.demand import daily_lane_demand

APP_STARTED_AT = datetime.now()

app = FastAPI(title="InsightIQ", description="AI-Based Intelligent Procurement Decision Support System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    build_database(force=False)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AllocationLine(BaseModel):
    supplier_id: str
    quantity: int
    unit_cost: Optional[float] = None
    estimated_arrival: Optional[str] = None


class ApproveOrderRequest(BaseModel):
    """Legacy single-supplier fields, or `allocations` for a (possibly split) plan."""
    sku_id: str
    supplier_id: Optional[str] = None
    supplier_name: Optional[str] = None
    quantity: Optional[int] = None
    unit_cost: Optional[float] = None
    estimated_arrival: Optional[str] = None
    backup_supplier_id: Optional[str] = None
    allocations: Optional[list[AllocationLine]] = None
    is_partial: bool = False


# ---------------------------------------------------------------------------
# System / dashboard endpoints
# ---------------------------------------------------------------------------

PRODUCT_ID_RETIRED = (
    "Procurement is SKU-based: product.csv has no link to any supplier or SKU, so product_id "
    "can no longer trigger procurement. Use sku_id (SKU_1..SKU_50)."
)


def _require_sku(conn, sku_id: str) -> None:
    if conn.execute("SELECT 1 FROM sku_supplier_lanes WHERE sku_id = ? LIMIT 1", (sku_id,)).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"SKU {sku_id} not found")


@app.get("/api/system-health")
def system_health():
    with connect() as conn:
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES_TO_BUILD + ["purchase_orders"]}
        urgent_count = len(perception_agent.urgent_requests(perception_agent.observe_all(conn)))
    uptime_seconds = (datetime.now() - APP_STARTED_AT).total_seconds()
    return {
        "status": "online",
        "uptime_seconds": round(uptime_seconds, 1),
        "datasets_loaded": counts,
        "active_alerts": urgent_count,
        "agents": [
            {"name": "Perception Agent", "status": "idle"},
            {"name": "Supplier Intelligence Agent", "status": "idle"},
            {"name": "Knowledge Agent", "status": "idle"},
            {"name": "Planning Agent", "status": "idle"},
        ],
    }


def _sku_summary(r: dict, candidate_counts: dict[str, int]) -> dict:
    return {
        "sku_id": r["sku_id"],
        "urgent": r["Urgent"],
        "urgent_warehouses": [lane["warehouse_id"] for lane in r["lanes"] if lane["Urgent"]],
        "lane_count": r["lane_count"],
        "inventory": r["Inventory"],
        "reorder_point": r["ReorderPoint"],
        "shortfall": r["Shortfall"],
        "required_quantity": r["RequiredQuantity"],
        "candidate_suppliers": candidate_counts.get(r["sku_id"], 0),
    }


def _candidate_counts(conn) -> dict[str, int]:
    rows = conn.execute("SELECT sku_id, COUNT(DISTINCT supplier_id) FROM sku_supplier_lanes GROUP BY sku_id").fetchall()
    return {sku: n for sku, n in rows}


@app.get("/api/dashboard")
def dashboard():
    with connect() as conn:
        requests = perception_agent.observe_all(conn)
        counts = _candidate_counts(conn)
        active_pos = pd.read_sql_query("SELECT * FROM purchase_orders ORDER BY created_at DESC", conn)
        avg_lead = conn.execute("SELECT AVG(lead_time_days) FROM sku_supplier_lanes").fetchone()[0]
        avg_fulfillment = conn.execute(
            "SELECT AVG(fulfillment_rate) FROM (SELECT DISTINCT supplier_id, fulfillment_rate FROM supplier_sku_profile)"
        ).fetchone()[0]
        as_of = perception_agent.snapshot_date(conn)[:10]

    urgent = perception_agent.urgent_requests(requests)
    kpis = {
        "critical_low_stock_skus": len(urgent),
        "active_purchase_orders": int((active_pos["status"] == "Approved").sum()) if not active_pos.empty else 0,
        "avg_supplier_lead_time_days": round(float(avg_lead), 1) if avg_lead is not None else None,
        "avg_supplier_fulfillment_rate": round(float(avg_fulfillment) * 100, 1) if avg_fulfillment is not None else None,
    }
    return {
        "snapshot_date": as_of,
        "kpis": kpis,
        "kpi_provenance": {"avg_supplier_lead_time_days": "REAL", "avg_supplier_fulfillment_rate": "SIMULATED"},
        "alerts": [_sku_summary(r, counts) for r in urgent],
        "recent_purchase_orders": active_pos.head(10).to_dict(orient="records") if not active_pos.empty else [],
        "generated_at": datetime.now().isoformat(),
    }


@app.get("/api/skus")
def list_skus(q: Optional[str] = None, urgent_only: bool = False):
    with connect() as conn:
        requests = perception_agent.observe_all(conn)
        counts = _candidate_counts(conn)
        as_of = perception_agent.snapshot_date(conn)[:10]
    rows = [_sku_summary(r, counts) for r in requests.values()]
    if urgent_only:
        rows = [r for r in rows if r["urgent"]]
    if q:
        rows = [r for r in rows if q.lower() in r["sku_id"].lower()]
    return {"snapshot_date": as_of, "skus": rows}


@app.get("/api/products")
def list_products():
    raise HTTPException(status_code=410, detail=PRODUCT_ID_RETIRED)


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------

@app.post("/api/run-agent-pipeline")
def run_agent_pipeline(sku_id: Optional[str] = None, product_id: Optional[int] = None):
    if sku_id is None:
        if product_id is not None:
            raise HTTPException(status_code=410, detail=PRODUCT_ID_RETIRED)
        raise HTTPException(status_code=422, detail="sku_id is required")
    with connect() as conn:
        _require_sku(conn, sku_id)
        return coordinator_agent.run_pipeline(conn, sku_id=sku_id)


# ---------------------------------------------------------------------------
# Supplier decision matrix
# ---------------------------------------------------------------------------

@app.get("/api/suppliers")
def suppliers(sku_id: str, required_quantity: Optional[float] = Query(None, ge=0)):
    """Candidate matrix for one SKU. Without required_quantity, the SKU's
    current requirement from the Perception Agent is used. When no acceptable
    single supplier exists, the split-order allocation is attached."""
    with connect() as conn:
        _require_sku(conn, sku_id)
        if required_quantity is None:
            required_quantity = perception_agent.observe_sku(conn, sku_id)["RequiredQuantity"]
        result = supplier_agent.rank_suppliers(conn, sku_id, required_quantity=required_quantity, top_n=None)
        knowledge = knowledge_agent.run({"focus_facts": []}, result)
        statuses = {c["supplier_id"]: c["supplier_status"] for c in knowledge["annotated_candidates"] + knowledge["ineligible_candidates"]}
        for c in result["ranked_candidates"] + result["ineligible_suppliers"]:
            c["supplier_status"] = statuses.get(c["supplier_id"])
        result["allocation_mode"] = knowledge["allocation_mode"]
        result["allocation"] = planning_agent.allocation_for(knowledge, required_quantity)
        return result


# ---------------------------------------------------------------------------
# Demand forecast
# ---------------------------------------------------------------------------

def _forecast_summary(method: dict) -> dict:
    artifact = method["artifact"] or {}
    return {
        "method": method["method"],
        "requested_method": method["requested"],
        "reason": method["reason"],
        "artifact_error": method["artifact_error"],
        "model": {k: artifact.get(k) for k in ("kind", "features", "alpha", "trained_through")} if artifact else None,
        "validation": artifact.get("validation"),
        "selection": artifact.get("selection"),
        "provenance": "REAL — Units_Sold from supply_chain_dataset1.csv; forecasts are model output, not observations",
    }


@app.get("/api/forecast/metrics")
def forecast_metrics():
    return _forecast_summary(forecast_service.resolve_method())


@app.get("/api/forecast")
def forecast(sku_id: str, horizon_days: int = Query(21, ge=1, le=90)):
    """Daily demand forecast for every warehouse lane of one SKU."""
    with connect() as conn:
        _require_sku(conn, sku_id)
        as_of = perception_agent.snapshot_date(conn)
        history = daily_lane_demand(conn, since=perception_agent._history_window_start(as_of))
    method = forecast_service.resolve_method()
    lanes = [c for c in history.columns if c[0] == sku_id]
    fc = forecast_service.forecast_daily(history[lanes], horizon_days, method["method"], method["model"])
    recent = history[lanes].tail(config.DEMAND_LOOKBACK_DAYS)
    return {
        "sku_id": sku_id,
        "as_of": as_of[:10],
        "horizon_days": horizon_days,
        "summary": _forecast_summary(method),
        "lanes": [
            {
                "warehouse_id": wh,
                "recent_mean_daily": round(float(recent[(sku, wh)].mean()), 2),
                "forecast_total": round(float(fc[(sku, wh)].sum()), 2),
                "forecast_mean_daily": round(float(fc[(sku, wh)].mean()), 2),
                "daily": [{"date": d.date().isoformat(), "units": round(float(v), 2)} for d, v in fc[(sku, wh)].items()],
            }
            for sku, wh in lanes
        ],
    }


# ---------------------------------------------------------------------------
# Procurement action & approval
# ---------------------------------------------------------------------------

def _validate_allocations(conn, payload: ApproveOrderRequest) -> tuple[list[dict], int]:
    """Re-derives everything server-side — requirement, availability, unit cost and
    risk status — so a stale or edited client plan cannot over-allocate a supplier,
    exceed the requirement or bypass Rule 3."""
    lines = payload.allocations or []
    if not lines:
        raise HTTPException(status_code=422, detail="allocations must not be empty")
    ids = [line.supplier_id for line in lines]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="each supplier may appear only once in an allocation")

    required = perception_agent.observe_sku(conn, payload.sku_id)["RequiredQuantity"]
    if required <= 0:
        raise HTTPException(status_code=422, detail=f"{payload.sku_id} has no current procurement requirement")

    matrix = {r["supplier_id"]: r for r in supplier_agent.candidate_matrix(conn, payload.sku_id).to_dict(orient="records")}
    validated = []
    for line in lines:
        if line.quantity <= 0:
            raise HTTPException(status_code=422, detail=f"{line.supplier_id}: quantity must be positive")
        candidate = matrix.get(line.supplier_id)
        if candidate is None or candidate.get("available_qty") is None or pd.isna(candidate["available_qty"]):
            raise HTTPException(status_code=422, detail=f"{line.supplier_id} does not supply {payload.sku_id} (no lane or profile)")
        available = int(candidate["available_qty"])
        if line.quantity > available:
            raise HTTPException(status_code=422, detail=(
                f"{line.supplier_id} has {available} units of {payload.sku_id} available; {line.quantity} requested"))
        status = knowledge_agent.evaluate_supplier(candidate, required, [], allow_partial=True)["supplier_status"]
        if status == "High Risk":
            raise HTTPException(status_code=422, detail=(
                f"{line.supplier_id} is High Risk (defect rate {candidate['defect_rate']} > {config.RULE_HIGH_RISK_DEFECT_RATE}); "
                "Rule 3 forbids allocating to it automatically"))
        unit_cost = float(candidate["unit_cost"])
        if line.unit_cost is not None and abs(line.unit_cost - unit_cost) > 0.01:
            raise HTTPException(status_code=422, detail=(
                f"{line.supplier_id}: unit cost {line.unit_cost} does not match the current {unit_cost:.4f} — re-run the pipeline"))
        validated.append({"supplier_id": line.supplier_id, "quantity": int(line.quantity), "unit_cost": unit_cost,
                          "estimated_arrival": line.estimated_arrival, "risk_status": status})

    total = sum(v["quantity"] for v in validated)
    if total > required:
        raise HTTPException(status_code=422, detail=f"allocations total {total} exceeds the requirement of {required}")
    if total < required and not payload.is_partial:
        raise HTTPException(status_code=422, detail=(
            f"allocations total {total} covers only part of the requirement ({required}); approve it explicitly as partial"))
    return validated, required


def _insert_order_group(conn, payload: ApproveOrderRequest, lines: list[dict], required: Optional[int]) -> list[dict]:
    """All rows of one approval in a single transaction: either every line is
    written or none is."""
    group = f"PO-{uuid.uuid4().hex[:10].upper()}"
    today = datetime.now().date().isoformat()
    ids = []
    try:
        conn.execute("BEGIN")
        for i, line in enumerate(lines, start=1):
            cur = conn.execute(
                """INSERT INTO purchase_orders
                   (sku_id, product_name, supplier_id, supplier_name, quantity, unit_cost, total_cost, order_date,
                    estimated_arrival, backup_supplier_id, status, order_group, allocation_line, allocation_count,
                    required_quantity, is_partial, risk_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload.sku_id, payload.sku_id, line["supplier_id"], line.get("supplier_name") or line["supplier_id"],
                    line["quantity"], line["unit_cost"], round(line["quantity"] * line["unit_cost"], 2), today,
                    line.get("estimated_arrival"), payload.backup_supplier_id if len(lines) == 1 else None,
                    "Approved", group, i, len(lines), required, int(bool(payload.is_partial)), line.get("risk_status"),
                ),
            )
            ids.append(cur.lastrowid)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    marks = ",".join("?" * len(ids))
    return [dict(r) for r in conn.execute(f"SELECT * FROM purchase_orders WHERE id IN ({marks}) ORDER BY allocation_line", ids)]


@app.post("/api/approve-order")
def approve_order(payload: ApproveOrderRequest):
    with connect() as conn:
        conn.isolation_level = None  # explicit BEGIN/COMMIT in _insert_order_group
        _require_sku(conn, payload.sku_id)

        if payload.allocations is not None:
            lines, required = _validate_allocations(conn, payload)
        else:
            if payload.supplier_id is None or payload.quantity is None or payload.unit_cost is None:
                raise HTTPException(status_code=422, detail="supplier_id, quantity and unit_cost are required (or send allocations)")
            if payload.quantity <= 0:
                raise HTTPException(status_code=422, detail="quantity must be positive")
            profile = conn.execute(
                "SELECT available_qty FROM supplier_sku_profile WHERE sku_id = ? AND supplier_id = ?",
                (payload.sku_id, payload.supplier_id),
            ).fetchone()
            if profile is None:
                raise HTTPException(status_code=422, detail=f"{payload.supplier_id} does not supply {payload.sku_id}")
            if payload.quantity > profile["available_qty"]:
                raise HTTPException(
                    status_code=422,
                    detail=f"{payload.supplier_id} has {profile['available_qty']} units of {payload.sku_id} available; "
                           f"{payload.quantity} requested",
                )
            lines = [{"supplier_id": payload.supplier_id, "supplier_name": payload.supplier_name,
                      "quantity": payload.quantity, "unit_cost": payload.unit_cost,
                      "estimated_arrival": payload.estimated_arrival, "risk_status": None}]
            required = None

        try:
            rows = _insert_order_group(conn, payload, lines, required)
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail=f"purchase order not saved (rolled back): {exc}") from exc

    return {
        "status": "approved",
        "order_group": rows[0]["order_group"],
        "is_partial": bool(payload.is_partial),
        "total_quantity": sum(r["quantity"] for r in rows),
        "total_cost": round(sum(r["total_cost"] for r in rows), 2),
        "purchase_order": rows[0],
        "purchase_orders": rows,
    }


@app.get("/api/orders")
def purchase_orders(limit: int = 50):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM purchase_orders ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"purchase_orders": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(config.FRONTEND_DIR / "index.html"))
