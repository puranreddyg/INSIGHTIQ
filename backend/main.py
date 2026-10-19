"""InsightIQ FastAPI application: REST gateway over the multi-agent engine."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import config
from backend.agents import coordinator_agent, perception_agent, supplier_agent
from backend.database import TABLES_TO_BUILD, build_database, connect

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

class ApproveOrderRequest(BaseModel):
    sku_id: str
    supplier_id: str
    supplier_name: Optional[str] = None
    quantity: int
    unit_cost: float
    estimated_arrival: Optional[str] = None
    backup_supplier_id: Optional[str] = None


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
    current requirement from the Perception Agent is used."""
    with connect() as conn:
        _require_sku(conn, sku_id)
        if required_quantity is None:
            required_quantity = perception_agent.observe_sku(conn, sku_id)["RequiredQuantity"]
        return supplier_agent.rank_suppliers(conn, sku_id, required_quantity=required_quantity, top_n=None)


# ---------------------------------------------------------------------------
# Procurement action & approval
# ---------------------------------------------------------------------------

@app.post("/api/approve-order")
def approve_order(payload: ApproveOrderRequest):
    if payload.quantity <= 0:
        raise HTTPException(status_code=422, detail="quantity must be positive")
    total_cost = round(payload.quantity * payload.unit_cost, 2)
    with connect() as conn:
        _require_sku(conn, payload.sku_id)
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

        cur = conn.execute(
            """INSERT INTO purchase_orders
               (sku_id, product_name, supplier_id, supplier_name, quantity, unit_cost,
                total_cost, order_date, estimated_arrival, backup_supplier_id, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                payload.sku_id, payload.sku_id, payload.supplier_id,
                payload.supplier_name or payload.supplier_id, payload.quantity, payload.unit_cost,
                total_cost, datetime.now().date().isoformat(), payload.estimated_arrival,
                payload.backup_supplier_id, "Approved",
            ),
        )
        conn.commit()
        po_id = cur.lastrowid
        row = conn.execute("SELECT * FROM purchase_orders WHERE id = ?", (po_id,)).fetchone()
    return {"status": "approved", "purchase_order": dict(row)}


class SplitLine(BaseModel):
    supplier_id: str
    quantity: int
    unit_cost: float
    estimated_arrival: Optional[str] = None


class ApproveSplitOrderRequest(BaseModel):
    sku_id: str
    allocations: list[SplitLine]
    backup_supplier_id: Optional[str] = None


@app.post("/api/approve-split-order")
def approve_split_order(payload: ApproveSplitOrderRequest):
    """Approve a split order: one purchase order per supplier, created all-or-nothing.
    Every line is validated first; if any line is invalid, no purchase order is created."""
    if not payload.allocations:
        raise HTTPException(status_code=422, detail="allocations must not be empty")
    supplier_ids = [line.supplier_id for line in payload.allocations]
    if len(set(supplier_ids)) != len(supplier_ids):
        raise HTTPException(status_code=422, detail="each supplier may appear only once in a split order")
    with connect() as conn:
        _require_sku(conn, payload.sku_id)
        for line in payload.allocations:
            if line.quantity <= 0:
                raise HTTPException(status_code=422, detail=f"{line.supplier_id}: quantity must be positive")
            profile = conn.execute(
                "SELECT available_qty FROM supplier_sku_profile WHERE sku_id = ? AND supplier_id = ?",
                (payload.sku_id, line.supplier_id),
            ).fetchone()
            if profile is None:
                raise HTTPException(status_code=422, detail=f"{line.supplier_id} does not supply {payload.sku_id}")
            if line.quantity > profile["available_qty"]:
                raise HTTPException(
                    status_code=422,
                    detail=f"{line.supplier_id} has {profile['available_qty']} units of {payload.sku_id} available; "
                           f"{line.quantity} requested",
                )

        po_ids = []
        for line in payload.allocations:
            cur = conn.execute(
                """INSERT INTO purchase_orders
                   (sku_id, product_name, supplier_id, supplier_name, quantity, unit_cost,
                    total_cost, order_date, estimated_arrival, backup_supplier_id, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload.sku_id, payload.sku_id, line.supplier_id, line.supplier_id, line.quantity,
                    line.unit_cost, round(line.quantity * line.unit_cost, 2), datetime.now().date().isoformat(),
                    line.estimated_arrival, payload.backup_supplier_id, "Approved",
                ),
            )
            po_ids.append(cur.lastrowid)
        conn.commit()
        rows = [dict(conn.execute("SELECT * FROM purchase_orders WHERE id = ?", (i,)).fetchone()) for i in po_ids]
    return {"status": "approved", "purchase_orders": rows}


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

class RevalidatingStaticFiles(StaticFiles):
    """Static files that browsers must revalidate on every load (cheap 304 via ETag).

    Without a Cache-Control header, browsers cache app.js/styles.css heuristically
    for days, so after a deploy users keep running the OLD frontend against the NEW
    API (blank KPI cards, empty charts) until they hard-refresh.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", RevalidatingStaticFiles(directory=str(config.FRONTEND_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(config.FRONTEND_DIR / "index.html"), headers={"Cache-Control": "no-cache"})
