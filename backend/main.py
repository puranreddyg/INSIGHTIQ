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
from backend.agents import coordinator_agent, supplier_agent
from backend.database import build_database, connect

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
    product_id: int
    supplier_id: str
    supplier_name: Optional[str] = None
    quantity: int
    unit_cost: float
    estimated_arrival: Optional[str] = None
    backup_supplier_id: Optional[str] = None


# ---------------------------------------------------------------------------
# System / dashboard endpoints
# ---------------------------------------------------------------------------

@app.get("/api/system-health")
def system_health():
    with connect() as conn:
        tables = ["products", "orders", "customers", "calendar", "procurement_kpi", "supply_chain", "supplier_profiles", "purchase_orders"]
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
        urgent_count = conn.execute("SELECT COUNT(*) FROM products WHERE urgent = 1").fetchone()[0]
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


@app.get("/api/dashboard")
def dashboard():
    with connect() as conn:
        products = pd.read_sql_query("SELECT * FROM products", conn)
        orders = pd.read_sql_query("SELECT * FROM orders", conn)
        supplier_profiles = pd.read_sql_query("SELECT * FROM supplier_profiles", conn)
        active_pos = pd.read_sql_query("SELECT * FROM purchase_orders ORDER BY created_at DESC", conn)

        urgent = products[products["urgent"] == 1].sort_values("deficit", ascending=False)

        kpis = {
            "critical_low_stock_skus": int(len(urgent)),
            "active_purchase_orders": int((active_pos["status"] == "Approved").sum()) if not active_pos.empty else 0,
            "avg_procurement_lead_time_days": round(float(orders["lead_time_days"].mean()), 1) if not orders.empty else 0,
            "avg_supplier_reliability": round(float(supplier_profiles["reliability_score"].mean()) * 100, 1) if not supplier_profiles.empty else 0,
        }

        alert_table = [
            {
                "product_id": int(r.product_id),
                "sku": f"SKU-{int(r.product_id):04d}",
                "product_name": r.product_name,
                "category": r.category,
                "inventory": int(r.units_in_stock),
                "reorder_level": int(r.reorder_level),
                "deficit": int(r.deficit),
                "unit_price": float(r.unit_price),
            }
            for r in urgent.head(50).itertuples()
        ]

        recent_orders = active_pos.head(10).to_dict(orient="records") if not active_pos.empty else []

    return {
        "kpis": kpis,
        "alerts": alert_table,
        "recent_purchase_orders": recent_orders,
        "generated_at": datetime.now().isoformat(),
    }


@app.get("/api/products")
def list_products(
    q: Optional[str] = None,
    urgent_only: bool = False,
    limit: int = Query(50, le=500),
):
    with connect() as conn:
        sql = "SELECT product_id, product_name, category, units_in_stock, reorder_level, deficit, urgent, unit_price FROM products WHERE 1=1"
        params: list = []
        if urgent_only:
            sql += " AND urgent = 1"
        if q:
            sql += " AND (product_name LIKE ? OR category LIKE ?)"
            params += [f"%{q}%", f"%{q}%"]
        sql += " ORDER BY deficit DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    return {"products": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------

@app.post("/api/run-agent-pipeline")
def run_agent_pipeline(product_id: int = Query(...)):
    with connect() as conn:
        product = conn.execute("SELECT product_id FROM products WHERE product_id = ?", (product_id,)).fetchone()
        if product is None:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        result = coordinator_agent.run_pipeline(conn, product_id=product_id)
    return result


# ---------------------------------------------------------------------------
# Supplier decision matrix
# ---------------------------------------------------------------------------

@app.get("/api/suppliers")
def suppliers(required_deficit: float = 0):
    with connect() as conn:
        if required_deficit and required_deficit > 0:
            result = supplier_agent.rank_suppliers(conn, required_deficit=required_deficit)
        else:
            result = supplier_agent.full_matrix(conn)
    return result


# ---------------------------------------------------------------------------
# Procurement action & approval
# ---------------------------------------------------------------------------

@app.post("/api/approve-order")
def approve_order(payload: ApproveOrderRequest):
    total_cost = round(payload.quantity * payload.unit_cost, 2)
    with connect() as conn:
        product = conn.execute(
            "SELECT product_name FROM products WHERE product_id = ?", (payload.product_id,)
        ).fetchone()
        if product is None:
            raise HTTPException(status_code=404, detail=f"Product {payload.product_id} not found")

        cur = conn.execute(
            """INSERT INTO purchase_orders
               (product_id, product_name, supplier_id, supplier_name, quantity, unit_cost,
                total_cost, order_date, estimated_arrival, backup_supplier_id, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                payload.product_id, product["product_name"], payload.supplier_id,
                payload.supplier_name or payload.supplier_id, payload.quantity, payload.unit_cost,
                total_cost, datetime.now().date().isoformat(), payload.estimated_arrival,
                payload.backup_supplier_id, "Approved",
            ),
        )
        conn.commit()
        po_id = cur.lastrowid
        row = conn.execute("SELECT * FROM purchase_orders WHERE id = ?", (po_id,)).fetchone()
    return {"status": "approved", "purchase_order": dict(row)}


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
