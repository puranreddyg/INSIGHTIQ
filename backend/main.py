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

import threading

from backend import config
from backend.agents import coordinator_agent, knowledge_agent, perception_agent, planning_agent, supplier_agent
from backend.database import TABLES_TO_BUILD, build_database, connect
from backend.ml import forecast_evaluation
from backend.ml.store import load_demand_model, load_supplier_meta

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
    threading.Thread(target=_warm_forecast_evaluation, daemon=True).start()


def _warm_forecast_evaluation() -> None:
    """The forecast comparison refits the model ~13 times; compute it once in the background."""
    try:
        with connect() as conn:
            sc = _supply_chain_frame(conn)
        for protocol in ("rolling", "fixed"):
            forecast_evaluation.evaluate(sc, protocol)
    except Exception:  # a failed warm-up only means the first request computes it
        pass


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
    is_partial: bool = False          # the order covers only part of the requirement (must be explicit)
    acknowledge_risk: bool = False    # the buyer accepts a High Risk supplier (Rule 3) for this order


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
        "effective_reorder_point": r["EffectiveReorderPoint"],
        "stockout_probability": r["StockoutProbability"],
        "forecast_daily_demand": r["DailyDemand"],
        "urgent_reasons": sorted({lane["UrgentReason"] for lane in r["lanes"] if lane["UrgentReason"]}),
        "shortfall": r["Shortfall"],
        "required_quantity": r["RequiredQuantity"],
        "required_quantity_baseline": r["required_quantity_baseline"],
        "candidate_suppliers": candidate_counts.get(r["sku_id"], 0),
    }


def _candidate_counts(conn) -> dict[str, int]:
    rows = conn.execute("SELECT sku_id, COUNT(DISTINCT supplier_id) FROM sku_supplier_lanes GROUP BY sku_id").fetchall()
    return {sku: n for sku, n in rows}


def _forecast_accuracy(model) -> dict | None:
    """Headline backtest: train on the past only, forecast the next 14 days (WAPE, lower is better)."""
    bt = model.metrics or {}
    test = bt.get("strict_h14")
    if not bt.get("available") or not test:
        return None
    model_err, base_err = test[f"model_k{bt['selected_k']}"], test["baseline_trailing_28d"]
    if not base_err:
        return None
    return {
        "horizon_days": 14,
        "model_error_pct": round(model_err * 100, 1),
        "previous_method_error_pct": round(base_err * 100, 1),
        "error_reduction_pct": round((1 - model_err / base_err) * 100, 0),
        "trained_rows": model.n_rows,
        "method": "train on the past only, forecast the next 14 days, repeated weekly through Q4",
    }


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
        model = load_demand_model(conn)

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
        "kpi_provenance": {"avg_supplier_lead_time_days": "REAL", "avg_supplier_fulfillment_rate": "LEARNED",
                           "critical_low_stock_skus": "MODEL"},
        "alerts": [_sku_summary(r, counts) for r in urgent],
        "forecast_accuracy": _forecast_accuracy(model),
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


def _supply_chain_frame(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT date, sku_id, warehouse_id, units_sold, promotion_flag, order_quantity FROM supply_chain", conn)


@app.get("/api/forecast-evaluation")
def forecast_eval(protocol: str = "rolling", sku_id: Optional[str] = None):
    """Out-of-sample comparison: actual demand vs baselines vs the learned model (Q4 2024 test period)."""
    if protocol not in ("rolling", "fixed"):
        raise HTTPException(status_code=422, detail="protocol must be 'rolling' or 'fixed'")
    with connect() as conn:
        if sku_id is not None:
            _require_sku(conn, sku_id)
        sc = _supply_chain_frame(conn)
    return forecast_evaluation.evaluate(sc, protocol, sku_id)


@app.get("/api/model")
def model_card():
    """What the predictive layer learned and how well it forecasts (for the UI's model panel)."""
    with connect() as conn:
        model = load_demand_model(conn)
        supplier_meta = load_supplier_meta(conn)
    return {
        "demand_forecaster": {
            "type": f"Seasonal regression: yearly Fourier terms (K={model.k}) + promotion lift that varies with the season",
            "trained_on": {"rows": model.n_rows, "lanes": model.n_lanes, "from": model.trained_from, "to": model.trained_through},
            "r2_in_sample": round(model.r2, 3),
            "daily_noise_sd": round(model.sigma_daily, 2),
            "backtest": model.metrics,
            "not_used": "The source column Demand_Forecast is derived from same-day sales (corr 0.95 vs 0.61 with next day), "
                        "so it is not a real forecast and is not used as an input.",
        },
        "supplier_estimator": supplier_meta,
        "field_provenance": config.FIELD_PROVENANCE,
        "service_level_z": config.SERVICE_LEVEL_Z,
    }


@app.get("/api/products")
def list_products():
    raise HTTPException(status_code=410, detail=PRODUCT_ID_RETIRED)


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------

@app.post("/api/run-agent-pipeline")
def run_agent_pipeline(sku_id: Optional[str] = None, product_id: Optional[int] = None,
                       budget: Optional[float] = Query(None, gt=0)):
    if sku_id is None:
        if product_id is not None:
            raise HTTPException(status_code=410, detail=PRODUCT_ID_RETIRED)
        raise HTTPException(status_code=422, detail="sku_id is required")
    with connect() as conn:
        _require_sku(conn, sku_id)
        return coordinator_agent.run_pipeline(conn, sku_id=sku_id, budget=budget)


# ---------------------------------------------------------------------------
# Supplier decision matrix
# ---------------------------------------------------------------------------

@app.get("/api/suppliers")
def suppliers(sku_id: str, required_quantity: Optional[float] = Query(None, ge=0)):
    """Candidate matrix for one SKU. Without required_quantity, the SKU's
    current requirement from the Perception Agent is used. When no single
    supplier can take the order, the multi-supplier plan the pipeline would
    make is attached as `allocation_preview` (cost, lead time, arrival dates,
    shortfall), so the matrix shows the same answer as the Action Hub."""
    with connect() as conn:
        _require_sku(conn, sku_id)
        fact = perception_agent.observe_sku(conn, sku_id)
        if required_quantity is None:
            required_quantity = fact["RequiredQuantity"]
        result = supplier_agent.rank_suppliers(conn, sku_id, required_quantity=required_quantity, top_n=None)
        if required_quantity > 0:
            knowledge = knowledge_agent.run({"focus_facts": []}, result)
            result["decision"] = knowledge["decision"]
            result["review_reason"] = knowledge["review_reason"]
            if knowledge["split_order"]:
                plan = planning_agent.build_split_plan(
                    conn, fact, knowledge["allocations"], knowledge["backup_candidates"],
                    required_quantity, knowledge["shortfall_qty"])
                preview = {k: plan[k] for k in ("allocations", "order_quantity", "required_quantity", "shortfall_qty",
                                                "partial_order", "total_expense", "estimated_arrival_date")}
                preview.update(budget=knowledge["budget"], quality_constraint_met=knowledge["quality_constraint_met"],
                               budget_constraint_met=knowledge["budget_constraint_met"])
                result["allocation_preview"] = preview
        return result


# ---------------------------------------------------------------------------
# Procurement action & approval
# ---------------------------------------------------------------------------

def _clean(row: dict) -> dict:
    """NaN / <NA> -> None so the Knowledge Agent's `or 0` / `is None` checks behave."""
    return {k: (None if (not isinstance(v, str) and pd.isna(v)) else v) for k, v in row.items()}


def _validate_lines(conn, sku_id: str, lines: list[dict], is_partial: bool, acknowledge_risk: bool) -> tuple[list[dict], int]:
    """Re-derive everything on the server; the client's numbers are never trusted.

    Checks: the SKU has a current requirement; every supplier serves the SKU and has a
    known capacity; quantity within capacity; unit cost taken from the ledger (a
    client cost that disagrees is rejected); High Risk suppliers (Rule 3) need an
    explicit acknowledgement; the total never exceeds the requirement, and is below
    it only when the order is explicitly marked partial.
    """
    if not lines:
        raise HTTPException(status_code=422, detail="allocations must not be empty")
    ids = [l["supplier_id"] for l in lines]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="each supplier may appear only once in an order")
    required = perception_agent.observe_sku(conn, sku_id)["RequiredQuantity"]
    if required <= 0:
        raise HTTPException(status_code=422, detail=f"{sku_id} has no current procurement requirement")
    matrix = {r["supplier_id"]: _clean(r) for r in supplier_agent.candidate_matrix(conn, sku_id).to_dict(orient="records")}

    out = []
    for l in lines:
        sid, qty = l["supplier_id"], l["quantity"]
        if qty <= 0:
            raise HTTPException(status_code=422, detail=f"{sid}: quantity must be positive")
        cand = matrix.get(sid)
        if cand is None:
            raise HTTPException(status_code=422, detail=f"{sid} does not supply {sku_id}")
        if cand.get("available_qty") is None:
            raise HTTPException(status_code=422, detail=f"{sid} has no known capacity for {sku_id}")
        if qty > int(cand["available_qty"]):
            raise HTTPException(status_code=422, detail=(
                f"{sid} has {int(cand['available_qty'])} units of {sku_id} available; {qty} requested"))
        unit_cost = round(float(cand["unit_cost"]), supplier_agent.DISPLAY_DECIMALS)
        if l.get("unit_cost") is not None and abs(l["unit_cost"] - unit_cost) > 0.01:
            raise HTTPException(status_code=422, detail=(
                f"{sid}: unit cost {l['unit_cost']} does not match the ledger cost {unit_cost}; re-run the pipeline"))
        status = knowledge_agent.evaluate_supplier(cand, qty, [])["supplier_status"]
        if status == "High Risk" and not acknowledge_risk:
            raise HTTPException(status_code=422, detail=(
                f"{sid} is High Risk (predicted defect rate {cand['defect_rate']:.1%} > "
                f"{config.RULE_HIGH_RISK_DEFECT_RATE:.0%}); set acknowledge_risk to order from it anyway"))
        out.append({"supplier_id": sid, "quantity": int(qty), "unit_cost": unit_cost,
                    "estimated_arrival": l.get("estimated_arrival"), "risk_status": status})

    total = sum(o["quantity"] for o in out)
    if total > required:
        raise HTTPException(status_code=422, detail=f"order total {total} exceeds the current requirement of {required}")
    if total < required and not is_partial:
        raise HTTPException(status_code=422, detail=(
            f"order total {total} covers only part of the requirement ({required}); mark it is_partial to approve it"))
    return out, required


def _insert_order(conn, sku_id: str, lines: list[dict], backup: Optional[str], is_partial: bool) -> list[dict]:
    """Insert every line inside the caller's open transaction; the group id ties them together."""
    group = f"PO-{uuid.uuid4().hex[:10].upper()}"
    today = datetime.now().date().isoformat()
    ids = []
    for l in lines:
        cur = conn.execute(
            """INSERT INTO purchase_orders
               (sku_id, product_name, supplier_id, supplier_name, quantity, unit_cost, total_cost, order_date,
                estimated_arrival, backup_supplier_id, status, order_group, is_partial)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sku_id, sku_id, l["supplier_id"], l["supplier_id"], l["quantity"], l["unit_cost"],
             round(l["quantity"] * l["unit_cost"], 2), today, l["estimated_arrival"], backup, "Approved",
             group, int(is_partial)),
        )
        ids.append(cur.lastrowid)
    marks = ",".join("?" * len(ids))
    return [dict(r) for r in conn.execute(f"SELECT * FROM purchase_orders WHERE id IN ({marks}) ORDER BY id", ids)]


def _approve(sku_id: str, lines: list[dict], backup: Optional[str], is_partial: bool, acknowledge_risk: bool) -> list[dict]:
    """Validate and save an order as ONE transaction (BEGIN IMMEDIATE also serialises
    concurrent approvals): either every purchase order is written or none is."""
    with connect() as conn:
        conn.isolation_level = None  # explicit transaction control
        _require_sku(conn, sku_id)
        try:
            conn.execute("BEGIN IMMEDIATE")
            validated, _ = _validate_lines(conn, sku_id, lines, is_partial, acknowledge_risk)
            rows = _insert_order(conn, sku_id, validated, backup, is_partial)
            conn.execute("COMMIT")
        except HTTPException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        except sqlite3.Error as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise HTTPException(status_code=500, detail=f"purchase order not saved (rolled back): {exc}") from exc
    return rows


@app.post("/api/approve-order")
def approve_order(payload: ApproveOrderRequest):
    line = {"supplier_id": payload.supplier_id, "quantity": payload.quantity,
            "unit_cost": payload.unit_cost, "estimated_arrival": payload.estimated_arrival}
    rows = _approve(payload.sku_id, [line], payload.backup_supplier_id, payload.is_partial, payload.acknowledge_risk)
    return {"status": "approved", "order_group": rows[0]["order_group"], "purchase_order": rows[0]}


class SplitLine(BaseModel):
    supplier_id: str
    quantity: int
    unit_cost: float
    estimated_arrival: Optional[str] = None


class ApproveSplitOrderRequest(BaseModel):
    sku_id: str
    allocations: list[SplitLine]
    backup_supplier_id: Optional[str] = None
    is_partial: bool = False
    acknowledge_risk: bool = False


@app.post("/api/approve-split-order")
def approve_split_order(payload: ApproveSplitOrderRequest):
    """Approve a split order: one purchase order per supplier, created all-or-nothing
    under one order_group. Every line is validated first; if any line is invalid,
    no purchase order is created."""
    lines = [l.model_dump() for l in payload.allocations]
    rows = _approve(payload.sku_id, lines, payload.backup_supplier_id, payload.is_partial, payload.acknowledge_risk)
    return {"status": "approved", "order_group": rows[0]["order_group"], "purchase_orders": rows,
            "total_quantity": sum(r["quantity"] for r in rows), "total_cost": round(sum(r["total_cost"] for r in rows), 2)}


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
