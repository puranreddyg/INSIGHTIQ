"""
Perception Agent — continuous environment observation (PEAS model).

Percepts:  product.csv (stock levels), order.csv (fulfillment/lead-time
           signals), customer.csv (outbound demand signals)
Actions:   emits structured state facts consumed by the downstream agents.
Environment: enterprise inventory + order ledger (partially observable,
             refreshed on each pipeline run).
Sensors:   SQLite queries against the cleaned `products`, `orders`, and
           `customers` tables built by backend/database.py.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd


def _row_to_fact(row: sqlite3.Row) -> dict:
    return {
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "category": row["category"],
        "supplier_id": row["supplier_id"],
        "SKU": f"SKU-{row['product_id']:04d}",
        "Inventory": row["units_in_stock"],
        "ReorderPoint": row["reorder_level"],
        "TargetStock": row["target_stock"],
        "Deficit": row["deficit"],
        "Urgent": bool(row["urgent"]),
        "UnitPrice": row["unit_price"],
    }


def observe_product(conn: sqlite3.Connection, product_id: int) -> Optional[dict]:
    """Perceive the current inventory state of a single SKU."""
    row = conn.execute(
        "SELECT * FROM products WHERE product_id = ?", (product_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_fact(row)


def scan_environment(conn: sqlite3.Connection, limit: int = 500) -> list[dict]:
    """Scan the full product ledger for urgent (below-reorder-point) SKUs."""
    rows = conn.execute(
        "SELECT * FROM products WHERE urgent = 1 ORDER BY deficit DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_fact(r) for r in rows]


def demand_signals(conn: sqlite3.Connection, top_n: int = 8) -> dict:
    """Aggregate outbound demand trend from customers joined with orders, and
    an overall order-volume trend for seasonality context."""
    orders = pd.read_sql_query(
        "SELECT customer_id, order_date, total_amount, lead_time_days FROM orders", conn,
        parse_dates=["order_date"],
    )
    customers = pd.read_sql_query("SELECT customer_id, country FROM customers", conn)

    merged = orders.merge(customers, on="customer_id", how="left")
    by_country = (
        merged.groupby("country")["total_amount"].sum().sort_values(ascending=False).head(top_n)
    )
    top_buyers = (
        merged.groupby("customer_id")["total_amount"].sum().sort_values(ascending=False).head(top_n)
    )
    monthly = merged.dropna(subset=["order_date"]).copy()
    monthly["month"] = monthly["order_date"].dt.to_period("M").astype(str)
    trend = monthly.groupby("month")["total_amount"].agg(["count", "sum"]).tail(12)

    avg_lead_time = float(orders["lead_time_days"].mean())

    return {
        "demand_by_country": [
            {"country": c, "total_amount": round(float(v), 2)} for c, v in by_country.items()
        ],
        "top_buyers": [
            {"customer_id": int(c), "total_amount": round(float(v), 2)} for c, v in top_buyers.items()
        ],
        "monthly_order_trend": [
            {"month": m, "order_count": int(r["count"]), "total_amount": round(float(r["sum"]), 2)}
            for m, r in trend.iterrows()
        ],
        "avg_order_lead_time_days": round(avg_lead_time, 1) if pd.notna(avg_lead_time) else None,
    }


def run(conn: sqlite3.Connection, product_id: Optional[int] = None) -> dict:
    """Entry point used by the Coordinator Agent."""
    if product_id is not None:
        fact = observe_product(conn, product_id)
        facts = [fact] if fact else []
    else:
        facts = scan_environment(conn, limit=1)

    urgent_scan = scan_environment(conn, limit=25)
    signals = demand_signals(conn)

    return {
        "agent": "Perception Agent",
        "focus_facts": facts,
        "urgent_skus": urgent_scan,
        "urgent_sku_count": len(urgent_scan),
        "demand_signals": signals,
    }
