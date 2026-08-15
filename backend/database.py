"""
Data ingestion layer: loads the 6 raw CSVs, cleans them, and materializes
cleaned + derived tables into a local SQLite database for fast agent queries.

Run standalone (`python backend/database.py`) to (re)build data/insightiq.db.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import numpy as np
import pandas as pd

from backend import config

# ---------------------------------------------------------------------------
# Low level helpers
# ---------------------------------------------------------------------------

def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """Several source CSVs have stray trailing commas that pandas reads as
    empty 'Unnamed: N' columns. Strip them."""
    keep = [c for c in df.columns if not str(c).startswith("Unnamed")]
    return df[keep].copy()


def get_connection() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connect():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Individual dataset cleaners
# ---------------------------------------------------------------------------

def _load_products() -> pd.DataFrame:
    df = pd.read_csv(config.CSV_PATHS["product"])
    df = _drop_unnamed(df)
    df["Discontinued"] = df["Discontinued"].astype(bool)
    df["ReorderLevel"] = df["ReorderLevel"].clip(lower=1)
    df["TargetStock"] = df["ReorderLevel"] * config.TARGET_STOCK_MULTIPLIER
    df["Deficit"] = (df["TargetStock"] - df["UnitsInStock"]).clip(lower=0)
    df["Urgent"] = df["UnitsInStock"] < df["ReorderLevel"]
    df.columns = [
        "product_id", "product_name", "category", "supplier_id", "unit_price",
        "units_in_stock", "units_on_order", "reorder_level", "discontinued",
        "target_stock", "deficit", "urgent",
    ]
    return df


def _load_orders() -> pd.DataFrame:
    df = pd.read_csv(config.CSV_PATHS["order"])
    df = _drop_unnamed(df)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    df["ShippedDate"] = pd.to_datetime(df["ShippedDate"], errors="coerce")
    lead = (df["ShippedDate"] - df["OrderDate"]).dt.days
    # Source data has inconsistencies (shipped dates before order dates for
    # still-pending orders); clip to a sane non-negative window for analytics.
    df["LeadTimeDays"] = lead.clip(lower=0, upper=90)
    df.columns = [
        "order_id", "customer_id", "order_date", "order_status", "shipped_date",
        "ship_address", "ship_country", "shipping_fee", "payment_method",
        "total_amount", "lead_time_days",
    ]
    return df


def _load_customers() -> pd.DataFrame:
    df = pd.read_csv(config.CSV_PATHS["customer"])
    df = _drop_unnamed(df)
    df["DateOfBirth"] = pd.to_datetime(df["DateOfBirth"], errors="coerce")
    df.columns = [
        "customer_id", "first_name", "last_name", "email", "phone_number",
        "address", "country", "date_of_birth", "gender",
    ]
    return df


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def _load_calendar() -> pd.DataFrame:
    df = pd.read_csv(config.CSV_PATHS["calendar"])
    df = _drop_unnamed(df)
    # The raw day/month/year/weekday/season columns are internally
    # inconsistent with the `date` field itself (classic messy-data
    # exercise CSV). We trust `date` as ground truth and recompute
    # every derived field from it instead of the noisy stored columns.
    parsed = pd.to_datetime(df["date"], errors="coerce")
    df = df.loc[parsed.notna()].copy()
    parsed = parsed.loc[parsed.notna()]
    df["date"] = parsed
    df["day"] = parsed.dt.day
    df["month"] = parsed.dt.month
    df["year"] = parsed.dt.year
    df["quarter"] = parsed.dt.quarter
    df["weekday"] = parsed.dt.day_name()
    df["week_number"] = parsed.dt.isocalendar().week.astype(int)
    df["is_weekend"] = parsed.dt.weekday >= 5
    df["season"] = df["month"].apply(_season_from_month)
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    return df[["date", "day", "month", "year", "quarter", "weekday", "week_number", "is_weekend", "season"]]


def _load_procurement_kpi() -> pd.DataFrame:
    df = pd.read_csv(config.CSV_PATHS["procurement_kpi"])
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors="coerce")
    df["DeliveryDays"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days
    df["DefectRate"] = (df["Defective_Units"] / df["Quantity"]).clip(lower=0, upper=1)
    df["CostVariance"] = ((df["Unit_Price"] - df["Negotiated_Price"]) / df["Unit_Price"]).clip(-1, 1)
    df["OnTime"] = df["Order_Status"] == "Delivered"
    df["CompliantFlag"] = df["Compliance"] == "Yes"
    df.columns = [
        "po_id", "supplier", "order_date", "delivery_date", "item_category",
        "order_status", "quantity", "unit_price", "negotiated_price",
        "defective_units", "compliance", "delivery_days", "defect_rate",
        "cost_variance", "on_time", "compliant_flag",
    ]
    return df


def _load_supply_chain() -> pd.DataFrame:
    df = pd.read_csv(config.CSV_PATHS["supply_chain"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.columns = [
        "date", "sku_id", "warehouse_id", "supplier_id", "region", "units_sold",
        "inventory_level", "supplier_lead_time_days", "reorder_point",
        "order_quantity", "unit_cost", "unit_price", "promotion_flag",
        "stockout_flag", "demand_forecast",
    ]
    return df


# ---------------------------------------------------------------------------
# Derived: unified supplier profile pool
# ---------------------------------------------------------------------------

def _build_supplier_profiles(kpi: pd.DataFrame, supply_chain: pd.DataFrame) -> pd.DataFrame:
    """Bridges two disjoint supplier ID spaces.

    supply_chain_dataset1.csv models a *live* catalog of 10 numeric supplier
    IDs (SUP_1..SUP_10) with real-time pricing/stock/lead-time telemetry, but
    carries no quality/rating signal. The Procurement KPI dataset models 5
    *named* legacy vendors (Alpha_Inc, Beta_Supplies, ...) with rich quality
    history (defect rate, compliance, cost variance) but no live stock/price.
    Neither dataset shares a join key with the other, so we deterministically
    round-robin-assign each live SUP_x a named-vendor KPI benchmark to
    reconstruct a single queryable supplier pool with both live logistics
    data and historical quality data, exactly as the Supplier Intelligence
    Agent needs for heuristic ranking.
    """
    sc_g = supply_chain.groupby("supplier_id").agg(
        avg_cost=("unit_cost", "mean"),
        avg_lead_time=("supplier_lead_time_days", "mean"),
        lead_time_std=("supplier_lead_time_days", "std"),
        avg_forecast_error=("demand_forecast", "mean"),
        region=("region", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
        sku_count=("sku_id", "nunique"),
    ).reset_index()

    latest_date = supply_chain["date"].max()
    latest_stock = (
        supply_chain[supply_chain["date"] == latest_date]
        .groupby("supplier_id")["inventory_level"].sum()
        .rename("stock_availability")
        .reset_index()
    )
    sc_g = sc_g.merge(latest_stock, on="supplier_id", how="left")
    sc_g["stock_availability"] = sc_g["stock_availability"].fillna(sc_g["avg_cost"] * 0).fillna(0)

    lo, hi = sc_g["lead_time_std"].min(), sc_g["lead_time_std"].max()
    span = (hi - lo) if hi > lo else 1.0
    sc_g["reliability_score"] = 1.0 - ((sc_g["lead_time_std"] - lo) / span)
    sc_g["reliability_score"] = sc_g["reliability_score"].clip(0.5, 1.0).round(3)

    kpi_g = kpi.groupby("supplier").agg(
        fulfillment_rate=("on_time", "mean"),
        compliance_rate=("compliant_flag", "mean"),
        defect_rate=("defect_rate", "mean"),
        cost_variance=("cost_variance", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
    ).reset_index()
    kpi_g["rating"] = (
        5 * (0.4 * kpi_g["fulfillment_rate"] + 0.3 * kpi_g["compliance_rate"]
             + 0.3 * (1 - kpi_g["defect_rate"].clip(0, 1)))
    ).clip(1, 5).round(2)
    kpi_g["defect_rate"] = kpi_g["defect_rate"].round(4)

    named_suppliers = sorted(kpi_g["supplier"].unique())
    live_suppliers = sorted(sc_g["supplier_id"].unique(), key=lambda s: int(s.split("_")[1]))
    mapping = {sup: named_suppliers[i % len(named_suppliers)] for i, sup in enumerate(live_suppliers)}
    sc_g["benchmarked_against"] = sc_g["supplier_id"].map(mapping)

    profiles = sc_g.merge(
        kpi_g[["supplier", "rating", "defect_rate", "cost_variance", "avg_delivery_days"]],
        left_on="benchmarked_against", right_on="supplier", how="left",
    ).drop(columns=["supplier"])

    profiles["supplier_name"] = profiles["supplier_id"] + " (" + profiles["benchmarked_against"].str.replace("_", " ") + " network)"
    profiles = profiles.rename(columns={"avg_delivery_days": "kpi_delivery_days"})
    # Blend live logistics lead time (authoritative, current) with historical
    # KPI delivery days for a stabilized delivery-time estimate.
    profiles["delivery_time_days"] = (
        0.7 * profiles["avg_lead_time"] + 0.3 * profiles["kpi_delivery_days"].fillna(profiles["avg_lead_time"])
    ).round(1)

    cols = [
        "supplier_id", "supplier_name", "benchmarked_against", "region", "sku_count",
        "avg_cost", "delivery_time_days", "avg_lead_time", "reliability_score",
        "stock_availability", "rating", "defect_rate", "cost_variance",
    ]
    return profiles[cols].sort_values("supplier_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

TABLES_TO_BUILD = [
    "products", "orders", "customers", "calendar", "procurement_kpi",
    "supply_chain", "supplier_profiles",
]


def build_database(force: bool = False) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.DB_PATH.exists() and not force:
        return

    products = _load_products()
    orders = _load_orders()
    customers = _load_customers()
    calendar = _load_calendar()
    kpi = _load_procurement_kpi()
    supply_chain = _load_supply_chain()
    supplier_profiles = _build_supplier_profiles(kpi, supply_chain)

    with connect() as conn:
        products.to_sql("products", conn, if_exists="replace", index=False)
        orders.to_sql("orders", conn, if_exists="replace", index=False)
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        calendar.to_sql("calendar", conn, if_exists="replace", index=False)
        kpi.to_sql("procurement_kpi", conn, if_exists="replace", index=False)
        supply_chain.to_sql("supply_chain", conn, if_exists="replace", index=False)
        supplier_profiles.to_sql("supplier_profiles", conn, if_exists="replace", index=False)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS purchase_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                product_name TEXT,
                supplier_id TEXT,
                supplier_name TEXT,
                quantity INTEGER,
                unit_cost REAL,
                total_cost REAL,
                order_date TEXT,
                estimated_arrival TEXT,
                backup_supplier_id TEXT,
                status TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_urgent ON products(urgent)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_calendar_date ON calendar(date)")
        conn.commit()


if __name__ == "__main__":
    build_database(force=True)
    with connect() as conn:
        for t in TABLES_TO_BUILD + ["purchase_orders"]:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"{t:20s} {n:>8} rows")
