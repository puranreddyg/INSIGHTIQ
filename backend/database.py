"""
Data ingestion layer: loads the 6 raw CSVs plus the simulated supplier-SKU
profile, cleans them, and materializes cleaned + derived tables into a local
SQLite database for fast agent queries. Raw CSVs are only ever read.

Run standalone (`python backend/database.py`) to (re)build data/insightiq.db.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

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
    """Reference only. product.csv has no link to any supplier or SKU (its
    SupplierID equals ProductID on every row), so it takes no part in
    procurement and carries no derived urgency/deficit columns."""
    df = pd.read_csv(config.CSV_PATHS["product"])
    df = _drop_unnamed(df)
    df["Discontinued"] = df["Discontinued"].astype(bool)
    df.columns = [
        "product_id", "product_name", "category", "supplier_id", "unit_price",
        "units_in_stock", "units_on_order", "reorder_level", "discontinued",
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
# Derived: SKU-supplier lanes (REAL) and supplier-SKU profile (SIMULATED)
# ---------------------------------------------------------------------------

LANE_ATTRIBUTES = ["supplier_id", "unit_cost", "supplier_lead_time_days"]


def _build_sku_supplier_lanes(supply_chain: pd.DataFrame) -> pd.DataFrame:
    """One row per (SKU, warehouse) lane with its supplier's real unit cost and
    lead time. These never change over the year in the source, so collapsing the
    365 daily rows is lossless; if that ever stops being true, fail loudly rather
    than silently averaging."""
    lanes = supply_chain.groupby(["sku_id", "warehouse_id"])[LANE_ATTRIBUTES]
    varying = lanes.nunique().gt(1).any(axis=1)
    if varying.any():
        raise ValueError(f"lane attributes vary over time for {list(varying[varying].index)[:5]}")
    return (
        lanes.first().reset_index()
        .rename(columns={"supplier_lead_time_days": "lead_time_days"})
        [["sku_id", "warehouse_id", "supplier_id", "unit_cost", "lead_time_days"]]
    )


PROFILE_COLUMNS = ["sku_id", "supplier_id", "available_qty", "defect_rate", "fulfillment_rate"]


def _load_supplier_sku_profile(supply_chain: pd.DataFrame) -> pd.DataFrame:
    """Load the supplier-SKU profile (simulated today, ERP-fed later) and reject
    anything that would invent a relationship or carry impossible values."""
    df = pd.read_csv(config.CSV_PATHS["supplier_sku_profile"])
    if list(df.columns) != PROFILE_COLUMNS:
        raise ValueError(f"supplier_sku_profile columns {list(df.columns)} != {PROFILE_COLUMNS}")
    if df.duplicated(["sku_id", "supplier_id"]).any():
        raise ValueError("supplier_sku_profile has duplicate (sku_id, supplier_id) rows")
    real_pairs = set(map(tuple, supply_chain[["sku_id", "supplier_id"]].drop_duplicates().to_numpy()))
    invented = [p for p in map(tuple, df[["sku_id", "supplier_id"]].to_numpy()) if p not in real_pairs]
    if invented:
        raise ValueError(f"supplier_sku_profile contains pairs absent from supply_chain: {invented[:5]}")
    if (df["available_qty"] < 0).any() or not df[["defect_rate", "fulfillment_rate"]].stack().between(0, 1).all():
        raise ValueError("supplier_sku_profile contains out-of-range values")
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

TABLES_TO_BUILD = [
    "products", "orders", "customers", "calendar", "procurement_kpi",
    "supply_chain", "sku_supplier_lanes", "supplier_sku_profile",
]

# Removed in schema v2: built on an unsupported round-robin mapping of KPI
# vendors onto SUP_x (docs/migration/DELETION_LOG.md).
OBSOLETE_TABLES = ["supplier_profiles"]
OBSOLETE_INDEXES = ["idx_products_urgent"]


def _schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _ensure_purchase_orders(conn: sqlite3.Connection) -> None:
    """Create the PO table, or migrate an existing one in place so previously
    approved purchase orders are kept."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS purchase_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku_id TEXT,
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
    columns = {r[1] for r in conn.execute("PRAGMA table_info(purchase_orders)").fetchall()}
    for name, sql_type in PURCHASE_ORDER_ADDED_COLUMNS.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE purchase_orders ADD COLUMN {name} {sql_type}")


# Columns added after the first schema. A split order is one row per supplier
# sharing an order_group; a legacy single-supplier PO is a group of one.
PURCHASE_ORDER_ADDED_COLUMNS = {
    "sku_id": "TEXT",
    "order_group": "TEXT",
    "allocation_line": "INTEGER",
    "allocation_count": "INTEGER",
    "required_quantity": "INTEGER",
    "is_partial": "INTEGER DEFAULT 0",
    "risk_status": "TEXT",
}


def build_database(force: bool = False) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.DB_PATH.exists() and not force:
        with connect() as conn:
            if _schema_version(conn) == config.SCHEMA_VERSION:
                _ensure_purchase_orders(conn)
                conn.commit()
                return

    supply_chain = _load_supply_chain()
    tables = {
        "products": _load_products(),
        "orders": _load_orders(),
        "customers": _load_customers(),
        "calendar": _load_calendar(),
        "procurement_kpi": _load_procurement_kpi(),
        "supply_chain": supply_chain,
        "sku_supplier_lanes": _build_sku_supplier_lanes(supply_chain),
        "supplier_sku_profile": _load_supplier_sku_profile(supply_chain),
    }

    with connect() as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
        for name in OBSOLETE_INDEXES:
            conn.execute(f"DROP INDEX IF EXISTS {name}")
        for name in OBSOLETE_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {name}")
        _ensure_purchase_orders(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_calendar_date ON calendar(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_supply_chain_date ON supply_chain(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_supply_chain_sku_date ON supply_chain(sku_id, date)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_lanes_sku_warehouse ON sku_supplier_lanes(sku_id, warehouse_id)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_profile_sku_supplier ON supplier_sku_profile(sku_id, supplier_id)")
        conn.execute(f"PRAGMA user_version = {config.SCHEMA_VERSION}")
        conn.commit()


if __name__ == "__main__":
    build_database(force=True)
    with connect() as conn:
        for t in TABLES_TO_BUILD + ["purchase_orders"]:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"{t:20s} {n:>8} rows")
