"""
Daily demand series and baseline forecasts per SKU × warehouse lane (REAL data).

Granularity of `supply_chain`: one row per (date, SKU, warehouse) and exactly one
supplier per lane all year, so no lane-day is counted twice today. The series is
still built defensively, in case a future feed repeats a lane-day across several
supplier rows:

  * exact duplicate rows are dropped;
  * Units_Sold is SUMMED over the remaining rows of a lane-day (each row is taken
    to be a distinct sale record);
  * Inventory_Level / Reorder_Point are lane-level STATE readings, so they are never
    summed: the perception layer takes one value per lane-day.

A day missing from a lane's calendar is filled with 0 sales (the source has none
missing: 365 days × 250 lanes).

Series are held "wide": rows = dates (daily, ascending), columns = (sku_id,
warehouse_id) lanes. Every baseline below looks only at rows it is given, so a
caller that slices history to dates < t can never leak information from t onward.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

LANE_KEYS = ["sku_id", "warehouse_id"]


def aggregate_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw supply-chain rows -> wide daily Units_Sold matrix (dates × lanes)."""
    df = raw.drop_duplicates().copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    wide = df.pivot_table(index="date", columns=LANE_KEYS, values="units_sold", aggfunc="sum")
    if wide.empty:
        return wide
    wide = wide.asfreq("D").fillna(0.0).astype(float)
    wide.columns = pd.MultiIndex.from_tuples(list(wide.columns), names=LANE_KEYS)
    return wide


def daily_lane_demand(conn: sqlite3.Connection, since: str | None = None) -> pd.DataFrame:
    """Wide daily demand from the `supply_chain` table, optionally from `since` on."""
    sql = "SELECT date, sku_id, warehouse_id, supplier_id, units_sold FROM supply_chain"
    params: tuple = ()
    if since is not None:
        sql += " WHERE date >= ?"
        params = (since,)
    return aggregate_daily(pd.read_sql_query(sql, conn, params=params))


# ---------------------------------------------------------------------------
# Baselines — each returns a (horizon × lanes) array of daily forecasts
# ---------------------------------------------------------------------------

def moving_average(history: np.ndarray, horizon: int, window: int) -> np.ndarray:
    """Mean of the last `window` days, held flat over the horizon."""
    rate = history[-window:].mean(axis=0)
    return np.tile(rate, (horizon, 1))


def seasonal_naive(history: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the last full season (weekly by default)."""
    last = history[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last, (reps, 1))[:horizon]


# ---------------------------------------------------------------------------
# Metrics — safe on zero demand
# ---------------------------------------------------------------------------

def error_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    """MAE, WAPE and bias. WAPE = Σ|e| / Σ|actual|; it is undefined (None) when
    total actual demand is zero, rather than dividing by zero. Per-point MAPE is
    not reported because zero-demand days make it unbounded."""
    actual = np.asarray(actual, dtype=float).ravel()
    forecast = np.asarray(forecast, dtype=float).ravel()
    err = forecast - actual
    denom = np.abs(actual).sum()
    return {
        "n": int(actual.size),
        "mae": round(float(np.abs(err).mean()), 4) if actual.size else None,
        "wape": round(float(np.abs(err).sum() / denom), 4) if denom > 0 else None,
        "bias": round(float(err.sum() / denom), 4) if denom > 0 else None,
    }
