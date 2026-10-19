"""
Out-of-sample comparison of the learned demand forecaster with simple baselines,
for the website's Forecast Evaluation page.

Data          supply_chain ledger: daily Units_Sold for 250 SKU-warehouse lanes.
Training      2024-01-01 .. CUTOFF (2024-09-30)        — the model never sees later data.
Test          CUTOFF+1 .. last date (2024-10-01 .. 2024-12-30, 91 days).

Two protocols:
  rolling   every 7 days the model is refitted on everything known so far and forecasts
            the next 7 days; the baselines are recomputed at the same moments. This is how
            the system is used (forecasts are refreshed), and it is fair to the baselines.
  fixed     one forecast made on CUTOFF for all 91 test days (harder; long horizon).

Methods
  model      learned forecaster (seasonal regression + promotion lift + lane scale), with
             future promotions unknown -> the historical promotion rate is used
  ma28       mean of each lane's last 28 days (the app's previous method)
  ma7        mean of each lane's last 7 days
Metrics (per lane per day, and on the all-lane daily total)
  MAE, RMSE, MAPE (days with zero actual sales are excluded from MAPE and counted), WAPE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import ORDER_CYCLE_FALLBACK_DAYS
from backend.ml import demand_forecaster

CUTOFF = "2024-09-30"
STEP_DAYS = 7
METHODS = {"model": "Learned model", "ma28": "Baseline: 28-day average (previous method)", "ma7": "Baseline: 7-day average"}


def _panel(sc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[tuple[str, str]]]:
    sc = sc.assign(date=pd.to_datetime(sc["date"]))
    sold = sc.pivot_table(index=["sku_id", "warehouse_id"], columns="date", values="units_sold", aggfunc="sum")
    return sold, pd.DatetimeIndex(sold.columns), list(sold.index)


def _forecast_block(sc: pd.DataFrame, sold: pd.DataFrame, dates: pd.DatetimeIndex, origin: pd.Timestamp,
                    horizon: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Forecasts (lanes x len(horizon)) made with information up to `origin` only."""
    past = sc[pd.to_datetime(sc["date"]) <= origin]
    model = demand_forecaster.fit(past, run_backtest=False)
    stats = demand_forecaster.lane_statistics(past, model, ORDER_CYCLE_FALLBACK_DAYS).set_index(["sku_id", "warehouse_id"])
    scale = stats.demand_scale.reindex(sold.index).fillna(1.0).to_numpy()
    curve = model.expected_daily(horizon)                       # pooled expected units/day
    hist = sold.loc[:, dates <= origin].to_numpy(float)
    n = len(horizon)
    return {
        "model": np.outer(scale, curve),
        "ma28": np.repeat(hist[:, -28:].mean(axis=1, keepdims=True), n, axis=1),
        "ma7": np.repeat(hist[:, -7:].mean(axis=1, keepdims=True), n, axis=1),
    }


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    nz = actual != 0
    return {
        "mae": round(float(np.abs(err).mean()), 3),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "mape_pct": round(float(np.abs(err[nz] / actual[nz]).mean() * 100), 2),
        "wape_pct": round(float(np.abs(err).sum() / np.abs(actual).sum() * 100), 2),
        "mape_excluded_zero_days": int((~nz).sum()),
        "n": int(actual.size),
    }


def _evaluate(sc: pd.DataFrame, protocol: str) -> tuple[dict, dict[str, np.ndarray], np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
    sold, dates, lanes = _panel(sc)
    cutoff = pd.Timestamp(CUTOFF)
    test_dates = dates[dates > cutoff]
    preds = {m: np.zeros((len(lanes), len(test_dates))) for m in METHODS}
    if protocol == "fixed":
        block = _forecast_block(sc, sold, dates, cutoff, test_dates)
        for m in METHODS:
            preds[m][:] = block[m]
    else:
        for start in range(0, len(test_dates), STEP_DAYS):
            horizon = test_dates[start:start + STEP_DAYS]
            origin = horizon[0] - pd.Timedelta(days=1)
            block = _forecast_block(sc, sold, dates, origin, horizon)
            for m in METHODS:
                preds[m][:, start:start + len(horizon)] = block[m]
    actual = sold.loc[:, test_dates].to_numpy(float)
    metrics = {
        "lane_daily": {m: _metrics(actual, preds[m]) for m in METHODS},
        "total_daily": {m: _metrics(actual.sum(axis=0), preds[m].sum(axis=0)) for m in METHODS},
    }
    return metrics, preds, actual, test_dates, sold


_CACHE: dict[tuple[str, str], dict] = {}


def _cached(sc: pd.DataFrame, protocol: str) -> dict:
    """The evaluation refits the model ~13 times, so results are cached per data snapshot."""
    key = (f"{len(sc)}|{sc['date'].max()}|{int(sc['units_sold'].sum())}", protocol)
    if key not in _CACHE:
        metrics, preds, actual, test_dates, sold = _evaluate(sc, protocol)
        _CACHE[key] = {"metrics": metrics, "preds": preds, "actual": actual, "test_dates": test_dates, "sold": sold}
    return _CACHE[key]


def evaluate(sc: pd.DataFrame, protocol: str = "rolling", sku_id: str | None = None) -> dict:
    """Full payload for the page: periods, metrics, chart series (total, or one SKU)."""
    if protocol not in ("rolling", "fixed"):
        raise ValueError("protocol must be 'rolling' or 'fixed'")
    sc = sc[["date", "sku_id", "warehouse_id", "units_sold", "promotion_flag", "order_quantity"]]
    r = _cached(sc, protocol)
    sold, test_dates = r["sold"], r["test_dates"]
    dates = pd.DatetimeIndex(sold.columns)
    if sku_id is not None:
        mask = np.array([s == sku_id for s, _ in sold.index])
        if not mask.any():
            raise KeyError(sku_id)
    else:
        mask = np.ones(len(sold.index), dtype=bool)
    hist = sold.to_numpy(float)[mask].sum(axis=0)
    series = {"dates": [d.strftime("%Y-%m-%d") for d in dates], "actual": [round(float(v), 2) for v in hist],
              "test_start_index": int(np.searchsorted(dates, test_dates[0]))}
    for m in METHODS:
        series[m] = [round(float(v), 2) for v in r["preds"][m][mask].sum(axis=0)]
    sku_metrics = None
    if sku_id is not None:
        a = r["actual"][mask]
        sku_metrics = {m: _metrics(a.sum(axis=0), r["preds"][m][mask].sum(axis=0)) for m in METHODS}
    m_total = r["metrics"]["total_daily"]
    m_lane = r["metrics"]["lane_daily"]
    best = min(METHODS, key=lambda k: m_lane[k]["mae"])
    return {
        "protocol": protocol,
        "methods": METHODS,
        "periods": {
            "data": [dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")],
            "train": [dates[0].strftime("%Y-%m-%d"), CUTOFF],
            "test": [test_dates[0].strftime("%Y-%m-%d"), test_dates[-1].strftime("%Y-%m-%d")],
            "test_days": int(len(test_dates)),
            "refit_every_days": STEP_DAYS if protocol == "rolling" else None,
            "lanes": int(len(sold.index)),
        },
        "metrics": r["metrics"],
        "sku_metrics": sku_metrics,
        "best_method_lane_mae": best,
        "model_beats_baselines": bool(all(m_lane["model"][k] < m_lane[b][k] for b in ("ma28", "ma7") for k in ("mae", "rmse"))
                                      and all(m_total["model"][k] < m_total[b][k] for b in ("ma28", "ma7") for k in ("mae", "rmse"))),
        "series_scope": sku_id or "all SKUs (sum of 250 lanes)",
        "series": series,
    }
