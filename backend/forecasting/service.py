"""
Forecast service used by the Perception Agent and the /api/forecast endpoints:
resolves which method to use (with fallback) and forecasts daily demand per lane.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend import config
from backend.forecasting import model as demand_model
from backend.forecasting.demand import moving_average, seasonal_naive

METHODS = {"auto", "moving_average", "seasonal_naive", "model"}


def resolve_method(requested: str | None = None, artifact_path: Path | None = None) -> dict:
    """Which forecast to use and why. The model is only ever chosen when its
    artifact loads and validates. "auto" takes the method with the lowest
    held-out horizon WAPE recorded in the artifact (baselines included); without
    a valid artifact it falls back to the moving average."""
    requested = requested or config.FORECAST_METHOD
    if requested not in METHODS:
        raise ValueError(f"unknown FORECAST_METHOD {requested!r}")
    artifact, model, error = demand_model.load_artifact(artifact_path or demand_model.ARTIFACT_PATH)
    info = {"requested": requested, "artifact_error": error, "artifact": artifact, "model": model}

    if requested in ("moving_average", "seasonal_naive"):
        return {**info, "method": requested, "reason": f"{requested} requested in config"}
    if model is None:
        return {**info, "method": "moving_average", "reason": f"fallback to moving average — {error}"}
    if requested == "model":
        return {**info, "method": "model", "reason": "model requested in config and artifact is valid"}
    sel = artifact.get("selection", {})
    best = sel.get("best_method")
    if best not in ("model", "moving_average", "seasonal_naive"):
        return {**info, "method": "moving_average", "reason": "fallback to moving average — artifact has no selection"}
    wapes = ", ".join(f"{m} {w}" for m, w in sel.get("horizon_wape", {}).items())
    return {**info, "method": best, "reason": f"{sel.get('criterion', 'lowest held-out WAPE')} ({wapes})"}


def forecast_daily(history: pd.DataFrame, horizon: int, method: str, model: dict | None = None) -> pd.DataFrame:
    """(horizon × lanes) daily demand forecast after the last row of `history`."""
    values = history.to_numpy(dtype=float)
    last = pd.Timestamp(history.index[-1])
    if method == "model" and model is not None and len(values) >= demand_model.MIN_HISTORY:
        fc = demand_model.recursive_forecast(model, values, last, horizon)
    elif method == "seasonal_naive" and len(values) >= 7:
        fc = seasonal_naive(values, horizon)
    else:
        fc = moving_average(values, horizon, min(config.DEMAND_LOOKBACK_DAYS, len(values)))
    index = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame(np.asarray(fc), index=index, columns=history.columns)
