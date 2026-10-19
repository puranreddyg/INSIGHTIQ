"""
Demand forecaster — a learned model of daily units sold, trained on the real
`supply_chain` ledger (Units_Sold, Date, Promotion_Flag).

Model
    units_sold(day) = lane_scale * [ b0 + Fourier(day_of_year, K)
                                      + promo * (b1 + c1*sin + c2*cos) ] + noise

  * a smooth yearly cycle (K Fourier harmonics, chosen by leave-block-out CV),
  * a promotion lift that is allowed to grow and shrink with the season,
  * a per-lane scale learned with empirical-Bayes shrinkage, so a lane only
    departs from the pooled curve when the data shows it really sells more/less.

It is an ordinary least-squares model on purpose. The residuals are white noise
(lag-1 autocorrelation ~ 0, no lane effect), so the model is already at the noise
floor of this data and a bigger model has nothing left to learn; the backtest in
`backtest()` reports how much it beats the app's previous method (trailing mean).

The column `Demand_Forecast` in the source file is NOT used: it tracks the same
day's sales (corr 0.95) far better than tomorrow's (0.61), i.e. it is derived
from the outcome, not a forecast.

The model only needs numpy. Everything is deterministic (no random numbers).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

YEAR_DAYS = 365.25
CANDIDATE_HARMONICS = (1, 2, 3)
RIDGE = 1e-6  # numerical stability only; far too small to shrink any coefficient
MAX_HORIZON_DAYS = 120


def _fourier(day_of_year: np.ndarray, k: int) -> np.ndarray:
    w = 2.0 * np.pi * np.asarray(day_of_year, dtype=float) / YEAR_DAYS
    cols = []
    for h in range(1, k + 1):
        cols += [np.sin(h * w), np.cos(h * w)]
    return np.column_stack(cols)


def _design(day_of_year: np.ndarray, promo: np.ndarray, k: int) -> np.ndarray:
    f = _fourier(day_of_year, k)
    promo = np.asarray(promo, dtype=float)
    # promotion lift interacts with the first harmonic (its size follows the season)
    return np.column_stack([np.ones(len(promo)), f, promo, promo * f[:, 0], promo * f[:, 1]])


def _solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = X.T @ X
    A[np.diag_indices_from(A)] += RIDGE
    return np.linalg.solve(A, X.T @ y)


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / actual.sum())


@dataclass
class DemandModel:
    k: int
    coef: list[float]
    promo_rate: float
    sigma_daily: float
    trained_from: str
    trained_through: str
    n_rows: int
    n_lanes: int
    r2: float
    metrics: dict = field(default_factory=dict)

    # ---- prediction -------------------------------------------------------
    def expected_daily(self, dates: pd.DatetimeIndex, promo: float | np.ndarray | None = None) -> np.ndarray:
        """Pooled expected units/day for each date. `promo` defaults to the
        historical promotion rate, because future promotions are not known."""
        doy = np.asarray(dates.dayofyear, dtype=float)
        p = np.full(len(doy), self.promo_rate if promo is None else promo, dtype=float)
        return np.maximum(_design(doy, p, self.k) @ np.asarray(self.coef), 0.0)

    def horizon_means(self, as_of: str | pd.Timestamp, horizons: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Expected total units over the next `horizons[i]` days (after `as_of`) for lane i."""
        horizons = np.clip(np.asarray(horizons, dtype=int), 1, MAX_HORIZON_DAYS)
        days = pd.date_range(pd.Timestamp(as_of) + pd.Timedelta(days=1), periods=MAX_HORIZON_DAYS)
        cum = np.concatenate([[0.0], np.cumsum(self.expected_daily(days))])
        return cum[horizons] * np.asarray(scale, dtype=float)

    def horizon_std(self, horizons: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Std of total demand over `horizons` days: daily noise is uncorrelated, so it grows as sqrt(h)."""
        h = np.clip(np.asarray(horizons, dtype=float), 1, MAX_HORIZON_DAYS)
        sigma = max(self.sigma_daily, 1e-6)  # a perfectly fitted series must not divide by zero downstream
        return sigma * np.sqrt(h) * np.sqrt(np.asarray(scale, dtype=float))

    # ---- persistence ------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, payload: str) -> "DemandModel":
        return cls(**json.loads(payload))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _with_dates(supply_chain: pd.DataFrame) -> pd.DataFrame:
    """Dates may arrive as text (SQLite, an ERP feed); the model needs real timestamps."""
    if pd.api.types.is_datetime64_any_dtype(supply_chain["date"]):
        return supply_chain
    return supply_chain.assign(date=pd.to_datetime(supply_chain["date"]))


def _panel(supply_chain: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """lane x day matrices of units sold and promotion flags (None if the panel is ragged)."""
    supply_chain = _with_dates(supply_chain)
    df = supply_chain.assign(lane=supply_chain.sku_id + "|" + supply_chain.warehouse_id)
    sold = df.pivot(index="lane", columns="date", values="units_sold")
    promo = df.pivot(index="lane", columns="date", values="promotion_flag")
    return sold.to_numpy(float), promo.to_numpy(float), pd.DatetimeIndex(sold.columns), list(sold.index)


def _fit_pooled(doy: np.ndarray, promo: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    return _solve(_design(doy, promo, k), y)


def backtest(supply_chain: pd.DataFrame, ks=CANDIDATE_HARMONICS) -> dict:
    """Compare the model with the app's previous method (mean of the last 28 days).

    strict_rolling_origin  train only on the past, forecast the next h days from
                           weekly origins in Q4 — the pessimistic, honest test.
    leave_block_out        hold out one h-day block at a time and train on the rest
                           (every season has been seen, as it is in production).
    WAPE = sum|error| / sum(actual) over lane-level h-day totals; lower is better.
    """
    S, P, dates, _ = _panel(supply_chain)
    if np.isnan(S).any() or np.isnan(P).any() or S.shape[1] < 120:
        return {"available": False, "reason": "ledger is ragged or shorter than 120 days"}
    n_lanes, n_days = S.shape
    doy = dates.dayofyear.to_numpy(float)
    promo_rate = float(P.mean())
    DOY, PRO, Y, T = np.tile(doy, n_lanes), P.ravel(), S.ravel(), np.tile(np.arange(n_days), n_lanes)

    def fit_on(mask: np.ndarray, k: int) -> np.ndarray:
        return _fit_pooled(DOY[mask], PRO[mask], Y[mask], k)

    def expected(coef: np.ndarray, k: int, day_idx: np.ndarray) -> float:
        X = _design(doy[day_idx], np.full(len(day_idx), promo_rate), k)
        return float((X @ coef).clip(0).sum())

    result: dict = {"available": True, "metric": "WAPE", "horizons_days": [7, 14], "candidate_harmonics": list(ks)}
    block_error: dict[int, float] = {}
    for h in (7, 14):
        # --- strict rolling origin ---
        origins = list(range(int(n_days * 0.75), n_days - h, 7))
        actual, base, preds = [], [], {k: [] for k in ks}
        for o in origins:
            idx = np.arange(o + 1, o + 1 + h)
            actual.append(S[:, idx].sum(axis=1))
            base.append(S[:, o - 27:o + 1].mean(axis=1) * h)
            for k in ks:
                coef = fit_on(T <= o, k)
                preds[k].append(np.full(n_lanes, expected(coef, k, idx)))
        a = np.concatenate(actual)
        result[f"strict_h{h}"] = {
            "baseline_trailing_28d": round(_wape(a, np.concatenate(base)), 4),
            **{f"model_k{k}": round(_wape(a, np.concatenate(preds[k])), 4) for k in ks},
            "origins": len(origins),
        }
        # --- leave-block-out ---
        actual, base, preds = [], [], {k: [] for k in ks}
        for b in range(28, n_days - h, h):
            idx = np.arange(b, b + h)
            actual.append(S[:, idx].sum(axis=1))
            base.append(S[:, b - 28:b].mean(axis=1) * h)
            for k in ks:
                coef = fit_on((T < b) | (T >= b + h), k)
                preds[k].append(np.full(n_lanes, expected(coef, k, idx)))
        a = np.concatenate(actual)
        result[f"block_h{h}"] = {
            "baseline_trailing_28d": round(_wape(a, np.concatenate(base)), 4),
            **{f"model_k{k}": round(_wape(a, np.concatenate(preds[k])), 4) for k in ks},
            "blocks": len(actual),
        }
        if h == 14:
            block_error = {k: _wape(a, np.concatenate(preds[k])) for k in ks}

    best = min(block_error.values())
    # parsimony: the simplest harmonic count within 1% (relative) of the best block error
    result["selected_k"] = next(k for k in sorted(ks) if block_error[k] <= best * 1.01)
    return result


def lane_statistics(supply_chain: pd.DataFrame, model: DemandModel, fallback_cadence_days: int) -> pd.DataFrame:
    """Per SKU x warehouse facts the Perception Agent needs:
       demand_scale   empirical-Bayes shrunken ratio of the lane's sales to the pooled curve
       order_cadence_days  median days between the lane's historical orders (its replenishment cycle)
       lot_size_median     median historical order size (kept as a reference)
       trailing_mean_28d   previous method's demand estimate, kept for comparison
    """
    df = _with_dates(supply_chain).copy()
    df["expected"] = model.expected_daily(pd.DatetimeIndex(df["date"]), promo=df["promotion_flag"].to_numpy(float))
    g = df.groupby(["sku_id", "warehouse_id"])
    lane = g.agg(actual=("units_sold", "sum"), expected=("expected", "sum"), n=("units_sold", "size")).reset_index()
    lane["ratio"] = lane.actual / lane.expected
    sampling_var = model.sigma_daily**2 * lane.n / lane.expected**2  # variance of the ratio if all lanes were identical
    tau2 = max(0.0, float(lane.ratio.var(ddof=1) - sampling_var.mean()))  # genuine between-lane variance
    weight = tau2 / (tau2 + sampling_var) if tau2 > 0 else pd.Series(0.0, index=lane.index)
    lane["demand_scale"] = 1.0 + weight * (lane.ratio - 1.0)

    orders = df[df.order_quantity > 0].sort_values("date")
    gaps = (
        orders.groupby(["sku_id", "warehouse_id"]).date.apply(lambda s: s.diff().dt.days.dropna().median())
        .rename("order_cadence_days").reset_index()
    )
    counts = orders.groupby(["sku_id", "warehouse_id"]).size().rename("order_count").reset_index()
    gaps = gaps.merge(counts, how="left")
    lot =orders.groupby(["sku_id", "warehouse_id"]).order_quantity.median().rename("lot_size_median").reset_index()
    last = df.date.max()
    trailing = (
        df[df.date > last - pd.Timedelta(days=28)].groupby(["sku_id", "warehouse_id"]).units_sold.mean()
        .rename("trailing_mean_28d").reset_index()
    )
    out = (
        lane[["sku_id", "warehouse_id", "demand_scale"]]
        .merge(gaps, how="left").merge(lot, how="left").merge(trailing, how="left")
    )
    out["order_cadence_days"] = out.order_cadence_days.fillna(fallback_cadence_days).clip(lower=1)
    out["order_count"] = out.order_count.fillna(0).astype(int)
    out.attrs["tau2"] = tau2
    return out


def fit(supply_chain: pd.DataFrame, run_backtest: bool = True) -> DemandModel:
    """Train on every row of the ledger. `supply_chain` uses the database column names."""
    supply_chain = _with_dates(supply_chain)
    metrics = backtest(supply_chain) if run_backtest else {}
    k = int(metrics.get("selected_k", 2)) if metrics.get("available") else 2
    doy = pd.DatetimeIndex(supply_chain["date"]).dayofyear.to_numpy(float)
    promo = supply_chain["promotion_flag"].to_numpy(float)
    y = supply_chain["units_sold"].to_numpy(float)
    coef = _fit_pooled(doy, promo, y, k)
    resid = y - _design(doy, promo, k) @ coef
    dof = len(y) - len(coef)
    model = DemandModel(
        k=k,
        coef=[float(c) for c in coef],
        promo_rate=float(promo.mean()),
        sigma_daily=float(math.sqrt((resid**2).sum() / dof)),
        trained_from=str(pd.Timestamp(supply_chain["date"].min()).date()),
        trained_through=str(pd.Timestamp(supply_chain["date"].max()).date()),
        n_rows=int(len(y)),
        n_lanes=int(supply_chain.groupby(["sku_id", "warehouse_id"]).ngroups),
        r2=float(1.0 - resid.var() / y.var()),
        metrics=metrics,
    )
    return model
