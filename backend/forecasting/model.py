"""
Lightweight predictive demand model: pooled ridge regression on SKU × warehouse
daily Units_Sold (REAL), implemented in NumPy so it adds no dependency.

Features for the demand on day t use ONLY values from days < t:

  lag_1, lag_2, lag_7      demand 1, 2 and 7 days earlier
  roll_mean_7, roll_mean_28 trailing means over the 7 / 28 days before t
  dow_0 .. dow_5           day-of-week indicators of t (Sunday = baseline)

Not used, deliberately:
  * month / annual seasonality — the data covers one year (2024-01-01..2024-12-30),
    so a yearly pattern cannot be learned from one cycle and validated on another.
  * Promotion_Flag — it is not known for the future days being forecast.
  * Demand_Forecast — a column of unknown origin that correlates 0.95 with same-day
    Units_Sold; it may be derived from the actuals, so it is treated as leakage.

Validation is strictly chronological: the last TEST_DAYS days are held out, the
scaler and coefficients are fitted on earlier days only, and two error views are
reported against the moving-average and seasonal-naive baselines:

  one_step  — next-day forecast for every lane-day in the test window
  horizon   — total demand over the next HORIZON_DAYS days from origins every
              ORIGIN_STEP days in the test window (recursive multi-step forecast),
              which is what replenishment actually consumes.

Under FORECAST_METHOD = "auto" the Perception Agent uses whichever of model,
moving average and seasonal naive had the lowest held-out horizon WAPE — the model
is only used if it actually won. After validation it is refitted on
all available days with the same settings; the stored metrics remain the held-out
ones. Training runs offline (`python -m backend.forecasting.train`) and the result
is a small JSON artifact, so nothing is retrained on a web request.
"""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from backend.forecasting.demand import error_metrics, moving_average, seasonal_naive

ARTIFACT_VERSION = 1
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "demand_model.json"

NUMERIC_FEATURES = ["lag_1", "lag_2", "lag_7", "roll_mean_7", "roll_mean_28"]
DOW_FEATURES = [f"dow_{i}" for i in range(6)]
FEATURES = NUMERIC_FEATURES + DOW_FEATURES
MIN_HISTORY = 28  # longest look-back any feature needs

RIDGE_ALPHA = 1.0
TEST_DAYS = 56
HORIZON_DAYS = 21
ORIGIN_STEP = 7
BASELINE_WINDOW = 28


# ---------------------------------------------------------------------------
# Features — only rows strictly before the target day are read
# ---------------------------------------------------------------------------

def _numeric_features(history: np.ndarray) -> np.ndarray:
    """(lanes × 5) features for the day right after the last row of `history`."""
    return np.column_stack([
        history[-1], history[-2], history[-7],
        history[-7:].mean(axis=0), history[-28:].mean(axis=0),
    ])


def _dow_features(day: pd.Timestamp, n: int) -> np.ndarray:
    out = np.zeros((n, len(DOW_FEATURES)))
    if day.weekday() < len(DOW_FEATURES):
        out[:, day.weekday()] = 1.0
    return out


def feature_rows(values: np.ndarray, dates: pd.DatetimeIndex, t: int) -> np.ndarray:
    """Features for target day index t, built from values[:t] only."""
    if t < MIN_HISTORY:
        raise ValueError(f"need {MIN_HISTORY} days of history, got {t}")
    return np.hstack([_numeric_features(values[:t]), _dow_features(dates[t], values.shape[1])])


def design_matrix(values: np.ndarray, dates: pd.DatetimeIndex, targets: range) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([feature_rows(values, dates, t) for t in targets])
    y = np.concatenate([values[t] for t in targets])
    return X, y


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> dict:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    Z = (X - mean) / scale
    intercept = float(y.mean())
    coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - intercept))
    return {"mean": mean, "scale": scale, "coef": coef, "intercept": intercept}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    z = (X - model["mean"]) / model["scale"]
    return np.clip(z @ model["coef"] + model["intercept"], 0.0, None)


def recursive_forecast(model: dict, history: np.ndarray, last_date: pd.Timestamp, horizon: int) -> np.ndarray:
    """(horizon × lanes) daily forecasts; each predicted day feeds the next."""
    ext = np.array(history[-MIN_HISTORY:], dtype=float)
    out = []
    for k in range(1, horizon + 1):
        day = pd.Timestamp(last_date) + timedelta(days=k)
        x = np.hstack([_numeric_features(ext), _dow_features(day, ext.shape[1])])
        p = predict(model, x)
        out.append(p)
        ext = np.vstack([ext, p])
    return np.array(out)


# ---------------------------------------------------------------------------
# Chronological evaluation
# ---------------------------------------------------------------------------

def evaluate(wide: pd.DataFrame, test_days: int = TEST_DAYS, horizon: int = HORIZON_DAYS,
             origin_step: int = ORIGIN_STEP, alpha: float = RIDGE_ALPHA) -> tuple[dict, dict]:
    values, dates = wide.to_numpy(dtype=float), pd.DatetimeIndex(wide.index)
    T = len(dates)
    split = T - test_days
    if split - MIN_HISTORY < 28 or test_days < horizon:
        raise ValueError("not enough history for a chronological train/test split")

    train_targets = range(MIN_HISTORY, split)
    model = fit_ridge(*design_matrix(values, dates, train_targets), alpha=alpha)

    test_targets = range(split, T)
    X_test, y_test = design_matrix(values, dates, test_targets)
    one_step = {
        "model": error_metrics(y_test, predict(model, X_test)),
        "moving_average": error_metrics(y_test, np.concatenate([values[t - BASELINE_WINDOW:t].mean(axis=0) for t in test_targets])),
        "seasonal_naive": error_metrics(y_test, np.concatenate([values[t - 7] for t in test_targets])),
    }

    origins = list(range(split, T - horizon + 1, origin_step))
    actual, fc = [], {"model": [], "moving_average": [], "seasonal_naive": []}
    for o in origins:
        hist = values[:o]
        actual.append(values[o:o + horizon].sum(axis=0))
        fc["model"].append(recursive_forecast(model, hist, dates[o - 1], horizon).sum(axis=0))
        fc["moving_average"].append(moving_average(hist, horizon, BASELINE_WINDOW).sum(axis=0))
        fc["seasonal_naive"].append(seasonal_naive(hist, horizon).sum(axis=0))
    horizon_metrics = {k: error_metrics(np.concatenate(actual), np.concatenate(v)) for k, v in fc.items()}

    metrics = {
        "train_period": [str(dates[MIN_HISTORY].date()), str(dates[split - 1].date())],
        "test_period": [str(dates[split].date()), str(dates[-1].date())],
        "lanes": int(values.shape[1]),
        "one_step": one_step,
        "horizon": {"days": horizon, "origins": [str(dates[o].date()) for o in origins], **horizon_metrics},
    }
    return metrics, model


def _fingerprint(wide: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(",".join(f"{a}|{b}" for a, b in wide.columns).encode())
    h.update(",".join(str(d.date()) for d in wide.index).encode())
    h.update(np.ascontiguousarray(wide.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()


def build_artifact(wide: pd.DataFrame, alpha: float = RIDGE_ALPHA) -> dict:
    metrics, _ = evaluate(wide, alpha=alpha)
    values, dates = wide.to_numpy(dtype=float), pd.DatetimeIndex(wide.index)
    final = fit_ridge(*design_matrix(values, dates, range(MIN_HISTORY, len(dates))), alpha=alpha)
    wapes = {m: metrics["horizon"][m]["wape"] for m in ("model", "moving_average", "seasonal_naive")}
    scored = {m: w for m, w in wapes.items() if w is not None}
    # Ties go to the simpler method (baselines listed first).
    best = min(["moving_average", "seasonal_naive", "model"], key=lambda m: scored.get(m, float("inf"))) if scored else "moving_average"
    best_baseline = min(w for m, w in scored.items() if m != "model") if len(scored) > 1 else None
    beats = wapes["model"] is not None and best_baseline is not None and wapes["model"] < best_baseline
    return {
        "version": ARTIFACT_VERSION,
        "kind": "pooled ridge regression (numpy)",
        "features": FEATURES,
        "alpha": alpha,
        "trained_through": str(dates[-1].date()),
        "data_fingerprint": _fingerprint(wide),
        "mean": final["mean"].round(10).tolist(),
        "scale": final["scale"].round(10).tolist(),
        "coef": final["coef"].round(10).tolist(),
        "intercept": round(final["intercept"], 10),
        "validation": metrics,
        "selection": {
            "criterion": f"lowest held-out {HORIZON_DAYS}-day horizon WAPE",
            "horizon_wape": wapes,
            "best_method": best,
            "model_beats_baseline": bool(beats),
        },
    }


def save_artifact(artifact: dict, path: Path = ARTIFACT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loading with validation — any problem means "use the baseline"
# ---------------------------------------------------------------------------

def _validate(artifact: dict) -> dict:
    if artifact.get("version") != ARTIFACT_VERSION:
        raise ValueError(f"artifact version {artifact.get('version')} != {ARTIFACT_VERSION}")
    if artifact.get("features") != FEATURES:
        raise ValueError("artifact features do not match the code")
    arrays = {k: np.asarray(artifact[k], dtype=float) for k in ("mean", "scale", "coef")}
    if any(a.shape != (len(FEATURES),) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("artifact coefficients are malformed or non-finite")
    if (arrays["scale"] <= 0).any() or not np.isfinite(float(artifact["intercept"])):
        raise ValueError("artifact scaler is invalid")
    return {**arrays, "intercept": float(artifact["intercept"])}


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime: float) -> tuple[dict | None, dict | None, str | None]:
    try:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        return artifact, _validate(artifact), None
    except Exception as exc:  # noqa: BLE001 — any failure falls back to the baseline
        return None, None, f"model artifact invalid: {exc}"


def load_artifact(path: Path = ARTIFACT_PATH) -> tuple[dict | None, dict | None, str | None]:
    """(artifact, model, error). model is None whenever the artifact is unusable."""
    if not Path(path).exists():
        return None, None, "model artifact missing"
    return _load_cached(str(path), Path(path).stat().st_mtime)
