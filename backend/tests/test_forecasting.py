"""
Forecasting: correct daily aggregation, zero-safe metrics, no future information in
training, artifact fallback, and the forecast-based replenishment quantity.
"""
from __future__ import annotations

import json
import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from backend import config
from backend.agents import perception_agent
from backend.forecasting import demand, model, service

# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def test_real_data_has_one_row_per_lane_day():
    raw = pd.read_csv(config.CSV_PATHS["supply_chain"], usecols=["Date", "SKU_ID", "Warehouse_ID", "Supplier_ID"])
    assert not raw.duplicated(["Date", "SKU_ID", "Warehouse_ID"]).any()
    assert raw.groupby(["SKU_ID", "Warehouse_ID"]).Supplier_ID.nunique().max() == 1


def _rows(*records):
    return pd.DataFrame(records, columns=["date", "sku_id", "warehouse_id", "supplier_id", "units_sold"])


def test_exact_duplicate_rows_are_not_double_counted():
    raw = _rows(("2024-01-01", "S", "W", "P1", 5), ("2024-01-01", "S", "W", "P1", 5), ("2024-01-02", "S", "W", "P1", 7))
    wide = demand.aggregate_daily(raw)
    assert wide[("S", "W")].tolist() == [5, 7]


def test_distinct_supplier_rows_on_one_lane_day_are_summed_once():
    raw = _rows(("2024-01-01", "S", "W", "P1", 5), ("2024-01-01", "S", "W", "P2", 3))
    assert demand.aggregate_daily(raw)[("S", "W")].tolist() == [8]


def test_missing_days_are_filled_with_zero_and_lanes_stay_separate():
    raw = _rows(("2024-01-01", "S", "W1", "P", 4), ("2024-01-03", "S", "W1", "P", 6), ("2024-01-01", "S", "W2", "P", 1))
    wide = demand.aggregate_daily(raw)
    assert list(wide.index.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert wide[("S", "W1")].tolist() == [4, 0, 6]
    assert wide[("S", "W2")].tolist() == [1, 0, 0]


def test_lane_state_is_not_summed_across_duplicate_supplier_rows():
    """Inventory / reorder point are read once per lane even if a feed repeats the lane-day."""
    c = sqlite3.connect(":memory:")
    days = pd.date_range("2025-01-01", periods=40, freq="D").strftime("%Y-%m-%d %H:%M:%S")
    rows = [{"date": d, "sku_id": "S", "warehouse_id": "W", "supplier_id": sup, "units_sold": 5, "inventory_level": 80,
             "reorder_point": 100, "order_quantity": 0, "supplier_lead_time_days": 4} for d in days for sup in ("P1", "P2")]
    pd.DataFrame(rows).to_sql("supply_chain", c, index=False)
    req = perception_agent.observe_sku(c, "S")
    assert req["lane_count"] == 1 and req["Inventory"] == 80
    assert req["lanes"][0]["DailyDemand"] == 10  # two sale records of 5 per day


# ---------------------------------------------------------------------------
# Baselines and metrics
# ---------------------------------------------------------------------------

def test_baselines():
    hist = np.arange(1, 29, dtype=float).reshape(-1, 1)
    assert demand.moving_average(hist, 3, 4).ravel().tolist() == [26.5] * 3
    assert demand.seasonal_naive(hist, 9).ravel().tolist() == [22, 23, 24, 25, 26, 27, 28, 22, 23]


def test_metrics_are_safe_on_zero_demand():
    m = demand.error_metrics(np.zeros(5), np.ones(5))
    assert m["mae"] == 1 and m["wape"] is None and m["bias"] is None
    assert demand.error_metrics([10, 0], [8, 1])["wape"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# No future information
# ---------------------------------------------------------------------------

def _synthetic(T=140, N=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="D")
    cols = pd.MultiIndex.from_tuples([("S", f"W{i}") for i in range(N)], names=demand.LANE_KEYS)
    return pd.DataFrame(rng.poisson(20, size=(T, N)).astype(float), index=idx, columns=cols)


def test_features_for_day_t_ignore_day_t_and_later():
    wide = _synthetic()
    values, dates = wide.to_numpy(), wide.index
    before = model.feature_rows(values, dates, 60)
    tampered = values.copy()
    tampered[60:] = 9999
    assert np.array_equal(model.feature_rows(tampered, dates, 60), before)


def test_training_uses_no_test_period_observations():
    """Changing every value in the held-out window must not change the fitted coefficients."""
    wide = _synthetic()
    _, fitted = model.evaluate(wide, test_days=28, horizon=14, origin_step=7)
    tampered = wide.copy()
    tampered.iloc[-28:] = 500.0
    _, fitted_tampered = model.evaluate(tampered, test_days=28, horizon=14, origin_step=7)
    for key in ("coef", "mean", "scale"):
        assert np.allclose(fitted[key], fitted_tampered[key])
    assert fitted["intercept"] == pytest.approx(fitted_tampered["intercept"])


def test_split_is_chronological():
    metrics, _ = model.evaluate(_synthetic(), test_days=28, horizon=14, origin_step=7)
    assert metrics["train_period"][1] < metrics["test_period"][0]
    assert all(o >= metrics["test_period"][0] for o in metrics["horizon"]["origins"])


def test_recursive_forecast_shape_and_non_negative():
    wide = _synthetic()
    _, fitted = model.evaluate(wide, test_days=28, horizon=14, origin_step=7)
    fc = model.recursive_forecast(fitted, wide.to_numpy(), wide.index[-1], 10)
    assert fc.shape == (10, 3) and (fc >= 0).all()


# ---------------------------------------------------------------------------
# Committed artifact and fallback
# ---------------------------------------------------------------------------

def test_committed_artifact_is_valid_and_honest():
    artifact, fitted, error = model.load_artifact()
    assert error is None and fitted is not None
    sel = artifact["selection"]
    wapes = sel["horizon_wape"]
    assert sel["best_method"] == min(wapes, key=wapes.get)
    assert sel["model_beats_baseline"] == (wapes["model"] < min(wapes["moving_average"], wapes["seasonal_naive"]))
    v = artifact["validation"]
    assert v["train_period"][1] < v["test_period"][0]


def test_auto_uses_the_validated_best_method():
    resolved = service.resolve_method("auto")
    artifact, _, _ = model.load_artifact()
    assert resolved["method"] == artifact["selection"]["best_method"]


def test_missing_artifact_falls_back_to_moving_average(tmp_path):
    resolved = service.resolve_method("model", artifact_path=tmp_path / "nope.json")
    assert resolved["method"] == "moving_average" and "missing" in resolved["reason"]


@pytest.mark.parametrize("corrupt", [
    "not json",
    json.dumps({"version": 999}),
    json.dumps({"version": model.ARTIFACT_VERSION, "features": model.FEATURES, "mean": [0] * 11,
                "scale": [1] * 11, "coef": [float("nan")] * 11, "intercept": 0}),
])
def test_invalid_artifact_falls_back_to_moving_average(tmp_path, corrupt):
    path = tmp_path / "bad.json"
    path.write_text(corrupt, encoding="utf-8")
    resolved = service.resolve_method("auto", artifact_path=path)
    assert resolved["method"] == "moving_average" and "invalid" in resolved["reason"]


def test_training_is_reproducible():
    wide = _synthetic(T=150)
    assert model.build_artifact(wide) == model.build_artifact(wide)


# ---------------------------------------------------------------------------
# Forecast-based replenishment quantity
# ---------------------------------------------------------------------------

def _lane_conn(inventory, daily=10, lead=4, order_every=10, days=60, reorder_point=100, noise=0):
    c = sqlite3.connect(":memory:")
    rng = np.random.default_rng(1)
    dates = pd.date_range("2025-01-01", periods=days, freq="D").strftime("%Y-%m-%d %H:%M:%S")
    rows = [{"date": d, "sku_id": "S", "warehouse_id": "W", "supplier_id": "P",
             "units_sold": daily + (int(rng.integers(-noise, noise + 1)) if noise else 0),
             "inventory_level": inventory, "reorder_point": reorder_point,
             "order_quantity": 120 if i % order_every == 0 else 0, "supplier_lead_time_days": lead}
            for i, d in enumerate(dates)]
    pd.DataFrame(rows).to_sql("supply_chain", c, index=False)
    return c


def test_quantity_is_target_minus_inventory_rounded_up(monkeypatch):
    monkeypatch.setattr(config, "FORECAST_METHOD", "moving_average")
    calc = perception_agent.observe_sku(_lane_conn(inventory=93), "S")["quantity_calculation"]
    assert calc["forecast_demand"] == 140 and calc["safety_stock"] == 0
    assert calc["quantity_before_rounding"] == 47 and calc["final_quantity"] == 50
    for key in ("forecast_demand", "safety_stock", "target_stock", "current_inventory", "quantity_before_rounding", "final_quantity", "explanation"):
        assert key in calc


def test_quantity_is_never_negative(monkeypatch):
    """Urgent (inventory < reorder point) but stock already covers L + R demand → 0."""
    monkeypatch.setattr(config, "FORECAST_METHOD", "moving_average")
    req = perception_agent.observe_sku(_lane_conn(inventory=900, reorder_point=1000), "S")
    assert req["Urgent"] is True
    assert req["quantity_calculation"]["quantity_before_rounding"] == 0 and req["RequiredQuantity"] == 0


def test_service_level_safety_stock(monkeypatch):
    monkeypatch.setattr(config, "FORECAST_METHOD", "moving_average")
    req = perception_agent.observe_sku(_lane_conn(inventory=50, noise=3), "S")
    lane = req["focus_lanes"][0]
    z = 1.6448536269514722
    # DemandStd is displayed at 2 dp, so allow for that rounding times √H
    assert lane["SafetyStock"] == pytest.approx(z * lane["DemandStd"] * math.sqrt(lane["HorizonDays"]), abs=0.05)
    assert lane["TargetStock"] == pytest.approx(lane["ForecastDemand"] + lane["SafetyStock"], abs=0.02)


def test_days_of_cover_safety_stock(monkeypatch):
    monkeypatch.setattr(config, "FORECAST_METHOD", "moving_average")
    monkeypatch.setattr(config, "SAFETY_STOCK_METHOD", "days_of_cover")
    calc = perception_agent.observe_sku(_lane_conn(inventory=93), "S")["quantity_calculation"]
    assert calc["safety_stock"] == 30  # 3 days × 10/day


def test_fixed_review_period_zero_is_a_pure_lead_time_horizon(monkeypatch):
    monkeypatch.setattr(config, "FORECAST_METHOD", "moving_average")
    monkeypatch.setattr(config, "FORECAST_REVIEW_PERIOD_SOURCE", "fixed")
    monkeypatch.setattr(config, "FORECAST_REVIEW_PERIOD_DAYS", 0)
    lane = perception_agent.observe_sku(_lane_conn(inventory=30), "S")["focus_lanes"][0]
    assert lane["HorizonDays"] == 4 and lane["ForecastDemand"] == 40


def test_median_fallback_is_still_available(monkeypatch):
    monkeypatch.setattr(config, "REQUIRED_QUANTITY_METHOD", "median")
    req = perception_agent.observe_sku(_lane_conn(inventory=93), "S")
    assert req["quantity_calculation"]["method"] == "median" and req["RequiredQuantity"] == 120


def test_urgency_rule_is_unchanged():
    assert perception_agent.observe_sku(_lane_conn(inventory=99, reorder_point=100), "S")["Urgent"] is True
    assert perception_agent.observe_sku(_lane_conn(inventory=100, reorder_point=100), "S")["Urgent"] is False
