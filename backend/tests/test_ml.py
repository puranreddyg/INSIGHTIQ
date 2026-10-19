"""
Tests for the predictive layer (backend/ml): it must learn from real data, beat the
previous method out of sample, be deterministic, and never invent supplier facts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend import database
from backend.ml import demand_forecaster, supplier_estimator


@pytest.fixture(scope="module")
def supply_chain():
    return database._load_supply_chain()


@pytest.fixture(scope="module")
def kpi():
    return database._load_procurement_kpi()


@pytest.fixture(scope="module")
def model(supply_chain):
    return demand_forecaster.fit(supply_chain)


# ---------------------------------------------------------------------------
# Demand forecaster
# ---------------------------------------------------------------------------

def test_forecaster_beats_the_previous_trailing_mean_out_of_sample(model):
    bt = model.metrics
    assert bt["available"]
    k = bt["selected_k"]
    for test in ("strict_h7", "strict_h14", "block_h7", "block_h14"):
        assert bt[test][f"model_k{k}"] < bt[test]["baseline_trailing_28d"], test
    # the honest (train-on-past-only) 14-day test should cut the error by at least a third
    assert bt["strict_h14"][f"model_k{k}"] < 0.67 * bt["strict_h14"]["baseline_trailing_28d"]


def test_forecaster_learns_the_real_seasonal_and_promotion_pattern(model):
    spring = model.expected_daily(pd.DatetimeIndex(["2025-03-20"]), promo=0.0)[0]
    autumn = model.expected_daily(pd.DatetimeIndex(["2025-09-20"]), promo=0.0)[0]
    promo = model.expected_daily(pd.DatetimeIndex(["2025-03-20"]), promo=1.0)[0]
    assert spring > 2 * autumn          # real data: ~30 units/day in March vs ~10 in September
    assert promo > spring               # promotions lift sales
    assert 0.5 < model.r2 < 0.9


def test_forecaster_does_not_use_the_leaky_demand_forecast_column(supply_chain):
    shuffled = supply_chain.assign(demand_forecast=supply_chain.demand_forecast.sample(frac=1, random_state=0).to_numpy())
    a = demand_forecaster.fit(supply_chain, run_backtest=False)
    b = demand_forecaster.fit(shuffled, run_backtest=False)
    assert a.coef == b.coef


def test_forecaster_is_deterministic_and_round_trips(supply_chain, model):
    again = demand_forecaster.fit(supply_chain)
    assert again.to_json() == model.to_json()
    assert demand_forecaster.DemandModel.from_json(model.to_json()).to_json() == model.to_json()


def test_horizon_uncertainty_grows_with_the_horizon(model):
    sd = model.horizon_std(np.array([1, 4, 16]), np.ones(3))
    assert sd[1] == pytest.approx(2 * sd[0]) and sd[2] == pytest.approx(4 * sd[0])


def test_lane_scales_are_shrunk_towards_the_pooled_curve(supply_chain, model):
    stats = demand_forecaster.lane_statistics(supply_chain, model, 14)
    assert len(stats) == 250
    assert stats.demand_scale.between(0.9, 1.1).all()
    assert (stats.order_cadence_days >= 1).all()


# ---------------------------------------------------------------------------
# Supplier estimator
# ---------------------------------------------------------------------------

def test_profile_has_no_random_values_and_only_real_pairs(supply_chain, kpi):
    a, meta_a = supplier_estimator.build_profile(supply_chain, kpi)
    b, meta_b = supplier_estimator.build_profile(supply_chain, kpi)
    pd.testing.assert_frame_equal(a, b)
    real = set(map(tuple, supply_chain[["sku_id", "supplier_id"]].drop_duplicates().to_numpy()))
    assert set(map(tuple, a[["sku_id", "supplier_id"]].to_numpy())) == real
    database._validate_supplier_profile(a, supply_chain)


def test_capacity_is_the_largest_order_the_supplier_actually_delivered(supply_chain, kpi):
    profile, _ = supplier_estimator.build_profile(supply_chain, kpi)
    orders = supply_chain[supply_chain.order_quantity > 0].groupby(["sku_id", "supplier_id"]).order_quantity
    largest, count = orders.max(), orders.size()
    for r in profile.itertuples():
        if r.capacity_orders >= 10:
            assert r.available_qty == largest[(r.sku_id, r.supplier_id)]
            assert r.capacity_orders == count[(r.sku_id, r.supplier_id)]
        else:
            assert pd.isna(r.available_qty)


def test_without_supplier_evidence_quality_sits_on_the_real_kpi_prior(supply_chain, kpi, tmp_path):
    none = tmp_path / "absent"
    profile, meta = supplier_estimator.build_profile(supply_chain, kpi, none / "links.json", none / "outcomes.csv")
    priors = meta["priors"]
    real_rates = (kpi.dropna(subset=["defective_units"]).defective_units / kpi.dropna(subset=["defective_units"]).quantity)
    assert priors["defect_prior_mean"] == pytest.approx(real_rates.clip(0, 1).mean())
    assert (profile.defect_rate == priors["defect_prior_mean"]).all()
    assert (profile.fulfillment_rate == priors["fulfillment_prior_mean"]).all()
    assert (profile.evidence_pos == 0).all()


def test_real_supplier_evidence_moves_the_estimate(supply_chain, kpi, tmp_path):
    outcomes = tmp_path / "supplier_outcomes.csv"
    pd.DataFrame({
        "supplier_id": ["SUP_3"] * 30,
        "quantity": [100] * 30,
        "defective_units": [20] * 30,       # 20% defects, far above the prior
        "order_status": ["Partially Delivered"] * 30,
    }).to_csv(outcomes, index=False)
    profile, _ = supplier_estimator.build_profile(supply_chain, kpi, tmp_path / "none.json", outcomes)
    sup3 = profile[profile.supplier_id == "SUP_3"].iloc[0]
    other = profile[profile.supplier_id == "SUP_4"].iloc[0]
    assert sup3.defect_rate > other.defect_rate and sup3.fulfillment_rate < other.fulfillment_rate
    assert sup3.evidence_pos == 30 and other.evidence_pos == 0
    assert 0.07 < sup3.defect_rate < 0.20   # credibility-weighted between prior and own rate


def test_vendor_link_must_name_real_entities(supply_chain, kpi, tmp_path):
    links = tmp_path / "vendor_links.json"
    links.write_text(json.dumps({"SUP_99": "Alpha_Inc"}))
    with pytest.raises(ValueError, match="not a supplier"):
        supplier_estimator.build_profile(supply_chain, kpi, links, tmp_path / "none.csv")
    links.write_text(json.dumps({"SUP_1": "Nobody_Ltd"}))
    with pytest.raises(ValueError, match="not a vendor"):
        supplier_estimator.build_profile(supply_chain, kpi, links, tmp_path / "none.csv")
