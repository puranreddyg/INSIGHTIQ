"""Validation of the simulated supplier-SKU profile against the raw source data."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from backend import config
from backend.simulation import supplier_sku_profile as sim


@pytest.fixture(scope="module")
def raw():
    return sim.load_raw()


@pytest.fixture(scope="module")
def generated(raw):
    return sim.generate(*raw)


@pytest.fixture(scope="module")
def committed():
    return pd.read_csv(config.CSV_PATHS["supplier_sku_profile"])


def test_regeneration_with_the_same_seed_is_identical(raw, generated):
    again, _ = sim.generate(*raw)
    pd.testing.assert_frame_equal(generated[0], again)


def test_committed_csv_equals_regeneration(generated, committed):
    pd.testing.assert_frame_equal(committed, generated[0].reset_index(drop=True))


def test_a_different_seed_gives_different_values(raw, generated):
    other, _ = sim.generate(*raw, seed=config.SIMULATION_SEED + 1)
    assert not other.equals(generated[0])


def test_every_pair_exists_in_source_and_none_are_missing(raw, committed):
    source = set(map(tuple, raw[0][["SKU_ID", "Supplier_ID"]].drop_duplicates().to_numpy()))
    simulated = set(map(tuple, committed[["sku_id", "supplier_id"]].to_numpy()))
    assert simulated - source == set(), "invented relationships"
    assert source - simulated == set(), "missing relationships"
    assert len(committed) == len(source) == 211


def test_no_duplicate_pairs_and_no_nulls(committed):
    assert not committed.duplicated(["sku_id", "supplier_id"]).any()
    assert not committed.isna().any().any()
    assert list(committed.columns) == ["sku_id", "supplier_id", "available_qty", "defect_rate", "fulfillment_rate"]


def test_available_qty_is_a_real_historical_order_size_of_that_pair(raw, committed):
    history = sim.order_quantity_history(raw[0])
    for r in committed.itertuples():
        assert r.available_qty in set(history[(r.sku_id, r.supplier_id)])


def test_available_qty_range_matches_history(committed):
    assert committed.available_qty.between(200, 499).all()


def test_available_qty_supports_an_eligibility_mix(committed):
    """At the median historical order size some pairs qualify and some do not."""
    share = (committed.available_qty >= 351).mean()
    assert 0.2 < share < 0.8


def test_every_pair_has_enough_order_history(raw):
    sizes = sim.order_quantity_history(raw[0]).map(len)
    assert sizes.min() >= config.MIN_ORDER_OBSERVATIONS_PER_PAIR


def test_quality_fields_are_supplier_level(committed):
    per_supplier = committed.groupby("supplier_id")[["defect_rate", "fulfillment_rate"]].nunique()
    assert (per_supplier == 1).all().all()


def test_defect_rate_is_within_zero_and_the_real_maximum(raw, committed):
    _, real_supplier_rates = sim.kpi_defect_rates(raw[1])
    assert committed.defect_rate.between(0, real_supplier_rates.max()).all()


def test_fulfillment_rate_is_a_valid_proportion(committed):
    assert committed.fulfillment_rate.between(0, 1).all()


def test_simulation_does_not_give_one_supplier_the_best_value_on_every_field(committed):
    q = committed.groupby("supplier_id")[["defect_rate", "fulfillment_rate"]].first()
    assert q.defect_rate.idxmin() != q.fulfillment_rate.idxmax() or q.defect_rate.nunique() == 1


def test_metadata_records_the_seed_and_cap():
    meta = json.loads(sim.OUT_META.read_text(encoding="utf-8"))
    assert meta["seed"] == config.SIMULATION_SEED == 20260926
    assert meta["defect_rate"]["cap_applied"] <= meta["defect_rate"]["cap_exact_real_max"]
    assert meta["rows"] == 211
