"""Forecast Evaluation page: honest out-of-sample metrics that match the plotted series."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import database
from backend.ml import forecast_evaluation as fe


@pytest.fixture(scope="module")
def sc():
    return database._load_supply_chain()


@pytest.fixture(scope="module")
def rolling(sc):
    return fe.evaluate(sc, "rolling")


def _recompute(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    nz = a != 0
    return {"mae": np.abs(p - a).mean(), "rmse": np.sqrt(((p - a) ** 2).mean()),
            "mape_pct": np.abs((p - a)[nz] / a[nz]).mean() * 100}


def test_periods_are_strictly_separated(rolling):
    P = rolling["periods"]
    assert P["train"] == ["2024-01-01", "2024-09-30"] and P["test"] == ["2024-10-01", "2024-12-30"]
    assert P["test_days"] == 91 and P["lanes"] == 250
    s = rolling["series"]
    assert s["dates"][s["test_start_index"]] == "2024-10-01"
    for m in rolling["methods"]:
        assert len(s[m]) == 91


def test_table_metrics_match_the_plotted_series(rolling):
    s, start = rolling["series"], rolling["series"]["test_start_index"]
    actual = s["actual"][start:]
    for m in rolling["methods"]:
        expect = _recompute(actual, s[m])
        got = rolling["metrics"]["total_daily"][m]
        for k in ("mae", "rmse", "mape_pct"):
            assert got[k] == pytest.approx(expect[k], rel=1e-3, abs=0.02), (m, k)


def test_no_future_data_leaks_into_the_forecasts(sc):
    """Changing Q4 sales must not change the forecast made on 30 Sep."""
    a = fe._evaluate(sc, "fixed")[1]
    tampered = sc.copy()
    q4 = pd.to_datetime(tampered["date"]) > pd.Timestamp(fe.CUTOFF)
    tampered.loc[q4, "units_sold"] = tampered.loc[q4, "units_sold"] * 3 + 7
    b = fe._evaluate(tampered, "fixed")[1]
    for m in fe.METHODS:
        np.testing.assert_allclose(a[m], b[m])


def test_baseline_is_the_last_28_days_before_the_origin(sc):
    preds = fe._evaluate(sc, "fixed")[1]
    d = sc.assign(date=pd.to_datetime(sc["date"]))
    lane = d[(d.sku_id == "SKU_1") & (d.warehouse_id == "WH_1")].sort_values("date")
    expect = lane[lane.date <= fe.CUTOFF].units_sold.tail(28).mean()
    sold, _, lanes = fe._panel(sc)
    i = lanes.index(("SKU_1", "WH_1"))
    assert preds["ma28"][i, 0] == pytest.approx(expect) and np.allclose(preds["ma28"][i], expect)


def test_reported_claim_matches_the_numbers(rolling):
    L, T = rolling["metrics"]["lane_daily"], rolling["metrics"]["total_daily"]
    beats = all(lvl["model"][k] < lvl[b][k] for lvl in (L, T) for b in ("ma28", "ma7") for k in ("mae", "rmse"))
    assert rolling["model_beats_baselines"] == beats
    # the verified result on this dataset: the model wins MAE/RMSE at both levels
    assert beats
    assert T["model"]["mape_pct"] < min(T["ma28"]["mape_pct"], T["ma7"]["mape_pct"])


def test_sku_scope_and_api(sc):
    r = fe.evaluate(sc, "rolling", "SKU_44")
    assert r["series_scope"] == "SKU_44" and r["sku_metrics"]["model"]["n"] == 91
    with pytest.raises(KeyError):
        fe.evaluate(sc, "rolling", "SKU_999")
    from backend.main import app
    with TestClient(app) as c:
        assert c.get("/api/forecast-evaluation").status_code == 200
        assert c.get("/api/forecast-evaluation?protocol=fixed&sku_id=SKU_2").json()["protocol"] == "fixed"
        assert c.get("/api/forecast-evaluation?protocol=bogus").status_code == 422
        assert c.get("/api/forecast-evaluation?sku_id=SKU_999").status_code == 404
        assert c.post("/api/run-agent-pipeline?sku_id=SKU_21&budget=0").status_code == 422
