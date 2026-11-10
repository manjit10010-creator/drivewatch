"""Tests for live scoring: POST /predict (step 5e).

The first block tests the plain functions (bands, request validation, scoring). The last block
(test_endpoint_*) sends real HTTP requests to the app through FastAPI's TestClient.
"""
import json
import logging

import numpy as np
import pytest

pytest.importorskip("fastapi")  # skipped on a machine without the API packages (for example the Windows venv)

from pydantic import ValidationError  # noqa: E402

from src.api.predict import FEATURES, MIN_FILLED, PredictRequest, band_for, score_one  # noqa: E402


# --- risk bands -------------------------------------------------------------------------------------------

def test_band_for_thresholds_are_inclusive():
    bands = {"high": 0.9, "medium": 0.6}
    assert band_for(0.95, bands) == "high"
    assert band_for(0.9, bands) == "high"
    assert band_for(0.8999, bands) == "medium"
    assert band_for(0.6, bands) == "medium"
    assert band_for(0.5999, bands) == "low"


# --- request validation ------------------------------------------------------------------------------------

def body_with_filled(n, full_body):
    """A body in which only the first n features are present (the others are left out)."""
    return {name: full_body[name] for name in FEATURES[:n]}


@pytest.mark.parametrize("n, accepted", [
    (0, False), (10, False), (MIN_FILLED - 1, False), (MIN_FILLED, True), (24, True), (25, True),
])
def test_request_needs_enough_filled_features(n, accepted, full_body):
    body = body_with_filled(n, full_body)
    if accepted:
        PredictRequest(**body)
    else:
        with pytest.raises(ValidationError, match="features are filled in"):
            PredictRequest(**body)


def test_explicit_nulls_count_as_not_filled(full_body):
    body = dict(full_body)
    for name in FEATURES[:len(FEATURES) - MIN_FILLED + 1]:  # one null too many
        body[name] = None
    with pytest.raises(ValidationError, match="features are filled in"):
        PredictRequest(**body)


def test_serial_number_alone_is_not_enough():
    with pytest.raises(ValidationError):
        PredictRequest(serial_number="ABC123")


def test_misspelled_feature_name_is_rejected(full_body):
    body = dict(full_body)
    del body["smart_5_raw"]
    body["smart_5_rawx"] = 1.0
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PredictRequest(**body)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "abc"])
def test_non_finite_or_non_numeric_value_is_rejected(bad, full_body):
    with pytest.raises(ValidationError):
        PredictRequest(**{**full_body, "temp_c": bad})


@pytest.mark.parametrize("serial", ["has space", "a;b", "", "x" * 65, "drop'table"])
def test_bad_serial_number_is_rejected(serial, full_body):
    with pytest.raises(ValidationError):
        PredictRequest(**{**full_body, "serial_number": serial})


def test_good_serial_number_and_no_serial_are_accepted(full_body):
    assert PredictRequest(**{**full_body, "serial_number": "ZHZ42XAP"}).serial_number == "ZHZ42XAP"
    assert PredictRequest(**full_body).serial_number is None


# --- score_one --------------------------------------------------------------------------------------------

def test_missing_values_reach_the_booster_as_nan(fake_model):
    score_one(fake_model, {"smart_197_raw": 5.0})
    x = fake_model.booster.last_x[0]
    assert np.isnan(x).sum() == len(FEATURES) - 1
    assert x[fake_model.features.index("smart_197_raw")] == 5.0


def test_row_follows_the_boosters_feature_order(make_model):
    reversed_names = FEATURES[::-1]  # the booster, not the request, decides the column order
    model = make_model(features=reversed_names)
    score_one(model, {"smart_5_raw": 7.0})
    x = model.booster.last_x[0]
    assert x[reversed_names.index("smart_5_raw")] == 7.0
    assert np.isnan(x).sum() == len(FEATURES) - 1


def test_top_features_are_sorted_by_size_and_zero_contributions_are_dropped(fake_model):
    values = {"smart_197_raw": 5000.0, "smart_5_raw": 100.0, "temp_c": 30.0, "n_obs_30": 0.0}
    score, band, top = score_one(fake_model, values)
    assert [c.feature for c in top] == ["smart_197_raw", "smart_5_raw", "temp_c"]  # n_obs_30=0 adds nothing
    assert top[0].value == 5000.0 and top[0].contribution == 5.0
    assert band == "high" and 0.9 < score < 1.0


def test_top_features_are_limited_to_top_n(fake_model, full_body):
    _, _, top = score_one(fake_model, full_body, top_n=3)
    assert len(top) == 3
    _, _, top = score_one(fake_model, full_body)
    assert len(top) == 5  # the default


def test_the_same_features_always_give_the_same_score(fake_model, full_body):
    assert score_one(fake_model, full_body)[0] == score_one(fake_model, full_body)[0]


# --- the endpoint -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("smart_197, band", [(1.0, "low"), (700.0, "medium"), (5000.0, "high")])
def test_endpoint_predict_returns_the_band(client, full_body, smart_197, band):
    r = client.post("/predict", json={**full_body, "smart_197_raw": smart_197})
    assert r.status_code == 200
    assert r.json()["risk_band"] == band


def test_endpoint_predict_response_shape(client, full_body):
    r = client.post("/predict", json={**full_body, "serial_number": "ZHZ42XAP", "smart_197_raw": 5000.0})
    assert r.status_code == 200
    data = r.json()
    assert set(data) == {"serial_number", "model_version", "risk_score", "risk_band", "top_features"}
    assert data["serial_number"] == "ZHZ42XAP" and data["model_version"] == 1
    assert 0.0 < data["risk_score"] < 1.0
    top = data["top_features"]
    assert 1 <= len(top) <= 5
    assert top[0]["feature"] == "smart_197_raw" and top[0]["value"] == 5000.0 and top[0]["contribution"] > 0


def test_endpoint_predict_rejects_an_empty_body(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422
    assert "features are filled in" in json.dumps(r.json())


def test_endpoint_predict_rejects_a_misspelled_feature(client, full_body):
    body = dict(full_body)
    del body["smart_5_raw"]
    body["smart_5_rawx"] = 1.0
    assert client.post("/predict", json=body).status_code == 422


def test_endpoint_predict_rejects_a_body_that_is_not_json(client):
    r = client.post("/predict", content="not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_endpoint_predict_only_accepts_post(client):
    assert client.get("/predict").status_code == 405


def test_endpoint_predict_answers_503_when_the_model_is_not_loaded(client_no_model, full_body):
    r = client_no_model.post("/predict", json=full_body)
    assert r.status_code == 503
    assert "model is not loaded" in r.json()["detail"]


def test_endpoint_responses_carry_the_timing_header(client, full_body):
    r = client.post("/predict", json=full_body)
    assert float(r.headers["X-Process-Time-ms"]) >= 0.0


def test_endpoint_predict_writes_one_json_log_line(client, full_body, caplog):
    caplog.set_level(logging.INFO, logger="drivewatch.predictions")
    r = client.post("/predict", json={**full_body, "serial_number": "ABC123"})
    lines = [json.loads(rec.getMessage()) for rec in caplog.records if rec.name == "drivewatch.predictions"]
    assert len(lines) == 1
    line = lines[0]
    assert line["event"] == "prediction" and line["serial_number"] == "ABC123"
    assert line["model_version"] == 1 and line["risk_band"] == r.json()["risk_band"]
    assert set(line) >= {"time", "risk_score", "ms"}


def test_endpoint_a_rejected_request_is_not_logged_as_a_prediction(client, caplog):
    caplog.set_level(logging.INFO, logger="drivewatch.predictions")
    client.post("/predict", json={})
    assert [rec for rec in caplog.records if rec.name == "drivewatch.predictions"] == []
