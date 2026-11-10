"""Live scoring for POST /predict (step 5d).

The fleet board and the drive pages read scores that were computed ahead of time (fleet_scores).
This module is the one place where the API runs the model itself: you send the 25 model features of
one drive on one day, and you get back a risk score, a risk band and the features that pushed the score.

The model is loaded ONCE when the API starts (load_model, called from the lifespan in main.py), never
per request. It comes from the MLflow registry (models:/drivewatch-risk@production), the same model
that score_fleet.py uses, so a live score and a stored score for the same features are identical.

The caller sends ready-made features. Computing them (rolling windows, deltas, error-day counters) needs
a drive's recent history and stays in the daily pipeline (build_scoring_features.py).
"""
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.api.schemas import RiskBand

log = logging.getLogger("drivewatch.api")
pred_log = logging.getLogger("drivewatch.predictions")  # one JSON line per prediction

TRACKING_URI = "sqlite:////workspace/mlflow.db"  # the repo is mounted at /workspace in the container
MODEL_NAME = "drivewatch-risk"
MODEL_URI = f"models:/{MODEL_NAME}@production"
BANDS_PATH = "reports/risk_bands.json"
TOP_N = 5  # how many contributing features to return
# A request needs at least this many of the 25 features filled in. In June 2026 every real drive-day of the
# 3 development models had 24 or 25 filled (check_nonnull.py), so 20 leaves room for drive models that do not
# report some SMART attributes while still rejecting an empty or nearly empty body, which LightGBM would
# otherwise score as a confident 'high' (all-missing values follow the learned missing-value branches).
# Re-check this number when the model is widened to all drive models.
MIN_FILLED = 20

# The 25 features of the production model (reports/selected_features_25.txt). load_model() checks that the
# registry model expects exactly these names, so a mismatch stops /predict instead of scoring garbage.
FEATURES = [
    "err_days30_smart_197", "err_days30_smart_188", "err_days30_smart_187",
    "err_days30_smart_5", "err_days30_smart_198", "err_days30_smart_199",
    "delta7_smart_187", "delta7_smart_5", "delta7_smart_188", "delta7_smart_197", "delta7_smart_198",
    "delta30_smart_197", "delta30_smart_198",
    "smart_193_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_12_raw",
    "smart_5_raw",
    "temp_c", "n_obs_30", "temp_std_14", "temp_std_30", "miss_smart_187",
]


def _feature():
    """A number that may be left out: a missing value reaches LightGBM as NaN, as it did in training."""
    return Field(None, allow_inf_nan=False)


class PredictRequest(BaseModel):
    """The 25 model features of one drive on one day. Leave a feature out (or null) when it is not reported."""
    model_config = ConfigDict(extra="forbid")  # a misspelled feature name is a 422, not a silently ignored field

    serial_number: Optional[str] = Field(None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$",
                                         description="Only used to label the prediction log")
    # Days in the last 30 on which the error counter was above zero
    err_days30_smart_197: Optional[float] = _feature()
    err_days30_smart_188: Optional[float] = _feature()
    err_days30_smart_187: Optional[float] = _feature()
    err_days30_smart_5: Optional[float] = _feature()
    err_days30_smart_198: Optional[float] = _feature()
    err_days30_smart_199: Optional[float] = _feature()
    # Change of the counter over the last 7 and 30 days
    delta7_smart_187: Optional[float] = _feature()
    delta7_smart_5: Optional[float] = _feature()
    delta7_smart_188: Optional[float] = _feature()
    delta7_smart_197: Optional[float] = _feature()
    delta7_smart_198: Optional[float] = _feature()
    delta30_smart_197: Optional[float] = _feature()
    delta30_smart_198: Optional[float] = _feature()
    # Latest raw SMART values
    smart_193_raw: Optional[float] = _feature()
    smart_9_raw: Optional[float] = _feature()
    smart_187_raw: Optional[float] = _feature()
    smart_188_raw: Optional[float] = _feature()
    smart_197_raw: Optional[float] = _feature()
    smart_12_raw: Optional[float] = _feature()
    smart_5_raw: Optional[float] = _feature()
    # Temperature, how many of the last 30 days the drive reported, and whether SMART 187 is reported at all
    temp_c: Optional[float] = _feature()
    n_obs_30: Optional[float] = _feature()
    temp_std_14: Optional[float] = _feature()
    temp_std_30: Optional[float] = _feature()
    miss_smart_187: Optional[float] = _feature()

    @model_validator(mode="after")
    def enough_features(self):
        filled = sum(getattr(self, name) is not None for name in FEATURES)
        if filled < MIN_FILLED:
            raise ValueError(f"Only {filled} of the {len(FEATURES)} features are filled in; "
                             f"at least {MIN_FILLED} are needed to score a drive.")
        return self


class Contribution(BaseModel):
    feature: str
    value: Optional[float]
    contribution: float  # in log-odds: positive pushes the risk up, negative pushes it down


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    serial_number: Optional[str]
    model_version: int
    risk_score: float
    risk_band: RiskBand
    top_features: list[Contribution]


@dataclass
class LoadedModel:
    booster: object
    version: int
    features: list  # in the order the booster expects them
    bands: dict     # {"high": ..., "medium": ...}


def load_model() -> LoadedModel:
    """Load the production model from the registry plus the band thresholds. Called once, at API start-up."""
    import mlflow  # imported here so that importing this module (for tests) stays light
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    booster = mlflow.lightgbm.load_model(MODEL_URI)
    version = int(MlflowClient().get_model_version_by_alias(MODEL_NAME, "production").version)
    feats = booster.feature_name()
    if set(feats) != set(FEATURES):
        raise RuntimeError(f"The registry model expects different features than predict.py knows.\n"
                           f"model only: {sorted(set(feats) - set(FEATURES))}\n"
                           f"code only:  {sorted(set(FEATURES) - set(feats))}")
    with open(BANDS_PATH) as f:
        th = json.load(f)["thresholds"]
    return LoadedModel(booster, version, feats, {"high": th["high"], "medium": th["medium"]})


def band_for(score: float, bands: dict) -> str:
    """Same rule as score_fleet.assign_band: the highest threshold the score reaches."""
    if score >= bands["high"]:
        return "high"
    if score >= bands["medium"]:
        return "medium"
    return "low"


def score_one(model: LoadedModel, values: dict, top_n: int = TOP_N):
    """Score one drive-day. `values` maps feature name -> number or None. Returns (score, band, top contributions)."""
    # Build the row in the booster's own feature order; None becomes NaN (LightGBM's native "missing").
    row = [np.nan if values.get(name) is None else float(values[name]) for name in model.features]
    x = np.array([row], dtype="float64")
    score = float(model.booster.predict(x, num_threads=1)[0])
    # Per-feature contributions to the raw (log-odds) score; the last column is the baseline, which we drop.
    contrib = np.asarray(model.booster.predict(x, pred_contrib=True, num_threads=1))[0][:-1]
    order = np.argsort(-np.abs(contrib))[:top_n]
    top = [Contribution(feature=model.features[i], value=None if np.isnan(row[i]) else row[i],
                        contribution=round(float(contrib[i]), 4))
           for i in order if contrib[i] != 0.0]
    return score, band_for(score, model.bands), top


router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict_drive(body: PredictRequest, request: Request):
    """Score one drive from its 25 features: risk score, risk band and the features that moved the score most."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="The model is not loaded; see the API start-up log.")
    t0 = time.perf_counter()
    values = body.model_dump(exclude={"serial_number"})
    score, band, top = score_one(model, values)
    pred_log.info(json.dumps({
        "event": "prediction",
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "serial_number": body.serial_number,
        "model_version": model.version,
        "risk_score": round(score, 6),
        "risk_band": band,
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }))
    return PredictResponse(serial_number=body.serial_number, model_version=model.version,
                           risk_score=round(score, 4), risk_band=band, top_features=top)
