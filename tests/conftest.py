"""Shared set-up for the DriveWatch API tests (step 5e).

The tests never touch the real warehouse, the MLflow registry or data/feedback.db:
- the model is a FakeBooster (a tiny stand-in with LightGBM's predict interface),
- the warehouse is a small in-memory DuckDB with a handful of fleet_scores rows,
- the feedback database is a fresh SQLite file in a temporary folder.

Run them inside the API image (it has duckdb, lightgbm and fastapi; pytest and httpx are added on the fly).
From the repo root, one line:
    docker run --rm -v "${PWD}:/workspace" drivewatch-api:slim sh -c "pip install -q pytest httpx && python -m pytest -q -p no:cacheprovider tests/test_predict.py tests/test_feedback.py"
"""
import numpy as np
import pytest

# Band thresholds used by the fake model. With FakeBooster below: all features 1.0 -> low,
# smart_197_raw=700 -> medium, smart_197_raw=5000 -> high.
BANDS = {"high": 0.9, "medium": 0.6}


class FakeBooster:
    """Same predict() interface as a LightGBM booster, but deterministic and instant.

    Every feature adds 0.001 * its value to the raw (log-odds) score; a missing value (NaN) adds nothing.
    The score is the sigmoid of the sum. last_x remembers the matrix of the latest call, so a test can
    check what actually reached the booster.
    """

    def __init__(self, feature_names):
        self._names = list(feature_names)
        self.last_x = None

    def feature_name(self):
        return list(self._names)

    def predict(self, x, pred_contrib=False, num_threads=1):
        self.last_x = np.array(x, dtype="float64")
        contrib = np.nan_to_num(self.last_x) * 0.001
        if pred_contrib:  # like LightGBM: one column per feature plus a last column with the baseline
            return np.hstack([contrib, np.zeros((contrib.shape[0], 1))])
        return 1.0 / (1.0 + np.exp(-contrib.sum(axis=1)))


@pytest.fixture
def make_model():
    """Factory: make_model(features=None, bands=None) -> LoadedModel around a FakeBooster."""
    def _make(features=None, bands=None):
        from src.api.predict import FEATURES, LoadedModel
        feats = list(features or FEATURES)
        return LoadedModel(booster=FakeBooster(feats), version=1, features=feats, bands=dict(bands or BANDS))
    return _make


@pytest.fixture
def fake_model(make_model):
    return make_model()


@pytest.fixture
def full_body():
    """A request body with all 25 features filled in (every value 1.0)."""
    from src.api.predict import FEATURES
    return {name: 1.0 for name in FEATURES}


@pytest.fixture
def feedback_path(tmp_path):
    return str(tmp_path / "feedback.db")


def _make_client(monkeypatch, feedback_path, model):
    """An API client whose warehouse, model and feedback file are all test doubles."""
    pytest.importorskip("httpx")  # FastAPI's TestClient needs it
    import duckdb
    from fastapi.testclient import TestClient
    from src.api import db, feedback, main, predict

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE fleet_scores (date DATE, serial_number VARCHAR, model VARCHAR, "
                "model_version INTEGER, risk_score DOUBLE, risk_band VARCHAR, rank_in_day INTEGER)")
    con.execute("INSERT INTO fleet_scores VALUES "
                "('2026-06-30', 'AAA111', 'ST12000NM0008', 1, 0.99, 'high', 1), "
                "('2026-06-30', 'BBB222', 'TOSHIBA MG07ACA14TA', 1, 0.01, 'low', 2)")
    monkeypatch.setattr(db, "open_db", lambda: con)

    # init_db() has the path as a default argument, so create the table here and then switch it off,
    # otherwise the API start-up would create (or touch) the real data/feedback.db.
    feedback.init_db(feedback_path)
    monkeypatch.setattr(feedback, "DB_PATH", feedback_path)
    monkeypatch.setattr(feedback, "init_db", lambda *args, **kwargs: None)

    def fake_load_model():
        if model is None:
            raise RuntimeError("no model in this test")
        return model
    monkeypatch.setattr(predict, "load_model", fake_load_model)
    return TestClient(main.app)


@pytest.fixture
def client(monkeypatch, feedback_path, fake_model):
    with _make_client(monkeypatch, feedback_path, fake_model) as c:  # `with` runs the start-up (lifespan)
        yield c


@pytest.fixture
def client_no_model(monkeypatch, feedback_path):
    """The API after a failed model load: it starts, and /predict answers 503."""
    with _make_client(monkeypatch, feedback_path, None) as c:
        yield c
