"""Shared set-up for the DriveWatch API tests (step 5e).

The tests never touch the real warehouse, the MLflow registry or data/feedback.db:
- the model is a FakeBooster (a tiny stand-in with LightGBM's predict interface),
- the warehouse is a small in-memory DuckDB (fleet_scores and features_scoring with a few drives),
- the feedback database is a fresh SQLite file in a temporary folder.

Run them inside the API image (it has duckdb, lightgbm and fastapi; pytest and httpx are added on the fly).
From the repo root, one short command per step (long pasted lines get cut off):
    docker run -d --rm --name dwtest -v "${PWD}:/workspace" drivewatch-api:slim sleep 600
    docker exec dwtest pip install -q pytest httpx
    docker exec dwtest python -m pytest -q -p no:cacheprovider tests/test_predict.py tests/test_feedback.py tests/test_fleet.py tests/test_fresh_clone.py
    docker rm -f dwtest
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


FLEET_SCORES_DDL = ("CREATE TABLE fleet_scores (date DATE, serial_number VARCHAR, model VARCHAR, "
                    "model_version INTEGER, risk_score DOUBLE, risk_band VARCHAR, rank_in_day INTEGER)")
FEATURES_SCORING_DDL = ("CREATE TABLE features_scoring (date DATE, serial_number VARCHAR, smart_5_raw DOUBLE, "
                        "smart_187_raw DOUBLE, smart_188_raw DOUBLE, smart_197_raw DOUBLE, temp_c DOUBLE)")


def _empty_warehouse():
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute(FLEET_SCORES_DDL)
    con.execute(FEATURES_SCORING_DDL)
    return con


def _feedback_warehouse():
    """Two drives on one day: what the /feedback tests need (AAA111 is band high, BBB222 is band low)."""
    con = _empty_warehouse()
    con.execute("INSERT INTO fleet_scores VALUES "
                "('2026-06-30', 'AAA111', 'ST12000NM0008', 1, 0.99, 'high', 1), "
                "('2026-06-30', 'BBB222', 'TOSHIBA MG07ACA14TA', 1, 0.01, 'low', 2)")
    return con


def _fleet_warehouse():
    """Two scored days. 2026-06-29: three drives. 2026-06-30 (latest, model version 2): four drives.

    features_scoring is deliberately incomplete, to test the LEFT JOIN in drive_history:
    ST_A has SMART values on both days, TOSH_A on the first day only (and with some NULLs), ST_B and TOSH_B on none.
    """
    con = _empty_warehouse()
    con.execute("INSERT INTO fleet_scores VALUES "
                "('2026-06-29', 'ST_A',   'ST12000NM0008',       1, 0.90,        'high',   1), "
                "('2026-06-29', 'TOSH_A', 'TOSHIBA MG07ACA14TA', 1, 0.50,        'medium', 2), "
                "('2026-06-29', 'ST_B',   'ST12000NM0008',       1, 0.01,        'low',    3), "
                "('2026-06-30', 'ST_A',   'ST12000NM0008',       2, 0.99,        'high',   1), "
                "('2026-06-30', 'TOSH_B', 'TOSHIBA MG07ACA14TA', 2, 0.80,        'high',   2), "
                "('2026-06-30', 'TOSH_A', 'TOSHIBA MG07ACA14TA', 2, 0.30,        'medium', 3), "
                "('2026-06-30', 'ST_B',   'ST12000NM0008',       2, 0.012345678, 'low',    4)")
    con.execute("INSERT INTO features_scoring VALUES "
                "('2026-06-29', 'ST_A',   100, 5,    0,    20,   31.0), "
                "('2026-06-30', 'ST_A',   200, 6,    0,    40,   32.0), "
                "('2026-06-29', 'TOSH_A', 3,   NULL, NULL, 7,    29.5)")
    return con


@pytest.fixture
def warehouse():
    """The two-day fleet warehouse (a DuckDB connection) for tests of the SQL in db.py."""
    pytest.importorskip("duckdb")
    return _fleet_warehouse()


@pytest.fixture
def empty_warehouse():
    pytest.importorskip("duckdb")
    return _empty_warehouse()


def _make_client(monkeypatch, feedback_path, model, con=None):
    """An API client whose warehouse, model and feedback file are all test doubles."""
    pytest.importorskip("httpx")  # FastAPI's TestClient needs it
    pytest.importorskip("duckdb")
    from fastapi.testclient import TestClient
    from src.api import db, feedback, main, predict

    con = con if con is not None else _feedback_warehouse()

    def fake_open_db():
        if isinstance(con, Exception):  # a test can pass an error to simulate a warehouse that cannot be opened
            raise con
        return con
    monkeypatch.setattr(db, "open_db", fake_open_db)

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


@pytest.fixture
def fleet_client(monkeypatch, feedback_path, fake_model, warehouse):
    """The API on top of the two-day fleet warehouse. The same connection is `warehouse`, so a test can break it."""
    with _make_client(monkeypatch, feedback_path, fake_model, con=warehouse) as c:
        yield c


@pytest.fixture
def empty_client(monkeypatch, feedback_path, fake_model, empty_warehouse):
    """The API when fleet_scores exists but holds no rows (before score_fleet.py has run)."""
    with _make_client(monkeypatch, feedback_path, fake_model, con=empty_warehouse) as c:
        yield c


@pytest.fixture
def no_warehouse_client(monkeypatch, feedback_path, fake_model):
    """The API on a fresh clone: data/gold/drivewatch.duckdb does not exist, so opening it fails at start-up."""
    import duckdb
    with _make_client(monkeypatch, feedback_path, fake_model, con=duckdb.Error("file does not exist")) as c:
        yield c


@pytest.fixture
def bare_client(monkeypatch, feedback_path):
    """The API with neither a warehouse nor a model (a fresh clone before anything has been built)."""
    import duckdb
    with _make_client(monkeypatch, feedback_path, None, con=duckdb.Error("file does not exist")) as c:
        yield c
