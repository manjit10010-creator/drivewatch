"""Tests for the API on a fresh clone (step 5e): data/ and mlruns/ are not in Git, so there is no warehouse
and no model yet. The API must still start, say clearly what is missing (503) and keep the parts that
do not need the missing piece working.
"""
import pytest

pytest.importorskip("fastapi")  # skipped on a machine without the API packages (for example the Windows venv)
duckdb = pytest.importorskip("duckdb")

from src.api import db  # noqa: E402


def test_opening_a_missing_warehouse_raises_a_duckdb_error(tmp_path):
    """The API start-up relies on this: a missing file is a duckdb.Error, which main.py catches."""
    missing = tmp_path / "nope.duckdb"
    with pytest.raises(duckdb.Error):
        db.open_db(str(missing))
    assert not missing.exists()  # a read-only open must not create the file


@pytest.mark.parametrize("path", ["/health", "/fleet/risk", "/drive/ST_A"])
def test_endpoint_without_a_warehouse_answers_503_with_a_clear_message(no_warehouse_client, path):
    r = no_warehouse_client.get(path)
    assert r.status_code == 503
    assert "could not be opened" in r.json()["detail"] and "README" in r.json()["detail"]


def test_endpoint_feedback_post_without_a_warehouse_is_503_and_stores_nothing(no_warehouse_client):
    body = {"serial_number": "AAA111", "alert_date": "2026-06-30", "verdict": "true_positive"}
    assert no_warehouse_client.post("/feedback", json=body).status_code == 503
    assert no_warehouse_client.get("/feedback").json() == []


def test_endpoint_feedback_list_works_without_a_warehouse(no_warehouse_client):
    r = no_warehouse_client.get("/feedback")
    assert r.status_code == 200 and r.json() == []


def test_endpoint_predict_works_without_a_warehouse(no_warehouse_client, full_body):
    r = no_warehouse_client.post("/predict", json=full_body)
    assert r.status_code == 200 and r.json()["model_version"] == 1


def test_endpoint_bare_api_starts_and_says_what_is_missing(bare_client, full_body):
    assert bare_client.get("/health").status_code == 503
    r = bare_client.post("/predict", json=full_body)
    assert r.status_code == 503 and "model is not loaded" in r.json()["detail"]
    assert bare_client.get("/feedback").status_code == 200


def test_endpoint_bare_api_still_documents_itself(bare_client):
    assert bare_client.get("/docs").status_code == 200
    assert "/predict" in bare_client.get("/openapi.json").json()["paths"]
