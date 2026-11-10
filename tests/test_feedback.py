"""Tests for alert feedback: POST /feedback and GET /feedback (step 5e).

The first block tests the SQLite functions directly; test_endpoint_* send HTTP requests to the app.
The fake warehouse holds two scored drives on 2026-06-30: AAA111 (band high) and BBB222 (band low).
"""
import datetime as dt
import sqlite3
from contextlib import closing

import pytest

pytest.importorskip("fastapi")  # skipped on a machine without the API packages (for example the Windows venv)

from src.api import feedback  # noqa: E402


def good(**changes):
    return {"serial_number": "AAA111", "alert_date": "2026-06-30", "verdict": "true_positive", **changes}


# --- the SQLite functions ---------------------------------------------------------------------------------

def test_init_db_can_run_twice(feedback_path):
    feedback.init_db(feedback_path)
    feedback.init_db(feedback_path)  # CREATE TABLE IF NOT EXISTS: a second start-up must not fail


def test_insert_and_list_roundtrip_newest_first(feedback_path):
    feedback.init_db(feedback_path)
    first = feedback.insert_feedback(feedback_path, "AAA111", dt.date(2026, 6, 30), "true_positive", 1, "high", None)
    second = feedback.insert_feedback(feedback_path, "BBB222", dt.date(2026, 6, 30), "false_positive", 1, "low", "fan noise")
    assert second > first
    rows = feedback.list_feedback(feedback_path)
    assert [r.id for r in rows] == [second, first]
    assert rows[0].serial_number == "BBB222" and rows[0].note == "fan noise"
    assert rows[1].alert_date == dt.date(2026, 6, 30) and rows[1].risk_band == "high"


def test_list_feedback_respects_the_limit(feedback_path):
    feedback.init_db(feedback_path)
    for _ in range(3):
        feedback.insert_feedback(feedback_path, "AAA111", dt.date(2026, 6, 30), "true_positive", 1, "high", None)
    assert len(feedback.list_feedback(feedback_path, limit=2)) == 2


def test_database_itself_rejects_an_unknown_verdict(feedback_path):
    feedback.init_db(feedback_path)
    with pytest.raises(sqlite3.IntegrityError):
        feedback.insert_feedback(feedback_path, "AAA111", dt.date(2026, 6, 30), "maybe", 1, "high", None)


# --- POST /feedback ---------------------------------------------------------------------------------------

def test_endpoint_feedback_is_stored_with_model_version_and_band_from_the_warehouse(client, feedback_path):
    r = client.post("/feedback", json=good(note="disk had bad sectors"))
    assert r.status_code == 201
    data = r.json()
    assert data["serial_number"] == "AAA111" and data["alert_date"] == "2026-06-30"
    assert data["verdict"] == "true_positive" and data["note"] == "disk had bad sectors"
    assert data["model_version"] == 1 and data["risk_band"] == "high"
    with closing(sqlite3.connect(feedback_path)) as con:
        assert con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1


def test_endpoint_feedback_ignores_a_band_sent_by_the_client(client):
    r = client.post("/feedback", json=good(risk_band="low", model_version=99))
    assert r.status_code == 201
    assert r.json()["risk_band"] == "high" and r.json()["model_version"] == 1


def test_endpoint_feedback_for_a_low_band_drive_is_allowed(client):
    r = client.post("/feedback", json=good(serial_number="BBB222", verdict="false_positive"))
    assert r.status_code == 201 and r.json()["risk_band"] == "low"


def test_endpoint_feedback_for_an_unknown_drive_is_404(client):
    assert client.post("/feedback", json=good(serial_number="NOSUCH1")).status_code == 404


def test_endpoint_feedback_for_a_date_without_a_score_is_404(client):
    assert client.post("/feedback", json=good(alert_date="2026-06-29")).status_code == 404


def test_endpoint_a_404_stores_nothing(client, feedback_path):
    client.post("/feedback", json=good(serial_number="NOSUCH1"))
    with closing(sqlite3.connect(feedback_path)) as con:
        assert con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0


@pytest.mark.parametrize("changes", [
    {"verdict": "maybe"},
    {"verdict": None},
    {"serial_number": "AAA 111"},
    {"serial_number": "x'; DROP TABLE feedback;--"},
    {"serial_number": ""},
    {"alert_date": "2026-13-45"},
    {"alert_date": "yesterday"},
    {"note": "x" * 501},
])
def test_endpoint_feedback_rejects_bad_input(client, changes):
    assert client.post("/feedback", json=good(**changes)).status_code == 422


def test_endpoint_feedback_needs_a_verdict(client):
    body = good()
    del body["verdict"]
    assert client.post("/feedback", json=body).status_code == 422


def test_endpoint_a_second_answer_is_a_new_row_not_an_overwrite(client, feedback_path):
    first = client.post("/feedback", json=good(verdict="true_positive")).json()
    second = client.post("/feedback", json=good(verdict="false_positive")).json()
    assert second["id"] > first["id"]
    with closing(sqlite3.connect(feedback_path)) as con:
        assert con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 2


# --- GET /feedback ----------------------------------------------------------------------------------------

def test_endpoint_get_feedback_starts_empty(client):
    r = client.get("/feedback")
    assert r.status_code == 200 and r.json() == []


def test_endpoint_get_feedback_lists_newest_first(client):
    client.post("/feedback", json=good(verdict="true_positive"))
    client.post("/feedback", json=good(verdict="false_positive"))
    rows = client.get("/feedback").json()
    assert [r["verdict"] for r in rows] == ["false_positive", "true_positive"]


def test_endpoint_get_feedback_limit(client):
    for _ in range(3):
        client.post("/feedback", json=good())
    assert len(client.get("/feedback?limit=2").json()) == 2


@pytest.mark.parametrize("limit", [0, 501, -1, "abc"])
def test_endpoint_get_feedback_rejects_a_bad_limit(client, limit):
    assert client.get(f"/feedback?limit={limit}").status_code == 422
