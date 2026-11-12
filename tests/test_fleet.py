"""Tests for the fleet endpoints: GET /health, GET /fleet/risk and GET /drive/{serial} (step 5e).

The first block tests the SQL in db.py against the small in-memory warehouse from conftest.py (`warehouse`):
  2026-06-29: ST_A (rank 1), TOSH_A (rank 2), ST_B (rank 3)
  2026-06-30: ST_A (rank 1), TOSH_B (rank 2), TOSH_A (rank 3), ST_B (rank 4), scored by model version 2
The second block (test_endpoint_*) sends HTTP requests to the app running on top of that warehouse.
"""
import datetime as dt

import pytest

pytest.importorskip("fastapi")  # skipped on a machine without the API packages (for example the Windows venv)
pytest.importorskip("duckdb")

from src.api import db  # noqa: E402

D29, D30 = dt.date(2026, 6, 29), dt.date(2026, 6, 30)
MG07 = "TOSHIBA MG07ACA14TA"


def serials(drives):
    return [d["serial_number"] for d in drives]


# --- the SQL in db.py -------------------------------------------------------------------------------------

def test_db_date_range(warehouse):
    assert db.date_range(warehouse.cursor()) == (D29, D30)


def test_db_date_range_of_an_empty_table(empty_warehouse):
    assert db.date_range(empty_warehouse.cursor()) == (None, None)


def test_db_health_info_describes_the_latest_day(warehouse):
    info = db.health_info(warehouse.cursor())
    assert info == {"model_version": 2, "data_date": D30, "drives_on_data_date": 4, "score_rows": 7}


def test_db_health_info_of_an_empty_table_is_none(empty_warehouse):
    assert db.health_info(empty_warehouse.cursor()) is None


def test_db_fleet_risk_is_ordered_by_rank_and_limited(warehouse):
    rows = db.fleet_risk(warehouse.cursor(), D30, 2)
    assert rows == [(1, "ST_A", "ST12000NM0008", 0.99, "high"), (2, "TOSH_B", MG07, 0.8, "high")]


def test_db_fleet_risk_filters_by_model_and_keeps_the_fleet_wide_rank(warehouse):
    rows = db.fleet_risk(warehouse.cursor(), D30, 50, MG07)
    assert [(r[0], r[1]) for r in rows] == [(2, "TOSH_B"), (3, "TOSH_A")]


def test_db_fleet_risk_treats_the_model_as_data_not_as_sql(warehouse):
    assert db.fleet_risk(warehouse.cursor(), D30, 50, "x' OR '1'='1") == []
    assert db.date_range(warehouse.cursor()) == (D29, D30)  # and the table is still there


def test_db_drive_history_is_oldest_first_and_joins_the_smart_values(warehouse):
    rows = db.drive_history(warehouse.cursor(), "ST_A")
    assert [r[0] for r in rows] == [D29, D30]
    assert rows[0] == (D29, "ST12000NM0008", 0.9, "high", 1, 100.0, 5.0, 0.0, 20.0, 31.0)
    assert rows[1] == (D30, "ST12000NM0008", 0.99, "high", 1, 200.0, 6.0, 0.0, 40.0, 32.0)


def test_db_drive_history_keeps_days_without_smart_values(warehouse):
    rows = db.drive_history(warehouse.cursor(), "TOSH_A")
    assert len(rows) == 2  # LEFT JOIN: the day without a features_scoring row is still there
    assert rows[0][5:] == (3.0, None, None, 7.0, 29.5)
    assert rows[1][5:] == (None, None, None, None, None)


def test_db_drive_history_of_an_unknown_drive_is_empty(warehouse):
    assert db.drive_history(warehouse.cursor(), "NOSUCH") == []


# --- GET /health ------------------------------------------------------------------------------------------

def test_endpoint_health(fleet_client):
    r = fleet_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": 2, "data_date": "2026-06-30",
                        "drives_on_data_date": 4, "score_rows": 7}


def test_endpoint_health_is_503_when_nothing_has_been_scored(empty_client):
    r = empty_client.get("/health")
    assert r.status_code == 503
    assert "fleet_scores is empty" in r.json()["detail"]


# --- GET /fleet/risk --------------------------------------------------------------------------------------

def test_endpoint_fleet_risk_defaults_to_the_latest_day(fleet_client):
    r = fleet_client.get("/fleet/risk")
    assert r.status_code == 200
    data = r.json()
    assert data["date"] == "2026-06-30" and data["model_filter"] is None and data["count"] == 4
    assert serials(data["drives"]) == ["ST_A", "TOSH_B", "TOSH_A", "ST_B"]
    assert [d["rank"] for d in data["drives"]] == [1, 2, 3, 4]
    assert [d["risk_band"] for d in data["drives"]] == ["high", "high", "medium", "low"]
    assert set(data["drives"][0]) == {"rank", "serial_number", "model", "risk_score", "risk_band"}


def test_endpoint_fleet_risk_rounds_scores_to_four_decimals(fleet_client):
    drives = fleet_client.get("/fleet/risk").json()["drives"]
    assert drives[-1]["risk_score"] == 0.0123  # stored as 0.012345678


def test_endpoint_fleet_risk_top(fleet_client):
    data = fleet_client.get("/fleet/risk", params={"top": 2}).json()
    assert data["count"] == 2 and serials(data["drives"]) == ["ST_A", "TOSH_B"]


def test_endpoint_fleet_risk_for_an_earlier_day(fleet_client):
    data = fleet_client.get("/fleet/risk", params={"date": "2026-06-29"}).json()
    assert data["date"] == "2026-06-29"
    assert serials(data["drives"]) == ["ST_A", "TOSH_A", "ST_B"]


def test_endpoint_fleet_risk_model_filter_keeps_the_fleet_wide_rank(fleet_client):
    data = fleet_client.get("/fleet/risk", params={"model": MG07}).json()
    assert data["model_filter"] == MG07 and data["count"] == 2
    assert [(d["rank"], d["serial_number"]) for d in data["drives"]] == [(2, "TOSH_B"), (3, "TOSH_A")]


def test_endpoint_fleet_risk_model_without_matches_is_an_empty_list(fleet_client):
    r = fleet_client.get("/fleet/risk", params={"model": "NO SUCH MODEL"})
    assert r.status_code == 200
    assert r.json()["count"] == 0 and r.json()["drives"] == []


def test_endpoint_fleet_risk_model_text_cannot_change_the_query(fleet_client):
    r = fleet_client.get("/fleet/risk", params={"model": "x' OR '1'='1"})
    assert r.status_code == 200 and r.json()["count"] == 0


def test_endpoint_fleet_risk_top_1000_is_allowed(fleet_client):
    assert fleet_client.get("/fleet/risk", params={"top": 1000}).status_code == 200


@pytest.mark.parametrize("params", [
    {"top": 0}, {"top": 1001}, {"top": -5}, {"top": "abc"}, {"date": "not-a-date"}, {"model": "x" * 65},
])
def test_endpoint_fleet_risk_rejects_bad_parameters(fleet_client, params):
    assert fleet_client.get("/fleet/risk", params=params).status_code == 422


@pytest.mark.parametrize("day", ["2026-06-28", "2026-07-01"])
def test_endpoint_fleet_risk_for_a_day_without_scores_is_404(fleet_client, day):
    r = fleet_client.get("/fleet/risk", params={"date": day})
    assert r.status_code == 404
    assert "2026-06-29 to 2026-06-30" in r.json()["detail"]  # tells the caller which dates exist


def test_endpoint_fleet_risk_is_503_when_nothing_has_been_scored(empty_client):
    r = empty_client.get("/fleet/risk")
    assert r.status_code == 503 and "fleet_scores is empty" in r.json()["detail"]


# --- GET /drive/{serial} ----------------------------------------------------------------------------------

def test_endpoint_drive_history(fleet_client):
    r = fleet_client.get("/drive/ST_A")
    assert r.status_code == 200
    data = r.json()
    assert data["serial_number"] == "ST_A" and data["model"] == "ST12000NM0008"
    assert (data["latest_date"], data["latest_score"], data["latest_band"], data["latest_rank"]) == \
           ("2026-06-30", 0.99, "high", 1)
    assert [day["date"] for day in data["history"]] == ["2026-06-29", "2026-06-30"]  # oldest first
    assert data["history"][0] == {"date": "2026-06-29", "risk_score": 0.9, "risk_band": "high", "rank_in_day": 1,
                                  "smart_5_raw": 100.0, "smart_187_raw": 5.0, "smart_188_raw": 0.0,
                                  "smart_197_raw": 20.0, "temp_c": 31.0}


def test_endpoint_drive_latest_values_come_from_the_last_day(fleet_client):
    data = fleet_client.get("/drive/TOSH_A").json()
    assert (data["latest_date"], data["latest_score"], data["latest_band"], data["latest_rank"]) == \
           ("2026-06-30", 0.3, "medium", 3)


def test_endpoint_drive_missing_smart_values_are_null(fleet_client):
    history = fleet_client.get("/drive/TOSH_A").json()["history"]
    assert history[0]["smart_5_raw"] == 3.0 and history[0]["smart_187_raw"] is None
    assert all(history[1][k] is None for k in
               ("smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "temp_c"))


def test_endpoint_drive_without_any_smart_values_still_has_its_scores(fleet_client):
    data = fleet_client.get("/drive/ST_B").json()
    assert len(data["history"]) == 2 and data["latest_rank"] == 4
    assert all(day["temp_c"] is None for day in data["history"])


def test_endpoint_unknown_drive_is_404(fleet_client):
    r = fleet_client.get("/drive/NOSUCH")
    assert r.status_code == 404 and "NOSUCH" in r.json()["detail"]


@pytest.mark.parametrize("serial", ["bad;serial", "bad%20serial", "x" * 65, "drop'table"])
def test_endpoint_drive_rejects_illegal_serial_numbers(fleet_client, serial):
    assert fleet_client.get(f"/drive/{serial}").status_code == 422


# --- when the warehouse breaks ----------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/health", "/fleet/risk", "/drive/ST_A"])
def test_endpoint_a_broken_warehouse_gives_503_not_a_crash(fleet_client, warehouse, path):
    warehouse.execute("DROP TABLE fleet_scores")
    r = fleet_client.get(path)
    assert r.status_code == 503
    assert "warehouse could not answer" in r.json()["detail"]
