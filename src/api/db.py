"""Read-only access to the gold DuckDB for the API.

Rules this file follows:
  - The API NEVER writes to the warehouse. It opens the file read-only; the batch scripts do the writing.
  - Every value that comes from a request goes in as a query parameter (the ? marks), never glued into the SQL
    text, so a request cannot change what the query does.
Each function takes a DuckDB cursor and returns plain Python values, which keeps them easy to test.
"""
import os

import duckdb

DB_PATH = os.environ.get("DRIVEWATCH_DB", "data/gold/drivewatch.duckdb")


def open_db(path=None):
    return duckdb.connect(path or DB_PATH, read_only=True)


def date_range(cur):
    """(first date, last date) in fleet_scores, or (None, None) when the table is empty."""
    return cur.execute("SELECT MIN(date), MAX(date) FROM fleet_scores").fetchone()


def health_info(cur):
    day, version, rows = cur.execute("SELECT MAX(date), MAX(model_version), COUNT(*) FROM fleet_scores").fetchone()
    if day is None:
        return None
    drives = cur.execute("SELECT COUNT(*) FROM fleet_scores WHERE date = ?", [day]).fetchone()[0]
    return {"model_version": version, "data_date": day, "drives_on_data_date": drives, "score_rows": rows}


def fleet_risk(cur, day, top, model=None):
    """The `top` riskiest drives on one day, riskiest first. rank is the rank across the whole fleet."""
    sql = ("SELECT rank_in_day, serial_number, model, risk_score, risk_band "
           "FROM fleet_scores WHERE date = ?")
    params = [day]
    if model:
        sql += " AND model = ?"
        params.append(model)
    sql += f" ORDER BY rank_in_day LIMIT {int(top)}"  # top is validated as an integer 1..1000 by the API
    return cur.execute(sql, params).fetchall()


def drive_history(cur, serial):
    """One row per scored day for a drive, with a few SMART values from the same day, oldest first."""
    return cur.execute(
        "SELECT s.date, s.model, s.risk_score, s.risk_band, s.rank_in_day, "
        "       f.smart_5_raw, f.smart_187_raw, f.smart_188_raw, f.smart_197_raw, f.temp_c "
        "FROM fleet_scores s "
        "LEFT JOIN features_scoring f ON f.serial_number = s.serial_number AND f.date = s.date "
        "WHERE s.serial_number = ? ORDER BY s.date",
        [serial],
    ).fetchall()
