"""Alert feedback for POST /feedback and GET /feedback (step 5d).

Ops mark an alert as a true or a false positive. Those answers are the labelled data for the next retraining,
so they are kept in their own small SQLite file (data/feedback.db), NOT in the gold DuckDB: the API opens the
warehouse read-only, and the pipeline rewrites it, so nothing of ours should live there.

The log is append-only: a second answer for the same alert is a new row, and the newest row wins when the
feedback is used for retraining.
"""
import datetime as dt
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.schemas import RiskBand

DB_PATH = "data/feedback.db"
Verdict = Literal["true_positive", "false_positive"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    serial_number TEXT    NOT NULL,
    alert_date    TEXT    NOT NULL,
    verdict       TEXT    NOT NULL CHECK (verdict IN ('true_positive', 'false_positive')),
    model_version INTEGER NOT NULL,
    risk_band     TEXT    NOT NULL,
    note          TEXT
)
"""


class FeedbackRequest(BaseModel):
    serial_number: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    alert_date: dt.date = Field(description="The scoring date of the alert, e.g. 2026-06-30")
    verdict: Verdict
    note: Optional[str] = Field(None, max_length=500)


class FeedbackRecord(BaseModel):
    id: int
    created_at: str
    serial_number: str
    alert_date: dt.date
    verdict: Verdict
    model_version: int
    risk_band: RiskBand
    note: Optional[str]


def init_db(path: str = DB_PATH) -> None:
    """Create the file and the table if they do not exist yet. Safe to call on every start."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with closing(sqlite3.connect(path, timeout=5)) as con, con:
        con.execute(SCHEMA)


def insert_feedback(path: str, serial: str, alert_date: dt.date, verdict: str, model_version: int,
                    risk_band: str, note: Optional[str]) -> int:
    """Store one answer and return its id. A fresh connection per call keeps this safe across threads."""
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(sqlite3.connect(path, timeout=5)) as con, con:  # `with con` commits (or rolls back on error)
        cur = con.execute(
            "INSERT INTO feedback (created_at, serial_number, alert_date, verdict, model_version, risk_band, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (created, serial, alert_date.isoformat(), verdict, model_version, risk_band, note))
        return cur.lastrowid


def list_feedback(path: str, limit: int = 50) -> list:
    """The newest answers first."""
    with closing(sqlite3.connect(path, timeout=5)) as con:
        rows = con.execute(
            "SELECT id, created_at, serial_number, alert_date, verdict, model_version, risk_band, note "
            "FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [FeedbackRecord(id=r[0], created_at=r[1], serial_number=r[2], alert_date=r[3], verdict=r[4],
                           model_version=r[5], risk_band=r[6], note=r[7]) for r in rows]


router = APIRouter()


@router.post("/feedback", response_model=FeedbackRecord, status_code=201)
def post_feedback(body: FeedbackRequest, request: Request):
    """Mark an alert as a true or false positive. The alert must exist: that drive scored on that date."""
    con = request.app.state.con
    if con is None:
        raise HTTPException(status_code=503, detail="The warehouse could not be opened when the API started, "
                                                    "so the alert cannot be checked. See the README.")
    cur = con.cursor()
    try:
        row = cur.execute("SELECT model_version, risk_band FROM fleet_scores WHERE serial_number = ? AND date = ?",
                          [body.serial_number, body.alert_date]).fetchone()
    finally:
        cur.close()
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"Drive {body.serial_number} has no score on {body.alert_date}, so there is no alert to mark.")
    new_id = insert_feedback(DB_PATH, body.serial_number, body.alert_date, body.verdict, int(row[0]), row[1], body.note)
    return list_feedback_by_id(DB_PATH, new_id)


def list_feedback_by_id(path: str, row_id: int) -> FeedbackRecord:
    with closing(sqlite3.connect(path, timeout=5)) as con:
        r = con.execute(
            "SELECT id, created_at, serial_number, alert_date, verdict, model_version, risk_band, note "
            "FROM feedback WHERE id = ?", (row_id,)).fetchone()
    return FeedbackRecord(id=r[0], created_at=r[1], serial_number=r[2], alert_date=r[3], verdict=r[4],
                          model_version=r[5], risk_band=r[6], note=r[7])


@router.get("/feedback", response_model=list[FeedbackRecord])
def get_feedback(limit: int = Query(50, ge=1, le=500, description="How many answers to return, newest first")):
    """The stored answers, newest first (the alert-triage screen will use this)."""
    return list_feedback(DB_PATH, limit)
