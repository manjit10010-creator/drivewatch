"""DriveWatch API (step 5c: health, fleet ranking, one drive's history).

Run from the repo root (inside the drivewatch-api image, see docker/api/Dockerfile):
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000/docs for an interactive page that documents and tests every endpoint.

The API only reads the table fleet_scores (written by src/models/score_fleet.py) and features_scoring.
It does not load the model yet; /predict (step 5d) will.
"""
import datetime as dt
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse

from src.api import db
from src.api.schemas import DriveDay, DriveResponse, FleetDrive, FleetRiskResponse, HealthResponse

log = logging.getLogger("drivewatch.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
SLOW_MS = 500  # the service target from the project plan


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open the warehouse once at startup (read-only), close it at shutdown.
    app.state.con = db.open_db()
    log.info("Opened %s read-only", db.DB_PATH)
    yield
    app.state.con.close()


app = FastAPI(title="DriveWatch API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def time_requests(request: Request, call_next):
    """Add an X-Process-Time-ms header to every response and warn when a request is slower than the target."""
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    if ms > SLOW_MS:
        log.warning("slow request: %s %s took %.0f ms", request.method, request.url.path, ms)
    return response


@app.exception_handler(duckdb.Error)
async def warehouse_error(request: Request, exc: duckdb.Error):
    log.error("warehouse error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": "The warehouse could not answer this request."})


def get_cursor(request: Request):
    """One cursor per request: DuckDB lets many cursors share one connection safely."""
    cur = request.app.state.con.cursor()
    try:
        yield cur
    finally:
        cur.close()


def resolve_day(cur, requested: Optional[dt.date]) -> dt.date:
    lo, hi = db.date_range(cur)
    if lo is None:
        raise HTTPException(status_code=503, detail="fleet_scores is empty; run src/models/score_fleet.py first.")
    if requested is None:
        return hi
    if not (lo <= requested <= hi):
        raise HTTPException(status_code=404, detail=f"No scores for {requested}; available dates are {lo} to {hi}.")
    return requested


@app.get("/health", response_model=HealthResponse)
def health(cur=Depends(get_cursor)):
    """Liveness check: the model version that produced the scores and the date of the latest data."""
    info = db.health_info(cur)
    if info is None:
        raise HTTPException(status_code=503, detail="fleet_scores is empty; run src/models/score_fleet.py first.")
    return HealthResponse(status="ok", **info)


@app.get("/fleet/risk", response_model=FleetRiskResponse)
def fleet_risk(
    top: int = Query(50, ge=1, le=1000, description="How many drives to return"),
    model: Optional[str] = Query(None, max_length=64, description="Only this drive model, e.g. ST12000NM0008"),
    day: Optional[dt.date] = Query(None, alias="date", description="Scoring date, default is the latest day"),
    cur=Depends(get_cursor),
):
    """The riskiest drives on one day, riskiest first. `rank` is the rank across the whole fleet."""
    the_day = resolve_day(cur, day)
    rows = db.fleet_risk(cur, the_day, top, model)
    drives = [FleetDrive(rank=r[0], serial_number=r[1], model=r[2], risk_score=round(r[3], 4), risk_band=r[4])
              for r in rows]
    return FleetRiskResponse(date=the_day, model_filter=model, count=len(drives), drives=drives)


@app.get("/drive/{serial}", response_model=DriveResponse)
def drive(
    serial: str = Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
    cur=Depends(get_cursor),
):
    """One drive's score history and a few SMART values for each scored day."""
    rows = db.drive_history(cur, serial)
    if not rows:
        raise HTTPException(status_code=404, detail=f"Drive {serial} has no scores in the warehouse.")
    history = [
        DriveDay(date=r[0], risk_score=round(r[2], 4), risk_band=r[3], rank_in_day=r[4],
                 smart_5_raw=r[5], smart_187_raw=r[6], smart_188_raw=r[7], smart_197_raw=r[8], temp_c=r[9])
        for r in rows
    ]
    last = rows[-1]
    return DriveResponse(serial_number=serial, model=last[1], latest_date=last[0],
                         latest_score=round(last[2], 4), latest_band=last[3], latest_rank=last[4], history=history)
