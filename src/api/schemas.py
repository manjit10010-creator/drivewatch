"""Pydantic schemas: the shape of every response the DriveWatch API returns.

FastAPI uses these to validate what we send, to document the API at /docs, and to turn Python objects
into JSON. A wrong type here fails loudly in the server instead of silently sending bad data.
"""
import datetime as dt
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

RiskBand = Literal["low", "medium", "high"]


class HealthResponse(BaseModel):
    # pydantic reserves the "model_" prefix; we have a field called model_version on purpose
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_version: int
    data_date: dt.date
    drives_on_data_date: int
    score_rows: int


class FleetDrive(BaseModel):
    rank: int
    serial_number: str
    model: str
    risk_score: float
    risk_band: RiskBand


class FleetRiskResponse(BaseModel):
    date: dt.date
    model_filter: Optional[str]
    count: int
    drives: list[FleetDrive]


class DriveDay(BaseModel):
    date: dt.date
    risk_score: float
    risk_band: RiskBand
    rank_in_day: int
    smart_5_raw: Optional[float]
    smart_187_raw: Optional[float]
    smart_188_raw: Optional[float]
    smart_197_raw: Optional[float]
    temp_c: Optional[float]


class DriveResponse(BaseModel):
    serial_number: str
    model: str
    latest_date: dt.date
    latest_score: float
    latest_band: RiskBand
    latest_rank: int
    history: list[DriveDay]
