"""Shared data loading and preprocessing for DriveWatch models (Phase 3 baselines, Phase 4 boosting).

Splits are by TIME with a 30-day gap, never random:
    train : 2025-02-01 .. 2025-11-30
    (gap) : December 2025 (31 days; a train row's label looks up to 30 days ahead)
    val   : 2026-01-01 .. 2026-03-31
    (gap) : April 2026 (30 days)
    test  : 2026-05-01 .. 2026-05-31   <- NOT used anywhere in this project until the final report.
Nothing in this file or in baselines.py reads the test period.
"""
import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/gold/drivewatch.duckdb"
TRAIN = ("2025-02-01", "2025-11-30")
VAL = ("2026-01-01", "2026-03-31")
META = {"date", "serial_number", "model", "manufacturer", "days_to_failure", "label"}


def connect():
    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=8")
    return con


def feature_columns(con, table):
    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    return [c for c in cols if c not in META]


def chunk_ranges(start, end, days=10):
    """Split [start, end] into windows of `days` days, so a chunk is about 1M rows."""
    out, cur, last = [], pd.Timestamp(start), pd.Timestamp(end)
    while cur <= last:
        stop = min(cur + pd.Timedelta(days=days - 1), last)
        out.append((cur.date().isoformat(), stop.date().isoformat()))
        cur = stop + pd.Timedelta(days=1)
    return out


def load_train_sample(con, table, feats, neg_frac, exclude_group=False):
    """All positive rows plus a deterministic random share of negative rows (undersampling).
    exclude_group=True leaves out the drives with hash(serial) % 5 = 0 (used by the grouped split)."""
    keep = int(round(neg_frac * 1000))
    grp = "AND hash(serial_number) % 5 <> 0" if exclude_group else ""
    sql = f"""
        SELECT date, serial_number, model, label, {", ".join(feats)}
        FROM {table}
        WHERE date BETWEEN DATE '{TRAIN[0]}' AND DATE '{TRAIN[1]}' {grp}
          AND (label = 1 OR hash(serial_number || CAST(date AS VARCHAR)) % 1000 < {keep})
    """
    return con.execute(sql).df()


def load_chunk(con, table, feats, start, end, only_group=False):
    """All rows (no sampling) between two dates; only_group=True keeps just hash(serial) % 5 = 0 drives."""
    grp = "AND hash(serial_number) % 5 = 0" if only_group else ""
    sql = f"""
        SELECT date, serial_number, model, label, {", ".join(feats)}
        FROM {table}
        WHERE date BETWEEN DATE '{start}' AND DATE '{end}' {grp}
    """
    return con.execute(sql).df()


def day_numbers(df):
    return (pd.to_datetime(df["date"]) - pd.Timestamp("1970-01-01")).dt.days.to_numpy().astype("int32")


def serial_hashes(df):
    return pd.util.hash_pandas_object(df["serial_number"], index=False).to_numpy()


class Preprocessor:
    """Turns the feature table into a numeric matrix for linear models.
    Counts and deltas: negative values clipped to 0, then log1p. Missing values: the TRAIN median.
    days_since_first_nonzero_*: a 'has it happened' flag plus log1p of the days (0 if never).
    Everything is standardised with TRAIN mean and std. Model name becomes one-hot columns."""

    def __init__(self, feats):
        self.since = [c for c in feats if c.startswith("days_since_first_nonzero_")]
        self.flags = [c for c in feats if c.startswith("miss_smart_")]
        self.temp = [c for c in feats if c.startswith("temp_")]
        self.logc = [c for c in feats if c not in self.since + self.flags + self.temp]
        self.names = []

    def fit(self, df):
        self.models = sorted(df["model"].unique())
        self.medians = {}
        for c in self.logc + self.temp:
            m = df[c].astype("float64").median()
            self.medians[c] = 0.0 if pd.isna(m) else float(m)
        X = self._raw(df)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def _raw(self, df):
        cols, names = [], []
        for c in self.logc:
            v = df[c].astype("float64").fillna(self.medians[c]).clip(lower=0).to_numpy()
            cols.append(np.log1p(v)); names.append(f"log1p({c})")
        for c in self.temp:
            cols.append(df[c].astype("float64").fillna(self.medians[c]).to_numpy()); names.append(c)
        for c in self.since:
            s = df[c].astype("float64")
            cols.append(s.notna().astype("float64").to_numpy()); names.append(f"has_{c[len('days_since_'):]}")
            cols.append(np.log1p(s.fillna(0).clip(lower=0).to_numpy())); names.append(f"log1p({c})")
        for c in self.flags:
            cols.append(df[c].astype("float64").to_numpy()); names.append(c)
        for m in self.models:
            cols.append((df["model"] == m).astype("float64").to_numpy()); names.append(f"model={m}")
        self.names = names
        return np.column_stack(cols)

    def transform(self, df):
        return ((self._raw(df) - self.mean) / self.std).astype("float32")
