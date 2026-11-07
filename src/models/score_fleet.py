"""Score the fleet with the PRODUCTION model from the MLflow registry and write the results to DuckDB.

Reads the 25 model features from a features table (default: features_scoring, built by
build_scoring_features.py), scores every drive-day with models:/drivewatch-risk@production, assigns a
risk band from reports/risk_bands.json (made by risk_bands.py) and writes table fleet_scores:

    date, serial_number, model, model_version, risk_score, risk_band, rank_in_day

rank_in_day = 1 is the riskiest drive that day. This is also step 4 of the daily pipeline, and the API
reads fleet_scores instead of scoring 95,000 drives on every request.

Before scoring it checks that the registry model gives exactly the same predictions as the model file
data/models/final_s1.txt on the first chunk (it should: the registry holds that file).

Run from the repo root, inside the drivewatch-ml Docker image:
    python src/models/score_fleet.py
    python src/models/score_fleet.py --start 2026-06-30 --end 2026-06-30      (one day only)
"""
import argparse
import json
import time
from datetime import date, timedelta

import duckdb
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from train_lgbm import load_feature_list

DB_PATH = "data/gold/drivewatch.duckdb"
TRACKING_URI = "sqlite:////workspace/mlflow.db"
MODEL_NAME = "drivewatch-risk"
MODEL_URI = f"models:/{MODEL_NAME}@production"
THREADS = 12


def load_production_model():
    """Return (booster, version number) of the registry model with alias 'production'."""
    mlflow.set_tracking_uri(TRACKING_URI)
    booster = mlflow.lightgbm.load_model(MODEL_URI)
    version = int(MlflowClient().get_model_version_by_alias(MODEL_NAME, "production").version)
    return booster, version


def date_chunks(d0, d1, step=10):
    cur = d0
    while cur <= d1:
        end = min(cur + timedelta(days=step - 1), d1)
        yield cur, end
        cur = end + timedelta(days=1)


def read_chunk(con, table, cols, d0, d1):
    sql = f"SELECT {', '.join(cols)} FROM {table} WHERE date BETWEEN DATE '{d0}' AND DATE '{d1}'"
    return con.execute(sql).df()


def predict(booster, df, feats):
    """NULLs become NaN, which LightGBM handles natively (same as in training)."""
    x = df[feats].to_numpy(dtype="float64", na_value=np.nan)
    return booster.predict(x, num_threads=THREADS)


def assign_band(scores, bands):
    return np.where(scores >= bands["high"], "high", np.where(scores >= bands["medium"], "medium", "low"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="features_scoring")
    p.add_argument("--out-table", default="fleet_scores")
    p.add_argument("--start", default=None, help="first date to score (default: first date in the table)")
    p.add_argument("--end", default=None, help="last date to score (default: last date in the table)")
    p.add_argument("--bands", default="reports/risk_bands.json")
    p.add_argument("--features-file", default="reports/selected_features_25.txt")
    p.add_argument("--check-file", default="data/models/final_s1.txt")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    booster, version = load_production_model()
    feats = booster.feature_name()
    file_feats = load_feature_list(args.features_file)
    if feats != file_feats:
        raise SystemExit(f"Registry model features differ from {args.features_file}; stopping.\n"
                         f"model: {feats}\nfile:  {file_feats}")
    print(f"Loaded {MODEL_URI} = version {version}, {len(feats)} features  ({time.time() - t0:.0f}s)")

    with open(args.bands) as f:
        bands = json.load(f)["thresholds"]
    print(f"Band thresholds: high >= {bands['high']:.4f}, medium >= {bands['medium']:.4f}")

    con = duckdb.connect(DB_PATH)
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=8")
    lo, hi = con.execute(f"SELECT MIN(CAST(date AS DATE)), MAX(CAST(date AS DATE)) FROM {args.table}").fetchone()
    d0 = date.fromisoformat(args.start) if args.start else lo
    d1 = date.fromisoformat(args.end) if args.end else hi
    print(f"Scoring {args.table} from {d0} to {d1}  (table holds {lo} to {hi})")

    cols = ["date", "serial_number", "model"] + feats
    parts = []
    checked = False
    for a, b in date_chunks(d0, d1):
        df = read_chunk(con, args.table, cols, a, b)
        if df.empty:
            continue
        scores = predict(booster, df, feats)
        if not checked:
            try:
                ref = lgb.Booster(model_file=args.check_file)
                diff = float(np.max(np.abs(ref.predict(df[feats].to_numpy(dtype="float64", na_value=np.nan),
                                                       num_threads=THREADS) - scores)))
                print(f"Registry model vs {args.check_file} on {len(df):,} rows: max abs difference = {diff:.2e} (want 0)")
            except Exception as exc:  # the check is a bonus; do not stop scoring because of it
                print(f"Could not compare with {args.check_file}: {exc}")
            checked = True
        parts.append(pd.DataFrame({
            "date": df["date"],
            "serial_number": df["serial_number"],
            "model": df["model"],
            "model_version": version,
            "risk_score": scores.astype("float32"),
            "risk_band": assign_band(scores, bands),
        }))
        print(f"  scored {a}..{b}: {len(df):,} rows  ({time.time() - t0:.0f}s)")

    if not parts:
        raise SystemExit("No rows were scored; check the date range and table.")
    out = pd.concat(parts, ignore_index=True)
    con.register("scores_df", out)
    con.execute(f"""
        CREATE OR REPLACE TABLE {args.out_table} AS
        SELECT CAST(date AS DATE) AS date, serial_number, model, model_version, risk_score, risk_band,
               CAST(row_number() OVER (PARTITION BY date ORDER BY risk_score DESC, serial_number) AS INTEGER)
                 AS rank_in_day
        FROM scores_df
    """)
    con.unregister("scores_df")

    n, drives, days = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT serial_number), COUNT(DISTINCT date) FROM {args.out_table}").fetchone()
    print(f"\n{args.out_table}: {n:,} rows, {drives:,} drives, {days} days")
    q = con.execute(f"SELECT quantile_cont(risk_score, [0.5, 0.9, 0.99, 0.999]), MAX(risk_score) FROM {args.out_table}").fetchone()
    print(f"Score quantiles: median {q[0][0]:.4f}, p90 {q[0][1]:.4f}, p99 {q[0][2]:.4f}, p99.9 {q[0][3]:.4f}, max {q[1]:.4f}")

    last = con.execute(f"SELECT MAX(date) FROM {args.out_table}").fetchone()[0]
    print(f"\nBand counts on the last day ({last}):")
    for band, cnt in con.execute(
            f"SELECT risk_band, COUNT(*) FROM {args.out_table} WHERE date = ? GROUP BY 1 ORDER BY 2 DESC", [last]).fetchall():
        print(f"  {band:<8} {cnt:,}")
    print(f"\nTop 15 riskiest drives on {last}:")
    print(f"  {'rank':>4}  {'serial_number':<18} {'model':<22} {'score':>7}  band")
    for r in con.execute(
            f"SELECT rank_in_day, serial_number, model, risk_score, risk_band FROM {args.out_table} "
            f"WHERE date = ? ORDER BY rank_in_day LIMIT 15", [last]).fetchall():
        print(f"  {r[0]:>4}  {r[1]:<18} {r[2]:<22} {r[3]:>7.4f}  {r[4]}")
    print(f"\nDone in {time.time() - t0:.0f}s")
    con.close()


if __name__ == "__main__":
    main()
