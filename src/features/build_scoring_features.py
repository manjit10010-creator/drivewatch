"""Build features for SCORING (no labels) into the gold DuckDB.

features_daily only holds rows whose 30-day label is known (scoring dates up to 2026-05-31). To score
the fleet today (June 2026 and later) we need the same features for recent days, without labels.

This script computes ONLY the columns the production model uses (reports/selected_features_25.txt),
for scoring dates >= --start, using the same formulas and the same DATE-based windows as
build_features.py. It reads a little history before --start (--lookback days) so the windows are full.

It refuses features that need the whole history (days_in_data, days_since_first_nonzero_*): those
are left-censored and were dropped from the model.

If features_daily already has rows for some of the same days, it recomputes them and compares every
column (a train/serve consistency check; the mismatch counts should be 0).

Run from the repo root, inside the drivewatch-spark Docker image:
    python src/features/build_scoring_features.py                      (June 2026 onwards -> features_scoring)
    python src/features/build_scoring_features.py --start 2026-04-01 --end 2026-04-30 --table features_scoring_check
"""
import argparse
import os
import re
import time

import duckdb

from build_features import COUNTERS, DB_PATH, DEV_MODELS, OTHER_RAW, TEMP_ATTR, TMP_DIR, col


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2026-06-01", help="first scoring date to output")
    p.add_argument("--end", default=None, help="last scoring date to output (default: last day in the data)")
    p.add_argument("--lookback", type=int, default=120, help="days of history read before --start")
    p.add_argument("--features-file", default="reports/selected_features_25.txt")
    p.add_argument("--table", default="features_scoring")
    p.add_argument("--all-models", action="store_true")
    return p.parse_args()


def read_feature_names(path):
    names = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.extend(t for t in re.split(r"[\s,]+", line) if t)
    return names


def feature_expressions():
    """Feature name -> SQL expression over the 'step' table. Same formulas as build_features.py."""
    f = {}
    for n in COUNTERS:
        f[f"smart_{n}_raw"] = f"CAST(c{n} AS DOUBLE)"
        f[f"delta7_smart_{n}"] = f"CAST(c{n} - first_value(c{n}) OVER w7 AS DOUBLE)"
        f[f"delta30_smart_{n}"] = f"CAST(c{n} - first_value(c{n}) OVER w30 AS DOUBLE)"
        f[f"err_days30_smart_{n}"] = f"CAST(SUM(inc{n}) OVER w30 AS INTEGER)"
    for n in OTHER_RAW:
        f[f"smart_{n}_raw"] = f"r{n}"
    f["temp_c"] = "temp"
    for w in (7, 14, 30):
        f[f"temp_mean_{w}"] = f"CAST(AVG(temp) OVER w{w} AS REAL)"
        f[f"temp_max_{w}"] = f"CAST(MAX(temp) OVER w{w} AS REAL)"
        f[f"temp_std_{w}"] = f"CAST(STDDEV_SAMP(temp) OVER w{w} AS REAL)"
    f["n_obs_30"] = "CAST(COUNT(*) OVER w30 AS INTEGER)"
    f["power_on_days"] = "CAST(r9 / 24.0 AS REAL)"
    for n in COUNTERS + OTHER_RAW + [TEMP_ATTR]:
        f[f"miss_smart_{n}"] = f"miss_smart_{n}"
    return f


WINDOWS = {
    "w7": "PARTITION BY serial_number ORDER BY day_num RANGE BETWEEN 6 PRECEDING AND CURRENT ROW",
    "w14": "PARTITION BY serial_number ORDER BY day_num RANGE BETWEEN 13 PRECEDING AND CURRENT ROW",
    "w30": "PARTITION BY serial_number ORDER BY day_num RANGE BETWEEN 29 PRECEDING AND CURRENT ROW",
}


def base_columns():
    cols = []
    for n in COUNTERS:
        cols.append(f"COALESCE(CAST({col(n)} AS DOUBLE), 0) AS c{n}")
        cols.append(f"CAST({col(n)} IS NULL AS INTEGER) AS miss_smart_{n}")
    for n in OTHER_RAW:
        cols.append(f"CAST({col(n)} AS DOUBLE) AS r{n}")
        cols.append(f"CAST({col(n)} IS NULL AS INTEGER) AS miss_smart_{n}")
    cols.append(
        f"CASE WHEN (CAST({col(TEMP_ATTR)} AS BIGINT) & 255) BETWEEN 0 AND 100 "
        f"THEN CAST(CAST({col(TEMP_ATTR)} AS BIGINT) & 255 AS DOUBLE) END AS temp"
    )
    cols.append(f"CAST({col(TEMP_ATTR)} IS NULL AS INTEGER) AS miss_smart_{TEMP_ATTR}")
    return cols


def main():
    args = parse_args()
    t0 = time.time()

    wanted = read_feature_names(args.features_file)
    exprs = feature_expressions()
    unknown = [n for n in wanted if n not in exprs]
    if unknown:
        print("These features cannot be built in scoring mode:", unknown)
        print("(days_in_data and days_since_first_nonzero_* need the full history and were dropped from the model.)")
        print("Buildable features:", sorted(exprs))
        raise SystemExit(1)
    print(f"{len(wanted)} features from {args.features_file}")

    os.makedirs(TMP_DIR, exist_ok=True)
    con = duckdb.connect(DB_PATH)
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=8")
    con.execute(f"SET temp_directory='{TMP_DIR}'")

    last_day = con.execute("SELECT MAX(CAST(date AS DATE)) FROM fact_daily_smart").fetchone()[0]
    end = args.end or str(last_day)
    print(f"Scoring dates {args.start} to {end}  (last day in fact_daily_smart: {last_day}, lookback {args.lookback} days)")

    if args.all_models:
        model_filter = "TRUE"
    else:
        model_filter = "model IN (" + ", ".join(f"'{m}'" for m in DEV_MODELS) + ")"

    base = f"""
    CREATE OR REPLACE TEMP TABLE base AS
    SELECT CAST(date AS DATE) AS date,
           CAST(date AS DATE) - DATE '2000-01-01' AS day_num,
           serial_number, model,
           {", ".join(base_columns())}
    FROM fact_daily_smart
    WHERE {model_filter}
      AND serial_number NOT IN (SELECT serial_number FROM excluded_drives)
      AND CAST(date AS DATE) >= CAST('{args.start}' AS DATE) - {args.lookback}
      AND CAST(date AS DATE) <= CAST('{end}' AS DATE)
    """
    inc_cols = ", ".join(
        f"CASE WHEN c{n} > LAG(c{n}) OVER o THEN 1 ELSE 0 END AS inc{n}" for n in COUNTERS
    )
    step = f"""
    CREATE OR REPLACE TEMP TABLE step AS
    SELECT *, {inc_cols}
    FROM base
    WINDOW o AS (PARTITION BY serial_number ORDER BY day_num)
    """
    select_list = ", ".join(f"{exprs[n]} AS {n}" for n in wanted)
    # only declare the windows the chosen features actually use
    used =[w for w in WINDOWS if re.search(rf"OVER {w}\b", select_list)]
    window_clause = ("WINDOW " + ", ".join(f"{w} AS ({WINDOWS[w]})" for w in used)) if used else ""
    final = f"""
    CREATE OR REPLACE TABLE {args.table} AS
    WITH f AS (
        SELECT date, serial_number, model, {select_list}
        FROM step
        {window_clause}
    )
    SELECT * FROM f WHERE date >= CAST('{args.start}' AS DATE)
    """

    print("1/3 base table...")
    con.execute(base)
    print(f"    {con.execute('SELECT COUNT(*) FROM base').fetchone()[0]:,} rows read  ({time.time() - t0:.0f}s)")
    print("2/3 counter-increase flags...")
    con.execute(step)
    print(f"3/3 windows and write to {args.table}...")
    con.execute(final)

    n, drives, d0, d1, nd = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT serial_number), MIN(date), MAX(date), COUNT(DISTINCT date) FROM {args.table}"
    ).fetchone()
    print(f"\n{args.table}: {n:,} rows, {drives:,} drives, {len(wanted)} features, dates {d0} to {d1} ({nd} days)")

    print("\nShare of NULL values per feature (temperature stats can be NULL for drives that do not report 194):")
    for name in wanted:
        share = con.execute(f"SELECT AVG(CASE WHEN {name} IS NULL THEN 1.0 ELSE 0.0 END) FROM {args.table}").fetchone()[0]
        if share and share > 0:
            print(f"  {name:<28} {share:.1%}")

    # Train/serve consistency: same days in features_daily must match column by column.
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "features_daily" in tables:
        overlap = con.execute(
            f"SELECT COUNT(*) FROM {args.table} s JOIN features_daily f USING (date, serial_number)"
        ).fetchone()[0]
        if overlap == 0:
            print("\nNo days in common with features_daily, so no consistency check was possible.")
        else:
            sc = con.execute(
                f"SELECT COUNT(*) FROM {args.table} WHERE date <= (SELECT MAX(date) FROM features_daily)"
            ).fetchone()[0]
            fc = con.execute(
                f"SELECT COUNT(*) FROM features_daily WHERE date BETWEEN CAST('{args.start}' AS DATE) "
                f"AND CAST('{end}' AS DATE)"
            ).fetchone()[0]
            print(f"\nConsistency check against features_daily: {overlap:,} drive-days in common "
                  f"(scoring rows in range {sc:,}, features_daily rows in range {fc:,})")
            cases = ", ".join(
                f"SUM(CASE WHEN (s.{c} IS NULL) <> (f.{c} IS NULL) OR abs(s.{c} - f.{c}) > 0.001 "
                f"THEN 1 ELSE 0 END) AS {c}" for c in wanted
            )
            row = con.execute(
                f"SELECT {cases} FROM {args.table} s JOIN features_daily f USING (date, serial_number)"
            ).fetchone()
            bad = {c: int(v or 0) for c, v in zip(wanted, row)}
            total = sum(bad.values())
            print(f"Mismatching values over all {len(wanted)} features: {total} (want 0)")
            for c, v in bad.items():
                if v:
                    print(f"  {c:<28} {v:,} mismatching rows")

    print(f"\nDone in {time.time() - t0:.0f}s")
    con.close()


if __name__ == "__main__":
    main()
