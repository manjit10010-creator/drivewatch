"""Build engineered features for DriveWatch (Phase 3) into the gold DuckDB.

One row per drive per day (the same rows as labels_h<H>, from --start on). Every feature uses only
information on or before the scoring day: all windows end at CURRENT ROW, and are DATE-based
(last N calendar days, via an integer day number), so the 2025-05-12 gap cannot shift a window.

Feature families (see section 8 of the plan):
  raw state      smart_{5,187,188,197,198,199,9,12,193}_raw, temp_c
  rolling stats  temp_c mean / max / std over 7, 14, 30 days
  change         delta7_/delta30_ of each error counter (today minus the earliest value in the window)
  event counters days_since_first_nonzero_ and err_days30_ (days in last 30 where the counter rose)
  static         model, manufacturer, power_on_days, days_in_data
  missingness    miss_smart_<n> flags (raw value not reported); counters are filled with 0 plus the flag
  coverage       n_obs_30 (rows present in the last 30 days, shows gaps / new drives)

Choices that are mine, not documented facts:
  - temp_c = low byte of smart_194_raw (same rule as quality_checks.py); values outside 0..100 become NULL.
  - days_since_first_nonzero_* only sees data from 2025-01-01, so it is left-censored for older drives.

Run from the repo root, inside the drivewatch-spark Docker image:
    python src/features/build_features.py                 (dev sample, horizon 30, rows from 2025-02-01)
    python src/features/build_features.py --horizon 7     (writes features_daily_h7)
Needs labels_h<H> and excluded_drives from build_labels.py.
"""
import argparse
import os
import time

import duckdb

DB_PATH = "data/gold/drivewatch.duckdb"
TMP_DIR = "data/duckdb_tmp"
DEV_MODELS = ["TOSHIBA MG07ACA14TA", "ST12000NM0008", "TOSHIBA MG08ACA16TA"]
COUNTERS = [5, 187, 188, 197, 198, 199]   # error counters that only grow
OTHER_RAW = [9, 12, 193]                  # power-on hours, power cycles, load cycles
TEMP_ATTR = 194


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--start", default="2025-02-01", help="first output date (earlier days are history only)")
    p.add_argument("--all-models", action="store_true")
    return p.parse_args()


def col(n):
    return f"smart_{n}_raw"


def build_sql(h, start, model_filter):
    labels = f"labels_h{h}"
    out = "features_daily" if h == 30 else f"features_daily_h{h}"

    base_cols = []
    for n in COUNTERS:
        base_cols.append(f"COALESCE(CAST({col(n)} AS DOUBLE), 0) AS c{n}")
        base_cols.append(f"CAST({col(n)} IS NULL AS INTEGER) AS miss_smart_{n}")
    for n in OTHER_RAW:
        base_cols.append(f"CAST({col(n)} AS DOUBLE) AS r{n}")
        base_cols.append(f"CAST({col(n)} IS NULL AS INTEGER) AS miss_smart_{n}")
    base_cols.append(
        f"CASE WHEN (CAST({col(TEMP_ATTR)} AS BIGINT) & 255) BETWEEN 0 AND 100 "
        f"THEN CAST(CAST({col(TEMP_ATTR)} AS BIGINT) & 255 AS DOUBLE) END AS temp"
    )
    base_cols.append(f"CAST({col(TEMP_ATTR)} IS NULL AS INTEGER) AS miss_smart_{TEMP_ATTR}")

    base = f"""
    CREATE OR REPLACE TEMP TABLE base AS
    SELECT CAST(date AS DATE) AS date,
           CAST(date AS DATE) - DATE '2000-01-01' AS day_num,
           serial_number, model,
           {", ".join(base_cols)}
    FROM fact_daily_smart
    WHERE {model_filter}
      AND serial_number NOT IN (SELECT serial_number FROM excluded_drives)
      AND CAST(date AS DATE) <= (SELECT MAX(date) FROM {labels})
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

    feats = []
    for n in COUNTERS:
        feats.append(f"CAST(c{n} AS DOUBLE) AS smart_{n}_raw")
        feats.append(f"CAST(c{n} - first_value(c{n}) OVER w7 AS DOUBLE) AS delta7_smart_{n}")
        feats.append(f"CAST(c{n} - first_value(c{n}) OVER w30 AS DOUBLE) AS delta30_smart_{n}")
        feats.append(f"CAST(SUM(inc{n}) OVER w30 AS INTEGER) AS err_days30_smart_{n}")
        feats.append(
            f"date_diff('day', MIN(CASE WHEN c{n} > 0 THEN date END) OVER wall, date) "
            f"AS days_since_first_nonzero_smart_{n}"
        )
    for n in OTHER_RAW:
        feats.append(f"r{n} AS smart_{n}_raw")
    feats.append("temp AS temp_c")
    for w in (7, 14, 30):
        feats.append(f"CAST(AVG(temp) OVER w{w} AS REAL) AS temp_mean_{w}")
        feats.append(f"CAST(MAX(temp) OVER w{w} AS REAL) AS temp_max_{w}")
        feats.append(f"CAST(STDDEV_SAMP(temp) OVER w{w} AS REAL) AS temp_std_{w}")
    feats.append("CAST(COUNT(*) OVER w30 AS INTEGER) AS n_obs_30")
    feats.append("CAST(day_num - MIN(day_num) OVER (PARTITION BY serial_number) AS INTEGER) AS days_in_data")
    feats.append("CAST(r9 / 24.0 AS REAL) AS power_on_days")
    feats.append(
        "CASE WHEN model LIKE 'TOSHIBA%' THEN 'TOSHIBA' WHEN model LIKE 'ST%' THEN 'SEAGATE' "
        "WHEN model LIKE 'WDC%' OR model LIKE 'WUH%' THEN 'WDC' WHEN model LIKE 'HGST%' THEN 'HGST' "
        "ELSE 'OTHER' END AS manufacturer"
    )
    for n in COUNTERS + OTHER_RAW + [TEMP_ATTR]:
        feats.append(f"miss_smart_{n}")

    final = f"""
    CREATE OR REPLACE TABLE {out} AS
    WITH f AS (
        SELECT date, serial_number, model, {", ".join(feats)}
        FROM step
        WINDOW
          w7   AS (PARTITION BY serial_number ORDER BY day_num RANGE BETWEEN 6  PRECEDING AND CURRENT ROW),
          w14  AS (PARTITION BY serial_number ORDER BY day_num RANGE BETWEEN 13 PRECEDING AND CURRENT ROW),
          w30  AS (PARTITION BY serial_number ORDER BY day_num RANGE BETWEEN 29 PRECEDING AND CURRENT ROW),
          wall AS (PARTITION BY serial_number ORDER BY day_num ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    )
    SELECT f.*, l.days_to_failure, l.label
    FROM f JOIN {labels} l USING (date, serial_number)
    WHERE f.date >= CAST('{start}' AS DATE)
    """
    return base, step, final, out, labels


def main():
    args = parse_args()
    t0 = time.time()
    os.makedirs(TMP_DIR, exist_ok=True)
    con = duckdb.connect(DB_PATH)
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=8")
    con.execute(f"SET temp_directory='{TMP_DIR}'")

    cols = {r[0] for r in con.execute("DESCRIBE fact_daily_smart").fetchall()}
    needed = [col(n) for n in COUNTERS + OTHER_RAW + [TEMP_ATTR]]
    missing = [c for c in needed if c not in cols]
    if missing:
        print("These columns are not in fact_daily_smart:", missing)
        print("Columns that are:", sorted(cols))
        raise SystemExit(1)

    if args.all_models:
        model_filter = "TRUE"
    else:
        model_filter = "model IN (" + ", ".join(f"'{m}'" for m in DEV_MODELS) + ")"

    base, step, final, out, labels = build_sql(args.horizon, args.start, model_filter)

    print("1/3 base table (typed columns, filled counters, missing flags)...")
    con.execute(base)
    print(f"    {con.execute('SELECT COUNT(*) FROM base').fetchone()[0]:,} rows  ({time.time() - t0:.0f}s)")
    print("2/3 counter-increase flags...")
    con.execute(step)
    print(f"3/3 windows and write to {out} (the slow step; can take several minutes)...")
    con.execute(final)

    n, pos, drives = con.execute(f"SELECT COUNT(*), SUM(label), COUNT(DISTINCT serial_number) FROM {out}").fetchone()
    expected = con.execute(f"SELECT COUNT(*) FROM {labels} WHERE date >= CAST('{args.start}' AS DATE)").fetchone()[0]
    ncols = len(con.execute(f"DESCRIBE {out}").fetchall())
    print(f"\n{out}: {n:,} rows, {drives:,} drives, {ncols} columns, {pos:,} positives ({pos / n:.4%})")
    print(f"Rows expected from {labels} (date >= {args.start}): {expected:,}  -> {'OK' if n == expected else 'MISMATCH'}")

    # Independent recomputation on 300 random rows: windows must end on the scoring day.
    bad = con.execute(f"""
        WITH s AS (SELECT serial_number, date FROM {out} USING SAMPLE 300 ROWS)
        SELECT COUNT(*) AS checked,
               SUM(CASE WHEN f.temp_mean_7 IS NULL AND c.m IS NULL THEN 0
                        WHEN abs(f.temp_mean_7 - c.m) < 0.01 THEN 0 ELSE 1 END) AS bad_temp_mean_7,
               SUM(CASE WHEN abs(f.delta7_smart_5 - c.d) < 0.001 THEN 0 ELSE 1 END) AS bad_delta7_smart_5
        FROM {out} f JOIN s USING (serial_number, date)
        JOIN LATERAL (
            SELECT
              (SELECT AVG(CASE WHEN (CAST(g.smart_194_raw AS BIGINT) & 255) BETWEEN 0 AND 100
                               THEN CAST(g.smart_194_raw AS BIGINT) & 255 END)
               FROM fact_daily_smart g
               WHERE g.serial_number = f.serial_number
                 AND CAST(g.date AS DATE) BETWEEN f.date - 6 AND f.date) AS m,
              COALESCE(CAST(f.smart_5_raw AS DOUBLE), 0) - COALESCE((
                SELECT CAST(g.smart_5_raw AS DOUBLE) FROM fact_daily_smart g
                WHERE g.serial_number = f.serial_number
                  AND CAST(g.date AS DATE) BETWEEN f.date - 6 AND f.date
                ORDER BY g.date LIMIT 1), 0) AS d
        ) c ON TRUE
    """).fetchone()
    print(f"Window check on {bad[0]} random rows: temp_mean_7 mismatches = {bad[1]}, delta7_smart_5 mismatches = {bad[2]} (want 0 and 0)")

    print("\nShare of NULL values for the main features (NULL temp stats are expected for drives that do not report 194):")
    for c in ["temp_c", "temp_mean_30", "temp_std_30", "days_since_first_nonzero_smart_5", "days_since_first_nonzero_smart_197", "power_on_days"]:
        share = con.execute(f"SELECT AVG(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0.0 END) FROM {out}").fetchone()[0]
        print(f"  {c:<40} {share:.1%}")
    print("\nPositives vs negatives, mean of two features:")
    for lab, cnt, d197, t in con.execute(
        f"SELECT label, COUNT(*), AVG(delta30_smart_197), AVG(temp_mean_30) FROM {out} GROUP BY label ORDER BY label"
    ).fetchall():
        print(f"  label={lab}: rows {cnt:,}  mean delta30_smart_197 = {d197:.3f}  mean temp_mean_30 = {t:.2f}")
    print(f"\nDone in {time.time() - t0:.0f}s")
    con.close()


if __name__ == "__main__":
    main()
