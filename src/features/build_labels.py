"""Build the H-day failure label for DriveWatch (Phase 3).

Label rule: label = 1 if the drive's FIRST failure date falls within the next H days.
  days_to_failure = failure_date - scoring_date
  day-zero convention (--day-zero):
    positive : label = 1 when 0 <= days_to_failure <= H  (the failure-day row counts)
    negative : label = 1 when 1 <= days_to_failure <= H  (the failure-day row is dropped)

Rows removed:
  1. rows after the drive's first failure date
  2. rows whose full H-day window is not observed (date > last data date - H)
  3. all rows of drives that reappear after their first failure (rule computed from the
     data, written to the table excluded_drives with a reason)

Two modes (run from the repo root, inside the drivewatch-spark Docker image):
  Verify (read-only, writes nothing; checks the definition against the EDA class balance):
    python src/features/build_labels.py --verify --all-models --start 2026-01-01 --end 2026-06-30 --horizon 30
  Build (writes labels_h<H> and excluded_drives into the gold DuckDB; dev sample by default):
    python src/features/build_labels.py --horizon 30
"""
import argparse
import datetime as dt
import os
import time

import duckdb

DB_PATH = "data/gold/drivewatch.duckdb"
TMP_DIR = "data/duckdb_tmp"
DEV_MODELS = ["TOSHIBA MG07ACA14TA", "ST12000NM0008", "TOSHIBA MG08ACA16TA"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizon", type=int, default=30, help="days ahead (7, 14, 30)")
    p.add_argument("--start", default="2025-01-01", help="first scoring date to label")
    p.add_argument("--end", default="2026-06-30", help="last scoring date to label")
    p.add_argument("--all-models", action="store_true", help="use every model, not just the dev sample")
    p.add_argument("--day-zero", choices=["positive", "negative"], default="positive")
    p.add_argument("--verify", action="store_true", help="read-only check of class balance; writes nothing")
    p.add_argument("--ignore-cutoff", action="store_true",
                   help="verify only: do not drop the last H days (reproduces the EDA notebook numerator)")
    return p.parse_args()


def connect(read_only):
    os.makedirs(TMP_DIR, exist_ok=True)
    con = duckdb.connect(DB_PATH, read_only=read_only)
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=8")
    con.execute(f"SET temp_directory='{TMP_DIR}'")
    return con


def verify(con, args, cutoff):
    h = args.horizon
    sql = """
    WITH ff AS (
        SELECT serial_number, MIN(CAST(date AS DATE)) AS fd
        FROM fact_daily_smart WHERE failure = 1 GROUP BY serial_number
    )
    SELECT
        COUNT(*) AS rows_all,
        COUNT(*) FILTER (WHERE ff.fd IS NULL OR CAST(s.date AS DATE) <= ff.fd) AS rows_before_or_on_failure,
        COUNT(*) FILTER (WHERE ff.fd IS NOT NULL
            AND date_diff('day', CAST(s.date AS DATE), ff.fd) BETWEEN 0 AND ?) AS pos_day_zero_counts,
        COUNT(*) FILTER (WHERE ff.fd IS NOT NULL
            AND date_diff('day', CAST(s.date AS DATE), ff.fd) BETWEEN 1 AND ?) AS pos_day_zero_excluded
    FROM fact_daily_smart s LEFT JOIN ff USING (serial_number)
    WHERE CAST(s.date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
    """
    end = dt.date.fromisoformat(args.end)
    if not args.ignore_cutoff:
        end = min(end, cutoff)
    rows_all, rows_pre, pos0, pos1 = con.execute(sql, [h, h, args.start, end.isoformat()]).fetchone()
    note = "cutoff IGNORED (EDA-style)" if args.ignore_cutoff else "no exclusions applied"
    print(f"\nVERIFY  range {args.start} to {end}  horizon {h}  ({note})")
    print(f"  rows in range (all)                   : {rows_all:,}")
    print(f"  rows on/before first failure          : {rows_pre:,}")
    print(f"  positives, failure-day row counts     : {pos0:,}   ({pos0 / rows_all:.4%} of all rows, {pos0 / rows_pre:.4%} of pre-failure rows)")
    print(f"  positives, failure-day row excluded   : {pos1:,}   ({pos1 / rows_all:.4%} of all rows, {pos1 / rows_pre:.4%} of pre-failure rows)")
    print("\n  EDA reference (H1 2026, 30 days): 60,436 positives of 51,928,255 labelable rows (0.1164%).")
    print("  Whichever line matches that is the convention the EDA used.")


def build(con, args, cutoff, max_date):
    h = args.horizon
    lo = 0 if args.day_zero == "positive" else 1
    start, end = args.start, min(dt.date.fromisoformat(args.end), cutoff).isoformat()
    t0 = time.time()

    # First failure per drive, over the whole fleet (so the exclusion rule is fleet-wide).
    con.execute("""
        CREATE OR REPLACE TEMP TABLE first_fail AS
        SELECT serial_number, MIN(CAST(date AS DATE)) AS failure_date
        FROM fact_daily_smart WHERE failure = 1 GROUP BY serial_number
    """)
    n_failed = con.execute("SELECT COUNT(*) FROM first_fail").fetchone()[0]
    print(f"Drives with a failure row: {n_failed:,}  (handover expects 6,861)")

    # Rule from the data: any drive with rows after its first failure date.
    con.execute("""
        CREATE OR REPLACE TABLE excluded_drives AS
        SELECT f.serial_number,
               any_value(s.model)               AS model,
               f.failure_date,
               COUNT(*)                          AS rows_after_failure,
               MIN(CAST(s.date AS DATE))         AS first_date_after,
               MAX(CAST(s.date AS DATE))         AS last_date_after,
               'reported rows after first failure date' AS reason
        FROM fact_daily_smart s
        JOIN first_fail f USING (serial_number)
        WHERE CAST(s.date AS DATE) > f.failure_date
        GROUP BY f.serial_number, f.failure_date
    """)
    n_ex, rows_ex = con.execute("SELECT COUNT(*), COALESCE(SUM(rows_after_failure), 0) FROM excluded_drives").fetchone()
    print(f"Excluded drives: {n_ex:,} with {rows_ex:,} rows after failure  (handover: 37 drives, 5,157 rows)")

    if args.all_models:
        model_filter, params = "TRUE", []
    else:
        model_filter = "s.model IN (" + ",".join("?" * len(DEV_MODELS)) + ")"
        params = list(DEV_MODELS)

    last_ok = "CAST(s.date AS DATE) <= f.failure_date" if lo == 0 else "CAST(s.date AS DATE) < f.failure_date"
    table = f"labels_h{h}"
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT CAST(s.date AS DATE) AS date,
               s.serial_number,
               s.model,
               CASE WHEN f.failure_date IS NULL THEN NULL
                    ELSE date_diff('day', CAST(s.date AS DATE), f.failure_date) END AS days_to_failure,
               CASE WHEN f.failure_date IS NOT NULL
                         AND date_diff('day', CAST(s.date AS DATE), f.failure_date) BETWEEN {lo} AND {h}
                    THEN 1 ELSE 0 END AS label
        FROM fact_daily_smart s
        LEFT JOIN first_fail f USING (serial_number)
        WHERE {model_filter}
          AND CAST(s.date AS DATE) BETWEEN CAST('{start}' AS DATE) AND CAST('{end}' AS DATE)
          AND (f.failure_date IS NULL OR {last_ok})
          AND s.serial_number NOT IN (SELECT serial_number FROM excluded_drives)
    """, params)

    n, pos, drives = con.execute(f"SELECT COUNT(*), SUM(label), COUNT(DISTINCT serial_number) FROM {table}").fetchone()
    print(f"\nTable {table}: {n:,} rows, {drives:,} drives, {pos:,} positives ({pos / n:.4%})")
    print(f"  scoring dates {start} to {end}; last data date {max_date}; cutoff for H={h} is {max_date - dt.timedelta(days=h)}")
    print(f"  day-zero convention: {args.day_zero}")
    print("\nPer model:")
    for model, mn, mp, md in con.execute(
        f"SELECT model, COUNT(*), SUM(label), COUNT(DISTINCT serial_number) FROM {table} GROUP BY model ORDER BY 2 DESC"
    ).fetchall():
        print(f"  {model:<28} rows {mn:>12,}  drives {md:>7,}  positives {mp:>8,}  ({mp / mn:.4%})")

    # Sanity checks on the rules.
    after = con.execute(f"""
        SELECT COUNT(*) FROM {table} l JOIN first_fail f USING (serial_number)
        WHERE l.date > f.failure_date""").fetchone()[0]
    dup = con.execute(f"SELECT COUNT(*) - COUNT(DISTINCT (date, serial_number)) FROM {table}").fetchone()[0]
    print(f"\nChecks: rows after failure = {after} (want 0), duplicate drive-days = {dup} (want 0)")
    print(f"Done in {time.time() - t0:.0f}s")


def main():
    args = parse_args()
    con = connect(read_only=args.verify)
    max_date = con.execute("SELECT MAX(CAST(date AS DATE)) FROM fact_daily_smart").fetchone()[0]
    cutoff = max_date - dt.timedelta(days=args.horizon)
    if args.verify:
        verify(con, args, cutoff)
    else:
        build(con, args, cutoff, max_date)
    con.close()


if __name__ == "__main__":
    main()
