"""Build the gold star schema in DuckDB from the silver Parquet layer.

Run from the repo root (PowerShell):
docker run --rm -v "${PWD}:/workspace" drivewatch-spark python src/etl/build_gold.py

Writes data/gold/drivewatch.duckdb. Safe to re-run: every table is replaced.
features_daily is left for Phase 3.
"""
import argparse
from pathlib import Path

import duckdb

# SMART attributes kept in fact_daily_smart (raw values). The other ~70 stay in silver.
SMART_IDS = [1, 3, 4, 5, 7, 9, 10, 12, 187, 188, 189, 190, 192, 193, 194, 197, 198, 199, 240, 241, 242]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", default="data/silver/**/*.parquet")
    parser.add_argument("--db", default="data/gold/drivewatch.duckdb")
    args = parser.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(args.db)
    con.execute("SET memory_limit='5GB'")
    con.execute("SET threads=4")
    con.execute("SET temp_directory='/tmp/duckdb_tmp'")
    src = f"read_parquet('{args.silver}', hive_partitioning=true)"

    smart_cols = ",\n            ".join(f"smart_{i}_raw" for i in SMART_IDS)

    print("Building fact_daily_smart ...")
    con.execute(f"""
        CREATE OR REPLACE TABLE fact_daily_smart AS
        SELECT date, serial_number, model, failure,
            {smart_cols}
        FROM {src}
    """)

    print("Building dim_drive ...")
    con.execute(f"""
        CREATE OR REPLACE TABLE dim_drive AS
        SELECT serial_number,
               MIN(model) AS model,
               MAX(capacity_bytes) AS capacity_bytes,
               MIN(date) AS first_seen,
               MAX(date) AS last_seen,
               MAX(failure) = 1 AS failed
        FROM {src}
        GROUP BY serial_number
    """)

    print("Building fact_failures ...")
    con.execute("""
        CREATE OR REPLACE TABLE fact_failures AS
        SELECT f.serial_number,
               d.model,
               f.failure_date,
               d.first_seen,
               DATEDIFF('day', d.first_seen, f.failure_date) AS days_in_service
        FROM (SELECT serial_number, MIN(date) AS failure_date
              FROM fact_daily_smart WHERE failure = 1 GROUP BY serial_number) f
        JOIN dim_drive d USING (serial_number)
    """)

    print("Building dim_model ...")
    con.execute("""
        CREATE OR REPLACE TABLE dim_model AS
        SELECT model,
               CASE
                 WHEN model LIKE 'ST%' THEN 'Seagate'
                 WHEN model LIKE 'TOSHIBA%' THEN 'Toshiba'
                 WHEN model LIKE 'HGST%' OR model LIKE 'HUH%' OR model LIKE 'HUS%' OR model LIKE 'HMS%' THEN 'HGST'
                 WHEN model LIKE 'HIT%' OR model LIKE 'HDS%' THEN 'Hitachi'
                 WHEN model LIKE 'WDC%' OR model LIKE 'WD%' OR model LIKE 'WUH%' THEN 'Western Digital'
                 WHEN model LIKE 'SAMSUNG%' OR model LIKE 'Samsung%' THEN 'Samsung'
                 ELSE 'Other'
               END AS manufacturer,
               MODE(capacity_bytes) AS capacity_bytes,
               ROUND(MODE(capacity_bytes) / 1e12, 1) AS capacity_tb,
               COUNT(*) AS drive_count,
               MIN(first_seen) AS first_seen,
               MAX(last_seen) AS last_seen,
               NULL::VARCHAR AS interface,
               NULL::INTEGER AS rpm,
               NULL::INTEGER AS release_year
        FROM dim_drive
        GROUP BY model
    """)

    print("Building dim_date ...")
    con.execute("""
        CREATE OR REPLACE TABLE dim_date AS
        SELECT d::DATE AS date,
               YEAR(d) AS year,
               QUARTER(d) AS quarter,
               MONTH(d) AS month,
               WEEKOFYEAR(d) AS week,
               DAYOFWEEK(d) AS day_of_week
        FROM generate_series(
            (SELECT MIN(date) FROM fact_daily_smart),
            (SELECT MAX(date) FROM fact_daily_smart),
            INTERVAL 1 DAY) t(d)
    """)

    print("\n=== Table sizes ===")
    for t in ["fact_daily_smart", "fact_failures", "dim_drive", "dim_model", "dim_date"]:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<18} {n:>12,} rows")

    print("\n=== Failures per quarter (Backblaze: Q1 1,030, Q2 1,498) ===")
    for year, quarter, n in con.execute("""
        SELECT YEAR(failure_date), QUARTER(failure_date), COUNT(*)
        FROM fact_failures GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchall():
        print(f"  {year} Q{quarter}: {n:,}")

    print("\n=== Manufacturer mix in dim_model ===")
    for m, models, drives in con.execute("""
        SELECT manufacturer, COUNT(*), SUM(drive_count)
        FROM dim_model GROUP BY 1 ORDER BY 3 DESC
    """).fetchall():
        print(f"  {m:<16} {models:>4} models {drives:>9,} drives")

    other = con.execute("""
        SELECT model, drive_count FROM dim_model
        WHERE manufacturer = 'Other' ORDER BY drive_count DESC LIMIT 8
    """).fetchall()
    if other:
        print("\nLargest models labelled 'Other' (tell me if any need a rule):")
        for model, n in other:
            print(f"  {model:<32} {n:,}")

    con.close()
    size_mb = Path(args.db).stat().st_size / 1024 / 1024
    print(f"\nWrote {args.db} ({size_mb:,.0f} MB)")


if __name__ == "__main__":
    main()
