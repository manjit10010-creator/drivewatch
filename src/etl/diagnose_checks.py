"""Diagnose the two failed quality checks on the 2025-2026 silver data.

Run from the repo root:
  docker run --rm -v "${PWD}:/workspace" drivewatch-spark python src/etl/diagnose_checks.py

Read-only: it only queries data/silver and prints tables.
"""
import duckdb
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

con = duckdb.connect()
con.execute(
    """
    CREATE VIEW s AS
    SELECT CAST(date AS DATE) AS date, serial_number, model, failure
    FROM read_parquet('data/silver/**/*.parquet', hive_partitioning = true)
    """
)


def show(title, sql):
    print(f"\n=== {title}")
    print(con.execute(sql).df().to_string(index=False))


# ---------------------------------------------------------------- check 1
show(
    "Rows per day around 2025-05-12",
    """
    SELECT date, count(*) AS rows
    FROM s
    WHERE date BETWEEN DATE '2025-05-08' AND DATE '2025-05-16'
    GROUP BY date ORDER BY date
    """,
)

show(
    "Drives present on 05-11 and 05-13 but missing on 05-12, by model (top 10)",
    """
    SELECT a.model, count(*) AS missing_drives
    FROM (SELECT serial_number, model FROM s WHERE date = DATE '2025-05-11') a
    JOIN (SELECT serial_number FROM s WHERE date = DATE '2025-05-13') c
      ON a.serial_number = c.serial_number
    LEFT JOIN (SELECT serial_number FROM s WHERE date = DATE '2025-05-12') b
      ON a.serial_number = b.serial_number
    WHERE b.serial_number IS NULL
    GROUP BY a.model ORDER BY missing_drives DESC LIMIT 10
    """,
)

show(
    "Failures per day around 2025-05-12",
    """
    SELECT date, sum(failure) AS failures
    FROM s
    WHERE date BETWEEN DATE '2025-05-08' AND DATE '2025-05-16'
    GROUP BY date ORDER BY date
    """,
)

# ---------------------------------------------------------------- check 2
con.execute(
    """
    CREATE TABLE f AS
    SELECT serial_number, min(date) AS fail_date, count(*) AS n_failure_rows
    FROM s WHERE failure = 1 GROUP BY serial_number
    """
)
con.execute(
    """
    CREATE TABLE later AS
    SELECT s.serial_number, s.model, s.date, s.failure, f.fail_date
    FROM s JOIN f ON s.serial_number = f.serial_number
    WHERE s.date > f.fail_date
    """
)

show(
    "Overall",
    """
    SELECT (SELECT count(*) FROM f) AS failed_drives,
           (SELECT count(DISTINCT serial_number) FROM later) AS reappearing_drives,
           (SELECT count(*) FROM later) AS later_rows,
           (SELECT sum(failure) FROM later) AS later_rows_with_failure_1,
           (SELECT count(*) FROM f WHERE n_failure_rows > 1) AS drives_with_2plus_failure_rows
    """,
)

show(
    "Gap in days from failure to first reappearance",
    """
    SELECT CASE WHEN gap = 1 THEN '1 day'
                WHEN gap <= 7 THEN '2-7 days'
                WHEN gap <= 30 THEN '8-30 days'
                ELSE '31+ days' END AS gap_bucket,
           count(*) AS drives
    FROM (SELECT serial_number, datediff('day', min(fail_date), min(date)) AS gap
          FROM later GROUP BY serial_number)
    GROUP BY gap_bucket ORDER BY min(gap)
    """,
)

show(
    "How many later rows per reappearing drive",
    """
    SELECT CASE WHEN n = 1 THEN '1 row' WHEN n <= 7 THEN '2-7 rows'
                WHEN n <= 30 THEN '8-30 rows' ELSE '31+ rows' END AS later_rows,
           count(*) AS drives
    FROM (SELECT serial_number, count(*) AS n FROM later GROUP BY serial_number)
    GROUP BY later_rows ORDER BY min(n)
    """,
)

show(
    "Reappearing drives by model (top 10)",
    """
    SELECT model, count(DISTINCT serial_number) AS drives
    FROM later GROUP BY model ORDER BY drives DESC LIMIT 10
    """,
)

show(
    "10 examples",
    """
    SELECT serial_number, model, min(fail_date) AS fail_date,
           min(date) AS first_later, max(date) AS last_later, count(*) AS later_rows
    FROM later GROUP BY serial_number, model
    ORDER BY later_rows DESC LIMIT 10
    """,
)
