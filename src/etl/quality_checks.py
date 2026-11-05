"""Data quality checks on the silver layer (plan section 6).

Run from the repo root (PowerShell):
docker run --rm -v "${PWD}:/workspace" drivewatch-spark python src/etl/quality_checks.py

Each check ends as PASS, WARN or FAIL. A WARN is a documented, understood
data quirk (see docs/decisions.md); a FAIL is a real problem. The script exits
with code 1 only if a check FAILS, so a pipeline can stop on it.
"""
import argparse
import re
import statistics
import sys

import duckdb

# Core columns every silver file must have, with their types.
CORE_COLUMNS = {
    "date": "DATE",
    "serial_number": "VARCHAR",
    "model": "VARCHAR",
    "capacity_bytes": "BIGINT",
    "failure": "INTEGER",
    "datacenter": "VARCHAR",
    "cluster_id": "INTEGER",
    "vault_id": "INTEGER",
    "pod_id": "INTEGER",
    "pod_slot_num": "INTEGER",
    "is_legacy_format": "BOOLEAN",
    "year": "BIGINT",
    "month": "BIGINT",
}
SMART_RE = re.compile(r"^smart_\d+_(raw|normalized)$")
EXPECTED_TOTAL_COLUMNS = 199  # 13 core + 93 SMART attributes x 2. Update if the schema changes.

DAY_COUNT_TOLERANCE = 0.10  # a day may differ from the median of the previous 7 days by 10%
TEMP_MAX_SHARE = 0.001  # fail if more than 0.1% of reported temperatures look impossible

# Days whose odd row count is understood and documented in docs/decisions.md.
# They are reported as WARN. Any other unusual day is still a FAIL.
KNOWN_GAP_DATES = {
    "2025-05-12": "partial Backblaze file: about 70k drives missing, only 2 failures that day",
}

# Failed drives that later reappear (probably returned to service, e.g. firmware pulls).
# Up to this share of failed drives is a WARN; above it is a FAIL. Labels drop all rows
# after a drive's first failure, so a small share is harmless. See docs/decisions.md.
REAPPEAR_WARN_SHARE = 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", default="data/silver/**/*.parquet")
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'")
    con.execute("SET threads=4")
    con.execute("SET temp_directory='/tmp/duckdb_tmp'")
    src = f"read_parquet('{args.silver}', hive_partitioning=true)"

    results = []  # (name, status, detail) with status PASS, WARN or FAIL

    def record(name, status, detail):
        results.append((name, status, detail))
        print(f"[{status}] {name}: {detail}")

    # 1. Schema matches the expected column list
    schema = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    found = {row[0]: row[1] for row in schema}
    problems = []
    for col, dtype in CORE_COLUMNS.items():
        if col not in found:
            problems.append(f"missing {col}")
        elif found[col] != dtype:
            problems.append(f"{col} is {found[col]}, expected {dtype}")
    for col in found:
        if col not in CORE_COLUMNS and not SMART_RE.match(col):
            problems.append(f"unexpected column {col}")
    if len(found) != EXPECTED_TOTAL_COLUMNS:
        problems.append(f"{len(found)} columns, expected {EXPECTED_TOTAL_COLUMNS}")
    record(
        "schema matches expected columns",
        "FAIL" if problems else "PASS",
        "; ".join(problems[:5]) or f"{len(found)} columns",
    )

    # 2. No duplicate (date, serial_number). A key can only repeat within one day,
    #    so checking one month at a time is exact and keeps memory low.
    months = con.execute(f"SELECT DISTINCT year, month FROM {src} ORDER BY 1, 2").fetchall()
    dupes = 0
    for year, month in months:
        dupes += con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT date, serial_number FROM {src}
                WHERE year = {year} AND month = {month}
                GROUP BY date, serial_number HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
    record("no duplicate (date, serial_number)", "FAIL" if dupes else "PASS", f"{dupes} duplicate keys")

    # 3. Row counts per day: no missing days, no big jumps against the previous 7 days.
    #    Days listed in KNOWN_GAP_DATES are a WARN; any other unusual day is a FAIL.
    daily = con.execute(f"SELECT date, COUNT(*) FROM {src} GROUP BY date ORDER BY date").fetchall()
    first, last = daily[0][0], daily[-1][0]
    expected_days = (last - first).days + 1
    known_days, unknown_days = [], []
    for i in range(1, len(daily)):
        window = [n for _, n in daily[max(0, i - 7):i]]
        median = statistics.median(window)
        day, n = daily[i]
        if abs(n - median) / median > DAY_COUNT_TOLERANCE:
            text = f"{day}: {n:,} vs median {int(median):,}"
            (known_days if str(day) in KNOWN_GAP_DATES else unknown_days).append((str(day), text))
    detail = f"{len(daily)} days of {expected_days} expected"
    if unknown_days:
        detail += f"; unusual days: {[t for _, t in unknown_days[:5]]}"
    if known_days:
        detail += f"; known gaps: {[t + ' (' + KNOWN_GAP_DATES[d] + ')' for d, t in known_days]}"
    if len(daily) != expected_days or unknown_days:
        status = "FAIL"
    elif known_days:
        status = "WARN"
    else:
        status = "PASS"
    record("daily row counts in expected range", status, detail)

    # 4. Failure flag only 0 or 1 (and never missing)
    bad_flags = con.execute(f"SELECT COUNT(*) FROM {src} WHERE failure IS NULL OR failure NOT IN (0, 1)").fetchone()[0]
    record("failure flag only 0 or 1", "FAIL" if bad_flags else "PASS", f"{bad_flags} bad values")

    # 5. A failed drive should not reappear after its failure date. A small share is a WARN
    #    (documented quirk); above REAPPEAR_WARN_SHARE of failed drives it is a FAIL.
    reappeared = con.execute(f"""
        WITH failed AS (
            SELECT serial_number, MIN(date) AS fail_date FROM {src} WHERE failure = 1 GROUP BY serial_number
        )
        SELECT COUNT(DISTINCT s.serial_number), COUNT(*)
        FROM {src} s JOIN failed f ON s.serial_number = f.serial_number AND s.date > f.fail_date
    """).fetchone()
    n_failed = con.execute(f"SELECT COUNT(DISTINCT serial_number) FROM {src} WHERE failure = 1").fetchone()[0]
    share = reappeared[0] / n_failed if n_failed else 0
    if reappeared[0] == 0:
        status = "PASS"
    elif share <= REAPPEAR_WARN_SHARE:
        status = "WARN"
    else:
        status = "FAIL"
    record(
        "failed drive never reappears",
        status,
        f"{reappeared[0]} of {n_failed:,} failed drives appear on later days "
        f"({share:.2%}, {reappeared[1]:,} rows; WARN up to {REAPPEAR_WARN_SHARE:.0%})",
    )

    # 6. Temperature plausible. For some models the raw value packs min/max temperature into
    #    the higher bytes, so we judge the lowest byte (raw & 255), which holds the current reading.
    total, bad, raw_over = con.execute(f"""
        SELECT COUNT(*),
               SUM(CASE WHEN (smart_194_raw & 255) > 100 THEN 1 ELSE 0 END),
               SUM(CASE WHEN smart_194_raw > 100 THEN 1 ELSE 0 END)
        FROM {src} WHERE smart_194_raw IS NOT NULL
    """).fetchone()
    temp_share = (bad or 0) / total if total else 0
    record(
        "temperature (smart_194_raw) plausible",
        "PASS" if temp_share <= TEMP_MAX_SHARE else "FAIL",
        f"{bad or 0:,} of {total:,} reported rows have a low byte above 100 ({temp_share:.4%}); "
        f"{raw_over or 0:,} rows have raw above 100 (packed values, informational)",
    )

    failed_checks = [name for name, status, _ in results if status == "FAIL"]
    warned_checks = [name for name, status, _ in results if status == "WARN"]
    print()
    if warned_checks:
        print(f"{len(warned_checks)} WARNING(S) (known, documented): {warned_checks}")
    if failed_checks:
        print(f"{len(failed_checks)} CHECK(S) FAILED: {failed_checks}")
        sys.exit(1)
    print("No failed checks.")


if __name__ == "__main__":
    main()
