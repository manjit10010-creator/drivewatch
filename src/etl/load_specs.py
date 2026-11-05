"""Load model specs and drive_type into dim_model (run AFTER build_gold.py).

build_gold.py recreates dim_model from scratch, so rerun this script whenever you rebuild gold.

Run from the repo root (PowerShell):
docker run --rm -v "${PWD}:/workspace" drivewatch-spark python src/etl/load_specs.py
"""
import argparse

import duckdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/gold/drivewatch.duckdb")
    parser.add_argument("--specs", default="reference/model_specs.csv")
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    # 1. Manufacturer fixes for non-HDD models that the name prefix rule missed
    con.execute("""
        UPDATE dim_model SET manufacturer = CASE
            WHEN model LIKE 'Seagate%' THEN 'Seagate'
            WHEN model LIKE 'CT%' THEN 'Crucial'
            WHEN model LIKE 'DELLBOSS%' THEN 'Dell'
            WHEN model LIKE 'MTFD%' THEN 'Micron'
            ELSE manufacturer END
        WHERE manufacturer = 'Other'
    """)

    # 2. drive_type: separate SSD / boot devices from spinning disks
    con.execute("ALTER TABLE dim_model ADD COLUMN IF NOT EXISTS drive_type VARCHAR")
    con.execute("""
        UPDATE dim_model SET drive_type = CASE
            WHEN model ILIKE '%SSD%' OR model LIKE 'Seagate%' OR model LIKE 'CT%' OR model LIKE 'MTFD%' OR model LIKE 'DELLBOSS%'
                THEN 'SSD/boot'
            ELSE 'HDD' END
    """)

    # 3. Specs from the reference CSV (blank cells stay NULL)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW specs AS
        SELECT * FROM read_csv('{args.specs}', header=true, all_varchar=true)
    """)
    con.execute("""
        UPDATE dim_model SET
            interface = NULLIF(s.interface, ''),
            rpm = TRY_CAST(NULLIF(s.rpm, '') AS INTEGER),
            release_year = TRY_CAST(NULLIF(s.release_year, '') AS INTEGER)
        FROM specs s
        WHERE dim_model.model = s.model
    """)

    unmatched = con.execute("""
        SELECT model FROM specs WHERE model NOT IN (SELECT model FROM dim_model)
    """).fetchall()
    if unmatched:
        print("Models in the CSV that are NOT in dim_model (check spelling):")
        for (m,) in unmatched:
            print(f"  {m}")

    # 4. Report
    total = con.execute("SELECT SUM(drive_count) FROM dim_model").fetchone()[0]
    with_rpm = con.execute("SELECT SUM(drive_count) FROM dim_model WHERE rpm IS NOT NULL").fetchone()[0] or 0
    with_year = con.execute("SELECT SUM(drive_count) FROM dim_model WHERE release_year IS NOT NULL").fetchone()[0] or 0
    print(f"\nDrives with RPM known:          {with_rpm:>9,} of {total:,} ({with_rpm / total:.1%})")
    print(f"Drives with release year known: {with_year:>9,} of {total:,} ({with_year / total:.1%})")

    print("\ndrive_type split:")
    for dtype, models, drives in con.execute("""
        SELECT drive_type, COUNT(*), SUM(drive_count) FROM dim_model GROUP BY 1 ORDER BY 3 DESC
    """).fetchall():
        print(f"  {dtype:<10} {models:>3} models {drives:>9,} drives")

    print("\nBiggest HDD models still missing RPM (fill these next):")
    for model, n in con.execute("""
        SELECT model, drive_count FROM dim_model
        WHERE rpm IS NULL AND drive_type = 'HDD' ORDER BY drive_count DESC LIMIT 12
    """).fetchall():
        print(f"  {model:<28} {n:>8,}")

    # 5. Q2 gap test: how many failures sit in SSD/boot models?
    print("\nFailures in SSD/boot models by quarter (Q2 gap vs Backblaze was +24):")
    for year, quarter, n in con.execute("""
        SELECT YEAR(f.failure_date), QUARTER(f.failure_date), COUNT(*)
        FROM fact_failures f JOIN dim_model m USING (model)
        WHERE m.drive_type = 'SSD/boot' GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchall():
        print(f"  {year} Q{quarter}: {n}")

    con.close()


if __name__ == "__main__":
    main()
