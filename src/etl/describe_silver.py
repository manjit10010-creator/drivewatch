"""Print the silver schema so we can write the data quality checks against real column names.

Run from the repo root (PowerShell):
docker run --rm -v "${PWD}:/workspace" drivewatch-spark python src/etl/describe_silver.py
"""
import duckdb

SILVER = "data/silver/**/*.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET threads=4")

src = f"read_parquet('{SILVER}', hive_partitioning=true)"

schema = con.sql(f"DESCRIBE SELECT * FROM {src}").fetchall()
print(f"Total columns: {len(schema)}\n")

print("First 12 columns:")
for name, dtype, *_ in schema[:12]:
    print(f"  {name:<28} {dtype}")

print("\nLast 6 columns:")
for name, dtype, *_ in schema[-6:]:
    print(f"  {name:<28} {dtype}")

keys = ("fail", "temp", "serial", "model", "capacity", "legacy", "date", "year", "month", "smart_194", "smart_9_", "smart_5_")
print("\nColumns whose names match key words:")
for name, dtype, *_ in schema:
    if any(k in name.lower() for k in keys):
        print(f"  {name:<28} {dtype}")

print("\nQuick facts:")
print(con.sql(f"""
    SELECT COUNT(*) AS rows,
           MIN(date) AS first_day,
           MAX(date) AS last_day,
           COUNT(DISTINCT date) AS days
    FROM {src}
""").fetchall())
