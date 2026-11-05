"""How many drives do the biggest models cover? Run from the repo root:
docker run --rm -v "${PWD}:/workspace" drivewatch-spark python src/etl/model_coverage.py
"""
import duckdb

con = duckdb.connect("data/gold/drivewatch.duckdb", read_only=True)
total = con.execute("SELECT SUM(drive_count) FROM dim_model").fetchone()[0]
rows = con.execute("""
    SELECT model, manufacturer, drive_count FROM dim_model ORDER BY drive_count DESC
""").fetchall()

print(f"Total drives: {total:,} across {len(rows)} models\n")
print(f"{'rank':<5}{'model':<34}{'drives':>9}{'cumulative %':>14}")
cum = 0
for i, (model, maker, n) in enumerate(rows[:25], start=1):
    cum += n
    print(f"{i:<5}{model:<34}{n:>9,}{cum / total:>13.1%}")

for k in (5, 12, 20):
    share = sum(r[2] for r in rows[:k]) / total
    print(f"\nTop {k} models cover {share:.1%} of drives")
