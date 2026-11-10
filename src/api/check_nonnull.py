"""How many of the 25 model features are filled in (not NULL) on each scored drive-day?

Used to choose MIN_FILLED for POST /predict: a request with fewer filled features than any real drive-day
is rejected, instead of being scored as if the missing values were real.

Read-only look at the warehouse. Run from the repo root with the API stopped:
    docker run --rm -v "${PWD}:/workspace" drivewatch-api:slim python src/api/check_nonnull.py
"""
import duckdb

from src.api.predict import FEATURES

DB_PATH = "data/gold/drivewatch.duckdb"


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    filled = " + ".join(f"({f} IS NOT NULL)::INT" for f in FEATURES)
    print(f"{len(FEATURES)} features; rows of features_scoring grouped by how many are filled in\n")
    rows = con.execute(
        f"SELECT n, COUNT(*) AS drive_days FROM (SELECT {filled} AS n FROM features_scoring) GROUP BY n ORDER BY n"
    ).fetchall()
    total = sum(r[1] for r in rows)
    cum = 0
    print(f"{'filled':>6} {'drive_days':>12} {'cumulative %':>13}")
    for n, c in rows:
        cum += c
        print(f"{n:>6} {c:>12,} {100 * cum / total:>12.4f}%")
    print("\nlowest 5 filled-counts by drive model:")
    for m, n, c in con.execute(
        f"SELECT model, n, COUNT(*) FROM (SELECT model, {filled} AS n FROM features_scoring) "
        f"GROUP BY model, n ORDER BY n, model LIMIT 5"
    ).fetchall():
        print(f"  {m:<22} filled={n:<3} drive_days={c:,}")


if __name__ == "__main__":
    main()
