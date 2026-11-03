"""Quick sanity check of data/bronze_parquet. Run: python -m src.ingest.check_parquet"""
from pathlib import Path

import duckdb

ROOT = Path("data/bronze_parquet")
GLOB = (ROOT / "*" / "part.parquet").as_posix()


def folder_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def main() -> None:
    con = duckdb.connect()
    src = f"read_parquet('{GLOB}', union_by_name=true, hive_partitioning=false)"

    print(f"Parquet size on disk: {folder_size_gb(ROOT):.2f} GB")
    zips = Path("data/bronze")
    if zips.exists():
        print(f"Zips on disk:         {folder_size_gb(zips):.2f} GB")

    print("\n-- overview")
    print(con.sql(f"SELECT count(*) AS rows, count(DISTINCT date) AS days, "
                  f"min(date) AS first_day, max(date) AS last_day, "
                  f"count(DISTINCT serial_number) AS drives FROM {src}").df().to_string(index=False))

    cols = con.sql(f"SELECT * FROM {src} LIMIT 0").columns
    smart = [c for c in cols if c.startswith("smart_")]
    other = [c for c in cols if not c.startswith("smart_")]
    print(f"\n-- {len(cols)} columns; {len(smart)} SMART columns")
    print("non-SMART columns:", other)

    print("\n-- rows per day (min / median / max)")
    print(con.sql(f"SELECT min(n) AS min, median(n) AS median, max(n) AS max FROM "
                  f"(SELECT date, count(*) AS n FROM {src} GROUP BY date)").df().to_string(index=False))

    print("\n-- failures per month")
    print(con.sql(f"SELECT substr(date,1,7) AS month, count(*) AS rows, "
                  f"sum(try_cast(failure AS INTEGER)) AS failures FROM {src} "
                  f"GROUP BY 1 ORDER BY 1").df().to_string(index=False))

    print("\n-- top 10 models by drive count")
    print(con.sql(f"SELECT model, count(DISTINCT serial_number) AS drives, "
                  f"sum(try_cast(failure AS INTEGER)) AS failures FROM {src} "
                  f"GROUP BY model ORDER BY drives DESC LIMIT 10").df().to_string(index=False))


if __name__ == "__main__":
    main()