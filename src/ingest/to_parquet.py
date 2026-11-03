"""Convert Backblaze Drive Stats zips (data/bronze/) into one Parquet file per day.

Run:  python -m src.ingest.to_parquet --years 2026
Output: data/bronze_parquet/date=YYYY-MM-DD/part.parquet

Design choices:
- Every column is read as text (all_varchar). Column sets changed over the years, so
  typing and schema alignment are the job of the silver ETL step, not this one.
- One CSV is extracted, converted and deleted at a time, so memory and disk stay small.
- Re-running skips days that already have a Parquet file.
"""
import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import duckdb

BRONZE = Path("data/bronze")
OUT = Path("data/bronze_parquet")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.csv$")


def day_from_name(name: str) -> str | None:
    """'data_Q1_2026/2026-01-05.csv' -> '2026-01-05'. None for non-day files."""
    if "__MACOSX" in name:
        return None
    m = DATE_RE.search(name)
    return m.group(1) if m else None


def csv_to_parquet(csv_path: Path, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT * FROM read_csv('{csv_path.as_posix()}', all_varchar=true, header=true)) "
        f"TO '{out_file.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
    )
    con.close()


def convert_zip(zip_path: Path, out_root: Path = OUT) -> tuple[int, int]:
    """Returns (converted, skipped)."""
    converted = skipped = 0
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmp:
        for member in sorted(zf.namelist()):
            day = day_from_name(member)
            if day is None:
                continue
            out_file = out_root / f"date={day}" / "part.parquet"
            if out_file.exists():
                skipped += 1
                continue
            extracted = Path(zf.extract(member, tmp))
            csv_to_parquet(extracted, out_file)
            extracted.unlink()
            converted += 1
    return converted, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--delete-zips", action="store_true",
                    help="delete each zip after a successful conversion (saves disk)")
    args = ap.parse_args()

    for year in args.years:
        for zip_path in sorted((BRONZE / str(year)).glob("*.zip")):
            c, s = convert_zip(zip_path)
            print(f"{zip_path.name}: {c} days converted, {s} skipped")
            if args.delete_zips:
                zip_path.unlink()


if __name__ == "__main__":
    main()