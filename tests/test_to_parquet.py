import sys
import types
import zipfile

# duckdb is only needed for the real conversion; stub it so the zip logic can be tested anywhere.
sys.modules.setdefault("duckdb", types.ModuleType("duckdb"))

from src.ingest import to_parquet  # noqa: E402


def test_day_from_name():
    assert to_parquet.day_from_name("data_Q1_2026/2026-01-05.csv") == "2026-01-05"
    assert to_parquet.day_from_name("2026-01-05.csv") == "2026-01-05"
    assert to_parquet.day_from_name("__MACOSX/data_Q1_2026/2026-01-05.csv") is None
    assert to_parquet.day_from_name("data_Q1_2026/readme.txt") is None


def test_convert_zip_converts_then_skips(tmp_path, monkeypatch):
    zpath = tmp_path / "data_Q1_2026.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("data_Q1_2026/2026-01-01.csv", "date,serial_number\n2026-01-01,A\n")
        zf.writestr("data_Q1_2026/2026-01-02.csv", "date,serial_number\n2026-01-02,B\n")
        zf.writestr("__MACOSX/junk.csv", "x")

    calls = []

    def fake_convert(csv_path, out_file):
        calls.append(out_file.parent.name)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(b"x")

    monkeypatch.setattr(to_parquet, "csv_to_parquet", fake_convert)
    out = tmp_path / "out"
    assert to_parquet.convert_zip(zpath, out) == (2, 0)
    assert sorted(calls) == ["date=2026-01-01", "date=2026-01-02"]
    assert to_parquet.convert_zip(zpath, out) == (0, 2)  # idempotent