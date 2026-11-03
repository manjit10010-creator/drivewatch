"""Download files listed in data/manifest.json into data/bronze/.

Run:  python -m src.ingest.downloader --years 2023
Skips files already downloaded, retries with backoff, waits between requests,
and records size, status and sha256 in the manifest.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

from src.ingest.scraper import MANIFEST_PATH, USER_AGENT

BRONZE = Path("data/bronze")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(entry: dict, session: requests.Session, retries: int = 3) -> None:
    dest = BRONZE / str(entry["year"]) / entry["file_name"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if entry["status"] == "done" and dest.exists():
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with session.get(entry["url"], stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) or None
                with tmp.open("wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=entry["file_name"]
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.replace(dest)
            entry.update(status="done", size_bytes=dest.stat().st_size, sha256=sha256_of(dest))
            return
        except requests.RequestException as exc:
            print(f"Attempt {attempt}/{retries} failed for {entry['file_name']}: {exc}")
            time.sleep(2 ** attempt)
    entry["status"] = "failed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--years", type=int, nargs="+", required=True, help="e.g. --years 2023")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between files")
    args = ap.parse_args()

    entries = json.loads(args.manifest.read_text())
    todo = [e for e in entries if e["year"] in args.years]
    if not todo:
        raise SystemExit(f"No manifest entries for years {args.years}")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for e in todo:
        download_one(e, session)
        args.manifest.write_text(json.dumps(entries, indent=2))  # save progress each file
        time.sleep(args.delay)

    done = sum(e["status"] == "done" for e in todo)
    print(f"{done}/{len(todo)} files downloaded to {BRONZE}")


if __name__ == "__main__":
    main()
