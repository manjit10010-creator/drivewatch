"""Scrape the Backblaze Drive Stats page and write a download manifest.

Run:  python -m src.ingest.scraper
Writes data/manifest.json. It never downloads the zips; that is downloader.py.
"""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

PAGE_URL = "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data"
USER_AGENT = "DriveWatch-learning-project/0.1 (personal ML portfolio; polite, rate-limited)"
MANIFEST_PATH = Path("data/manifest.json")

# Matches names like data_Q1_2023.zip, data_2013.zip, data_2014.zip, 2015.zip
FILE_RE = re.compile(r"(?:data_)?(?:Q([1-4])_)?((?:19|20)\d{2})\.zip$", re.IGNORECASE)


def parse_links(html: str, base_url: str = PAGE_URL) -> list[dict]:
    """Extract every Drive Stats zip link from the page HTML."""
    soup = BeautifulSoup(html, "html.parser")
    seen, entries = set(), []
    for a in soup.find_all("a", href=True):
        url = urljoin(base_url, a["href"])
        name = Path(urlparse(url).path).name
        m = FILE_RE.search(name)
        if not m or url in seen:
            continue
        seen.add(url)
        quarter, year = m.group(1), int(m.group(2))
        entries.append({
            "file_name": name,
            "url": url,
            "year": year,
            "quarter": int(quarter) if quarter else None,
            "size_bytes": None,
            "status": "pending",
            "sha256": None,
        })
    entries.sort(key=lambda e: (e["year"], e["quarter"] or 0))
    return entries


def merge_with_existing(new: list[dict], path: Path) -> list[dict]:
    """Keep download status/checksum for files already in an old manifest."""
    if not path.exists():
        return new
    old = {e["url"]: e for e in json.loads(path.read_text())}
    return [old.get(e["url"], e) for e in new]


def fetch_sizes(entries: list[dict], session: requests.Session, delay: float = 1.0) -> None:
    """HEAD each URL for its size. Polite: one request per `delay` seconds."""
    for e in entries:
        if e["size_bytes"] is not None:
            continue
        try:
            r = session.head(e["url"], allow_redirects=True, timeout=30)
            size = r.headers.get("Content-Length")
            e["size_bytes"] = int(size) if size else None
        except requests.RequestException as exc:
            print(f"HEAD failed for {e['file_name']}: {exc}")
        time.sleep(delay)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=PAGE_URL)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--no-sizes", action="store_true", help="skip HEAD requests")
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    resp = session.get(args.url, timeout=30)
    resp.raise_for_status()

    entries = merge_with_existing(parse_links(resp.text, args.url), args.manifest)
    if not entries:
        raise SystemExit("No zip links found. The page layout may have changed; inspect the HTML.")
    if not args.no_sizes:
        fetch_sizes(entries, session)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(entries, indent=2))
    print(f"Wrote {len(entries)} entries to {args.manifest}")


if __name__ == "__main__":
    main()
