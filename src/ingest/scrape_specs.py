"""Fill missing drive-model specs (interface, RPM) by reading product pages.

You put a product-page URL in reference/model_specs.csv (column source_url) for each model
marked 'todo'. This script fetches those pages politely, finds the RPM and interface in the
page text, and writes them back to the CSV with status 'scraped'. Read the results before
trusting them: a retailer page can list several variants.

Run from the repo root (PowerShell, Windows venv):
python -m src.ingest.scrape_specs
"""
import csv
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

CSV_PATH = Path("reference/model_specs.csv")
FIELDS = ["model", "interface", "rpm", "release_year", "source_url", "status"]
USER_AGENT = "DriveWatch-learning-project/0.1 (personal portfolio project; contact: manjit10010@gmail.com)"
DELAY_SECONDS = 5

RPM_FULL = re.compile(r"\b(5[,.]?400|5[,.]?900|7[,.]?200|10[,.]?000|15[,.]?000)\s*(?:rpm|r/min)\b", re.I)
RPM_SHORT = re.compile(r"\b(5\.4|7\.2|10|15)\s*k\s*rpm\b", re.I)
SHORT_TO_RPM = {"5.4": 5400, "7.2": 7200, "10": 10000, "15": 15000}

_robots = {}


def allowed_by_robots(url):
    host = urlparse(url)
    base = f"{host.scheme}://{host.netloc}"
    if base not in _robots:
        parser = RobotFileParser()
        parser.set_url(f"{base}/robots.txt")
        try:
            parser.read()
        except Exception:
            parser = None  # robots.txt unreachable: treat as no rules
        _robots[base] = parser
    parser = _robots[base]
    return True if parser is None else parser.can_fetch(USER_AGENT, url)


def extract_rpm(text):
    found = Counter()
    for m in RPM_FULL.findall(text):
        found[int(re.sub(r"[,.]", "", m))] += 1
    for m in RPM_SHORT.findall(text):
        found[SHORT_TO_RPM[m]] += 1
    return found.most_common(1)[0][0] if len(found) == 1 else None  # None if none or conflicting


def extract_interface(text):
    has_sata = re.search(r"\bSATA\b", text, re.I) is not None
    has_sas = re.search(r"\bSAS\b", text, re.I) is not None
    if has_sata and not has_sas:
        return "SATA"
    if has_sas and not has_sata:
        return "SAS"
    return None  # both or neither: ambiguous


def read_rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows):
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = read_rows()
    todo = [r for r in rows if r["source_url"] and (not r["rpm"] or not r["interface"]) and r["status"] != "checked"]
    print(f"{len(todo)} models to scrape (rows with a source_url and missing specs)")

    for row in todo:
        url = row["source_url"]
        print(f"\n{row['model']}: {url}")
        if not allowed_by_robots(url):
            print("  skipped: robots.txt does not allow this page")
            continue
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        except requests.RequestException as exc:
            print(f"  failed: {exc}")
            continue
        if resp.status_code != 200:
            print(f"  failed: HTTP {resp.status_code}")
            continue
        if "pdf" in resp.headers.get("Content-Type", "").lower():
            print("  skipped: PDF (read it by hand and fill the CSV)")
            continue

        text = BeautifulSoup(resp.text, "html.parser").get_text(" ", strip=True)
        rpm, interface = extract_rpm(text), extract_interface(text)
        print(f"  found rpm={rpm}, interface={interface}")
        if rpm and not row["rpm"]:
            row["rpm"] = str(rpm)
        if interface and not row["interface"]:
            row["interface"] = interface
        if rpm or interface:
            row["status"] = "scraped"
        time.sleep(DELAY_SECONDS)

    write_rows(rows)
    print("\nDone. Open reference/model_specs.csv and check any 'scraped' rows by eye.")


if __name__ == "__main__":
    main()
