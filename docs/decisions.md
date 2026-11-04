# Decisions log

Short notes on why choices were made. Add one entry per real decision.

## 2026-11-03: Scraper user-agent
Generic project name, no personal contact details. Add a contact address yourself if you want sites to be able to reach you.

## 2026-11-03: Bronze layout
`data/bronze/<year>/<file>.zip`, with `data/manifest.json` as the source of truth for download status and checksums.

## Backblaze terms and robots.txt (checked 2026-11-03)

**Terms of use for the Drive Stats data** (from the Backblaze Drive Stats page):
1. Cite Backblaze as the source. The README credits "Backblaze Drive Stats".
2. I am solely responsible for how I use the data.
3. I may not sell the dataset itself. DriveWatch uses it for a learning project and portfolio, not for resale.

**robots.txt** (https://www.backblaze.com/robots.txt, read 2026-11-03):
- The rules for all crawlers (`User-agent: *`) block only a few paths: `/api/install_backblaze`, `/win32/`, `/mac/`, `/linux/`, `/gift/`, `/gift_download/`, `/gen/`, `/fix_billing_problem.htm`, `/partials/`, `/feed/` and `/?`.
- The Drive Stats page that my scraper reads is not on that list, so scraping it is allowed.
- The file also has a content signal that allows search, AI input and AI training.
- Decision: the scraper keeps a descriptive user-agent string, waits between requests, and skips files it already has.
- The zip files are served from a different host (check the URLs in `data/manifest.json`), so this robots.txt applies only to the page the scraper reads.

**Data credit:** Backblaze Drive Stats, https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data

## Spark threads (2026-11-04)
CPU is an i9-13900HX (24 cores, 32 logical). The plan's rule (cores minus 2) gives local[22], but with about 6 GB for Spark that leaves about 270 MB per task and risks out-of-memory errors. Decision: start with local[8] (about 750 MB per task); try local[12] if memory allows. WSL2 is capped at 8 GB RAM, 4 GB swap and 12 processors in .wslconfig.