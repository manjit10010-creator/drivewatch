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

## Phase 2 decisions (2026-11-05)

- **zstd compression for silver Parquet.** Set `spark.sql.parquet.compression.codec` to `zstd`. One month of silver dropped from 610 MB to 477 MB.
- **`reference/` folder for hand-maintained data.** `reference/model_specs.csv` lives at the repo root, not under `data/`, because `data/` is git-ignored and the specs should be versioned.
- **Drive specs came from web searches, not a scraper.** Retailer pages and Toshiba datasheets were used. `scrape_specs.py` was written but never run; it is kept for the scraping learning goal.
- **`release_year` is left blank on purpose.** Sources gave conflicting kinds of dates (announcement, launch, retail), so none is trusted yet.
- **`rpm` and `interface` carry almost no signal.** All 21 filled models are SATA 7200 rpm, so these are constant across about 96% of the fleet.
- **Q2 2026 failure count is +24 (1.6%) above Backblaze's report** (1,522 vs 1,498). Cause unexplained. SSD/boot models had zero failures, so boot drives are not the reason. Gate still passes at 1.6%.
- **Dev sample:** TOSHIBA MG07ACA14TA, ST12000NM0008, TOSHIBA MG08ACA16TA. About 96k drives (26% of the fleet) and about 750 failures in H1 2026.
- **Adding 2025 Q3 and Q4** so rolling features and a time-based split with a 30-day gap have enough history.
## Failure counts vs Backblaze reports (2026-11-05)

- Our failure rows vs Backblaze: 2025 Q1 1,067 vs 1,093 (-2.4%), Q3 1,254 vs 1,273 (-1.5%), full year 4,327 vs 4,317 (+0.2%), 2026 Q1 1,030 vs 1,030, Q2 1,522 vs 1,498 (+1.6%). Gate passed (within about 2.5%).
- Differences are unexplained. Backblaze reports apply filters we do not, and quarter assignment may differ.
- `fact_failures` keeps each drive's first failure only, so quarter counts from it undercount the 18 repeat failures. Compare against failure rows in `fact_daily_smart` instead.

## Reappearing drives excluded from labels (2026-11-06)

- **Decision:** drives that report again after their first failure date (37 drives, 0.54% of failed drives) are excluded entirely from the labelled dataset used for training and evaluation.
- **Why:** many of these "failures" look like drives pulled and returned to service (Backblaze notes firmware-update pulls for Toshiba MG08ACA16TEY). A positive label on a drive that was never unhealthy is noise.
- **How:** the rule is computed from the data when labels are built (any drive with rows after its first failure date), not a hard-coded list, and the excluded drives are written to a table with the reason.
- **Trade-off:** it uses hindsight, which is fine when cleaning a training set but impossible when scoring live. It is stated as a limit in the model card.

## Label definition and EDA class-balance correction
- Label = 1 if the drive's first failure date is 0 to H days after the scoring date (failure-day row counts as positive). Rows after the first failure and rows whose H-day window is not fully observed are dropped. The 37 drives that reappear after failing are excluded and listed in `excluded_drives`.
- The EDA notebook's 0.1164% positive rate (30 days, H1 2026) was inconsistent: its numerator counted rows from the last 30 days, which its denominator excluded. Reproducing it without the cutoff gives 60,406 positives against the notebook's 60,436 (difference probably repeat failures). With the cutoff applied the figure is 0.1127% (failure-day row counted) or 0.1084% (excluded).
- Dev sample, H=30: 49,519,299 rows, 97,260 drives, 54,853 positives (0.1108%).

## Phase 3 baselines (dev sample, H = 30)
- Split: train 2025-02-01..2025-11-30, validation 2026-01-01..2026-03-31, test 2026-05-01..2026-05-31 (untouched); 30-day gaps between periods.
- Validation (8,556,532 rows, 9,668 positives, 0.113%): rule "any reallocated sectors" PR-AUC 0.0075; graded error-count rule 0.0256; logistic regression (class weights, 3% of healthy rows) 0.1558, recall 0.574 at 1% false alarms, precision 0.515 at top 20 per week.
- Grouped check (20% of drives held out): logistic regression PR-AUC 0.150.
- To test in Phase 4: drop days_in_data and days_since_first_nonzero_* (left-censored by the 2025-01-01 data start).