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

## Phase 4: LightGBM and feature selection (2026-11-07, validation only; test period untouched)

**Setup.** LightGBM trained on all positive rows plus 3% of healthy rows (train 2025-02-01..2025-11-30). Metrics are exact on the full validation period (8,556,532 rows, 9,668 positives, 0.113%). Every run is logged in MLflow (experiment `drivewatch-phase4`) and saved in `reports/lgbm_*.json`. All numbers are single-seed; run-to-run noise is about 0.005 PR-AUC, so smaller differences are treated as ties.

**Result.** Best validation PR-AUC is about 0.19 (logistic regression 0.1558; rule baselines 0.0075 and 0.0256). ROC-AUC is not better than the logistic regression (about 0.86 for both). Recall at 1% false-alarm rate is about 0.60 (logistic regression 0.574).

**Decisions.**
1. Dropped the left-censored features `days_in_data` and `days_since_first_nonzero_*`: PR-AUC 0.1904 without against 0.1891 with them (no cost), and the model now relies on `smart_197_raw` and `err_days30_smart_197` instead of a feature whose meaning shifts over calendar time.
2. No class weighting: `scale_pos_weight` 5 gave 0.1812 against 0.1891. Undersampling healthy rows to 3% is kept; 10% gave 0.1785 at 100 trees for about 3x the data.
3. Use 100 to 200 trees: validation PR-AUC fell with more trees in every run. Regularization (15 leaves, min child 500) changed the best score by about 0.003, which is within noise.
4. Feature selection (`src/models/select_features.py`, `reports/feature_selection.json`): 8 constant `miss_smart_*` flags dropped; 12 highly correlated duplicates dropped (|rho| >= 0.95, simpler feature kept); 29 features left; ranked by permutation importance. Five features did not hurt when permuted: `delta7_smart_198`, `delta7_smart_199`, `temp_std_7`, `smart_199_raw`, `smart_198_raw`.
5. Selected set: the top 25 by permutation importance (`reports/selected_features_25.txt`). Exact validation PR-AUC at 200 trees: all 49 features 0.1925, 29 features 0.1873, top 25 0.1914, top 15 0.1869. These are within noise of each other, so this is a judgment, not a pass of the 1% rule: the top 25 keeps temperature and the raw reallocated-sector count, which the top 15 loses.

**Checks.** Grouped by drive (20% of drives held out, train period): PR-AUC 0.174 to 0.178 against 0.150 for the logistic regression, so the model generalises to unseen drives. SHAP: reallocated sectors (5), pending sectors (197), uncorrectable errors (187) and temperature raise risk, as expected.

**Open items and known limits.**
- `smart_9_raw` (power-on hours), `smart_12_raw` (power cycles) and `smart_193_raw` (load cycles) are important in SHAP, but their effect is not a clear physical failure signal. They may identify drive cohorts or models. Removing them lowered PR-AUC by about 0.005 to 0.008. To be tested with per-model-family models.
- The SHAP direction of `delta30_smart_197` (negative) and `n_obs_30` (negative) is unexplained; rank correlation is only a rough summary of non-monotone effects.
- The weighted PR-AUC printed by `select_features.py` is about 0.02 lower than the exact value (0.171 against 0.1914 for the same 25 features). Use it only to compare feature sets, never as a reported result.
- Feature selection used the validation period, so the validation score of the chosen set is slightly optimistic; the test period is the unbiased figure.


## Phase 4: LightGBM and feature selection (2026-11-07, validation only; test period untouched)

**Setup.** LightGBM trained on all positive rows plus 3% of healthy rows (train 2025-02-01..2025-11-30). Metrics are exact on the full validation period (8,556,532 rows, 9,668 positives, 0.113%). Every run is logged in MLflow (experiment `drivewatch-phase4`) and saved in `reports/lgbm_*.json`. All numbers are single-seed; run-to-run noise is about 0.005 PR-AUC, so smaller differences are treated as ties.

**Result.** Best validation PR-AUC is about 0.19 (logistic regression 0.1558; rule baselines 0.0075 and 0.0256). ROC-AUC is not better than the logistic regression (about 0.86 for both). Recall at 1% false-alarm rate is about 0.60 (logistic regression 0.574).

**Decisions.**
1. Dropped the left-censored features `days_in_data` and `days_since_first_nonzero_*`: PR-AUC 0.1904 without against 0.1891 with them (no cost), and the model now relies on `smart_197_raw` and `err_days30_smart_197` instead of a feature whose meaning shifts over calendar time.
2. No class weighting: `scale_pos_weight` 5 gave 0.1812 against 0.1891. Undersampling healthy rows to 3% is kept; 10% gave 0.1785 at 100 trees for about 3x the data.
3. Use 100 to 200 trees: validation PR-AUC fell with more trees in every run. Regularization (15 leaves, min child 500) changed the best score by about 0.003, which is within noise.
4. Feature selection (`src/models/select_features.py`, `reports/feature_selection.json`): 8 constant `miss_smart_*` flags dropped; 12 highly correlated duplicates dropped (|rho| >= 0.95, simpler feature kept); 29 features left; ranked by permutation importance. Five features did not hurt when permuted: `delta7_smart_198`, `delta7_smart_199`, `temp_std_7`, `smart_199_raw`, `smart_198_raw`.
5. Selected set: the top 25 by permutation importance (`reports/selected_features_25.txt`). Exact validation PR-AUC at 200 trees: all 49 features 0.1925, 29 features 0.1873, top 25 0.1914, top 15 0.1869. These are within noise of each other, so this is a judgment, not a pass of the 1% rule: the top 25 keeps temperature and the raw reallocated-sector count, which the top 15 loses.

**Checks.** Grouped by drive (20% of drives held out, train period): PR-AUC 0.174 to 0.178 against 0.150 for the logistic regression, so the model generalises to unseen drives. SHAP: reallocated sectors (5), pending sectors (197), uncorrectable errors (187) and temperature raise risk, as expected.

**Open items and known limits.**
- `smart_9_raw` (power-on hours), `smart_12_raw` (power cycles) and `smart_193_raw` (load cycles) are important in SHAP, but their effect is not a clear physical failure signal. They may identify drive cohorts or models. Removing them lowered PR-AUC by about 0.005 to 0.008. To be tested with per-model-family models.
- The SHAP direction of `delta30_smart_197` (negative) and `n_obs_30` (negative) is unexplained; rank correlation is only a rough summary of non-monotone effects.
- The weighted PR-AUC printed by `select_features.py` is about 0.02 lower than the exact value (0.171 against 0.1914 for the same 25 features). Use it only to compare feature sets, never as a reported result.
- Feature selection used the validation period, so the validation score of the chosen set is slightly optimistic; the test period is the unbiased figure.


## Phase 4: global model against per-model-family models (2026-11-07, validation only; test period untouched)

Same top-25 features, LightGBM settings (15 leaves, min child 500) and 3% healthy-row sampling for all variants; exact validation metrics; single seed (`src/models/compare_models.py`, `reports/compare_models.json`).

| variant (200 trees) | PR-AUC | recall@1%FPR | P@50 |
|---|---|---|---|
| global, no model column | 0.1914 | 0.606 | 0.438 |
| global + `model` as a feature | 0.1859 | 0.606 | 0.443 |
| one model per drive model (scores pooled) | 0.1761 | 0.592 | 0.438 |

PR-AUC within each drive model at 200 trees (global / global + model column / per-model): ST12000NM0008 0.294 / 0.289 / 0.291; TOSHIBA MG07ACA14TA 0.181 / 0.183 / 0.179; TOSHIBA MG08ACA16TA 0.093 / 0.090 / 0.086.

**Decision.** Use a single global model without a model column (200 trees, 3% healthy-row sampling). The per-model models were lower overall at every tree count (by 0.010 to 0.018), probably because each has only 9,400 to 12,400 training positives instead of 32,100 (hypothesis, not tested). Adding the model as a feature made no difference within noise. Within each drive model, the global model ranks failures as well as the dedicated model, which argues against the lifetime counters (`smart_9_raw`, `smart_12_raw`, `smart_193_raw`) acting only as drive-model markers.

**Limits.** Single seed (noise about 0.005 PR-AUC). Predictability differs a lot by drive model: PR-AUC about 0.29 for ST12000NM0008 against about 0.09 for MG08ACA16TA. Only the three development-sample models were compared; widening to all models is still to do. Results may not transfer to other fleets or drive models without retraining.


## Phase 4: final configuration, seed check and business view (2026-11-07, validation only; test period untouched)

**Final configuration.** Global LightGBM, 25 features (`reports/selected_features_25.txt`), 200 trees, 15 leaves, min child 500, all positives plus 3% of healthy rows, no class weighting, no model column. Trained with 5 seeds (`src/models/final_model.py`, `reports/final_validation.json`). The model registered for serving is seed 1, chosen in advance (not by validation score); an ensemble of seeds is an option for later.

**Validation, exact (mean, range over 5 seeds).** PR-AUC 0.188 (0.182 to 0.195); ROC-AUC 0.861; recall at 1% false-alarm rate 0.608 (0.606 to 0.611); recall at 0.1% 0.32; weekly precision in the top 20 / 50 / 100: 0.54 / 0.44 / 0.37. Baselines: logistic regression PR-AUC 0.156; rules 0.0075 and 0.0256.

**Business view.** Every 7 days the K riskiest drives not yet replaced are replaced; a failing drive counts as caught if picked on a day when it fails within 30 days; 13 weekly decisions; 488 distinct drives fail within the validation labels.

| K per week | model caught (range over seeds) | recall | precision | rule: smart_5 + smart_197 | random (mean of 20) |
|---|---|---|---|---|---|
| 20 | 75.2 (71 to 78) | 15.4% | 28.9% | 29 (5.9%) | 0.2 |
| 50 | 109.0 (106 to 112) | 22.3% | 16.8% | 49 (10.0%) | 0.9 |
| 100 | 126.4 (122 to 132) | 25.9% | 9.7% | 77 (15.8%) | 2.1 |

The weekly precision at top K (0.54 at K = 20) is about twice the distinct-drive precision (28.9%) because it re-counts the same failing drive in consecutive weeks. Report the business-view figures to the operations persona.

**Correction to earlier Phase 4 entries: seed noise.** The entries above assumed run-to-run noise of about 0.005 PR-AUC. The 5-seed range is 0.013 (0.1816 to 0.1949). Differences below about 0.01 in the single-seed comparisons are therefore ties: the feature-set sizes (49, 29, 25, 15), regularization, 10% sampling, class weight 5, and the removal of the lifetime counters (0.005 to 0.008). The conclusions that still hold: every seed beats the logistic regression; the per-model models were below the global model's lowest seed at all three tree counts; adding a model column made no difference. The choices of no class weighting, 3% sampling and the 25-feature set stand because they are simpler, not because they were shown to be better.

**Limits.** Development sample only (3 drive models, about 26% of the fleet); widening to all models is still to do before final results. Business-view simplifications: one decision per 7 days, early picks count as caught if the drive fails within 30 days, failures between decision days with no earlier warning cannot be caught. The test period has not been used.
