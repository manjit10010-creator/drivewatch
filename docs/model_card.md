# DriveWatch model card (draft)

Status: draft. Metrics, final model and intended-use details are filled in after Phase 4.

## Data

Backblaze Drive Stats, 2025-01-01 to 2026-06-30 (credit: Backblaze). One operator's data centers.

## Known limitations

- **Single fleet.** Results come from one operator's fleet. SMART attributes differ by drive model, so the model will not transfer to another fleet without retraining.
- **Reappearing drives excluded.** 37 drives (0.54% of failed drives) that reappeared after a recorded failure are excluded from training and evaluation, because many of those "failures" look like drives pulled and returned to service (for example firmware updates), not real failures. This uses hindsight and cannot be applied when scoring live, so a live system may see such drives and score them as at-risk.
- **Partial source day.** Backblaze's file for 2025-05-12 is incomplete (about 70k drives missing, 2 failures that day). A few failure events on that day may be missing, and rolling features use date-based windows to avoid distortion.
- **Failure counts differ slightly from Backblaze's reports** (within about 2.5% per quarter); the cause is unexplained.
- **Sudden failures.** Drives that fail with no earlier SMART warning cannot be predicted from SMART data (to be quantified in error analysis).
## Phase 4 update: final model (validation results only)

Status: the test month (2026-05-01 to 2026-05-31) has NOT been used. All figures below are validation figures. Feature selection used the validation period, so they are slightly optimistic.

### Intended use
Rank drives by risk of failing within 30 days, so an operations team with a fixed weekly replacement budget can replace the riskiest drives first. It is a ranking tool, not a guarantee: many failures give no SMART warning.

### Model
- Global LightGBM (one model for all drive models), registered in MLflow as `drivewatch-risk`, version 1, alias `production` (seed 1, chosen in advance, not by validation score).
- 25 features (`reports/selected_features_25.txt`): top 25 by permutation importance after dropping constant and highly correlated columns.
- 200 trees, 15 leaves, min child samples 500, no class weights, no model column.
- Training data: 2025-02-01 to 2025-11-30; all positives plus a 3% deterministic sample of healthy rows (905,329 rows, 32,141 positives).
- Left-censored features (`days_in_data`, `days_since_first_nonzero_*`) are dropped.

### Data
- Backblaze Drive Stats, 2025-01-01 to 2026-06-30.
- Development sample: three drive models (TOSHIBA MG07ACA14TA, ST12000NM0008, TOSHIBA MG08ACA16TA), about 96k drives, 26% of the fleet.
- Label: 1 if the drive's first failure is 0 to 30 days after the scoring date. Rows after failure and rows with an unobserved window are dropped; 37 drives that reappear after failure are excluded.
- Split by time with 30-day gaps. Validation: 2026-01-01 to 2026-03-31 (8,556,532 rows, 9,668 positives, 0.113%), never sampled, so metrics are exact.

### Validation results
| Model | PR-AUC | ROC-AUC | Recall at 1% false alarms |
|---|---|---|---|
| Rule: any smart_5 > 0 | 0.0075 | n/a | n/a |
| Graded rule | 0.0256 | n/a | n/a |
| Logistic regression | 0.1558 | 0.861 | 0.574 |
| LightGBM final (mean of 5 seeds) | 0.188 (0.182 to 0.195) | 0.861 | 0.608 |

Random-guess PR-AUC is 0.113% (the positive rate). Seed noise is about 0.013 peak-to-peak, so differences of 0.01 or less between configurations are ties.

Grouped check (20% of drives held out): LightGBM 0.174 to 0.178 vs logistic regression 0.150.

### Business view (13 weekly decisions, replaced drives leave the pool; 488 distinct drives fail in the validation labels)
| Replacements per week | Model catches | Rule (smart_5 + smart_197) | Random |
|---|---|---|---|
| 20 | 75 (15.4%, precision 28.9%) | 29 | 0.2 |
| 50 | 109 (22.3%, precision 16.8%) | 49 | 0.9 |
| 100 | 126 (25.9%, precision 9.7%) | 77 | 2.1 |

Weekly precision at top K (0.54 / 0.44 / 0.37 for K = 20 / 50 / 100) re-counts drives across weeks and overstates; quote this table instead.

### Known weaknesses
- Roughly three quarters of failures are missed even at 100 replacements per week; sudden failures with no SMART warning are expected.
- Predictability varies by drive model: PR-AUC about 0.29 for ST12000NM0008, 0.18 for MG07ACA14TA, 0.09 for MG08ACA16TA.
- The model relies on lifetime counters (`smart_9_raw`, `smart_12_raw`, `smart_193_raw`) whose effect is not a clear physical signal; they may mark drive cohorts. Two SHAP directions are unexplained (`delta30_smart_197` negative, `n_obs_30` negative).
- Trained and validated on three drive models from one operator's fleet. It will not transfer to other models or fleets without retraining.
- Not yet checked on the full fleet or on the test month.

### Phase 4 gate
Best model beats the baselines on validation (0.188 vs 0.156 logistic regression and 0.026 best rule), it is registered, and this model card is drafted.
