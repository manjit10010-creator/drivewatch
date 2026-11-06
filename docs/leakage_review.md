# Leakage review (Phase 3 gate)

Date: 2026-11-06. Scope: `features_daily` (dev sample: TOSHIBA MG07ACA14TA, ST12000NM0008, TOSHIBA MG08ACA16TA), labels at H = 30, baselines in `src/models/baselines.py`.

## Rule
A feature for scoring day D may use only SMART readings dated D or earlier. The label looks forward; nothing else may.

## Checks and results

| Check | How | Result |
|---|---|---|
| All rolling windows end on the scoring day | Every window is `... AND CURRENT ROW`, date-based (integer day number), so the 2025-05-12 gap cannot shift a window | Recomputed `temp_mean_7` and `delta7_smart_5` independently from `fact_daily_smart` for 300 random rows: 0 mismatches. The other features use the same window definitions but were NOT each re-computed |
| Counters' event features look backward only | `days_since_first_nonzero_*` uses `UNBOUNDED PRECEDING .. CURRENT ROW`; `err_days30_*` uses the 30-day window ending on D | Code review only |
| Label and failure information are not model inputs | `days_to_failure` and `label` are stored in `features_daily` but excluded from the feature list (`META` in `src/models/dataset.py`) | 62 table columns minus 6 non-feature columns = 56 features used, as printed by the baseline run |
| Preprocessing fitted on training data only | Medians, means and standard deviations come from the training sample; validation rows are only transformed | Code review (`Preprocessor.fit` is called once, on the train sample) |
| Split is by time with a gap | Train 2025-02-01..2025-11-30, validation 2026-01-01..2026-03-31, test 2026-05-01..2026-05-31; gaps of 31 and 30 days, so no label window crosses a boundary | By construction |
| No random split | No random row split anywhere | By construction |
| Test period untouched | `dataset.py` and `baselines.py` never query the test dates | Code review; test is to be used once, in the final report |
| Undersampling does not leak | Healthy rows are undersampled in the training period only, by a deterministic hash; validation is never sampled | By construction |

## Known limits (state these in the model card)
1. **Hindsight in training data selection.** The 37 drives that reappear after their first failure are excluded from labelled data. The rule needs future data, so it cannot be applied live. It affects which rows are used for training and evaluation, not any feature value.
2. **Same drives in train and validation.** The time split puts the same drives in both periods (as in deployment, where drives are scored day after day). The grouped-by-drive run is the check for generalisation to unseen drives.
3. **Left-censored features.** `days_in_data` and `days_since_first_nonzero_*` depend on the data starting on 2025-01-01, so their meaning shifts over calendar time. Not leakage, but a drift risk. To be tested without them in feature selection.
4. **Temperature rule.** `temp_c` is the low byte of `smart_194_raw`, a rule chosen here and not a documented fact.
Grouped-by-drive check (20% of drives held out): PR-AUC 0.150 against 0.156 on the time split, so the model generalises to unseen drives.

## Status
Gate item "leakage review done": complete for the checks above. Items to repeat when features change: the independent recomputation and the feature-list check.
