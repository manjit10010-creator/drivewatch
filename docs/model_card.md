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