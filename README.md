# DriveWatch

DriveWatch scores every hard drive in a fleet each day for its chance of failing within 30 days, using Backblaze's public Drive Stats data. It is built end to end: scrape and download, process with Spark, store in a DuckDB warehouse, train and select a model, serve it with FastAPI, and explore it in a Streamlit app, all run by a daily pipeline.

## Problem framing

The question DriveWatch answers: **which drives in the fleet should be replaced first, before they fail?**

| Item | Definition |
|---|---|
| Persona | Data-center operations lead who can replace a limited number of drives per week |
| Prediction unit | One drive on one day (a row per drive per day) |
| Target (label) | 1 if the drive fails within the next 30 days, else 0. Horizon is a parameter (7, 14 and 30 days can all be tested) |
| Output | Risk score per drive per day, and a ranked replace-first list |
| Stretch target | Remaining useful life in days (regression or survival model) |
| Success metrics | Precision at top K (if ops can replace 50 drives a week, how many were real failures), recall at a fixed false-alarm rate, PR-AUC |
| Metrics to avoid | Plain accuracy: with failures under 1 in 1,000 rows, predicting "never fails" scores above 99% |
| Service targets | API answers in under 500 ms; daily pipeline finishes before the next day's file arrives; app loads in under 3 s |
| Out of scope | Real-time streaming, mixing in other vendors' data, hardware-level root-cause analysis |

### Assumptions and limits

- The data comes from one operator's data centers (Backblaze).
- SMART attributes differ by drive model.
- Results will not transfer to another fleet without retraining.

## Data source and credit

Data: [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). Thank you to Backblaze for publishing it. Check their terms of use before redistributing anything derived from it. This repo does not include the data.

## Scale ladder

1. Sample: 1 drive model, 1 year. Used for all development.
2. Medium: all models, 2 to 3 years. Used for model comparison.
3. Full: all available years. Used once, for the final run.

## Repo layout

```
src/ingest/      scraper.py, downloader.py
src/etl/         Spark jobs, quality checks
src/features/    build_features.py
src/models/      train.py, evaluate.py
src/api/         FastAPI service
app/             Streamlit app
pipelines/       daily DAG
notebooks/       EDA and features
tests/
docs/            model_card.md, decisions.md
data/            git-ignored: bronze/ silver/ gold/
```

## Phase 1 quick start

```powershell
# inside WSL2 or a Windows venv
python -m venv .venv
.venv\Scripts\activate          # WSL: source .venv/bin/activate
pip install -r requirements.txt

# 1. Build the manifest (no downloads yet)
python -m src.ingest.scraper

# 2. Look at it, then download one year
python -m src.ingest.downloader --years 2023
```

## Status

- [x] Phase 1: problem framing, repo scaffold, scraper and downloader code
- [ ] Phase 1: first download, Parquet conversion, first EDA notebook
- [ ] Phase 2: ETL and warehouse
- [ ] Phase 3: features
- [ ] Phase 4: modeling
- [ ] Phase 5: serving
- [ ] Phase 6: automation and ship
