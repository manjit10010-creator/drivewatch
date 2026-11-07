"""Derive risk-band thresholds from the VALIDATION period, using the production model from the registry.

The model was trained on a 3% sample of healthy rows, so its raw scores are inflated and a fixed cut-off
such as 0.5 would mean nothing. Instead the bands are tied to false-alarm rates measured on validation:

    high    score >= the score that only 1% of healthy drive-days reach   (recall about 0.6)
    medium  score >= the score that only 5% of healthy drive-days reach
    low     everything else
    (the 0.1% level is stored for information)

It also reports PR-AUC of the registry model on the full validation period. For model version 1 (seed 1)
this should be about 0.1908 with recall at 1% false-alarm rate about 0.607: a check that the registry model
and features_daily reproduce the Phase 4 numbers. The test period is not touched.

Run from the repo root, inside the drivewatch-ml Docker image:
    python src/models/risk_bands.py
"""
import argparse
import json
import time
from datetime import date

import duckdb
import numpy as np
from sklearn.metrics import average_precision_score

from score_fleet import DB_PATH, date_chunks, load_production_model, predict, read_chunk


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="features_daily")
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--end", default="2026-03-31")
    p.add_argument("--out", default="reports/risk_bands.json")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    if date.fromisoformat(args.end) >= date(2026, 5, 1):
        raise SystemExit("Refusing to score the test period (2026-05-01 onwards). Use validation dates only.")

    booster, version = load_production_model()
    feats = booster.feature_name()
    print(f"Loaded production model version {version}, {len(feats)} features")

    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("SET threads=8")
    d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
    cols = ["date", "label"] + feats

    score_parts, label_parts = [], []
    for a, b in date_chunks(d0, d1):
        df = read_chunk(con, args.table, cols, a, b)
        if df.empty:
            continue
        score_parts.append(predict(booster, df, feats).astype("float32"))
        label_parts.append(df["label"].to_numpy(dtype="int8"))
        print(f"  scored {a}..{b}: {len(df):,} rows  ({time.time() - t0:.0f}s)")
    scores = np.concatenate(score_parts)
    labels = np.concatenate(label_parts)
    pos, neg = scores[labels == 1], scores[labels == 0]
    print(f"\nValidation rows: {len(scores):,}, positives: {len(pos):,} ({len(pos) / len(scores):.3%})")
    pr_auc = float(average_precision_score(labels, scores))
    print(f"PR-AUC of the registry model on validation: {pr_auc:.4f}  (Phase 4, seed 1: 0.1908)")

    levels = {"high": 0.99, "medium": 0.95, "very_high_info": 0.999}
    thresholds = {k: float(np.quantile(neg, q)) for k, q in levels.items()}
    stats = {}
    print(f"\n{'level':<16} {'false-alarm rate':>16} {'threshold':>10} {'recall':>8} {'share of drive-days flagged':>28}")
    for k, q in levels.items():
        t = thresholds[k]
        recall = float((pos >= t).mean())
        flagged = float((scores >= t).mean())
        stats[k] = {"false_alarm_rate": round(1 - q, 4), "recall": recall, "share_flagged": flagged}
        print(f"{k:<16} {1 - q:>16.1%} {t:>10.4f} {recall:>8.3f} {flagged:>28.2%}")

    out = {
        "meta": {"model": "drivewatch-risk", "version": version, "table": args.table, "start": args.start,
                 "end": args.end, "rows": int(len(scores)), "positives": int(len(pos)), "pr_auc": pr_auc},
        "thresholds": {"high": thresholds["high"], "medium": thresholds["medium"],
                       "very_high_info": thresholds["very_high_info"]},
        "stats": stats,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}  ({time.time() - t0:.0f}s)")
    con.close()


if __name__ == "__main__":
    main()
