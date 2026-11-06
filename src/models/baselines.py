"""Phase 3 baselines: rule baseline and logistic regression, time-based split, validation only.

Rule baselines (the bar any model must beat):
  rule_any_realloc   : flag a drive when smart_5_raw (reallocated sectors) > 0   (the plan's example)
  rule_graded_errors : rank by log1p(smart_5 + smart_197 + smart_198 raw), for the ranking metrics
Logistic regression: class_weight='balanced', trained on all positive rows plus a sample of the negative
rows (undersampling), standardised features (see dataset.Preprocessor).
Validation rows are NEVER sampled: all of them are scored in 10-day chunks, so the metrics are exact.
The test period is not touched.

Run from the repo root, inside the drivewatch-ml Docker image:
    python src/models/baselines.py
    python src/models/baselines.py --grouped      (second check: hold out 20% of drives inside the train period)
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from dataset import (TRAIN, VAL, Preprocessor, chunk_ranges, connect, day_numbers, feature_columns,
                     load_chunk, load_train_sample, serial_hashes)
from evaluate import binary_rule_summary, evaluate_scores, print_report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="features_daily")
    p.add_argument("--neg-frac", type=float, default=0.03, help="share of healthy training rows kept")
    p.add_argument("--ks", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grouped", action="store_true", help="grouped-by-drive check inside the train period")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    con = connect()
    feats = feature_columns(con, args.table)
    eval_range = TRAIN if args.grouped else VAL
    label = "grouped split (20% of drives held out, train period)" if args.grouped else "time split (validation period)"
    print(f"Table {args.table}: {len(feats)} feature columns. Evaluating on: {label}")

    tr = load_train_sample(con, args.table, feats, args.neg_frac, exclude_group=args.grouped)
    ytr = tr["label"].to_numpy()
    print(f"Train sample {TRAIN[0]}..{TRAIN[1]}: {len(tr):,} rows, {int(ytr.sum()):,} positives "
          f"({ytr.mean():.2%} of the sample; {args.neg_frac:.0%} of healthy rows kept)  [{time.time() - t0:.0f}s]")

    pp = Preprocessor(feats).fit(tr)
    Xtr = pp.transform(tr)
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=args.seed)
    lr.fit(Xtr, ytr)
    print(f"Logistic regression fitted ({Xtr.shape[1]} inputs)  [{time.time() - t0:.0f}s]")
    top = np.argsort(-np.abs(lr.coef_[0]))[:12]
    print("Largest coefficients (standardised inputs; sign = direction of risk):")
    for i in top:
        print(f"  {pp.names[i]:<46}{lr.coef_[0][i]:>8.3f}")
    del tr, Xtr

    days, serials, ys, s_lr, s_graded, f_any = [], [], [], [], [], []
    for a, b in chunk_ranges(*eval_range):
        df = load_chunk(con, args.table, feats, a, b, only_group=args.grouped)
        if df.empty:
            continue
        days.append(day_numbers(df))
        serials.append(serial_hashes(df))
        ys.append(df["label"].to_numpy().astype("int8"))
        s_lr.append(lr.decision_function(pp.transform(df)).astype("float32"))
        errs = df["smart_5_raw"].fillna(0) + df["smart_197_raw"].fillna(0) + df["smart_198_raw"].fillna(0)
        s_graded.append(np.log1p(errs.clip(lower=0).to_numpy()).astype("float32"))
        f_any.append((df["smart_5_raw"].fillna(0).to_numpy() > 0))
        print(f"  scored {a}..{b}: {len(df):,} rows  [{time.time() - t0:.0f}s]")
    days, serials, y = np.concatenate(days), np.concatenate(serials), np.concatenate(ys)
    s_lr, s_graded, f_any = np.concatenate(s_lr), np.concatenate(s_graded), np.concatenate(f_any)
    print(f"Evaluation set: {len(y):,} rows, {int(y.sum()):,} positives ({y.mean():.4%})")

    results = {
        "rule_any_realloc": evaluate_scores(days, serials, y, f_any.astype("float32"), args.ks),
        "rule_graded_errors": evaluate_scores(days, serials, y, s_graded, args.ks),
        "logreg_balanced": evaluate_scores(days, serials, y, s_lr, args.ks),
    }
    results["rule_any_realloc"]["binary_rule"] = binary_rule_summary(y, f_any)
    print_report(results, args.ks)
    b = results["rule_any_realloc"]["binary_rule"]
    print(f"\nRule 'any reallocated sectors > 0' as a yes/no alarm: flags {b['flagged_share']:.2%} of drive-days, "
          f"precision {b['precision']:.2%}, recall {b['recall']:.1%}")

    os.makedirs("reports", exist_ok=True)
    out = f"reports/baselines_{args.table}{'_grouped' if args.grouped else ''}.json"
    meta = {"table": args.table, "eval": label, "train": TRAIN, "val": VAL, "neg_frac": args.neg_frac,
            "n_features": len(feats), "seed": args.seed}
    with open(out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"\nSaved {out}   (done in {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
