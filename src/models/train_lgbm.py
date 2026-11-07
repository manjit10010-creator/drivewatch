"""Phase 4: LightGBM on the same split and metrics as the Phase 3 baselines (validation only).

Same data path as baselines.py: all positive training rows plus a hash sample of healthy rows
(--neg-frac), validation is NEVER sampled and is scored in 10-day chunks, so metrics are exact.
The test period is not touched.

Differences from the logistic regression: no scaling or log transforms (trees do not need them),
NULLs are passed to LightGBM as they are (it learns where missing values go), text columns such as
manufacturer become pandas categories.

The model is scored at several numbers of trees (--checkpoints) in one pass, so you can see where
validation PR-AUC stops improving without refitting.

Imbalance handling is controlled by two switches:
  --neg-frac   share of healthy training rows kept (undersampling)
  --spw        scale_pos_weight, the extra weight on positive rows (1.0 = no class weighting)

Run from the repo root, inside the drivewatch-ml Docker image:
    python src/models/train_lgbm.py
    python src/models/train_lgbm.py --spw 5 --tag spw5
    python src/models/train_lgbm.py --drop days_in_data days_since_first_nonzero --tag nodays
    python src/models/train_lgbm.py --grouped
"""
import argparse
import json
import os
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from dataset import (TRAIN, VAL, chunk_ranges, connect, day_numbers, feature_columns,
                     load_chunk, load_train_sample, serial_hashes)
from evaluate import evaluate_scores, print_report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="features_daily")
    p.add_argument("--neg-frac", type=float, default=0.03, help="share of healthy training rows kept")
    p.add_argument("--spw", type=float, default=1.0, help="scale_pos_weight (1.0 = no class weighting)")
    p.add_argument("--rounds", type=int, default=600, help="maximum number of trees")
    p.add_argument("--checkpoints", type=int, nargs="+", default=[100, 200, 400, 600],
                   help="score the model at these numbers of trees")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--min-child", type=int, default=100, help="min_child_samples")
    p.add_argument("--threads", type=int, default=12, help="CPU threads (leave some for Windows)")
    p.add_argument("--drop", nargs="*", default=[], help="drop feature columns whose name contains any of these")
    p.add_argument("--features-file", default=None,
                   help="use only these features: a .txt (one name per line) or reports/feature_selection.json "
                        "(all surviving features, ranked by permutation importance)")
    p.add_argument("--top-k", type=int, default=None, help="with the .json features file: keep the top K by importance")
    p.add_argument("--ks", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grouped", action="store_true", help="grouped-by-drive check inside the train period")
    p.add_argument("--tag", default="base", help="short name for this run (used in file names)")
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def load_feature_list(path, top_k=None):
    if path.endswith(".json"):
        with open(path) as f:
            imp = json.load(f)["step4_permutation_importance"]
        ranked = [n for n, _ in sorted(imp.items(), key=lambda t: -t[1])]
        return ranked[:top_k] if top_k else ranked
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    return names[:top_k] if top_k else names


def flatten_numbers(d, prefix=""):
    """Turn a nested results dict into {name: float} with MLflow-safe names."""
    out = {}
    for k, v in d.items():
        name = re.sub(r"[^A-Za-z0-9_.\- /]", "_", f"{prefix}{k}")
        if isinstance(v, dict):
            out.update(flatten_numbers(v, name + "."))
        elif isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
            out[name] = float(v)
    return out


def log_to_mlflow(args, params, feats, results, report_path, model_path, extra):
    import mlflow
    mlflow.set_tracking_uri("sqlite:////workspace/mlflow.db")
    mlflow.set_experiment("drivewatch-phase4")
    with mlflow.start_run(run_name=f"lgbm_{args.tag}"):
        mlflow.log_params({**params, "table": args.table, "neg_frac": args.neg_frac, "n_features": len(feats),
                           "train": str(TRAIN), "val": str(VAL), "grouped": args.grouped,
                           "dropped": ",".join(args.drop), **extra})
        for name, res in results.items():
            for k, v in flatten_numbers(res, f"{name}.").items():
                mlflow.log_metric(k, v)
        mlflow.log_text("\n".join(feats), "features.txt")
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(model_path)


def main():
    args = parse_args()
    t0 = time.time()
    con = connect()
    all_feats = feature_columns(con, args.table)
    base_feats = load_feature_list(args.features_file, args.top_k) if args.features_file else all_feats
    unknown = [c for c in base_feats if c not in all_feats]
    if unknown:
        raise SystemExit(f"Features not found in table {args.table}: {unknown}")
    feats = [c for c in base_feats if not any(s in c for s in args.drop)]
    dropped = [c for c in all_feats if c not in feats]
    eval_range = TRAIN if args.grouped else VAL
    label = "grouped split (20% of drives held out, train period)" if args.grouped else "time split (validation period)"
    src = f"from {args.features_file}" + (f" (top {args.top_k})" if args.top_k else "") if args.features_file else "all columns"
    print(f"Table {args.table}: {len(feats)} features used ({src}; {len(dropped)} left out). Evaluating on: {label}")

    tr = load_train_sample(con, args.table, feats, args.neg_frac, exclude_group=args.grouped)
    ytr = tr["label"].to_numpy()
    print(f"Train sample {TRAIN[0]}..{TRAIN[1]}: {len(tr):,} rows, {int(ytr.sum()):,} positives "
          f"({ytr.mean():.2%} of the sample)  [{time.time() - t0:.0f}s]")

    # text columns -> categories with the categories fixed from the training sample
    cat_levels = {c: sorted(tr[c].dropna().unique().tolist()) for c in feats if tr[c].dtype == object}
    if cat_levels:
        print("Categorical columns:", {c: len(v) for c, v in cat_levels.items()})

    def prep(df):
        X = df[feats].copy()
        for c, levels in cat_levels.items():
            X[c] = pd.Categorical(X[c], categories=levels)
        return X

    params = {"objective": "binary", "learning_rate": args.lr, "num_leaves": args.leaves,
              "min_child_samples": args.min_child, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 1.0, "scale_pos_weight": args.spw,
              "num_threads": args.threads, "seed": args.seed, "verbose": -1}
    Xtr = prep(tr)
    booster = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=args.rounds)
    print(f"LightGBM fitted ({args.rounds} trees)  [{time.time() - t0:.0f}s]")
    del tr, Xtr

    imp = booster.feature_importance("gain")
    order = np.argsort(-imp)[:15]
    print("Top features by gain (share of total):")
    for i in order:
        print(f"  {feats[i]:<46}{imp[i] / imp.sum():>8.1%}")

    checkpoints = sorted({c for c in args.checkpoints if c <= args.rounds} | {args.rounds})
    days, serials, ys = [], [], []
    scores = {it: [] for it in checkpoints}
    for a, b in chunk_ranges(*eval_range):
        df = load_chunk(con, args.table, feats, a, b, only_group=args.grouped)
        if df.empty:
            continue
        X = prep(df)
        days.append(day_numbers(df))
        serials.append(serial_hashes(df))
        ys.append(df["label"].to_numpy().astype("int8"))
        for it in checkpoints:
            scores[it].append(booster.predict(X, num_iteration=it, raw_score=True).astype("float32"))
        print(f"  scored {a}..{b}: {len(df):,} rows  [{time.time() - t0:.0f}s]")
    days, serials, y = np.concatenate(days), np.concatenate(serials), np.concatenate(ys)
    print(f"Evaluation set: {len(y):,} rows, {int(y.sum()):,} positives ({y.mean():.4%})")

    results = {f"lgbm_{args.tag}_{it}trees": evaluate_scores(days, serials, y, np.concatenate(scores[it]), args.ks)
               for it in checkpoints}
    print_report(results, args.ks)
    print("\nBar to beat (validation): logistic regression PR-AUC 0.1558, recall@1%FPR 0.574, P@50 0.422")

    suffix = f"{args.tag}{'_grouped' if args.grouped else ''}"
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    report_path = f"reports/lgbm_{suffix}.json"
    model_path = f"data/models/lgbm_{suffix}.txt"
    meta = {"table": args.table, "eval": label, "train": TRAIN, "val": VAL, "neg_frac": args.neg_frac,
            "n_features": len(feats), "dropped": dropped, "params": params, "seed": args.seed,
            "top_features_by_gain": {feats[i]: float(imp[i] / imp.sum()) for i in order}}
    with open(report_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    booster.save_model(model_path)
    print(f"\nSaved {report_path} and {model_path}")

    if not args.no_mlflow:
        try:
            log_to_mlflow(args, params, feats, results, report_path, model_path,
                          {"eval_rows": len(y), "eval_positives": int(y.sum()), "rounds": args.rounds})
            print("Logged to MLflow (mlflow.db in the repo root).")
        except Exception as e:  # never lose a finished run because of logging
            print(f"MLflow logging failed ({type(e).__name__}: {e}); results are still saved in reports/")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
