"""Phase 4: one global model against per-model-family models (validation only; the test period is not touched).

Three variants, same features (default: the top-25 list), same training sample, same settings:
  global            one LightGBM, no model column
  global_model_col  one LightGBM with the drive model as a categorical feature
  per_model         one LightGBM per drive model, each trained only on that model's rows

Scores from all variants are log-odds from models trained on the same healthy-row sampling rate, so pooling
the per-model scores for the overall ranking is consistent (each family's own base rate is built in).
Overall metrics are exact on the full validation period; per-model PR-AUC is also printed.

Run from the repo root, inside the drivewatch-ml Docker image:
    python src/models/compare_models.py
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from dataset import (TRAIN, VAL, chunk_ranges, connect, day_numbers, feature_columns, load_chunk,
                     load_train_sample, serial_hashes)
from evaluate import evaluate_scores, print_report
from train_lgbm import flatten_numbers, load_feature_list


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="features_daily")
    p.add_argument("--features-file", default="reports/selected_features_25.txt")
    p.add_argument("--neg-frac", type=float, default=0.03)
    p.add_argument("--rounds", type=int, default=300, help="maximum number of trees")
    p.add_argument("--checkpoints", type=int, nargs="+", default=[100, 200, 300])
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--leaves", type=int, default=15)
    p.add_argument("--min-child", type=int, default=500)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--ks", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    con = connect()
    feats = load_feature_list(args.features_file)
    unknown = [c for c in feats if c not in feature_columns(con, args.table)]
    if unknown:
        raise SystemExit(f"Features not found in {args.table}: {unknown}")
    print(f"{len(feats)} features from {args.features_file}")

    mapping = (con.execute(f"SELECT DISTINCT serial_number, model FROM {args.table}").df()
               .drop_duplicates("serial_number").set_index("serial_number")["model"])

    def with_model(df):
        if "model" in df.columns:
            return df
        if "serial_number" not in df.columns:
            raise SystemExit("The loaded data has neither a 'model' nor a 'serial_number' column; "
                             "paste me the load_train_sample code in dataset.py and I will adapt this script.")
        df = df.copy()
        df["model"] = df["serial_number"].map(mapping)
        return df

    tr = with_model(load_train_sample(con, args.table, feats, args.neg_frac))
    ytr = tr["label"].to_numpy()
    models = sorted(tr["model"].dropna().unique().tolist())
    print(f"Train sample: {len(tr):,} rows, {int(ytr.sum()):,} positives; drive models: {models}  [{time.time() - t0:.0f}s]")
    for m in models:
        mk = (tr["model"] == m).to_numpy()
        print(f"  {m:<20} {int(mk.sum()):>9,} rows  {int(ytr[mk].sum()):>7,} positives")

    params = {"objective": "binary", "learning_rate": args.lr, "num_leaves": args.leaves,
              "min_child_samples": args.min_child, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": args.threads, "seed": args.seed, "verbose": -1}

    def train(X, y):
        return lgb.train(params, lgb.Dataset(X, y), num_boost_round=args.rounds)

    b_global = train(tr[feats], ytr)
    Xb = tr[feats].copy()
    Xb["model"] = pd.Categorical(tr["model"], categories=models)
    b_col = train(Xb, ytr)
    b_per = {}
    for m in models:
        mk = (tr["model"] == m).to_numpy()
        b_per[m] = train(tr.loc[mk, feats], ytr[mk])
    print(f"Trained global, global+model column, and {len(models)} per-model boosters  [{time.time() - t0:.0f}s]")
    del tr, Xb

    cps = sorted({c for c in args.checkpoints if c <= args.rounds} | {args.rounds})
    variants = ("global", "global_model_col", "per_model")
    store = {(v, it): [] for v in variants for it in cps}
    days, serials, ys, midx = [], [], [], []
    for a, b in chunk_ranges(*VAL):
        df = load_chunk(con, args.table, feats, a, b)
        if df.empty:
            continue
        df = with_model(df)
        codes = pd.Categorical(df["model"], categories=models).codes.astype("int8")
        if (codes < 0).any():
            raise SystemExit("Validation contains a drive model that is not in the training sample.")
        X = df[feats]
        Xc = X.copy()
        Xc["model"] = pd.Categorical(df["model"], categories=models)
        days.append(day_numbers(df))
        serials.append(serial_hashes(df))
        ys.append(df["label"].to_numpy().astype("int8"))
        midx.append(codes)
        for it in cps:
            store[("global", it)].append(b_global.predict(X, num_iteration=it, raw_score=True).astype("float32"))
            store[("global_model_col", it)].append(b_col.predict(Xc, num_iteration=it, raw_score=True).astype("float32"))
            s = np.zeros(len(df), dtype="float32")
            for k, m in enumerate(models):
                mk = codes == k
                if mk.any():
                    s[mk] = b_per[m].predict(X[mk], num_iteration=it, raw_score=True)
            store[("per_model", it)].append(s)
        print(f"  scored {a}..{b}: {len(df):,} rows  [{time.time() - t0:.0f}s]")
    days, serials = np.concatenate(days), np.concatenate(serials)
    y, midx = np.concatenate(ys), np.concatenate(midx)
    print(f"Evaluation set: {len(y):,} rows, {int(y.sum()):,} positives ({y.mean():.4%})")

    results, per_model_ap = {}, {}
    for (v, it), parts in store.items():
        s = np.concatenate(parts)
        name = f"{v}_{it}trees"
        results[name] = evaluate_scores(days, serials, y, s, args.ks)
        per_model_ap[name] = {m: float(average_precision_score(y[midx == k], s[midx == k]))
                              for k, m in enumerate(models)}
    print_report(results, args.ks)

    print("\nPR-AUC within each drive model (validation):")
    print(f"{'variant':<30}" + "".join(f"{m:>20}" for m in models))
    for name, d in per_model_ap.items():
        print(f"{name:<30}" + "".join(f"{d[m]:>20.4f}" for m in models))
    for k, m in enumerate(models):
        mk = midx == k
        print(f"  {m}: {int(mk.sum()):,} validation rows, {int(y[mk].sum()):,} positives ({y[mk].mean():.3%})")
    print("\nReference (49 features, global, exact validation): PR-AUC 0.1925 at 200 trees; top-25 global 0.1914")

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/compare_models.json"
    meta = {"features_file": args.features_file, "n_features": len(feats), "params": params, "models": models,
            "train": TRAIN, "val": VAL, "neg_frac": args.neg_frac, "seed": args.seed}
    with open(report_path, "w") as f:
        json.dump({"meta": meta, "results": results, "per_model_prauc": per_model_ap}, f, indent=2)
    print(f"\nSaved {report_path}")

    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri("sqlite:////workspace/mlflow.db")
            mlflow.set_experiment("drivewatch-phase4")
            with mlflow.start_run(run_name="compare_models"):
                mlflow.log_params({"features_file": args.features_file, "n_features": len(feats), "neg_frac": args.neg_frac,
                                   "leaves": args.leaves, "min_child": args.min_child, "rounds": args.rounds,
                                   "seed": args.seed, "models": ",".join(models)})
                for name, res in results.items():
                    for k, v in flatten_numbers(res, f"{name}.").items():
                        mlflow.log_metric(k, v)
                mlflow.log_text("\n".join(feats), "features.txt")
                mlflow.log_artifact(report_path)
            print("Logged to MLflow.")
        except Exception as e:
            print(f"MLflow logging failed ({type(e).__name__}: {e}); results are saved in reports/")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
