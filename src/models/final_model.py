"""Phase 4: final model configuration, multi-seed validation check, and the business view (validation only;
the test period is not touched).

Model: global LightGBM, top-25 feature list, 200 trees, 15 leaves, min child 500, all positives + 3% healthy rows.
It is trained with several seeds (the training sample itself is fixed, so seeds change only LightGBM's
randomness). Every model is saved in data/models/ so the one-time test run can use exactly these models.

Business view (simulation on the validation period):
  Every 7 days (the first scored day of each block) the policy picks the K highest-scoring drives that have not
  been replaced yet and replaces them. A failing drive counts as caught if it is picked on a day when its label
  is 1 (it fails within 30 days of that day). Replaced drives leave the pool. The denominator is the number of
  distinct drives that fail within the validation labels. Policies: the model, a simple rule (highest
  smart_5 + smart_197 raw counts) and random picks (mean of 20 draws).

Run from the repo root, inside the drivewatch-ml Docker image:
    python src/models/final_model.py
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np

from dataset import TRAIN, VAL, chunk_ranges, connect, day_numbers, feature_columns, load_chunk, \
    load_train_sample, serial_hashes
from evaluate import evaluate_scores, print_report
from train_lgbm import flatten_numbers, load_feature_list


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="features_daily")
    p.add_argument("--features-file", default="reports/selected_features_25.txt")
    p.add_argument("--neg-frac", type=float, default=0.03)
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--leaves", type=int, default=15)
    p.add_argument("--min-child", type=int, default=500)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--ks", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--random-draws", type=int, default=20)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def simulate(days, serials, y, score, k_per_week, starts):
    """Weekly replacement policy. Returns (distinct failing drives caught, drives replaced)."""
    replaced = np.empty(0, dtype=serials.dtype)
    caught, n_replaced = set(), 0
    for d0 in starts:
        idx = np.flatnonzero(days == d0)
        if len(replaced):
            idx = idx[~np.isin(serials[idx], replaced)]
        if len(idx) == 0:
            continue
        k = min(k_per_week, len(idx))
        top = idx[np.argpartition(-score[idx], k - 1)[:k]]
        n_replaced += len(top)
        caught.update(serials[top][y[top] == 1].tolist())
        replaced = np.concatenate([replaced, serials[top]])
    return len(caught), n_replaced


def main():
    args = parse_args()
    t0 = time.time()
    con = connect()
    feats = load_feature_list(args.features_file)
    unknown = [c for c in feats if c not in feature_columns(con, args.table)]
    if unknown:
        raise SystemExit(f"Features not found in {args.table}: {unknown}")
    print(f"{len(feats)} features from {args.features_file}; seeds {args.seeds}")

    tr = load_train_sample(con, args.table, feats, args.neg_frac)
    ytr = tr["label"].to_numpy()
    print(f"Train sample: {len(tr):,} rows, {int(ytr.sum()):,} positives  [{time.time() - t0:.0f}s]")

    os.makedirs("data/models", exist_ok=True)
    boosters = {}
    for sd in args.seeds:
        params = {"objective": "binary", "learning_rate": args.lr, "num_leaves": args.leaves,
                  "min_child_samples": args.min_child, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                  "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": args.threads, "seed": sd,
                  "deterministic": True, "force_row_wise": True, "verbose": -1}
        b = lgb.train(params, lgb.Dataset(tr[feats], ytr), num_boost_round=args.trees)
        b.save_model(f"data/models/final_s{sd}.txt")
        boosters[sd] = b
    print(f"Trained and saved {len(boosters)} models in data/models/final_s*.txt  [{time.time() - t0:.0f}s]")
    del tr

    rule_cols = [c for c in ("smart_5_raw", "smart_197_raw") if c in feats]
    days, serials, ys, rule = [], [], [], []
    scores = {sd: [] for sd in args.seeds}
    for a, b in chunk_ranges(*VAL):
        df = load_chunk(con, args.table, feats, a, b)
        if df.empty:
            continue
        X = df[feats]
        days.append(day_numbers(df))
        serials.append(serial_hashes(df))
        ys.append(df["label"].to_numpy().astype("int8"))
        r = sum((df[c].fillna(0).clip(lower=0) for c in rule_cols), start=0)
        rule.append(np.log1p(np.asarray(r, dtype="float64")).astype("float32"))
        for sd, bo in boosters.items():
            scores[sd].append(bo.predict(X, raw_score=True).astype("float32"))
        print(f"  scored {a}..{b}: {len(df):,} rows  [{time.time() - t0:.0f}s]")
    days, serials, y, rule = (np.concatenate(v) for v in (days, serials, ys, rule))
    scores = {sd: np.concatenate(v) for sd, v in scores.items()}
    print(f"Evaluation set: {len(y):,} rows, {int(y.sum()):,} positives ({y.mean():.4%})")

    # ---- exact metrics per seed, then mean and range
    results = {f"seed{sd}": evaluate_scores(days, serials, y, scores[sd], args.ks) for sd in args.seeds}
    print_report(results, args.ks)
    flat = {name: flatten_numbers(res) for name, res in results.items()}
    summary = {}
    print(f"\nMean and range over {len(args.seeds)} seeds (validation, exact):")
    for key in sorted(next(iter(flat.values()))):
        vals = np.array([f[key] for f in flat.values()])
        summary[key] = {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max())}
        print(f"  {key:<34}{vals.mean():>9.4f}   ({vals.min():.4f} to {vals.max():.4f})")

    # ---- business view
    ud = np.unique(days)
    blocks = (ud - ud[0]) // 7
    starts = [ud[blocks == b][0] for b in np.unique(blocks)]
    total = len(np.unique(serials[y == 1]))
    print(f"\nBusiness view: {len(starts)} weekly decisions; {total} distinct drives fail within the validation labels")
    print(f"{'K/week':<8}{'policy':<26}{'caught':>14}{'recall':>9}{'precision':>11}")
    rng = np.random.default_rng(0)
    business = {}
    for k in args.ks:
        model_runs = [simulate(days, serials, y, scores[sd], k, starts) for sd in args.seeds]
        caught = np.array([c for c, _ in model_runs])
        repl = np.mean([n for _, n in model_runs])
        rule_c, rule_n = simulate(days, serials, y, rule + rng.random(len(rule)).astype("float32") * 1e-3, k, starts)
        rand_runs = [simulate(days, serials, y, rng.random(len(y)).astype("float32"), k, starts)
                     for _ in range(args.random_draws)]
        rand_c = float(np.mean([c for c, _ in rand_runs]))
        rows = [(f"model (mean of {len(caught)} seeds)", caught.mean(), repl),
                ("rule: smart_5 + smart_197", rule_c, rule_n),
                (f"random (mean of {args.random_draws})", rand_c, np.mean([n for _, n in rand_runs]))]
        for name, c, n in rows:
            print(f"{k:<8}{name:<26}{c:>8.1f}/{total:<5}{c / total:>9.1%}{c / n:>11.1%}")
        print(f"{'':<8}{'  model range over seeds:':<26}{caught.min():>8.0f} to {caught.max():.0f}")
        business[k] = {"total_failing": total, "model_caught": caught.tolist(), "model_replaced_mean": float(repl),
                       "rule_caught": rule_c, "random_caught_mean": rand_c}

    os.makedirs("reports", exist_ok=True)
    out = "reports/final_validation.json"
    with open(out, "w") as f:
        json.dump({"meta": {"features_file": args.features_file, "n_features": len(feats), "trees": args.trees,
                            "leaves": args.leaves, "min_child": args.min_child, "neg_frac": args.neg_frac,
                            "seeds": args.seeds, "train": TRAIN, "val": VAL},
                   "per_seed": results, "summary": summary, "business_view": business}, f, indent=2)
    print(f"\nSaved {out}")

    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri("sqlite:////workspace/mlflow.db")
            mlflow.set_experiment("drivewatch-phase4")
            with mlflow.start_run(run_name="final_validation"):
                mlflow.log_params({"features_file": args.features_file, "n_features": len(feats), "trees": args.trees,
                                   "leaves": args.leaves, "min_child": args.min_child, "neg_frac": args.neg_frac,
                                   "seeds": ",".join(map(str, args.seeds))})
                for key, v in summary.items():
                    mlflow.log_metric(f"mean.{key}", v["mean"])
                    mlflow.log_metric(f"min.{key}", v["min"])
                    mlflow.log_metric(f"max.{key}", v["max"])
                mlflow.log_text("\n".join(feats), "features.txt")
                mlflow.log_artifact(out)
            print("Logged to MLflow.")
        except Exception as e:
            print(f"MLflow logging failed ({type(e).__name__}: {e}); results are saved in reports/")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
