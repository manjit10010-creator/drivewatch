"""Phase 4: feature selection, following the plan's order. Validation only; the test period is not touched.

  1. Remove constant and near-empty columns.
  2. Remove highly correlated duplicates (rank correlation), keeping the simpler feature.
  3. Rank by mutual information with the label (informational, nothing is dropped here).
  4. Permutation importance on a validation sample, using a LightGBM trained on the survivors.
  5. SHAP check: do the top features push risk in a physically sensible direction?
  Then: retrain on the top-k features by permutation importance (2 seeds each) and keep the smallest k
  whose validation PR-AUC is within 1% (relative) of the full set.

Validation sample = all positives + a random 5% of healthy rows; PR-AUC is computed with sample weights
(healthy rows weigh 1/0.05), which approximates the exact full-validation PR-AUC. Use train_lgbm.py for exact numbers.

Run from the repo root, inside the drivewatch-ml Docker image:
    python src/models/select_features.py
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score

from dataset import TRAIN, VAL, chunk_ranges, connect, feature_columns, load_chunk, load_train_sample


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="features_daily")
    p.add_argument("--neg-frac", type=float, default=0.03, help="healthy share kept in the training sample")
    p.add_argument("--val-neg-frac", type=float, default=0.05, help="healthy share kept in the validation sample")
    p.add_argument("--drop", nargs="*", default=["days_in_data", "days_since_first_nonzero"],
                   help="exclude columns whose name contains any of these (left-censored features)")
    p.add_argument("--corr-threshold", type=float, default=0.95)
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3, help="permutation repeats per feature")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--tolerance", type=float, default=0.01, help="allowed relative PR-AUC loss")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def simplicity(name):
    """Lower = simpler. Used to decide which of two correlated features stays."""
    if name.startswith("delta"):
        return 2
    if name.startswith(("err_days", "temp_mean", "temp_max", "temp_std")):
        return 1
    return 0


def wap(y, s, w):
    return average_precision_score(y, s, sample_weight=w)


def fit(args, X, y, seed):
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 500,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "num_threads": args.threads, "seed": seed, "verbose": -1}
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=args.trees)


def main():
    args = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    con = connect()
    feats = [c for c in feature_columns(con, args.table) if not any(s in c for s in args.drop)]
    log = {"start_features": len(feats), "excluded_by_flag": args.drop}
    print(f"{len(feats)} candidate features (columns containing {args.drop} excluded)")

    tr = load_train_sample(con, args.table, feats, args.neg_frac)
    ytr = tr["label"].to_numpy()
    non_numeric = [c for c in feats if not pd.api.types.is_numeric_dtype(tr[c])]
    assert not non_numeric, f"non-numeric feature columns not supported here: {non_numeric}"
    print(f"Train sample: {len(tr):,} rows, {int(ytr.sum()):,} positives  [{time.time() - t0:.0f}s]")

    # ---- step 1: constant / near-empty
    drop1 = []
    for c in feats:
        s = tr[c]
        mode_share = s.value_counts(dropna=False, normalize=True).iloc[0]
        if s.nunique(dropna=True) <= 1 or mode_share >= 0.9995:
            drop1.append((c, round(float(mode_share), 5)))
    feats = [c for c in feats if c not in {d[0] for d in drop1}]
    print(f"\nStep 1 constant or near-empty (>=99.95% one value): dropped {len(drop1)}: {drop1}")
    log["step1_dropped"] = drop1

    # ---- step 2: correlated duplicates (Spearman via ranks, pairwise complete)
    sub = tr[feats].sample(min(300_000, len(tr)), random_state=args.seed)
    corr = sub.rank().corr().abs()
    pairs = [(corr.iloc[i, j], feats[i], feats[j]) for i in range(len(feats)) for j in range(i + 1, len(feats))
             if corr.iloc[i, j] >= args.corr_threshold]
    pairs.sort(reverse=True)
    dropped2, kept = [], set(feats)
    for r, a, b in pairs:
        if a in kept and b in kept:
            loser = max((a, b), key=lambda n: (simplicity(n), len(n), n))
            winner = b if loser == a else a
            kept.discard(loser)
            dropped2.append((loser, winner, round(float(r), 3)))
    feats = [c for c in feats if c in kept]
    print(f"\nStep 2 correlated pairs |rho| >= {args.corr_threshold}: dropped {len(dropped2)} (dropped, kept instead, rho)")
    for d in dropped2:
        print(f"  {d[0]:<40} <- {d[1]:<40} {d[2]}")
    log["step2_dropped"] = dropped2
    print(f"{len(feats)} features left  [{time.time() - t0:.0f}s]")

    # ---- step 3: mutual information (informational)
    mi_idx = rng.choice(len(tr), size=min(100_000, len(tr)), replace=False)
    Xmi = tr[feats].iloc[mi_idx].rank(pct=True).fillna(-1.0).to_numpy()
    mi = mutual_info_classif(Xmi, ytr[mi_idx], n_neighbors=3, random_state=args.seed, n_jobs=args.threads)
    mi_rank = sorted(zip(feats, mi), key=lambda t: -t[1])
    print(f"\nStep 3 mutual information with the label (100k-row sample; informational, nothing dropped here)  [{time.time() - t0:.0f}s]")
    for name, v in mi_rank[:12]:
        print(f"  {name:<46}{v:>8.4f}")
    print("  ...  lowest five:", [(n, round(float(v), 4)) for n, v in mi_rank[-5:]])
    log["step3_mutual_info"] = {n: float(v) for n, v in mi_rank}

    # ---- validation sample
    frames = []
    for a, b in chunk_ranges(*VAL):
        df = load_chunk(con, args.table, feats, a, b)
        if df.empty:
            continue
        keep = (df["label"].to_numpy() == 1) | (rng.random(len(df)) < args.val_neg_frac)
        frames.append(df.loc[keep, feats + ["label"]])
    va = pd.concat(frames, ignore_index=True)
    yva = va["label"].to_numpy()
    wva = np.where(yva == 1, 1.0, 1.0 / args.val_neg_frac)
    print(f"\nValidation sample: {len(va):,} rows, {int(yva.sum()):,} positives (weights restore the real base rate)  [{time.time() - t0:.0f}s]")

    # ---- step 4: permutation importance
    model = fit(args, tr[feats], ytr, args.seed)
    Xv = va[feats].to_numpy(dtype=np.float32)
    base = wap(yva, model.predict(Xv, raw_score=True), wva)
    print(f"\nStep 4 permutation importance (drop in weighted PR-AUC; baseline {base:.4f})  [{time.time() - t0:.0f}s]")
    perm = {}
    for j, c in enumerate(feats):
        orig = Xv[:, j].copy()
        drops = []
        for _ in range(args.repeats):
            Xv[:, j] = rng.permutation(orig)
            drops.append(base - wap(yva, model.predict(Xv, raw_score=True), wva))
        Xv[:, j] = orig
        perm[c] = float(np.mean(drops))
    perm_rank = sorted(perm.items(), key=lambda t: -t[1])
    for name, v in perm_rank[:20]:
        print(f"  {name:<46}{v:>9.4f}")
    zero = [n for n, v in perm_rank if v <= 0]
    print(f"  features whose permutation does not hurt (<= 0): {len(zero)}: {zero}")
    log["step4_permutation_importance"] = perm
    log["baseline_weighted_prauc"] = float(base)

    # ---- step 5: SHAP sanity check
    pos = np.flatnonzero(yva == 1)
    neg = np.flatnonzero(yva == 0)
    idx = np.concatenate([rng.choice(pos, min(5000, len(pos)), replace=False),
                          rng.choice(neg, min(15000, len(neg)), replace=False)])
    Xs = va[feats].iloc[idx]
    sv = shap.TreeExplainer(model).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)[:15]
    print(f"\nStep 5 SHAP (20k-row sample, enriched in positives; direction = rank correlation of value with SHAP, + means higher value raises risk)  [{time.time() - t0:.0f}s]")
    shap_out = {}
    for i in order:
        d = pd.Series(Xs.iloc[:, i].to_numpy()).corr(pd.Series(sv[:, i]), method="spearman")
        shap_out[feats[i]] = {"mean_abs_shap": float(mean_abs[i]), "direction": None if pd.isna(d) else float(d)}
        print(f"  {feats[i]:<46}{mean_abs[i]:>8.3f}   direction {d:+.2f}")
    log["step5_shap_top15"] = shap_out

    # ---- smallest set within tolerance
    ranked = [n for n, _ in perm_rank]
    ks = sorted({k for k in (8, 10, 12, 15, 20, 25, 30, 35) if k < len(ranked)} | {len(ranked)})
    seeds = [1, 2]
    res = {}
    print(f"\nTop-k feature sets by permutation importance, retrained (mean of {len(seeds)} seeds, weighted PR-AUC)  [{time.time() - t0:.0f}s]")
    for k in ks:
        cols = ranked[:k]
        sc = []
        for sd in seeds:
            m = fit(args, tr[cols], ytr, sd)
            sc.append(wap(yva, m.predict(va[cols].to_numpy(dtype=np.float32), raw_score=True), wva))
        res[k] = {"mean": float(np.mean(sc)), "scores": [float(s) for s in sc]}
        print(f"  k={k:<3} PR-AUC {np.mean(sc):.4f}  (seeds: {', '.join(f'{s:.4f}' for s in sc)})")
    full = res[ks[-1]]["mean"]
    chosen = next(k for k in ks if res[k]["mean"] >= (1 - args.tolerance) * full)
    selected = ranked[:chosen]
    print(f"\nSmallest k within {args.tolerance:.0%} of the full set ({full:.4f}): k = {chosen}")
    print("Seed-to-seed spread for the full set:", f"{abs(res[ks[-1]]['scores'][0] - res[ks[-1]]['scores'][1]):.4f}",
          "(if this is larger than the tolerance, treat the choice as approximate)")
    print("Selected:", selected)
    log.update({"topk_results": res, "chosen_k": chosen, "selected_features": selected})

    os.makedirs("reports", exist_ok=True)
    with open("reports/feature_selection.json", "w") as f:
        json.dump(log, f, indent=2)
    with open("reports/selected_features.txt", "w") as f:
        f.write("\n".join(selected) + "\n")
    print(f"\nSaved reports/feature_selection.json and reports/selected_features.txt  (done in {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
