"""Evaluation metrics for DriveWatch (primary: PR-AUC, precision at top K per week, recall at a fixed
false-alarm rate; secondary: ROC-AUC). Plain accuracy is deliberately not reported.

Inputs are plain numpy arrays: y (0/1), s (score, higher = riskier), days (days since 1970-01-01),
serials (any per-drive id, e.g. a hash).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y, s, targets=(0.01, 0.001)):
    """Share of failing rows caught when at most `target` of healthy rows may raise a false alarm."""
    fpr, tpr, _ = roc_curve(y, s)
    return {t: float(tpr[np.searchsorted(fpr, t, side="right") - 1]) for t in targets}


def weekly_precision_at_k(days, serials, y, s, ks=(20, 50, 100)):
    """Per calendar week: give each drive its highest score that week, take the K riskiest drives,
    and measure how many of them have label = 1 on some day of that week. Averaged over weeks.
    This mirrors the ops persona: 'we can replace K drives a week'."""
    wk = days.astype("int64") // 7
    sc = pd.factorize(serials)[0]
    df = pd.DataFrame({"wk": wk, "sc": sc, "y": y, "s": s})
    g = df.groupby(["wk", "sc"], sort=False).agg(s=("s", "max"), y=("y", "max")).reset_index()
    g = g.sort_values(["wk", "s"], ascending=[True, False], kind="stable")
    g["rank"] = g.groupby("wk").cumcount()
    out = {"base_rate_drive_weeks": float(g["y"].mean()), "weeks": int(g["wk"].nunique())}
    for k in ks:
        sel = g[g["rank"] < k]
        out[f"top{k}"] = {
            "mean_precision": float(sel.groupby("wk")["y"].mean().mean()),
            "true_picks": int(sel["y"].sum()),
            "picks": int(len(sel)),
        }
    return out


def evaluate_scores(days, serials, y, s, ks=(20, 50, 100)):
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "pr_auc": float(average_precision_score(y, s)),
        "pr_auc_random_baseline": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, s)),
        "recall_at_fpr": recall_at_fpr(y, s),
        "weekly": weekly_precision_at_k(days, serials, y, s, ks),
    }


def binary_rule_summary(y, flag):
    flag = flag.astype(bool)
    tp = int((flag & (y == 1)).sum())
    return {
        "flagged_share": float(flag.mean()),
        "precision": tp / max(int(flag.sum()), 1),
        "recall": tp / max(int(y.sum()), 1),
    }


def print_report(results, ks=(20, 50, 100)):
    print(f"\n{'model':<26}{'PR-AUC':>9}{'ROC-AUC':>9}{'rec@1%FPR':>11}{'rec@0.1%':>10}" + "".join(f"{'P@' + str(k):>8}" for k in ks))
    for name, r in results.items():
        rf = r["recall_at_fpr"]
        line = f"{name:<26}{r['pr_auc']:>9.4f}{r['roc_auc']:>9.4f}{rf[0.01]:>11.3f}{rf[0.001]:>10.3f}"
        line += "".join(f"{r['weekly'][f'top{k}']['mean_precision']:>8.3f}" for k in ks)
        print(line)
    any_r = next(iter(results.values()))
    print(f"\nRandom-guess PR-AUC (positive rate): {any_r['pr_auc_random_baseline']:.4%}; "
          f"weekly drive-week base rate: {any_r['weekly']['base_rate_drive_weeks']:.2%}")
