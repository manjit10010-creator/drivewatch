"""Check that live scoring (what POST /predict does) gives the same answer as the stored fleet_scores.

Takes one drive on one day, reads its 25 features from the gold table features_scoring, scores them with the
production model through the same function the API uses, and compares with the row in fleet_scores (made earlier
by score_fleet.py). It also checks that the per-feature contributions add up to the score, and writes the
features as a ready-to-send request body, data/sample_predict.json, for testing POST /predict by hand.

Run from the repo root with the API stopped (a read-only look at the warehouse):
    docker run --rm -e GIT_PYTHON_REFRESH=quiet -v "${PWD}:/workspace" drivewatch-api python -m src.api.check_predict
"""
import argparse
import json
import math
from datetime import date

import duckdb
import numpy as np

from src.api import predict

DB_PATH = "data/gold/drivewatch.duckdb"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serial", default="ZHZ42XAP", help="the drive to check (default: the riskiest drive on 2026-06-30)")
    p.add_argument("--date", default="2026-06-30")
    p.add_argument("--out", default="data/sample_predict.json")
    args = p.parse_args()
    day = date.fromisoformat(args.date)

    model = predict.load_model()
    print(f"Loaded {predict.MODEL_URI} = version {model.version}, {len(model.features)} features")

    con = duckdb.connect(DB_PATH, read_only=True)
    row = con.execute(f"SELECT {', '.join(model.features)} FROM features_scoring WHERE serial_number = ? AND date = ?",
                      [args.serial, day]).fetchone()
    stored = con.execute("SELECT risk_score, risk_band, rank_in_day FROM fleet_scores WHERE serial_number = ? AND date = ?",
                         [args.serial, day]).fetchone()
    con.close()
    if row is None or stored is None:
        raise SystemExit(f"No row for {args.serial} on {day} in features_scoring / fleet_scores.")

    # NULL (and NaN) become None, which is also what an API caller sends for "not reported"
    values = {name: (None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v))
              for name, v in zip(model.features, row)}

    score, band, top = predict.score_one(model, values)
    diff = abs(float(np.float32(score)) - float(stored[0]))
    print(f"\n{args.serial} on {day}")
    print(f"  live score   {score:.6f}  band {band}")
    print(f"  stored score {stored[0]:.6f}  band {stored[1]}  (rank {stored[2]} that day)")
    ok_score = diff < 1e-6
    ok_band = band == stored[1]
    print(f"  difference {diff:.2e}  -> {'PASS' if ok_score and ok_band else 'FAIL'} (score and band must match)")

    # The contributions (log-odds) plus the baseline must add up to the raw score; the sigmoid gives the probability.
    x = np.array([[np.nan if values[n] is None else values[n] for n in model.features]], dtype="float64")
    contrib = np.asarray(model.booster.predict(x, pred_contrib=True, num_threads=1))[0]
    from_contrib = 1.0 / (1.0 + math.exp(-float(contrib.sum())))
    ok_sum = abs(from_contrib - score) < 1e-9
    print(f"  sigmoid(sum of contributions) = {from_contrib:.6f}  -> {'PASS' if ok_sum else 'FAIL'}")

    print("\nTop contributing features (positive = pushes the risk up):")
    for c in top:
        print(f"  {c.feature:<24} value {c.value!s:>14}   contribution {c.contribution:+.4f}")

    body = {"serial_number": args.serial, **values}
    with open(args.out, "w") as f:
        json.dump(body, f, indent=2)
    print(f"\nWrote {args.out}: a request body for POST /predict")
    if not (ok_score and ok_band and ok_sum):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
