"""Register the chosen final model in the MLflow model registry with the alias 'production'.

The model was chosen on validation only (seed 1, decided in advance); the test period has not been used.
Logs the model, its feature list and its validation metrics from reports/final_validation.json.

Run from the repo root, inside the drivewatch-ml Docker image:
    python src/models/register_model.py
"""
import argparse
import json

import lightgbm as lgb
import mlflow
from mlflow import MlflowClient

from train_lgbm import flatten_numbers, load_feature_list


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-file", default="data/models/final_s1.txt")
    p.add_argument("--features-file", default="reports/selected_features_25.txt")
    p.add_argument("--report", default="reports/final_validation.json")
    p.add_argument("--seed-key", default="seed1", help="which seed's validation metrics to attach")
    p.add_argument("--name", default="drivewatch-risk")
    return p.parse_args()


def main():
    args = parse_args()
    mlflow.set_tracking_uri("sqlite:////workspace/mlflow.db")
    mlflow.set_experiment("drivewatch-phase4")

    booster = lgb.Booster(model_file=args.model_file)
    feats = load_feature_list(args.features_file)
    if booster.feature_name() != feats:
        raise SystemExit("The model's feature names or order differ from the features file; refusing to register.\n"
                         f"model: {booster.feature_name()}\nfile:  {feats}")
    with open(args.report) as f:
        rep = json.load(f)
    metrics = flatten_numbers(rep["per_seed"][args.seed_key], "validation.")

    with mlflow.start_run(run_name="register_final"):
        mlflow.log_params({"model_file": args.model_file, "n_features": len(feats), "seed_key": args.seed_key,
                           "trees": rep["meta"]["trees"], "leaves": rep["meta"]["leaves"],
                           "min_child": rep["meta"]["min_child"], "neg_frac": rep["meta"]["neg_frac"],
                           "train": str(rep["meta"]["train"]), "val": str(rep["meta"]["val"])})
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_text("\n".join(feats), "features.txt")
        try:
            mlflow.lightgbm.log_model(booster, name="model", registered_model_name=args.name)
        except TypeError:  # older MLflow API
            mlflow.lightgbm.log_model(booster, artifact_path="model", registered_model_name=args.name)

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{args.name}'")
    latest = str(max(int(v.version) for v in versions))
    client.set_registered_model_alias(args.name, "production", latest)
    client.set_model_version_tag(args.name, latest, "note",
                                 "chosen on validation only; test period not used; dev sample (3 drive models)")
    print(f"Registered {args.name} version {latest} with alias 'production' ({len(feats)} features).")
    print(f"Validation PR-AUC of this seed: {metrics.get('validation.pr_auc', float('nan')):.4f}")


if __name__ == "__main__":
    main()
