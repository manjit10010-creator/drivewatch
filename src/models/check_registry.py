"""Read-only check of the MLflow model registry. Registers and changes nothing."""
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

NAME = "drivewatch-risk"

if not os.path.exists("mlflow.db"):
    print("mlflow.db not found in the working directory; run from the repo root.")
    sys.exit(1)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

models = client.search_registered_models()
print("Registered models:", [m.name for m in models] or "none")

versions = client.search_model_versions(f"name='{NAME}'")
if not versions:
    print(f"No versions of '{NAME}' found. Registration did not happen.")
    sys.exit(0)

for v in sorted(versions, key=lambda x: int(x.version)):
    print(
        f"version {v.version}  aliases={getattr(v, 'aliases', [])}  "
        f"run_id={v.run_id}  status={v.status}"
    )

try:
    prod = client.get_model_version_by_alias(NAME, "production")
    print(f"Alias 'production' -> version {prod.version}")
except Exception as exc:
    print("Alias 'production' not set:", exc)
