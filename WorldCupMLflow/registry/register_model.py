"""
Register the best model in the MLflow Model Registry and transition it
through lifecycle stages: None -> Staging -> Production.

This demonstrates Solution Guide step 6 (Model Registry).
Run from the project root:  python registry/register_model.py
"""

import time
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

MODEL_NAME = "WorldCupPredictor"
EXPERIMENT = "WorldCup-Match-Prediction"

# ── 1. Find the best run by CROSS-VALIDATED accuracy ─────────
# cv_accuracy is more reliable than single-holdout accuracy on this small set.
experiment = client.get_experiment_by_name(EXPERIMENT)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.cv_accuracy > 0",
    order_by=["metrics.cv_accuracy DESC"],
    max_results=1,
)
best = runs[0]
run_id = best.info.run_id
run_name = best.data.tags.get("mlflow.runName", "unknown")
cv_acc = best.data.metrics["cv_accuracy"]
print(f"Best run: '{run_name}' ({run_id})  cv_accuracy={cv_acc:.3f}")

# ── 2. Pick the right artifact path for that run ─────────────
# Each training run logs its model under a different artifact name.
artifact_map = {
    "XGBoost": "xgb_model",
    "RandomForest": "rf_model",
    "LogisticRegression": "logistic_model",
    "KNN": "knn_model",
    "StackingEnsemble": "stacking_model",
    "Hyperopt-XGB": "model",  # nested hyperopt runs don't log a model by default
}
artifact = artifact_map.get(run_name, "model")
model_uri = f"runs:/{run_id}/{artifact}"
print(f"Registering: {model_uri}")

# ── 3. Register a new version ────────────────────────────────
result = mlflow.register_model(model_uri, MODEL_NAME)
version = result.version
print(f"Registered '{MODEL_NAME}' version {version}")

# Wait for registration to finish
for _ in range(10):
    mv = client.get_model_version(MODEL_NAME, version)
    if mv.status == "READY":
        break
    time.sleep(1)

# ── 4. Transition: Staging then Production ───────────────────
client.transition_model_version_stage(
    name=MODEL_NAME, version=version, stage="Staging"
)
print(f"➡️  Version {version} moved to STAGING")

client.transition_model_version_stage(
    name=MODEL_NAME, version=version, stage="Production",
    archive_existing_versions=True,
)
print(f"✅ Version {version} promoted to PRODUCTION")
