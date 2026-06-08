"""
One-time MLflow configuration check.
Run from the project root:  python mlflow_setup/setup_mlflow.py
"""

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("WorldCup-Match-Prediction")

print("✅ MLflow configured")
print("   Tracking URI:", mlflow.get_tracking_uri())
print("   Experiment  : WorldCup-Match-Prediction")
print("\nStart the UI with:")
print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
