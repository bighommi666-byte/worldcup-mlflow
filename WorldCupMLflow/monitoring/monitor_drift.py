"""
Performance / data-drift monitoring (Solution Guide step 5).

We simulate incoming match data and use a Kolmogorov-Smirnov test to detect
distribution drift in a key feature (goals scored), then log the result to
MLflow. In production you would run this on a schedule against live data.

Run from the project root:  python monitoring/monitor_drift.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlflow
from scipy.stats import ks_2samp

from utils.data_loader import load_raw

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("WorldCup-Monitoring")

# Reference distribution = historical home goals from the real dataset
df = load_raw()
reference = df["Home Team Goals"].values.astype(float)

# Simulate "incoming" data with a slight shift (modern football = fewer goals)
rng = np.random.default_rng(42)
incoming = rng.normal(loc=reference.mean() - 0.4, scale=1.0, size=60)
incoming = np.clip(incoming, 0, None)

stat, p_value = ks_2samp(reference, incoming)
drift = p_value < 0.05

with mlflow.start_run(run_name="Goals-Distribution-Drift"):
    mlflow.log_metric("reference_mean", float(reference.mean()))
    mlflow.log_metric("incoming_mean", float(incoming.mean()))
    mlflow.log_metric("ks_statistic", float(stat))
    mlflow.log_metric("p_value", float(p_value))
    mlflow.log_metric("drift_detected", int(drift))

status = "⚠️  DRIFT DETECTED — consider retraining" if drift else "✅ No significant drift"
print(f"KS statistic = {stat:.4f}  |  p-value = {p_value:.4f}")
print(f"Reference mean goals = {reference.mean():.2f}  |  Incoming = {incoming.mean():.2f}")
print(status)
