"""
Load the PRODUCTION model from the registry and make a prediction for a
real fixture. This demonstrates Solution Guide step 5 (deployment / serving).

Run from the project root:  python deployment/serve_model.py

For real REST serving you can also run:
    mlflow models serve -m "models:/WorldCupPredictor/Production" -p 1234 --no-conda
then POST JSON to http://localhost:1234/invocations
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlflow

from utils.data_loader import load_data, FEATURES

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load production model
model = mlflow.pyfunc.load_model("models:/WorldCupPredictor/Production")

# Rebuild scaler + team stats so we can featurise a new fixture the same way
_, _, _, _, scaler, team_table = load_data()
LABELS = {0: "Home/listed-first team WINS 🏆",
          1: "DRAW 🤝",
          2: "Away/listed-second team WINS 🏆"}


def build_features(home, away, is_knockout, year):
    g = team_table["__GLOBAL__"]
    h = team_table.get(home, g)
    a = team_table.get(away, g)
    yr_norm = (year - 1930) / (2022 - 1930)  # same normalisation as training
    row = [
        h["win_rate"], h["avg_gf"], h["avg_ga"],
        a["win_rate"], a["avg_gf"], a["avg_ga"],
        int(is_knockout), yr_norm,
    ]
    return scaler.transform([row])


def predict(home, away, is_knockout=1, year=2026):
    X = build_features(home, away, is_knockout, year)
    pred = int(model.predict(X)[0])
    print(f"\n{home}  vs  {away}  (knockout={bool(is_knockout)}, {year})")
    print(f"  → Prediction: {LABELS[pred]}")
    return pred


if __name__ == "__main__":
    # Example fixtures (teams must match cleaned names in the dataset)
    predict("Brazil", "France", is_knockout=1)
    predict("Germany", "Argentina", is_knockout=1)
    predict("Spain", "Netherlands", is_knockout=0)
    predict("Brazil", "Qatar", is_knockout=0)
    predict("Germany", "Saudi Arabia", is_knockout=0)