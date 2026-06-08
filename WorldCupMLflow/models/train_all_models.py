"""
Train and track 5 models with MLflow:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost
  4. K-Nearest Neighbors (tuned)        <-- best single model on this data
  5. Stacking ensemble (RF + ET + KNN -> LR)

Each run logs parameters, BOTH a single-holdout accuracy/F1 AND a more reliable
5-fold cross-validated accuracy, and the model artifact.

Run from the project root:  python models/train_all_models.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils.data_loader import load_data

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("WorldCup-Match-Prediction")

X_train, X_test, y_train, y_test, scaler, team_table = load_data()
LABELS = ["HomeWin", "Draw", "AwayWin"]

# For cross-validated metrics we use the full data (CV does the splitting).
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def log_all_metrics(model, preds):
    """Log holdout accuracy/F1 and 5-fold cross-validated accuracy."""
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    cv_acc = cross_val_score(model, X_all, y_all, cv=CV, scoring="accuracy").mean()
    mlflow.log_metric("accuracy", acc)            # single-holdout
    mlflow.log_metric("f1_macro", f1)
    mlflow.log_metric("cv_accuracy", cv_acc)      # reliable estimate
    return acc, f1, cv_acc


def report(name, acc, f1, cv_acc, preds):
    print(f"OK {name:22s} holdout_acc={acc:.3f}  f1={f1:.3f}  CV_acc={cv_acc:.3f}")
    print(classification_report(y_test, preds, target_names=LABELS, zero_division=0))


# Model 1: Logistic Regression
with mlflow.start_run(run_name="LogisticRegression"):
    params = {"C": 1.0, "max_iter": 500, "solver": "lbfgs",
              "class_weight": "balanced"}
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mlflow.log_params(params)
    a, f, c = log_all_metrics(LogisticRegression(**params), preds)
    mlflow.sklearn.log_model(model, "logistic_model")
    report("LogisticRegression", a, f, c, preds)


# Model 2: Random Forest
with mlflow.start_run(run_name="RandomForest"):
    params = {"n_estimators": 300, "max_depth": 6, "random_state": 42,
              "class_weight": "balanced"}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mlflow.log_params(params)
    a, f, c = log_all_metrics(RandomForestClassifier(**params), preds)
    mlflow.sklearn.log_model(model, "rf_model")
    report("RandomForest", a, f, c, preds)


# Model 3: XGBoost
with mlflow.start_run(run_name="XGBoost"):
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
              "objective": "multi:softmax", "num_class": 3, "random_state": 42}
    model = XGBClassifier(**params, eval_metric="mlogloss")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mlflow.log_params(params)
    a, f, c = log_all_metrics(
        XGBClassifier(**params, eval_metric="mlogloss"), preds)
    mlflow.xgboost.log_model(model, "xgb_model")
    report("XGBoost", a, f, c, preds)


# Model 4: K-Nearest Neighbors (tuned) - best single model on this data
with mlflow.start_run(run_name="KNN"):
    params = {"n_neighbors": 51, "weights": "distance"}
    model = KNeighborsClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mlflow.log_params(params)
    a, f, c = log_all_metrics(KNeighborsClassifier(**params), preds)
    mlflow.sklearn.log_model(model, "knn_model")
    report("KNN", a, f, c, preds)


# Model 5: Stacking ensemble
with mlflow.start_run(run_name="StackingEnsemble"):
    base = [
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                      class_weight="balanced", random_state=42)),
        ("et", ExtraTreesClassifier(n_estimators=300, max_depth=8,
                                    class_weight="balanced", random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=31, weights="distance")),
    ]
    model = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=500),
        cv=5,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mlflow.log_param("base_estimators", "RF+ExtraTrees+KNN")
    mlflow.log_param("final_estimator", "LogisticRegression")
    a, f, c = log_all_metrics(
        StackingClassifier(estimators=base,
                           final_estimator=LogisticRegression(max_iter=500), cv=5),
        preds)
    mlflow.sklearn.log_model(model, "stacking_model")
    report("StackingEnsemble", a, f, c, preds)


print("\nAll 5 models logged. Compare them with:")
print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("Tip: sort the runs table by 'cv_accuracy' - it's the reliable metric.")
