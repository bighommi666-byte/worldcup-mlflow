"""
Hyperparameter tuning for XGBoost using Hyperopt (TPE algorithm).

KEY IMPROVEMENT: the objective optimizes 5-FOLD CROSS-VALIDATED accuracy, not a
single 171-row holdout. With a small test set, single-holdout accuracy is noisy,
so tuning against it just chases noise. Cross-validation gives a reliable signal.

Each trial is logged as a NESTED MLflow run under one parent run.
Run from the project root:  python tuning/hyperopt_tuning.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils.data_loader import load_data

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("WorldCup-Match-Prediction")

X_train, X_test, y_train, y_test, scaler, team_table = load_data()
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def objective(params):
    params["max_depth"] = int(params["max_depth"])
    params["n_estimators"] = int(params["n_estimators"])

    with mlflow.start_run(run_name="Hyperopt-XGB", nested=True):
        model = XGBClassifier(
            **params,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
        # 5-fold cross-validated accuracy = the reliable objective
        cv_acc = cross_val_score(model, X_all, y_all, cv=CV,
                                 scoring="accuracy").mean()
        mlflow.log_params(params)
        mlflow.log_metric("cv_accuracy", cv_acc)

    return {"loss": -cv_acc, "status": STATUS_OK}


search_space = {
    "max_depth":        hp.quniform("max_depth", 2, 8, 1),
    "n_estimators":     hp.quniform("n_estimators", 50, 400, 50),
    "learning_rate":    hp.uniform("learning_rate", 0.01, 0.3),
    "subsample":        hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
    "reg_lambda":       hp.uniform("reg_lambda", 0.0, 3.0),
}

with mlflow.start_run(run_name="Hyperopt-Parent"):
    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=40,          # more trials than before
        trials=trials,
    )
    best_acc = -min(t["result"]["loss"] for t in trials.trials)
    mlflow.log_params({f"best_{k}": v for k, v in best.items()})
    mlflow.log_metric("best_cv_accuracy", best_acc)
    print("Best params:", best)
    print(f"Best CV accuracy: {best_acc:.3f}")
