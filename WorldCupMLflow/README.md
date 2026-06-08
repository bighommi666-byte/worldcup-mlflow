# FIFA World Cup Match Outcome Prediction — MLOps with MLflow

**Course:** AIN-3009 · Artificial Intelligence Engineering · Bahçeşehir University
**Domain:** Sports Analytics
**Task:** Multi-class classification — predict the result of a World Cup match
(Home/listed-first win · Draw · Away/listed-second win)

This project implements the full machine-learning lifecycle with **MLflow**:
experiment tracking, model development, hyperparameter tuning, a model
registry with staging→production transitions, deployment, and drift monitoring.

---

## Dataset

`data/WorldCupMatches.csv` — historical World Cup matches **1930–2022**
(from the Kaggle *FIFA World Cup* dataset). After cleaning and de-duplication there are
**964 real matches** (the raw file contains ~3,700 empty rows that are dropped).

**Cleaning performed** (see `utils/data_loader.py`):
- Drops empty rows with no team / goal data.
- Fixes encoding artifacts in team names (e.g. `rn">Republic of Ireland`).
- Normalises variants (`Germany FR`→`Germany`, `IR Iran`→`Iran`, etc.).

**Features** (leakage-safe — team strength computed on the training split only):
`home_win_rate, home_avg_gf, home_avg_ga, away_win_rate, away_avg_gf,
away_avg_ga, Is_Knockout, Year_Norm`

> Note: most World Cup matches are at neutral venues, so "home"/"away" is
> largely the order teams are listed in. The historical team-strength features
> give the models real signal beyond that positional bias.

---

## Project structure

```
WorldCupMLflow/
├── data/WorldCupMatches.csv
├── utils/data_loader.py          # cleaning + feature engineering
├── mlflow_setup/setup_mlflow.py  # config check
├── models/train_all_models.py    # 5 models, logged to MLflow
├── tuning/hyperopt_tuning.py     # Hyperopt + nested runs
├── registry/register_model.py    # register + staging→production
├── deployment/serve_model.py     # load production model, predict
├── monitoring/monitor_drift.py   # KS drift test, logged to MLflow
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick start (local)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run the pipeline IN ORDER (from the project root)
python models/train_all_models.py     # train + log 3 models
python tuning/hyperopt_tuning.py       # hyperparameter search
python registry/register_model.py      # register best → Production
python deployment/serve_model.py       # predict example fixtures
python monitoring/monitor_drift.py     # drift check

# 3. Explore everything in the UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
#   → open http://localhost:5000
```

> ⚠️ Always run scripts from the **project root** (so `data/` and the
> `sqlite:///mlflow.db` path resolve correctly), not from inside subfolders.

---

## Running the MLflow server with Docker

```bash
docker compose up --build
#   → open http://localhost:5000
```

The SQLite DB and artifacts are mounted as volumes, so your runs persist
on the host between container restarts. Train the models on the host first
(so `mlflow.db` exists), then bring up the container to browse them — or run
the training scripts pointing at the same DB.

---

## Mapping to the assignment objectives

| # | Objective | Where |
|---|-----------|-------|
| 1 | Experiment tracking | `models/train_all_models.py` logs params/metrics/artifacts |
| 2 | Model training & tuning | 3 models + `tuning/hyperopt_tuning.py` |
| 3 | Model deployment | `deployment/serve_model.py` + `mlflow models serve` |
| 4 | Performance monitoring | `monitoring/monitor_drift.py` (KS drift test) |
| 5 | Model Registry | `registry/register_model.py` (staging→production) |

---

## Results (cross-validated accuracy)

Match prediction is genuinely hard; with a strong majority-class baseline
(~57% home wins), the models reach the following **5-fold cross-validated**
accuracy (the reliable metric — single-holdout numbers on the 171-row test set
are noisy, so sort the MLflow runs table by `cv_accuracy`):

| Model | CV Accuracy |
|-------|-------------|
| Majority baseline | ~0.573 |
| Logistic Regression | ~0.569 |
| Random Forest | ~0.595 |
| XGBoost | ~0.58 |
| **KNN (k=51, distance)** | **~0.637** |
| **Stacking (RF+ET+KNN→LR)** | **~0.633** |

The point of the project is demonstrating the **MLflow lifecycle**, not maximising
accuracy. The KNN / Stacking models are the strongest here because the
team-strength features create a space where similar matchups have similar
outcomes. Fill in your own exact numbers from the MLflow UI after running.
