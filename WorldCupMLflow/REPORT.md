# Project Report — Development and Evaluation of a Machine Learning Lifecycle Management System using MLflow

**Course:** AIN-3009 — MLOps
**Department:** Artificial Intelligence Engineering, Bahçeşehir University
**Student:** _[Your Name — Student Number]_
**Date:** _[date]_

---

## 1. Introduction

This project designs and implements a complete machine-learning lifecycle for a
**sports-analytics** problem: predicting the outcome of FIFA World Cup matches.
Given two national teams and the match context, the system predicts one of three
outcomes — a win for the first (home/listed) team, a draw, or a win for the
second (away/listed) team. The entire lifecycle — experiment tracking, model
development, hyperparameter tuning, model registry, deployment, and monitoring —
is managed with **MLflow**, an open-source MLOps platform.

The choice of domain is deliberately distinct from the common healthcare/churn
examples. Match-outcome prediction is a well-known, genuinely difficult
multi-class problem, which makes it a good vehicle for demonstrating the MLOps
tooling rather than chasing high accuracy.

## 2. Dataset

The data is the Kaggle *FIFA World Cup* dataset (`WorldCupMatches.csv`), covering
World Cup matches from **1930 to 2014**.

**Data quality issues found and fixed:**
- The raw file has ~4,500 rows but only **852 are real matches**; the remainder
  are empty rows, which are dropped.
- Several team names contain an HTML encoding artifact (e.g. the prefix on
  `Republic of Ireland`, `Bosnia and Herzegovina`), which would otherwise split
  one team into multiple identities. These are cleaned.
- Variants such as `Germany FR` and `IR Iran` are normalised to canonical names.

**Target variable** (`Outcome`): `0 = Home/listed win`, `1 = Draw`, `2 = Away/listed win`.
The class distribution is imbalanced (~57% / ~22% / ~20%), which we address with
balanced class weights.

**Features** (8 total): historical **win rate**, **average goals scored**, and
**average goals conceded** for each team, plus a **knockout-stage flag** and a
**normalised year**. Critically, the team-strength statistics are computed on the
**training split only** and then mapped onto the test split, so there is **no data
leakage** from test matches into the features.

> _Methodological note:_ Because most World Cup matches are at neutral venues,
> the home/away distinction is largely positional. The historical team-strength
> features are what give the models meaningful predictive signal.

## 3. Tools and Environment

| Tool | Purpose |
|------|---------|
| **MLflow** | Experiment tracking, model registry, deployment, UI |
| **scikit-learn** | Logistic Regression, Random Forest, preprocessing, metrics |
| **XGBoost** | Gradient-boosted trees model |
| **Hyperopt** | Bayesian (TPE) hyperparameter search |
| **SQLite** | Backend store for MLflow experiment metadata |
| **Docker / docker-compose** | Containerised MLflow tracking server |

MLflow is configured with a SQLite backend store (`sqlite:///mlflow.db`) and a
local artifact root. _[Insert screenshot of `mlflow ui` landing page.]_

## 4. Model Development and Experiment Tracking

Five models were developed and each was logged as a separate MLflow run with its
parameters, metrics, and serialized model artifact. For each model we log both a
single-holdout accuracy/macro-F1 **and** a 5-fold **cross-validated accuracy**.
Because the test set is small (171 matches), single-holdout accuracy is noisy, so
**cross-validated accuracy (`cv_accuracy`) is the metric used for comparison and
model selection.**

1. **Logistic Regression** — linear baseline with balanced class weights.
2. **Random Forest** — 300 trees, depth 6, balanced class weights.
3. **XGBoost** — 200 estimators, depth 4, learning rate 0.1.
4. **K-Nearest Neighbors** — k=51, distance-weighted (best single model here).
5. **Stacking ensemble** — RF + ExtraTrees + KNN, combined by a logistic
   meta-learner with 5-fold internal CV.

_[Insert screenshot of the MLflow experiments table sorted by `cv_accuracy`.]_

**Cross-validated results** _(fill in your exact numbers from the UI):_

| Model | CV Accuracy | Macro-F1 (holdout) |
|-------|-------------|--------------------|
| Logistic Regression | _0.xx_ | _0.xx_ |
| Random Forest | _0.xx_ | _0.xx_ |
| XGBoost | _0.xx_ | _0.xx_ |
| KNN | _0.xx_ | _0.xx_ |
| Stacking | _0.xx_ | _0.xx_ |

The strongest models (KNN, Stacking) reach ~0.63–0.64, clearly above the ~0.57
majority baseline. A note on draws: like most football models, all five rarely
predict draws — draws are the hardest class to separate, which depresses macro-F1
even when accuracy improves.

## 5. Hyperparameter Tuning

XGBoost was tuned with **Hyperopt** using the Tree-structured Parzen Estimator
(TPE) algorithm over 40 trials. The search space covered `max_depth`,
`n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`,
`min_child_weight`, and `reg_lambda`. Crucially, each trial is scored by **5-fold
cross-validated accuracy** rather than a single noisy holdout, so the search
optimizes real signal. Each trial is logged as a **nested MLflow run** under a
single parent run, so the full search is browsable and comparable in the UI.

_[Insert screenshot of the nested Hyperopt runs and the parallel-coordinates or
metrics plot.]_

**Best hyperparameters found:** _[paste the printed `best` dict]_
**Best tuned accuracy:** _0.xx_

## 6. Model Registry

The best run is registered in the **MLflow Model Registry** under the name
`WorldCupPredictor`. The registered version is then transitioned through the
lifecycle stages: **None → Staging → Production**, with older production versions
automatically archived.

_[Insert screenshot of the Models page showing the version in the Production stage.]_

## 7. Deployment

The production model is loaded directly from the registry
(`models:/WorldCupPredictor/Production`) and used to predict example fixtures
(e.g. Brazil vs France in a knockout match). The same cleaning and scaling
pipeline used in training is reused to featurise new fixtures.

For REST-style serving, MLflow's built-in server exposes the model as an HTTP
endpoint:

```bash
mlflow models serve -m "models:/WorldCupPredictor/Production" -p 1234 --no-conda
```

_[Insert screenshot/terminal output of a prediction.]_

## 8. Performance Monitoring

Data drift is monitored with a **Kolmogorov-Smirnov (KS) two-sample test** on the
distribution of goals scored, comparing the historical reference distribution to
(simulated) incoming match data. The KS statistic, p-value, and a binary
`drift_detected` flag are logged to a dedicated `WorldCup-Monitoring` experiment.
In production this job would run on a schedule against live data, triggering a
retraining alert when drift is detected (p < 0.05).

_[Insert screenshot of the monitoring run / drift metric.]_

## 9. Results and Insights

- Match-outcome prediction is **inherently hard**; with a strong majority-class
  baseline (~57%), the models reached roughly **0.55–0.60 accuracy**. The most
  informative features were the teams' historical win rates and goal averages.
- The value delivered here is the **end-to-end MLOps workflow**: every
  experiment is reproducible and comparable, models are versioned and promoted
  through governed stages, and drift is monitored — exactly the lifecycle
  management that matters for real-world deployments.

## 10. Conclusion and Lessons Learned

Building this pipeline gave hands-on experience with the full ML lifecycle in
MLflow. The biggest practical lesson was that **data quality dominates**: the raw
dataset's empty rows and corrupted team names would have silently degraded the
models, and catching them early was essential. MLflow's tracking made it trivial
to compare models objectively, and the registry plus staging/production stages
provide the governance needed to manage models safely over time.

_[Optional: future work — add more features (FIFA rankings, recent form),
calibrate probabilities, automate retraining on drift.]_
