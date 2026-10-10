"""CV experiment tracking, held-out evaluation and complete pipeline persistence."""
import argparse
import json
import os
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.common import (CATEGORICAL, FEATURES, NUMERIC, SEED, TARGET,
                        canonical_features, fingerprint, versions, write_json)


def build_pipeline(model):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    return Pipeline([("preprocess", prep), ("model", model)])


def metrics(y, predictions):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, predictions))),
            "MAE": float(mean_absolute_error(y, predictions)), "R2": float(r2_score(y, predictions))}


def train(split_dir="data/processed", models="models", reports="reports", quick=False):
    split_dir, models, reports = map(Path, [split_dir, models, reports])
    models.mkdir(parents=True, exist_ok=True)
    experiments = reports / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    split_info = json.loads((split_dir / "split_metadata.json").read_text())
    for name in ["train", "test"]:
        if fingerprint(split_dir / f"{name}.csv") != split_info[f"{name}_sha256"]:
            raise ValueError(f"{name} artifact hash mismatch")
    tr, te = [pd.read_csv(split_dir / f"{name}.csv") for name in ["train", "test"]]
    X, y = canonical_features(tr), tr[TARGET]
    X_test, y_test = canonical_features(te), te[TARGET]
    if len(X) < 10:
        raise ValueError("Insufficient training observations")
    cv = KFold(n_splits=3 if quick else 5, shuffle=True, random_state=SEED)
    baseline = build_pipeline(DummyRegressor(strategy="mean"))
    baseline_cv = -cross_val_score(baseline, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1).mean()
    candidates = {
        "random_forest": (RandomForestRegressor(random_state=SEED, n_jobs=1),
            {"model__n_estimators": [20] if quick else [150],
             "model__max_depth": [8] if quick else [None, 12],
             "model__min_samples_leaf": [2] if quick else [1, 3]}),
        "gradient_boosting": (GradientBoostingRegressor(random_state=SEED),
            {"model__n_estimators": [20] if quick else [100, 180],
             "model__learning_rate": [0.1] if quick else [0.05, 0.1],
             "model__max_depth": [2] if quick else [2, 3]}),
    }
    searches, rows = {}, []
    # Bounded parallelism avoids exhausting hosted runner or Colab memory.
    jobs = min(2, os.cpu_count() or 1)
    for name, (model, grid) in candidates.items():
        search = GridSearchCV(build_pipeline(model), grid, scoring="neg_root_mean_squared_error",
                              cv=cv, n_jobs=jobs, refit=True, return_train_score=True, error_score="raise")
        search.fit(X, y)
        results = pd.DataFrame(search.cv_results_)
        results.insert(0, "algorithm", name)
        results.to_csv(experiments / f"{name}_trials.csv", index=False)
        write_json(experiments / f"{name}_configuration.json", {
            "algorithm": name, "base_parameters": model.get_params(), "search_grid": grid,
            "best_parameters": search.best_params_, "cv_folds": cv.n_splits,
            "selection_metric": "CV RMSE", "seed": SEED,
        })
        rows.append({"algorithm": name, "cv_RMSE": float(-search.best_score_),
                     "best_parameters": search.best_params_})
        searches[name] = search
    comparison = pd.DataFrame(rows).sort_values("cv_RMSE")
    comparison.to_csv(experiments / "comparison.csv", index=False)
    winner = comparison.iloc[0]["algorithm"]
    pipeline = searches[winner].best_estimator_
    baseline.fit(X, y)
    pred = pipeline.predict(X_test)
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite holdout predictions")
    evaluation = {"selected_algorithm": winner, "baseline_cv_RMSE": float(baseline_cv),
                  "model": metrics(y_test, pred), "baseline": metrics(y_test, baseline.predict(X_test)),
                  "holdout_rows": len(te), "baseline_cv_beaten": bool(comparison.iloc[0].cv_RMSE < baseline_cv),
                  "quick_test_mode": bool(quick)}
    write_json(reports / "evaluation.json", evaluation)
    pd.DataFrame({"actual": y_test, "predicted": pred, "residual": y_test - pred}).to_csv(
        reports / "holdout_predictions.csv", index=False)
    joblib.dump(pipeline, models / "model.joblib", compress=3)
    # Category choices and sample come from training only; missing values become JSON null.
    sample = X.iloc[0].astype(object).where(X.iloc[0].notna(), None).to_dict()
    metadata = {
        "algorithm": winner, "best_parameters": searches[winner].best_params_,
        "features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL,
        "categories": {c: sorted(X[c].dropna().unique().tolist()) for c in CATEGORICAL},
        "sample_input": sample, "evaluation": evaluation, "split": split_info,
        "packages": versions(), "python": platform.python_version(),
        "source_commit": os.environ.get("GITHUB_SHA", "local"),
        "model_sha256": fingerprint(models / "model.joblib"),
        "prediction_grain": "Product/store row revenue; period and currency unspecified",
    }
    write_json(models / "metadata.json", metadata)
    prep = pipeline.named_steps["preprocess"]
    pd.DataFrame({"feature": prep.get_feature_names_out(),
                  "importance": pipeline.named_steps["model"].feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(reports / "feature_importance.csv", index=False)
    print(comparison.to_string(index=False))
    print(json.dumps(evaluation, indent=2))
    print("Baseline CV improvement:", evaluation["baseline_cv_beaten"])
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="data/processed")
    p.add_argument("--models", default="models")
    p.add_argument("--reports", default="reports")
    p.add_argument("--quick", action="store_true", help="Integration tests only; not academic final training")
    a = p.parse_args()
    train(a.splits, a.models, a.reports, a.quick)
