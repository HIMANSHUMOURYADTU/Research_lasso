"""Bagging and boosting on the same Communities and Crime splits.

Trees are median-imputed inside the training fold. They are not standardized.
Settings are the library defaults, except the forest and the bagged trees use
300 trees and a minimum leaf size of 2 so a single community cannot own a leaf.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import crime_regression as cr

SEEDS = (0, 1, 7, 21, 42)
OUT = Path(__file__).resolve().parent / "ensemble_results.json"


def make_models(seed: int) -> dict:
    tree = DecisionTreeRegressor(min_samples_leaf=2, random_state=seed)
    return {
        "Bagging": BaggingRegressor(
            estimator=tree,
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=seed),
    }


def one_split(X, y, seed: int) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cr.TEST_SIZE, random_state=seed
    )
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    row = {"seed": seed}
    for name, model in make_models(seed).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        row[name] = {
            "r2": round(float(r2_score(y_test, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        }
        print(
            f"seed {seed:3}  {name:22} "
            f"R2 {row[name]['r2']:.4f}  RMSE {row[name]['rmse']:.4f}",
            flush=True,
        )
    return row


def main() -> None:
    df = cr.load_frame()
    predictive = [c for c in df.columns if c not in cr.NON_PREDICTIVE and c != cr.TARGET]
    high_missing = [
        c for c in predictive if float(df[c].isna().mean()) > cr.MISSING_DROP_THRESHOLD
    ]
    features = [c for c in predictive if c not in high_missing]
    X = df[features]
    y = df[cr.TARGET].to_numpy()
    rows = [one_split(X, y, seed) for seed in SEEDS]
    names = ("Bagging", "RandomForest", "GradientBoosting", "HistGradientBoosting")
    summary = {}
    for name in names:
        r2 = np.array([row[name]["r2"] for row in rows])
        rmse = np.array([row[name]["rmse"] for row in rows])
        summary[name] = {
            "mean_r2": round(float(r2.mean()), 4),
            "std_r2": round(float(r2.std(ddof=1)), 4),
            "mean_rmse": round(float(rmse.mean()), 4),
        }
    payload = {"seeds": list(SEEDS), "splits": rows, "summary": summary}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
