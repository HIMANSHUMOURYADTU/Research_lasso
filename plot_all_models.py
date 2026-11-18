"""Figures for every fitted technique: linear penalties, bagging, and boosting.

Seed 42 matches crime_regression.py and ensemble_comparison.py.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import crime_regression as cr
import ensemble_comparison as ec

FIG = cr.ROOT / "figures"
BLUE = "#1d4e89"
ORANGE = "#d4760a"
GREEN = "#2f7d4a"
PURPLE = "#5c3d86"
TEAL = "#1f6f6a"
BROWN = "#8a5a12"
RED = "#8a3b3b"
NAVY = "#243e73"

# Seed-42 test R^2 and RMSE, linear models from the primary fit.
LINEAR = [
    ("Least squares", 0.6379, 0.1317),
    ("Ridge", 0.6399, 0.1313),
    ("Lasso", 0.6421, 0.1309),
    ("Elastic net", 0.6418, 0.1310),
]


def save(fig, name: str) -> None:
    path = FIG / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def load_xy():
    df = cr.load_frame()
    predictive = [c for c in df.columns if c not in cr.NON_PREDICTIVE and c != cr.TARGET]
    high = [c for c in predictive if float(df[c].isna().mean()) > cr.MISSING_DROP_THRESHOLD]
    features = [c for c in predictive if c not in high]
    X = df[features]
    y = df[cr.TARGET].to_numpy()
    return train_test_split(X, y, test_size=cr.TEST_SIZE, random_state=cr.RANDOM_STATE)


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    ens = json.loads((cr.ROOT / "ensemble_results.json").read_text(encoding="utf-8"))
    comp = json.loads((FIG / "comparison.json").read_text(encoding="utf-8"))
    seed42 = next(row for row in ens["splits"] if row["seed"] == 42)
    trees = [
        ("Bagging", seed42["Bagging"]["r2"], seed42["Bagging"]["rmse"]),
        ("Random forest", seed42["RandomForest"]["r2"], seed42["RandomForest"]["rmse"]),
        ("Gradient boosting", seed42["GradientBoosting"]["r2"], seed42["GradientBoosting"]["rmse"]),
        ("Hist. boosting", seed42["HistGradientBoosting"]["r2"], seed42["HistGradientBoosting"]["rmse"]),
    ]
    names = [n for n, _, _ in LINEAR + trees]
    r2s = [v for _, v, _ in LINEAR + trees]
    rmses = [v for _, _, v in LINEAR + trees]
    colors = [BLUE, ORANGE, GREEN, PURPLE, TEAL, BROWN, RED, NAVY]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), constrained_layout=True)
    y = np.arange(len(names))
    axes[0].barh(y, r2s, color=colors)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Test R²")
    axes[0].set_title("Holdout R², seed 42")
    for i, v in enumerate(r2s):
        axes[0].text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=8)
    axes[1].barh(y, rmses, color=colors)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names)
    axes[1].set_xlabel("Test RMSE (normalized crime rate)")
    axes[1].set_title("Holdout RMSE, seed 42")
    for i, v in enumerate(rmses):
        axes[1].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    axes[1].set_xlim(0, 0.18)
    fig.suptitle("Every fitted model on the same 399 test communities")
    save(fig, "19_all_models_holdout.png")

    X_train, X_test, y_train, y_test = load_xy()
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)
    order = ["Bagging", "RandomForest", "GradientBoosting", "HistGradientBoosting"]
    labels = {
        "Bagging": "Bagging",
        "RandomForest": "Random forest",
        "GradientBoosting": "Gradient boosting",
        "HistGradientBoosting": "Hist. boosting",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, key in zip(axes.ravel(), order):
        model = ec.make_models(42)[key]
        model.fit(X_train_i, y_train)
        pred = model.predict(X_test_i)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        ax.scatter(y_test, pred, s=12, alpha=0.4, color=BLUE, linewidths=0)
        ax.plot([0, 1], [0, 1], color=RED, linewidth=1.2)
        ax.set_title(f"{labels[key]}   RMSE {rmse:.3f}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 1.15)
    for ax in axes[1]:
        ax.set_xlabel("Actual crime rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted crime rate")
    fig.suptitle("Tree models on the seed-42 test communities")
    save(fig, "20_tree_predicted_vs_actual.png")

    linear_means = [
        comp["repeated_splits"]["summary"][k]["mean_r2"]
        for k in ("Linear", "Ridge", "Lasso", "ElasticNet")
    ]
    tree_means = [
        ens["summary"][k]["mean_r2"]
        for k in ("Bagging", "RandomForest", "GradientBoosting", "HistGradientBoosting")
    ]
    mean_names = [
        "Least squares",
        "Ridge",
        "Lasso",
        "Elastic net",
        "Bagging",
        "Random forest",
        "Gradient boosting",
        "Hist. boosting",
    ]
    means = linear_means + tree_means
    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    xpos = np.arange(len(mean_names))
    ax.bar(xpos, means, color=colors)
    ax.set_xticks(xpos)
    ax.set_xticklabels(mean_names, rotation=28, ha="right")
    ax.set_ylim(0.60, 0.68)
    ax.set_ylabel("Mean test R² across five splits")
    ax.set_title("Five-split average. The axis starts at 0.60 so the gap is visible.")
    for i, v in enumerate(means):
        ax.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontsize=8)
    save(fig, "21_five_split_means.png")


if __name__ == "__main__":
    main()
