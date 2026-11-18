"""Save EDA, diagnostic, and gradient-descent figures for Communities and Crime.

Gradient descent minimizes training MSE on the same standardized 100-feature
design used by the linear model. The step size is set from the largest
eigenvalue of the Hessian so one run is slow, one is near the stable limit,
and one is too large and diverges.

Run from this folder:
  python plot_analysis.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import crime_regression as cr

FIG = cr.ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE = "#1d4e89"
ORANGE = "#d4760a"
GREEN = "#2f7d4a"
RED = "#b3403a"
GRAY = "#5c6770"
PURPLE = "#5c4d7a"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#d0d4d8",
        "axes.labelcolor": "#1c1c1c",
        "xtick.color": "#1c1c1c",
        "ytick.color": "#1c1c1c",
        "text.color": "#1c1c1c",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "medium",
        "axes.grid": True,
        "grid.color": "#e6e8eb",
        "grid.linewidth": 0.6,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def batch_gd(X: np.ndarray, y: np.ndarray, lr: float, max_iter: int = 4000):
    """Minimize mean squared error. Returns weights, intercept, and history.

    History arrays include the starting point (iteration 0, before any step).
    Stops early if the loss diverges.
    """
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0
    mse = np.empty(max_iter + 1)
    weights = np.empty((max_iter + 1, p))
    intercepts = np.empty(max_iter + 1)
    weights[0] = w
    intercepts[0] = b
    pred = np.full(n, b)
    err = pred - y
    mse[0] = float(np.mean(err**2))
    last = 0
    for t in range(1, max_iter + 1):
        grad_w = (2.0 / n) * (X.T @ err)
        grad_b = (2.0 / n) * float(np.sum(err))
        w = w - lr * grad_w
        b = b - lr * grad_b
        pred = X @ w + b
        err = pred - y
        value = float(np.mean(err**2))
        weights[t] = w
        intercepts[t] = b
        mse[t] = value
        last = t
        if not np.isfinite(value) or value > 1e3:
            break
    return {
        "w": w,
        "b": b,
        "mse": mse[: last + 1],
        "weights": weights[: last + 1],
        "intercepts": intercepts[: last + 1],
        "diverged": bool(mse[last] > 1.0 or not np.isfinite(mse[last])),
    }


def downsample(values: np.ndarray, k: int = 28) -> tuple[np.ndarray, np.ndarray]:
    idx = np.unique(np.round(np.linspace(0, len(values) - 1, k)).astype(int))
    return idx, values[idx]


def main() -> None:
    df = cr.load_frame()
    y_all = df[cr.TARGET].astype(float)
    predictive = [c for c in df.columns if c not in cr.NON_PREDICTIVE and c != cr.TARGET]
    high_missing = [c for c in predictive if float(df[c].isna().mean()) > cr.MISSING_DROP_THRESHOLD]
    features = [c for c in predictive if c not in high_missing]

    model_df = df[features + [cr.TARGET]].apply(pd.to_numeric, errors="coerce")
    eda = model_df.fillna(model_df.median(numeric_only=True))
    corr = eda.corr(numeric_only=True)[cr.TARGET].drop(cr.TARGET)
    order = corr.abs().sort_values(ascending=False)

    # --- target distribution ---
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(y_all, bins=20, range=(0, 1), color=BLUE, edgecolor="white")
    ax.axvline(y_all.mean(), color=ORANGE, linewidth=1.6, label=f"Mean {y_all.mean():.2f}")
    ax.axvline(y_all.median(), color=GREEN, linewidth=1.6, label=f"Median {y_all.median():.2f}")
    ax.set_xlabel("ViolentCrimesPerPop (normalized 0–1)")
    ax.set_ylabel("Communities")
    ax.set_title("Most communities have a low normalized violent-crime rate")
    ax.legend(frameon=False)
    save(fig, "01_target_distribution.png")

    # --- missingness ---
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [
        "22 LEMAS police fields\n(each)",
        "community id",
        "county id",
        "OtherPerCap",
    ]
    missing_pct = [84.0, 100 * df["community"].isna().mean(), 100 * df["county"].isna().mean(), 100 * df["OtherPerCap"].isna().mean()]
    ax.barh(labels[::-1], missing_pct[::-1], color=[BLUE, GRAY, GRAY, ORANGE][::-1])
    ax.set_xlabel("Percent of the 1,994 communities missing")
    ax.set_xlim(0, 100)
    ax.set_title("Police-survey fields are missing for most communities")
    save(fig, "02_missing_values.png")

    # --- correlation with target ---
    top = order.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    vals = [float(corr[name]) for name in top.index[::-1]]
    colors = [RED if v > 0 else BLUE for v in vals]
    ax.barh(list(top.index[::-1]), vals, color=colors)
    ax.axvline(0, color="#1c1c1c", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with ViolentCrimesPerPop")
    ax.set_title("Family structure has the strongest linear association")
    save(fig, "03_correlation_with_target.png")

    # --- heatmap ---
    heat_cols = list(order.head(12).index) + [cr.TARGET]
    heat = eda[heat_cols].corr()
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    image = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(heat_cols)))
    ax.set_yticks(range(len(heat_cols)))
    ax.set_xticklabels(heat_cols, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(heat_cols, fontsize=8)
    ax.set_title("Top predictors move together, not as separate causes")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson r")
    ax.grid(False)
    save(fig, "04_correlation_heatmap.png")

    # --- scatters ---
    scatter_feats = list(order.head(6).index)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), sharey=True, constrained_layout=True)
    for ax, feat in zip(axes.ravel(), scatter_feats):
        ax.scatter(eda[feat], eda[cr.TARGET], s=8, alpha=0.28, color=BLUE, linewidths=0)
        ax.set_xlabel(f"{feat} (normalized 0–1)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"r = {corr[feat]:+.2f}", fontsize=11)
    axes[0, 0].set_ylabel("ViolentCrimesPerPop")
    axes[1, 0].set_ylabel("ViolentCrimesPerPop")
    fig.suptitle("Six strongest features against the crime rate (all 1,994 communities)")
    save(fig, "05_feature_scatters.png")

    # --- models, same split as crime_regression.py ---
    X = model_df[features]
    y = model_df[cr.TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cr.TEST_SIZE, random_state=cr.RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=cr.RANDOM_STATE)
    specs = cr.make_models(cv, cr.ALPHAS, cr.RANDOM_STATE)
    pipes = {}
    test_pred = {}
    for name, est in specs.items():
        pipe = cr.make_pipeline(est)
        pipe.fit(X_train, y_train)
        pipes[name] = pipe
        test_pred[name] = pipe.predict(X_test)

    # --- holdout metrics ---
    names = ["Linear", "Ridge", "Lasso", "Elastic net"]
    keys = ["Linear", "Ridge", "Lasso", "ElasticNet"]
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    test_r2 = [r2_score(y_test, test_pred[n]) for n in keys]
    test_rmse = [float(np.sqrt(mean_squared_error(y_test, test_pred[n]))) for n in keys]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(names, test_r2, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Test R²")
    axes[0].set_title("Holdout R²")
    for i, v in enumerate(test_r2):
        axes[0].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].bar(names, test_rmse, color=colors)
    axes[1].set_ylabel("Test RMSE (0–1 crime rate)")
    axes[1].set_title("Holdout RMSE")
    for i, v in enumerate(test_rmse):
        axes[1].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("399 held-out communities", y=1.03)
    save(fig, "06_holdout_metrics.png")

    # --- predicted vs actual ---
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, key, label in zip(axes.ravel(), keys, names):
        pred = test_pred[key]
        ax.scatter(y_test, pred, s=12, alpha=0.4, color=BLUE, linewidths=0)
        ax.plot([0, 1], [0, 1], color=RED, linewidth=1.2)
        ax.set_title(f"{label}  RMSE {np.sqrt(mean_squared_error(y_test, pred)):.3f}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 1.15)
    for ax in axes[1]:
        ax.set_xlabel("Actual crime rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted crime rate")
    fig.suptitle("Test-set predictions against the actual normalized crime rate")
    save(fig, "07_predicted_vs_actual.png")

    # --- residuals ---
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.0), constrained_layout=True)
    for i, (key, label) in enumerate(zip(keys, names)):
        resid = y_test - test_pred[key]
        axes[0, i].scatter(test_pred[key], resid, s=10, alpha=0.4, color=PURPLE, linewidths=0)
        axes[0, i].axhline(0, color=RED, linewidth=1)
        axes[0, i].set_title(label)
        axes[0, i].set_xlabel("Predicted crime rate")
        axes[1, i].hist(resid, bins=22, color=PURPLE, edgecolor="white")
        axes[1, i].set_xlabel("Residual (actual − predicted)")
    axes[0, 0].set_ylabel("Residual")
    axes[1, 0].set_ylabel("Test communities")
    fig.suptitle("Holdout residuals. Spread stays wide at high predicted crime.")
    save(fig, "08_residuals.png")

    # --- coefficients ---
    focus = []
    coefs = {}
    for key in keys:
        coef = pipes[key].named_steps["model"].coef_.ravel()
        coefs[key] = coef
        for idx in np.argsort(-np.abs(coef))[:8]:
            feat = features[int(idx)]
            if feat not in focus:
                focus.append(feat)
    focus = focus[:10]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ypos = np.arange(len(focus))
    height = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for offset, key, label, color in zip(offsets, keys, names, colors):
        mapping = dict(zip(features, coefs[key]))
        ax.barh(
            ypos + offset * height,
            [mapping[f] for f in focus],
            height=height,
            color=color,
            label=label,
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(focus)
    ax.axvline(0, color="#1c1c1c", linewidth=0.8)
    ax.set_xlabel("Standardized coefficient (change in crime rate per 1 SD)")
    ax.set_title("Least squares inflates copies; elastic net and lasso shrink them")
    ax.legend(frameon=False)
    save(fig, "09_coefficients.png")

    # --- gradient descent ---
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    X_s = scaler.transform(imputer.transform(X_train))
    y_s = y_train.astype(float)
    n, p = X_s.shape
    design = np.column_stack([np.ones(n), X_s])
    beta_ols = np.linalg.lstsq(design, y_s, rcond=None)[0]
    b_ols = float(beta_ols[0])
    w_ols = beta_ols[1:]
    ols_mse = float(np.mean((design @ beta_ols - y_s) ** 2))

    gram = (design.T @ design) / n
    lipschitz_gram = float(np.linalg.eigvalsh(gram)[-1])
    # MSE Hessian is 2 * gram, so the stable step is 1 / largest eigenvalue of gram.
    max_stable = 1.0 / lipschitz_gram
    schedules = {
        "slow": 0.15 * max_stable,
        "chosen": 0.8 * max_stable,
        "too_large": 1.35 * max_stable,
    }
    runs = {name: batch_gd(X_s, y_s, lr, max_iter=5000) for name, lr in schedules.items()}

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), constrained_layout=True)
    early = 250
    for name, color, label in (
        ("slow", BLUE, f"Slow lr={schedules['slow']:.4f}"),
        ("chosen", GREEN, f"Chosen lr={schedules['chosen']:.4f}"),
    ):
        series = runs[name]["mse"]
        axes[0].plot(np.arange(early), series[:early], color=color, linewidth=1.8, label=label)
        axes[1].plot(np.arange(len(series)), series, color=color, linewidth=1.6, label=label)
    for ax in axes:
        ax.axhline(ols_mse, color=GRAY, linestyle="--", linewidth=1.2, label=f"Least squares {ols_mse:.4f}")
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Training MSE")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_xlim(0, early)
    axes[0].set_ylim(0, 0.13)
    axes[0].set_title("First 250 steps")
    axes[1].set_ylim(0.015, 0.04)
    axes[1].set_title("Same runs, zoomed once the loss is small")
    fig.suptitle(
        f"Learning rate {schedules['too_large']:.3f} diverges in a few steps and is left off these axes"
    )
    save(fig, "10_gradient_descent_loss.png")

    chosen = runs["chosen"]
    dist = np.sqrt(
        (chosen["intercepts"] - b_ols) ** 2 + np.sum((chosen["weights"] - w_ols) ** 2, axis=1)
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].plot(np.arange(400), chosen["mse"][:400], color=GREEN, linewidth=1.8)
    axes[0].axhline(ols_mse, color=GRAY, linestyle="--", linewidth=1.2, label="Least squares")
    axes[0].set_xlabel("Gradient step")
    axes[0].set_ylabel("Training MSE")
    axes[0].set_title("Loss over the first 400 steps")
    axes[0].legend(frameon=False)
    axes[1].plot(dist, color=ORANGE, linewidth=1.8)
    axes[1].set_xlabel("Gradient step")
    axes[1].set_ylabel("Distance to least-squares weights")
    axes[1].set_title("Weights keep moving after the loss flattens")
    fig.suptitle(
        f"Batch gradient descent, learning rate {schedules['chosen']:.4f}, {p} standardized features",
        y=1.03,
    )
    save(fig, "11_gradient_descent_weights_vs_loss.png")

    highlight_idx = np.argsort(-np.abs(w_ols))[:4]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    steps = np.arange(chosen["weights"].shape[0])
    ax.plot(steps, chosen["weights"], color="#c5cad1", linewidth=0.6, alpha=0.7)
    highlight_colors = [BLUE, RED, ORANGE, GREEN]
    for color, idx in zip(highlight_colors, highlight_idx):
        feat = features[int(idx)]
        ax.plot(steps, chosen["weights"][:, idx], color=color, linewidth=1.8, label=feat)
        ax.axhline(w_ols[idx], color=color, linestyle=":", linewidth=1)
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("Standardized coefficient")
    ax.set_title("Weights start at 0. Dotted lines are the least-squares targets")
    ax.legend(frameon=False, loc="best")
    save(fig, "12_gradient_descent_coefficient_paths.png")

    # 2D contour: strongest collinear family-structure pair, plus a milder pair.
    def contour_pair(feat_a: str, feat_b: str, filename: str, title: str) -> None:
        ia, ib = features.index(feat_a), features.index(feat_b)
        pair = X_s[:, [ia, ib]]
        # Closed form on these two columns only, y centered so the bowl is in the weight plane.
        y_c = y_s - y_s.mean()
        pair = pair - pair.mean(axis=0)
        w_star = np.linalg.lstsq(pair, y_c, rcond=None)[0]
        # A fraction of the stable step keeps the path from bouncing across the bowl.
        lr2 = 0.22 / float(np.linalg.eigvalsh((pair.T @ pair) / n)[-1])
        w = np.array([0.75, 0.35])
        path = [w.copy()]
        for _ in range(160):
            err = pair @ w - y_c
            grad = (2.0 / n) * (pair.T @ err)
            w = w - lr2 * grad
            path.append(w.copy())
            if float(np.linalg.norm(w - w_star)) < 0.01:
                break
        path_a = np.array(path)
        span = np.vstack([path_a, w_star.reshape(1, 2)])
        lo = span.min(axis=0) - 0.35
        hi = span.max(axis=0) + 0.35
        g1, g2 = np.meshgrid(np.linspace(lo[0], hi[0], 80), np.linspace(lo[1], hi[1], 80))
        grid = np.column_stack([g1.ravel(), g2.ravel()])
        loss = np.mean((grid @ pair.T - y_c.reshape(1, -1)) ** 2, axis=1).reshape(g1.shape)
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        levels = np.quantile(loss, np.linspace(0.02, 0.92, 14))
        fill = ax.contourf(g1, g2, loss, levels=np.unique(levels), cmap="Blues")
        ax.contour(g1, g2, loss, levels=np.unique(levels), colors="white", linewidths=0.4, alpha=0.7)
        ax.plot(path_a[:, 0], path_a[:, 1], color=ORANGE, linewidth=1.8, marker="o", markersize=3.5)
        ax.scatter([path_a[0, 0]], [path_a[0, 1]], color=RED, s=40, zorder=4, label="Start")
        ax.scatter([w_star[0]], [w_star[1]], color=GREEN, marker="*", s=140, zorder=4, label="Least squares")
        ax.set_xlabel(f"{feat_a} weight")
        ax.set_ylabel(f"{feat_b} weight")
        ax.set_title(title)
        ax.legend(frameon=False, loc="best")
        fig.colorbar(fill, ax=ax, fraction=0.046, pad=0.04).set_label("Training MSE")
        ax.grid(False)
        save(fig, filename)

    fam_a, fam_b = "PctKids2Par", "PctIlleg"
    contour_pair(
        fam_a,
        fam_b,
        "13_gradient_descent_contour_family.png",
        "Descent on two family-structure features\n(a narrow valley, because the features overlap)",
    )

    mild = None
    for i, a in enumerate(list(order.head(8).index)):
        for b in list(order.head(8).index)[i + 1 :]:
            rho = abs(float(eda[[a, b]].corr().iloc[0, 1]))
            if mild is None or rho < mild[0]:
                mild = (rho, a, b)
    contour_pair(
        mild[1],
        mild[2],
        "14_gradient_descent_contour_bowl.png",
        f"Descent on {mild[1]} and {mild[2]}\n(less overlap, r = {mild[0]:.2f}, so the bowl is rounder)",
    )

    # Snapshots where the weight vector has closed a set fraction of the gap.
    snap_idx = [i for i in (0, 10, 25, 50, 100, 250, 1000, 5000) if i < len(dist)]

    shown = [features[int(i)] for i in highlight_idx]
    shown_ols = [round(float(w_ols[i]), 4) for i in highlight_idx]
    snapshots = []
    for idx in snap_idx:
        snapshots.append(
            {
                "iter": int(idx),
                "mse": round(float(chosen["mse"][idx]), 5),
                "distance": round(float(dist[idx]), 4),
                "coefs": [round(float(chosen["weights"][idx, j]), 4) for j in highlight_idx],
            }
        )

    loss_idx, loss_chosen = downsample(chosen["mse"], 24)
    _, loss_slow = downsample(runs["slow"]["mse"], 24)
    # Slow run may be shorter or longer; align by recomputing its own index if needed.
    slow_idx, loss_slow = downsample(runs["slow"]["mse"], 24)
    labels = [str(int(i)) for i in loss_idx]
    # Use the chosen-run iteration labels. Slow series is plotted on its own
    # downsampled grid only when the lengths match; otherwise resample by fraction.
    def on_grid(series: np.ndarray, grid: np.ndarray) -> list[float]:
        grid = grid[grid < len(series)]
        return [round(float(series[i]), 5) for i in grid]

    canvas = {
        "max_stable_lr": round(max_stable, 6),
        "learning_rates": {k: round(float(v), 6) for k, v in schedules.items()},
        "ols_mse": round(ols_mse, 5),
        "ols_intercept": round(b_ols, 4),
        "final_mse": round(float(chosen["mse"][-1]), 5),
        "final_distance": round(float(dist[-1]), 4),
        "start_mse": round(float(chosen["mse"][0]), 5),
        "start_distance": round(float(dist[0]), 4),
        "n_steps": int(len(chosen["mse"]) - 1),
        "diverged_large_step": bool(runs["too_large"]["diverged"]),
        "large_step_last_mse": round(float(runs["too_large"]["mse"][-1]), 4),
        "large_step_iters": int(len(runs["too_large"]["mse"]) - 1),
        "features": shown,
        "ols_coefs": shown_ols,
        "snapshots": snapshots,
        "loss_labels": [str(int(i)) for i in loss_idx],
        "loss_chosen": on_grid(chosen["mse"], loss_idx),
        "loss_slow": on_grid(runs["slow"]["mse"], slow_idx),
        "loss_slow_labels": [str(int(i)) for i in slow_idx],
        "mild_pair": [mild[1], mild[2], round(float(mild[0]), 3)],
        "train_mse_models": {
            name: round(float(np.mean((y_train - pipes[name].predict(X_train)) ** 2)), 5)
            for name in keys
        },
        "test_rmse": {name: round(float(v), 4) for name, v in zip(keys, test_rmse)},
        "gd_test_rmse": None,
    }
    # Test RMSE of the GD solution, using the training scaler.
    X_test_s = scaler.transform(imputer.transform(X_test))
    gd_pred = X_test_s @ chosen["w"] + chosen["b"]
    canvas["gd_test_rmse"] = round(float(np.sqrt(mean_squared_error(y_test, gd_pred))), 4)
    canvas["gd_test_r2"] = round(float(r2_score(y_test, gd_pred)), 4)
    canvas["linear_test_rmse"] = round(float(np.sqrt(mean_squared_error(y_test, test_pred["Linear"]))), 4)
    canvas["mae_chosen_vs_ols"] = round(float(np.max(np.abs(chosen["w"] - w_ols))), 6)

    (FIG / "gd_canvas.json").write_text(json.dumps(canvas, indent=2), encoding="utf-8")
    print(json.dumps({k: canvas[k] for k in (
        "max_stable_lr", "learning_rates", "ols_mse", "final_mse", "final_distance",
        "start_mse", "n_steps", "diverged_large_step", "features", "snapshots",
        "gd_test_r2", "gd_test_rmse", "mae_chosen_vs_ols", "mild_pair",
    )}, indent=2))


if __name__ == "__main__":
    main()
