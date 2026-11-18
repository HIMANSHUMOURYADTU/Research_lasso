"""Animate a regression line fitting, and re-score linear, ridge, and lasso.

The animation is one feature on purpose. PctKids2Par is the strongest linear
associate of the crime rate, so the moving line is visible. The full model is
a 100-feature hyperplane and cannot be drawn as one line.

The repeated splits refit least squares, ridge, lasso, and elastic net on five
80/20 cuts, retuning each penalty inside that training set.

Run:
  python animate_fit.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

import crime_regression as cr

FIG = cr.ROOT / "figures"
FIG.mkdir(exist_ok=True)
FEATURE = "PctKids2Par"
SEEDS = [0, 1, 7, 21, 42]
ALPHAS = np.logspace(-3, 2, 16)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#d0d4d8",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.grid": True,
        "grid.color": "#e6e8eb",
        "grid.linewidth": 0.6,
    }
)


def prepare():
    import pandas as pd

    df = cr.load_frame()
    predictive = [c for c in df.columns if c not in cr.NON_PREDICTIVE and c != cr.TARGET]
    high_missing = [c for c in predictive if float(df[c].isna().mean()) > cr.MISSING_DROP_THRESHOLD]
    features = [c for c in predictive if c not in high_missing]
    model_df = df[features + [cr.TARGET]].apply(pd.to_numeric, errors="coerce")
    return model_df, features


def repeated_splits(model_df, features) -> dict:
    rows = []
    X = model_df[features]
    y = model_df[cr.TARGET].to_numpy()
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cr.TEST_SIZE, random_state=seed
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        specs = cr.make_models(cv, ALPHAS, seed)
        scores = {"seed": seed}
        for name, est in specs.items():
            pipe = cr.make_pipeline(est)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            scores[name] = {
                "r2": round(float(r2_score(y_test, pred)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            }
            model = pipe.named_steps["model"]
            if hasattr(model, "alpha_"):
                scores[name]["alpha"] = round(float(np.ravel(model.alpha_)[0]), 6)
            if hasattr(model, "l1_ratio_"):
                scores[name]["l1_ratio"] = round(float(np.ravel(model.l1_ratio_)[0]), 4)
            if name in ("Lasso", "ElasticNet"):
                scores[name]["nonzero"] = int(np.sum(np.abs(model.coef_) > 1e-8))
        rows.append(scores)
        print(
            f"seed {seed:3}  "
            f"lin {scores['Linear']['r2']:.3f}  "
            f"ridge {scores['Ridge']['r2']:.3f}  "
            f"lasso {scores['Lasso']['r2']:.3f}  "
            f"enet {scores['ElasticNet']['r2']:.3f}"
        )

    summary = {}
    names = ("Linear", "Ridge", "Lasso", "ElasticNet")
    for name in names:
        r2 = np.array([row[name]["r2"] for row in rows])
        rmse = np.array([row[name]["rmse"] for row in rows])
        summary[name] = {
            "mean_r2": round(float(r2.mean()), 4),
            "std_r2": round(float(r2.std(ddof=1)), 4),
            "mean_rmse": round(float(rmse.mean()), 4),
        }
    wins = {name: 0 for name in names}
    for row in rows:
        best = max(names, key=lambda n: row[n]["r2"])
        wins[best] += 1
    for name in wins:
        summary[name]["wins"] = wins[name]
    return {"seeds": SEEDS, "splits": rows, "summary": summary}


def line_frames(model_df):
    y = model_df[cr.TARGET].to_numpy()
    x = model_df[FEATURE].to_numpy()
    X_train, _, y_train, _, x_train, _ = train_test_split(
        model_df[[FEATURE]],
        y,
        x,
        test_size=cr.TEST_SIZE,
        random_state=cr.RANDOM_STATE,
    )
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    # PctKids2Par has no missing values. Guard anyway.
    if np.isnan(x_train).any():
        x_train = np.where(np.isnan(x_train), np.nanmedian(x_train), x_train)

    n = len(y_train)
    design = np.column_stack([np.ones(n), x_train])
    b_ols, w_ols = np.linalg.lstsq(design, y_train, rcond=None)[0]
    gram = (design.T @ design) / n
    max_stable = 1.0 / float(np.linalg.eigvalsh(gram)[-1])
    # A moderate step still converges; we then keep fewer frames so the GIF stays short.
    lr = 0.55 * max_stable

    # Start with the wrong tilt: crime rising as two-parent households rise.
    w, b = 0.9, 0.05
    history = []
    for step in range(2500):
        err = b + w * x_train - y_train
        history.append((step, float(w), float(b), float(np.mean(err**2))))
        if step > 8 and abs(w - float(w_ols)) < 0.015 and abs(b - float(b_ols)) < 0.015:
            break
        w -= lr * (2.0 / n) * float(np.sum(err * x_train))
        b -= lr * (2.0 / n) * float(np.sum(err))

    early = np.linspace(0, max(len(history) // 4, 1), 24)
    late = np.linspace(len(history) // 4, len(history) - 1, 16)
    picks = np.unique(np.round(np.r_[early, late]).astype(int))
    shown = [history[int(i)] for i in picks]

    xs = np.linspace(0, 1, 50)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
    ax, ax_loss = axes
    ax.scatter(x_train, y_train, s=10, alpha=0.28, color="#1d4e89", linewidths=0)
    ax.plot(xs, b_ols + w_ols * xs, color="#8d96a0", linestyle="--", linewidth=1.3, label="Least squares")
    (line,) = ax.plot([], [], color="#d4760a", linewidth=2.2, label="Gradient step")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.12, 1.05)
    ax.set_xlabel("PctKids2Par (normalized 0–1)")
    ax.set_ylabel("ViolentCrimesPerPop (normalized 0–1)")
    ax.legend(frameon=False, loc="upper right")
    title = ax.set_title("")

    losses = [row[3] for row in history]
    steps = [row[0] for row in history]
    ax_loss.set_xlim(0, steps[-1])
    ax_loss.set_ylim(0, max(losses) * 1.08)
    ax_loss.set_xlabel("Gradient step")
    ax_loss.set_ylabel("Training MSE")
    ols_mse = float(np.mean((b_ols + w_ols * x_train - y_train) ** 2))
    ax_loss.axhline(ols_mse, color="#8d96a0", linestyle="--", linewidth=1)
    (loss_line,) = ax_loss.plot([], [], color="#2f7d4a", linewidth=1.8)
    ax_loss.set_title("Training error of this one-feature line")

    images = []
    for step, wt, bt, _loss in shown:
        line.set_data(xs, bt + wt * xs)
        upto = steps.index(step) + 1
        loss_line.set_data(steps[:upto], losses[:upto])
        title.set_text(f"Step {step}   slope {wt:+.2f}   intercept {bt:.2f}")
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()))

    gif_path = FIG / "15_line_fitting.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=140,
        loop=0,
        disposal=2,
    )
    for name, frame in (
        ("16_line_fit_start.png", images[0]),
        ("17_line_fit_middle.png", images[len(images) // 2]),
        ("18_line_fit_end.png", images[-1]),
    ):
        frame.convert("RGB").save(FIG / name, quality=95)
        print(f"wrote {FIG / name}")
    plt.close(fig)
    print(f"wrote {gif_path}  frames={len(images)}  lr={lr:.4f}  steps_run={len(history)}")
    return {
        "gif": str(gif_path),
        "frames": len(images),
        "learning_rate": round(float(lr), 4),
        "start_slope": history[0][1],
        "end_slope": round(history[-1][1], 4),
        "end_intercept": round(history[-1][2], 4),
        "ols_slope": round(float(w_ols), 4),
        "ols_intercept": round(float(b_ols), 4),
        "start_mse": round(history[0][3], 4),
        "end_mse": round(history[-1][3], 4),
        "steps": int(history[-1][0]),
    }


def main() -> None:
    import sys

    model_df, features = prepare()
    if "--animation-only" in sys.argv:
        print(json.dumps(line_frames(model_df), indent=2))
        return
    splits = repeated_splits(model_df, features)
    anim = line_frames(model_df)
    payload = {"repeated_splits": splits, "animation": anim}
    out = FIG / "comparison.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(splits["summary"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
