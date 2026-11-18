"""Communities and Crime: EDA plus linear, ridge, and lasso regression.

Source files:
  communities.names  UCI attribute dictionary
  communities.data   1,994 communities, 128 attributes, '?' = missing

Target is ViolentCrimesPerPop, already normalized to [0, 1].
Non-predictive fields (state, county, community, communityname, fold) are
dropped, as specified in the dictionary. Police-survey (LEMAS) fields that
are missing for most communities are dropped rather than imputed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
NAMES_PATH = ROOT / "communities.names"
DATA_PATH = ROOT / "communities.data"
OUT_PATH = ROOT / "analysis_results.json"

NON_PREDICTIVE = ["state", "county", "community", "communityname", "fold"]
TARGET = "ViolentCrimesPerPop"
MISSING_DROP_THRESHOLD = 0.50
RANDOM_STATE = 42
TEST_SIZE = 0.20
ALPHAS = np.logspace(-4, 2, 40)
# Mix of L1 and L2. 1 is the lasso, values near 0 are mostly ridge.
L1_RATIOS = (0.1, 0.5, 0.7, 0.9, 0.95)
MODEL_NAMES = ("Linear", "Ridge", "Lasso", "ElasticNet")


def load_column_names() -> list[str]:
    text = NAMES_PATH.read_text(encoding="utf-8", errors="replace")
    header = text.split("Attribute Information:")[0]
    cols = re.findall(r"@attribute\s+(\S+)\s+\S+", header)
    if not cols:
        raise RuntimeError("Could not parse @attribute names from communities.names")
    return cols


def load_frame() -> pd.DataFrame:
    cols = load_column_names()
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=cols,
        na_values="?",
        keep_default_na=True,
    )
    if df.shape[1] != len(cols):
        raise RuntimeError(f"Expected {len(cols)} columns, found {df.shape[1]}")
    return df


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> dict:
    n = int(len(y_true))
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = n - n_features - 1
    adj = float(1 - (1 - r2) * (n - 1) / denom) if denom > 0 else None
    return {
        "r2": round(r2, 4),
        "adj_r2": None if adj is None else round(adj, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "n": n,
        "n_features_in_adj": int(n_features),
    }


def make_models(cv, alphas, random_state: int, l1_ratios=L1_RATIOS) -> dict:
    """The four estimators compared in the report. Penalties are chosen by CV."""
    return {
        "Linear": LinearRegression(),
        "Ridge": RidgeCV(alphas=alphas, cv=cv),
        "Lasso": LassoCV(
            alphas=alphas,
            cv=cv,
            max_iter=50000,
            random_state=random_state,
        ),
        "ElasticNet": ElasticNetCV(
            l1_ratio=list(l1_ratios),
            alphas=alphas,
            cv=cv,
            max_iter=50000,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def coef_table(names: list[str], coef: np.ndarray, k: int = 12) -> list[dict]:
    order = np.argsort(-np.abs(coef))
    rows = []
    for idx in order[:k]:
        rows.append(
            {
                "feature": names[int(idx)],
                "coefficient": round(float(coef[idx]), 4),
                "abs_coefficient": round(float(abs(coef[idx])), 4),
            }
        )
    return rows


def decile_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    frame = pd.DataFrame({"y": y_true, "pred": y_pred})
    try:
        frame["bin"] = pd.qcut(frame["pred"], 10, duplicates="drop")
    except ValueError:
        frame["bin"] = pd.cut(frame["pred"], 10)
    grouped = frame.groupby("bin", observed=True).agg(
        mean_predicted=("pred", "mean"),
        mean_actual=("y", "mean"),
        n=("y", "size"),
    )
    labels = [f"D{i}" for i in range(1, len(grouped) + 1)]
    return {
        "labels": labels,
        "mean_predicted": [round(float(v), 4) for v in grouped["mean_predicted"]],
        "mean_actual": [round(float(v), 4) for v in grouped["mean_actual"]],
        "mean_residual": [
            round(float(a - p), 4)
            for a, p in zip(grouped["mean_actual"], grouped["mean_predicted"])
        ],
    }


def target_histogram(y: pd.Series, bins: int = 12) -> dict:
    counts, edges = np.histogram(y.to_numpy(), bins=bins, range=(0.0, 1.0))
    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(bins)]
    return {"labels": labels, "counts": [int(c) for c in counts]}


def main() -> None:
    df = load_frame()
    n_rows, n_cols = df.shape

    missing_counts = df.isna().sum()
    missing_rows = []
    for col, count in missing_counts[missing_counts > 0].sort_values(ascending=False).items():
        missing_rows.append(
            {
                "feature": col,
                "missing": int(count),
                "pct": round(100.0 * float(count) / n_rows, 2),
            }
        )

    y_all = df[TARGET].astype(float)
    hist = target_histogram(y_all)
    target_summary = {
        "mean": round(float(y_all.mean()), 4),
        "median": round(float(y_all.median()), 4),
        "std": round(float(y_all.std()), 4),
        "min": round(float(y_all.min()), 4),
        "max": round(float(y_all.max()), 4),
        "skew": round(float(y_all.skew()), 4),
        "pct_above_0_5": round(100.0 * float((y_all > 0.5).mean()), 2),
        "pct_at_least_0_9": round(100.0 * float((y_all >= 0.9).mean()), 2),
    }

    predictive = [c for c in df.columns if c not in NON_PREDICTIVE and c != TARGET]
    high_missing = [
        c
        for c in predictive
        if float(df[c].isna().mean()) > MISSING_DROP_THRESHOLD
    ]
    features = [c for c in predictive if c not in high_missing]

    model_df = df[features + [TARGET]].copy()
    for col in features:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df[TARGET] = pd.to_numeric(model_df[TARGET], errors="coerce")

    # EDA correlations use a median fill so the single OtherPerCap gap does not
    # drop a row. Model fitting imputes inside the train fold only.
    eda = model_df.copy()
    eda[features] = eda[features].fillna(eda[features].median(numeric_only=True))

    corr = eda[features + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
    corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)
    top_corr = [
        {"feature": feat, "correlation": round(float(corr_sorted.loc[feat]), 4)}
        for feat in corr_sorted.index[:15]
    ]

    feature_corr = eda[features].corr().abs()
    pairs = []
    cols = list(feature_corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = float(feature_corr.iloc[i, j])
            if value >= 0.90:
                pairs.append(
                    {
                        "a": cols[i],
                        "b": cols[j],
                        "abs_correlation": round(value, 4),
                    }
                )
    pairs.sort(key=lambda row: row["abs_correlation"], reverse=True)

    # VIF on the standardized, median-filled design matrix.
    x_vif = StandardScaler().fit_transform(eda[features].to_numpy())
    vifs = []
    for i, name in enumerate(features):
        y_i = x_vif[:, i]
        x_i = np.delete(x_vif, i, axis=1)
        r2_i = r2_score(y_i, LinearRegression().fit(x_i, y_i).predict(x_i))
        vif = float("inf") if r2_i >= 0.999 else float(1.0 / (1.0 - r2_i))
        vifs.append(vif)
    finite_vifs = [v for v in vifs if np.isfinite(v)]
    vif_summary = {
        "n_features": len(features),
        "n_vif_over_10": int(sum(v > 10 for v in finite_vifs)),
        "n_vif_over_30": int(sum(v > 30 for v in finite_vifs)),
        "n_infinite": int(sum(not np.isfinite(v) for v in vifs)),
        "median_vif": round(float(np.median(finite_vifs)), 2) if finite_vifs else None,
        "max_finite_vif": round(float(np.max(finite_vifs)), 2) if finite_vifs else None,
    }

    X = model_df[features]
    y = model_df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    baseline_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=float)
    baseline = regression_metrics(y_test, baseline_pred, n_features=0)
    baseline["name"] = "Mean baseline"

    specs = make_models(cv, ALPHAS, RANDOM_STATE)

    models = {}
    fitted = {}
    for name, estimator in specs.items():
        pipe = make_pipeline(estimator)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)
        coef = pipe.named_steps["model"].coef_.ravel()
        n_nonzero = int(np.sum(np.abs(coef) > 1e-8))
        entry = {
            "name": name,
            "train": regression_metrics(y_train, pred_train, n_nonzero),
            "test": regression_metrics(y_test, pred_test, n_nonzero),
            "n_nonzero": n_nonzero,
            "intercept": round(float(pipe.named_steps["model"].intercept_), 4),
            "top_coefficients": coef_table(features, coef, k=12),
            "calibration": decile_calibration(y_test, pred_test),
        }
        model = pipe.named_steps["model"]
        if hasattr(model, "alpha_"):
            entry["alpha"] = round(float(np.ravel(model.alpha_)[0]), 6)
        if hasattr(model, "l1_ratio_"):
            entry["l1_ratio"] = round(float(np.ravel(model.l1_ratio_)[0]), 4)
        models[name] = entry

    # Alpha path for the charts. Scaling is fit once on the training set here;
    # the alpha actually used was chosen inside the pipeline, which refits the
    # imputer and scaler on each CV fold. The two peaks agree.
    alpha_path = {
        "alphas": [round(float(a), 6) for a in ALPHAS],
        "ridge_r2": [],
        "lasso_r2": [],
        "elastic_r2": [],
    }
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    X_train_s = scaler.transform(imputer.transform(X_train))
    elastic_l1 = float(models["ElasticNet"]["l1_ratio"])

    # Recompute mean fold R² across the alpha grid for the tuning charts.
    # Elastic net is traced at the l1_ratio chosen above.
    from sklearn.model_selection import cross_val_score

    for alpha in ALPHAS:
        ridge_scores = cross_val_score(
            Ridge(alpha=float(alpha)),
            X_train_s,
            y_train,
            cv=cv,
            scoring="r2",
        )
        lasso_scores = cross_val_score(
            Lasso(alpha=float(alpha), max_iter=50000, random_state=RANDOM_STATE),
            X_train_s,
            y_train,
            cv=cv,
            scoring="r2",
        )
        elastic_scores = cross_val_score(
            ElasticNet(
                alpha=float(alpha),
                l1_ratio=elastic_l1,
                max_iter=50000,
                random_state=RANDOM_STATE,
            ),
            X_train_s,
            y_train,
            cv=cv,
            scoring="r2",
        )
        alpha_path["ridge_r2"].append(round(float(ridge_scores.mean()), 4))
        alpha_path["lasso_r2"].append(round(float(lasso_scores.mean()), 4))
        alpha_path["elastic_r2"].append(round(float(elastic_scores.mean()), 4))

    # Shared coefficient view: union of each model's top 8 features.
    focus = []
    for name in MODEL_NAMES:
        for row in models[name]["top_coefficients"][:8]:
            if row["feature"] not in focus:
                focus.append(row["feature"])
    focus = focus[:10]
    coef_lookup = {}
    for name in MODEL_NAMES:
        coef = fitted[name].named_steps["model"].coef_.ravel()
        mapping = {feat: round(float(c), 4) for feat, c in zip(features, coef)}
        coef_lookup[name] = [mapping[feat] for feat in focus]

    best_name = max(models, key=lambda n: models[n]["test"]["r2"])

    results = {
        "dataset": {
            "rows": int(n_rows),
            "columns": int(n_cols),
            "non_predictive_dropped": NON_PREDICTIVE,
            "high_missing_dropped": high_missing,
            "n_high_missing_dropped": len(high_missing),
            "n_features_used": len(features),
            "features_used": features,
            "missing_drop_threshold_pct": int(MISSING_DROP_THRESHOLD * 100),
            "remaining_missing_cells": int(model_df[features].isna().sum().sum()),
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "test_fraction": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "missing": missing_rows,
        "target": target_summary,
        "target_histogram": hist,
        "top_correlations": top_corr,
        "collinear_pairs": pairs[:12],
        "n_collinear_pairs_ge_0_90": len(pairs),
        "vif": vif_summary,
        "baseline": baseline,
        "models": models,
        "alpha_path": {
            "labels": [f"{a:.4g}" for a in ALPHAS[::4]],
            "ridge_r2": alpha_path["ridge_r2"][::4],
            "lasso_r2": alpha_path["lasso_r2"][::4],
            "elastic_r2": alpha_path["elastic_r2"][::4],
            "full_alphas": alpha_path["alphas"],
            "full_ridge_r2": alpha_path["ridge_r2"],
            "full_lasso_r2": alpha_path["lasso_r2"],
            "full_elastic_r2": alpha_path["elastic_r2"],
            "elastic_l1_ratio": models["ElasticNet"]["l1_ratio"],
        },
        "coefficient_compare": {
            "features": focus,
            "Linear": coef_lookup["Linear"],
            "Ridge": coef_lookup["Ridge"],
            "Lasso": coef_lookup["Lasso"],
            "ElasticNet": coef_lookup["ElasticNet"],
        },
        "best_test_model": best_name,
        "selected_alphas": {
            "Ridge": models["Ridge"].get("alpha"),
            "Lasso": models["Lasso"].get("alpha"),
            "ElasticNet": models["ElasticNet"].get("alpha"),
            "ElasticNet_l1_ratio": models["ElasticNet"].get("l1_ratio"),
        },
        "notes": [
            "All numeric attributes in this release are already normalized to roughly 0-1 with 3-SD clipping, so RMSE is on that normalized crime-rate scale, not raw crimes per 100,000.",
            "Coefficients are on standardized predictors: each value is the change in normalized ViolentCrimesPerPop for a one-standard-deviation increase in that feature, holding the other included features fixed.",
            "These are associations. Family structure, poverty, and racial composition move together, so a coefficient is not a causal effect of that variable.",
        ],
    }

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Rows: {n_rows}  Columns: {n_cols}")
    print(f"Features used: {len(features)}  Dropped for missingness: {len(high_missing)}")
    print(f"Train/test: {len(y_train)}/{len(y_test)}")
    print(f"Target mean={target_summary['mean']} median={target_summary['median']} skew={target_summary['skew']}")
    print(f"Collinear pairs |r|>=0.90: {len(pairs)}  VIF>10: {vif_summary['n_vif_over_10']}")
    print(f"Baseline test RMSE={baseline['rmse']} R2={baseline['r2']}")
    for name, entry in models.items():
        test = entry["test"]
        train = entry["train"]
        alpha = entry.get("alpha", "-")
        l1 = entry.get("l1_ratio")
        mix = f" l1_ratio={l1}" if l1 is not None else ""
        print(
            f"{name:11} alpha={alpha}{mix} nonzero={entry['n_nonzero']:3} "
            f"train R2={train['r2']:.4f} test R2={test['r2']:.4f} "
            f"test RMSE={test['rmse']:.4f} test MAE={test['mae']:.4f}"
        )
    print(f"Best on test R2: {best_name}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
