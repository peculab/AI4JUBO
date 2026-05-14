"""
Generate revision tables and figures for the JUBO 6-month mortality paper.

Outputs are written to C:/AI4JUBO/RESULTS by default.

Examples
--------
Run with local files:
    python revision_generate_results.py --training training_data_1014.xlsx --external external_validation_1014.xlsx

Run with Google Sheets, matching the original notebook:
    python revision_generate_results.py --use-google-sheets

Notes
-----
The original notebook labels the best XGBoost-style model as
"HybridXGBRF (Our Approach)". This script preserves that label for
continuity with the submitted manuscript tables, but also writes a
model_identity_note.txt file because the final notebook appears to use
an XGBClassifier object under that label.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: xgboost. Install it before running this script.") from exc

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: plotly. Install it before running this script.") from exc


WORKDIR = Path(r"C:\AI4JUBO")
DEFAULT_RESULTS_DIR = WORKDIR / "RESULTS"

TRAINING_SHEET_URL = "https://docs.google.com/spreadsheets/d/1qljyp9lq3QsZ7O2O7FQxm7taEWQi3F3bZgNMcQ7NJeE/edit?usp=sharing"
EXTERNAL_SHEET_URL = "https://docs.google.com/spreadsheets/d/1NFAhP8NUVsxzEq55siFA0yHvnXY5GWqiKGSOKC4y1Qg/edit?usp=sharing"
TRAINING_WORKSHEET = "training_data_1014"
EXTERNAL_WORKSHEET = "external_validation_1014"

OUTCOME_COL = "死亡標記"
DROP_COLUMNS = ["H01_NUM", "觀察天數"]

MODEL_ORDER = [
    "HybridXGBRF (Our Approach)",
    "XGBClassifier",
    "RandomForestClassifier",
    "LogisticRegression (max_iter=1000)",
    "Ridge",
    "Elastic",
    "Lasso",
    "LogisticRegression (max_iter=200)",
]

FEATURE_NAME_MAP = {
    "六個月內住院次數": "Hospitalizations within 6 Months",
    "ADL_last_score": "ADL Last Score",
    "BW_diff_seq": "Body Weight Change",
    "ADL_std": "ADL Standard Deviation",
    "ADL_Min": "ADL Minimum",
    "性別_is_male": "Male",
    "BW_last": "Body Weight (Last)",
    "ADL_總分_max": "ADL Total Max",
    "BW_first": "Body Weight (First)",
    "DNR_flag": "DNR Flag",
    "ADL_first_score": "ADL First Score",
    "意識總分_diff": "Consciousness Score Difference",
    "預估年齡": "Estimated Age",
    "last_ 意識總分": "Consciousness Score (Last)",
    "使用呼吸輔具": "Use of Respiratory Aid",
    "ADL_diff_seq": "ADL Change",
    "diff_has_feeding_tube": "Feeding Tube Change",
    "ADL_last_CouldNot": "ADL Last Could Not Perform",
    "ADL_明顯惡化": "ADL Significant Deterioration",
    "ADL_Max": "ADL Maximum",
    "ADL_first_CouldNot": "ADL First Could Not Perform",
    "diff_has_denture": "Denture Change",
    "last_has_denture": "Has Denture (Last)",
    "had_fall": "Had Fall",
    "last_has_feeding_tube": "Has Feeding Tube (Last)",
    "first_has_denture": "Has Denture (First)",
    "意識總分Max": "Consciousness Score Max",
    "first_has_feeding_tube": "Has Feeding Tube (First)",
    "first_ 意識總分": "Consciousness Score (First)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate revision analyses into RESULTS.")
    parser.add_argument("--training", type=str, default=None, help="Path to training_data_1014 csv/xlsx.")
    parser.add_argument("--external", type=str, default=None, help="Path to external_validation_1014 csv/xlsx.")
    parser.add_argument("--use-google-sheets", action="store_true", help="Read the original Google Sheets.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Output directory.")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap replicates for final outputs.")
    parser.add_argument("--n-boot-fast", type=int, default=500, help="Bootstrap replicates for subgroup/calibration.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP outputs.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip 5-fold internal CV.")
    parser.add_argument(
        "--supplement-data-dir",
        type=str,
        default=str(WORKDIR / "DATA"),
        help="Directory containing supplemental revision data such as excluded residents and area_size.xlsx.",
    )
    return parser.parse_args()


def ensure_results_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)
    (path / "tables").mkdir(exist_ok=True)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return numeric_clean(df)


def read_google_sheet(url: str, worksheet_name: str) -> pd.DataFrame:
    public_xlsx_url = url.split("/edit", 1)[0] + "/export?format=xlsx"
    try:
        return numeric_clean(pd.read_excel(public_xlsx_url, sheet_name=worksheet_name))
    except Exception:
        pass

    try:
        import gspread
        from google.colab import auth  # type: ignore

        auth.authenticate_user()
        from google.auth import default  # type: ignore

        creds, _ = default()
        gc = gspread.authorize(creds)
    except Exception:
        try:
            import gspread

            gc = gspread.oauth()
        except Exception as exc:
            raise RuntimeError(
                "Could not authenticate Google Sheets. Use --training and --external local files, "
                "or run in Colab with Google authentication."
            ) from exc
    ws = gc.open_by_url(url).worksheet(worksheet_name)
    return numeric_clean(pd.DataFrame(ws.get_all_records()))


def numeric_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
    return out


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.training and args.external:
        return read_table(args.training), read_table(args.external)
    if args.use_google_sheets:
        return (
            read_google_sheet(TRAINING_SHEET_URL, TRAINING_WORKSHEET),
            read_google_sheet(EXTERNAL_SHEET_URL, EXTERNAL_WORKSHEET),
        )
    raise SystemExit(
        "Please provide --training and --external local files, or use --use-google-sheets.\n"
        "The workspace currently does not contain local csv/xlsx source data."
    )


def select_features(train_df: pd.DataFrame, missing_cutoff: float = 0.30) -> list[str]:
    missing_ratio = train_df.isna().mean()
    features = missing_ratio[missing_ratio < missing_cutoff].index.tolist()
    if OUTCOME_COL not in features:
        features.append(OUTCOME_COL)
    for col in DROP_COLUMNS:
        if col in features:
            features.remove(col)
    return features


def prepare_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")
    data = df[features].copy().fillna(0)
    X = data.drop(columns=[OUTCOME_COL])
    y = df[OUTCOME_COL].astype(int)
    return X, y


def build_models() -> dict[str, object]:
    return {
        "HybridXGBRF (Our Approach)": XGBClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=3,
            random_state=42,
            eval_metric="logloss",
            subsample=1.0,
            verbosity=0,
        ),
        "LogisticRegression (max_iter=200)": LogisticRegression(max_iter=200),
        "XGBClassifier": XGBClassifier(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=5,
            random_state=42,
            eval_metric="logloss",
        ),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression (max_iter=1000)": LogisticRegression(max_iter=1000),
        "Ridge": make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l2", solver="saga", max_iter=1000, random_state=42),
        ),
        "Lasso": make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l1", solver="saga", max_iter=1000, random_state=42),
        ),
        "Elastic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                max_iter=1000,
                random_state=42,
            ),
        ),
    }


def get_positive_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba[:, 1] if np.ndim(proba) == 2 else proba, dtype=float)
    if hasattr(model, "decision_function"):
        margin = model.decision_function(X)
        margin = np.clip(margin, -50, 50)
        return 1.0 / (1.0 + np.exp(-margin))
    return np.asarray(model.predict(X), dtype=float)


def point_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "PPV": tp / (tp + fp) if (tp + fp) else np.nan,
        "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUROC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "Brier": brier_score_loss(y_true, y_proba),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def stratified_bootstrap_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 2000,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    keys = ["Accuracy", "Precision", "Recall", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "AUROC", "Brier"]
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        m = point_metrics(y_true[idx], y_proba[idx], threshold=threshold)
        for k in keys:
            boot[k].append(m[k])
    out = {}
    for k, vals in boot.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "SD": float(np.nanstd(arr, ddof=1)),
            "CI_low": float(np.nanpercentile(arr, 2.5)),
            "CI_high": float(np.nanpercentile(arr, 97.5)),
        }
    return out


def fmt_ci(value: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def save_df(df: pd.DataFrame, path_base: Path) -> None:
    df.to_csv(path_base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    df.to_excel(path_base.with_suffix(".xlsx"), index=False)


def save_plot(fig: go.Figure, path_base: Path) -> None:
    fig.write_html(str(path_base.with_suffix(".html")))
    try:
        fig.write_image(str(path_base.with_suffix(".png")), scale=3)
    except Exception:
        pass


def internal_cv(
    models: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    n_boot: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_probs_all: dict[str, np.ndarray] = {}
    trained_models: dict[str, object] = {}
    fold_rows = []
    summary_rows = []
    y_arr = y.to_numpy()

    for model_name, model in models.items():
        print(f"[CV] {model_name}")
        oof_probs = np.zeros(len(y_arr), dtype=float)
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            estimator = clone(model)
            try:
                estimator.fit(X_train, y_train)
            except ValueError:
                imputer = SimpleImputer(strategy="median")
                X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
                X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
                estimator.fit(X_train, y_train)
            prob = get_positive_proba(estimator, X_test)
            oof_probs[test_idx] = prob
            m = point_metrics(y_test.to_numpy(), prob)
            m.update({"Model": model_name, "Fold": fold})
            fold_metrics.append(m)
            fold_rows.append(m)
        oof_probs_all[model_name] = oof_probs

        final_estimator = clone(model)
        final_estimator.fit(X.fillna(0), y)
        trained_models[model_name] = final_estimator

        point = point_metrics(y_arr, oof_probs)
        ci = stratified_bootstrap_metrics(y_arr, oof_probs, n_boot=n_boot, random_state=random_state)
        row = {"Model": model_name}
        for metric in ["AUROC", "Accuracy", "Precision", "Recall", "Sensitivity", "Specificity", "F1", "Brier"]:
            row[metric] = point[metric]
            row[f"{metric} 95% CI"] = fmt_ci(point[metric], ci[metric]["CI_low"], ci[metric]["CI_high"])
            row[f"{metric} SD"] = ci[metric]["SD"]
        summary_rows.append(row)

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(fold_rows),
        oof_probs_all,
        y_arr,
        trained_models,
    )


def external_validation(
    models: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_ext: pd.DataFrame,
    y_ext: pd.Series,
    n_boot: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    summary_rows = []
    pretty_rows = []
    probs_all: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}
    y_arr = y_ext.to_numpy()

    for model_name, model in models.items():
        print(f"[External] {model_name}")
        estimator = clone(model)
        estimator.fit(X_train.fillna(0), y_train)
        fitted[model_name] = estimator
        prob = get_positive_proba(estimator, X_ext.fillna(0))
        probs_all[model_name] = prob
        point = point_metrics(y_arr, prob)
        ci = stratified_bootstrap_metrics(y_arr, prob, n_boot=n_boot, random_state=random_state)
        row = {"Model": model_name}
        pretty = {"Model": model_name}
        for metric in ["AUROC", "Accuracy", "Precision", "Recall", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "Brier"]:
            row[metric] = point[metric]
            row[f"{metric} SD"] = ci[metric]["SD"]
            row[f"{metric} 95% CI Low"] = ci[metric]["CI_low"]
            row[f"{metric} 95% CI High"] = ci[metric]["CI_high"]
            pretty[metric] = fmt_ci(point[metric], ci[metric]["CI_low"], ci[metric]["CI_high"])
        summary_rows.append(row)
        pretty_rows.append(pretty)
    return pd.DataFrame(summary_rows), pd.DataFrame(pretty_rows), probs_all, fitted


def threshold_table(y_true: np.ndarray, y_proba: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    n = len(y_true)
    for threshold in thresholds:
        m = point_metrics(y_true, y_proba, threshold)
        net_benefit = (m["TP"] / n) - (m["FP"] / n) * (threshold / (1 - threshold))
        rows.append({"Threshold": threshold, "Net Benefit": net_benefit, **m})
    return pd.DataFrame(rows)


def decision_curve_df(y_true: np.ndarray, prob_dict: dict[str, np.ndarray], thresholds: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    n = len(y_true)
    prevalence = y_true.mean()
    rows = []
    for threshold in thresholds:
        rows.append({"Model": "Treat None", "Threshold": threshold, "Net Benefit": 0.0})
        treat_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        rows.append({"Model": "Treat All", "Threshold": threshold, "Net Benefit": treat_all})
        for model_name, probs in prob_dict.items():
            pred = (probs >= threshold).astype(int)
            tp = np.sum((pred == 1) & (y_true == 1))
            fp = np.sum((pred == 1) & (y_true == 0))
            nb = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            rows.append({"Model": model_name, "Threshold": threshold, "Net Benefit": nb})
    return pd.DataFrame(rows)


def plot_decision_curve(dca: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for model, sub in dca.groupby("Model", sort=False):
        dash = "dash" if model == "Treat None" else "dot" if model == "Treat All" else "solid"
        fig.add_trace(
            go.Scatter(
                x=sub["Threshold"],
                y=sub["Net Benefit"],
                mode="lines",
                name=model,
                line=dict(dash=dash),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Threshold probability",
        yaxis_title="Net benefit",
        template="plotly_white",
        width=900,
        height=600,
    )
    return fig


def calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    eps = 1e-6
    p = np.clip(y_proba, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    try:
        lr.fit(logit_p, y_true)
        intercept = float(lr.intercept_[0])
        slope = float(lr.coef_[0][0])
    except Exception:
        intercept = np.nan
        slope = np.nan
    expected = float(np.sum(y_proba))
    observed = float(np.sum(y_true))
    return {
        "Calibration Intercept": intercept,
        "Calibration Slope": slope,
        "Observed": observed,
        "Expected": expected,
        "Observed/Expected": observed / expected if expected else np.nan,
        "Brier": brier_score_loss(y_true, y_proba),
    }


def calibration_metrics_with_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    point = calibration_metrics(y_true, y_proba)
    boot = {k: [] for k in point.keys()}
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        metrics = calibration_metrics(y_true[idx], y_proba[idx])
        for key, value in metrics.items():
            boot[key].append(value)

    rows = []
    for key, value in point.items():
        arr = np.asarray(boot[key], dtype=float)
        rows.append(
            {
                "Metric": key,
                "Estimate": value,
                "95% CI Low": float(np.nanpercentile(arr, 2.5)),
                "95% CI High": float(np.nanpercentile(arr, 97.5)),
                "Formatted": fmt_ci(value, float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))),
            }
        )
    return pd.DataFrame(rows)


def risk_decile_table(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["Risk Decile"] = pd.qcut(df["p"], q=10, duplicates="drop")
    rows = []
    for i, (_, sub) in enumerate(df.groupby("Risk Decile", observed=True), start=1):
        rows.append(
            {
                "Decile": i,
                "N": len(sub),
                "Deaths": int(sub["y"].sum()),
                "Mean Predicted Risk": sub["p"].mean(),
                "Observed Risk": sub["y"].mean(),
                "Min Predicted Risk": sub["p"].min(),
                "Max Predicted Risk": sub["p"].max(),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration_with_histogram(y_true: np.ndarray, prob_dict: dict[str, np.ndarray], title: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash")), row=1, col=1)
    for model_name, probs in prob_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
        fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name=model_name), row=1, col=1)
    main_model = "HybridXGBRF (Our Approach)"
    if main_model in prob_dict:
        fig.add_trace(
            go.Histogram(x=prob_dict[main_model], nbinsx=30, name="Predicted probability distribution"),
            row=2,
            col=1,
        )
    fig.update_layout(title=title, template="plotly_white", width=900, height=750, bargap=0.05)
    fig.update_yaxes(title_text="Observed event rate", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Predicted probability", row=2, col=1)
    return fig


def paired_bootstrap_auc(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    name_a: str,
    name_b: str,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        diffs.append(roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx]))
    diffs = np.asarray(diffs)
    point = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    p_two_sided = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return pd.DataFrame(
        [
            {
                "Model A": name_a,
                "Model B": name_b,
                "AUROC A": roc_auc_score(y_true, prob_a),
                "AUROC B": roc_auc_score(y_true, prob_b),
                "AUROC Difference A-B": point,
                "95% CI Low": np.percentile(diffs, 2.5),
                "95% CI High": np.percentile(diffs, 97.5),
                "Bootstrap P Value": min(float(p_two_sided), 1.0),
            }
        ]
    )


def subgroup_masks(raw_df: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {}
    if "ADL_明顯惡化" in raw_df.columns:
        masks["ADL improvement"] = raw_df["ADL_明顯惡化"] == 0
        masks["ADL decline"] = raw_df["ADL_明顯惡化"] == 1
    if "性別_is_male" in raw_df.columns:
        masks["Female"] = raw_df["性別_is_male"] == 0
        masks["Male"] = raw_df["性別_is_male"] == 1
    if "預估年齡" in raw_df.columns:
        masks["Age <= 85"] = raw_df["預估年齡"] <= 85
        masks["Age > 85"] = raw_df["預估年齡"] > 85
    return masks


def subgroup_performance_with_ci(
    y_true: pd.Series,
    y_proba: np.ndarray,
    raw_df: pd.DataFrame,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    for name, mask in subgroup_masks(raw_df).items():
        mask = mask.fillna(False).to_numpy(dtype=bool)
        if mask.sum() == 0:
            continue
        yt = y_true.to_numpy()[mask]
        yp = y_proba[mask]
        point = point_metrics(yt, yp)
        ci = stratified_bootstrap_metrics(yt, yp, n_boot=n_boot, random_state=random_state)
        row = {
            "Subgroup": name,
            "Support (n)": int(len(yt)),
            "Positives (n)": int(np.sum(yt)),
        }
        for metric in ["Accuracy", "Precision", "Recall", "F1", "AUROC"]:
            row[metric] = point[metric]
            row[f"{metric} 95% CI"] = fmt_ci(point[metric], ci[metric]["CI_low"], ci[metric]["CI_high"])
        rows.append(row)
    return pd.DataFrame(rows)


def missingness_indicator_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = df[OUTCOME_COL].astype(int)
    for col in df.columns:
        if col == OUTCOME_COL:
            continue
        miss = df[col].isna().astype(int)
        if miss.sum() == 0:
            continue
        miss_rate_dead = miss[y == 1].mean()
        miss_rate_alive = miss[y == 0].mean()
        rows.append(
            {
                "Variable": col,
                "Missing N": int(miss.sum()),
                "Missing Percent": miss.mean(),
                "Missing Percent Dead": miss_rate_dead,
                "Missing Percent Alive": miss_rate_alive,
                "Difference Dead-Alive": miss_rate_dead - miss_rate_alive,
            }
        )
    return pd.DataFrame(rows).sort_values("Missing Percent", ascending=False)


def facility_missingness(df: pd.DataFrame) -> pd.DataFrame:
    facility_candidates = ["dbname", "facility_id", "institution_id", "機構代碼", "機構ID", "H01_NUM"]
    facility_col = next((c for c in facility_candidates if c in df.columns and df[c].notna().any()), None)
    if facility_col is None:
        return pd.DataFrame([{"Note": "No usable facility identifier column found."}])
    feature_cols = [c for c in df.columns if c != OUTCOME_COL]
    rows = []
    for facility, sub in df.groupby(facility_col):
        rows.append(
            {
                "Facility": facility,
                "N": len(sub),
                "Overall Missing Percent": sub[feature_cols].isna().mean().mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("Overall Missing Percent", ascending=False)


def load_supplemental_revision_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    complete_path = data_dir / "analysis_data_filtering_out_0514.csv"
    broad_path = data_dir / "analysis_data_filtering_out_included_ADL_missing_0514.csv"
    area_path = data_dir / "area_size.xlsx"
    if complete_path.exists():
        out["excluded_adl_complete"] = pd.read_csv(complete_path)
    if broad_path.exists():
        out["excluded_with_adl_missing"] = pd.read_csv(broad_path)
    if area_path.exists():
        out["facility_area_size"] = pd.read_excel(area_path, sheet_name="訓練資料_機構大小")
        out["excluded_exit_reason_summary"] = pd.read_excel(area_path, sheet_name="排除個案_結案原因分析")
        out["excluded_region_summary_source"] = pd.read_excel(area_path, sheet_name="排除個案_所在區域分析")
    return out


def add_facility_area_size(df: pd.DataFrame, facility_area_size: pd.DataFrame | None) -> pd.DataFrame:
    if facility_area_size is None or "dbname" not in df.columns:
        return df.copy()
    cols = [c for c in ["dbname", "核定床數", "機構大小層級", "區域"] if c in facility_area_size.columns]
    if "dbname" not in cols:
        return df.copy()
    left = df.copy()
    right = facility_area_size[cols].drop_duplicates("dbname").copy()
    if left["dbname"].notna().sum() == 0:
        return left
    left["dbname"] = left["dbname"].astype(str)
    right["dbname"] = right["dbname"].astype(str)
    return left.merge(right, on="dbname", how="left")


def anti_join_excluded_rows(broad: pd.DataFrame, complete: pd.DataFrame) -> pd.DataFrame:
    key_cols = [c for c in ["H01_NUM", "dbname", "入家日期"] if c in broad.columns and c in complete.columns]
    if not key_cols:
        return broad.copy()
    complete_keys = set(map(tuple, complete[key_cols].astype(str).to_numpy()))
    broad_keys = list(map(tuple, broad[key_cols].astype(str).to_numpy()))
    return broad[[key not in complete_keys for key in broad_keys]].copy()


def format_mean_sd(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "NA"
    return f"{values.mean():.1f} ({values.std(ddof=1):.1f})"


def format_median_iqr(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "NA"
    return f"{values.median():.1f} ({values.quantile(0.25):.1f}-{values.quantile(0.75):.1f})"


def format_n_pct(series: pd.Series, positive_value: object = 1) -> str:
    values = series.dropna()
    if values.empty:
        return "NA"
    n = int((values == positive_value).sum())
    return f"{n} ({n / len(values) * 100:.1f}%)"


def baseline_comparison_table(included_df: pd.DataFrame, excluded_df: pd.DataFrame) -> pd.DataFrame:
    variables = [
        ("Age, years", "預估年齡", "continuous"),
        ("Male sex", "性別_is_male", "binary"),
        ("DNR", "DNR_flag", "binary"),
        ("Initial ADL score", "ADL_first_score", "continuous"),
        ("Maximum ADL score", "ADL_總分_max", "continuous"),
        ("Initial body weight", "BW_first", "continuous"),
        ("Hospitalizations within 6 months", "六個月內住院次數", "continuous"),
        ("Initial feeding tube", "first_has_feeding_tube", "binary"),
        ("Respiratory aid / oxygen use", "使用呼吸輔具", "binary"),
        ("Fall history", "had_fall", "binary"),
        ("Approved beds", "核定床數", "continuous"),
        ("Small facility (<50 beds)", "機構大小層級", "category_small"),
        ("Large facility (>150 beds)", "機構大小層級", "category_large"),
    ]
    rows = [
        {
            "Characteristic": "N residents",
            "Included residents": str(len(included_df)),
            "Excluded residents": str(len(excluded_df)),
            "Missing/Notes": "",
        }
    ]
    for label, col, kind in variables:
        if col not in included_df.columns and col not in excluded_df.columns:
            continue
        inc = included_df[col] if col in included_df.columns else pd.Series(dtype=float)
        exc = excluded_df[col] if col in excluded_df.columns else pd.Series(dtype=float)
        if kind == "continuous":
            inc_text = format_mean_sd(inc)
            exc_text = format_mean_sd(exc)
        elif kind == "binary":
            inc_text = format_n_pct(inc)
            exc_text = format_n_pct(exc)
        elif kind == "category_small":
            inc_text = format_n_pct(inc, "小")
            exc_text = format_n_pct(exc, "小")
        else:
            inc_text = format_n_pct(inc, "大")
            exc_text = format_n_pct(exc, "大")
        rows.append(
            {
                "Characteristic": label,
                "Included residents": inc_text,
                "Excluded residents": exc_text,
                "Missing/Notes": f"Included missing={int(inc.isna().sum())}; excluded missing={int(exc.isna().sum())}",
            }
        )
    return pd.DataFrame(rows)


def excluded_baseline_summary(excluded_df: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame()
    return baseline_comparison_table(empty, excluded_df).rename(columns={"Excluded residents": "Excluded residents summary"})


def summarize_by_facility_stratum(df: pd.DataFrame, group_col: str, cohort_name: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    feature_cols = [c for c in df.columns if c not in [OUTCOME_COL, "dbname", "H01_NUM"]]
    rows = []
    for group, sub in df.dropna(subset=[group_col]).groupby(group_col):
        row = {
            "Cohort": cohort_name,
            "Stratum": group,
            "N residents": len(sub),
            "N facilities": sub["dbname"].nunique() if "dbname" in sub.columns else np.nan,
            "Overall missing percent": sub[feature_cols].isna().mean().mean(),
        }
        if OUTCOME_COL in sub.columns:
            row["Death/event rate"] = pd.to_numeric(sub[OUTCOME_COL], errors="coerce").mean()
        if "觀察天數" in sub.columns:
            row["Median observation days"] = pd.to_numeric(sub["觀察天數"], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows)


def write_supplemental_revision_outputs(
    results_dir: Path,
    data_dir: Path,
    included_df: pd.DataFrame | None = None,
) -> dict[str, int | str]:
    data = load_supplemental_revision_data(data_dir)
    if not data:
        return {"supplemental_data_status": f"No supplemental DATA files found in {data_dir}"}

    facility_info = data.get("facility_area_size")
    excluded_complete = data.get("excluded_adl_complete")
    excluded_broad = data.get("excluded_with_adl_missing")
    excluded_primary = excluded_broad if excluded_broad is not None else excluded_complete
    manifest_update: dict[str, int | str] = {"supplemental_data_dir": str(data_dir)}

    if "excluded_exit_reason_summary" in data:
        save_df(data["excluded_exit_reason_summary"], results_dir / "tables" / "excluded_exit_reason_summary")
    if "excluded_region_summary_source" in data:
        save_df(data["excluded_region_summary_source"], results_dir / "tables" / "excluded_region_summary")

    if excluded_primary is not None:
        excluded_primary = add_facility_area_size(excluded_primary, facility_info)
        save_df(excluded_baseline_summary(excluded_primary), results_dir / "tables" / "excluded_residents_baseline_summary")
        manifest_update["excluded_residents_n"] = int(len(excluded_primary))
        if excluded_complete is not None and excluded_broad is not None:
            adl_missing_only = add_facility_area_size(anti_join_excluded_rows(excluded_broad, excluded_complete), facility_info)
            save_df(excluded_baseline_summary(adl_missing_only), results_dir / "tables" / "excluded_adl_missing_baseline_summary")
            manifest_update["excluded_adl_missing_n"] = int(len(adl_missing_only))

    stratum_frames_size = []
    stratum_frames_region = []
    if included_df is not None:
        included_aug = add_facility_area_size(included_df, facility_info)
        stratum_frames_size.append(summarize_by_facility_stratum(included_aug, "機構大小層級", "Included analytic cohort"))
        stratum_frames_region.append(summarize_by_facility_stratum(included_aug, "區域", "Included analytic cohort"))
    if excluded_primary is not None:
        stratum_frames_size.append(summarize_by_facility_stratum(excluded_primary, "機構大小層級", "Excluded residents"))
        stratum_frames_region.append(summarize_by_facility_stratum(excluded_primary, "區域", "Excluded residents"))
    if stratum_frames_size:
        size_df = pd.concat([x for x in stratum_frames_size if not x.empty], ignore_index=True)
        if not size_df.empty:
            save_df(size_df, results_dir / "tables" / "facility_size_missingness_and_outcome")
    if stratum_frames_region:
        region_df = pd.concat([x for x in stratum_frames_region if not x.empty], ignore_index=True)
        if not region_df.empty:
            save_df(region_df, results_dir / "tables" / "facility_region_missingness_and_outcome")
    if included_df is not None and excluded_primary is not None:
        save_df(
            baseline_comparison_table(add_facility_area_size(included_df, facility_info), excluded_primary),
            results_dir / "tables" / "included_vs_excluded_insufficient_followup",
        )
    return manifest_update


def plot_roc(prob_dict: dict[str, np.ndarray], y_true: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    for model_name in [m for m in MODEL_ORDER if m in prob_dict]:
        fpr, tpr, _ = roc_curve(y_true, prob_dict[model_name])
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC={auc(fpr, tpr):.3f})")
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        width=900,
        height=650,
    )
    return fig


def plot_confusion_matrices(prob_dict: dict[str, np.ndarray], y_true: np.ndarray, title: str) -> go.Figure:
    available = [m for m in MODEL_ORDER if m in prob_dict]
    n_cols = 2
    n_rows = math.ceil(len(available) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=available)
    for i, model_name in enumerate(available):
        row = i // n_cols + 1
        col = i % n_cols + 1
        pred = (prob_dict[model_name] >= 0.5).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[1, 0])
        row_sums = cm.sum(axis=1, keepdims=True)
        pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
        text = [[f"{pct[r, c]:.1f}%<br>({cm[r, c]})" for c in range(2)] for r in range(2)]
        fig.add_trace(
            go.Heatmap(
                z=pct,
                text=text,
                texttemplate="%{text}",
                x=["Predicted 1", "Predicted 0"],
                y=["Actual 1", "Actual 0"],
                coloraxis="coloraxis",
            ),
            row=row,
            col=col,
        )
    fig.update_layout(
        title=title,
        coloraxis=dict(colorscale="Blues", cmin=0, cmax=100),
        template="plotly_white",
        width=950,
        height=max(450, 330 * n_rows),
    )
    return fig


def shap_outputs(model: object, X: pd.DataFrame, results_dir: Path, random_state: int) -> None:
    try:
        import shap
    except Exception:
        print("[SHAP] shap is not installed; skipping SHAP outputs.")
        return

    sample = X.sample(n=min(1000, len(X)), random_state=random_state)
    try:
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            base_score = booster.attr("base_score")
            if isinstance(base_score, str) and base_score.startswith("[") and base_score.endswith("]"):
                cleaned = re.split(r"[,\s]+", base_score.strip("[]").strip())[0]
                booster.set_attr(base_score=cleaned)
                booster.set_param({"base_score": cleaned})
            explainer = shap.TreeExplainer(booster)
            values = explainer.shap_values(sample)
            if isinstance(values, list):
                values = values[1]
            values = np.asarray(values)
        else:
            explainer = shap.Explainer(lambda z: get_positive_proba(model, pd.DataFrame(z, columns=X.columns)), sample)
            exp = explainer(sample)
            values = np.asarray(getattr(exp, "values", exp))
            if values.ndim == 3:
                values = values[:, :, 1]
    except Exception as exc:
        print(f"[SHAP] Tree/auto explainer failed: {exc}")
        print("[SHAP] Falling back to KernelSHAP on final predict_proba output.")
        try:
            background = X.sample(n=min(100, len(X)), random_state=random_state)
            sample = X.sample(n=min(300, len(X)), random_state=random_state + 1)

            def f_prob(z):
                return get_positive_proba(model, pd.DataFrame(z, columns=X.columns))

            explainer = shap.KernelExplainer(f_prob, background)
            values = explainer.shap_values(sample, nsamples=100)
            if isinstance(values, list):
                values = values[0]
            values = np.asarray(values)
        except Exception as exc2:
            print(f"[SHAP] Failed: {exc2}")
            return

    mean_abs = np.abs(values).mean(axis=0)
    imp = pd.DataFrame(
        {
            "Feature": sample.columns,
            "Feature_EN": [FEATURE_NAME_MAP.get(c, c) for c in sample.columns],
            "MeanAbsSHAP": mean_abs,
        }
    ).sort_values("MeanAbsSHAP", ascending=False)
    save_df(imp, results_dir / "tables" / "shap_feature_importance")

    top = imp.head(12).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=top["MeanAbsSHAP"],
            y=top["Feature_EN"],
            orientation="h",
        )
    )
    fig.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="Mean absolute SHAP value",
        yaxis_title="Feature",
        template="plotly_white",
        width=900,
        height=650,
    )
    save_plot(fig, results_dir / "figures" / "shap_feature_importance")


def write_manifest(results_dir: Path, payload: dict) -> None:
    with (results_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    results_dir = ensure_results_dir(Path(args.results_dir))

    train_df, external_df = load_data(args)
    features = select_features(train_df)
    X, y = prepare_xy(train_df, features)
    ex_X, ex_y = prepare_xy(external_df, features)
    models = build_models()

    save_df(pd.DataFrame({"Feature": features}), results_dir / "tables" / "selected_features")
    save_df(missingness_indicator_analysis(train_df), results_dir / "tables" / "missingness_indicator_development")
    save_df(missingness_indicator_analysis(external_df), results_dir / "tables" / "missingness_indicator_external")
    save_df(facility_missingness(train_df), results_dir / "tables" / "facility_missingness_development")
    save_df(facility_missingness(external_df), results_dir / "tables" / "facility_missingness_external")

    with (results_dir / "model_identity_note.txt").open("w", encoding="utf-8") as f:
        f.write(
            "The original final notebook labels the leading model as 'HybridXGBRF (Our Approach)', "
            "but the all_models dictionary appears to assign this label to an XGBClassifier. "
            "Confirm whether the manuscript should describe this as the final blended HybridXGBRF "
            "or as an XGBoost-style selected tree-based model before final submission.\n"
        )

    oof_probs_all = {}
    oof_true = y.to_numpy()
    if not args.skip_cv:
        cv_summary, cv_folds, oof_probs_all, oof_true, _ = internal_cv(
            models, X, y, n_boot=args.n_boot_fast, random_state=args.random_state
        )
        save_df(cv_summary, results_dir / "tables" / "table3_internal_cv_performance_with_ci")
        save_df(cv_folds, results_dir / "tables" / "internal_cv_fold_metrics")
        save_plot(plot_roc(oof_probs_all, oof_true, "Internal Cross-Validation ROC"), results_dir / "figures" / "internal_cv_roc")
        save_plot(
            plot_confusion_matrices(oof_probs_all, oof_true, "Internal CV Confusion Matrices"),
            results_dir / "figures" / "internal_cv_confusion_matrices",
        )
        save_plot(
            plot_calibration_with_histogram(oof_true, oof_probs_all, "Internal CV Calibration with Histogram"),
            results_dir / "figures" / "internal_cv_calibration_with_histogram",
        )
        dca_internal = decision_curve_df(oof_true, oof_probs_all, np.linspace(0.05, 0.95, 19))
        save_df(dca_internal, results_dir / "tables" / "decision_curve_internal_cv")
        save_plot(plot_decision_curve(dca_internal, "Decision Curve Analysis: Internal CV"), results_dir / "figures" / "decision_curve_internal_cv")

    ext_summary, ext_pretty, ext_probs, fitted_models = external_validation(
        models, X, y, ex_X, ex_y, n_boot=args.n_boot, random_state=args.random_state
    )
    save_df(ext_summary, results_dir / "tables" / "table4_external_validation_full_with_ci")
    save_df(ext_pretty, results_dir / "tables" / "table4_external_validation_paper_friendly")
    save_plot(plot_roc(ext_probs, ex_y.to_numpy(), "Temporal External Validation ROC"), results_dir / "figures" / "external_validation_roc")
    save_plot(
        plot_confusion_matrices(ext_probs, ex_y.to_numpy(), "External Validation Confusion Matrices"),
        results_dir / "figures" / "external_validation_confusion_matrices",
    )
    save_plot(
        plot_calibration_with_histogram(ex_y.to_numpy(), ext_probs, "External Validation Calibration with Histogram"),
        results_dir / "figures" / "external_validation_calibration_with_histogram",
    )

    main_model = "HybridXGBRF (Our Approach)"
    xgb_model = "XGBClassifier"
    if main_model in ext_probs:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
        save_df(
            threshold_table(ex_y.to_numpy(), ext_probs[main_model], thresholds),
            results_dir / "tables" / "threshold_tradeoff_external_hybridxgbrf",
        )
        save_df(
            risk_decile_table(ex_y.to_numpy(), ext_probs[main_model]),
            results_dir / "tables" / "risk_decile_calibration_external_hybridxgbrf",
        )
        save_df(
            pd.DataFrame([calibration_metrics(ex_y.to_numpy(), ext_probs[main_model])]),
            results_dir / "tables" / "calibration_metrics_external_hybridxgbrf",
        )
        save_df(
            calibration_metrics_with_ci(
                ex_y.to_numpy(),
                ext_probs[main_model],
                n_boot=args.n_boot_fast,
                random_state=args.random_state,
            ),
            results_dir / "tables" / "calibration_metrics_external_hybridxgbrf_with_ci",
        )
        save_df(
            subgroup_performance_with_ci(
                ex_y,
                ext_probs[main_model],
                external_df,
                n_boot=args.n_boot_fast,
                random_state=args.random_state,
            ),
            results_dir / "tables" / "table5_subgroup_performance_with_ci",
        )
        if not args.skip_shap:
            shap_outputs(fitted_models[main_model], ex_X.fillna(0), results_dir, args.random_state)

    dca_external = decision_curve_df(ex_y.to_numpy(), ext_probs, np.linspace(0.05, 0.95, 19))
    save_df(dca_external, results_dir / "tables" / "decision_curve_external_validation")
    save_plot(plot_decision_curve(dca_external, "Decision Curve Analysis: External Validation"), results_dir / "figures" / "decision_curve_external_validation")

    if main_model in ext_probs and xgb_model in ext_probs:
        save_df(
            paired_bootstrap_auc(
                ex_y.to_numpy(),
                ext_probs[main_model],
                ext_probs[xgb_model],
                main_model,
                xgb_model,
                n_boot=args.n_boot,
                random_state=args.random_state,
            ),
            results_dir / "tables" / "paired_bootstrap_auroc_external_hybrid_vs_xgb",
        )
    if oof_probs_all and main_model in oof_probs_all and xgb_model in oof_probs_all:
        save_df(
            paired_bootstrap_auc(
                oof_true,
                oof_probs_all[main_model],
                oof_probs_all[xgb_model],
                main_model,
                xgb_model,
                n_boot=args.n_boot_fast,
                random_state=args.random_state,
            ),
            results_dir / "tables" / "paired_bootstrap_auroc_internal_hybrid_vs_xgb",
        )

    supplemental_manifest = write_supplemental_revision_outputs(
        results_dir,
        Path(args.supplement_data_dir),
        included_df=pd.concat([train_df, external_df], ignore_index=True),
    )

    manifest = {
        "results_dir": str(results_dir),
        "training_n": int(len(train_df)),
        "external_n": int(len(external_df)),
        "features_n": int(len(features) - 1),
        "main_model_label": main_model,
        "tables_dir": str(results_dir / "tables"),
        "figures_dir": str(results_dir / "figures"),
        **supplemental_manifest,
    }
    write_manifest(results_dir, manifest)

    print(f"\nDone. Results written to: {results_dir}")
    print("Tables: ", results_dir / "tables")
    print("Figures:", results_dir / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
