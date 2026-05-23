"""
Survival sensitivity analyses for the JUBO revision.

This script uses the currently available columns:
    - 觀察天數: follow-up/observation days
    - 死亡標記: event indicator

It creates pragmatic time-to-event sensitivity outputs under a 180-day
administrative horizon. Residents without death before 180 days are censored
at min(observation days, 180).

Outputs are written under:
    ./RESULTS/tables
    ./RESULTS/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import revision_generate_results as base

try:
    from statsmodels.duration.hazard_regression import PHReg
except Exception as exc:  # pragma: no cover
    raise SystemExit("statsmodels PHReg is required for this script.") from exc


WORKDIR = Path(__file__).resolve().parent
RESULTS = WORKDIR / "RESULTS"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
TIME_COL = "觀察天數"
EVENT_COL = "死亡標記"
HORIZON = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-google-sheets", action="store_true", default=True)
    parser.add_argument("--training", default=None)
    parser.add_argument("--external", default=None)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path_base: Path) -> None:
    df.to_csv(path_base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    df.to_excel(path_base.with_suffix(".xlsx"), index=False)


def save_fig(fig: go.Figure, path_base: Path) -> None:
    fig.write_html(str(path_base.with_suffix(".html")))
    try:
        fig.write_image(str(path_base.with_suffix(".png")), scale=3)
    except Exception:
        pass


def survival_outcome(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing required time column: {TIME_COL}")
    if EVENT_COL not in df.columns:
        raise ValueError(f"Missing required event column: {EVENT_COL}")
    raw_time = pd.to_numeric(df[TIME_COL], errors="coerce").fillna(horizon).clip(lower=0).to_numpy(float)
    raw_event = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype(int).to_numpy()
    event = ((raw_event == 1) & (raw_time <= horizon)).astype(int)
    time = np.minimum(raw_time, horizon)
    time = np.where(time <= 0, 0.5, time)
    return time, event


def fit_ml_risk_model(X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, random_state: int) -> np.ndarray:
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        random_state=random_state,
        eval_metric="logloss",
        subsample=1.0,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1]


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    tied = 0.0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        mask = time[i] < time
        if not np.any(mask):
            continue
        comparable += mask.sum()
        concordant += np.sum(risk[i] > risk[mask])
        tied += np.sum(risk[i] == risk[mask])
    if comparable == 0:
        return np.nan
    return float((concordant + 0.5 * tied) / comparable)


def cumulative_dynamic_auc(time: np.ndarray, event: np.ndarray, risk: np.ndarray, months: list[int]) -> pd.DataFrame:
    rows = []
    for month in months:
        t = month * 30
        case = (event == 1) & (time <= t)
        control = time > t
        keep = case | control
        if case.sum() == 0 or control.sum() == 0:
            auc_val = np.nan
        else:
            auc_val = roc_auc_score(case[keep].astype(int), risk[keep])
        rows.append(
            {
                "Month": month,
                "Day": t,
                "Cases by time t": int(case.sum()),
                "Controls event-free beyond t": int(control.sum()),
                "Excluded censored before t": int((~keep).sum()),
                "Cumulative/Dynamic AUC": auc_val,
            }
        )
    return pd.DataFrame(rows)


def km_curve(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    event_times = np.sort(np.unique(time[event == 1]))
    surv = 1.0
    rows = [{"Time": 0.0, "Survival": 1.0, "At risk": int(len(time)), "Events": 0}]
    for t in event_times:
        at_risk = np.sum(time >= t)
        d = np.sum((time == t) & (event == 1))
        if at_risk > 0:
            surv *= 1 - d / at_risk
        rows.append({"Time": float(t), "Survival": float(surv), "At risk": int(at_risk), "Events": int(d)})
    return pd.DataFrame(rows)


def km_by_risk_group(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = ["Low predicted risk", "Medium predicted risk", "High predicted risk"]
    group = pd.qcut(risk, q=3, labels=labels, duplicates="drop")
    all_rows = []
    summary = []
    for label in group.categories:
        mask = np.asarray(group == label)
        km = km_curve(time[mask], event[mask])
        km["Risk group"] = str(label)
        all_rows.append(km)
        summary.append(
            {
                "Risk group": str(label),
                "N": int(mask.sum()),
                "Deaths within horizon": int(event[mask].sum()),
                "Event percent": float(event[mask].mean()),
                "Median predicted risk": float(np.median(risk[mask])),
            }
        )
    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(summary)


def plot_km(km: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for group, sub in km.groupby("Risk group", sort=False):
        fig.add_trace(
            go.Scatter(
                x=sub["Time"],
                y=sub["Survival"],
                mode="lines",
                line_shape="hv",
                name=group,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Days since LTCF admission",
        yaxis_title="Survival probability",
        template="plotly_white",
        width=900,
        height=600,
        yaxis=dict(range=[0, 1.01]),
    )
    return fig


def logrank_test(time: np.ndarray, event: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    # Global k-sample log-rank statistic.
    groups = pd.Series(groups)
    cats = list(groups.cat.categories)
    event_times = np.sort(np.unique(time[event == 1]))
    observed = np.zeros(len(cats))
    expected = np.zeros(len(cats))
    var = np.zeros((len(cats), len(cats)))
    for t in event_times:
        at_risk_all = time >= t
        d_all = ((time == t) & (event == 1)).sum()
        n_all = at_risk_all.sum()
        if n_all <= 1 or d_all == 0:
            continue
        n_g = np.array([np.sum(at_risk_all & (groups == c)) for c in cats], dtype=float)
        d_g = np.array([np.sum((time == t) & (event == 1) & (groups == c)) for c in cats], dtype=float)
        observed += d_g
        expected += d_all * n_g / n_all
        factor = d_all * (n_all - d_all) / (n_all - 1)
        for i in range(len(cats)):
            for j in range(len(cats)):
                if i == j:
                    var[i, j] += factor * (n_g[i] / n_all) * (1 - n_g[i] / n_all)
                else:
                    var[i, j] -= factor * (n_g[i] / n_all) * (n_g[j] / n_all)
    diff = observed - expected
    # Drop last group because covariance matrix is singular.
    v_reduced = var[:-1, :-1]
    d_reduced = diff[:-1]
    try:
        chi2 = float(d_reduced.T @ np.linalg.pinv(v_reduced) @ d_reduced)
        from scipy.stats import chi2 as chi2_dist

        p = float(chi2_dist.sf(chi2, len(cats) - 1))
    except Exception:
        chi2, p = np.nan, np.nan
    return pd.DataFrame(
        [
            {
                "Groups": ", ".join(map(str, cats)),
                "Chi-square": chi2,
                "df": len(cats) - 1,
                "P value": p,
            }
        ]
    )


def cox_on_risk_score(time: np.ndarray, event: np.ndarray, risk: np.ndarray, label: str) -> pd.DataFrame:
    df = pd.DataFrame({"time": time, "event": event, "risk_per_0_1": risk * 10})
    model = PHReg.from_formula("time ~ risk_per_0_1", status="event", data=df)
    res = model.fit(disp=False)
    beta = float(res.params[0])
    se = float(res.bse[0])
    hr = float(np.exp(beta))
    lo = float(np.exp(beta - 1.96 * se))
    hi = float(np.exp(beta + 1.96 * se))
    p = float(res.pvalues[0])
    return pd.DataFrame(
        [
            {
                "Analysis": label,
                "Predictor": "ML predicted risk per 0.10 increase",
                "Hazard Ratio": hr,
                "95% CI Low": lo,
                "95% CI High": hi,
                "P value": p,
                "N": int(len(time)),
                "Events": int(event.sum()),
            }
        ]
    )


def multivariable_cox_baseline(time: np.ndarray, event: np.ndarray, X: pd.DataFrame, max_features: int = 12) -> pd.DataFrame:
    # Keep features with highest univariate absolute correlation with event for a stable pragmatic Cox baseline.
    X_num = X.copy().replace([np.inf, -np.inf], np.nan)
    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_num), columns=X.columns)
    scores = []
    for col in X_imp.columns:
        try:
            scores.append((col, abs(np.corrcoef(X_imp[col], event)[0, 1])))
        except Exception:
            scores.append((col, 0))
    top_cols = [c for c, _ in sorted(scores, key=lambda x: -x[1])[:max_features]]
    X_std = pd.DataFrame(StandardScaler().fit_transform(X_imp[top_cols]), columns=[f"x{i}" for i in range(len(top_cols))])
    df = pd.concat([pd.DataFrame({"time": time, "event": event}), X_std], axis=1)
    formula = "time ~ " + " + ".join(X_std.columns)
    model = PHReg.from_formula(formula, status="event", data=df)
    res = model.fit(disp=False)
    rows = []
    for i, original_col in enumerate(top_cols):
        beta = float(res.params[i])
        se = float(res.bse[i])
        rows.append(
            {
                "Feature": original_col,
                "Feature_EN": base.FEATURE_NAME_MAP.get(original_col, original_col),
                "Hazard Ratio per 1 SD": float(np.exp(beta)),
                "95% CI Low": float(np.exp(beta - 1.96 * se)),
                "95% CI High": float(np.exp(beta + 1.96 * se)),
                "P value": float(res.pvalues[i]),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    ensure_dirs()
    train_df, ext_df = base.load_data(args)
    features = base.select_features(train_df)
    preprocessing = base.fit_preprocessing(train_df, features)
    X_train, y_train = base.prepare_xy(train_df, features, preprocessing)
    X_ext, y_ext = base.prepare_xy(ext_df, features, preprocessing)

    train_time, train_event = survival_outcome(train_df, args.horizon)
    ext_time, ext_event = survival_outcome(ext_df, args.horizon)

    ext_risk = fit_ml_risk_model(X_train, y_train, X_ext, args.random_state)
    train_risk = fit_ml_risk_model(X_train, y_train, X_train, args.random_state)

    months = [1, 2, 3, 4, 5, 6]
    auc_ext = cumulative_dynamic_auc(ext_time, ext_event, ext_risk, months)
    auc_train = cumulative_dynamic_auc(train_time, train_event, train_risk, months)
    auc_ext["Cohort"] = "Temporal external validation"
    auc_train["Cohort"] = "Development"
    save_df(pd.concat([auc_train, auc_ext], ignore_index=True), TABLES / "survival_time_dependent_auc")

    cindex = pd.DataFrame(
        [
            {
                "Cohort": "Development",
                "Harrell C-index": harrell_c_index(train_time, train_event, train_risk),
                "N": len(train_time),
                "Events": int(train_event.sum()),
            },
            {
                "Cohort": "Temporal external validation",
                "Harrell C-index": harrell_c_index(ext_time, ext_event, ext_risk),
                "N": len(ext_time),
                "Events": int(ext_event.sum()),
            },
        ]
    )
    save_df(cindex, TABLES / "survival_c_index")

    cox_risk = pd.concat(
        [
            cox_on_risk_score(train_time, train_event, train_risk, "Development"),
            cox_on_risk_score(ext_time, ext_event, ext_risk, "Temporal external validation"),
        ],
        ignore_index=True,
    )
    save_df(cox_risk, TABLES / "survival_cox_ml_risk_score")

    try:
        cox_base = multivariable_cox_baseline(train_time, train_event, X_train)
        save_df(cox_base, TABLES / "survival_cox_baseline_development_top_features")
    except Exception as exc:
        pd.DataFrame([{"Note": f"Multivariable Cox baseline failed: {exc}"}]).to_excel(
            TABLES / "survival_cox_baseline_development_top_features.xlsx", index=False
        )

    km_ext, km_summary = km_by_risk_group(ext_time, ext_event, ext_risk)
    save_df(km_ext, TABLES / "survival_km_curve_external_by_risk_group")
    save_df(km_summary, TABLES / "survival_km_external_risk_group_summary")
    save_fig(plot_km(km_ext, "180-day Kaplan-Meier Curves by Predicted Risk Group"), FIGURES / "survival_km_external_by_risk_group")

    groups = pd.qcut(ext_risk, q=3, labels=["Low predicted risk", "Medium predicted risk", "High predicted risk"], duplicates="drop")
    save_df(logrank_test(ext_time, ext_event, groups), TABLES / "survival_logrank_external_risk_groups")

    fig_auc = go.Figure()
    for cohort, sub in pd.concat([auc_train, auc_ext]).groupby("Cohort", sort=False):
        fig_auc.add_trace(
            go.Scatter(x=sub["Month"], y=sub["Cumulative/Dynamic AUC"], mode="lines+markers", name=cohort)
        )
    fig_auc.update_layout(
        title="Cumulative/Dynamic AUC by Prediction Horizon",
        xaxis_title="Months since LTCF admission",
        yaxis_title="AUC",
        yaxis=dict(range=[0.5, 1.0]),
        template="plotly_white",
        width=900,
        height=600,
    )
    save_fig(fig_auc, FIGURES / "survival_time_dependent_auc")

    note = (
        "Survival sensitivity analysis used available observation days and death indicators under a 180-day "
        "administrative horizon. Residents without death before 180 days were censored at min(observation days, 180). "
        "Because post-discharge death ascertainment and censoring mechanisms may be incomplete in the analytic "
        "dataset, these analyses should be presented as sensitivity analyses rather than replacing the primary "
        "binary 6-month mortality model.\n"
    )
    (RESULTS / "survival_analysis_note.txt").write_text(note, encoding="utf-8")
    print("Survival outputs written to RESULTS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
