from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Revision" / "20260523"
TMP = ROOT / "RESULTS_20260523_completion"
sys.path.insert(0, str(ROOT))

import revision_generate_results as rgr  # noqa: E402


def load_or_fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_cache = OUT / "training_data_1014_cached_for_completion.csv"
    ext_cache = OUT / "external_validation_1014_cached_for_completion.csv"
    if train_cache.exists() and ext_cache.exists():
        return pd.read_csv(train_cache), pd.read_csv(ext_cache)

    class Args:
        training = None
        external = None
        use_google_sheets = True

    train_df, ext_df = rgr.load_data(Args())
    train_df.to_csv(train_cache, index=False, encoding="utf-8-sig")
    ext_df.to_csv(ext_cache, index=False, encoding="utf-8-sig")
    return train_df, ext_df


def fit_external_predictions(train_df: pd.DataFrame, ext_df: pd.DataFrame):
    features = rgr.select_features(train_df)
    preprocessing = rgr.fit_preprocessing(train_df, features)
    X_train, y_train = rgr.prepare_xy(train_df, features, preprocessing)
    X_ext, y_ext = rgr.prepare_xy(ext_df, features, preprocessing)
    models = rgr.build_models()
    probs = {}
    fitted = {}
    for name, model in models.items():
        estimator = clone(model)
        estimator.fit(X_train, y_train)
        probs[name] = rgr.get_positive_proba(estimator, X_ext)
        fitted[name] = estimator
    return features, preprocessing, X_train, y_train, X_ext, y_ext, probs, fitted


def write_threshold_06_09(y_ext: pd.Series, probs: dict[str, np.ndarray]) -> None:
    p = probs["HybridXGBRF (Our Approach)"]
    full = rgr.threshold_table(y_ext.to_numpy(), p, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    high = full[full["Threshold"].isin([0.6, 0.7, 0.8, 0.9])].copy()
    with pd.ExcelWriter(OUT / "threshold_tradeoff_external_hybridxgbrf_0p1_to_0p9.xlsx") as writer:
        full.to_excel(writer, index=False, sheet_name="Thresholds 0.1-0.9")
        high.to_excel(writer, index=False, sheet_name="Thresholds 0.6-0.9")
    full.to_csv(OUT / "threshold_tradeoff_external_hybridxgbrf_0p1_to_0p9.csv", index=False, encoding="utf-8-sig")


def calibration_all_models(train_df, features, X_train, y_train, X_ext, y_ext, ext_probs) -> None:
    models = rgr.build_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    internal_rows = []
    for name, model in models.items():
        oof = np.zeros(len(y_train))
        for tr, te in skf.split(X_train, y_train):
            fold_pre = rgr.fit_preprocessing(train_df.iloc[tr], features)
            X_tr, y_tr = rgr.prepare_xy(train_df.iloc[tr], features, fold_pre)
            X_te, _ = rgr.prepare_xy(train_df.iloc[te], features, fold_pre)
            est = clone(model)
            est.fit(X_tr, y_tr)
            oof[te] = rgr.get_positive_proba(est, X_te)
        m = rgr.calibration_metrics(y_train.to_numpy(), oof)
        internal_rows.append({"Cohort": "Internal CV", "Model": name, **m})

    external_rows = []
    for name, p in ext_probs.items():
        m = rgr.calibration_metrics(y_ext.to_numpy(), p)
        external_rows.append({"Cohort": "Temporal external validation", "Model": name, **m})

    out = pd.DataFrame(internal_rows + external_rows)
    out["Formatted Intercept"] = out["Calibration Intercept"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    out["Formatted Slope"] = out["Calibration Slope"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    out["Formatted O/E"] = out["Observed/Expected"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    out["Formatted Brier"] = out["Brier"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    with pd.ExcelWriter(OUT / "calibration_metrics_all_models_internal_external.xlsx") as writer:
        out.to_excel(writer, index=False, sheet_name="All models")
        out[out["Cohort"] == "Internal CV"].to_excel(writer, index=False, sheet_name="Internal CV")
        out[out["Cohort"] != "Internal CV"].to_excel(writer, index=False, sheet_name="External")


def missing_indicator_regression(train_df, ext_df) -> None:
    # Best practical reviewer-facing specification:
    # death outcome ~ missingness indicators for SHAP-highlighted documentation domains
    # adjusted for age, sex, DNR, and recent hospitalization count.
    targets = {
        "ADL_second_missing": "ADL_last_score",
        "Body_weight_missing": "BW_first",
        "Facility_id_missing": "dbname",
    }
    adjusters = ["預估年齡", "性別_is_male", "DNR_flag", "六個月內住院次數"]
    rows = []
    for cohort, df in [("Development", train_df), ("Temporal external validation", ext_df)]:
        work = pd.DataFrame(index=df.index)
        y = pd.to_numeric(df["死亡標記"], errors="coerce").astype(int)
        for new, col in targets.items():
            work[new] = df[col].isna().astype(int)
        for col in adjusters:
            work[col] = pd.to_numeric(df[col], errors="coerce")
            work[col] = work[col].fillna(work[col].median())
        for var in targets:
            X = work[[var] + adjusters]
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            coef = float(model.coef_[0][0])
            or_ = float(np.exp(coef))
            rows.append(
                {
                    "Cohort": cohort,
                    "Missingness indicator": var,
                    "Source variable": targets[var],
                    "N": len(df),
                    "Missing N": int(work[var].sum()),
                    "Missing percent": float(work[var].mean()),
                    "Adjusted odds ratio for death": or_,
                    "Log-odds coefficient": coef,
                    "Adjustment variables": ", ".join(adjusters),
                    "Outcome": "6-month death indicator",
                }
            )
    out = pd.DataFrame(rows)
    out["Formatted OR"] = out["Adjusted odds ratio for death"].map(lambda x: f"{x:.3f}")
    with pd.ExcelWriter(OUT / "missingness_indicator_key_features_regression.xlsx") as writer:
        out.to_excel(writer, index=False, sheet_name="Logistic regression")


def multiple_imputation_decision_table() -> None:
    rows = [
        {
            "Item": "Decision",
            "Conclusion": "Do not add a separate multiple-imputation model as the primary revision output.",
            "Rationale": "The manuscript's prespecified strategy treats binary/count documentation absence as 0 and continuous/scale missingness as development-mean-equivalent after standardization. This matches routine-care documentation patterns and avoids mixing structurally absent binary flags with stochastic imputation.",
        },
        {
            "Item": "Sensitivity support",
            "Conclusion": "Use missingness indicators and facility-level missingness analyses instead.",
            "Rationale": "These directly address reviewer concern about documentation bias and whether missingness patterns differ by outcome/facility.",
        },
        {
            "Item": "If requested later",
            "Conclusion": "Multiple imputation can be a future sensitivity analysis only if the target estimand and imputation model are prespecified.",
            "Rationale": "A post-hoc MI result would require full refitting and synchronized manuscript updates.",
        },
    ]
    pd.DataFrame(rows).to_excel(OUT / "multiple_imputation_decision_and_rationale.xlsx", index=False)


def main() -> None:
    train_df, ext_df = load_or_fetch_data()
    features, preprocessing, X_train, y_train, X_ext, y_ext, probs, fitted = fit_external_predictions(train_df, ext_df)
    write_threshold_06_09(y_ext, probs)
    calibration_all_models(train_df, features, X_train, y_train, X_ext, y_ext, probs)
    missing_indicator_regression(train_df, ext_df)
    multiple_imputation_decision_table()
    print("completed remaining revision tables")


if __name__ == "__main__":
    main()
