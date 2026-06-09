from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2_contingency, norm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Revision" / "20260609"

TRAINING_CACHE = ROOT / "Revision" / "20260523" / "training_data_1014_cached_for_completion.csv"
EXTERNAL_CACHE = ROOT / "Revision" / "20260523" / "external_validation_1014_cached_for_completion.csv"
FACILITY_ROSTER = ROOT / "DATA" / "area_size.xlsx"
SHAP_FEATURES = ROOT / "RESULTS" / "tables" / "shap_feature_importance.xlsx"
INTERNAL_CALIBRATION = OUT / "calibration_metrics_internal_hybridxgbrf_with_ci.csv"
INTERNAL_RISK_DECILE = OUT / "risk_decile_calibration_internal_hybridxgbrf.csv"

TRAINING_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1qljyp9lq3QsZ7O2O7FQxm7taEWQi3F3bZgNMcQ7NJeE/export?format=xlsx"
)
EXTERNAL_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1NFAhP8NUVsxzEq55siFA0yHvnXY5GWqiKGSOKC4y1Qg/export?format=xlsx"
)

OUTCOME_COL = "死亡標記"
FACILITY_COL = "dbname"


def numeric_clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(",", "").str.strip(), errors="coerce"))


def load_development_cohort_raw() -> pd.DataFrame:
    return pd.read_excel(TRAINING_SHEET_URL, sheet_name=0)


def load_external_cohort_raw() -> pd.DataFrame:
    return pd.read_excel(EXTERNAL_SHEET_URL, sheet_name=0)


def load_development_cache() -> pd.DataFrame:
    return pd.read_csv(TRAINING_CACHE)


def load_external_cache() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL_CACHE)


def load_facility_roster() -> pd.DataFrame:
    roster = pd.read_excel(FACILITY_ROSTER, sheet_name="訓練資料_機構大小")
    keep = ["dbname", "核定床數", "機構大小層級", "區域"]
    return roster[keep].dropna(subset=["dbname"]).copy()


def load_model_feature_list() -> pd.DataFrame:
    features = pd.read_excel(SHAP_FEATURES)
    features = features[["Feature", "Feature_EN", "MeanAbsSHAP"]].copy()
    features["Included in all-feature missingness"] = True
    return features


def feature_columns(df: pd.DataFrame, model_features: pd.DataFrame) -> list[str]:
    features = model_features["Feature"].tolist()
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Model features missing from Development cohort source: {missing}")
    return features


def validate_raw_matches_cache(raw: pd.DataFrame, cache: pd.DataFrame) -> tuple[bool, list[tuple[str, int]]]:
    common = [c for c in cache.columns if c in raw.columns]
    raw_num = numeric_clean(raw[common])
    mismatches: list[tuple[str, int]] = []
    for col in common:
        a = raw_num[col].reset_index(drop=True)
        b = pd.to_numeric(cache[col], errors="coerce").reset_index(drop=True)
        same = (a.isna() & b.isna()) | np.isclose(a.fillna(0), b.fillna(0), rtol=0, atol=1e-8)
        if not bool(same.all()):
            mismatches.append((col, int((~same).sum())))
    return len(mismatches) == 0, mismatches


def add_roster(raw: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    roster2 = roster.drop_duplicates(FACILITY_COL).copy()
    return raw.merge(roster2, on=FACILITY_COL, how="left")


def summarize_facility_strata(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    y_all = pd.to_numeric(df[OUTCOME_COL], errors="coerce").astype(int)

    def summarize(label: str, mask: pd.Series) -> dict[str, object]:
        sub = df.loc[mask].copy()
        y = y_all.loc[sub.index]
        missing = sub[features].isna()

        def miss(mask2: pd.Series | np.ndarray | None = None) -> float:
            ss = missing.loc[mask2] if mask2 is not None else missing
            return float(ss.mean().mean()) if len(ss) else np.nan

        return {
            "Variable": label,
            "N facilities": int(sub[FACILITY_COL].nunique(dropna=True)),
            "N residents": int(len(sub)),
            "ALL Feature Missing Percent (Overall missing percent)": miss(),
            "ALL Feature Missing Percent/ Dead cohort": miss(y == 1),
            "ALL Feature Missing Percent/ Alive cohort": miss(y == 0),
            "Death rate": float(y.mean()) if len(y) else np.nan,
            "Status/Notes": (
                "Computed from confirmed resident-level dbname in the original training_data_1014 Google Sheet, "
                "merged to DATA/area_size.xlsx facility roster."
            ),
        }

    rows = [
        summarize("facility_size <50", df["核定床數"] < 50),
        summarize("facility_size 50-150", df["核定床數"].between(50, 150, inclusive="both")),
        summarize("facility_size >150", df["核定床數"] > 150),
        summarize("Institutional regions_North", df["區域"] == "北部"),
        summarize("Institutional regions_Central", df["區域"] == "中部"),
        summarize("Institutional regions_South", df["區域"] == "南部"),
        summarize("Institutional regions_East", df["區域"] == "東部"),
    ]
    return pd.DataFrame(rows)


def dbname_missingness_chi_square(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = df[features].isna()
    detail = (
        pd.DataFrame(
            {
                FACILITY_COL: df[FACILITY_COL],
                "Missing feature cells": missing.sum(axis=1),
                "Observed feature cells": len(features) - missing.sum(axis=1),
            }
        )
        .groupby(FACILITY_COL, dropna=False)
        .sum()
        .reset_index()
    )
    detail["Total feature cells"] = detail["Missing feature cells"] + detail["Observed feature cells"]
    detail["Overall missing percent"] = detail["Missing feature cells"] / detail["Total feature cells"]
    detail["N residents/rows"] = df.groupby(FACILITY_COL, dropna=False).size().to_numpy()

    contingency = detail[["Missing feature cells", "Observed feature cells"]].to_numpy()
    chi2, p_value, dof, _ = chi2_contingency(contingency)
    total = contingency.sum()
    cramers_v = float(np.sqrt(chi2 / (total * (min(contingency.shape) - 1))))
    chi_square = pd.DataFrame(
        [
            {
                "Test": "dbname x all-feature missing/observed cells",
                "Chi-square statistic": chi2,
                "df": dof,
                "P value": p_value,
                "Table-ready P value": "<0.001" if p_value < 0.001 else f"{p_value:.3f}",
                "Cramer's V": cramers_v,
                "N dbname groups": int(detail[FACILITY_COL].nunique(dropna=False)),
                "Total feature cells": int(total),
                "Missing feature cells": int(contingency[:, 0].sum()),
                "Observed feature cells": int(contingency[:, 1].sum()),
                "Important note": (
                    "Computed using confirmed resident-level dbname from the original training_data_1014 Google Sheet. "
                    "The local numeric cache has dbname missing because text identifiers were coerced to NaN."
                ),
            }
        ]
    )
    return detail, chi_square


def h01_diagnostic(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = df[features].isna()
    detail = (
        pd.DataFrame(
            {
                "H01_NUM": df["H01_NUM"],
                "Missing feature cells": missing.sum(axis=1),
                "Observed feature cells": len(features) - missing.sum(axis=1),
            }
        )
        .groupby("H01_NUM", dropna=False)
        .sum()
        .reset_index()
    )
    detail["Total feature cells"] = detail["Missing feature cells"] + detail["Observed feature cells"]
    detail["Overall missing percent"] = detail["Missing feature cells"] / detail["Total feature cells"]
    detail["N residents/rows"] = df.groupby("H01_NUM", dropna=False).size().to_numpy()

    contingency = detail[["Missing feature cells", "Observed feature cells"]].to_numpy()
    chi2, p_value, dof, _ = chi2_contingency(contingency)
    total = contingency.sum()
    cramers_v = float(np.sqrt(chi2 / (total * (min(contingency.shape) - 1))))
    summary = pd.DataFrame(
        [
            {
                "Test": "H01_NUM x all-feature missing/observed cells",
                "Chi-square statistic": chi2,
                "df": dof,
                "P value": p_value,
                "Table-ready P value": "<0.001" if p_value < 0.001 else f"{p_value:.3f}",
                "Cramer's V": cramers_v,
                "N H01_NUM groups": int(detail["H01_NUM"].nunique(dropna=False)),
                "Total feature cells": int(total),
                "Missing feature cells": int(contingency[:, 0].sum()),
                "Observed feature cells": int(contingency[:, 1].sum()),
                "Important limitation": (
                    "Diagnostic only. H01_NUM has 2,057 groups in the Development cohort and is not the LTCF count; "
                    "confirmed facility-level analyses should use dbname."
                ),
            }
        ]
    )
    return detail, summary


def overall_development_missingness(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    y = pd.to_numeric(df[OUTCOME_COL], errors="coerce").astype(int)

    def feature_missing_percent(mask: pd.Series | np.ndarray | None = None) -> float:
        subset = df.loc[mask, features] if mask is not None else df[features]
        return float(subset.isna().mean().mean())

    return pd.DataFrame(
        [
            {
                "Cohort/Subset": "Development cohort overall",
                "N residents": len(df),
                "Deaths": int(y.sum()),
                "All model features counted": len(features),
                "ALL Feature Missing Percent (Overall missing percent)": feature_missing_percent(),
                "ALL Feature Missing Percent/ Dead cohort": feature_missing_percent(y == 1),
                "ALL Feature Missing Percent/ Alive cohort": feature_missing_percent(y == 0),
                "Death rate": float(y.mean()),
                "Notes": (
                    "Overall Development cohort values across the 29 SHAP/model features listed in the Feature list sheet."
                ),
            }
        ]
    )


def fit_adjusted_logistic_pvalue(y: pd.Series, X: pd.DataFrame, target: str) -> dict[str, float | str]:
    y_arr = pd.to_numeric(y, errors="coerce").astype(float).to_numpy()
    X_num = X.apply(pd.to_numeric, errors="coerce").copy()
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    if X_num[target].nunique(dropna=False) < 2:
        return {
            "Adjusted odds ratio for death": np.nan,
            "Log-odds coefficient": np.nan,
            "Standard error": np.nan,
            "Wald z": np.nan,
            "P value": np.nan,
            "Model status": "Not estimable because the missingness indicator has no variation.",
        }
    for col in X_num.columns:
        if col == target:
            continue
        sd = X_num[col].std(ddof=0)
        if pd.notna(sd) and sd > 0:
            X_num[col] = (X_num[col] - X_num[col].mean()) / sd

    design = np.column_stack([np.ones(len(X_num)), X_num.to_numpy(dtype=float)])
    target_index = list(X_num.columns).index(target) + 1

    def nll(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(np.sum(np.logaddexp(0, eta) - y_arr * eta))

    def grad(beta: np.ndarray) -> np.ndarray:
        eta = design @ beta
        p = 1 / (1 + np.exp(-eta))
        return design.T @ (p - y_arr)

    result = minimize(nll, np.zeros(design.shape[1]), jac=grad, method="L-BFGS-B")
    beta = result.x
    eta = design @ beta
    p = 1 / (1 + np.exp(-eta))
    weights = p * (1 - p)
    hessian = design.T @ (design * weights[:, None])
    cov = np.linalg.pinv(hessian)
    se = float(np.sqrt(cov[target_index, target_index]))
    coef = float(beta[target_index])
    z = coef / se if se > 0 else np.nan
    p_value = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {
        "Adjusted odds ratio for death": float(np.exp(coef)),
        "Log-odds coefficient": coef,
        "Standard error": se,
        "Wald z": z,
        "P value": p_value,
        "Model status": "Converged" if result.success else f"Optimization warning: {result.message}",
    }


def missingness_indicator_key_features_regression(train_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "ADL_second_missing": "ADL_last_score",
        "Body_weight_missing": "BW_first",
        "Consciousness_total_max_missing": "意識總分Max",
        "Facility_id_missing": "dbname",
    }
    adjusters = ["預估年齡", "性別_is_male", "DNR_flag", "六個月內住院次數"]
    rows: list[dict[str, object]] = []
    for cohort, df in [("Development", train_df), ("Temporal external validation", external_df)]:
        y = pd.to_numeric(df[OUTCOME_COL], errors="coerce").astype(int)
        work = pd.DataFrame(index=df.index)
        for indicator, source in targets.items():
            work[indicator] = df[source].isna().astype(int)
        for col in adjusters:
            work[col] = pd.to_numeric(df[col], errors="coerce")

        for indicator, source in targets.items():
            X = work[[indicator] + adjusters]
            stats = fit_adjusted_logistic_pvalue(y, X, indicator)
            rows.append(
                {
                    "Cohort": cohort,
                    "Missingness indicator": indicator,
                    "Source variable": source,
                    "N": len(df),
                    "Missing N": int(work[indicator].sum()),
                    "Missing percent": float(work[indicator].mean()),
                    **stats,
                    "Adjustment variables": ", ".join(adjusters),
                    "Outcome": "6-month death indicator",
                }
            )
    out = pd.DataFrame(rows)
    out["Formatted OR"] = out["Adjusted odds ratio for death"].map(
        lambda x: "Not estimable" if pd.isna(x) else f"{x:.3f}"
    )
    out["Formatted P value"] = out["P value"].map(
        lambda x: "Not estimable" if pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}")
    )
    return out


def data_availability_audit(
    raw: pd.DataFrame,
    cache: pd.DataFrame,
    external_raw: pd.DataFrame,
    external_cache: pd.DataFrame,
    roster: pd.DataFrame,
    features: list[str],
    cache_match: bool,
    cache_mismatches: list[tuple[str, int]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Item": "Development raw source", "Finding": TRAINING_SHEET_URL},
            {"Item": "Development raw rows", "Finding": len(raw)},
            {"Item": "Development raw usable dbname rows", "Finding": int(raw[FACILITY_COL].notna().sum())},
            {"Item": "Development raw unique dbname", "Finding": int(raw[FACILITY_COL].nunique(dropna=True))},
            {"Item": "Development raw unique H01_NUM", "Finding": int(raw["H01_NUM"].nunique(dropna=False))},
            {"Item": "Development cache source", "Finding": str(TRAINING_CACHE.relative_to(ROOT))},
            {"Item": "Development cache usable dbname rows", "Finding": int(cache[FACILITY_COL].notna().sum())},
            {"Item": "Raw numeric-clean result matches cache", "Finding": cache_match},
            {"Item": "Raw/cache mismatched columns", "Finding": cache_mismatches if cache_mismatches else "None"},
            {"Item": "External raw rows", "Finding": len(external_raw)},
            {"Item": "External raw usable dbname rows", "Finding": int(external_raw[FACILITY_COL].notna().sum())},
            {"Item": "External raw unique dbname", "Finding": int(external_raw[FACILITY_COL].nunique(dropna=True))},
            {"Item": "External raw unique H01_NUM", "Finding": int(external_raw["H01_NUM"].nunique(dropna=False))},
            {"Item": "External cache usable dbname rows", "Finding": int(external_cache[FACILITY_COL].notna().sum())},
            {"Item": "Facility roster source", "Finding": str(FACILITY_ROSTER.relative_to(ROOT))},
            {"Item": "Facility roster rows", "Finding": len(roster)},
            {"Item": "Facility roster unique dbname", "Finding": int(roster[FACILITY_COL].nunique(dropna=True))},
            {"Item": "All prediction features counted", "Finding": len(features)},
            {
                "Item": "Conclusion",
                "Finding": (
                    "Confirmed dbname is present in the original raw Google Sheets. The local numeric cache lost dbname "
                    "because text identifiers were coerced to NaN. Facility-level analyses in this workbook use the raw "
                    "confirmed dbname and match the cache row order after numeric cleaning."
                ),
            },
        ]
    )


def write_notes(dev: pd.DataFrame, features: list[str], chi_square: pd.DataFrame) -> None:
    note = OUT / "0609_development_facility_missingness_form_notes.md"
    cramers_v = chi_square.loc[0, "Cramer's V"]
    note.write_text(
        "\n".join(
            [
                "# 0609 Development Facility Missingness Form Notes",
                "",
                "## What Was Requested",
                "",
                "The PDF requested a Development cohort table stratified by facility size and institutional region, including N facilities, N residents, all-feature missing percent overall, all-feature missing percent among dead residents, all-feature missing percent among alive residents, and death rate. It also requested a chi-square test for institution ID by overall all-feature missing percent.",
                "",
                "## Key Correction",
                "",
                "The local Development cohort cache (`Revision/20260523/training_data_1014_cached_for_completion.csv`) has `dbname` missing for all 23,901 rows because the original notebook/script applied numeric coercion to the full Google Sheet. The original `training_data_1014` Google Sheet retains confirmed resident-level `dbname`. After numeric cleaning, the raw Google Sheet matches the local cache row-for-row, so the confirmed `dbname` can be safely reattached by row order.",
                "",
                "Confirmed Development cohort counts from the raw sheet:",
                "",
                "- Development residents: 23,901",
                "- Development deaths: 5,272",
                "- Confirmed unique `dbname`: 493",
                "- Unique `H01_NUM`: 2,057 (`H01_NUM` is not the LTCF count)",
                "- Facility roster rows in `DATA/area_size.xlsx`: 493",
                "",
                "## Files Produced",
                "",
                "- `0609_development_facility_missingness_form.xlsx`",
                "- `0609_development_facility_missingness_form.csv`",
                "- `0609_dbname_missingness_chi_square.csv`",
                "- `0609_h01num_missingness_chi_square.csv` (diagnostic only)",
                "- `0609_missingness_indicator_key_features_regression_with_p.xlsx`",
                "- `0609_missingness_indicator_key_features_regression_with_p.csv`",
                "",
                "## All Features",
                "",
                "The all-feature missingness calculation uses the 29 model predictor features listed in `RESULTS/tables/shap_feature_importance.xlsx`, matching the feature list shown in the 0609 PDF. `死亡標記` from `selected_features.xlsx` is the outcome and is not counted as a predictor feature.",
                "",
                "## Facility-Level Table",
                "",
                "`0609_development_facility_missingness_form.csv` is now the confirmed facility-level table. It uses raw resident-level `dbname` merged to `DATA/area_size.xlsx`, so N residents, dead/alive all-feature missingness, and death rate are estimable by facility size and institutional region.",
                "",
                "## Chi-Square Test",
                "",
                "`0609_dbname_missingness_chi_square.csv` is the confirmed facility-level chi-square test: `dbname x all-feature missing/observed cells`. The older `H01_NUM` result is retained only as a diagnostic because `H01_NUM` has 2,057 groups and should not be interpreted as the number of LTCFs.",
                "",
                f"dbname chi-square statistic: {chi_square.loc[0, 'Chi-square statistic']:.6f}",
                f"dbname chi-square df: {int(chi_square.loc[0, 'df'])}",
                "dbname chi-square p value: <0.001",
                f"dbname chi-square Cramer's V: {cramers_v:.6f}",
                "",
                f"Number of prediction features counted: {len(features)}",
                f"Development cohort rows: {len(dev)}",
                f"Development cohort deaths: {int(pd.to_numeric(dev[OUTCOME_COL], errors='coerce').sum())}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_readme() -> None:
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# 0609 請蔡老師協助部分：結果檔案對應",
                "",
                "來源 PDF：`0609_請蔡老師協助的部分.pdf`",
                "",
                "本資料夾已依 PDF 逐項整理結果。優先開啟：",
                "",
                "`0609_development_facility_missingness_form.xlsx`",
                "",
                "## 重要修正：機構數與 dbname",
                "",
                "原本 0609 cache 檔 `Revision/20260523/training_data_1014_cached_for_completion.csv` 的 `dbname` 全部為空，是因為原始 notebook/script 對整份 Google Sheet 做 numeric coercion，文字型機構代碼被轉成 NaN。原始 `training_data_1014` Google Sheet 仍保留 confirmed resident-level `dbname`，且 numeric-clean 後與本地 cache row-by-row 一致。",
                "",
                "目前可驗證的 Development cohort 數字：",
                "",
                "- 住民數：23,901",
                "- 死亡數：5,272",
                "- confirmed unique `dbname`：493",
                "- unique `H01_NUM`：2,057，這不是機構數",
                "- `DATA/area_size.xlsx` facility roster：493 家",
                "",
                "因此正式 facility-level 分析使用 `dbname`，不再用 `H01_NUM` 代表機構。",
                "",
                "## 1. Internal calibration metrics with 95% CI",
                "",
                "- `calibration_metrics_internal_hybridxgbrf_with_ci.xlsx`",
                "- `0609_development_facility_missingness_form.xlsx`，sheet `Internal calibration CI`",
                "",
                "## 2. Internal risk-decile calibration",
                "",
                "- `risk_decile_calibration_internal_hybridxgbrf.xlsx`",
                "- `0609_development_facility_missingness_form.xlsx`，sheet `Internal risk decile`",
                "",
                "## 3. Development cohort facility size / institutional region missingness table",
                "",
                "正式表：",
                "",
                "- `0609_development_facility_missingness_form.xlsx`，sheet `Requested form`",
                "- `0609_development_facility_missingness_form.csv`",
                "",
                "此表已用原始 Google Sheet 的 confirmed `dbname` 接回 `DATA/area_size.xlsx`，因此可估計 N facilities、N residents、overall/dead/alive all-feature missingness 與 death rate。",
                "",
                "## 4. Chi-square test: institution ID x overall missingness",
                "",
                "正式 facility-level 檢定：",
                "",
                "- `0609_development_facility_missingness_form.xlsx`，sheet `Chi-square dbname`",
                "- `0609_dbname_missingness_chi_square.csv`",
                "",
                "結果為 `dbname x all-feature missing/observed cells`：N dbname groups = 493，df = 492，P < 0.001。",
                "",
                "診斷用 H01_NUM 檔案：",
                "",
                "- `0609_h01num_missingness_chi_square.csv`",
                "",
                "`H01_NUM` 有 2,057 groups，只能視為 documentation/resident identifier 診斷，不可解讀為 LTCF 家數。",
                "",
                "## 5. ALL Feature 清單",
                "",
                "- `0609_development_facility_missingness_form.xlsx`，sheet `Feature list`",
                "- 來源：`../../RESULTS/tables/shap_feature_importance.xlsx`",
                "",
                "ALL Feature 使用 29 個模型 predictor features；`死亡標記` 是 outcome，不納入 all-feature missingness 的 predictor feature 計算。",
                "",
                "## 6. missingness_indicator_key_features_regression 補意識總分Max_missing 與 P value",
                "",
                "- `0609_missingness_indicator_key_features_regression_with_p.xlsx`",
                "- `0609_missingness_indicator_key_features_regression_with_p.csv`",
                "- `0609_development_facility_missingness_form.xlsx`，sheet `Key missingness regression`",
                "",
                "已新增：",
                "",
                "- `Consciousness_total_max_missing`",
                "- Source variable：`意識總分Max`",
                "- P value / formatted P value",
                "",
                "## 7. 可重跑程式",
                "",
                "- `generate_0609_facility_missingness_form.py`",
                "",
                "此程式會重建主 workbook、confirmed dbname facility-level 表、dbname chi-square 表、H01_NUM 診斷表，以及補 P value 的 missingness indicator regression 表。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    dev_raw = load_development_cohort_raw()
    external_raw = load_external_cohort_raw()
    dev_cache = load_development_cache()
    external_cache = load_external_cache()
    roster = load_facility_roster()
    model_features = load_model_feature_list()
    features = feature_columns(dev_raw, model_features)

    cache_match, cache_mismatches = validate_raw_matches_cache(dev_raw, dev_cache)
    dev = add_roster(dev_raw, roster)
    requested = summarize_facility_strata(dev, features)
    overall_missingness = overall_development_missingness(dev, features)
    dbname_detail, dbname_chi_square = dbname_missingness_chi_square(dev, features)
    h01_detail, h01_summary = h01_diagnostic(dev, features)
    key_missingness_regression = missingness_indicator_key_features_regression(dev_raw, external_raw)
    audit = data_availability_audit(
        dev_raw,
        dev_cache,
        external_raw,
        external_cache,
        roster,
        features,
        cache_match,
        cache_mismatches,
    )
    internal_calibration = pd.read_csv(INTERNAL_CALIBRATION)
    internal_risk_decile = pd.read_csv(INTERNAL_RISK_DECILE)

    output_xlsx = OUT / "0609_development_facility_missingness_form.xlsx"
    with pd.ExcelWriter(output_xlsx) as writer:
        internal_calibration.to_excel(writer, index=False, sheet_name="Internal calibration CI")
        internal_risk_decile.to_excel(writer, index=False, sheet_name="Internal risk decile")
        requested.to_excel(writer, index=False, sheet_name="Requested form")
        overall_missingness.to_excel(writer, index=False, sheet_name="Overall dev missingness")
        model_features.to_excel(writer, index=False, sheet_name="Feature list")
        dbname_chi_square.to_excel(writer, index=False, sheet_name="Chi-square dbname")
        dbname_detail.sort_values("Overall missing percent", ascending=False).to_excel(
            writer, index=False, sheet_name="dbname missingness detail"
        )
        h01_summary.to_excel(writer, index=False, sheet_name="H01 diagnostic")
        h01_detail.sort_values("Overall missing percent", ascending=False).to_excel(
            writer, index=False, sheet_name="H01 missingness detail"
        )
        key_missingness_regression.to_excel(writer, index=False, sheet_name="Key missingness regression")
        audit.to_excel(writer, index=False, sheet_name="Data availability audit")

    requested.to_csv(OUT / "0609_development_facility_missingness_form.csv", index=False, encoding="utf-8-sig")
    dbname_chi_square.to_csv(OUT / "0609_dbname_missingness_chi_square.csv", index=False, encoding="utf-8-sig")
    dbname_detail.to_csv(OUT / "0609_dbname_missingness_detail.csv", index=False, encoding="utf-8-sig")
    h01_summary.to_csv(OUT / "0609_h01num_missingness_chi_square.csv", index=False, encoding="utf-8-sig")
    key_missingness_regression.to_csv(
        OUT / "0609_missingness_indicator_key_features_regression_with_p.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with pd.ExcelWriter(OUT / "0609_missingness_indicator_key_features_regression_with_p.xlsx") as writer:
        key_missingness_regression.to_excel(writer, index=False, sheet_name="Logistic regression")

    # Keep the old exploratory filenames from being mistaken as current formal outputs.
    deprecated = pd.DataFrame(
        [
            {
                "Status": "Deprecated",
                "Reason": (
                    "Confirmed resident-level dbname is available from the original training_data_1014 Google Sheet. "
                    "Use 0609_development_facility_missingness_form.csv and 0609_dbname_missingness_chi_square.csv."
                ),
            }
        ]
    )
    deprecated.to_csv(OUT / "0609_exploratory_mapped_facility_missingness_form.csv", index=False, encoding="utf-8-sig")
    deprecated.to_csv(OUT / "0609_exploratory_mapping_audit.csv", index=False, encoding="utf-8-sig")

    write_notes(dev_raw, features, dbname_chi_square)
    write_readme()


if __name__ == "__main__":
    main()
