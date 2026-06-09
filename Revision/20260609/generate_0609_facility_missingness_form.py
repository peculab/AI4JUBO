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
LINKAGE_SOURCES = [
    ROOT / "DATA" / "analysis_data_filtering_out_0514.csv",
    ROOT / "DATA" / "analysis_data_filtering_out_included_ADL_missing_0514.csv",
    ROOT / "Revision" / "20260523" / "analysis_data_filtering_out_included_ADL_missing_0523.csv",
]

OUTCOME_COL = "死亡標記"
IDENTIFIER_COLS = {"H01_NUM", "dbname", "入家日期", "結案日期", OUTCOME_COL, "觀察天數"}


def fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value * 100:.1f}%"


def load_development_cohort() -> pd.DataFrame:
    return pd.read_csv(TRAINING_CACHE)


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
        raise ValueError(f"Model features missing from Development cohort cache: {missing}")
    return features


def requested_form_from_available_sources(roster: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    not_estimable = "Not estimable from current project files"

    size_rows = [
        ("facility_size <50", roster["核定床數"] < 50),
        ("facility_size 50-150", roster["核定床數"].between(50, 150, inclusive="both")),
        ("facility_size >150", roster["核定床數"] > 150),
    ]
    region_rows = [
        ("Institutional regions_North", roster["區域"] == "北部"),
        ("Institutional regions_Central", roster["區域"] == "中部"),
        ("Institutional regions_South", roster["區域"] == "南部"),
        ("Institutional regions_East", roster["區域"] == "東部"),
    ]

    for variable, mask in [*size_rows, *region_rows]:
        sub = roster.loc[mask]
        rows.append(
            {
                "Variable": variable,
                "N facilities": int(sub["dbname"].nunique()),
                "N residents": not_estimable,
                "ALL Feature Missing Percent (Overall missing percent)": not_estimable,
                "ALL Feature Missing Percent/ Dead cohort": not_estimable,
                "ALL Feature Missing Percent/ Alive cohort": not_estimable,
                "Death rate": not_estimable,
                "Status/Notes": (
                    "N facilities is from DATA/area_size.xlsx facility roster. "
                    "Resident-level Development cohort dbname/facility linkage is absent in "
                    "Revision/20260523/training_data_1014_cached_for_completion.csv, so resident counts, "
                    "dead/alive all-feature missingness, and death rate are not estimable from current project files."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_h01_dbname_mode_map() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in LINKAGE_SOURCES:
        if path.exists():
            tmp = pd.read_csv(path, usecols=["H01_NUM", "dbname"])
            tmp["Linkage source"] = str(path.relative_to(ROOT))
            frames.append(tmp)
    linkage = pd.concat(frames, ignore_index=True).dropna(subset=["H01_NUM", "dbname"])
    counts = linkage.groupby(["H01_NUM", "dbname"]).size().reset_index(name="Pair count")
    total = counts.groupby("H01_NUM")["Pair count"].sum().rename("Total linkage rows")
    n_db = counts.groupby("H01_NUM")["dbname"].nunique().rename("Candidate dbname count")
    mode = counts.sort_values(["H01_NUM", "Pair count", "dbname"], ascending=[True, False, True]).drop_duplicates("H01_NUM")
    mode = mode.merge(total, on="H01_NUM", how="left").merge(n_db, on="H01_NUM", how="left")
    mode["Mode share"] = mode["Pair count"] / mode["Total linkage rows"]
    mode = mode.rename(columns={"dbname": "Mapped dbname"})

    diagnostic = pd.DataFrame(
        [
            {
                "Linkage method": "H01_NUM to dbname by modal value from excluded/supplemental local files",
                "N H01_NUM with any candidate dbname": int(mode["H01_NUM"].nunique()),
                "Median candidate dbname count per H01_NUM": float(mode["Candidate dbname count"].median()),
                "Median mode share": float(mode["Mode share"].median()),
                "Important limitation": (
                    "Many H01_NUM values map to many dbname candidates in the available local files. "
                    "This mapping is exploratory and should not be treated as confirmed resident-level facility linkage."
                ),
            }
        ]
    )
    return mode, diagnostic


def exploratory_mapped_form(
    df: pd.DataFrame,
    roster: pd.DataFrame,
    features: list[str],
    h01_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work = work.merge(
        h01_map[["H01_NUM", "Mapped dbname", "Candidate dbname count", "Mode share"]],
        on="H01_NUM",
        how="left",
    )
    roster2 = roster.rename(columns={"dbname": "Mapped dbname"}).copy()
    work = work.merge(roster2[["Mapped dbname", "核定床數", "機構大小層級", "區域"]], on="Mapped dbname", how="left")
    y = pd.to_numeric(work[OUTCOME_COL], errors="coerce").astype(int)

    def summarize(label: str, mask: pd.Series) -> dict[str, object]:
        sub = work.loc[mask].copy()
        if sub.empty:
            return {
                "Variable": label,
                "N facilities": 0,
                "N residents": 0,
                "ALL Feature Missing Percent (Overall missing percent)": np.nan,
                "ALL Feature Missing Percent/ Dead cohort": np.nan,
                "ALL Feature Missing Percent/ Alive cohort": np.nan,
                "Death rate": np.nan,
                "Mapped resident coverage note": "No mapped residents in this stratum.",
            }
        y_sub = pd.to_numeric(sub[OUTCOME_COL], errors="coerce").astype(int)

        def miss(mask2: pd.Series | np.ndarray | None = None) -> float:
            ss = sub.loc[mask2, features] if mask2 is not None else sub[features]
            return float(ss.isna().mean().mean()) if len(ss) else np.nan

        return {
            "Variable": label,
            "N facilities": int(sub["Mapped dbname"].nunique(dropna=True)),
            "N residents": int(len(sub)),
            "ALL Feature Missing Percent (Overall missing percent)": miss(),
            "ALL Feature Missing Percent/ Dead cohort": miss(y_sub == 1),
            "ALL Feature Missing Percent/ Alive cohort": miss(y_sub == 0),
            "Death rate": float(y_sub.mean()),
            "Mapped resident coverage note": (
                "Exploratory values based on modal H01_NUM-to-dbname mapping; "
                f"median mode share in this stratum={sub['Mode share'].median():.3f}; "
                f"median candidate dbname count={sub['Candidate dbname count'].median():.1f}."
            ),
        }

    rows = [
        summarize("facility_size <50", work["核定床數"] < 50),
        summarize("facility_size 50-150", work["核定床數"].between(50, 150, inclusive="both")),
        summarize("facility_size >150", work["核定床數"] > 150),
        summarize("Institutional regions_North", work["區域"] == "北部"),
        summarize("Institutional regions_Central", work["區域"] == "中部"),
        summarize("Institutional regions_South", work["區域"] == "南部"),
        summarize("Institutional regions_East", work["區域"] == "東部"),
    ]
    form = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(
        [
            {
                "Metric": "Development rows",
                "Value": len(work),
            },
            {
                "Metric": "Rows with modal H01_NUM-to-dbname mapping",
                "Value": int(work["Mapped dbname"].notna().sum()),
            },
            {
                "Metric": "Rows with mapped dbname and facility size/region",
                "Value": int(work["機構大小層級"].notna().sum()),
            },
            {
                "Metric": "Mapped resident coverage",
                "Value": float(work["Mapped dbname"].notna().mean()),
            },
            {
                "Metric": "Mapped roster coverage",
                "Value": float(work["機構大小層級"].notna().mean()),
            },
            {
                "Metric": "Weighted median mode share",
                "Value": float(work.loc[work["Mapped dbname"].notna(), "Mode share"].median()),
            },
            {
                "Metric": "Weighted median candidate dbname count",
                "Value": float(work.loc[work["Mapped dbname"].notna(), "Candidate dbname count"].median()),
            },
        ]
    )
    return form, diagnostics


def h01_missingness_chi_square(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = df[features].isna()
    by_h01 = (
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
    by_h01["Total feature cells"] = by_h01["Missing feature cells"] + by_h01["Observed feature cells"]
    by_h01["Overall missing percent"] = by_h01["Missing feature cells"] / by_h01["Total feature cells"]
    by_h01["N residents/rows"] = df.groupby("H01_NUM", dropna=False).size().to_numpy()

    contingency = by_h01[["Missing feature cells", "Observed feature cells"]].to_numpy()
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    total = contingency.sum()
    cramers_v = float(np.sqrt(chi2 / (total * (min(contingency.shape) - 1))))

    chi_square = pd.DataFrame(
        [
            {
                "Test": "H01_NUM x all-feature missing/observed cells",
                "Chi-square statistic": chi2,
                "df": dof,
                "P value": p_value,
                "Table-ready P value": "<0.001" if p_value < 0.001 else f"{p_value:.3f}",
                "Cramer's V": cramers_v,
                "N H01_NUM groups": int(by_h01["H01_NUM"].nunique(dropna=False)),
                "Total feature cells": int(total),
                "Missing feature cells": int(contingency[:, 0].sum()),
                "Observed feature cells": int(contingency[:, 1].sum()),
                "Important limitation": (
                    "This is computed using H01_NUM because the analytic Development cohort cache has empty dbname. "
                    "It should be treated as an exploratory facility/documentation-identifier test unless H01_NUM is confirmed "
                    "to be the intended institution ID."
                ),
            }
        ]
    )
    return by_h01, chi_square


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
                "Notes": "Overall Development cohort values across the 29 SHAP/model features listed in the Feature list sheet; not stratified by facility because dbname is unavailable in the analytic cache.",
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


def data_availability_audit(df: pd.DataFrame, roster: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    dbname_usable = df["dbname"].notna() & ~df["dbname"].astype(str).str.strip().isin(["", "nan", "None"])
    return pd.DataFrame(
        [
            {
                "Item": "Development cohort source",
                "Finding": str(TRAINING_CACHE.relative_to(ROOT)),
            },
            {
                "Item": "Development cohort rows",
                "Finding": len(df),
            },
            {
                "Item": "Development cohort deaths",
                "Finding": int(pd.to_numeric(df[OUTCOME_COL], errors="coerce").sum()),
            },
            {
                "Item": "All prediction features counted",
                "Finding": len(features),
            },
            {
                "Item": "dbname missing rows in Development cache",
                "Finding": int(df["dbname"].isna().sum()),
            },
            {
                "Item": "Usable dbname rows in Development cache",
                "Finding": int(dbname_usable.sum()),
            },
            {
                "Item": "Unique H01_NUM groups in Development cache",
                "Finding": int(df["H01_NUM"].nunique(dropna=False)),
            },
            {
                "Item": "Facility roster source",
                "Finding": str(FACILITY_ROSTER.relative_to(ROOT)),
            },
            {
                "Item": "Facility roster rows",
                "Finding": len(roster),
            },
            {
                "Item": "Conclusion",
                "Finding": (
                    "The requested resident-level facility size/region table cannot be fully populated from current project files "
                    "because the Development cohort cache lacks usable dbname/facility linkage."
                ),
            },
        ]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load_development_cohort()
    external = pd.read_csv(EXTERNAL_CACHE)
    roster = load_facility_roster()
    model_features = load_model_feature_list()
    features = feature_columns(dev, model_features)

    requested = requested_form_from_available_sources(roster)
    overall_missingness = overall_development_missingness(dev, features)
    h01_missingness, chi_square = h01_missingness_chi_square(dev, features)
    key_missingness_regression = missingness_indicator_key_features_regression(dev, external)
    audit = data_availability_audit(dev, roster, features)
    internal_calibration = pd.read_csv(INTERNAL_CALIBRATION)
    internal_risk_decile = pd.read_csv(INTERNAL_RISK_DECILE)
    h01_map, h01_linkage_diagnostic = build_h01_dbname_mode_map()
    exploratory_form, exploratory_diagnostics = exploratory_mapped_form(dev, roster, features, h01_map)

    output_xlsx = OUT / "0609_development_facility_missingness_form.xlsx"
    with pd.ExcelWriter(output_xlsx) as writer:
        internal_calibration.to_excel(writer, index=False, sheet_name="Internal calibration CI")
        internal_risk_decile.to_excel(writer, index=False, sheet_name="Internal risk decile")
        requested.to_excel(writer, index=False, sheet_name="Requested form")
        exploratory_form.to_excel(writer, index=False, sheet_name="Exploratory mapped form")
        exploratory_diagnostics.to_excel(writer, index=False, sheet_name="Exploratory mapping audit")
        overall_missingness.to_excel(writer, index=False, sheet_name="Overall dev missingness")
        model_features.to_excel(writer, index=False, sheet_name="Feature list")
        chi_square.to_excel(writer, index=False, sheet_name="Chi-square")
        key_missingness_regression.to_excel(writer, index=False, sheet_name="Key missingness regression")
        h01_linkage_diagnostic.to_excel(writer, index=False, sheet_name="H01 linkage method")
        h01_map.sort_values(["Candidate dbname count", "Mode share"], ascending=[False, True]).to_excel(
            writer, index=False, sheet_name="H01 dbname mode map"
        )
        h01_missingness.sort_values("Overall missing percent", ascending=False).to_excel(
            writer, index=False, sheet_name="H01 missingness detail"
        )
        audit.to_excel(writer, index=False, sheet_name="Data availability audit")

    requested.to_csv(OUT / "0609_development_facility_missingness_form.csv", index=False, encoding="utf-8-sig")
    exploratory_form.to_csv(
        OUT / "0609_exploratory_mapped_facility_missingness_form.csv",
        index=False,
        encoding="utf-8-sig",
    )
    exploratory_diagnostics.to_csv(
        OUT / "0609_exploratory_mapping_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    chi_square.to_csv(OUT / "0609_h01num_missingness_chi_square.csv", index=False, encoding="utf-8-sig")
    key_missingness_regression.to_csv(
        OUT / "0609_missingness_indicator_key_features_regression_with_p.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with pd.ExcelWriter(OUT / "0609_missingness_indicator_key_features_regression_with_p.xlsx") as writer:
        key_missingness_regression.to_excel(writer, index=False, sheet_name="Logistic regression")

    summary = OUT / "0609_development_facility_missingness_form_notes.md"
    summary.write_text(
        "\n".join(
            [
                "# 0609 Development Facility Missingness Form Notes",
                "",
                "## What Was Requested",
                "",
                "The PDF requested a Development cohort table stratified by facility size and institutional region, including N facilities, N residents, all-feature missing percent overall, all-feature missing percent among dead residents, all-feature missing percent among alive residents, and death rate. It also requested a chi-square test for institution ID by overall all-feature missing percent.",
                "",
                "## Files Produced",
                "",
                "- `0609_development_facility_missingness_form.xlsx`",
                "- `0609_development_facility_missingness_form.csv`",
                "- `0609_exploratory_mapped_facility_missingness_form.csv`",
                "- `0609_exploratory_mapping_audit.csv`",
                "- `0609_h01num_missingness_chi_square.csv`",
                "- `0609_missingness_indicator_key_features_regression_with_p.xlsx`",
                "- `0609_missingness_indicator_key_features_regression_with_p.csv`",
                "",
                "## All Features",
                "",
                "The all-feature missingness calculation uses the 29 model predictor features listed in `RESULTS/tables/shap_feature_importance.xlsx`, matching the feature list shown in the 0609 PDF. `死亡標記` from `selected_features.xlsx` is the outcome and is not counted as a predictor feature.",
                "",
                "## Data Limitation",
                "",
                "The current project files do not contain a usable resident-level `dbname` / facility linkage for the Development cohort model cache (`Revision/20260523/training_data_1014_cached_for_completion.csv`). In that file, `dbname` is empty for all 23,901 rows. Therefore, resident counts, dead/alive all-feature missingness, and death rates by facility size or institutional region cannot be estimated reliably from the saved Development cohort cache.",
                "",
                "The `N facilities` column in the requested form was filled from `DATA/area_size.xlsx`, sheet `訓練資料_機構大小`. Other resident-level columns are marked as not estimable in the workbook.",
                "",
                "## Exploratory Mapping Attempt",
                "",
                "An additional sheet, `Exploratory mapped form`, uses a modal H01_NUM-to-dbname map derived from local excluded/supplemental files and then merges to `DATA/area_size.xlsx`. This provides a complete numeric table, but it is exploratory. The `Exploratory mapping audit` sheet reports coverage and ambiguity. In the current data, many H01_NUM values have many candidate dbname values, so this should not replace confirmed resident-level facility linkage.",
                "",
                "## Chi-square Test",
                "",
                "Because `dbname` is absent in the Development cohort cache, the chi-square sheet uses `H01_NUM` as the only available repeated identifier in the analytic cache. This should be treated as exploratory unless `H01_NUM` is confirmed to be the intended institution ID.",
                "",
                f"Number of prediction features counted: {len(features)}",
                f"Development cohort rows: {len(dev)}",
                f"Development cohort deaths: {int(pd.to_numeric(dev[OUTCOME_COL], errors='coerce').sum())}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
