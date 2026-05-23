from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PREV = ROOT / "Revision" / "01_需蔡老師補做資料表"
CSV = BASE / "analysis_data_filtering_out_included_ADL_missing_0523.csv"
FACILITY_XLSX = ROOT / "Revision" / "20260516" / "機構區域與大小.xlsx"


def fmt_mean_sd(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "NA"
    return f"{s.mean():.1f} ({s.std(ddof=1):.1f})"


def fmt_median_iqr(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "NA"
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{s.median():.1f} ({q1:.1f}-{q3:.1f})"


def fmt_binary(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return "NA"
    n = int((x == 1).sum())
    return f"{n} ({n / len(x) * 100:.1f}%)"


def p_cont(a: pd.Series, b: pd.Series) -> str:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return "NA"
    p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def p_binary(a: pd.Series, b: pd.Series) -> str:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if a.empty or b.empty:
        return "NA"
    table = np.array(
        [
            [(a == 1).sum(), (a != 1).sum()],
            [(b == 1).sum(), (b != 1).sum()],
        ]
    )
    if (table < 5).any():
        p = stats.fisher_exact(table).pvalue
    else:
        p = stats.chi2_contingency(table, correction=False).pvalue
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(df[x].astype(str), df[y], color=color)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def copy_prior_outputs() -> list[str]:
    files = [
        "included_vs_excluded_insufficient_followup.xlsx",
        "included_vs_excluded_insufficient_followup_with_p.xlsx",
        "development_cohort_plus_current_excluded_insufficient_followup.xlsx",
        "internal_cv_calibration_with_histogram.png",
        "external_validation_calibration_with_histogram.png",
        "paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx",
        "paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx",
        "table3_internal_cv_performance_with_ci.xlsx",
        "table4_external_validation_full_with_ci.xlsx",
        "table4_external_validation_paper_friendly.xlsx",
        "table5_subgroup_performance_with_ci.xlsx",
        "threshold_tradeoff_external_hybridxgbrf.xlsx",
        "decision_curve_external_validation.png",
        "decision_curve_external_validation.xlsx",
        "survival_cox_ml_risk_score.xlsx",
        "survival_time_dependent_auc.xlsx",
        "survival_km_external_by_risk_group.png",
        "survival_c_index.xlsx",
        "missingness_indicator_development.xlsx",
        "missingness_indicator_external.xlsx",
        "facility_missingness_development.xlsx",
        "facility_missingness_external.xlsx",
        "shap_feature_importance.png",
        "shap_feature_importance.xlsx",
        "calibration_metrics_external_hybridxgbrf_with_ci.xlsx",
        "risk_decile_calibration_external_hybridxgbrf.xlsx",
        "selected_features.xlsx",
    ]
    copied = []
    for name in files:
        src = PREV / name
        if src.exists():
            dst = BASE / name
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def main() -> None:
    df = pd.read_csv(CSV)
    n = len(df)
    copied = copy_prior_outputs()

    baseline_specs = [
        ("N residents", None, "n"),
        ("Age, years", "預估年齡", "mean"),
        ("Male sex", "性別_is_male", "binary"),
        ("DNR", "DNR_flag", "binary"),
        ("Observation days", "觀察天數", "median"),
        ("Death marker", "死亡標記", "binary"),
        ("Initial ADL score", "ADL_first_score", "mean"),
        ("Last ADL score", "ADL_last_score", "mean"),
        ("Maximum ADL score", "ADL_總分_max", "mean"),
        ("ADL change", "ADL_diff_seq", "mean"),
        ("Hospitalizations within 6 months", "六個月內住院次數", "mean"),
        ("Initial body weight", "BW_first", "mean"),
        ("Last body weight", "BW_last", "mean"),
        ("Body weight change", "BW_diff_seq", "mean"),
        ("Use of respiratory aid", "使用呼吸輔具", "binary"),
        ("Feeding tube at first record", "first_has_feeding_tube", "binary"),
        ("Had fall", "had_fall", "binary"),
        ("First consciousness total", "first_ 意識總分", "mean"),
        ("Last consciousness total", "last_ 意識總分", "mean"),
    ]
    rows = []
    for label, col, typ in baseline_specs:
        if typ == "n":
            value, missing = str(n), 0
        elif typ == "mean":
            value, missing = fmt_mean_sd(df[col]), int(df[col].isna().sum())
        elif typ == "median":
            value, missing = fmt_median_iqr(df[col]), int(df[col].isna().sum())
        else:
            value, missing = fmt_binary(df[col]), int(df[col].isna().sum())
        rows.append({"Characteristic": label, "Excluded residents": value, "Missing N": missing})
    baseline = pd.DataFrame(rows)

    adl_missing = df["ADL_總分_max"].isna() | df["ADL_first_score"].isna()
    comp_rows = []
    for label, col, typ in baseline_specs:
        if typ == "n":
            comp_rows.append(
                {
                    "Characteristic": label,
                    "ADL available": int((~adl_missing).sum()),
                    "ADL missing": int(adl_missing.sum()),
                    "P value": "",
                }
            )
            continue
        if typ == "mean":
            avail = fmt_mean_sd(df.loc[~adl_missing, col])
            miss = fmt_mean_sd(df.loc[adl_missing, col])
            pval = p_cont(df.loc[~adl_missing, col], df.loc[adl_missing, col])
        elif typ == "median":
            avail = fmt_median_iqr(df.loc[~adl_missing, col])
            miss = fmt_median_iqr(df.loc[adl_missing, col])
            pval = p_cont(df.loc[~adl_missing, col], df.loc[adl_missing, col])
        else:
            avail = fmt_binary(df.loc[~adl_missing, col])
            miss = fmt_binary(df.loc[adl_missing, col])
            pval = p_binary(df.loc[~adl_missing, col], df.loc[adl_missing, col])
        comp_rows.append(
            {
                "Characteristic": label,
                "ADL available": avail,
                "ADL missing": miss,
                "P value": pval,
                "ADL available missing N": int(df.loc[~adl_missing, col].isna().sum()),
                "ADL missing group missing N": int(df.loc[adl_missing, col].isna().sum()),
            }
        )
    adl_compare = pd.DataFrame(comp_rows)

    missing_rows = []
    for col in df.columns:
        miss = df[col].isna()
        missing_rows.append(
            {
                "Variable": col,
                "Missing N": int(miss.sum()),
                "Missing Percent": miss.mean(),
                "Nonmissing N": int((~miss).sum()),
            }
        )
    missingness = pd.DataFrame(missing_rows).sort_values(
        ["Missing Percent", "Variable"], ascending=[False, True]
    )

    feature_cols = [
        c
        for c in df.columns
        if c not in {"H01_NUM", "dbname", "入家日期", "結案日期", "死亡標記"}
    ]
    facility = (
        df.groupby("dbname", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "N residents": len(g),
                    "Overall missing percent": g[feature_cols].isna().mean().mean(),
                    "ADL missing percent": (
                        g["ADL_總分_max"].isna() | g["ADL_first_score"].isna()
                    ).mean(),
                    "Body weight first missing percent": g["BW_first"].isna().mean(),
                    "Median observation days": g["觀察天數"].median(),
                    "Death/event rate": pd.to_numeric(g["死亡標記"], errors="coerce").mean(),
                }
            )
        )
        .reset_index()
    )

    if FACILITY_XLSX.exists():
        fmap = pd.read_excel(FACILITY_XLSX, sheet_name="訓練資料_機構大小")
        fmap = fmap[["dbname", "核定床數", "機構大小層級", "區域"]].dropna(subset=["dbname"])
        facility = facility.merge(fmap, on="dbname", how="left")

    size_summary = pd.DataFrame()
    region_summary = pd.DataFrame()
    if "機構大小層級" in facility.columns:
        tmp = df.merge(facility[["dbname", "機構大小層級", "區域"]], on="dbname", how="left")
        size_summary = (
            tmp.groupby("機構大小層級", dropna=False)
            .agg(
                **{
                    "N residents": ("H01_NUM", "size"),
                    "N facilities": ("dbname", "nunique"),
                    "Median observation days": ("觀察天數", "median"),
                    "ADL missing percent": (
                        "ADL_總分_max",
                        lambda s: (
                            tmp.loc[s.index, "ADL_總分_max"].isna()
                            | tmp.loc[s.index, "ADL_first_score"].isna()
                        ).mean(),
                    ),
                    "Overall missing percent": (
                        "H01_NUM",
                        lambda s: tmp.loc[s.index, feature_cols].isna().mean().mean(),
                    ),
                }
            )
            .reset_index()
            .rename(columns={"機構大小層級": "Facility size"})
        )
        region_summary = (
            tmp.groupby("區域", dropna=False)
            .agg(
                **{
                    "N residents": ("H01_NUM", "size"),
                    "N facilities": ("dbname", "nunique"),
                    "Median observation days": ("觀察天數", "median"),
                    "ADL missing percent": (
                        "ADL_總分_max",
                        lambda s: (
                            tmp.loc[s.index, "ADL_總分_max"].isna()
                            | tmp.loc[s.index, "ADL_first_score"].isna()
                        ).mean(),
                    ),
                    "Overall missing percent": (
                        "H01_NUM",
                        lambda s: tmp.loc[s.index, feature_cols].isna().mean().mean(),
                    ),
                }
            )
            .reset_index()
            .rename(columns={"區域": "Facility region"})
        )

    with pd.ExcelWriter(BASE / "excluded_residents_baseline_summary_0523.xlsx") as writer:
        baseline.to_excel(writer, index=False, sheet_name="Baseline")
        adl_compare.to_excel(writer, index=False, sheet_name="ADL missing comparison")
        missingness.to_excel(writer, index=False, sheet_name="Variable missingness")

    with pd.ExcelWriter(BASE / "facility_missingness_and_outcome_0523.xlsx") as writer:
        facility.sort_values("N residents", ascending=False).to_excel(
            writer, index=False, sheet_name="Facility level"
        )
        if not size_summary.empty:
            size_summary.to_excel(writer, index=False, sheet_name="By facility size")
        if not region_summary.empty:
            region_summary.to_excel(writer, index=False, sheet_name="By region")

    plot_label_map = {
        "結案日期": "Discharge date",
        "first_ 意識_V": "First consciousness V",
        "last_ 意識_V": "Last consciousness V",
        "first_ 意識_E": "First consciousness E",
        "last_ 意識_E": "Last consciousness E",
        "first_ 意識_M": "First consciousness M",
        "last_ 意識_M": "Last consciousness M",
        "first_ 意識總分": "First consciousness total",
        "last_ 意識總分": "Last consciousness total",
        "意識總分Max": "Consciousness total max",
        "意識總分_diff": "Consciousness total change",
        "diff_has_denture": "Denture change flag",
        "diff_has_feeding_tube": "Feeding tube change flag",
        "disease_Max": "Disease max",
        "disease_Min": "Disease min",
        "disease_diff_seq": "Disease change",
        "disease_first_score": "Disease first score",
        "disease_last_score": "Disease last score",
        "disease_std": "Disease SD",
        "BW_diff_seq": "Body weight change",
        "BW_first": "Body weight first",
        "BW_last": "Body weight last",
        "ADL_last_score": "ADL last score",
        "ADL_diff_seq": "ADL change",
        "ADL_std": "ADL SD",
        "ADL_first_score": "ADL first score",
        "ADL_總分_max": "ADL total max",
        "ADL_Max": "ADL max",
        "ADL_Min": "ADL min",
        "觀察天數": "Observation days",
        "預估年齡": "Age",
    }

    top_missing = missingness.head(15).sort_values("Missing Percent").copy()
    top_missing["Plot label"] = top_missing["Variable"].map(plot_label_map).fillna(
        top_missing["Variable"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_missing["Plot label"], top_missing["Missing Percent"] * 100, color="#4c78a8")
    ax.set_xlabel("Missing percent")
    ax.set_title("Variables with highest missingness among excluded residents")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(BASE / "excluded_variable_missingness_top15_0523.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(pd.to_numeric(df["觀察天數"], errors="coerce").dropna(), bins=40, color="#59a14f")
    ax.set_title("Observation days among excluded residents")
    ax.set_xlabel("Observation days")
    ax.set_ylabel("Residents")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(BASE / "excluded_observation_days_histogram_0523.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(pd.to_numeric(df["預估年齡"], errors="coerce").dropna(), bins=35, color="#f28e2b")
    ax.set_title("Age distribution among excluded residents")
    ax.set_xlabel("Age, years")
    ax.set_ylabel("Residents")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(BASE / "excluded_age_histogram_0523.png", dpi=180)
    plt.close(fig)

    if not size_summary.empty:
        plot_df = size_summary.dropna(subset=["Facility size"]).copy()
        plot_df["Facility size"] = plot_df["Facility size"].replace(
            {"小": "Small", "中": "Medium", "大": "Large"}
        )
        save_bar(
            plot_df,
            "Facility size",
            "ADL missing percent",
            "ADL missingness by facility size",
            BASE / "excluded_adl_missingness_by_facility_size_0523.png",
            "#b07aa1",
        )
    if not region_summary.empty:
        plot_df = region_summary.dropna(subset=["Facility region"]).copy()
        plot_df["Facility region"] = plot_df["Facility region"].replace(
            {"北部": "North", "中部": "Central", "南部": "South", "東部": "East", "離島": "Islands"}
        )
        save_bar(
            plot_df,
            "Facility region",
            "ADL missing percent",
            "ADL missingness by facility region",
            BASE / "excluded_adl_missingness_by_region_0523.png",
            "#e15759",
        )

    response_md = f"""# Reviewer Response Outputs for 2026-05-23

Source document: `01_請蔡老師協助的部分2.docx`  
New data source: `analysis_data_filtering_out_included_ADL_missing_0523.csv`  
Excluded residents in 0523 file: `{n:,}`

## Reviewer W. Excluded residents and ADL-missing sensitivity

Response:
We regenerated the excluded-resident descriptive analyses using the May 23 data extract. The excluded file contains {n:,} residents. We summarized demographics, observation time, ADL-derived variables, body weight, care-related variables, variable-level missingness, and facility-level documentation patterns. Because the 0523 file contains the excluded residents only, these outputs should be interpreted as characterization of the excluded population and its ADL-missing subset; direct included-versus-excluded comparisons should use the existing analytic-cohort tables copied into this folder.

Files:
- `excluded_residents_baseline_summary_0523.xlsx`
- `excluded_variable_missingness_top15_0523.png`
- `excluded_observation_days_histogram_0523.png`
- `excluded_age_histogram_0523.png`

## Reviewer W. Included versus excluded residents

Response:
The included-versus-excluded baseline comparison from the prior regenerated analytic results has been copied into this folder for response assembly. The 0523 CSV itself does not include the analytic included cohort, so no new p-value comparison against included residents was recalculated from this file alone.

Files:
- `included_vs_excluded_insufficient_followup.xlsx` if present from prior output set
- `included_vs_excluded_insufficient_followup_with_p.xlsx` if present from prior output set
- `excluded_residents_baseline_summary_0523.xlsx`

## Reviewer W. Calibration plots with predicted probability histogram

Response:
Calibration plots with predicted-risk histograms were already generated in the regenerated model outputs and copied here. These figures show calibration curves with the predicted probability distribution underneath.

Files:
- `internal_cv_calibration_with_histogram.png`
- `external_validation_calibration_with_histogram.png`

## Reviewer BK. Paired bootstrap comparison

Response:
Paired bootstrap AUROC comparisons between HybridXGBRF and XGBoost are included. The manuscript response should describe the difference as small and avoid overstating superiority.

Files:
- `paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`
- `paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx`

## Reviewer BK. 95% confidence intervals for performance metrics

Response:
Performance tables with bootstrap 95% confidence intervals for internal validation, external validation, and subgroup analyses are included.

Files:
- `table3_internal_cv_performance_with_ci.xlsx`
- `table4_external_validation_full_with_ci.xlsx`
- `table4_external_validation_paper_friendly.xlsx`
- `table5_subgroup_performance_with_ci.xlsx`

## Reviewer BK. Threshold-specific tradeoffs and decision curve analysis

Response:
Threshold-specific PPV, NPV, sensitivity, specificity, F1, and decision-curve outputs are included. These support wording that lower thresholds may be more appropriate for screening, whereas higher thresholds prioritize PPV/specificity.

Files:
- `threshold_tradeoff_external_hybridxgbrf.xlsx`
- `decision_curve_external_validation.xlsx`
- `decision_curve_external_validation.png`

## Reviewer BK. Survival sensitivity analysis

Response:
Survival sensitivity outputs are included as supplementary analyses. Because the data structure is based on an administrative 180-day binary outcome and follow-up is not fully continuous after discharge, these should be described as sensitivity analyses, with interval/censoring limitations noted.

Files:
- `survival_cox_ml_risk_score.xlsx`
- `survival_time_dependent_auc.xlsx`
- `survival_c_index.xlsx`
- `survival_km_external_by_risk_group.png`

## Reviewer BK. Missingness indicators and facility-level missingness

Response:
The 0523 excluded-resident file was used to regenerate variable missingness, ADL-missing subgroup comparisons, and facility-level missingness summaries. These outputs help address whether documentation patterns and facility-level missingness may affect interpretation.

Files:
- `excluded_residents_baseline_summary_0523.xlsx`
- `facility_missingness_and_outcome_0523.xlsx`
- `excluded_adl_missingness_by_facility_size_0523.png`
- `excluded_adl_missingness_by_region_0523.png`
- `missingness_indicator_development.xlsx`
- `missingness_indicator_external.xlsx`
- `facility_missingness_development.xlsx`
- `facility_missingness_external.xlsx`

## Reviewer BK. Multiple imputation

Response:
The response should clarify that the primary regenerated model uses development-fitted preprocessing with binary/count missing values imputed as 0 and continuous/scale values handled as mean-equivalent imputation after standardization. Multiple imputation can be discussed as a sensitivity option if added later, but no multiple-imputation model result is generated from the 0523 excluded-only file.

Files:
- `selected_features.xlsx`
- `excluded_residents_baseline_summary_0523.xlsx`

## Reviewer BK. SHAP explanation

Response:
The SHAP feature-importance output for the final model has been copied here and can be cited as model-agnostic explanation of final predicted probabilities.

Files:
- `shap_feature_importance.png`
- `shap_feature_importance.xlsx`

## Reviewer BK. Calibration metrics and risk-decile calibration

Response:
Calibration metrics and risk-decile calibration outputs are included. These support adding intercept, slope, O/E ratio, Brier score, and decile-level calibration summaries.

Files:
- `calibration_metrics_external_hybridxgbrf_with_ci.xlsx`
- `risk_decile_calibration_external_hybridxgbrf.xlsx`

## Reproducibility details

Response:
The generated 0523 files were created by `generate_0523_revision_outputs.py`. The copied model outputs come from the regenerated revision output folder. Random seeds, package versions, preprocessing, and source code details should be cited from the project scripts and README materials.

Copied prior-output files: {", ".join(copied)}
"""
    (BASE / "reviewer_response_outputs_20260523.md").write_text(response_md, encoding="utf-8")
    print(f"Generated 0523 outputs in {BASE}")
    print(f"Copied {len(copied)} prior-output files")


if __name__ == "__main__":
    main()
