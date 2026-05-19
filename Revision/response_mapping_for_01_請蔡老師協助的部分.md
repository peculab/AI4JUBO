# `01_請蔡老師協助的部分.docx` 對應回覆、圖表與檔名

更新日期：2026-05-18 晚間重跑後  
工作區：`C:\AI4JUBO`

本文依 `01_請蔡老師協助的部分.docx` 的順序整理可直接回覆 reviewer 的內容與對應檔案。所有模型 performance、calibration、SHAP、survival、threshold、DCA、missingness 相關數字已改以 2026-05-18 晚間重跑後的 `RESULTS` 為準。

## 0. 本次一致性更新重點

已完成：

- 程式碼與原始投稿 missing-data 說明對齊：binary/count 缺失補 0；continuous/scale 以 development cohort mean/SD 標準化後，missing z-score 補 0，等同原始尺度補 mean。
- `revision_generate_results.py --use-google-sheets` 已重跑。
- `survival_generate_results.py --use-google-sheets` 已重跑。
- 最新 xlsx/png 已同步到 `Revision/01_需蔡老師補做資料表/`。
- `Revision/missing_value_imputation_code_comparison.md` 已更新為最新補值策略。
- `Revision/reviewer_BK_W_coverage_check.md` 已更新為最新 reviewer point-to-point coverage。

注意：`jubodeath_v9_puredata_paper.ipynb` 是原始投稿追溯來源；修稿後重跑結果以 `revision_generate_results.py`、`survival_generate_results.py`、`RESULTS`、以及 `Revision/01_需蔡老師補做資料表` 內 2026-05-18 20:48-20:54 檔案為準。

## 1. Reviewer W：排除個案 baseline analysis 與 selection bias

### 對應檔案

- `RESULTS/tables/included_vs_excluded_insufficient_followup.xlsx`
- `RESULTS/tables/included_vs_excluded_insufficient_followup_with_p.xlsx`
- `RESULTS/tables/development_cohort_plus_current_excluded_insufficient_followup.xlsx`
- `RESULTS/tables/excluded_residents_baseline_summary.xlsx`
- `RESULTS/tables/excluded_adl_missing_baseline_summary.xlsx`
- `RESULTS/tables/excluded_exit_reason_summary.xlsx`
- `RESULTS/tables/excluded_residents_by_facility_size_20260516.xlsx`
- `RESULTS/tables/excluded_residents_by_region_20260516.xlsx`
- `RESULTS/figures/excluded_facility_region_size_20260516.png`

以上最新版本也已複製到：

- `Revision/01_需蔡老師補做資料表/`

### 目前可回覆內容

已補 included analytic residents 與 insufficient-follow-up excluded residents 的 baseline comparison。Included analytic cohort 為 development + temporal external validation 合併後 `N = 30,117`；insufficient-follow-up excluded residents 為 `N = 19,756`。另補 excluded residents 的 exit/discharge reason summary、ADL-missing excluded subset summary、facility size 與 region stratification。

需注意限制：included cohort 的 Google Sheets analytic extract 目前缺乏穩定可與 `area_size.xlsx` 合併的 resident-level `dbname`，因此 included vs excluded 的 resident-level facility size/region statistical comparison 仍不完整。可呈現 development facility roster characterization 與 excluded-resident stratification，但不能宣稱已完成完整 resident-level facility-size/region comparison。

### 可貼入 response letter

Thank you for pointing out the potential selection bias introduced by excluding residents without sufficient follow-up for 6-month outcome ascertainment. We added a baseline comparison between included analytic residents and residents excluded because of insufficient follow-up. The new supplementary tables compare available demographic, functional, clinical, and care-related characteristics between the two groups and summarize discharge/exit reasons among excluded residents. We also added facility-size and facility-region summaries for excluded residents using approved bed capacity and regional information available from the data source. Because facility identifiers were not consistently available in the analytic Google Sheets extract used for the included cohort, facility-size and region stratification should be interpreted primarily as excluded-resident characterization rather than a complete resident-level included-versus-excluded facility comparison. We expanded the Limitations section to clarify how exclusion due to incomplete follow-up and incomplete facility linkage may affect generalizability.

## 2. Reviewer W：Calibration plot 加 predicted probability histogram

### 對應檔案

- `RESULTS/figures/internal_cv_calibration_with_histogram.png`
- `RESULTS/figures/external_validation_calibration_with_histogram.png`
- `RESULTS/tables/calibration_metrics_external_hybridxgbrf_with_ci.xlsx`
- `RESULTS/tables/risk_decile_calibration_external_hybridxgbrf.xlsx`

### 可回覆內容

Calibration plots 已更新為上下兩層：上方為 calibration curve，下方為 predicted probability histogram。External calibration numeric metrics 也已補：intercept、slope、O/E ratio、Brier score 與 bootstrap 95% CI。

最新 external calibration：

- Calibration intercept = 0.516 (95% CI 0.447-0.597)
- Calibration slope = 1.271 (95% CI 1.217-1.342)
- O/E ratio = 1.128 (95% CI 1.108-1.149)
- Brier score = 0.112 (95% CI 0.108-0.116)

### 可貼入 response letter

Thank you for this helpful suggestion. We revised the calibration plots to include the distribution of predicted probabilities. The upper panel shows observed versus predicted risk across calibration bins, and the lower panel shows the histogram of predicted probabilities. We also added numerical calibration metrics, including calibration intercept, calibration slope, observed/expected ratio, Brier score, and risk-decile calibration with bootstrap 95% confidence intervals.

## 3. Reviewer W / BK：Missing data、補值、facility-level missingness

### 對應檔案

- `Revision/missing_value_imputation_code_comparison.md`
- `RESULTS/tables/missingness_indicator_development.xlsx`
- `RESULTS/tables/missingness_indicator_external.xlsx`
- `RESULTS/tables/facility_missingness_development.xlsx`
- `RESULTS/tables/facility_missingness_external.xlsx`
- `RESULTS/tables/selected_features.xlsx`

### 目前可回覆內容

已將程式碼和投稿文字對齊：

- Candidate predictors missingness >= 30% in development cohort：排除。
- Binary/count variables：missing 補 0。
- Continuous/scale variables：development cohort 或 CV training fold mean/SD 標準化，missing z-score 補 0，等同原始尺度補 mean。
- External validation：使用 development-fitted preprocessing，不用 external 估計 mean/SD。
- Internal CV：每 fold 只用 training fold 估 preprocessing。

ADL wording 要分清楚：嚴重/incomplete ADL assessment 可作為 cohort exclusion；納入分析後 retained ADL-derived features 的剩餘 missingness 則依 continuous/scale mean-equivalent imputation 處理。

### 可貼入 response letter

We revised the analysis code to align with the missing-data strategy described in the manuscript. Candidate predictors with 30% or greater missingness in the development cohort were excluded. For retained binary or count variables typically documented when present, missing values were imputed as 0. Continuous measures and scale scores were standardized using means and standard deviations estimated from the development cohort only; missing standardized values were then imputed as 0, corresponding to mean imputation on the original scale. The same development-cohort preprocessing parameters were applied unchanged to the temporal external validation cohort. In internal cross-validation, preprocessing parameters were estimated within each training fold and applied to the corresponding validation fold. We also added missingness-indicator and facility-level missingness summaries to evaluate documentation patterns and potential bias.

## 4. Reviewer BK：HybridXGBRF vs XGBoost paired comparison

### 對應檔案

- `RESULTS/tables/paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`
- `RESULTS/tables/paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx`
- `RESULTS/model_identity_note.txt`

### 最新結果

- Internal AUROC difference Hybrid - XGB = 0.0032 (95% CI 0.0018-0.0048), P < 0.001
- External AUROC difference Hybrid - XGB = 0.0036 (95% CI 0.0012-0.0060), P = 0.003

### 回覆口徑

雖然最新 paired bootstrap 達統計顯著，但差異幅度很小。Manuscript 不應強調 clinically large superiority。建議改成：selected tree-based models showed comparable high discrimination, with only a small AUROC difference between the selected model and XGBoost.

### 可貼入 response letter

Thank you for highlighting that the improvement over XGBoost was small. We added paired bootstrap AUROC comparisons between the selected model and XGBoost using the same bootstrap samples. Although the AUROC difference was statistically distinguishable in the regenerated analysis, the absolute difference was small. We therefore revised the manuscript to avoid overstating superiority and instead describe the selected tree-based models as showing comparable high discrimination.

## 5. Reviewer BK：Performance metrics 95% CI

### 對應檔案

- `RESULTS/tables/table3_internal_cv_performance_with_ci.xlsx`
- `RESULTS/tables/table4_external_validation_full_with_ci.xlsx`
- `RESULTS/tables/table4_external_validation_paper_friendly.xlsx`
- `RESULTS/tables/table5_subgroup_performance_with_ci.xlsx`

### 最新主要結果

Internal CV HybridXGBRF：

- AUROC = 0.888 (95% CI 0.883-0.893)
- Accuracy = 0.866 (95% CI 0.862-0.869)
- Precision = 0.788 (95% CI 0.775-0.799)
- Recall = 0.536 (95% CI 0.523-0.549)
- Specificity = 0.959 (95% CI 0.956-0.962)
- F1 = 0.638 (95% CI 0.626-0.649)
- Brier = 0.101 (95% CI 0.099-0.103)

External validation HybridXGBRF：

- AUROC = 0.898 (95% CI 0.889-0.906)
- Accuracy = 0.848 (95% CI 0.841-0.856)
- Precision / PPV = 0.855 (95% CI 0.836-0.875)
- Recall / Sensitivity = 0.567 (95% CI 0.545-0.590)
- Specificity = 0.961 (95% CI 0.956-0.967)
- NPV = 0.847 (95% CI 0.840-0.854)
- F1 = 0.682 (95% CI 0.663-0.701)
- Brier = 0.112 (95% CI 0.108-0.116)

## 6. Reviewer BK：Threshold tradeoffs、PPV/NPV、DCA

### 對應檔案

- `RESULTS/tables/threshold_tradeoff_external_hybridxgbrf.xlsx`
- `RESULTS/tables/decision_curve_external_validation.xlsx`
- `RESULTS/figures/decision_curve_external_validation.png`
- `RESULTS/figures/external_validation_confusion_matrices.png`

### 最新 threshold examples

External HybridXGBRF：

- Threshold 0.10：Sensitivity 0.934, PPV 0.469, NPV 0.956
- Threshold 0.20：Sensitivity 0.861, PPV 0.577, NPV 0.930
- Threshold 0.30：Sensitivity 0.723, PPV 0.714, NPV 0.888
- Threshold 0.40：Sensitivity 0.641, PPV 0.801, NPV 0.867
- Threshold 0.50：Sensitivity 0.567, PPV 0.855, NPV 0.847

### 可貼入 response letter

We added threshold-specific analyses showing sensitivity, specificity, PPV, NPV, F1 score, confusion-matrix counts, and net benefit across clinically plausible thresholds. These results clarify that lower thresholds may be preferable when the intended use is screening or prompting clinical review, whereas higher thresholds may be used for prioritizing limited resources. We also added decision-curve analysis to evaluate clinical utility beyond AUROC.

## 7. Reviewer BK：Survival / time-to-event sensitivity

### 對應檔案

- `RESULTS/tables/survival_time_dependent_auc.xlsx`
- `RESULTS/tables/survival_c_index.xlsx`
- `RESULTS/tables/survival_cox_ml_risk_score.xlsx`
- `RESULTS/tables/survival_km_external_risk_group_summary.xlsx`
- `RESULTS/tables/survival_logrank_external_risk_groups.xlsx`
- `RESULTS/figures/survival_time_dependent_auc.png`
- `RESULTS/figures/survival_km_external_by_risk_group.png`

### 最新結果

- Development Harrell C-index = 0.866
- External Harrell C-index = 0.860
- External Cox HR per 0.10 predicted risk increase = 1.641 (95% CI 1.614-1.669)
- External cumulative/dynamic AUC at months 1-5 = 0.916, 0.924, 0.918, 0.908, 0.902
- 180-day AUC is NA because no event-free controls remain under the 180-day administrative binary outcome setup.
- External KM risk-group event rates: low 2.5%, medium 17.2%, high 66.3%.

### 可貼入 response letter

We added survival sensitivity analyses using available observation days and death indicators under a 180-day administrative horizon. These analyses included Harrell's C-index, Cox models using the ML predicted risk score, cumulative/dynamic AUC at monthly horizons, and Kaplan-Meier curves by predicted-risk strata. These results support the risk-stratification ability of the model in a time-to-event framework, but we present them as sensitivity analyses rather than replacing the primary binary 6-month mortality model because post-discharge death ascertainment and censoring mechanisms may be incomplete in the available dataset.

## 8. Reviewer BK：SHAP final output explanation

### 對應檔案

- `RESULTS/tables/shap_feature_importance.xlsx`
- `RESULTS/figures/shap_feature_importance.png`
- `RESULTS/model_identity_note.txt`

### 最新 SHAP top features

1. Hospitalizations within 6 months
2. ADL change
3. Body weight change
4. ADL last score
5. ADL total max
6. Male
7. ADL maximum
8. ADL standard deviation
9. Body weight first
10. Use of respiratory aid

### 可貼入 response letter

Thank you for identifying this ambiguity. We clarified the SHAP analysis and regenerated the SHAP feature-importance results using a model-agnostic KernelSHAP approach applied to the final `predict_proba` output. Therefore, the revised SHAP results explain the final predicted probability rather than only an intermediate model component. We revised the Methods and figure caption accordingly.

## 9. Reviewer BK：Code、random seed、package versions、preprocessing

### 對應檔案

- `revision_generate_results.py`
- `survival_generate_results.py`
- `README.md`
- `RESULTS/README_RESULTS.md`
- `RESULTS/package_versions.txt`
- `RESULTS/tables/selected_features.xlsx`
- `Revision/missing_value_imputation_code_comparison.md`

### 可貼入 response letter

Thank you for this suggestion. We revised the reproducibility description to include preprocessing details, model settings, random seeds, and package versions. The analysis used fixed random seeds for cross-validation, model fitting where applicable, and bootstrap resampling. We also prepared analysis scripts documenting preprocessing, model evaluation, bootstrap confidence intervals, calibration analyses, threshold-specific analyses, decision-curve analysis, SHAP workflow, and survival sensitivity analyses. Because the individual-level data cannot be publicly shared due to privacy and data-use restrictions, the code and preprocessing details are provided to support reproducibility within the constraints of data governance.

## 10. 仍需 manuscript 人工修正的地方

- Abstract / Results：若使用重跑結果，請改為 external AUROC 0.898、Brier 0.112 等最新數字。
- HybridXGBRF wording：不要過度宣稱明顯優於 XGBoost；模型命名需處理 notebook 中 Hybrid label 實際對應 XGBClassifier 的問題。
- Methods：補上新版 preprocessing，尤其 continuous/scale z-score 後 missing 補 0。
- Predictor timing：將 model 定位成 early-stay / longitudinal routine-care risk model，或清楚說明 last/max/diff/std predictors 的 measurement window。
- Limitations：補 documentation bias、selection bias、facility linkage limitation、COVID-era limitation、longitudinal predictors potential leakage。
- Figure captions：Calibration plot caption 加 predicted-risk histogram；SHAP caption 加 model-agnostic final predicted probability output。
- Prose cleanup：移除 em dash，補 Taiwan LTCF context 與 international generalizability。
