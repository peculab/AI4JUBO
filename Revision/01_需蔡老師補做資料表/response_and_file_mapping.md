# 補做資料表對應回覆與檔名清單

來源資料夾：

- `C:\AI4JUBO\Revision\01_需蔡老師補做資料表`

可用結果資料夾：

- 表格：`C:\AI4JUBO\RESULTS\tables`
- 圖檔：`C:\AI4JUBO\RESULTS\figures`

## 2026-05-18 完成狀態

這個資料夾內原本需要補做的兩份 Word 表格已完成：

- `COMPLETED_95CI_tables.docx`
  - 已補 Table 3 internal CV performance 的 95% CI。
  - 已補 Table 4 external validation performance 的 95% CI。
  - 已補 Table 5 subgroup performance 的 95% CI。
  - Table 4 另補上 Brier score 95% CI row。
- `COMPLETED_Development_Cohort_included_vs_excluded_insufficient_followup.docx`
  - 已把原本 `n = ???` 的 excluded subjects due to insufficient follow-up 補為 `n = 19,756`。
  - 已補 age、age group、sex、DNR、ADL first/max/last、GCS、dentures、tube feeding、respiratory support、body weight、hospitalizations 等目前資料可產出的欄位。
  - 已補 p 值；CIRS-G 與 falls 因目前 insufficient-follow-up excluded-resident 資料不可用，保留 `NA`。

同一資料夾也已放入可直接交付的 Excel 表：

- `included_vs_excluded_insufficient_followup.xlsx`
- `included_vs_excluded_insufficient_followup_with_p.xlsx`
- `development_cohort_plus_current_excluded_insufficient_followup.xlsx`
- `table3_internal_cv_performance_with_ci.xlsx`
- `table4_external_validation_full_with_ci.xlsx`
- `table4_external_validation_paper_friendly.xlsx`
- `table5_subgroup_performance_with_ci.xlsx`
- `calibration_metrics_external_hybridxgbrf_with_ci.xlsx`
- `facility_region_size_overview_20260516.xlsx`
- `excluded_residents_by_facility_size_20260516.xlsx`
- `excluded_residents_by_region_20260516.xlsx`

也已放入可直接交付的機構區域/大小圖檔：

- `training_facility_region_size_20260516.png`
- `excluded_facility_region_size_20260516.png`

新版與投稿版數據不同的原因已整理在：

- `C:\AI4JUBO\Revision\20260516\data_discrepancy_and_table_notes.md`

最重要的判讀：新版 included/excluded comparison 的 included 端是 development + external validation 合併後 `N = 30,117`，而投稿版 Development Cohort appendix 是 development cohort only `N = 23,901`；此外新版 binary/categorical 百分比使用非缺失分母，因此 tube feeding、respiratory support 等百分比會和投稿版不同。

2026-05-18 晚間補入 `Revision\20260516\機構區域與大小.xlsx` 後，已進一步補上機構區域/大小總覽。此檔可支持 training/development facilities 的機構層級區域/大小描述，以及 excluded residents 的區域與機構大小分層；但它不是 resident-level included cohort linkage，因此仍不能做完整 resident-level included vs excluded facility-size/region statistical comparison。

重要版本提醒：

- 這批 `RESULTS` 是 2026-05-08 從原 notebook 的 Google Sheets 重新跑出來的結果。
- 重新跑出的 external validation 指標與投稿版數字不完全相同。例如重新跑出主模型 external AUROC 約 0.887，但投稿版摘要原本為 0.878。若要維持投稿版數字，請不要直接替換主文數值；若決定採用重跑版，則 Table 4、Results、Abstract 需同步更新。
- 原 notebook 中 `"HybridXGBRF (Our Approach)"` 實際似乎是 `XGBClassifier` 物件標籤，建議回覆中避免過度宣稱 hybrid 明顯優於 XGBoost。

---

## A. `(Revision)95CI.docx`

這份檔案要補三件事：

1. Table 3 internal CV performance 的 95% CI。
2. Table 4 external validation performance 的 95% CI。
3. Table 5 subgroup performance 的 95% CI。

### 對應檔案

Table 3：

- `RESULTS\tables\table3_internal_cv_performance_with_ci.xlsx`
- `RESULTS\tables\table3_internal_cv_performance_with_ci.csv`

Table 4：

- `RESULTS\tables\table4_external_validation_paper_friendly.xlsx`
- `RESULTS\tables\table4_external_validation_full_with_ci.xlsx`

Table 5：

- `RESULTS\tables\table5_subgroup_performance_with_ci.xlsx`
- `RESULTS\tables\table5_subgroup_performance_with_ci.csv`

可搭配圖：

- `RESULTS\figures\internal_cv_roc.png`
- `RESULTS\figures\external_validation_roc.png`
- `RESULTS\figures\internal_cv_calibration_with_histogram.png`
- `RESULTS\figures\external_validation_calibration_with_histogram.png`
- `RESULTS\figures\internal_cv_confusion_matrices.png`
- `RESULTS\figures\external_validation_confusion_matrices.png`

### 可貼到 Response Letter 的回答

Reviewer comment:

> Confidence intervals are missing for several important results. Recall, precision, F1, Brier score, calibration measures, and subgroup performance should also include uncertainty estimates.

Response:

Thank you for this important suggestion. We have revised the performance reporting to include uncertainty estimates for the main discrimination, classification, and calibration metrics. Specifically, we added 95% confidence intervals for AUROC, accuracy, precision, recall/sensitivity, specificity, F1 score, and Brier score in the internal cross-validation and temporal external validation analyses. We also added 95% confidence intervals for subgroup performance estimates of the selected model. These revised estimates are now presented in the revised Table 3, Table 4, and Table 5.

Manuscript change:

We revised the model performance tables to report point estimates with 95% confidence intervals. The subgroup performance table was also updated to include 95% confidence intervals for accuracy, precision, recall, F1 score, and AUROC.

### 補入論文時可寫的 Results 句子

Internal cross-validation:

In internal cross-validation, the selected model achieved an AUROC of 0.875 (95% CI 0.870-0.880), with accuracy 0.850 (95% CI 0.846-0.853), precision 0.774 (95% CI 0.760-0.787), recall 0.451 (95% CI 0.438-0.464), F1 score 0.570 (95% CI 0.557-0.582), and Brier score 0.109 (95% CI 0.107-0.110).

External validation, if using regenerated RESULTS:

In the temporal external validation cohort, the selected model achieved an AUROC of 0.887 (95% CI 0.878-0.895), with accuracy 0.828 (95% CI 0.821-0.835), precision 0.856 (95% CI 0.835-0.876), recall 0.482 (95% CI 0.460-0.505), F1 score 0.617 (95% CI 0.596-0.636), and Brier score 0.122 (95% CI 0.118-0.125).

Subgroups, if using regenerated RESULTS:

Across prespecified subgroups, AUROC remained high in residents with ADL improvement, female residents, male residents, and both age strata, while performance was lower in the ADL decline subgroup, reflecting the smaller sample size and lower number of events in that subgroup.

---

## B. HybridXGBRF vs XGBoost paired comparison

這不是原 `95CI.docx` 的表格內容，但它直接回應 Reviewer BK 對「HybridXGBRF 是否真的優於 XGBoost」的質疑。

### 對應檔案

Internal paired bootstrap：

- `RESULTS\tables\paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`

External paired bootstrap：

- `RESULTS\tables\paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx`

### 目前重跑結果重點

Internal:

- Hybrid AUROC = 0.8751
- XGBoost AUROC = 0.8740
- Difference = 0.0011
- 95% CI = -0.0001 to 0.0025
- Bootstrap P = 0.064

External:

- Hybrid AUROC = 0.8868
- XGBoost AUROC = 0.8864
- Difference = 0.0004
- 95% CI = -0.0017 to 0.0026
- Bootstrap P = 0.694

### 可貼到 Response Letter 的回答

Reviewer comment:

> The improvement of HybridXGBRF over XGBoost is too small to support a strong superiority claim. Paired bootstrap testing or another paired statistical comparison is needed.

Response:

We agree with the reviewer. We performed paired bootstrap comparisons of AUROC between the selected model and XGBoost using paired predictions from the same participants. The AUROC difference was small and not statistically significant in the external validation cohort. Therefore, we revised the manuscript to avoid claiming clear superiority of HybridXGBRF over XGBoost. Instead, we now describe the tree-based models as showing comparable high discrimination, with the selected model retained because of its overall balance of discrimination, calibration, and implementation considerations.

Manuscript change:

We softened claims of model superiority in the Abstract, Results, and Discussion, and added the paired bootstrap comparison results to the supplementary material.

---

## C. Threshold-specific tradeoffs, PPV/NPV, and Decision Curve Analysis

這對應 Reviewer BK 的 clinical utility 要求。

### 對應檔案

Threshold table：

- `RESULTS\tables\threshold_tradeoff_external_hybridxgbrf.xlsx`

Decision curve：

- `RESULTS\figures\decision_curve_external_validation.png`
- `RESULTS\tables\decision_curve_external_validation.xlsx`

Internal decision curve，如需要：

- `RESULTS\figures\decision_curve_internal_cv.png`
- `RESULTS\tables\decision_curve_internal_cv.xlsx`

### 可貼到 Response Letter 的回答

Reviewer comment:

> AUROC alone does not show whether the model improves clinical decisions. Decision-curve analysis, PPV/NPV at clinically meaningful thresholds, and threshold-specific tradeoffs should be added.

Response:

Thank you for this suggestion. We added threshold-specific analyses in the temporal external validation cohort. For clinically relevant probability thresholds, we report sensitivity, specificity, PPV, NPV, F1 score, confusion-matrix counts, and net benefit. We also added decision-curve analysis comparing the selected model with alternative models and treat-all/treat-none strategies. These additions clarify how the model behaves when used as a lower-threshold screening tool versus a higher-threshold prioritization tool.

Manuscript change:

We added a threshold-specific tradeoff table and a decision-curve analysis figure to the revised supplementary materials, and we expanded the Results and Discussion to describe the clinical implications of missed high-risk residents and threshold selection.

### 補入論文可寫的句子

At a threshold of 0.20, the selected model prioritized sensitivity, with sensitivity 0.844, specificity 0.736, PPV 0.562, and NPV 0.921. At a threshold of 0.50, the model prioritized positive predictive value, with sensitivity 0.482, specificity 0.968, PPV 0.856, and NPV 0.823. These results indicate that lower thresholds may be more appropriate for screening and prompting clinical review, whereas higher thresholds may be more appropriate for prioritizing limited supportive-care resources.

---

## D. Calibration slope/intercept, O/E ratio, and risk-decile calibration

這對應 Reviewer BK 對 calibration 的要求，也可搭配 Reviewer W 建議在 calibration plot 裡加入 predicted probability distribution。

### 對應檔案

Calibration metrics：

- `RESULTS\tables\calibration_metrics_external_hybridxgbrf.xlsx`

Risk-decile table：

- `RESULTS\tables\risk_decile_calibration_external_hybridxgbrf.xlsx`

Calibration plot with histogram：

- `RESULTS\figures\external_validation_calibration_with_histogram.png`
- `RESULTS\figures\internal_cv_calibration_with_histogram.png`

### 目前重跑結果重點

External calibration metrics:

- Calibration intercept = 0.591
- Calibration slope = 1.323
- Observed deaths = 1781
- Expected deaths = 1538.1
- O/E ratio = 1.158
- Brier score = 0.122

### 可貼到 Response Letter 的回答

Reviewer comment:

> Calibration is not evaluated deeply enough. Calibration slope, intercept, observed/expected ratios, and risk-decile calibration would make the results more clinically interpretable.

Response:

We agree and have expanded the calibration assessment. In addition to calibration plots and Brier scores, we now report calibration intercept, calibration slope, observed/expected ratio, and risk-decile calibration in the external validation cohort. We also revised the calibration figure to include the distribution of predicted probabilities, as suggested by the reviewer, to make the range and density of predictions easier to interpret.

Manuscript change:

We added numerical calibration metrics and a risk-decile calibration table to the supplementary materials and revised the calibration figure to include the distribution of predicted risks.

---

## E. SHAP final predictor explanation

這對應 Reviewer BK 質疑 SHAP 是否解釋 final hybrid output。

### 對應檔案

SHAP table：

- `RESULTS\tables\shap_feature_importance.xlsx`

SHAP figure：

- `RESULTS\figures\shap_feature_importance.png`

### 目前重跑結果 top features

Top features in regenerated KernelSHAP output:

1. Body Weight Change
2. Hospitalizations within 6 Months
3. ADL Last Score
4. ADL Minimum
5. ADL Standard Deviation
6. Male
7. Body Weight (Last)
8. Body Weight (First)

### 可貼到 Response Letter 的回答

Reviewer comment:

> SHAP explanations are not clearly tied to the final hybrid predictor. The authors should clarify whether SHAP explains the blended HybridXGBRF output or only one model component.

Response:

Thank you for identifying this important ambiguity. We clarified the SHAP analysis and revised the interpretability description to specify the model output being explained. To avoid component-level ambiguity, we regenerated SHAP feature-importance estimates using a model-agnostic KernelSHAP approach applied to the final `predict_proba` output. Thus, the revised SHAP results should be interpreted as explanations of the final probability output rather than explanations of only an intermediate component.

Manuscript change:

We revised the Methods and figure caption for the SHAP analysis to specify that model-agnostic SHAP was used to explain the final predicted probability output.

---

## F. `(Revision)Development_Cohort.docx 的副本.docx`

這份是「追蹤未達 6 個月的排除個案」baseline comparison。2026-05-18 已用 `DATA\analysis_data_filtering_out_included_ADL_missing_0514.csv`、`DATA\analysis_data_filtering_out_0514.csv` 與 `DATA\area_size.xlsx` 補完目前資料可產出的欄位。完成版 Word 為 `COMPLETED_Development_Cohort_included_vs_excluded_insufficient_followup.docx`，對應 Excel 為 `development_cohort_plus_current_excluded_insufficient_followup.xlsx`。

### 目前狀態

已完成：

- Excluded subjects due to insufficient follow-up: `n = 19,756`。
- 已填入目前資料可產出的 baseline 欄位與 p 值。
- CIRS-G 與 falls 因目前 insufficient-follow-up excluded-resident 檔案無可用資料，保留為 `NA`。
- 新版 `included_vs_excluded_insufficient_followup.xlsx` 也已補上 `P value` 欄位。

已完成檔案：

- `COMPLETED_Development_Cohort_included_vs_excluded_insufficient_followup.docx`
- `development_cohort_plus_current_excluded_insufficient_followup.xlsx`
- `included_vs_excluded_insufficient_followup_with_p.xlsx`
- `C:\AI4JUBO\Revision\20260516\data_discrepancy_and_table_notes.md`

資料限制與判讀：

- 本次 comparison 的 included 端若使用 `included_vs_excluded_insufficient_followup`，是 development + external validation 合併後 `N = 30,117`；投稿版 Development Cohort appendix 是 development cohort only `N = 23,901`。
- 新版 binary/categorical 百分比使用非缺失分母，因此與投稿版使用 cohort total 分母的欄位會不同。
- facility size / region 可用於 excluded residents，但 included cohort 缺穩定可合併的 `dbname`，所以 included facility-size/region 仍需在回覆與限制中說明。

建議輸出檔名：

- `RESULTS\tables\included_vs_excluded_insufficient_followup.xlsx`

### 可貼到 Response Letter 的回答，若資料補得出來

Reviewer comment:

> The exclusion criteria may introduce selection bias. The authors should compare included and excluded residents more fully and discuss how exclusions may affect mortality prediction.

Response:

We agree that exclusions may introduce selection bias. We therefore added a baseline comparison between included residents and residents excluded because of insufficient follow-up to ascertain the 6-month outcome. The revised supplementary table reports demographic, functional, clinical, care-related, and facility-region characteristics for both groups. We also expanded the Limitations section to acknowledge that model generalizability may be strongest for residents with sufficiently complete follow-up and outcome ascertainment.

Manuscript change:

We added a supplementary table comparing included residents with excluded residents and expanded the Limitations section to discuss potential selection bias from incomplete follow-up and missing outcome ascertainment.

### 可貼到 Response Letter 的回答，若資料補不出來

Response:

We agree with the reviewer that exclusion due to insufficient follow-up may introduce selection bias. Unfortunately, the available analytic dataset used for model development does not retain sufficient baseline information for residents excluded before outcome ascertainment to support a complete included-versus-excluded comparison. We have therefore revised the Limitations section to state this limitation explicitly and to clarify that the model is intended for residents whose 6-month outcome can be ascertained from the available LTCF records. We also added missingness and documentation-bias analyses for the available analytic cohorts to better characterize data completeness.

Manuscript change:

We expanded the Limitations section to acknowledge the potential for selection bias due to insufficient follow-up and incomplete outcome ascertainment.

---

## G. 建議貼表/貼圖順序

Reviewer response 或 supplement 建議順序：

1. Revised Table 3: `table3_internal_cv_performance_with_ci.xlsx`
2. Revised Table 4: `table4_external_validation_paper_friendly.xlsx`
3. Revised Table 5: `table5_subgroup_performance_with_ci.xlsx`
4. New Table Sx: `paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx`
5. New Table Sx: `threshold_tradeoff_external_hybridxgbrf.xlsx`
6. New Figure Sx: `decision_curve_external_validation.png`
7. New Table Sx: `calibration_metrics_external_hybridxgbrf.xlsx`
8. New Table Sx: `risk_decile_calibration_external_hybridxgbrf.xlsx`
9. Revised/New Figure Sx: `external_validation_calibration_with_histogram.png`
10. New Table/Figure Sx: `shap_feature_importance.xlsx` and `shap_feature_importance.png`
11. Completed Table Sx: `included_vs_excluded_insufficient_followup_with_p.xlsx`
12. Completed Table Sx: `development_cohort_plus_current_excluded_insufficient_followup.xlsx`
13. Completed Word handoff: `COMPLETED_Development_Cohort_included_vs_excluded_insufficient_followup.docx`
