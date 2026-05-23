# Reviewer Response Outputs for 2026-05-23

Source document: `01_請蔡老師協助的部分2.docx`  
New data source: `analysis_data_filtering_out_included_ADL_missing_0523.csv`  
Comparison source for existing manuscript numbers: `C:\AI4JUBO\FINAL`  
Excluded residents in 0523 file: `19,756`

## 數字一致性總原則

中文說明：
本檔案用來整理 reviewer response 可以引用的文字、圖表與表格。為避免修稿時前後數字不一致，原本已存在於 `C:\AI4JUBO\FINAL` 投稿稿件與 Figure 1-7 的主要模型數字，原則上維持 FINAL 版本；本次 20260523 新增的分析只作為補充資料或 reviewer response 的新增敏感度/輔助分析，不直接取代原稿既有 performance 數字，除非後續決定正式重跑並同步更新 manuscript、tables、figures、abstract、results 與 response letter。

English note:
Existing manuscript performance numbers should remain aligned with the submitted FINAL version unless the manuscript is comprehensively updated. Newly generated 20260523 analyses should be labeled as additional or supplementary analyses.

## 與 FINAL 原稿數字核對

以下數字來自 `C:\AI4JUBO\FINAL\Machine_Learning_Model_for_Predicting_6-Month_Mortality_in_LTC_Facilities.docx`，應作為原本既有結果的基準。

| 項目 | FINAL 原稿既有數字 | 本次處理原則 |
| --- | --- | --- |
| Development cohort | n = 23,901; deaths = 5,272; mortality = 22.1% | 維持 |
| External validation cohort | n = 6,216; deaths = 1,781; mortality = 28.7% | 維持 |
| Internal CV HybridXGBRF AUROC | 0.875, 95% CI 0.862-0.889 | 維持原稿；不要改成新表的 0.888 |
| Internal CV HybridXGBRF Brier | Abstract/Results: 0.109; Table 3 rounded: 0.11 | 維持原稿，但見下方注意事項 |
| External HybridXGBRF AUROC | 0.878, 95% CI 0.866-0.889 | 維持原稿；不要改成新表的 0.898 |
| External HybridXGBRF accuracy | 0.851 | 維持 |
| External HybridXGBRF F1 | 0.572 | 維持 |
| Table 3 internal performance | AUROC 約 0.88; precision 0.77; recall/sensitivity 0.45; specificity 0.96; F1 0.57; accuracy 0.85; Brier 0.11 | 維持 |
| Table 4 external performance | AUROC 約 0.88; precision 0.79; recall 0.45; F1 0.57; accuracy 0.85 | 維持 |
| Table 5 subgroup AUROC | ADL improvement 0.90; ADL decline 0.73; female 0.89; age <=85 約 0.88; age >85 0.89; male 0.88 | 維持 |
| FINAL Figure 5-7 SHAP explanation | Hospitalizations within 6 months, ADL impairment/decline, and weight loss are top explanatory domains | 維持 |

### 需要人工確認的 FINAL 內部差異

中文說明：
FINAL 原稿本身有一個需確認之處：摘要與模型結果段落寫 internal Brier score = 0.109，Table 3 四捨五入為 0.11；但 calibration 文字段落另寫 internal Brier score = 0.128、external Brier score = 0.122。這不是 20260523 新分析造成的差異，而是 FINAL 原稿內部既有文字可能未完全一致。建議修稿時擇一套正式數字後同步更新所有段落。

English note:
The submitted FINAL manuscript appears to contain an internal inconsistency for Brier score wording. This should be resolved during manuscript revision, but the 20260523 supplementary outputs should not be used to silently overwrite the existing manuscript numbers.

## 新增分析可引用數字

以下數字是本次 20260523 新增或補充分析，可在 reviewer response 或 supplementary materials 中標註為新增分析。

| 新增項目 | 可引用數字 |
| --- | --- |
| 0523 excluded residents file | n = 19,756 |
| ADL available vs ADL missing within excluded residents | ADL available n = 12,895; ADL missing n = 6,861 |
| Excluded residents age | 76.6 (14.9) years |
| Excluded residents male sex | 9,929 (50.3%) |
| Excluded residents DNR | 5,032 (25.5%) |
| Excluded residents observation days | median 29.0 days, IQR 10.0-66.0 |
| ADL missing group observation days | median 16.0 days, IQR 5.0-41.8 |
| ADL available group observation days | median 37.0 days, IQR 15.0-81.0 |
| Facility-level file | 592 facility identifiers in the 0523 excluded file |
| Facility size ADL missing percent | small 34.8%, medium 32.2%, large 26.3%; unlinked 93.0% |
| Facility region ADL missing percent | north 27.3%, central 42.0%, south 30.6%, east 60.2%; unlinked 93.0% |

## Reviewer W. Excluded Residents and ADL-Missing Sensitivity

中文說明：
這題主要回覆 reviewer 對 selection bias / excluded subjects 的疑慮。本次使用 0523 CSV 重新整理 excluded residents 的基本資料、ADL missing subgroup、變項缺失比例、觀察天數分布、年齡分布，以及 facility-level missingness。因為 0523 CSV 只有 excluded residents，不能單獨用它重算 included vs excluded 的完整 p value；included vs excluded 的比較仍應引用已複製到本資料夾的既有 analytic-cohort 表格。

Response:
We regenerated the excluded-resident descriptive analyses using the May 23 data extract. The excluded file contains 19,756 residents. We summarized demographics, observation time, ADL-derived variables, body weight, care-related variables, variable-level missingness, and facility-level documentation patterns. Because the 0523 file contains the excluded residents only, these outputs should be interpreted as characterization of the excluded population and its ADL-missing subset; direct included-versus-excluded comparisons should use the existing analytic-cohort tables copied into this folder.

Files:
- `excluded_residents_baseline_summary_0523.xlsx`
- `excluded_variable_missingness_top15_0523.png`
- `excluded_observation_days_histogram_0523.png`
- `excluded_age_histogram_0523.png`

## Reviewer W. Included Versus Excluded Residents

中文說明：
這題要特別避免把 0523 excluded-only CSV 當成完整 cohort comparison。若 response letter 需要寫 included vs excluded，應使用 `included_vs_excluded_insufficient_followup.xlsx` 或 `included_vs_excluded_insufficient_followup_with_p.xlsx`。FINAL 原稿既有 development/external cohort 數字仍維持 23,901 與 6,216，不被本次 excluded-only 檔案改動。

Response:
The included-versus-excluded baseline comparison from the prior regenerated analytic results has been copied into this folder for response assembly. The 0523 CSV itself does not include the analytic included cohort, so no new p-value comparison against included residents was recalculated from this file alone.

Files:
- `included_vs_excluded_insufficient_followup.xlsx`
- `included_vs_excluded_insufficient_followup_with_p.xlsx`
- `development_cohort_plus_current_excluded_insufficient_followup.xlsx`
- `excluded_residents_baseline_summary_0523.xlsx`

## Reviewer W. Calibration Plots With Predicted Probability Histogram

中文說明：
Reviewer 要的是 calibration plot 補上 predicted probability histogram。這是圖形呈現的新增，不應改掉 FINAL 原稿已存在的 main performance 數字。若 manuscript 只補圖與 caption，可沿用 FINAL 的 AUROC、accuracy、F1 等既有數字。

Response:
Calibration plots with predicted-risk histograms were generated for the regenerated model outputs and copied here. These figures show calibration curves with the predicted probability distribution underneath. These plots should be described as an added visualization of predicted-risk distribution and calibration, while preserving the existing FINAL manuscript performance numbers unless all model results are formally updated.

Files:
- `internal_cv_calibration_with_histogram.png`
- `external_validation_calibration_with_histogram.png`

## Reviewer BK. Paired Bootstrap Comparison

中文說明：
這是 reviewer 新要求的 paired statistical comparison，屬於新增補充分析。因為 FINAL 原稿原本沒有 paired bootstrap p value，所以可在 response 裡新增描述；但不要因此把 FINAL 原稿的 AUROC 主數字改成重跑表格中的 0.888/0.898。建議文字強調「差異幅度小」，避免過度宣稱 HybridXGBRF 優於 XGBoost。

Response:
Paired bootstrap AUROC comparisons between HybridXGBRF and XGBoost are included as additional analyses. The manuscript response should describe the difference as small and avoid overstating superiority. Existing manuscript AUROC values should remain aligned with FINAL unless the entire model-result set is updated.

Files:
- `paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`
- `paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx`

## Reviewer BK. 95% Confidence Intervals for Performance Metrics

中文說明：
Reviewer 要求 recall、precision、F1、Brier、calibration measures 補 95% CI。這是新增補充表格。若只在 supplementary 補 CI，正文仍可維持 FINAL 原稿的主要點估計與四捨五入數字。若要把 CI 寫進正文，需確認與 FINAL 的原始分析一致；目前 20260523 資料夾中的 CI 表格是重跑後版本，數字不應直接覆蓋 FINAL 既有 performance。

Response:
Performance tables with bootstrap 95% confidence intervals for internal validation, external validation, and subgroup analyses are included. These should be used as supplementary confidence-interval tables. If the manuscript text is not comprehensively regenerated, the existing FINAL point estimates should remain unchanged.

Files:
- `table3_internal_cv_performance_with_ci.xlsx`
- `table4_external_validation_full_with_ci.xlsx`
- `table4_external_validation_paper_friendly.xlsx`
- `table5_subgroup_performance_with_ci.xlsx`

## Reviewer BK. Threshold-Specific Tradeoffs and Decision Curve Analysis

中文說明：
Threshold tradeoff 與 DCA 是新增分析，用來回應 reviewer 對 recall 低、AUROC 不足以說明臨床效益的疑慮。這些可作為 supplementary 新表/新圖，不需要更動 FINAL 原本 Figure 1-7 或 Table 3-5 的主要數字。可在 response letter 說明低 threshold 較適合 screening，高 threshold 較適合資源優先排序。

`01_請蔡老師協助的部分2.docx` 特別寫到：「decision curve / BK4_decision curve / 線條幾乎全水平、model 間差異看不出來 / 可嘗試把 Y 軸 Net Benefit 的 scale 小一點」。因此已重新掃描整個 `Revision` 與 `RESULTS` 中所有 decision-curve/DCA 來源檔，確認真正屬於這類圖的共有兩組：internal CV decision curve 與 external validation decision curve。兩張圖都已重畫並集中放在 `Revision\20260523`。

重畫原則：
- 使用對應的 `decision_curve_*.xlsx` 作為資料來源。
- 將 Net Benefit 的 Y 軸下限縮到 `-2`，避免 Treat All 在高 threshold 掉到約 -13 至 -15 時把模型線壓扁。
- 保留小幅正向上限到 `0.30`，因為 HybridXGBRF、XGBClassifier、RandomForest 等模型的 net benefit 多位於 0 到 0.26 左右；若上限固定為 0，主要模型線會被裁掉，反而無法比較。
- 這是圖形視覺化重畫，不改動 underlying xlsx 數值，也不改動 FINAL 原稿主要 performance 數字。

Response:
Threshold-specific PPV, NPV, sensitivity, specificity, F1, and decision-curve outputs are included. These support wording that lower thresholds may be more appropriate for screening, whereas higher thresholds prioritize PPV/specificity. These are additional decision-analytic outputs and should not replace the existing FINAL model-performance tables unless all outputs are updated together.

Files:
- `threshold_tradeoff_external_hybridxgbrf.xlsx`
- `decision_curve_external_validation.xlsx`
- `decision_curve_external_validation.png`
- `decision_curve_internal_cv.xlsx`
- `decision_curve_internal_cv.png`
- `BK4_decision_curve_external_validation_yaxis_minus2.png`
- `BK4_decision_curve_internal_cv_yaxis_minus2.png`
- `decision_curve_redraw_manifest_20260523.xlsx`
- `decision_curve_redraw_manifest_20260523.md`

## Reviewer BK. Survival Sensitivity Analysis

中文說明：
Survival analysis 是新增 sensitivity analysis。FINAL 原稿主架構是 180-day binary mortality prediction，因此 survival 結果應放在 supplementary 或 response 中作為補充，不宜改寫成主要 survival model。文字需說明資料不是完整連續 time-to-event follow-up，且 180 天行政性 outcome 會限制一般 Cox/time-dependent AUC 的解讀。

Response:
Survival sensitivity outputs are included as supplementary analyses. Because the data structure is based on an administrative 180-day binary outcome and follow-up is not fully continuous after discharge, these should be described as sensitivity analyses, with interval/censoring limitations noted.

Files:
- `survival_cox_ml_risk_score.xlsx`
- `survival_time_dependent_auc.xlsx`
- `survival_c_index.xlsx`
- `survival_km_external_by_risk_group.png`

## Reviewer BK. Missingness Indicators and Facility-Level Missingness

中文說明：
這是本次 0523 CSV 最主要新增內容之一。可用來回覆 documentation bias、ADL missing 是否影響結果、facility-level missingness 是否集中於特定機構大小或區域。注意 0523 檔案死亡標記皆為 0，因為它是 insufficient follow-up / excluded file，因此 facility 表裡的 death/event rate 不能拿來解釋死亡風險，只能解釋 excluded residents 的資料完整性與追蹤不足特徵。

Response:
The 0523 excluded-resident file was used to regenerate variable missingness, ADL-missing subgroup comparisons, and facility-level missingness summaries. These outputs help address whether documentation patterns and facility-level missingness may affect interpretation. Because the 0523 file represents excluded residents, facility-level outcome rates from this file should not be interpreted as model outcome performance.

Files:
- `excluded_residents_baseline_summary_0523.xlsx`
- `facility_missingness_and_outcome_0523.xlsx`
- `excluded_adl_missingness_by_facility_size_0523.png`
- `excluded_adl_missingness_by_region_0523.png`
- `missingness_indicator_development.xlsx`
- `missingness_indicator_external.xlsx`
- `facility_missingness_development.xlsx`
- `facility_missingness_external.xlsx`

## Reviewer BK. Multiple Imputation

中文說明：
目前 20260523 沒有新增 multiple imputation model 結果。建議 response 不要宣稱已完成 multiple imputation；可說明主分析採用 prespecified pragmatic imputation，並補上 missingness indicator / facility-level missingness 作為 documentation-bias 檢查。若 reviewer 強烈要求 MI，需要另外正式新增 MI sensitivity analysis，且要同步產生表格。

Response:
The response should clarify that the primary model uses development-fitted preprocessing with binary/count missing values imputed as 0 and continuous/scale values handled as mean-equivalent imputation after standardization. Multiple imputation can be discussed as a possible sensitivity option if added later, but no multiple-imputation model result is generated from the 0523 excluded-only file.

Files:
- `selected_features.xlsx`
- `excluded_residents_baseline_summary_0523.xlsx`

## Reviewer BK. SHAP Explanation

中文說明：
FINAL 原稿 Figure 5-7 已有 SHAP beeswarm、forest、force plots；本次只需在 response 補充說明 SHAP 是解釋 final predicted probability 的工具，並維持 FINAL 原稿對重要因子的描述：recent hospitalizations、ADL impairment/decline、weight loss。不要把新表格的 feature ordering 改寫成與 FINAL 圖不一致的敘述。

Response:
The SHAP feature-importance output for the final model has been copied here and can be cited as model explanation of final predicted probabilities. The response should remain consistent with the FINAL manuscript interpretation, emphasizing recent hospitalizations, ADL impairment or decline, and weight loss as the main explanatory domains.

Files:
- `shap_feature_importance.png`
- `shap_feature_importance.xlsx`

## Reviewer BK. Calibration Metrics and Risk-Decile Calibration

中文說明：
Calibration metrics 與 risk-decile calibration 是 reviewer 新要求的補充數值。可新增到 supplementary。若正文不全面更新模型 performance，請避免把 calibration metrics 中的 regenerated Brier `0.112` 直接拿去取代 FINAL 文字中的 Brier 數字。FINAL 內部 Brier 差異應另行統一。

Response:
Calibration metrics and risk-decile calibration outputs are included. These support adding intercept, slope, O/E ratio, Brier score, and decile-level calibration summaries as supplementary calibration evidence. Brier score wording should be harmonized with the FINAL manuscript before submission.

Files:
- `calibration_metrics_external_hybridxgbrf_with_ci.xlsx`
- `risk_decile_calibration_external_hybridxgbrf.xlsx`

## Reproducibility Details

中文說明：
本次 0523 新增輸出由 `generate_0523_revision_outputs.py` 產生。若要回覆 code、random seed、package versions、preprocessing 細節，可引用 project scripts、README、model candidate appendix 與 preprocessing/missing-data 說明。這部分通常不涉及改動 FINAL 圖表數字。

Response:
The generated 0523 files were created by `generate_0523_revision_outputs.py`. The copied model outputs come from the regenerated revision output folder. Random seeds, package versions, preprocessing, and source code details should be cited from the project scripts and README materials.

Copied prior-output files:
`included_vs_excluded_insufficient_followup.xlsx`, `included_vs_excluded_insufficient_followup_with_p.xlsx`, `development_cohort_plus_current_excluded_insufficient_followup.xlsx`, `internal_cv_calibration_with_histogram.png`, `external_validation_calibration_with_histogram.png`, `paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`, `paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx`, `table3_internal_cv_performance_with_ci.xlsx`, `table4_external_validation_full_with_ci.xlsx`, `table4_external_validation_paper_friendly.xlsx`, `table5_subgroup_performance_with_ci.xlsx`, `threshold_tradeoff_external_hybridxgbrf.xlsx`, `decision_curve_external_validation.png`, `decision_curve_external_validation.xlsx`, `survival_cox_ml_risk_score.xlsx`, `survival_time_dependent_auc.xlsx`, `survival_km_external_by_risk_group.png`, `survival_c_index.xlsx`, `missingness_indicator_development.xlsx`, `missingness_indicator_external.xlsx`, `facility_missingness_development.xlsx`, `facility_missingness_external.xlsx`, `shap_feature_importance.png`, `shap_feature_importance.xlsx`, `calibration_metrics_external_hybridxgbrf_with_ci.xlsx`, `risk_decile_calibration_external_hybridxgbrf.xlsx`, `selected_features.xlsx`.
