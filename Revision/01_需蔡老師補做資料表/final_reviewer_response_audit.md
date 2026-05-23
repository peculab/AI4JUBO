# Final reviewer response audit

更新日期：2026-05-18 晚間重跑後  
工作區：`../..`

本文是最後一次總核對紀錄，用於確認 reviewer response、程式碼、重跑結果、圖表與交付資料夾是否一致。

## 1. 總結判定

目前可完整回覆 Reviewer BK 與 Reviewer W 的主要技術性意見。已完成的實質補強包括：

- 缺失值補值邏輯已改成與原始投稿文字一致。
- Internal CV、external validation、calibration、threshold analysis、DCA、SHAP、missingness/facility missingness、included-vs-excluded comparison、survival sensitivity analyses 已重跑或整理。
- `RESULTS/tables`、`RESULTS/figures` 與 `Revision/01_需蔡老師補做資料表` 內主要交付檔案已核對存在。
- `Revision/response_mapping_for_01_請蔡老師協助的部分.md`、delivery folder 內同名副本、`response_and_file_mapping.md` hash 一致。
- `Revision/reviewer_BK_W_coverage_check.md` 與 delivery folder 內副本 hash 一致。
- 已掃描主要回覆文件與 README，未再發現舊版核心數字 `0.878`、`0.887` 等殘留。

## 2. 缺失值補值與程式碼一致性

最終主模型 preprocessing：

- Candidate predictors missingness >= 30% in development cohort：排除。
- Binary/count variables：missing 補 `0`。
- Continuous/scale variables：以 development cohort 或 CV training fold mean/SD 做 z-score，missing standardized value 補 `0`，等同原始尺度補 mean。
- External validation：固定使用 development cohort fitted preprocessing，不用 external cohort 重估 mean/SD。
- Internal CV：每 fold 只用 training fold fitted preprocessing，再套到 validation fold。

程式對應：

- `revision_generate_results.py::fit_preprocessing()`
- `revision_generate_results.py::apply_preprocessing()`
- `revision_generate_results.py::prepare_xy()`
- `survival_generate_results.py` 的 ML risk model 使用同一套 preprocessing。

需誠實註明：

- `survival_generate_results.py::multivariable_cox_baseline()` 仍使用 median imputation，但這是 Cox baseline sensitivity 的獨立流程，不是最終 ML 主模型補值策略。
- `revision_generate_results.py` 內仍保留一段 `SimpleImputer(strategy="median")` exception fallback；目前正式重跑路徑已先完成 preprocessing，該段不是主模型預設補值策略。

## 3. 最新核心數字

| 項目 | 最新結果 |
| --- | --- |
| Development cohort | n = 23,901 |
| Temporal external validation cohort | n = 6,216 |
| Selected predictors | 29 predictors |
| Internal CV HybridXGBRF AUROC | 0.888 (95% CI 0.883-0.893) |
| Internal CV HybridXGBRF Brier | 0.101 (95% CI 0.099-0.103) |
| External HybridXGBRF AUROC | 0.898 (95% CI 0.889-0.906) |
| External HybridXGBRF accuracy | 0.848 (95% CI 0.841-0.856) |
| External HybridXGBRF precision / PPV | 0.855 (95% CI 0.836-0.875) |
| External HybridXGBRF recall / sensitivity | 0.567 (95% CI 0.545-0.590) |
| External HybridXGBRF specificity | 0.961 (95% CI 0.956-0.967) |
| External HybridXGBRF NPV | 0.847 (95% CI 0.840-0.854) |
| External HybridXGBRF F1 | 0.682 (95% CI 0.663-0.701) |
| External HybridXGBRF Brier | 0.112 (95% CI 0.108-0.116) |
| External calibration intercept | 0.516 (95% CI 0.447-0.597) |
| External calibration slope | 1.271 (95% CI 1.217-1.342) |
| External O/E ratio | 1.128 (95% CI 1.108-1.149) |
| Paired bootstrap external Hybrid vs XGB AUROC difference | 0.0036 (95% CI 0.0012-0.0060), P = 0.003 |
| Paired bootstrap internal Hybrid vs XGB AUROC difference | 0.0032 (95% CI 0.0018-0.0048), P < 0.001 |
| Survival external Harrell C-index | 0.860 |
| External Cox HR per 0.10 ML risk increase | 1.641 (95% CI 1.614-1.669) |
| External time-dependent AUC, months 1-5 | 0.916, 0.924, 0.918, 0.908, 0.902 |
| External KM risk groups | low 2.5%, medium 17.2%, high 66.3% |

## 4. Reviewer BK coverage

| # | 狀態 | 可回覆內容 | 對應檔案 |
| --- | --- | --- | --- |
| BK-1 measurement window | 可回覆，manuscript 需補文字 | 補 predictor measurement window；說明 first/last/max/diff/std 是 routine-care longitudinal features，不應寫成 pure admission-time prediction。 | `selected_features.xlsx`; `predictor_measurement_window_table.xlsx` |
| BK-2 Hybrid vs XGB | 已回覆 | 已補 paired bootstrap。差異統計顯著但很小，文字需降調為 comparable high discrimination / small AUROC difference。 | `paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`; `paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx` |
| BK-3 95% CI | 已回覆 | Internal/external/subgroup performance 與 calibration metrics 皆有 95% CI。 | `table3_internal_cv_performance_with_ci.xlsx`; `table4_external_validation_full_with_ci.xlsx`; `table5_subgroup_performance_with_ci.xlsx`; `calibration_metrics_external_hybridxgbrf_with_ci.xlsx` |
| BK-4 recall limited | 已回覆，manuscript 需降調 | 以 threshold table 說明 low threshold 可提升 sensitivity，高 threshold 可提升 PPV/specificity。 | `threshold_tradeoff_external_hybridxgbrf.xlsx`; `external_validation_confusion_matrices.png` |
| BK-5 AUROC insufficient | 已回覆 | 已補 PPV/NPV、threshold tradeoff、DCA、confusion matrix。 | `decision_curve_external_validation.xlsx`; `decision_curve_external_validation.png`; `threshold_tradeoff_external_hybridxgbrf.xlsx` |
| BK-6 baseline comparators | 部分回覆 | 若正式臨床分數缺必要欄位，需說明無法可靠計算；以 survival/Cox sensitivity 補強。 | `survival_c_index.xlsx`; `survival_cox_ml_risk_score.xlsx`; `survival_time_dependent_auc.xlsx` |
| BK-7 time-to-event | 已回覆，作 sensitivity | 已補 C-index、Cox ML risk score、time-dependent AUC、KM risk groups、log-rank。180-day AUC NA 需說明 administrative horizon 下無 event-free controls。 | `survival_*` tables/figures |
| BK-8 missingness/documentation bias | 已回覆 | 補 missingness indicator、facility missingness；主模型補值改為 binary/count 0、continuous/scale mean-equivalent。 | `missingness_indicator_*`; `facility_missingness_*`; `missing_value_imputation_code_comparison.md` |
| BK-9 selection bias | 已回覆，需註明限制 | 已補 included 30,117 vs insufficient-follow-up excluded 19,756 comparison、exit reasons、excluded region/size。Included resident-level facility linkage 仍有限。 | `included_vs_excluded_insufficient_followup_with_p.xlsx`; `excluded_*` tables |
| BK-10 SHAP final model | 已回覆，命名需修 | Revised SHAP 使用 KernelSHAP on final predict_proba output。HybridXGBRF vs XGBClassifier naming ambiguity 需在 manuscript 處理。 | `shap_feature_importance.xlsx`; `shap_feature_importance.png`; `model_identity_note.txt` |
| BK-11 calibration metrics | 已回覆 | 已補 intercept、slope、O/E、Brier、risk-decile calibration 與 95% CI。 | `calibration_metrics_external_hybridxgbrf_with_ci.xlsx`; `risk_decile_calibration_external_hybridxgbrf.xlsx` |
| BK-12 reproducibility | 已回覆 | scripts、random seeds、package versions、preprocessing details 可提供；個資資料不可公開。 | `README.md`; `RESULTS/README_RESULTS.md`; `package_versions.txt`; scripts |
| BK-13 dynamic prediction literature | 文字型回覆 | Discussion/Future Work 補 dynamic prediction、longitudinal updating、time-to-event framing。 | manuscript text; `survival_*` |

## 5. Reviewer W coverage

| # | 狀態 | 可回覆內容 | 對應檔案 |
| --- | --- | --- | --- |
| W-1 Taiwan LTCF context | manuscript 需補文字 | 補台灣 LTCF 情境與國際外推限制。 | manuscript Discussion |
| W-2 COVID-era | manuscript 需補 limitation | 2024 temporal validation 仍有 high discrimination，但未做 dedicated COVID/calendar-period analysis。 | `table4_external_validation_*`; `survival_*` |
| W-3 actionable prediction | 已有分析，需降調文字 | 改成 may support risk stratification and clinical review，不寫 standalone decision-making。 | `threshold_tradeoff_external_hybridxgbrf.xlsx`; DCA |
| W-4 excluded residents | 已回覆 | 已補 baseline、p values、exit reasons、facility size/region for excluded residents，並說明 included resident-level facility linkage limitation。 | included/excluded/excluded facility tables |
| W-5 predictor selection | manuscript/appendix 需補文字 | 補 candidate predictor rationale、availability、measurement timing、retention/exclusion logic。 | `selected_features.xlsx`; `predictor_measurement_window_table.xlsx` |
| W-7 em dash cleanup | manuscript 文字工作 | Word 全文 search em dash 並改成 comma/parenthesis/semicolon。 | manuscript docx |
| W-8 calibration histogram | 已回覆 | Calibration plots 已含 predicted-risk histogram。 | `internal_cv_calibration_with_histogram.png`; `external_validation_calibration_with_histogram.png` |
| W-9 ADL imputation/exclusion | 已可回覆，Methods 需澄清 | 區分 ADL incomplete assessment exclusion 與納入 cohort 後 retained ADL-derived features 的 mean-equivalent imputation。 | `missing_value_imputation_code_comparison.md`; `excluded_adl_missing_baseline_summary.xlsx` |

## 6. 圖表產出確認

下列 PNG 圖檔已確認存在於 `RESULTS/figures` 且已同步到 `Revision/01_需蔡老師補做資料表`：

- `internal_cv_roc.png`
- `external_validation_roc.png`
- `internal_cv_calibration_with_histogram.png`
- `external_validation_calibration_with_histogram.png`
- `internal_cv_confusion_matrices.png`
- `external_validation_confusion_matrices.png`
- `decision_curve_internal_cv.png`
- `decision_curve_external_validation.png`
- `shap_feature_importance.png`
- `survival_time_dependent_auc.png`
- `survival_km_external_by_risk_group.png`
- `training_facility_region_size_20260516.png`
- `excluded_facility_region_size_20260516.png`

抽查 hash 結果：主要 PNG 圖檔在 `RESULTS/figures` 與 delivery folder 內容一致。

## 7. 表格產出確認

下列主要 XLSX 表格已確認存在於 `Revision/01_需蔡老師補做資料表`，且抽查 hash 與 `RESULTS/tables` 一致：

- `table3_internal_cv_performance_with_ci.xlsx`
- `table4_external_validation_full_with_ci.xlsx`
- `table4_external_validation_paper_friendly.xlsx`
- `table5_subgroup_performance_with_ci.xlsx`
- `calibration_metrics_external_hybridxgbrf_with_ci.xlsx`
- `risk_decile_calibration_external_hybridxgbrf.xlsx`
- `threshold_tradeoff_external_hybridxgbrf.xlsx`
- `decision_curve_external_validation.xlsx`
- `decision_curve_internal_cv.xlsx`
- `paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx`
- `paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`
- `missingness_indicator_development.xlsx`
- `missingness_indicator_external.xlsx`
- `facility_missingness_development.xlsx`
- `facility_missingness_external.xlsx`
- `included_vs_excluded_insufficient_followup_with_p.xlsx`
- `excluded_residents_baseline_summary.xlsx`
- `excluded_adl_missing_baseline_summary.xlsx`
- `excluded_exit_reason_summary.xlsx`
- `excluded_residents_by_facility_size_20260516.xlsx`
- `excluded_residents_by_region_20260516.xlsx`
- `selected_features.xlsx`
- `shap_feature_importance.xlsx`
- `survival_c_index.xlsx`
- `survival_cox_ml_risk_score.xlsx`
- `survival_time_dependent_auc.xlsx`
- `survival_km_external_risk_group_summary.xlsx`
- `survival_logrank_external_risk_groups.xlsx`

## 8. 投稿前仍需人工套回 Word 的項目

- Abstract / Results / Tables / Figure captions：更新所有 performance numbers，尤其 external AUROC `0.898`、Brier `0.112`、calibration intercept `0.516`、slope `1.271`、O/E `1.128`。
- HybridXGBRF wording：不要宣稱 clinically large superiority；需處理 HybridXGBRF label 與 XGBClassifier 實作命名 ambiguity。
- Methods：補上新版 missing-data handling 與 CV/external preprocessing 防 leakage 的說明。
- Predictor timing：避免 pure admission-time prediction；改寫為 early-stay / longitudinal routine-care risk model，或清楚說明 measurement window。
- Limitations：加入 documentation bias、selection bias、facility linkage limitation、COVID-era limitation、longitudinal predictors potential leakage、international generalizability。
- SHAP caption：寫 model-agnostic KernelSHAP on final predicted probability output。
- Calibration caption：寫 calibration curve plus predicted-risk histogram。
- 全文搜尋 em dash 並改成較正式的標點。

## 9. 建議 response letter 總回覆段落

We thank the reviewers for identifying areas requiring stronger validation, calibration, and transparency. In response, we revised the preprocessing code to match the missing-data strategy described in the manuscript, regenerated all main and sensitivity outputs, and added confidence intervals for internal and external performance metrics, paired bootstrap comparisons with XGBoost, threshold-specific PPV/NPV and sensitivity/specificity tradeoffs, decision-curve analysis, calibration plots with predicted-risk histograms, numerical calibration metrics with bootstrap confidence intervals, risk-decile calibration, missingness and facility-level missingness summaries, included-versus-excluded resident comparisons, facility size and region characterization, model-agnostic SHAP explanations of the final predicted probability output, and survival sensitivity analyses using Cox models, Harrell's C-index, time-dependent AUC, and Kaplan-Meier curves by predicted-risk strata.

We also revised the manuscript text to clarify predictor measurement windows, preprocessing and missing-data handling, model reproducibility, and the intended clinical role of the model. We toned down claims of HybridXGBRF superiority and actionable prediction, describing the model as a tool that may support risk stratification and clinical review. We expanded the Limitations to address selection bias from excluded residents, documentation bias, facility-level heterogeneity, incomplete facility linkage, potential leakage from longitudinal predictors, pandemic-era effects, and generalizability beyond Taiwanese long-term care facilities.
