# Reviewer BK / Reviewer W point-to-point coverage check

來源檔案：

- `C:\AI4JUBO\Revision\00_Reviewer BK_External Peer-Review.docx`
- `C:\AI4JUBO\Revision\00_Reviewer W_External Peer-Review.docx`
- `C:\AI4JUBO\Revision\01_請蔡老師協助的部分.docx`

目前可用輸出：

- 表格：`C:\AI4JUBO\RESULTS\tables`
- 圖檔：`C:\AI4JUBO\RESULTS\figures`
- 主分析程式：`C:\AI4JUBO\revision_generate_results.py`
- Survival 分析程式：`C:\AI4JUBO\survival_generate_results.py`
- 詳細 mapping：`C:\AI4JUBO\Revision\response_mapping_for_01_請蔡老師協助的部分.md`

## 最新總結

2026-05-13 更新後，原本最明確的資料缺口「included vs excluded because of insufficient follow-up」已經有可用表格回應；排除者 exit reason、機構核定床數/大小/區域分層、calibration metrics 95% CI 也已補出。

目前 reviewers 的統計與補圖表要求大多可 point-to-point 回覆。仍需特別小心的部分不是沒有 results，而是需要在 manuscript 文字中明確降調或承認限制：

- predictor measurement window / possible leakage：已新增 `predictor_measurement_window_table.xlsx` 支援回答，但若 manuscript 宣稱 admission-time prediction，`last/max/diff/std/change` 類 longitudinal predictors 仍需清楚界定 prediction index，或改寫成 early-stay/longitudinal risk updating。
- formal clinical risk score baseline：目前有 logistic/RF/XGB/Cox/survival sensitivity，但沒有 MDS-CHESS、ADEPT 等正式臨床風險分數。若資料欄位不足，回覆信應明確說明無法合理計算，並把 Cox/survival sensitivity 作為替代補強。
- Taiwan LTCF context、COVID-era impact、actionable claim、continuous prediction literature、刪除 em dash：這些是正文與引用問題，不是 results 可以單獨完成的問題。
- alternative imputation sensitivity：目前有 missingness indicator、facility missingness、preprocessing clarification；若 reviewer 強烈要求「測試不同插補策略」，仍可再補一張 sensitivity table，但目前可先用現有結果與文字澄清回應。

## Reviewer BK point-to-point

| # | Reviewer BK comment | 可用 results / files | 覆蓋狀態 | 建議回覆方向 |
|---|---|---|---|---|
| BK-1 | Every feature should be linked to a clear measurement window. | `RESULTS\tables\selected_features.xlsx`; `RESULTS\tables\predictor_measurement_window_table.xlsx`; `RESULTS\tables\shap_feature_importance.xlsx` | **可回答，但需正文補強** | 新增 predictor measurement appendix，逐一列出 demographic、first recorded、last recorded、change/variability、prior utilization 等 timing。需承認 `last/max/diff/std/BW_diff` 屬 longitudinal predictors，若用於 admission-time prediction 有 leakage risk；建議 manuscript 改寫成 early-stay/longitudinal routine-care risk model，或補 early-window-only sensitivity。 |
| BK-2 | HybridXGBRF improvement over XGBoost is too small; need paired comparison. | `paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`; `paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx` | **已充分回答** | External ΔAUROC = 0.0004，95% CI -0.0017 to 0.0026，bootstrap P = 0.716。回覆應改成 tree-based models had comparable high discrimination，避免宣稱 Hybrid 顯著優於 XGBoost。 |
| BK-3 | Need 95% CI for recall, precision, F1, Brier, calibration, subgroup performance. | `table3_internal_cv_performance_with_ci.xlsx`; `table4_external_validation_full_with_ci.xlsx`; `table4_external_validation_paper_friendly.xlsx`; `table5_subgroup_performance_with_ci.xlsx`; `calibration_metrics_external_hybridxgbrf_with_ci.xlsx` | **已充分回答** | 可直接回覆已補 AUROC、accuracy、precision、recall/sensitivity、specificity、PPV、NPV、F1、Brier、subgroup performance 與 calibration metrics 的 95% CI。 |
| BK-4 | Recall is limited; discuss missed high-risk residents and alternative thresholds. | `threshold_tradeoff_external_hybridxgbrf.xlsx`; `external_validation_confusion_matrices.png`; `decision_curve_external_validation.png` | **已充分回答，需正文降調** | Threshold 0.20 sensitivity 0.844、NPV 0.921；threshold 0.50 PPV 0.856、specificity 0.968。文字上說明低 threshold 適合作為 screening/review trigger，高 threshold 適合 resource prioritization；不能寫成 standalone decision tool。 |
| BK-5 | AUROC alone does not show clinical utility; add DCA, PPV/NPV, threshold tradeoffs. | `decision_curve_external_validation.xlsx`; `decision_curve_external_validation.png`; `threshold_tradeoff_external_hybridxgbrf.xlsx` | **已充分回答** | 可回覆已新增 decision curve analysis、treat-all/treat-none comparison、threshold-specific sensitivity/specificity/PPV/NPV/F1/confusion counts/net benefit。 |
| BK-6 | Baselines incomplete; add clinical risk-score baselines or survival/time-to-event baselines. | `table3_internal_cv_performance_with_ci.xlsx`; `table4_external_validation_full_with_ci.xlsx`; `survival_c_index.xlsx`; `survival_cox_ml_risk_score.xlsx`; `survival_time_dependent_auc.xlsx`; `survival_km_external_risk_group_summary.xlsx`; `survival_km_external_by_risk_group.png` | **部分回答** | Survival sensitivity 已補強：external Harrell C-index 0.849；risk-score Cox HR per 0.10 increase 1.65；1-5 month AUC 0.907/0.912/0.901/0.893/0.889。正式 clinical risk score 若欄位不足，需在回覆信明確說明 cannot be reliably computed from available routine LTCF variables。 |
| BK-7 | Binary 6-month outcome ignores censoring/time-to-death; need survival modeling or sensitivity analyses. | `survival_c_index.xlsx`; `survival_cox_ml_risk_score.xlsx`; `survival_time_dependent_auc.xlsx`; `survival_km_external_risk_group_summary.xlsx`; `survival_logrank_external_risk_groups.xlsx`; `survival_km_external_by_risk_group.png`; `survival_time_dependent_auc.png` | **已充分回答，需限制說明** | 可回覆已新增 survival sensitivity analyses。需說明這是 sensitivity analysis，不取代 primary binary 180-day model；6-month cumulative/dynamic AUC 因 180-day administrative censoring 無 event-free controls，因此不可估。 |
| BK-8 | Missing binary/count variables coded as absence may mix absence and documentation missingness; test missingness, imputation, facility-level missingness. | `missingness_indicator_development.xlsx`; `missingness_indicator_external.xlsx`; `facility_missingness_development.xlsx`; `facility_missingness_external.xlsx`; `facility_size_missingness_and_outcome.xlsx`; `facility_region_missingness_and_outcome.xlsx`; `included_vs_excluded_insufficient_followup.xlsx` | **可回答，若要更強可補 imputation sensitivity** | 現有 results 可回應 missingness patterns、facility-level heterogeneity、排除者機構大小/區域分層。若 reviewer 要求「不同插補策略」實證比較，仍可另補 `alternative_imputation_performance.xlsx`。 |
| BK-9 | Exclusion criteria may introduce selection bias; compare included and excluded residents. | `included_vs_excluded_insufficient_followup.xlsx`; `excluded_residents_baseline_summary.xlsx`; `excluded_adl_missing_baseline_summary.xlsx`; `excluded_exit_reason_summary.xlsx`; `excluded_region_summary.xlsx`; `facility_size_missingness_and_outcome.xlsx`; `facility_region_missingness_and_outcome.xlsx` | **已可回答，需註明限制** | 已補 included 30,117 vs excluded 19,756 baseline comparison。排除原因前三項：返家/家屬自行照顧 51.5%、空白未填 18.0%、轉院/轉介 16.9%。需註明 included cohort 的機構大小/區域因缺穩定 `dbname` 不能完整合併。 |
| BK-10 | SHAP should explain final hybrid predictor, not only one component. | `shap_feature_importance.xlsx`; `shap_feature_importance.png`; `model_identity_note.txt` | **可回答，但模型命名需一致** | 回覆可說 revised SHAP uses model-agnostic explanation of final `predict_proba` output。另需處理 manuscript 中 HybridXGBRF 標籤與實際 XGBClassifier 物件可能不一致的問題，避免過度宣稱 blended hybrid。 |
| BK-11 | Calibration needs intercept, slope, O/E ratio, risk-decile calibration. | `calibration_metrics_external_hybridxgbrf.xlsx`; `calibration_metrics_external_hybridxgbrf_with_ci.xlsx`; `risk_decile_calibration_external_hybridxgbrf.xlsx`; `external_validation_calibration_with_histogram.png` | **已充分回答** | Calibration intercept 0.591 (95% CI 0.525-0.662)，slope 1.323 (1.261-1.393)，O/E 1.158 (1.138-1.179)，Brier 0.122 (0.118-0.125)。 |
| BK-12 | Code/data availability: provide code, preprocessing, random seeds, package versions, data-access statement. | `README.md`; `RESULTS\README_RESULTS.md`; `RESULTS\package_versions.txt`; `revision_generate_results.py`; `survival_generate_results.py`; `selected_features.xlsx`; `predictor_measurement_window_table.xlsx` | **已回答** | 回覆應說 individual-level data cannot be publicly shared due to privacy/data-use restrictions, but scripts, preprocessing details, seeds, package versions, and aggregate outputs are documented. |
| BK-13 | Recent work on continuous prediction/event completion in temporal data. | `survival_*` outputs; `predictor_measurement_window_table.xlsx` | **正文補充為主** | 不需要大工程重跑。可在 Discussion/Future Work 補 dynamic prediction / longitudinal updating / event completion literature，並說本研究目前是 fixed 6-month horizon plus survival sensitivity。 |
| BK minor-1 | Abstract overstates benefit of hybrid model. | `paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx` | **已可支撐文字修改** | Abstract/Results 改成 comparable discrimination，不寫 superior。 |
| BK minor-2 | Tables hard to read; add CI/highlight best values. | `table4_external_validation_paper_friendly.xlsx`; all `*_with_ci.xlsx` | **結果已支援，仍需 manuscript formatting** | 可用 paper-friendly table，投稿版 Word 表格需人工 bold best values、加 CI。 |
| BK minor-3 | Limitations should mention documentation bias, facility clustering, leakage from post-admission longitudinal variables. | `missingness_indicator_*`; `facility_missingness_*`; `facility_size_missingness_and_outcome.xlsx`; `predictor_measurement_window_table.xlsx` | **可回答，但需正文補強** | Limitations 明確加入 documentation bias、facility-level heterogeneity/clustering、longitudinal predictors potential leakage。 |

## Reviewer W point-to-point

| # | Reviewer W comment | 可用 results / files | 覆蓋狀態 | 建議回覆方向 |
|---|---|---|---|---|
| W-1 | Need more Taiwan LTCF context and international generalizability framing. | No results required; can cite cohort size and external validation from `manifest.json`/tables | **正文補充為主** | 在 Introduction/Discussion 補台灣 LTCF 的照護角色、住民 frailty、與其他國家的差異；說明外推到不同 financing/admission policy 的 LTCF 需謹慎。 |
| W-2 | COVID pandemic may affect LTCF mortality and results. | `table4_external_validation_full_with_ci.xlsx`; survival outputs | **部分可支撐，正文補充為主** | 可用 temporal external validation 說明模型在 later cohort 仍有 discrimination，但目前沒有 dedicated calendar-period/COVID table。Discussion 加 pandemic-era limitation。若要更強，可補 `performance_by_calendar_period.xlsx`。 |
| W-3 | Claims about actionable prediction need references and caution. | `threshold_tradeoff_external_hybridxgbrf.xlsx`; `decision_curve_external_validation.png` | **結果可支撐，正文需降調與補引用** | 回覆說已新增 threshold/DCA 來界定可能使用情境，並把 actionable 改成 may support risk stratification and clinical review。需補 clinical AI / palliative care trigger / LTCF risk stratification references。 |
| W-4 | Excluded residents without sufficient follow-up need baseline analysis / sensitivity check. | `included_vs_excluded_insufficient_followup.xlsx`; `excluded_residents_baseline_summary.xlsx`; `excluded_exit_reason_summary.xlsx`; `facility_size_missingness_and_outcome.xlsx`; `facility_region_missingness_and_outcome.xlsx` | **已可回答** | 直接回覆已新增 included vs excluded baseline comparison、exit reason summary、facility-size/region characterization，並在 Limitations 說明 selection bias 與 facility linkage 限制。 |
| W-5 | Prediction model review/selection methods should be described in appendix and cited. | `selected_features.xlsx`; `predictor_measurement_window_table.xlsx`; `shap_feature_importance.xlsx` | **部分可回答，正文/appendix 補充為主** | 建議新增 appendix paragraph/table：candidate predictor domain、clinical rationale、measurement timing、preprocessing、selection/retention reason。現有 files 可支撐，但仍需補文獻引用。 |
| W-7 | Remove long em dash signs. | No results required | **文字修改** | 全文搜尋 em dash 並替換成 comma/parenthesis/semicolon。 |
| W-8 | Add distribution histogram of predicted probabilities in calibration plots. | `internal_cv_calibration_with_histogram.png`; `external_validation_calibration_with_histogram.png` | **已充分回答** | 回覆可直接說 calibration plots now include predicted-risk histogram below observed-vs-predicted calibration panel。 |
| W-9 | ADL imputation/exclusion wording appears contradictory. | `missingness_indicator_development.xlsx`; `missingness_indicator_external.xlsx`; `selected_features.xlsx`; `predictor_measurement_window_table.xlsx`; `excluded_adl_missing_baseline_summary.xlsx` | **可回答，需 Methods 澄清** | Methods 要分清楚：ADL assessment incompleteness as exclusion criterion for a specific analytic subset vs limited missingness in retained predictors handled through preprocessing. Continuous/scale variables imputed at development mean; standardized missing z-score equals 0. |

## 已足以 point-to-point 回覆的 reviewer concerns

- BK-2 paired bootstrap comparison.
- BK-3 performance and calibration 95% CI.
- BK-4/BK-5 threshold tradeoff, PPV/NPV, clinical utility, DCA.
- BK-7 survival/time-to-event sensitivity.
- BK-9 included vs excluded residents and exit reason summary.
- BK-10 SHAP final-output explanation, provided manuscript model naming is corrected.
- BK-11 calibration slope/intercept/OE/risk-decile calibration.
- BK-12 code/seed/package/preprocessing transparency.
- W-4 excluded residents baseline comparison.
- W-8 calibration histogram.
- W-9 imputation clarification, provided Methods text is revised.

## 仍不能只靠 RESULTS 自動解決的 concerns

這些不是 results 缺失，而是 manuscript/reply wording 必須處理：

- Taiwan LTCF context and international generalizability.
- COVID-era discussion.
- Actionable AI claim references and cautious wording.
- Formal clinical risk scores unavailable unless source variables permit calculation.
- Continuous prediction/event completion literature.
- Em dash / prose cleanup.
- Potential leakage from longitudinal predictors if the intended use is admission-time prediction.

## 建議新增到 response letter 的總體段落

We thank the reviewers for identifying areas that required stronger validation, calibration, and transparency. In response, we added confidence intervals for internal and external performance metrics, paired bootstrap comparisons between the selected model and XGBoost, threshold-specific tradeoff analyses including PPV and NPV, decision-curve analysis, calibration plots with predicted-risk histograms, numerical calibration metrics with bootstrap confidence intervals, risk-decile calibration, SHAP explanations of the final predicted probability output, missingness and facility-level missingness summaries, included-versus-excluded resident comparisons with exit-reason summaries, and survival sensitivity analyses using Cox models, Harrell's C-index, time-dependent AUC, and Kaplan-Meier curves by predicted-risk strata.

We also revised the manuscript text to clarify predictor measurement windows, preprocessing and missing-data handling, model reproducibility, and the intended clinical role of the model. In particular, we toned down claims of HybridXGBRF superiority and actionable prediction, describing the model instead as a tool that may support risk stratification and clinical review. We expanded the Limitations to address selection bias from excluded residents, documentation bias, facility-level heterogeneity, possible leakage from longitudinal predictors, pandemic-era effects, and generalizability beyond Taiwanese long-term care facilities.

## 投稿前人工檢查清單

- Confirm manuscript wording does not claim HybridXGBRF is significantly superior to XGBoost.
- Confirm the intended prediction time point: admission, early-stay, or longitudinal updating.
- Add or cite `predictor_measurement_window_table.xlsx` in appendix/supplement.
- Decide whether formal clinical risk scores are impossible from available variables; if yes, say so directly.
- Add Taiwan LTCF context, COVID-era limitation, and cautious actionable-AI wording.
- Replace em dash characters in manuscript prose.
- Ensure final manuscript numbers match the regenerated `RESULTS` values if using the current rerun.
