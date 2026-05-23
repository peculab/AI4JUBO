# Reviewer BK / Reviewer W point-to-point coverage check

更新日期：2026-05-18 晚間重跑後  
工作區：`..`

## 一、目前判定

已檢查並對齊下列來源：

- `Revision/response_mapping_for_01_請蔡老師協助的部分.md`
- `Revision/reviewer_BK_W_coverage_check.md`
- `jubodeath_v9_puredata_paper.ipynb`
- `Revision/01_需蔡老師補做資料表/`
- `revision_generate_results.py`
- `survival_generate_results.py`
- `RESULTS/tables/` 與 `RESULTS/figures/`

2026-05-18 晚間已將 preprocessing 改成與原始投稿文字一致，並重跑全部主要 revision 與 survival sensitivity outputs：

- Binary/count variables：missing 補 `0`。
- Continuous/scale variables：以 development cohort 或 CV training fold 的 mean/SD 做 z-score，missing standardized value 補 `0`，等同原始尺度補 mean。
- External validation：固定使用 development cohort fitted preprocessing，不用 external cohort 重估 mean/SD。
- Internal CV：每 fold 只用 training fold fitted preprocessing，再套到 validation fold。

`Revision/01_需蔡老師補做資料表/` 已同步最新重跑後的主要 xlsx/png 檔案。之後回覆審查委員與更新 manuscript 時，數字應以 2026-05-18 20:48-20:54 產出的 `RESULTS` 與該 Revision delivery folder 為準。

## 二、Notebook 與目前 revision code 的一致性判讀

`jubodeath_v9_puredata_paper.ipynb` 是原始投稿結果追溯來源，但不是此次重跑後的唯一依據。檢查重點如下：

- Notebook 原始流程中可見 `dfNew = dfNew.fillna(0)` 與 `ex_X = ex_X.fillna(0)`，也有模型不支援 NaN 時的 `SimpleImputer(strategy='median')` fallback。
- 原始投稿文字寫 continuous/scale variables z-score 後 missing 補 0，等同 mean imputation。為了讓程式與投稿文字一致，已在 `revision_generate_results.py` 新增 `fit_preprocessing()` / `apply_preprocessing()`，正式採用 binary/count 補 0、continuous/scale z-score 後補 0。
- Notebook 中 `HybridXGBRF (Our Approach)` 在 final `all_models` 似乎實際對應 `XGBClassifier`。修稿回覆應避免宣稱 blended HybridXGBRF 顯著優於 XGBoost；應寫成 selected tree-based / XGBoost-style model，或在 manuscript 中明確重新定義模型名稱。
- 目前 SHAP 已由 revision script 以 model-agnostic KernelSHAP fallback 套用 final `predict_proba` output；因此修稿版可說 revised SHAP explains the final predicted probability output。

## 三、最新關鍵結果

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
| External KM risk groups | low 2.5%, medium 17.2%, high 66.3% deaths within horizon |
| Top SHAP features | hospitalizations within 6 months, ADL change, body weight change, ADL last score, ADL total max |

## 四、Reviewer BK point-to-point

| # | Reviewer concern | 覆蓋狀態 | 對應檔案 | 回覆重點 |
| --- | --- | --- | --- | --- |
| BK-1 | Every feature needs a clear measurement window. | 可回覆，但 manuscript 需補文字 | `selected_features.xlsx`; `predictor_measurement_window_table.xlsx`; `revision_generate_results.py` | 補 predictor measurement window appendix。需清楚說明 first/last/max/diff/std 是 early-stay/longitudinal routine-care predictors，不應過度宣稱純 admission-time prediction。 |
| BK-2 | HybridXGBRF improvement over XGBoost is small; need paired comparison. | 已回覆 | `paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`; `paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx` | 最新 paired bootstrap 顯示 external AUROC difference 0.0036 (0.0012-0.0060), P=0.003；但幅度仍小，文字應寫 comparable high discrimination / modest difference，不要誇大 superiority。 |
| BK-3 | Need 95% CI for precision, recall, F1, Brier, calibration, subgroup. | 已回覆 | `table3_internal_cv_performance_with_ci.xlsx`; `table4_external_validation_full_with_ci.xlsx`; `table4_external_validation_paper_friendly.xlsx`; `table5_subgroup_performance_with_ci.xlsx`; `calibration_metrics_external_hybridxgbrf_with_ci.xlsx` | 已補 internal/external/subgroup performance 與 calibration metrics 95% CI。 |
| BK-4 | Recall is limited; discuss missed high-risk residents and thresholds. | 已回覆，需降調 clinical claim | `threshold_tradeoff_external_hybridxgbrf.xlsx`; `external_validation_confusion_matrices.png` | Threshold 0.20 sensitivity 0.861, NPV 0.930；threshold 0.50 PPV 0.855, specificity 0.961。說明低 threshold 適合 screening/review trigger，高 threshold 適合 resource prioritization。 |
| BK-5 | AUROC alone insufficient; add DCA, PPV/NPV, threshold tradeoffs. | 已回覆 | `decision_curve_external_validation.xlsx`; `decision_curve_external_validation.png`; `threshold_tradeoff_external_hybridxgbrf.xlsx` | 已補 DCA、net benefit、PPV/NPV、threshold-specific confusion counts。 |
| BK-6 | Baselines incomplete; add clinical risk score or survival/time-to-event baselines. | 部分回覆 | `survival_c_index.xlsx`; `survival_cox_ml_risk_score.xlsx`; `survival_time_dependent_auc.xlsx`; `survival_km_external_risk_group_summary.xlsx` | 已補 survival sensitivity。若 MDS-CHESS/ADEPT 等正式臨床分數缺必要欄位，回覆需明確說無法可靠計算，並以 survival/Cox sensitivity 補強。 |
| BK-7 | Binary 6-month outcome ignores censoring/time-to-death. | 已回覆，需標為 sensitivity | `survival_*` tables/figures | 已補 C-index、Cox risk score、time-dependent AUC、KM risk groups、log-rank。180 天 AUC 因 administrative horizon 下無 event-free controls 為 NA，應誠實說明。 |
| BK-8 | Missing binary/count coded absence may reflect documentation missingness; need missingness/facility checks. | 已回覆，且 preprocessing 已對齊投稿 | `missingness_indicator_*`; `facility_missingness_*`; `facility_size_missingness_and_outcome.xlsx`; `facility_region_missingness_and_outcome.xlsx`; `missing_value_imputation_code_comparison.md` | 已補 missingness indicator 與 facility missingness。Methods 要清楚寫 binary/count 補 0；continuous/scale z-score 後補 0 等同 mean。 |
| BK-9 | Exclusion criteria may cause selection bias; compare included vs excluded. | 已回覆，需註明 facility linkage 限制 | `included_vs_excluded_insufficient_followup.xlsx`; `excluded_residents_baseline_summary.xlsx`; `excluded_exit_reason_summary.xlsx`; `excluded_residents_by_facility_size_20260516.xlsx`; `excluded_residents_by_region_20260516.xlsx` | 已補 included 30,117 vs excluded 19,756 baseline comparison、p values、exit reasons、excluded residents region/size。Included cohort resident-level facility linkage 仍有限。 |
| BK-10 | SHAP should explain final hybrid predictor, not one component. | 可回覆，但模型命名要修 | `shap_feature_importance.xlsx`; `shap_feature_importance.png`; `model_identity_note.txt` | Revised SHAP 使用 KernelSHAP on final `predict_proba` output。Manuscript 仍需修正 HybridXGBRF vs XGBClassifier naming ambiguity。 |
| BK-11 | Add calibration intercept, slope, O/E, risk-decile calibration. | 已回覆 | `calibration_metrics_external_hybridxgbrf.xlsx`; `calibration_metrics_external_hybridxgbrf_with_ci.xlsx`; `risk_decile_calibration_external_hybridxgbrf.xlsx`; `external_validation_calibration_with_histogram.png` | 最新 intercept 0.516, slope 1.271, O/E 1.128, Brier 0.112，均有 95% CI。 |
| BK-12 | Provide code, seeds, package versions, preprocessing details. | 已回覆 | `README.md`; `RESULTS/README_RESULTS.md`; `package_versions.txt`; `revision_generate_results.py`; `survival_generate_results.py`; `selected_features.xlsx` | 可公開 scripts、aggregate outputs、preprocessing details、random seeds、package versions；individual-level data 因 privacy/data-use restrictions 不公開。 |
| BK-13 | Discuss continuous prediction/event-completion literature. | 主要是 manuscript 文字 | `survival_*`; `predictor_measurement_window_table.xlsx` | Discussion/Future Work 補 dynamic prediction / longitudinal updating / time-to-event literature；目前是 fixed 6-month model plus survival sensitivity。 |

## 五、Reviewer W point-to-point

| # | Reviewer concern | 覆蓋狀態 | 對應檔案 | 回覆重點 |
| --- | --- | --- | --- | --- |
| W-1 | Add Taiwan LTCF context and international generalizability framing. | 需 manuscript 文字 | 主文 Discussion | 補台灣 LTCF 照護情境、住民組成、資料可得性，說明外推到不同制度需謹慎。 |
| W-2 | COVID-era may affect LTCF mortality/results. | 需 manuscript 文字，可用 temporal validation 支撐 | `table4_external_validation_*`; `survival_*` | 說明 2024 temporal validation 仍有 high discrimination，但未做 dedicated COVID/calendar-period analysis，列為 limitation。 |
| W-3 | Actionable prediction claims need references and caution. | 已有支撐圖表，需降調文字 | `threshold_tradeoff_external_hybridxgbrf.xlsx`; `decision_curve_external_validation.png` | 將 actionable 改成 may support risk stratification and clinical review，不寫 standalone decision-making。 |
| W-4 | Excluded residents without sufficient follow-up need baseline analysis/sensitivity. | 已回覆 | `included_vs_excluded_insufficient_followup.xlsx`; `excluded_exit_reason_summary.xlsx`; `excluded_residents_by_facility_size_20260516.xlsx`; `excluded_residents_by_region_20260516.xlsx` | 已補 baseline、p value、exit reasons、facility size/region for excluded residents。需說明 included resident-level facility linkage 限制。 |
| W-5 | Predictor review/selection methods should be clearer. | 需 appendix/manuscript 補文字 | `selected_features.xlsx`; `predictor_measurement_window_table.xlsx` | 補 candidate predictor rationale、measurement timing、availability、preprocessing、retention/exclusion logic。 |
| W-7 | Remove long em dash signs. | 純文字修改 | manuscript docx | 全文 search em dash 並改成 comma/parenthesis/semicolon。 |
| W-8 | Add predicted-probability histogram to calibration plots. | 已回覆 | `internal_cv_calibration_with_histogram.png`; `external_validation_calibration_with_histogram.png` | Calibration plots 已含 predicted-risk histogram。 |
| W-9 | ADL imputation/exclusion wording contradictory. | 已可回覆，需 Methods 澄清 | `missing_value_imputation_code_comparison.md`; `missingness_indicator_*`; `selected_features.xlsx`; `excluded_adl_missing_baseline_summary.xlsx` | 區分：嚴重/incomplete ADL assessment 可作 exclusion；納入 cohort 後 retained ADL-derived features 的剩餘缺失依 continuous/scale mean-equivalent imputation 處理。 |

## 六、可貼入 Response Letter 的總回覆段落

We thank the reviewers for identifying areas requiring stronger validation, calibration, and transparency. In response, we revised the preprocessing code to match the missing-data strategy described in the manuscript, regenerated all main and sensitivity outputs, and added confidence intervals for internal and external performance metrics, paired bootstrap comparisons with XGBoost, threshold-specific PPV/NPV and sensitivity/specificity tradeoffs, decision-curve analysis, calibration plots with predicted-risk histograms, numerical calibration metrics with bootstrap confidence intervals, risk-decile calibration, missingness and facility-level missingness summaries, included-versus-excluded resident comparisons, facility size and region characterization, model-agnostic SHAP explanations of the final predicted probability output, and survival sensitivity analyses using Cox models, Harrell's C-index, time-dependent AUC, and Kaplan-Meier curves by predicted-risk strata.

We also revised the manuscript text to clarify predictor measurement windows, preprocessing and missing-data handling, model reproducibility, and the intended clinical role of the model. We toned down claims of HybridXGBRF superiority and actionable prediction, describing the model as a tool that may support risk stratification and clinical review. We expanded the Limitations to address selection bias from excluded residents, documentation bias, facility-level heterogeneity, incomplete facility linkage, potential leakage from longitudinal predictors, pandemic-era effects, and generalizability beyond Taiwanese long-term care facilities.

## 七、投稿前必須人工確認

- Manuscript 所有 performance numbers 是否改成 2026-05-18 重跑後數字，尤其 external AUROC 0.898、Brier 0.112。
- Abstract/Results 不要再寫 HybridXGBRF 明顯或臨床上大幅優於 XGBoost；paired AUROC difference 雖統計顯著但幅度小。
- Methods 的 missing-data handling 要與 `missing_value_imputation_code_comparison.md` 一致。
- SHAP caption 要寫 model-agnostic KernelSHAP / final predicted probability output。
- Predictor measurement window 要避免純 admission-time 說法，或明確改為 early-stay/longitudinal routine-care risk model。
- Limitations 要加入 documentation bias、selection bias、facility linkage、COVID-era、longitudinal predictor leakage。
- 若要把 `Revision/01_需蔡老師補做資料表` 內檔案交給老師，請使用 2026-05-18 20:48-20:54 更新時間的檔案。
