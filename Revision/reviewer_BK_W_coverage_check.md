# Reviewer BK / Reviewer W 補充圖表覆蓋檢查

來源檔案：

- `C:\AI4JUBO\Revision\00_Reviewer BK_External Peer-Review.docx`
- `C:\AI4JUBO\Revision\00_Reviewer W_External Peer-Review.docx`

目前補充輸出：

- 表格：`C:\AI4JUBO\RESULTS\tables`
- 圖檔：`C:\AI4JUBO\RESULTS\figures`
- 主分析程式：`C:\AI4JUBO\revision_generate_results.py`
- Survival 分析程式：`C:\AI4JUBO\survival_generate_results.py`

## 總結判斷

目前補充的圖表已經可以回答多數「統計驗證」類挑戰，包括 performance confidence intervals、paired bootstrap、calibration histogram、threshold tradeoff、decision curve analysis、SHAP、missingness/facility missingness，以及 survival sensitivity analysis。

但還沒有完全覆蓋所有審稿意見。主要缺口有五個：

1. **Included vs excluded because of insufficient follow-up**：目前沒有完整 baseline comparison table。
2. **Predictor measurement window / possible leakage**：目前有 selected features，但還沒有清楚列出每個 predictor 的量測時間窗；若 `ADL last/max/difference`、`body weight change` 等變項使用了 outcome 前整段追蹤資料，會被 reviewer 視為 potential information leakage。
3. **Clinical risk score baseline**：目前可用 logistic/XGB/RF/Cox/survival sensitivity 回答，但沒有 MDS-CHESS、ADEPT 或其他正式臨床風險分數的外部比較；若資料欄位不足，需要明確寫成 limitation。
4. **Taiwan LTCF context、COVID-era impact、actionable claim 的文獻與語氣**：這些需要 manuscript text/references，不是補圖表即可解決。
5. **Alternative imputation sensitivity**：目前有 missingness indicator 與 facility missingness，但還沒有一張「不同 imputation strategy 下 model performance 是否穩定」的表。

建議投稿前優先補齊第 1、2 點；其中第 2 點最重要，因為它直接影響模型是否真的能在 admission / early-stay 時點使用。

---

## Reviewer BK 覆蓋檢查

| BK 挑戰 | 目前可用圖表/檔案 | 覆蓋狀態 | 判斷與下一步 |
|---|---|---|---|
| Clarify timing of predictor measurement and avoid information leakage | `RESULTS\tables\selected_features.xlsx`; `RESULTS\tables\shap_feature_importance.xlsx`; `RESULTS\figures\shap_feature_importance.png` | **部分回答，仍需補強** | selected features 和 SHAP 可說明模型用哪些變項，但不能單獨回答「每個變項是在預測前何時量測」。建議新增 `RESULTS\tables\predictor_measurement_window_table.xlsx`，列出每個 predictor 的來源、量測時間窗、是否 baseline/early-stay/prior 6 months，以及是否可能包含 outcome 後資訊。若 ADL last/max/difference 或 body weight change 來自 6-month follow-up 期間，需重新定義 index date 或重跑 early-window-only model。 |
| Hybrid model only marginally better than XGBoost; need statistical comparison | `RESULTS\tables\paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`; `RESULTS\tables\paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx` | **已充分回答** | External AUROC difference 約 0.00036，95% CI 跨 0，P = 0.694。建議 response 不要主張 hybrid 顯著優於 XGBoost，改寫為 tree-based approaches had comparable discrimination。 |
| Need 95% CI for performance metrics | `RESULTS\tables\table3_internal_cv_performance_with_ci.xlsx`; `RESULTS\tables\table4_external_validation_full_with_ci.xlsx`; `RESULTS\tables\table5_subgroup_performance_with_ci.xlsx` | **已充分回答** | 已可補進 main table 或 supplement。若正文篇幅有限，main table 放 AUROC、sensitivity、specificity、PPV、NPV、F1、Brier，完整 CI 放 supplement。 |
| AUROC not enough for clinical decision-making | `RESULTS\tables\threshold_tradeoff_external_hybridxgbrf.xlsx`; `RESULTS\tables\decision_curve_external_validation.xlsx`; `RESULTS\figures\decision_curve_external_validation.png` | **已充分回答** | 可用 threshold table 說明 0.20 threshold sensitivity 較高、0.50 threshold PPV/specificity 較高；DCA 可支持 net benefit。需在 Results/Discussion 補文字解釋 threshold 適用情境。 |
| Recall/sensitivity modest; need explain missed high-risk residents and threshold tradeoff | `RESULTS\tables\threshold_tradeoff_external_hybridxgbrf.xlsx`; `RESULTS\figures\external_validation_confusion_matrices.png`; `RESULTS\figures\decision_curve_external_validation.png` | **大致已回答** | 圖表足夠，但正文需降調：此模型較適合作為 risk stratification / review trigger，不應寫成單獨決策工具。 |
| Need stronger baselines, preferably clinical risk scores or survival models | `RESULTS\tables\table3_internal_cv_performance_with_ci.xlsx`; `RESULTS\tables\table4_external_validation_full_with_ci.xlsx`; `RESULTS\tables\survival_cox_ml_risk_score.xlsx`; `RESULTS\tables\survival_c_index.xlsx`; `RESULTS\figures\survival_time_dependent_auc.png` | **部分回答** | 已補 Cox/survival sensitivity，但未補正式臨床風險分數。若資料沒有 MDS-CHESS/ADEPT 所需欄位，建議在 response 寫明 cannot be computed from available LTCF variables，並把 Cox sensitivity 作為 time-to-event robustness check。 |
| Binary 6-month outcome ignores censoring/time-to-event | `RESULTS\tables\survival_c_index.xlsx`; `RESULTS\tables\survival_cox_ml_risk_score.xlsx`; `RESULTS\tables\survival_time_dependent_auc.xlsx`; `RESULTS\tables\survival_km_external_risk_group_summary.xlsx`; `RESULTS\tables\survival_logrank_external_risk_groups.xlsx`; `RESULTS\figures\survival_km_external_by_risk_group.png`; `RESULTS\figures\survival_time_dependent_auc.png` | **已大幅補強，但需寫限制** | Survival sensitivity 已可回答：external C-index 0.849、risk-score Cox HR per 0.10 risk increase 1.65、KM tertile 分層明顯。需註明 6-month time-dependent AUC 因 administrative censoring 在 180 天不穩定/不可估。 |
| Missing values coded as absence may bias model | `RESULTS\tables\missingness_indicator_development.xlsx`; `RESULTS\tables\missingness_indicator_external.xlsx`; `RESULTS\tables\facility_missingness_development.xlsx`; `RESULTS\tables\facility_missingness_external.xlsx` | **部分回答** | 目前可描述 missingness pattern 與 facility-level missingness，但還沒有 alternative imputation sensitivity performance table。建議補 `alternative_imputation_performance.xlsx`，至少比較 current preprocessing vs median/mode imputation plus missingness indicators。 |
| Exclusion criteria may introduce selection bias | 尚缺：`RESULTS\tables\included_vs_excluded_insufficient_followup.xlsx` | **尚未充分回答** | 這是目前最大缺口之一。若資料端能取得 excluded residents baseline，需補 included vs excluded table。若無法取得，response 必須明確承認 limitation。 |
| SHAP should explain final trained model, not only one component | `RESULTS\tables\shap_feature_importance.xlsx`; `RESULTS\figures\shap_feature_importance.png`; `RESULTS\model_identity_note.txt` | **部分至充分，取決於 manuscript wording** | 目前 SHAP 是針對 final predict_proba output 的 KernelSHAP，可回答 reviewer。但 `model_identity_note.txt` 顯示原 notebook 的 HybridXGBRF 標籤和實際模型物件可能不一致；需先在文稿中統一模型名稱。 |
| Calibration needs deeper reporting | `RESULTS\tables\calibration_metrics_external_hybridxgbrf.xlsx`; `RESULTS\tables\risk_decile_calibration_external_hybridxgbrf.xlsx`; `RESULTS\figures\external_validation_calibration_with_histogram.png` | **已充分回答** | 已有 calibration intercept、slope、Brier、O/E、decile calibration 與 histogram。 |
| Code/seed/package/preprocessing reproducibility | `README.md`; `RESULTS\package_versions.txt`; `revision_generate_results.py`; `survival_generate_results.py` | **已回答** | README 已整理 Code / seed / package / preprocessing。若投稿系統需匿名 code availability statement，需另加一句 repository available upon reasonable request 或 supplemental code。 |
| Need recent literature on continuous prediction/event completion | 無圖表需求 | **尚需正文補充** | 這是 Discussion/Future work 的文字與引用問題。建議加入 continuous risk updating、event completion、dynamic prediction 相關文獻，說明本研究目前是 fixed 6-month horizon，未來可發展成 longitudinal updating。 |
| Abstract/result tone overstates clinical utility | 無圖表需求 | **尚需文字修改** | 建議把 "supports proactive interventions" 改為 "may support risk stratification and clinical review"，避免 reviewer 認為超出證據。 |
| Table readability / bold best values / avoid overclaim | `RESULTS\tables\table4_external_validation_paper_friendly.xlsx` | **部分回答** | paper-friendly table 可支援重排，但 manuscript table 仍需人工格式化。 |
| Limitations: documentation bias, facility clustering, leakage risk | `RESULTS\tables\facility_missingness_external.xlsx`; `RESULTS\tables\missingness_indicator_external.xlsx` | **部分回答** | documentation/facility missingness 已有表；leakage risk 仍需 predictor time-window table 或 early-window sensitivity。 |

---

## Reviewer W 覆蓋檢查

| W 挑戰 | 目前可用圖表/檔案 | 覆蓋狀態 | 判斷與下一步 |
|---|---|---|---|
| Need Taiwan LTCF context and generalizability framing | 無圖表需求 | **尚需正文補充** | 需要在 Introduction/Discussion 補台灣 LTCF 的照護場域、住民特性、照護紀錄可得性，並說明外推到其他國家或照護體系需謹慎。 |
| COVID-era data may affect mortality and generalizability | 目前無年份/疫情分層表 | **部分回答不足** | 可用 external validation 表支持 2024 cohort 仍有表現，但沒有專門回答疫情影響。建議正文加入 pandemic-era limitation；若資料有年份，可再補 `performance_by_calendar_period.xlsx`。 |
| Actionable AI claims need references/caution | `RESULTS\tables\threshold_tradeoff_external_hybridxgbrf.xlsx`; `RESULTS\figures\decision_curve_external_validation.png` | **部分回答** | DCA/threshold table 支持 clinical review trigger，但仍需文獻與語氣降調。不要寫模型可直接決定介入；應寫可協助辨識需照護團隊複評的住民。 |
| Excluded residents without sufficient follow-up need baseline comparison | 尚缺：`RESULTS\tables\included_vs_excluded_insufficient_followup.xlsx` | **尚未充分回答** | 與 BK selection bias 相同，是目前最明確的資料缺口。 |
| Predictor review/selection method needs clearer appendix | `RESULTS\tables\selected_features.xlsx`; `RESULTS\tables\shap_feature_importance.xlsx` | **部分回答** | 目前有模型選用特徵與重要性，但還沒有完整的 candidate predictor appendix：candidate variable、clinical rationale、source、measurement window、preprocessing、included/excluded reason。 |
| Calibration plot should include predicted probability histogram | `RESULTS\figures\internal_cv_calibration_with_histogram.png`; `RESULTS\figures\external_validation_calibration_with_histogram.png` | **已充分回答** | 這一點已完成。可在 response 直接列出新增 figure。 |
| ADL imputation description appears contradictory | `RESULTS\tables\missingness_indicator_development.xlsx`; `RESULTS\tables\missingness_indicator_external.xlsx` | **部分回答** | missingness 表可支撐說明，但 manuscript Methods 必須明確寫清楚 ADL missing 的處理邏輯。若實際作法是 missing encoded as 0 或 no-record category，要避免寫成 imputed as normal function。 |
| Remove awkward em dash / improve wording | 無圖表需求 | **尚需文字修改** | 需全文人工編修，不屬於 RESULTS 圖表。 |

---

## 可直接放進回覆信的整體說法

We thank both reviewers for identifying several areas where the original submission required clearer validation, calibration, and transparency. In response, we added confidence intervals for internal and external performance metrics, paired bootstrap comparisons between the final model and XGBoost, calibration plots with predicted-risk histograms, calibration intercept/slope and decile calibration summaries, decision curve analysis, threshold-specific tradeoff tables, SHAP-based model explanations for the final prediction output, missingness and facility-level missingness summaries, and time-to-event sensitivity analyses including Cox models, Harrell's C-index, time-dependent AUC, and Kaplan-Meier curves by predicted-risk strata.

Some reviewer concerns require manuscript clarification rather than figures alone. We therefore also revised the Methods and Discussion to clarify the predictor measurement window, preprocessing, missing-data handling, model reproducibility, and the intended use of the model as a risk stratification and clinical review support tool rather than a stand-alone decision-making system. We also expanded the Limitations to address selection bias from excluded residents, documentation bias, facility-level heterogeneity, potential pandemic-era effects, and generalizability beyond Taiwanese long-term care facilities.

---

## 投稿前最建議補的檔案

1. `RESULTS\tables\included_vs_excluded_insufficient_followup.xlsx`
   - 用來回答 BK/W 都提到的 exclusion / selection bias。

2. `RESULTS\tables\predictor_measurement_window_table.xlsx`
   - 用來回答 predictor timing、preprocessing、potential leakage。

3. `RESULTS\tables\candidate_predictor_appendix.xlsx`
   - 用來回答 W 對 predictor review / selection method 的要求。

4. `RESULTS\tables\alternative_imputation_performance.xlsx`
   - 用來把 missingness concern 從「描述」提升到「敏感性分析」。

5. 可選：`RESULTS\tables\performance_by_calendar_period.xlsx`
   - 若資料可分年或疫情前/中/後，可更直接回答 COVID-era generalizability。

---

## 最終判斷

目前補充圖表已足以支撐以下挑戰：

- performance 95% CI
- internal/external ROC and metrics
- paired bootstrap AUROC comparison
- threshold tradeoff
- decision curve analysis
- calibration plot with histogram
- calibration intercept/slope and decile calibration
- SHAP model explanation
- missingness/facility missingness descriptive analysis
- survival/time-to-event sensitivity analysis
- code/package/seed/preprocessing reproducibility

目前仍不足以完全支撐以下挑戰：

- insufficient-follow-up excluded residents 的 baseline comparison
- predictor measurement window and leakage risk
- formal clinical risk score baseline comparison
- alternative imputation performance sensitivity
- Taiwan LTCF context, COVID-era discussion, actionable AI claim references
- manuscript wording, abstract tone, and limitations revision

