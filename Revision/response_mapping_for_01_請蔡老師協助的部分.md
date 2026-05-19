# `01_請蔡老師協助的部分.docx` 對應回覆、圖表與檔名

來源檔案：`C:\AI4JUBO\Revision\01_請蔡老師協助的部分.docx`

可用輸出：

- 表格：`C:\AI4JUBO\RESULTS\tables`
- 圖檔：`C:\AI4JUBO\RESULTS\figures`
- 主分析程式：`C:\AI4JUBO\revision_generate_results.py`
- Survival 分析程式：`C:\AI4JUBO\survival_generate_results.py`
- 套件版本：`C:\AI4JUBO\RESULTS\package_versions.txt`

## 2026-05-18 更新：included vs excluded 表、p 值與投稿版數據差異追查

本次已完成 Reviewer W / BK 對「excluded because of insufficient follow-up」與補充表的更新：

- `RESULTS\tables\included_vs_excluded_insufficient_followup.xlsx` 已新增 `P value` 欄位。
- `RESULTS\tables\included_vs_excluded_insufficient_followup_with_p.xlsx` 為同內容的 p-value 標示副本。
- `RESULTS\tables\development_cohort_plus_current_excluded_insufficient_followup.xlsx` 為仿投稿版 Development Cohort appendix 格式的新表，保留投稿版 development cohort 欄位與數字，並補上目前資料可產出的 insufficient-follow-up excluded residents 欄位。
- `Revision\20260516\data_discrepancy_and_table_notes.md` 已整理新版程式輸出與投稿最終版數字不同的原因。
- `Revision\01_需蔡老師補做資料表\COMPLETED_Development_Cohort_included_vs_excluded_insufficient_followup.docx` 已將原本 placeholder 的 Development Cohort Word 表補完。
- `Revision\01_需蔡老師補做資料表\COMPLETED_95CI_tables.docx` 已將 95% CI Word 表補完。

主要差異原因如下：

- 新版 `included_vs_excluded_insufficient_followup` 的 included cohort 是 development + external validation 合併後的 analytic residents，`N = 30,117`；投稿版 Table 1 / Development Cohort appendix 是 development cohort only，`N = 23,901`。因此 age、ADL、tube feeding、respiratory support、falls、body weight、hospitalizations 等數字不會完全一致。
- 新版表格的 binary/categorical 百分比使用非缺失分母。例如 initial feeding tube 是 `2,689 / 27,330 = 9.8%`，因為 included 端此欄有 2,787 人缺失；投稿版多數欄位使用 cohort total 作分母，例如 `2,081 / 23,901 = 8.7%`。
- facility size / region 目前可由 `DATA\area_size.xlsx` 合併到 excluded residents；但 Google Sheets analytic-cohort extract 不保留穩定可合併的 included cohort `dbname`，所以 included facility-size/region 欄位仍需標註限制。
- CIRS-G 與部分原投稿版 excluded-without-ADL 欄位不在目前 insufficient-follow-up excluded-resident 檔案內，因此 manuscript-style 新表以 `NA` 保留，不做推估。

本次補上的 p 值規則：

- 連續變項使用 Welch two-sample t test。
- binary/categorical 變項使用 nonmissing 2 x 2 chi-square test。
- 任一組沒有可用資料或欄位無法比較者標示 `NA`。

## 2026-05-18 更新：機構區域與大小補充資料

新增來源檔案：

- `Revision\20260516\機構區域與大小.xlsx`

已補出的對應表格：

- `RESULTS\tables\facility_region_size_overview_20260516.xlsx`
- `RESULTS\tables\excluded_residents_by_facility_size_20260516.xlsx`
- `RESULTS\tables\excluded_residents_by_region_20260516.xlsx`

已補出的對應圖檔：

- `RESULTS\figures\training_facility_region_size_20260516.png`
- `RESULTS\figures\excluded_facility_region_size_20260516.png`

同一批表圖也已複製到：

- `Revision\01_需蔡老師補做資料表\facility_region_size_overview_20260516.xlsx`
- `Revision\01_需蔡老師補做資料表\excluded_residents_by_facility_size_20260516.xlsx`
- `Revision\01_需蔡老師補做資料表\excluded_residents_by_region_20260516.xlsx`
- `Revision\01_需蔡老師補做資料表\training_facility_region_size_20260516.png`
- `Revision\01_需蔡老師補做資料表\excluded_facility_region_size_20260516.png`

目前可補強的內容：

- Training/development facility roster 共 493 家機構：小型 311 家（63.1%）、中型 148 家（30.0%）、大型 34 家（6.9%）。
- Training/development facility roster 的區域：南部 332 家（67.3%）、中部 100 家（20.3%）、北部 54 家（11.0%）、東部 7 家（1.4%）。
- Excluded residents with insufficient follow-up 共 19,756 人，區域分布：北部 2,112（10.7%）、中部 4,019（20.3%）、南部 13,447（68.1%）、東部 174（0.9%）、離島 4（0.0%）。
- Excluded residents 依機構大小合併後：小型 9,218（46.7%）、中型 8,211（41.6%）、大型 2,233（11.3%）、Unknown 94（0.5%）。比舊版 merge 少了許多 unknown，表示新檔案改善了 excluded 端的機構大小 linkage。
- Excluded-resident facility roster 共 586 家有機構大小：小型 363 家（61.9%）、中型 182 家（31.1%）、大型 41 家（7.0%）。

仍需限制說明：

- `機構區域與大小.xlsx` 的 training/development 部分是機構層級 roster，不是 resident-level included cohort distribution。因此目前可以說明 development/training facilities 的區域與大小組成，但不能把 23,901 或 30,117 位 included residents 精準分配到機構大小/區域。
- Excluded residents 的區域分布是 workbook 已彙整的人數，可直接使用；excluded residents 的機構大小可用 `dbname` 合併到 resident-level excluded data，但仍有 94 人（0.5%）無法取得機構大小。
- 因 included cohort 仍缺 resident-level stable facility linkage，正式 included vs excluded resident-level facility-size/region statistical comparison 仍不能完整執行；回覆時應描述為 facility-level characterization plus excluded-resident stratification。

可放入 Response Letter 的補充句：

Using the newly provided facility-region and facility-size workbook, we further characterized the facility composition of the development facility roster and the excluded residents. The development facility roster included 493 facilities, most of which were small facilities and located in southern Taiwan. Among residents excluded because of insufficient follow-up, 68.1% were from southern Taiwan, and resident-level linkage to facility size showed that 46.7% were from small facilities, 41.6% from medium facilities, and 11.3% from large facilities, with facility size unavailable for 0.5%. Because the newly provided development-cohort file is a facility-level roster rather than a resident-level analytic dataset, we interpreted these results as facility-level characterization and excluded-resident stratification rather than a complete resident-level included-versus-excluded facility comparison.

重要提醒：

- `RESULTS` 的主要模型結果原先於 2026-05-08 從原 notebook 的 Google Sheets 重新跑出；2026-05-13 已再用 `DATA` 內新補資料更新排除個案、機構規模/區域、external calibration CI 等補充表。部分 external validation 數值與投稿版摘要不同；若採用重跑結果，Abstract、Results、Table 4 要同步更新。
- 原 notebook 中 `"HybridXGBRF (Our Approach)"` 實際上似乎是 `XGBClassifier` 物件標籤。回覆時建議避免強調 Hybrid 明顯優於 XGBoost，改成「tree-based models showed comparable high discrimination」。
- 本檔已依照 `01_請蔡老師協助的部分.docx` 的順序排列：先 Reviewer W 三項，再 Reviewer BK 九項；2026-05-13 已同步更新新補資料可支援的回答與對應檔案。
- 2026-05-13 已補齊原先未完成的三個子項目：排除者「離開護家的原因」基本分析、排除者機構核定床數/機構大小與區域分層、calibration intercept/slope 的 bootstrap 95% CI。需注意：目前 `area_size.xlsx` 可和新 `DATA` 的排除個案用 `dbname` 合併；原 Google Sheets development/external cohort 經匯入後缺少可用 `dbname` 文字欄位，因此機構大小/區域分層目前主要支援排除者資料，而非完整 included cohort 的機構分層比較。

---

## (Reviewer W)

### 1. 進行排除個案的基本 data 分析

原始需求：

沒有 6 個月死亡 outcome 被排除的個案，進行 baseline data 的分析與比較。比較 included vs excluded 的基本變項，例如年齡、性別、初始 ADL、體重、住院次數、DNR、初始管灌、初始氧氣使用、機構區域。另需增加離開護家的原因基本分析。撈出被排除者的 baseline characteristics，做 included vs excluded 比較表。

目前狀態：

已用 `DATA\analysis_data_filtering_out_included_ADL_missing_0514.csv`、`DATA\analysis_data_filtering_out_0514.csv` 和 `DATA\area_size.xlsx` 補出排除個案 baseline、ADL missing 排除者摘要、離開護家原因、以及排除者機構大小/區域分層。`included_vs_excluded_insufficient_followup.xlsx` 也已產出，included 端使用原 Google Sheets development + external cohort，excluded 端使用 `DATA` 的排除個案資料。

需提醒：included 端目前可比較年齡、性別、DNR、ADL、體重、住院次數、管灌、氧氣使用等欄位；但 included cohort 匯入後沒有可和 `area_size.xlsx` 穩定合併的 `dbname` 文字欄位，因此機構床數/區域分層目前只適合用排除者資料呈現，或等資料端提供 included cohort 的 `dbname`/機構代碼後再補完整。

已補出的檔案：

- `RESULTS\tables\included_vs_excluded_insufficient_followup.xlsx`
- `RESULTS\tables\excluded_residents_baseline_summary.xlsx`
- `RESULTS\tables\excluded_adl_missing_baseline_summary.xlsx`
- `RESULTS\tables\excluded_exit_reason_summary.xlsx`
- `RESULTS\tables\excluded_region_summary.xlsx`
- `RESULTS\tables\facility_size_missingness_and_outcome.xlsx`
- `RESULTS\tables\facility_region_missingness_and_outcome.xlsx`

目前新資料已提供或部分提供欄位：

- age
- sex
- ADL first/max/last if available
- body weight
- hospitalization count
- DNR
- tube feeding
- respiratory support / oxygen use
- falls（目前排除者 `had_fall` 幾乎皆缺失，表格中需保留為資料限制）
- GCS
- facility region（排除者可用 `area_size.xlsx` 合併）
- approved bed capacity / facility size（排除者可用 `area_size.xlsx` 合併）
- exit/discharge reason from LTCF（已由 `area_size.xlsx` 的 `排除個案_結案原因分析` 提供彙整）

目前補出結果重點：

- 排除個案總數：19,756；其中 ADL missing 排除者：6,715。
- Included analytic cohort：30,117；排除個案：19,756。
- 排除者平均年齡 76.6 歲，男性 50.3%，DNR 25.5%，初始 ADL score 32.6，初始體重 55.2 kg。
- 排除者離開/結案原因前三項：返家照護/家屬自行照顧 10,184（51.5%）、空白/未填寫 3,559（18.0%）、轉院/轉介其他機構 3,345（16.9%）。
- 排除者機構大小分層：小型 8,996、中型 8,005、大型 2,115；區域分層以南部最多 13,175。

可貼上的 Response Letter 回覆：

Thank you for pointing out the potential selection bias introduced by excluding residents without sufficient follow-up for 6-month outcome ascertainment. We added a baseline comparison between included residents and residents excluded because of insufficient follow-up. The new supplementary tables compare available demographic, functional, clinical, and care-related characteristics between the two groups and summarize discharge/exit reasons among excluded residents. We also added facility-size and facility-region summaries for excluded residents using approved bed capacity and regional information available from the data source. Because facility identifiers were not consistently available in the analytic Google Sheets extract used for the included cohort, facility-size and region stratification should be interpreted primarily as a characterization of excluded residents. We expanded the Limitations section to clarify how exclusion due to incomplete follow-up and incomplete facility linkage may affect model generalizability.

建議論文修改位置：

- Methods / Study Population：補 excluded residents without sufficient follow-up 的定義。
- Limitations：補 selection bias 與 outcome ascertainment limitation。
- Supplement：若資料端補得出來，新增 `Multimedia Appendix X. Characteristics of included and excluded residents`。
- Supplement：若離開護家原因可取得，新增或併入 `Exit/discharge reasons among excluded residents`。

---

### 2. 補做 distribution histogram of the predicted probabilities in the calibration plots

原始需求：

Calibration plot 下方加 predicted probability histogram，或另做一張圖。X 軸為 predicted risk，Y 軸為 number of residents。可分 internal / external validation。

對應圖檔：

Internal:

- `RESULTS\figures\internal_cv_calibration_with_histogram.png`
- `RESULTS\figures\internal_cv_calibration_with_histogram.html`

External:

- `RESULTS\figures\external_validation_calibration_with_histogram.png`
- `RESULTS\figures\external_validation_calibration_with_histogram.html`

可貼上的 Response Letter 回覆：

Thank you for this helpful suggestion. We revised the calibration plots to include the distribution of predicted probabilities. The upper panel shows the observed versus predicted risk across calibration bins, and the lower panel shows the histogram of predicted probabilities. This addition helps readers interpret both calibration performance and the density of predictions across the risk range.

建議圖說：

Calibration plots with predicted probability distributions. The upper panel shows observed event rates against mean predicted probabilities by risk bins, and the lower panel shows the distribution of predicted 6-month mortality probabilities.

建議論文修改位置：

- Results / Calibration paragraph。
- Figure caption for calibration plots。

---

### 3. 差補資料的寫法需要再次檢視確認

原始需求：

目前寫法為：其他連續變項或量表少量缺失，用 z-score 後補 0，即 development cohort mean。需要確認這些連續變項用來建機器學習模型時，是用原始數值，還是標準化後的數值。

對應檔案：

- `RESULTS\tables\missingness_indicator_development.xlsx`
- `RESULTS\tables\missingness_indicator_external.xlsx`
- `RESULTS\tables\selected_features.xlsx`

可貼上的 Response Letter 回覆：

Thank you for requesting clarification. We revised the preprocessing description to distinguish between binary/count variables and continuous or scale variables. Binary and count variables that are routinely documented when present were coded as 0 when missing, reflecting absence or non-documentation in the source workflow. For continuous or scale variables, missing values were imputed at the development-cohort mean. In implementation, this is equivalent to imputing 0 after z-score standardization when standardized inputs are used. All preprocessing parameters, including means and standard deviations for standardization, were estimated from the development cohort only and then applied unchanged to the external validation cohort to avoid information leakage.

建議 Methods 文字：

For continuous or scale variables, missing values were imputed using the mean estimated from the development cohort. When standardized inputs were used, this corresponded to imputing missing z-scores as 0. All standardization parameters were estimated in the development cohort and applied unchanged to the temporal external validation cohort.

建議論文修改位置：

- Methods / Predictor Measurement and Preprocessing。
- Methods / Missing Data Handling。

---

## (Reviewer BK)

### 4. Paired bootstrap testing or another paired statistical comparison (HybridXGBRF vs XGBoost)

原始需求：

新增 paired bootstrap。在 validation cohort 中重抽樣 1000 或 5000 次，每次保留同一批 bootstrap sample，同時計算 HybridXGBRF 與 XGBoost 的 AUROC。每次計算 `ΔAUROC = AUROC_Hybrid − AUROC_XGB`，取得 ΔAUROC 的 95% CI。若 CI 包含 0，就不能說 Hybrid 顯著優於 XGBoost。

對應表格：

- `RESULTS\tables\paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`
- `RESULTS\tables\paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx`

目前重跑結果：

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
- Bootstrap P = 0.716

可貼上的 Response Letter 回覆：

We agree that the performance difference between HybridXGBRF and XGBoost should not be overstated. We performed paired bootstrap comparisons of AUROC using paired predictions from the same participants. In the temporal external validation cohort, the AUROC difference was 0.0004, with a 95% CI from -0.0017 to 0.0026 and a bootstrap P value of 0.716. These findings indicate that HybridXGBRF and XGBoost had statistically comparable discrimination. We therefore revised the Abstract, Results, and Discussion to avoid claiming clear superiority of the hybrid model and instead describe the tree-based models as showing comparable high discrimination.

建議論文修改：

把「HybridXGBRF achieved superior performance」改成：

HybridXGBRF and XGBoost demonstrated comparable high discrimination, with only a small AUROC difference in paired bootstrap analysis.

---

### 5. Recall, precision, F1, Brier score, calibration measures 都需要補 95% CI

原始需求：

原表格補充 95% CI。增加 calibration measures 表格資料，含 95% CI。

對應表格：

Internal CV:

- `RESULTS\tables\table3_internal_cv_performance_with_ci.xlsx`

External validation:

- `RESULTS\tables\table4_external_validation_full_with_ci.xlsx`
- `RESULTS\tables\table4_external_validation_paper_friendly.xlsx`

Subgroup:

- `RESULTS\tables\table5_subgroup_performance_with_ci.xlsx`

Calibration metrics:

- `RESULTS\tables\calibration_metrics_external_hybridxgbrf.xlsx`
- `RESULTS\tables\calibration_metrics_external_hybridxgbrf_with_ci.xlsx`
- `RESULTS\tables\risk_decile_calibration_external_hybridxgbrf.xlsx`

注意：

- Word 原始需求寫到 calibration measures 也要有 95% CI。2026-05-13 已補出 bootstrap CI 版本：
  - `RESULTS\tables\calibration_metrics_external_hybridxgbrf_with_ci.xlsx`

目前重跑結果重點：

Internal selected model:

- AUROC = 0.875 (95% CI 0.870-0.880)
- Precision = 0.774 (95% CI 0.760-0.787)
- Recall = 0.451 (95% CI 0.438-0.464)
- F1 = 0.570 (95% CI 0.557-0.582)
- Brier = 0.109 (95% CI 0.107-0.110)

External selected model, if using regenerated RESULTS:

- AUROC = 0.887 (95% CI 0.879-0.895)
- Precision = 0.856 (95% CI 0.836-0.878)
- Recall = 0.482 (95% CI 0.461-0.505)
- F1 = 0.617 (95% CI 0.596-0.637)
- Brier = 0.122 (95% CI 0.118-0.125)

External calibration metrics with bootstrap 95% CI:

- Calibration intercept = 0.591 (95% CI 0.525-0.662)
- Calibration slope = 1.323 (95% CI 1.261-1.393)
- Observed/Expected ratio = 1.158 (95% CI 1.138-1.179)
- Brier score = 0.122 (95% CI 0.118-0.125)

可貼上的 Response Letter 回覆：

Thank you for this suggestion. We added uncertainty estimates for the main performance metrics. The revised tables now include 95% confidence intervals for AUROC, accuracy, precision, recall/sensitivity, specificity, F1 score, and Brier score in both internal cross-validation and temporal external validation. We also added 95% confidence intervals for subgroup performance. Calibration was further summarized using calibration intercept, calibration slope, observed/expected ratio, Brier score, and risk-decile calibration, and bootstrap 95% confidence intervals were added for the numerical calibration metrics in the external validation cohort.

建議 Results 文字：

In internal cross-validation, the selected model achieved an AUROC of 0.875 (95% CI 0.870-0.880), precision of 0.774 (95% CI 0.760-0.787), recall of 0.451 (95% CI 0.438-0.464), F1 score of 0.570 (95% CI 0.557-0.582), and Brier score of 0.109 (95% CI 0.107-0.110).

---

### 6. 是否可做 threshold-specific tradeoffs

原始需求：

可選幾個代表性 threshold，例如最高 sensitivity、Youden index、或原本 threshold 附近範圍。需要補做 decision curve。PB/PPV 已有部分資料，但 NPV、threshold-specific tradeoffs 需補完整。

對應表格：

Threshold-specific tradeoff:

- `RESULTS\tables\threshold_tradeoff_external_hybridxgbrf.xlsx`

目前重跑結果重點：

At threshold 0.20:

- Sensitivity = 0.844
- Specificity = 0.736
- PPV = 0.562
- NPV = 0.921

At threshold 0.50:

- Sensitivity = 0.482
- Specificity = 0.968
- PPV = 0.856
- NPV = 0.823

可貼上的 Response Letter 回覆：

Thank you for this suggestion. We added threshold-specific analyses in the temporal external validation cohort, including sensitivity, specificity, PPV, NPV, F1 score, and confusion-matrix counts across clinically relevant thresholds. These analyses show how lower thresholds may be useful for screening and prompting clinical review, whereas higher thresholds may be more appropriate for prioritizing limited supportive-care resources.

建議 Results 文字：

At a threshold of 0.20, the model prioritized sensitivity, with sensitivity 0.844, specificity 0.736, PPV 0.562, and NPV 0.921. At a threshold of 0.50, the model prioritized PPV and specificity, with sensitivity 0.482, specificity 0.968, PPV 0.856, and NPV 0.823.

---

### 7. 需要補做 decision curve

原始需求：

需要補做 decision curve。PB/PPV 我們已有部分資料，但 NPV、threshold-specific tradeoffs 需補完整。

對應表格與圖：

Decision curve:

- `RESULTS\figures\decision_curve_external_validation.png`
- `RESULTS\tables\decision_curve_external_validation.xlsx`

Internal decision curve:

- `RESULTS\figures\decision_curve_internal_cv.png`
- `RESULTS\tables\decision_curve_internal_cv.xlsx`

可貼上的 Response Letter 回覆：

We agree that AUROC alone does not establish clinical utility. We therefore added decision-curve analysis comparing the selected model with alternative models and treat-all/treat-none strategies. The decision-curve analysis summarizes net benefit across clinically relevant threshold probabilities and complements the threshold-specific performance table.

---

### 8. 是否能補做 survival analysis

原始需求：

如果有 time-to-death 和 censoring time，survival analysis 應可做。可補 time-dependent AUC，例如 1、2、3、4、5、6 個月或 6 個月一個時間點。

已產出檔案：

程式：

- `C:\AI4JUBO\survival_generate_results.py`

表格：

- `RESULTS\tables\survival_time_dependent_auc.xlsx`
- `RESULTS\tables\survival_c_index.xlsx`
- `RESULTS\tables\survival_cox_ml_risk_score.xlsx`
- `RESULTS\tables\survival_cox_baseline_development_top_features.xlsx`
- `RESULTS\tables\survival_km_external_risk_group_summary.xlsx`
- `RESULTS\tables\survival_km_curve_external_by_risk_group.xlsx`
- `RESULTS\tables\survival_logrank_external_risk_groups.xlsx`

圖檔：

- `RESULTS\figures\survival_time_dependent_auc.png`
- `RESULTS\figures\survival_km_external_by_risk_group.png`

說明：

- `RESULTS\survival_analysis_note.txt`

分析方式：

這是一組 sensitivity analysis，不取代主要 binary 180-day mortality model。使用 `觀察天數` 作為 time-to-event / follow-up time，`死亡標記` 作為 event indicator，180 天作為 administrative horizon。未於 180 天內死亡者 censor at `min(觀察天數, 180)`。

目前結果重點：

C-index:

- Development cohort: Harrell C-index = 0.854
- Temporal external validation cohort: Harrell C-index = 0.849

Cox model using ML predicted risk score:

- Development cohort: HR per 0.10 predicted-risk increase = 1.67 (95% CI 1.66-1.69)
- Temporal external validation cohort: HR per 0.10 predicted-risk increase = 1.65 (95% CI 1.62-1.68)

External time-dependent AUC:

- 1 month: 0.907
- 2 months: 0.912
- 3 months: 0.901
- 4 months: 0.893
- 5 months: 0.889
- 6 months: not estimable in this cumulative/dynamic formulation because all non-events are administratively censored at 180 days, leaving no event-free controls beyond t.

External risk groups:

- Low predicted risk: 53/2072 deaths within 180 days (2.6%)
- Medium predicted risk: 371/2072 deaths within 180 days (17.9%)
- High predicted risk: 1357/2072 deaths within 180 days (65.5%)
- Global log-rank test across risk groups: chi-square = 2740.3, P < .001

可貼上的 Response Letter 回覆：

We agree that time-to-event analyses provide an important complementary perspective to the binary 6-month mortality endpoint. We therefore added survival sensitivity analyses using the available observation time and death indicator. We used a 180-day administrative horizon; residents without death before 180 days were censored at the earlier of their observed follow-up time or 180 days. These analyses were presented as sensitivity analyses because post-discharge outcome ascertainment and censoring mechanisms may be incomplete in the analytic dataset.

In the temporal external validation cohort, the model-based risk score showed strong time-to-event discrimination, with a Harrell C-index of 0.849. In a Cox proportional hazards model, each 0.10 increase in predicted risk was associated with a higher hazard of death (HR 1.65, 95% CI 1.62-1.68). Cumulative/dynamic AUCs were 0.907, 0.912, 0.901, 0.893, and 0.889 at 1, 2, 3, 4, and 5 months, respectively. Kaplan-Meier curves stratified by predicted-risk tertiles showed clear separation across risk groups; 180-day mortality was 2.6%, 17.9%, and 65.5% in the low-, medium-, and high-risk groups, respectively (global log-rank P < .001).

建議 Methods 文字：

As a sensitivity analysis, we evaluated time-to-event performance using available observation time and death status. Follow-up time was administratively truncated at 180 days. Residents without death before 180 days were censored at the earlier of observed follow-up time or 180 days. We summarized survival discrimination using Harrell's C-index and cumulative/dynamic AUCs at monthly horizons. We also fitted Cox proportional hazards models using the model-predicted risk score and plotted Kaplan-Meier curves by predicted-risk tertiles in the external validation cohort.

建議 Limitations 文字：

The survival analyses should be interpreted as sensitivity analyses because the source dataset was primarily structured for 180-day binary outcome prediction. Post-discharge mortality ascertainment and censoring mechanisms may be incomplete for some residents, and the primary analysis therefore remains the prespecified binary 6-month mortality model.

---

### 9. 進行測試有缺值的 missing 指標

原始需求：

每個重要變項新增一個是否有 missing 的欄位，1/0，來做比較分析。可以比較是否有機構差異、結果差異。

Word 原文另有資料端需求：

- 請確認是否能提供機構區域。
- 請確認是否能提供機構核定床數。
- 若可取得核定床數，可依機構規模分層，例如小規模 `<50` 床、大規模 `>150` 床。
- 若有品質指標或 `品質X` 欄位，也可納入 missingness 或 outcome 差異比較。

對應表格：

Missingness indicators:

- `RESULTS\tables\missingness_indicator_development.xlsx`
- `RESULTS\tables\missingness_indicator_external.xlsx`

Facility-level missingness:

- `RESULTS\tables\facility_missingness_development.xlsx`
- `RESULTS\tables\facility_missingness_external.xlsx`

機構規模/區域分層：

- `RESULTS\tables\facility_size_missingness_and_outcome.xlsx`
- `RESULTS\tables\facility_region_missingness_and_outcome.xlsx`

目前狀態：

- 2026-05-13 已用 `DATA\area_size.xlsx` 補出排除個案的機構核定床數、機構大小層級與區域分層。
- `facility_size_missingness_and_outcome.xlsx` 目前包含排除者小/中/大型機構分層：小型 8,996、中型 8,005、大型 2,115。
- `facility_region_missingness_and_outcome.xlsx` 目前包含排除者區域分層：北部 1,962、中部 3,818、南部 13,175、東部 161。
- `facility_missingness_development/external` 仍受原 Google Sheets 匯入欄位限制；若要對 included analytic cohort 做正式機構大小/區域比較，需要資料端提供可合併 `area_size.xlsx` 的 included cohort `dbname` 或機構代碼。

可貼上的 Response Letter 回覆：

Thank you for this important point. We added missingness indicator summaries to evaluate whether missingness differed by outcome status and to better characterize documentation patterns in the routine-care dataset. We also summarized facility-level missingness to assess heterogeneity in documentation completeness across facilities. Using newly available approved bed capacity and regional information, we further summarized excluded residents by facility size and facility region to examine documentation completeness and follow-up patterns across facility strata. Because the analytic Google Sheets extract did not retain a stable facility identifier for all included residents, facility-size and region analyses were interpreted as supplementary characterization of excluded residents, and the Limitations section was expanded to discuss documentation bias, facility-level variation, and incomplete facility linkage.

建議 Limitations 文字：

Because the study used routinely collected LTCF data, missingness may reflect both true absence of a condition and documentation practices. Facility-level differences in documentation completeness may also contribute to measurement heterogeneity. We therefore added missingness indicator and facility-level missingness summaries and interpreted model results in light of potential documentation bias.

---

### 10. 請老師協助補充說明：SHAP 解釋的是混合 HybridXGBRF 的輸出，還是僅解釋其中一個模型

原始需求：

需要確認 SHAP 圖到底是哪個 model 做的。

對應檔案：

SHAP table:

- `RESULTS\tables\shap_feature_importance.xlsx`

SHAP figure:

- `RESULTS\figures\shap_feature_importance.png`

目前重跑 SHAP 結果：

這次補出的 SHAP 使用 KernelSHAP，套用於 final `predict_proba` output。

Top features:

1. Body Weight Change
2. Hospitalizations within 6 Months
3. ADL Last Score
4. ADL Minimum
5. ADL Standard Deviation
6. Male
7. Body Weight (Last)
8. Body Weight (First)

可貼上的 Response Letter 回覆：

Thank you for identifying this ambiguity. We clarified the SHAP analysis and regenerated the SHAP feature-importance results using a model-agnostic KernelSHAP approach applied to the final `predict_proba` output. Therefore, the revised SHAP results explain the final predicted probability rather than only an intermediate model component. We revised the Methods and figure caption accordingly.

建議 Methods 文字：

To avoid ambiguity regarding component-level explanations, SHAP values in the revised analysis were estimated using a model-agnostic KernelSHAP approach applied to the final predicted probability output.

---

### 11. 需要補 calibration metrics，包括 intercept 和 slope

原始需求：

需要補 calibration metrics，包括 intercept 和 slope。

對應表格與圖：

Calibration metrics:

- `RESULTS\tables\calibration_metrics_external_hybridxgbrf.xlsx`

Risk-decile calibration:

- `RESULTS\tables\risk_decile_calibration_external_hybridxgbrf.xlsx`

Calibration plot:

- `RESULTS\figures\external_validation_calibration_with_histogram.png`
- `RESULTS\figures\internal_cv_calibration_with_histogram.png`

目前重跑結果：

- Calibration intercept = 0.591 (95% CI 0.525-0.662)
- Calibration slope = 1.323 (95% CI 1.261-1.393)
- Observed deaths = 1781
- Expected deaths = 1538.1
- O/E ratio = 1.158 (95% CI 1.138-1.179)
- Brier score = 0.122 (95% CI 0.118-0.125)

可貼上的 Response Letter 回覆：

We agree that graphical calibration alone is insufficient. We added numerical calibration metrics, including calibration intercept, calibration slope, observed/expected ratio, Brier score, and risk-decile calibration in the temporal external validation cohort. We also added bootstrap 95% confidence intervals for the numerical calibration metrics. These results are now presented in the supplementary materials, and the calibration figure was revised to include the predicted-probability distribution.

---

### 12. 是否能提供 code、random seed、package versions、preprocessing 細節

原始需求：

是否能提供 code、random seed、package versions、preprocessing 細節。

對應檔案：

程式：

- `C:\AI4JUBO\revision_generate_results.py`
- `C:\AI4JUBO\survival_generate_results.py`

Package versions:

- `C:\AI4JUBO\RESULTS\package_versions.txt`

Features:

- `RESULTS\tables\selected_features.xlsx`

Model identity note:

- `RESULTS\model_identity_note.txt`

可貼上的 Response Letter 回覆：

Thank you for this suggestion. We revised the reproducibility description to include preprocessing details, model settings, random seeds, and package versions. The analysis used fixed random seeds for cross-validation, model fitting where applicable, and bootstrap resampling. We also prepared analysis scripts documenting preprocessing, model evaluation, bootstrap confidence intervals, calibration analyses, threshold-specific analyses, decision-curve analysis, SHAP workflow, and survival sensitivity analyses. Because the individual-level data cannot be publicly shared due to privacy and data-use restrictions, the code and preprocessing details are provided to support reproducibility within the constraints of data governance.

建議 Data/Code Availability 文字：

The individual-level dataset is not publicly available because of privacy and data-use restrictions. The analysis pipeline, preprocessing details, random seeds, and package versions are documented in the revision analysis scripts and supplementary reproducibility materials. All preprocessing parameters were estimated in the development cohort and applied unchanged to the temporal external validation cohort.

---

## 依照 Word 順序的貼表 / 貼圖總清單與對應解釋

### Reviewer W 對應檔案

1. 排除個案 baseline analysis

   對應檔案：

   - `RESULTS\tables\included_vs_excluded_insufficient_followup.xlsx`
   - `RESULTS\tables\excluded_residents_baseline_summary.xlsx`
   - `RESULTS\tables\excluded_adl_missing_baseline_summary.xlsx`
   - `RESULTS\tables\excluded_exit_reason_summary.xlsx`
   - `RESULTS\tables\excluded_region_summary.xlsx`
   - `RESULTS\tables\facility_size_missingness_and_outcome.xlsx`
   - `RESULTS\tables\facility_region_missingness_and_outcome.xlsx`

   對應解釋：

   這些檔案用來回答 Reviewer W 對 selection bias 的疑慮。`included_vs_excluded_insufficient_followup.xlsx` 比較納入分析者與因無法確認 6 個月死亡 outcome 而排除者的 baseline characteristics；`excluded_residents_baseline_summary.xlsx` 與 `excluded_adl_missing_baseline_summary.xlsx` 進一步整理排除者與 ADL missing 排除者特徵；`excluded_exit_reason_summary.xlsx` 補充排除者離開護家或追蹤中斷原因；`facility_size_missingness_and_outcome.xlsx` 與 `facility_region_missingness_and_outcome.xlsx` 則用來呈現排除者的機構床數/大小與區域分層。需在回覆信註明 included cohort 的機構大小/區域仍缺穩定機構代碼可合併，因此這部分目前作為排除者補充分析。

2. Calibration histogram

   對應檔案：

   - `RESULTS\figures\internal_cv_calibration_with_histogram.png`
   - `RESULTS\figures\external_validation_calibration_with_histogram.png`

   對應解釋：

   這兩張圖用來回答 reviewer 要求在 calibration plot 加上 predicted probability distribution 的意見。圖的上半部呈現 predicted risk 與 observed event rate 的 calibration；下半部 histogram 顯示模型預測風險分布，讓讀者知道大多數住民落在哪些 risk range。

3. Imputation / missingness

   對應檔案：

   - `RESULTS\tables\missingness_indicator_development.xlsx`
   - `RESULTS\tables\missingness_indicator_external.xlsx`
   - `RESULTS\tables\selected_features.xlsx`

   對應解釋：

   這些檔案用來支持 missing-data handling 的說明。`missingness_indicator_*` 表格可呈現各重要變項缺值是否與 outcome 相關；`selected_features.xlsx` 則可確認模型實際使用的變項。回覆時要特別說清楚：連續/量表變項是在 development cohort 估計平均值與標準差，標準化後 missing z-score 補 0，等同補 development-cohort mean；binary/count 變項則依資料紀錄邏輯處理。

### Reviewer BK 對應檔案

1. Paired bootstrap

   對應檔案：

   - `RESULTS\tables\paired_bootstrap_auroc_internal_hybrid_vs_xgb.xlsx`
   - `RESULTS\tables\paired_bootstrap_auroc_external_hybrid_vs_xgb.xlsx`

   對應解釋：

   這兩張表用來回答 HybridXGBRF 是否真的優於 XGBoost。paired bootstrap 使用同一批 bootstrap sample 同時計算兩個模型的 AUROC，再計算 ΔAUROC 與 95% CI。External validation 的 ΔAUROC 很小且 CI 包含 0，因此回覆時不應主張 Hybrid 顯著優於 XGBoost，應改寫成兩者 discrimination 相近，較簡單的 XGBoost 可能已足夠。

2. Metrics 95% CI

   對應檔案：

   - `RESULTS\tables\table3_internal_cv_performance_with_ci.xlsx`
   - `RESULTS\tables\table4_external_validation_paper_friendly.xlsx`
   - `RESULTS\tables\table4_external_validation_full_with_ci.xlsx`
   - `RESULTS\tables\table5_subgroup_performance_with_ci.xlsx`

   對應解釋：

   這些表用來補 reviewer 要求的 recall、precision、F1、Brier score 等指標的 95% CI。`table3` 對應 internal cross-validation，`table4` 對應 temporal external validation，`table5` 對應 subgroup performance。`paper_friendly` 版本適合放進正文或 supplement，`full_with_ci` 版本保留較完整欄位供檢查。

3. Threshold-specific tradeoffs

   對應檔案：

   - `RESULTS\tables\threshold_tradeoff_external_hybridxgbrf.xlsx`

   對應解釋：

   這張表用來回答不同 threshold 下 sensitivity、specificity、PPV、NPV、F1 與 confusion-matrix counts 如何變化。它可以支持「低 threshold 適合提高 sensitivity、作為篩檢/提醒；高 threshold 適合提高 PPV/specificity、用於資源優先排序」的回覆。

4. Decision curve

   對應檔案：

   - `RESULTS\figures\decision_curve_external_validation.png`
   - `RESULTS\tables\decision_curve_external_validation.xlsx`
   - `RESULTS\figures\decision_curve_internal_cv.png`
   - `RESULTS\tables\decision_curve_internal_cv.xlsx`

   對應解釋：

   Decision curve 用來回答 AUROC 以外的 clinical utility 問題。圖表呈現不同 threshold probability 下的 net benefit，並與 treat-all / treat-none strategy 比較。這可以放在 Results 或 supplement，文字上要說明模型是輔助 risk stratification 和 clinical review，不是單獨決策工具。

5. Survival

   對應檔案：

   - `RESULTS\tables\survival_time_dependent_auc.xlsx`
   - `RESULTS\tables\survival_c_index.xlsx`
   - `RESULTS\tables\survival_cox_ml_risk_score.xlsx`
   - `RESULTS\tables\survival_km_external_risk_group_summary.xlsx`
   - `RESULTS\figures\survival_time_dependent_auc.png`
   - `RESULTS\figures\survival_km_external_by_risk_group.png`

   對應解釋：

   這些檔案用來回答 reviewer 對 binary 6-month outcome 忽略 censoring/time-to-event 的疑慮。`survival_c_index` 和 `survival_time_dependent_auc` 呈現 time-to-event discrimination；`survival_cox_ml_risk_score` 檢查模型預測風險是否和死亡 hazard 相關；KM 圖與 risk-group summary 顯示低、中、高風險組的存活曲線明顯分離。這組分析應定位為 sensitivity analysis，不取代主要 binary 180-day model。

6. Missing indicators / facility missingness

   對應檔案：

   - `RESULTS\tables\missingness_indicator_development.xlsx`
   - `RESULTS\tables\missingness_indicator_external.xlsx`
   - `RESULTS\tables\facility_missingness_development.xlsx`
   - `RESULTS\tables\facility_missingness_external.xlsx`

   機構規模/區域分層檔案：

   - `RESULTS\tables\facility_size_missingness_and_outcome.xlsx`
   - `RESULTS\tables\facility_region_missingness_and_outcome.xlsx`

   對應解釋：

   `missingness_indicator_*` 表格用來比較重要變項是否缺值與 outcome 的關聯；`facility_missingness_*` 表格用來檢查不同機構是否有文件紀錄完整度差異。2026-05-13 已用 `area_size.xlsx` 補出排除者的 facility size/region stratification，回應 Word 原文提到的小規模 `<50` 床、大規模 `>150` 床，以及機構差異/結果差異。若要把 included analytic cohort 也納入同一套機構分層，仍需 included cohort 的穩定 `dbname` 或機構代碼。

7. SHAP

   對應檔案：

   - `RESULTS\tables\shap_feature_importance.xlsx`
   - `RESULTS\figures\shap_feature_importance.png`

   對應解釋：

   這兩個檔案用來回答 SHAP 到底解釋哪個 model 的問題。此次補做的 SHAP 是 model-agnostic KernelSHAP，套用在 final `predict_proba` output，因此回覆時可說 revised SHAP explains the final predicted probability output，而不是只解釋單一 component model。注意 manuscript 也要同步釐清 HybridXGBRF 的模型名稱與實際模型物件。

8. Calibration metrics

   對應檔案：

   - `RESULTS\tables\calibration_metrics_external_hybridxgbrf.xlsx`
   - `RESULTS\tables\calibration_metrics_external_hybridxgbrf_with_ci.xlsx`
   - `RESULTS\tables\risk_decile_calibration_external_hybridxgbrf.xlsx`
   - `RESULTS\figures\external_validation_calibration_with_histogram.png`

   對應解釋：

   `calibration_metrics_external_hybridxgbrf.xlsx` 用來補 calibration intercept、slope、observed/expected ratio、Brier score；`calibration_metrics_external_hybridxgbrf_with_ci.xlsx` 已補 bootstrap 95% CI；`risk_decile_calibration_*` 用來呈現各風險分層的 predicted vs observed risk；calibration histogram 圖用來視覺化校準與預測風險分布。

9. Code / seed / package / preprocessing

   對應檔案：

   - `C:\AI4JUBO\revision_generate_results.py`
   - `C:\AI4JUBO\survival_generate_results.py`
   - `C:\AI4JUBO\RESULTS\package_versions.txt`
   - `RESULTS\tables\selected_features.xlsx`

   對應解釋：

   這些檔案用來回答 reproducibility。`revision_generate_results.py` 包含主要 performance、bootstrap CI、threshold、DCA、calibration、missingness、SHAP 的產出流程；`survival_generate_results.py` 包含 survival sensitivity analysis；`package_versions.txt` 對應套件版本；`selected_features.xlsx` 對應模型使用變項。回覆信可說明 code、random seed、package versions、preprocessing details 已整理，但 individual-level data 因隱私與資料使用限制不能公開。
