# 0609 請蔡老師協助部分：結果檔案對應與機構數修正

來源 PDF：`0609_請蔡老師協助的部分.pdf`

本資料夾已依 0609 PDF 需求重新整理。請優先開啟：

`0609_development_facility_missingness_form.xlsx`

## 最重要結論

原先看到的三個機構相關數字不可混用：

- `493`：目前可由原始 `training_data_1014` Google Sheet 與 `DATA/area_size.xlsx` 共同確認的 Development cohort LTCF / `dbname` 數。
- `2,057`：Development cohort 中的 unique `H01_NUM` groups，不是 LTCF 家數。
- `122`：舊版 exploratory modal `H01_NUM -> dbname` mapping 後得到的 mapped dbname 數，已不作為正式結果。

因此，正式 facility-level 分析均改用 confirmed resident-level `dbname`。

## 為什麼本地 cache 的 dbname 是空的

`Revision/20260523/training_data_1014_cached_for_completion.csv` 中 `dbname` 全部缺失，是因為原始 notebook/script 對整份 Google Sheet 做了 numeric coercion：

`pd.to_numeric(..., errors="coerce")`

文字型機構代碼，例如 `C001`，因此被轉成 `NaN`。

但是原始 `training_data_1014` Google Sheet 仍保留 confirmed resident-level `dbname`。經驗證，原始 Google Sheet 經 numeric-clean 後與本地 cache row-by-row 一致，所以可以安全地用原始 sheet 的 `dbname` 接回本地分析資料。

目前可驗證的 Development cohort 數字：

- Residents：23,901
- Deaths：5,272
- Confirmed unique `dbname`：493
- Unique `H01_NUM`：2,057，僅作診斷用，不是機構數
- `DATA/area_size.xlsx` facility roster：493 家

## 正式使用的檔案

### 1. Facility size / institutional region missingness table

正式表：

- `0609_development_facility_missingness_form.xlsx`，sheet `Requested form`
- `0609_development_facility_missingness_form.csv`

此表已使用原始 Google Sheet 的 confirmed `dbname`，並 merge 至 `DATA/area_size.xlsx`。因此現在可以正式呈現：

- N facilities
- N residents
- ALL Feature Missing Percent overall
- ALL Feature Missing Percent among dead residents
- ALL Feature Missing Percent among alive residents
- Death rate

分層合計檢查：

- Facility size strata：493 facilities，23,901 residents
- Institutional region strata：493 facilities，23,901 residents

### 2. Chi-square test by confirmed institution ID

正式 facility-level chi-square：

- `0609_development_facility_missingness_form.xlsx`，sheet `Chi-square dbname`
- `0609_dbname_missingness_chi_square.csv`

檢定為：

`dbname x all-feature missing/observed cells`

結果：

- N `dbname` groups：493
- Total feature cells：693,129
- Missing feature cells：54,243
- Observed feature cells：638,886
- Chi-square statistic：140,785.975
- df：492
- P value：<0.001
- Cramer's V：0.451

### 3. dbname-level missingness detail

- `0609_dbname_missingness_detail.csv`
- `0609_development_facility_missingness_form.xlsx`，sheet `dbname missingness detail`

此檔列出每個 confirmed `dbname` 的 residents/rows、missing feature cells、observed feature cells，以及 overall missing percent。

## 診斷用或已停用的檔案

### H01_NUM diagnostic only

- `0609_h01num_missingness_chi_square.csv`
- `0609_development_facility_missingness_form.xlsx`，sheet `H01 diagnostic`

`H01_NUM` 有 2,057 groups，因此不可解讀為 LTCF 家數。此檔僅保留作為 documentation/resident identifier 診斷，不作為正式 facility-level chi-square。

### 舊 exploratory mapping 已停用

- `0609_exploratory_mapped_facility_missingness_form.csv`
- `0609_exploratory_mapping_audit.csv`

這兩個檔案已改為 `Deprecated` 說明。因為 confirmed `dbname` 已可由原始 Google Sheet 取得，不再需要使用 modal `H01_NUM -> dbname` exploratory mapping。

## 其他 0609 輸出

### Internal calibration metrics with 95% CI

- `calibration_metrics_internal_hybridxgbrf_with_ci.xlsx`
- `calibration_metrics_internal_hybridxgbrf_with_ci.csv`
- `0609_development_facility_missingness_form.xlsx`，sheet `Internal calibration CI`

### Internal risk-decile calibration

- `risk_decile_calibration_internal_hybridxgbrf.xlsx`
- `risk_decile_calibration_internal_hybridxgbrf.csv`
- `0609_development_facility_missingness_form.xlsx`，sheet `Internal risk decile`

### ALL Feature 清單

- `0609_development_facility_missingness_form.xlsx`，sheet `Feature list`
- 來源：`../../RESULTS/tables/shap_feature_importance.xlsx`

ALL Feature 使用 29 個模型 predictor features。`死亡標記` 是 outcome，不納入 all-feature missingness 的 predictor feature 計算。

### Missingness indicator regression with P values

- `0609_missingness_indicator_key_features_regression_with_p.xlsx`
- `0609_missingness_indicator_key_features_regression_with_p.csv`
- `0609_development_facility_missingness_form.xlsx`，sheet `Key missingness regression`

已新增：

- `Consciousness_total_max_missing`
- Source variable：`意識總分Max`
- P value / formatted P value

### SHAP forest plot

依照參考圖樣式重新繪製的 SHAP forest plot：

- `0609_shap_forest_plot_top8.png`
- `0609_shap_forest_plot_top8.jpg`
- `0609_shap_forest_plot_top8.html`
- `0609_shap_forest_plot_data.xlsx`
- `0609_shap_forest_plot_data.csv`

此圖使用目前最新正式 SHAP feature importance 表：

- `../../RESULTS/tables/shap_feature_importance.csv`

Forest plot 的 point estimate 使用該表中的 `MeanAbsSHAP`。誤差線使用 temporal external validation cohort 上的 XGBoost TreeSHAP contribution 進行 bootstrap 後估計 interval width，並縮放回正式 `MeanAbsSHAP` 尺度。因此圖的 x 軸與目前正式 SHAP summary table 一致，約為 0.00 到 0.10。

Top 8 features：

- Hospitalizations within 6 Months
- ADL Change
- Body Weight Change
- ADL Last Score
- ADL Total Max
- Male
- ADL Maximum
- ADL Standard Deviation

## 可重跑程式

- `generate_0609_facility_missingness_form.py`
- `redraw_shap_forest_plot.py`

`generate_0609_facility_missingness_form.py` 會重新讀取原始 Google Sheet 的 confirmed `dbname`，重建：

- 主 workbook
- confirmed dbname facility-level table
- dbname chi-square table
- dbname-level missingness detail
- H01_NUM diagnostic table
- missingness indicator regression with P values

`redraw_shap_forest_plot.py` 會重建：

- SHAP forest plot PNG / JPG / HTML
- SHAP forest plot data CSV / XLSX

## 建議 manuscript / reviewer response 口徑

若沒有其他舊版原始資料能支持 497 家，建議主文與回覆統一改為：

> After excluding individuals who did not meet the inclusion criteria, 23,901 residents from 493 LTCFs were included in the model development cohort.

若希望保留較保守但不精確的敘述，也可寫：

> ... from approximately 500 LTCFs ...

但不建議再使用 497，因為目前專案內可由 raw sheet 與 facility roster 共同驗證的數字是 493。
