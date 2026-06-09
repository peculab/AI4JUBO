# 0609 請蔡老師協助部分：結果檔案對應

來源 PDF：`0609_請蔡老師協助的部分.pdf`

本資料夾已依 PDF 逐項整理結果。若要快速檢查全部需求，優先開啟：

`0609_development_facility_missingness_form.xlsx`

## 1. Internal calibration metrics with 95% CI

PDF 需求：

`calibration_metrics_internal_hybridxgbrf_with_ci`

請看：

- `calibration_metrics_internal_hybridxgbrf_with_ci.xlsx`
- `0609_development_facility_missingness_form.xlsx`，sheet `Internal calibration CI`

## 2. Internal risk-decile calibration

PDF 需求：

`risk_decile_calibration_internal_hybridxgbrf`

請看：

- `risk_decile_calibration_internal_hybridxgbrf.xlsx`
- `0609_development_facility_missingness_form.xlsx`，sheet `Internal risk decile`

## 3. Development cohort facility size / institutional region missingness table

PDF 需求欄位：

- Variable
- N facilities
- N residents
- ALL Feature Missing Percent (Overall missing percent)
- ALL Feature Missing Percent / Dead cohort
- ALL Feature Missing Percent / Alive cohort
- Death rate

正式保守版請看：

- `0609_development_facility_missingness_form.xlsx`，sheet `Requested form`
- `0609_development_facility_missingness_form.csv`

因 Development cohort cache 內 `dbname` 全部缺失，resident-level facility size / region 欄位無法由目前專案檔案可靠估計。`Requested form` 中只填入可由 `DATA/area_size.xlsx` 可靠取得的 `N facilities`，其餘 resident-level 欄位標示為不可估計。

若要看使用目前本地資料嘗試對應 `dbname` 後的完整探索性數字，請看：

- `0609_development_facility_missingness_form.xlsx`，sheet `Exploratory mapped form`
- `0609_exploratory_mapped_facility_missingness_form.csv`

探索性 mapping 的覆蓋率與限制請看：

- `0609_development_facility_missingness_form.xlsx`，sheet `Exploratory mapping audit`
- `0609_development_facility_missingness_form.xlsx`，sheet `H01 linkage method`

注意：探索性表使用其他本地資料建立 `H01_NUM -> dbname` 眾數對照後再接 `DATA/area_size.xlsx`。雖可對到大部分 rows，但許多 `H01_NUM` 有多個候選 `dbname`，因此不應視為 confirmed resident-level facility linkage。

## 4. Chi-square test: institution ID x overall missing percent

PDF 需求：

卡方檢定：機構 ID x Overall missing percent (ALL features 平均缺失率)

請看：

- `0609_development_facility_missingness_form.xlsx`，sheet `Chi-square`
- `0609_h01num_missingness_chi_square.csv`

目前因 Development cohort cache 缺少可用 `dbname`，此檢定以 `H01_NUM` 作為唯一可用的 repeated identifier，檢定 `H01_NUM x all-feature missing/observed cells`。此結果應視為 exploratory，除非確認 `H01_NUM` 即為本研究要使用的機構 ID。

## 5. ALL Feature 清單

PDF 問題：

「ALL Feature 是不是下面這些？」

請看：

- `0609_development_facility_missingness_form.xlsx`，sheet `Feature list`
- 來源：`..\..\RESULTS\tables\shap_feature_importance.xlsx`

結論：是。ALL Feature 使用 PDF 第 2-3 頁列出的 29 個模型 predictor features，與 `shap_feature_importance.xlsx` 一致。`selected_features.xlsx` 中的 `死亡標記` 是 outcome，不納入 all-feature missingness 的 predictor feature 計算。

## 6. missingness_indicator_key_features_regression 補意識總分Max_missing 與 P value

PDF 需求：

`missingness_indicator_key_features_regression` 增加 `意識總分Max_missing`，並呈現 P 值。

請看：

- `0609_missingness_indicator_key_features_regression_with_p.xlsx`
- `0609_missingness_indicator_key_features_regression_with_p.csv`
- `0609_development_facility_missingness_form.xlsx`，sheet `Key missingness regression`

已新增：

- `Consciousness_total_max_missing`
- Source variable：`意識總分Max`
- P value / formatted P value

## 7. 可重跑程式

本次 0609 產出程式：

- `generate_0609_facility_missingness_form.py`

此程式會重建主 workbook、探索性 mapping 表、chi-square 表，以及補 P 值的 missingness indicator regression 表。
