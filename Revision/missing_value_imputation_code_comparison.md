# 缺失值補值方法與程式碼比對說明

本文整理最終投稿與 reviewer response 可使用的 missing data / imputation 說明，並對照目前 `..` 的更新後程式碼。此次已將程式碼調整為與原始投稿文字一致：二元/計數型欄位缺失補 `0`；連續/量表欄位先用 development cohort 的 mean 與 SD 做 z-score normalization，缺失的 standardized value 補 `0`，等同於在原始尺度補 development-cohort mean。

## 投稿可用描述

本研究在建模前先進行缺失資料篩選與補值。所有候選變項先以 development cohort 的缺失比例進行篩選；缺失比例達 30% 或以上的變項不納入主要機器學習特徵矩陣。此門檻只由 development cohort 決定，之後相同的變項集合會原封不動套用於 temporal external validation cohort，以避免使用外部驗證資料決定特徵選擇。

納入模型的變項皆轉換為數值型態。二元或計數型照護紀錄欄位若缺失，補為 `0`，代表 absence、未記錄事件、或未標記該照護狀態。連續變項與量表分數則使用 development cohort 估計 mean 與 SD 後進行 z-score normalization；缺失的 standardized values 補為 `0`，因此等同於在原始尺度補 development-cohort mean。所有 development cohort 估計出的 preprocessing 參數會固定套用於 temporal external validation cohort，不會用 external cohort 重新估計，以避免 information leakage。

內部 5-fold cross-validation 中，特徵篩選由 development cohort 的整體規則決定；每一個 fold 內的連續/量表變項 mean 與 SD 只使用該 fold 的 training partition 估計，再套用到同 fold 的 validation partition。最終外部驗證模型則使用完整 development cohort 估計 preprocessing 參數後訓練，並套用到 2024 temporal external validation cohort。

## 程式碼比對

| 步驟 | 更新後處理方式 | 程式碼位置 |
| --- | --- | --- |
| 讀入資料 | 讀取 Google Sheets 或本地 CSV/XLSX；移除逗號與前後空白後轉 numeric，無法轉換者成為 `NaN`。 | `revision_generate_results.py::numeric_clean()` |
| 特徵篩選 | 以 development cohort 計算每欄缺失比例，保留 missingness `< 0.30` 的欄位。 | `revision_generate_results.py::select_features()` |
| 非模型欄位 | `H01_NUM` 與 `觀察天數` 從主要分類特徵矩陣移除。 | `revision_generate_results.py::select_features()` |
| 二元/計數欄位 | 缺失補 `0`。 | `ZERO_IMPUTE_FEATURES`, `apply_preprocessing()` |
| 連續/量表欄位 | 用 training data 的 mean/SD 轉為 z-score；缺失 z-score 補 `0`。 | `fit_preprocessing()`, `apply_preprocessing()` |
| 外部驗證 | 使用 full development cohort fitted preprocessing，不使用 external cohort 重估 mean/SD。 | `main()`, `external_validation()` |
| 內部 CV | 每個 fold 用 training fold fitted preprocessing，再轉換 validation fold。 | `internal_cv(raw_df=train_df, features=features)` |
| SHAP | 使用同一套 development-fitted preprocessing 後的 external validation feature matrix。 | `shap_outputs(fitted_models[main_model], ex_X, ...)` |
| Survival ML risk sensitivity | 使用同一套 development-fitted preprocessing。 | `survival_generate_results.py::main()` |
| Cox baseline sensitivity | Cox baseline 仍以 median imputation 作為 survival sensitivity 的獨立流程。 | `survival_generate_results.py::multivariable_cox_baseline()` |

## 每個建模欄位遇到缺失值時的處理

| 欄位 | 類型 | 缺失值處理 |
| --- | --- | --- |
| `死亡標記` | Outcome label | 不補值；作為 y 轉為 integer。若 outcome 缺失，實務上不能納入 supervised learning。 |
| `性別_is_male` | 二元 | 補 `0`。 |
| `預估年齡` | 連續 | 以 development/training fold mean 與 SD 標準化；缺失 z-score 補 `0`，等同原始尺度補 mean。 |
| `DNR_flag` | 二元 | 補 `0`。 |
| `ADL_總分_max` | 量表 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `ADL_first_score` | 量表 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `ADL_last_score` | 量表 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `ADL_diff_seq` | 變化量/連續 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `ADL_std` | 連續 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `ADL_Max` | 量表 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `ADL_Min` | 量表 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `ADL_明顯惡化` | 二元 | 補 `0`。 |
| `ADL_first_CouldNot` | 二元 | 補 `0`。 |
| `ADL_last_CouldNot` | 二元 | 補 `0`。 |
| `六個月內住院次數` | 計數 | 補 `0`。 |
| `first_has_denture` | 二元 | 補 `0`。 |
| `last_has_denture` | 二元 | 補 `0`。 |
| `diff_has_denture` | 二元/變化指標 | 補 `0`。 |
| `first_ 意識總分` | 量表 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `last_ 意識總分` | 量表 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `意識總分Max` | 量表 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `意識總分_diff` | 變化量/連續 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `使用呼吸輔具` | 二元 | 補 `0`。 |
| `first_has_feeding_tube` | 二元 | 補 `0`。 |
| `last_has_feeding_tube` | 二元 | 補 `0`。 |
| `diff_has_feeding_tube` | 二元/變化指標 | 補 `0`。 |
| `had_fall` | 二元事件 | 補 `0`。 |
| `BW_first` | 連續 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `BW_last` | 連續 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |
| `BW_diff_seq` | 變化量/連續 | z-score 後缺失補 `0`，等同原始尺度補 mean。 |

## 不進入主要分類模型的欄位

| 欄位/類型 | 處理方式 |
| --- | --- |
| development cohort missingness `>= 30%` 的候選變項 | 不納入主要模型特徵矩陣。 |
| `H01_NUM` | 識別碼，從特徵矩陣移除。 |
| `觀察天數` | 從主要 binary classifier 移除；只在 survival sensitivity analysis 作為 time variable。 |
| `dbname`、`結案日期`、`入家日期` 等非主要模型欄位 | 不進入主要模型；可用於補充 missingness 或機構層級分析。 |

## 可放入 response letter 的英文說法

> We revised the analysis code to align with the missing-data strategy described in the manuscript. Candidate predictors with 30% or greater missingness in the development cohort were excluded. For retained binary or count variables typically documented when present, missing values were imputed as 0. Continuous measures and scale scores were standardized using means and standard deviations estimated from the development cohort only; missing standardized values were then imputed as 0, corresponding to mean imputation on the original scale. The same development-cohort preprocessing parameters were applied unchanged to the temporal external validation cohort. In internal cross-validation, preprocessing parameters were estimated within each training fold and applied to the corresponding validation fold.
