# FINAL 投稿版數據與程式碼對照審查

日期：2026-05-08  
工作區：`..`

## 結論

目前與 `FINAL` 投稿版主要數據最一致的程式來源是：

`../jubodeath_v9_puredata_paper.ipynb`

其他 `JuboDeath_V2` 到 `JuboDeath_V9_*` 多數是歷史版本或局部分析；只有 `jubodeath_v9_puredata_paper.ipynb` 同時包含投稿版的 cohort size、主要模型表、外部驗證 bootstrap CI、次族群分析、ROC/calibration/SHAP 圖表流程。

需特別注意：在 `jubodeath_v9_puredata_paper.ipynb` cell 20 雖然定義了真正的 `HybridXGBRF` blending class，但 cell 29 的 `all_models` 內，標記為 `"HybridXGBRF (Our Approach)"` 的實際物件是 `XGBClassifier(...)`，不是 cell 20 的 blending class。也就是說，投稿版表格中的「HybridXGBRF」數值與 notebook 結果一致，但程式語義上需要重新確認：目前 SHAP 與 performance 很可能是在解釋/評估 XGBoost component 或 XGB-style model，而不是 blended HybridXGBRF final probability。

這點正好對應 Reviewer BK 的 SHAP 質疑，修稿時不能再直接寫「SHAP explains the blended HybridXGBRF output」，除非重新用 final blended probability 重跑 model-agnostic SHAP。

## 與投稿文章一致的 FINAL 數據

投稿主文 `FINAL\Machine_Learning_Model_for_Predicting_6-Month_Mortality_in_LTC_Facilities.docx`：

| 項目 | 投稿版數據 | 對應程式/儲存格 |
|---|---:|---|
| Development cohort | n = 23,901 | `jubodeath_v9_puredata_paper.ipynb` cell 13 |
| Development deaths | 5,272 / 23,901 = 22.1% | cell 13/14 與主文 Table 1 |
| External validation cohort | n = 6,216 | cell 9 |
| External deaths | 1,781 / 6,216 = 28.7% | cell 9 與主文 Table 2 |
| Internal CV Hybrid AUROC | 0.875, 95% CI 0.862-0.889 | cell 49 |
| Internal CV Hybrid Brier | 0.108553，投稿四捨五入 0.109/0.11 | cell 49 |
| External Hybrid accuracy | 0.851464，投稿四捨五入 0.851/0.85 | cell 68/70 |
| External Hybrid F1 | 0.572289，投稿四捨五入 0.572/0.57 | cell 68/70 |
| External Hybrid AUROC | 0.878147，95% CI 0.866439-0.889445 | cell 68/70 |
| Subgroup ADL improvement AUROC | 0.897020，投稿四捨五入 0.90 | cell 60 |
| Subgroup sex/age AUROC | female 0.890752, male 0.879600, age <=85 0.883859, age >85 0.892212 | cell 60 |

## 可沿用的程式區塊

| 用途 | 可沿用儲存格 | 狀態 |
|---|---|---|
| Cohort descriptive statistics | cells 9, 13, 14 | 與投稿主文 Table 1/2 一致 |
| Missingness table | cells 10, 15 | 與 Multimedia Appendix 4 大致一致 |
| Model candidate setup | cells 20, 29 | 數值可追溯，但 `HybridXGBRF` 名稱與實際模型需確認 |
| 5-fold CV model performance | cells 30, 49 | 與投稿 Table 3 一致 |
| Confusion matrix figure | cell 31 | 對應 Figure 2 |
| Internal calibration plot | cells 32, 33, 67 | 對應 Figure 3 |
| Decision curve analysis, internal OOF | cells 34, 35 | 已有初稿，但投稿版尚未納入，且需補 external DCA |
| SHAP ranking / forest / beeswarm / force plot | cells 36-48 | 對應 Figure 5-7，但需釐清是否 final hybrid output |
| External validation model comparison | cell 66 | 對應 Figure 1/4 與外部驗證 |
| External validation 95% CI | cells 68, 70, 71 | 對應 Reviewer 要求的一部分，可補進 Table 4 |
| Subgroup performance | cells 59, 60 | 對應 Table 5，但目前只有 point estimate，缺 95% CI |

## 不建議作為修稿依據的程式

`run_v13_3.py` 是 LangChain / LLM 測試程式，與本篇死亡預測模型無關。

`JuboDeath_V2.ipynb` 到 `JuboDeath_V8_pureData.ipynb` 以及 `JuboDeath_V9_pureData*.ipynb` 仍可作為歷史參考，但目前沒有同時命中 external AUROC 0.878147、accuracy 0.851464、F1 0.572289、完整 external_summary_df 等投稿最終結果，因此不要優先從這些版本修改。

## Revision 需要補的數據與圖表

### 1. Performance 95% CI

Reviewer BK 要求 recall、precision、F1、Brier score、calibration、subgroup performance 都要 uncertainty estimates。

目前已有：

- External validation accuracy/precision/recall/F1/AUROC bootstrap 95% CI：cell 68/70。

仍需補：

- Internal CV Table 3 的 precision、recall、F1、Brier score 95% CI。
- External Table 4 的 Brier score 95% CI。
- Subgroup Table 5 的 accuracy、precision、recall、F1、AUROC 95% CI。

建議新增表格：

- Revised Table 3. Internal cross-validation performance with 95% CI.
- Revised Table 4. Temporal external validation performance with 95% CI.
- Revised Table 5. Subgroup performance of the selected model with 95% CI.

### 2. HybridXGBRF vs XGBoost paired comparison

Reviewer BK 指出 HybridXGBRF 對 XGBoost 的提升太小。

目前缺：

- paired bootstrap AUROC difference。
- DeLong test 或 paired permutation test。

建議補：

- Internal OOF：用 `oof_probs_all["HybridXGBRF (Our Approach)"]` vs `oof_probs_all["XGBClassifier"]`。
- External：用 final model external probabilities。
- 回覆時若差異不顯著，摘要與結論改成 "tree-based models showed comparable discrimination"，避免宣稱 Hybrid 明顯優於 XGBoost。

### 3. Threshold-specific clinical utility

Reviewer BK 要求 alternative thresholds、PPV/NPV、threshold-specific tradeoffs、decision curve analysis。

目前已有：

- Internal OOF decision curve：cells 34/35。

仍需補：

- External validation DCA。
- Threshold table：例如 threshold = 0.10, 0.20, 0.30, 0.40, 0.50。
- 每個 threshold 報 TP、FP、FN、TN、sensitivity、specificity、PPV、NPV、F1、net benefit。

建議新增圖表：

- Figure Sx. Decision curve analysis in the temporal external validation cohort.
- Table Sx. Threshold-specific tradeoffs for HybridXGBRF in the external validation cohort.

### 4. Calibration numeric metrics

Reviewer BK 要求 calibration slope、intercept、observed/expected ratio、risk-decile calibration。

目前已有：

- Calibration plots：Figure 3/4，cells 32/33/67。
- Brier score：Table 3，external notebook可計算。

仍需補：

- Calibration intercept。
- Calibration slope。
- Observed/Expected ratio。
- Risk decile table：每個 decile 的 n、mean predicted risk、observed risk、death n。

建議新增：

- Table Sx. Calibration metrics in development and external validation cohorts.
- Table Sx. Risk-decile calibration table for the external validation cohort.
- 若 Reviewer W 要求 histogram，可把 calibration plot 改為下方加 predicted probability distribution rug/histogram。

### 5. Included vs excluded comparison

Reviewer BK 與 Reviewer W 都要求處理 selection bias。

目前已有：

- FINAL Appendix 7：development cohort vs excluded subjects without ADL data。
- Revision 內已有 `01_需蔡老師補做資料表\(Revision)Development_Cohort.docx 的副本.docx`，但「追蹤未達6個月的排除個案」仍是 placeholder，n = ???。

仍需補：

- Excluded due to insufficient follow-up 的 n 與 baseline characteristics。
- 最好分成 included、missing ADL excluded、insufficient follow-up excluded、implausible/duplicate excluded。
- 加上 SMD，而不只 P value。

建議新增表格：

- Multimedia Appendix X. Characteristics of included and excluded residents.

### 6. Missingness / imputation sensitivity

Reviewer BK 質疑 binary/count missing coded as absence。

目前已有：

- Missingness summary：FINAL Appendix 4，notebook cells 10/15。

仍需補：

- Missingness indicator analysis：重要變項各加 missing flag，看 missingness 與 mortality 關係。
- Alternative imputation sensitivity：binary missing = 0 vs mode/missing category；continuous z-score mean imputation。
- Facility-level missingness：若有 facility ID，需報各機構 missing rate distribution，或至少 summary。

建議新增：

- Table Sx. Association between missingness indicators and 6-month mortality.
- Table Sx. Model performance under alternative imputation strategies.
- Figure Sx. Facility-level missingness distribution/heatmap。

### 7. Survival / time-to-event

Reviewer BK 要求 Cox、random survival forest、gradient boosting survival 或 sensitivity。

目前 notebook 有 `from lifelines import CoxPHFitter`，但沒有看到完成的 survival model 結果。

需確認資料是否真的有：

- admission date。
- death date。
- censoring/discharge date。
- follow-up time。

若有，建議補：

- Cox proportional hazards baseline。
- Time-dependent AUC at 180 days，或 C-index。

若沒有完整 censoring/time-to-death：

- 回覆需明確說明本研究是 pragmatic binary 180-day screening model。
- 補 included vs excluded baseline comparison 作為 sensitivity/selection-bias assessment。

### 8. SHAP 必須修正說法

目前程式顯示：

- cell 20 定義 HybridXGBRF blending。
- cell 29 實際 `all_models["HybridXGBRF (Our Approach)"]` 是 `XGBClassifier(...)`。
- cell 36 SHAP 使用 `xgb_model = all_models["HybridXGBRF (Our Approach)"]`。

因此目前 SHAP 圖很可能不是 final blended Hybrid probability 的 direct explanation。

修稿有兩個選項：

1. 若維持目前 SHAP 圖：文字改成 component-level / selected tree-based model explanation，不能說是 blended final output 的 SHAP。
2. 若要完整回應 Reviewer：用 model-agnostic SHAP / KernelSHAP / permutation SHAP 對 `HybridXGBRF.predict_proba()` 的 final blended probability 重跑 Figure 5-7。

## 建議優先順序

1. 先確認 `HybridXGBRF` 最終模型到底是 blended model 還是 XGBClassifier 命名。
2. 補 paired Hybrid vs XGB statistical comparison，並同步降調摘要與結論。
3. 補 external validation 的 threshold table、PPV/NPV、DCA。
4. 補 calibration slope/intercept/OE/risk-decile table。
5. 補 subgroup 95% CI。
6. 補 included vs excluded，尤其 insufficient follow-up 那組。
7. 補 missingness indicator / imputation sensitivity / facility-level missingness。
8. 視資料可得性決定 survival analysis；若不能做，要在 response letter 和 limitations 說清楚。
