# 0609 Development Facility Missingness Form Notes

## What Was Requested

The PDF requested a Development cohort table stratified by facility size and institutional region, including N facilities, N residents, all-feature missing percent overall, all-feature missing percent among dead residents, all-feature missing percent among alive residents, and death rate. It also requested a chi-square test for institution ID by overall all-feature missing percent.

## Files Produced

- `0609_development_facility_missingness_form.xlsx`
- `0609_development_facility_missingness_form.csv`
- `0609_exploratory_mapped_facility_missingness_form.csv`
- `0609_exploratory_mapping_audit.csv`
- `0609_h01num_missingness_chi_square.csv`
- `0609_missingness_indicator_key_features_regression_with_p.xlsx`
- `0609_missingness_indicator_key_features_regression_with_p.csv`

## All Features

The all-feature missingness calculation uses the 29 model predictor features listed in `RESULTS/tables/shap_feature_importance.xlsx`, matching the feature list shown in the 0609 PDF. `死亡標記` from `selected_features.xlsx` is the outcome and is not counted as a predictor feature.

## Data Limitation

The current project files do not contain a usable resident-level `dbname` / facility linkage for the Development cohort model cache (`Revision/20260523/training_data_1014_cached_for_completion.csv`). In that file, `dbname` is empty for all 23,901 rows. Therefore, resident counts, dead/alive all-feature missingness, and death rates by facility size or institutional region cannot be estimated reliably from the saved Development cohort cache.

The `N facilities` column in the requested form was filled from `DATA/area_size.xlsx`, sheet `訓練資料_機構大小`. Other resident-level columns are marked as not estimable in the workbook.

## Exploratory Mapping Attempt

An additional sheet, `Exploratory mapped form`, uses a modal H01_NUM-to-dbname map derived from local excluded/supplemental files and then merges to `DATA/area_size.xlsx`. This provides a complete numeric table, but it is exploratory. The `Exploratory mapping audit` sheet reports coverage and ambiguity. In the current data, many H01_NUM values have many candidate dbname values, so this should not replace confirmed resident-level facility linkage.

## Chi-square Test

Because `dbname` is absent in the Development cohort cache, the chi-square sheet uses `H01_NUM` as the only available repeated identifier in the analytic cache. This should be treated as exploratory unless `H01_NUM` is confirmed to be the intended institution ID.

Number of prediction features counted: 29
Development cohort rows: 23901
Development cohort deaths: 5272
