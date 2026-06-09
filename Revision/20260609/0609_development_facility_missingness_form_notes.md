# 0609 Development Facility Missingness Form Notes

## What Was Requested

The PDF requested a Development cohort table stratified by facility size and institutional region, including N facilities, N residents, all-feature missing percent overall, all-feature missing percent among dead residents, all-feature missing percent among alive residents, and death rate. It also requested a chi-square test for institution ID by overall all-feature missing percent.

## Key Correction

The local Development cohort cache (`Revision/20260523/training_data_1014_cached_for_completion.csv`) has `dbname` missing for all 23,901 rows because the original notebook/script applied numeric coercion to the full Google Sheet. The original `training_data_1014` Google Sheet retains confirmed resident-level `dbname`. After numeric cleaning, the raw Google Sheet matches the local cache row-for-row, so the confirmed `dbname` can be safely reattached by row order.

Confirmed Development cohort counts from the raw sheet:

- Development residents: 23,901
- Development deaths: 5,272
- Confirmed unique `dbname`: 493
- Unique `H01_NUM`: 2,057 (`H01_NUM` is not the LTCF count)
- Facility roster rows in `DATA/area_size.xlsx`: 493

## Files Produced

- `0609_development_facility_missingness_form.xlsx`
- `0609_development_facility_missingness_form.csv`
- `0609_dbname_missingness_chi_square.csv`
- `0609_h01num_missingness_chi_square.csv` (diagnostic only)
- `0609_missingness_indicator_key_features_regression_with_p.xlsx`
- `0609_missingness_indicator_key_features_regression_with_p.csv`

## All Features

The all-feature missingness calculation uses the 29 model predictor features listed in `RESULTS/tables/shap_feature_importance.xlsx`, matching the feature list shown in the 0609 PDF. `死亡標記` from `selected_features.xlsx` is the outcome and is not counted as a predictor feature.

## Facility-Level Table

`0609_development_facility_missingness_form.csv` is now the confirmed facility-level table. It uses raw resident-level `dbname` merged to `DATA/area_size.xlsx`, so N residents, dead/alive all-feature missingness, and death rate are estimable by facility size and institutional region.

## Chi-Square Test

`0609_dbname_missingness_chi_square.csv` is the confirmed facility-level chi-square test: `dbname x all-feature missing/observed cells`. The older `H01_NUM` result is retained only as a diagnostic because `H01_NUM` has 2,057 groups and should not be interpreted as the number of LTCFs.

dbname chi-square statistic: 140785.975224
dbname chi-square df: 492
dbname chi-square p value: <0.001
dbname chi-square Cramer's V: 0.450685

Number of prediction features counted: 29
Development cohort rows: 23901
Development cohort deaths: 5272
