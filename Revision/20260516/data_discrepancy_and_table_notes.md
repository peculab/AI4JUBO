# Data discrepancy and table notes

## Main reason for differences

1. `included_vs_excluded_insufficient_followup` uses all included analytic residents, development plus external validation, N = 30,117. The submitted manuscript Table 1 uses only the development cohort, N = 23,901. This explains why baseline values such as age, ADL, tube feeding, respiratory support, falls, body weight, and hospitalizations do not exactly match the submitted Table 1.

2. The generated revision table reports percentages among nonmissing values for binary/categorical fields. For example, initial feeding tube is 2,689 / 27,330 = 9.8% because 2,787 included residents are missing that field. The submitted manuscript table generally used the cohort total as the denominator, for example 2,081 / 23,901 = 8.7%.

3. Facility size variables are available for excluded residents through `DATA/area_size.xlsx`, but the Google Sheets analytic-cohort import does not retain a stable facility linkage for included residents. Therefore included facility-size cells remain `NA` in the current generated comparison table.

4. Some original manuscript variables, including CIRS-G and parts of the original excluded-without-ADL appendix, are not available in the current insufficient-follow-up excluded-resident file. These are kept as `NA` in the manuscript-style reconstruction instead of being imputed.

## Updated outputs

- `RESULTS/tables/included_vs_excluded_insufficient_followup.csv` and `.xlsx`: original generated table with `P value` added.
- `RESULTS/tables/included_vs_excluded_insufficient_followup_with_p.csv` and `.xlsx`: duplicate p-value-labeled copy.
- `RESULTS/tables/development_cohort_plus_current_excluded_insufficient_followup.csv` and `.xlsx`: manuscript-style table preserving the submitted Development Cohort appendix columns and adding what can currently be produced for excluded residents with insufficient follow-up/outcome ascertainment.
- `RESULTS/tables/facility_region_size_overview_20260516.csv` and `.xlsx`: overview of the newly provided facility-region and facility-size workbook.
- `RESULTS/tables/excluded_residents_by_facility_size_20260516.csv` and `.xlsx`: resident-level excluded cohort linked to facility size by `dbname`.
- `RESULTS/tables/excluded_residents_by_region_20260516.csv` and `.xlsx`: excluded-resident region distribution supplied in the workbook.
- `RESULTS/figures/training_facility_region_size_20260516.png` and `RESULTS/figures/excluded_facility_region_size_20260516.png`: bar charts for facility-region/size characterization.

## Facility Region And Size Update

The newly added `Revision/20260516/機構區域與大小.xlsx` improves the facility-size and region response, especially for excluded residents:

- Training/development facility roster: 493 facilities; small 311, medium 148, large 34; south 332, central 100, north 54, east 7.
- Excluded residents by region: north 2,112, central 4,019, south 13,447, east 174, islands 4.
- Excluded residents by facility size after `dbname` linkage: small 9,218, medium 8,211, large 2,233, unknown 94.

This workbook is still not a resident-level included analytic cohort file. It supports facility-level characterization of the development/training facility roster and resident-level excluded-resident stratification, but it does not fully support resident-level included versus excluded facility-size/region testing.

## Statistical tests

Continuous variables use Welch two-sample t tests from available means, SDs, and nonmissing denominators. Binary/categorical rows use chi-square tests on nonmissing 2 x 2 tables. Rows with unavailable data in either group are marked `NA`.
