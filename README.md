# AI4JUBO: JUBO 6-Month Mortality Prediction Analysis

This repository contains the analysis materials for the manuscript:

**Development and Temporal External Validation of a Parsimonious, Interpretable Machine Learning Model for Predicting 6-Month Mortality in Long-Term Care Facilities**

The repository has been cleaned to keep only the submitted manuscript analysis, revision scripts, generated revision outputs, and submission/revision documents.

## Repository Contents

```text
AI4JUBO/
|-- FINAL/
|   `-- Submitted manuscript, appendices, and original submitted figures
|-- Revision/
|   `-- Reviewer comments, response mapping files, and revision planning notes
|-- RESULTS/
|   |-- tables/    Generated CSV/XLSX revision tables
|   |-- figures/   Generated HTML/PNG revision figures
|   |-- package_versions.txt
|   |-- model_identity_note.txt
|   `-- survival_analysis_note.txt
|-- jubodeath_v9_puredata_paper.ipynb
|-- revision_generate_results.py
|-- survival_generate_results.py
|-- README.md
`-- LICENSE
```

## Main Analysis Code

- `jubodeath_v9_puredata_paper.ipynb`  
  Final notebook corresponding to the submitted manuscript analyses.

- `revision_generate_results.py`  
  Generates revision tables and figures, including:
  - internal cross-validation performance with 95% CI
  - temporal external validation performance with 95% CI
  - subgroup performance with 95% CI
  - paired bootstrap AUROC comparison
  - threshold-specific PPV/NPV/sensitivity/specificity tradeoffs
  - decision-curve analysis
  - calibration intercept, slope, O/E ratio, risk-decile calibration
  - missingness indicator summaries
  - facility-level missingness summaries
  - model-agnostic KernelSHAP feature importance

- `survival_generate_results.py`  
  Generates survival sensitivity analyses using available observation time and death status:
  - Harrell C-index
  - Cox model using predicted risk score
  - cumulative/dynamic AUC at monthly horizons
  - Kaplan-Meier curves by predicted-risk tertile
  - log-rank test across risk groups

## Data Source

The individual-level data are not stored in this repository because of privacy and data-use restrictions.

The analysis scripts read the same Google Sheets data sources referenced in the final notebook:

- Development cohort: `training_data_1014`
- Temporal external validation cohort: `external_validation_1014`

The scripts also support local CSV/XLSX files if the data are exported locally.

## Reproducing Revision Outputs

Run from `C:\AI4JUBO`.

Generate main revision results:

```powershell
python revision_generate_results.py --use-google-sheets
```

Generate survival sensitivity results:

```powershell
python survival_generate_results.py --use-google-sheets
```

If local files are available:

```powershell
python revision_generate_results.py --training training_data_1014.xlsx --external external_validation_1014.xlsx
```

Outputs are written to:

```text
RESULTS/tables/
RESULTS/figures/
```

## Random Seeds

The revision scripts use fixed seeds for reproducibility:

- Default random seed: `42`
- Stratified 5-fold cross-validation: `random_state=42`
- Bootstrap confidence intervals: `random_state=42`
- XGBoost models: `random_state=42`
- Random Forest models: `random_state=42`
- SHAP sampling: `random_state=42`

Bootstrap settings:

- Main external-validation bootstrap: `n_boot=2000`
- Internal/subgroup calibration bootstrap: `n_boot_fast=500`

## Preprocessing Summary

Feature selection and preprocessing follow the submitted analysis workflow:

1. Candidate variables with missingness below the prespecified cutoff in the development cohort are retained for modeling.
2. Identifier and follow-up-time columns (`H01_NUM`, `觀察天數`) are excluded from the classification feature matrix.
3. Binary/count variables routinely documented when present are coded as `0` when missing, reflecting absence or non-documentation in the routine workflow.
4. Continuous or scale variables with missingness are imputed at the development-cohort mean. When standardized inputs are used, this is equivalent to imputing missing z-scores as `0`.
5. All preprocessing parameters are estimated from the development cohort and applied unchanged to the temporal external validation cohort.
6. The primary outcome is binary death within 180 days after admission.

## Model and Evaluation Summary

The scripts compare:

- `HybridXGBRF (Our Approach)` label used in the final notebook
- XGBoost classifier
- Random Forest classifier
- Logistic regression baselines
- Ridge, Lasso, and Elastic Net logistic regression baselines

Important note:

The final notebook labels the leading model as `HybridXGBRF (Our Approach)`, but the final `all_models` dictionary appears to assign this label to an `XGBClassifier` object. For revision transparency, see:

```text
RESULTS/model_identity_note.txt
```

For reviewer response, claims of superiority over XGBoost should be phrased cautiously because paired bootstrap comparisons show very small AUROC differences.

## Package Versions

The package versions used in the generated revision outputs are recorded in:

```text
RESULTS/package_versions.txt
```

Current recorded versions include:

```text
Python: 3.10.11
numpy: 1.26.4
pandas: 2.3.3
scikit-learn: 1.7.2
xgboost: 3.1.1
plotly: 6.4.0
shap: 0.49.1
openpyxl: 3.1.5
kaleido: 1.1.0
```

## Survival Sensitivity Analysis Note

Survival analyses are sensitivity analyses, not replacements for the primary binary 180-day mortality model.

They use:

- `觀察天數` as available observation/follow-up time
- `死亡標記` as death indicator
- 180-day administrative horizon
- Censoring at `min(觀察天數, 180)` for residents without death before 180 days

See:

```text
RESULTS/survival_analysis_note.txt
```

## Data and Code Availability Statement

The individual-level dataset cannot be publicly shared because of privacy and data-use restrictions. The analysis scripts, preprocessing details, random seeds, package versions, and generated aggregate outputs are provided to support reproducibility within the limits of data governance.
