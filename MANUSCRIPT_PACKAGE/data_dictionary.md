# Data Dictionary and Privacy-Preserving Schema

The individual-level source data have been removed from this repository and are not distributed with the code. This document describes the analysis schema and variable attributes at a reproducibility level without exposing real resident records.

## Privacy Scope

- The real development and temporal external validation datasets contain resident-level long-term care facility records.
- Real individual-level rows, facility rosters, and raw supplemental spreadsheets are not included in the repository.
- The repository provides code, aggregate outputs, documentation, and synthetic demonstration data only.
- Synthetic files in `../SYNTHETIC_DATA/` are simulated and must not be interpreted as clinical or operational results.

## Dataset Roles

| Dataset role | Description | Availability in repository |
| --- | --- | --- |
| Development cohort | Resident-level records used for feature selection, preprocessing, model development, and internal 5-fold cross-validation | Removed |
| Temporal external validation cohort | Later-period resident-level records used for external model validation | Removed |
| Supplemental facility files | Facility size, region, and excluded-resident support files used for reviewer-response summaries | Removed |
| Synthetic demonstration data | Simulated rows with the same analysis roles and compatible column structure | Included |

## Required Analysis Columns

| Column role | Type | Used for | Synthetic example |
| --- | --- | --- | --- |
| Resident identifier | String or numeric identifier | Row tracing and exclusion diagnostics; excluded from model predictors | `SYN000001` |
| Observation days | Numeric days | Survival sensitivity analysis and exclusion diagnostics; excluded from model predictors | `180` |
| Death indicator | Binary 0/1 | Primary 180-day mortality outcome and survival event indicator | `0` or `1` |
| Facility identifier | String identifier | Facility-level missingness summaries when available; not a model predictor | `FAC001` |
| Facility size stratum | Categorical | Facility-level descriptive summaries | `Small`, `Medium`, `Large` |
| Facility region | Categorical | Facility-level descriptive summaries | `North`, `Central`, `South`, `East` |

## Model Predictor Domains

The manuscript model uses a parsimonious set of structured predictors. Column names in the original data may contain non-English source labels; these labels remain in the code only where required to match the original analysis workflow.

| Predictor domain | Data type | Example manuscript label | Preprocessing attribute |
| --- | --- | --- | --- |
| Demographics | Binary or continuous | Male, estimated age | Binary variables are 0/1; continuous variables are standardized from development-cohort parameters |
| Activities of daily living | Continuous or binary | ADL last score, ADL change, ADL maximum, ADL deterioration flag | Continuous ADL features use mean imputation after standardization; binary indicators use zero imputation |
| Body weight trajectory | Continuous | Body weight first value, last value, and change | Continuous; development-cohort mean and standard deviation applied to validation data |
| Consciousness score | Continuous | Consciousness score first, last, maximum, and change | Continuous; same preprocessing as other scale variables |
| Care devices and functional supports | Binary | Feeding tube, denture, respiratory aid | Binary/count features use zero imputation when missing |
| Prior events and directives | Binary or count | Hospitalizations within 6 months, fall history, DNR flag | Binary/count features use zero imputation when missing |

## Manuscript Predictor Crosswalk

The following table describes the model-facing predictor attributes using English labels. It does not include real resident values.

| English predictor label | Domain | Attribute type | Missing-data handling |
| --- | --- | --- | --- |
| Hospitalizations within 6 Months | Prior utilization | Count | Zero imputation |
| ADL Last Score | Activities of daily living | Continuous score | Development-cohort mean after standardization |
| Body Weight Change | Body weight trajectory | Continuous change | Development-cohort mean after standardization |
| ADL Standard Deviation | Activities of daily living | Continuous variability metric | Development-cohort mean after standardization |
| ADL Minimum | Activities of daily living | Continuous score | Development-cohort mean after standardization |
| Male | Demographics | Binary indicator | Zero imputation |
| Body Weight (Last) | Body weight trajectory | Continuous measurement | Development-cohort mean after standardization |
| ADL Total Max | Activities of daily living | Continuous score | Development-cohort mean after standardization |
| Body Weight (First) | Body weight trajectory | Continuous measurement | Development-cohort mean after standardization |
| DNR Flag | Care directive | Binary indicator | Zero imputation |
| ADL First Score | Activities of daily living | Continuous score | Development-cohort mean after standardization |
| Consciousness Score Difference | Consciousness status | Continuous change | Development-cohort mean after standardization |
| Estimated Age | Demographics | Continuous | Development-cohort mean after standardization |
| Consciousness Score (Last) | Consciousness status | Continuous score | Development-cohort mean after standardization |
| Use of Respiratory Aid | Care support | Binary indicator | Zero imputation |
| ADL Change | Activities of daily living | Continuous change | Development-cohort mean after standardization |
| Feeding Tube Change | Care support | Binary/change indicator | Zero imputation |
| ADL Last Could Not Perform | Activities of daily living | Binary indicator | Zero imputation |
| ADL Significant Deterioration | Activities of daily living | Binary indicator | Zero imputation |
| ADL Maximum | Activities of daily living | Continuous score | Development-cohort mean after standardization |
| ADL First Could Not Perform | Activities of daily living | Binary indicator | Zero imputation |
| Denture Change | Care support | Binary/change indicator | Zero imputation |
| Has Denture (Last) | Care support | Binary indicator | Zero imputation |
| Had Fall | Prior event | Binary indicator | Zero imputation |
| Has Feeding Tube (Last) | Care support | Binary indicator | Zero imputation |
| Has Denture (First) | Care support | Binary indicator | Zero imputation |
| Consciousness Score Max | Consciousness status | Continuous score | Development-cohort mean after standardization |
| Has Feeding Tube (First) | Care support | Binary indicator | Zero imputation |
| Consciousness Score (First) | Consciousness status | Continuous score | Development-cohort mean after standardization |

## Missingness Handling

The reproducibility scripts apply the manuscript preprocessing rules:

1. Candidate variables with missingness below the prespecified cutoff in the development cohort are retained.
2. Resident identifiers and follow-up-time columns are excluded from the classification feature matrix.
3. Binary and count variables that are routinely documented are imputed as `0` when missing.
4. Continuous or scale variables are standardized using development-cohort parameters, and missing standardized values are imputed as `0`.
5. Preprocessing parameters are learned from the development cohort and applied unchanged to the temporal external validation cohort.

## Synthetic Demonstration Data

The synthetic data generator creates compatible but artificial files:

- `../SYNTHETIC_DATA/training_data_1014_synthetic.csv`
- `../SYNTHETIC_DATA/external_validation_1014_synthetic.csv`

The generated values are random simulations designed to exercise the code paths. They do not preserve row-level distributions from the real data and should not be used to report model performance.
