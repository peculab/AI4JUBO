# Code Catalog

This catalog identifies the manuscript-facing scripts and separates them from historical notebook/provenance files.

## Canonical Manuscript Scripts

### `../revision_generate_results.py`

Primary reproducibility script for the binary 180-day mortality prediction analysis. Use this script for manuscript tables and figures related to:

- internal 5-fold cross-validation performance;
- temporal external validation performance;
- calibration metrics and risk-decile calibration;
- subgroup analyses;
- threshold tradeoff tables;
- paired bootstrap AUROC comparisons;
- decision-curve analysis;
- missingness and facility-level summaries;
- model-agnostic SHAP feature importance.

Recommended command:

```powershell
python revision_generate_results.py --use-synthetic-data
```

Optional local-file command:

```powershell
python revision_generate_results.py --training training_data_1014.xlsx --external external_validation_1014.xlsx
```

Synthetic smoke-test command:

```powershell
python SYNTHETIC_DATA/generate_synthetic_data.py
python revision_generate_results.py --training SYNTHETIC_DATA/training_data_1014_synthetic.csv --external SYNTHETIC_DATA/external_validation_1014_synthetic.csv --results-dir SYNTHETIC_DATA/demo_results --n-boot 20 --n-boot-fast 20 --skip-cv --skip-shap
```

### `../survival_generate_results.py`

Sensitivity-analysis script for survival-style checks using available observation time and death status. These outputs should be described as sensitivity analyses, not replacements for the primary binary 180-day mortality model.

Recommended command:

```powershell
python survival_generate_results.py --use-synthetic-data
```

## Provenance Files

### `../jubodeath_v9_puredata_paper.ipynb`

Original notebook retained in the local ignored `../ARCHIVE_DO_NOT_CITE/` folder for provenance and auditability. It is not the recommended manuscript-facing reproducibility entry point.

### `../jubodeath_v9_puredata_paper.py`

Python export of the original notebook, retained in the local ignored `../ARCHIVE_DO_NOT_CITE/` folder. It contains notebook-style cells and historical exploratory sections. Do not cite this file as the consolidated analysis pipeline unless a reader specifically needs to audit the original notebook workflow.

## Historical Revision Scripts

Dated scripts inside the local ignored `../ARCHIVE_DO_NOT_CITE/Revision/YYYYMMDD/` folder document reviewer-response iterations. They are retained for traceability but should not be treated as separate manuscript versions. The current manuscript-facing workflow is consolidated through `revision_generate_results.py`, `survival_generate_results.py`, and the files listed in `results_catalog.md`.
