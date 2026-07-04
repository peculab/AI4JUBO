# Manuscript Reproducibility Package

This folder is the recommended entry point for manuscript review, revision, and reproducibility checks.

The repository contains several dated revision folders because the analysis was updated step by step during reviewer response preparation. For manuscript-facing use, treat the files listed here as the consolidated package. Older dated folders are retained only for provenance and audit trails.

## Recommended Files to Cite

- `../revision_generate_results.py`: canonical script for the primary binary 180-day mortality model, internal cross-validation, temporal external validation, calibration, subgroup, threshold, missingness, facility, SHAP, and decision-curve outputs.
- `../survival_generate_results.py`: canonical script for survival sensitivity analyses.
- `../RESULTS/`: canonical output directory for regenerated manuscript tables and figures.
- `../RESULTS/tables/`: manuscript-facing CSV/XLSX tables, including the official 20260609 addendum tables.
- `../RESULTS/figures/`: manuscript-facing PNG/JPG/HTML figures, including the official 20260609 addendum figures.

## How to Regenerate Outputs

The real resident-level data have been removed from this repository. Authorized users can regenerate manuscript outputs from local restricted data exports kept outside version control. The public code does not include embedded Google Sheet URLs or real source-data links. Public readers can use synthetic files only to inspect the code workflow.

Run from the repository root:

```powershell
python revision_generate_results.py --use-synthetic-data
python survival_generate_results.py --use-synthetic-data
```

If local exported files are available, use:

```powershell
python revision_generate_results.py --training training_data_1014.xlsx --external external_validation_1014.xlsx
python survival_generate_results.py --training training_data_1014.xlsx --external external_validation_1014.xlsx
```

Primary outputs are written to:

```text
RESULTS/tables/
RESULTS/figures/
```

## Synthetic Demonstration

Generate privacy-preserving fake input files:

```powershell
python SYNTHETIC_DATA/generate_synthetic_data.py
```

Run a fast smoke test:

```powershell
python revision_generate_results.py --training SYNTHETIC_DATA/training_data_1014_synthetic.csv --external SYNTHETIC_DATA/external_validation_1014_synthetic.csv --results-dir SYNTHETIC_DATA/demo_results --n-boot 20 --n-boot-fast 20 --skip-cv --skip-shap
```

The synthetic files contain no real resident or facility records. Their output estimates should not be cited as manuscript results.

## What to Use in the Manuscript

- Use `RESULTS/tables/table3_internal_cv_performance_with_ci.*` for internal cross-validation performance.
- Use `RESULTS/tables/table4_external_validation_full_with_ci.*` and `table4_external_validation_paper_friendly.*` for temporal external validation.
- Use `RESULTS/tables/table5_subgroup_performance_with_ci.*` for subgroup analyses.
- Use `RESULTS/tables/calibration_metrics_*` and `risk_decile_calibration_*` for calibration reporting.
- Use `RESULTS/tables/paired_bootstrap_auroc_*` for paired AUROC comparisons.
- Use `RESULTS/tables/survival_*` and `RESULTS/figures/survival_*` only as sensitivity analyses.
- Use `RESULTS/tables/0609_development_facility_missingness_form.*` for the latest facility-level missingness form.
- Use `RESULTS/figures/0609_shap_forest_plot_top8.*` and `RESULTS/figures/0609_shap_beeswarm_top8.*` for the latest revised SHAP displays.

## Navigation

- `code_catalog.md`: short English description of each script and which one should be considered canonical.
- `data_dictionary.md`: privacy-preserving description of real data fields, variable attributes, and synthetic demo files.
- `results_catalog.md`: direct links to the manuscript-facing tables and figures.
- `legacy_revision_map.md`: explains why dated revision folders remain in the repository and how they should be interpreted.

## Important Reproducibility Notes

- Individual-level data have been removed from this repository because of privacy and data-use restrictions.
- The repository includes synthetic demonstration data only; synthetic outputs are not manuscript results.
- Random seeds are fixed at `42` unless explicitly overridden.
- Main external-validation bootstrap intervals use `--n-boot 2000`.
- Calibration and subgroup bootstrap intervals use `--n-boot-fast 500` unless the script is rerun with a different value.
- The final manuscript should avoid overclaiming superiority over XGBoost because paired bootstrap AUROC differences are small.
