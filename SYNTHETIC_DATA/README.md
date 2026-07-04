# Synthetic Demonstration Data

This folder contains privacy-preserving demonstration data for code walkthroughs.

The files in this folder are fully simulated. They do not contain real residents, real facilities, or real source records. Their only purpose is to show how local CSV inputs should be shaped when running the analysis scripts.

## Generate the Demo Files

Run from the repository root:

```powershell
python SYNTHETIC_DATA/generate_synthetic_data.py
```

This creates:

```text
SYNTHETIC_DATA/training_data_1014_synthetic.csv
SYNTHETIC_DATA/external_validation_1014_synthetic.csv
```

## Run a Fast Code Demonstration

The full manuscript pipeline is designed for real analyses and can be computationally expensive. For a local smoke test with synthetic data, use small bootstrap counts and skip SHAP:

```powershell
python revision_generate_results.py --training SYNTHETIC_DATA/training_data_1014_synthetic.csv --external SYNTHETIC_DATA/external_validation_1014_synthetic.csv --results-dir SYNTHETIC_DATA/demo_results --n-boot 20 --n-boot-fast 20 --skip-cv --skip-shap
```

The resulting performance estimates are meaningless because the data are simulated. Use them only to confirm that the code can read local files and write output tables/figures.

## Privacy Notice

Do not place real resident-level data in this folder. Real data exports should remain outside version control and should be handled under the applicable data-use and privacy rules.
