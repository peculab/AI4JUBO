# Archive Map

Legacy revision materials have been centralized locally under `../ARCHIVE_DO_NOT_CITE/`. This folder is ignored by Git and is not part of the public manuscript reproducibility package.

## Manuscript-Facing Rule

Use `../RESULTS/` as the canonical regenerated output directory for manuscript-facing tables and figures. The 20260609 facility-missingness and SHAP addendum files that remain manuscript-facing have been copied into:

- `../RESULTS/tables/`
- `../RESULTS/figures/`

Do not cite files directly from `../ARCHIVE_DO_NOT_CITE/`, and do not commit this folder to the public repository.

## Current Public Structure

- `../README.md`: repository-level entry point.
- `../MANUSCRIPT_PACKAGE/`: reproducibility package documentation and catalogs.
- `../RESULTS/`: canonical manuscript-facing aggregate outputs.
- `../SYNTHETIC_DATA/`: privacy-preserving fake data and demo instructions.
- `../DATA/README.md`: data-availability notice; real source data removed.
- `../revision_generate_results.py`: canonical binary-outcome analysis script.
- `../survival_generate_results.py`: canonical survival sensitivity script.

## Archived Contents

- Local `../ARCHIVE_DO_NOT_CITE/FINAL/`: submitted manuscript figures and submission records from the earlier package.
- Local `../ARCHIVE_DO_NOT_CITE/RESULTS_20260523_completion/`: earlier revision-output snapshot.
- Local `../ARCHIVE_DO_NOT_CITE/Revision/`: dated reviewer-response working folders and intermediate outputs.
- Local `../ARCHIVE_DO_NOT_CITE/jubodeath_v9_puredata_paper.ipynb`: original notebook retained for provenance.
- Local `../ARCHIVE_DO_NOT_CITE/jubodeath_v9_puredata_paper.py`: Python export of the original notebook retained for provenance.

## How to Avoid Version Confusion

When preparing manuscript text, tables, captions, response letters, or code-availability statements:

1. Start from `MANUSCRIPT_PACKAGE/README.md`.
2. Use direct file links in `results_catalog.md`.
3. Use `code_catalog.md` to identify the canonical scripts.
4. Treat `ARCHIVE_DO_NOT_CITE/` as an internal audit trail only.
