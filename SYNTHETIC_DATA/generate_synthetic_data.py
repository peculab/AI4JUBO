"""
Create synthetic CSV inputs that match the manuscript analysis schema.

The generated data are random and privacy-preserving. They are intended only
for code demonstrations and smoke tests; they are not clinically meaningful.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
ANALYSIS_SCRIPT = ROOT / "revision_generate_results.py"


def _load_analysis_schema() -> dict[str, object]:
    tree = ast.parse(ANALYSIS_SCRIPT.read_text(encoding="utf-8"))
    wanted = {"OUTCOME_COL", "DROP_COLUMNS", "FEATURE_NAME_MAP", "ZERO_IMPUTE_FEATURES"}
    schema: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    schema[target.id] = ast.literal_eval(node.value)
    missing = wanted - set(schema)
    if missing:
        raise RuntimeError(f"Could not read required schema constants from {ANALYSIS_SCRIPT}: {sorted(missing)}")
    return schema


def _make_frame(n_rows: int, seed: int, cohort: str) -> pd.DataFrame:
    schema = _load_analysis_schema()
    outcome_col = str(schema["OUTCOME_COL"])
    drop_columns = list(schema["DROP_COLUMNS"])
    feature_name_map = dict(schema["FEATURE_NAME_MAP"])
    zero_impute_features = set(schema["ZERO_IMPUTE_FEATURES"])

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=np.arange(n_rows))

    df["H01_NUM"] = np.arange(1, n_rows + 1)
    if drop_columns:
        time_col = drop_columns[1]
    else:
        time_col = "observation_days"
    df[time_col] = rng.integers(30, 241, size=n_rows)

    latent_risk = rng.normal(0, 1, size=n_rows)

    for col in feature_name_map:
        label = feature_name_map[col].lower()
        if col in zero_impute_features or any(token in label for token in ["male", "flag", "tube", "denture", "fall", "aid"]):
            value = rng.binomial(1, 0.25, size=n_rows).astype(float)
        elif "age" in label:
            value = np.clip(rng.normal(84, 8, size=n_rows), 50, 105)
        elif "body weight" in label:
            value = rng.normal(0, 6, size=n_rows)
        elif "hospital" in label:
            value = rng.poisson(0.6, size=n_rows).astype(float)
        else:
            value = rng.normal(0, 1, size=n_rows)

        missing = rng.random(n_rows) < 0.08
        value = value.astype(float)
        value[missing] = np.nan
        df[col] = value

    age_col = _first_feature_containing(feature_name_map, "Estimated Age")
    adl_col = _first_feature_containing(feature_name_map, "ADL Last Score")
    hosp_col = _first_feature_containing(feature_name_map, "Hospitalizations within 6 Months")
    male_col = _first_feature_containing(feature_name_map, "Male")

    linear = latent_risk
    if age_col in df:
        linear += (pd.to_numeric(df[age_col], errors="coerce").fillna(84).to_numpy() - 84) / 12
    if adl_col in df:
        linear += 0.35 * pd.to_numeric(df[adl_col], errors="coerce").fillna(0).to_numpy()
    if hosp_col in df:
        linear += 0.45 * pd.to_numeric(df[hosp_col], errors="coerce").fillna(0).to_numpy()
    if male_col in df:
        linear += 0.15 * pd.to_numeric(df[male_col], errors="coerce").fillna(0).to_numpy()

    external_shift = 0.15 if cohort == "external" else 0.0
    probability = 1 / (1 + np.exp(-(linear - 1.4 + external_shift)))
    df[outcome_col] = rng.binomial(1, probability, size=n_rows)

    return df


def _first_feature_containing(feature_name_map: dict[str, str], label: str) -> str | None:
    for source_name, english_label in feature_name_map.items():
        if english_label == label:
            return source_name
    return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    training = _make_frame(n_rows=600, seed=42, cohort="training")
    external = _make_frame(n_rows=240, seed=2024, cohort="external")

    training.to_csv(OUT_DIR / "training_data_1014_synthetic.csv", index=False, encoding="utf-8-sig")
    external.to_csv(OUT_DIR / "external_validation_1014_synthetic.csv", index=False, encoding="utf-8-sig")

    print("Synthetic demonstration data written to:")
    print(OUT_DIR / "training_data_1014_synthetic.csv")
    print(OUT_DIR / "external_validation_1014_synthetic.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
