from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Revision" / "20260609"
TRAINING_CACHE = ROOT / "Revision" / "20260523" / "training_data_1014_cached_for_completion.csv"
EXTERNAL_CACHE = ROOT / "Revision" / "20260523" / "external_validation_1014_cached_for_completion.csv"
OFFICIAL_SHAP_IMPORTANCE = ROOT / "RESULTS" / "tables" / "shap_feature_importance.csv"

sys.path.insert(0, str(ROOT))
import revision_generate_results as rgr

TOP_N = 8
N_BOOT = 1000
RANDOM_STATE = 42


def bootstrap_mean_abs_ci(values: np.ndarray, random_state: int = RANDOM_STATE, n_boot: int = N_BOOT) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    abs_values = np.abs(values)
    n, p = abs_values.shape
    means = abs_values.mean(axis=0)
    boot = np.empty((n_boot, p), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = abs_values[idx].mean(axis=0)
    return pd.DataFrame(
        {
            "MeanAbsSHAP": means,
            "CI_low": np.percentile(boot, 2.5, axis=0),
            "CI_high": np.percentile(boot, 97.5, axis=0),
        }
    )


def make_forest_plot(df: pd.DataFrame) -> go.Figure:
    plot_df = df.sort_values("MeanAbsSHAP", ascending=True).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["MeanAbsSHAP"],
            y=plot_df["Feature_EN"],
            mode="markers",
            marker=dict(size=7, color="#5567d9", line=dict(width=1, color="#5567d9")),
            error_x=dict(
                type="data",
                symmetric=False,
                array=plot_df["CI_high"] - plot_df["MeanAbsSHAP"],
                arrayminus=plot_df["MeanAbsSHAP"] - plot_df["CI_low"],
                color="#5567d9",
                thickness=1.5,
                width=4,
            ),
            hovertemplate=(
                "%{y}<br>"
                "Mean |SHAP|=%{x:.5f}<br>"
                "95% CI=%{customdata[0]:.5f}-%{customdata[1]:.5f}<extra></extra>"
            ),
            customdata=np.stack([plot_df["CI_low"], plot_df["CI_high"]], axis=1),
            showlegend=False,
        )
    )
    fig.update_layout(
        template="plotly",
        width=900,
        height=430,
        margin=dict(l=210, r=40, t=25, b=70),
        paper_bgcolor="white",
        plot_bgcolor="#e5ecf6",
        xaxis_title="Mean |SHAP|",
        yaxis_title="Feature",
        font=dict(family="Arial", size=13, color="#25364a"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="white", zeroline=False, tickformat=".2f")
    fig.update_yaxes(showgrid=True, gridcolor="white", autorange="reversed")
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAINING_CACHE)
    external_df = pd.read_csv(EXTERNAL_CACHE)
    features = rgr.select_features(train_df)
    preprocessing = rgr.fit_preprocessing(train_df, features)
    train_x, train_y = rgr.prepare_xy(train_df, features, preprocessing)
    external_x, _ = rgr.prepare_xy(external_df, features, preprocessing)

    model = rgr.build_models()["HybridXGBRF (Our Approach)"]
    model.fit(train_x, train_y)

    dmatrix = xgb.DMatrix(external_x, feature_names=list(external_x.columns))
    contribs = np.asarray(model.get_booster().predict(dmatrix, pred_contribs=True))
    margin_shap_values = contribs[:, :-1]
    proba = model.predict_proba(external_x)[:, 1]
    # xgboost pred_contribs is on the raw margin scale. The current official SHAP
    # importance table is on the final-output scale, so use a first-order
    # probability-scale approximation only to estimate interval widths.
    prob_scale_shap_values = margin_shap_values * (proba * (1 - proba))[:, None]

    forest_df = bootstrap_mean_abs_ci(prob_scale_shap_values)
    forest_df.insert(0, "Feature", external_x.columns)
    forest_df.insert(1, "Feature_EN", [rgr.FEATURE_NAME_MAP.get(c, c) for c in external_x.columns])
    forest_df = forest_df.rename(
        columns={
            "MeanAbsSHAP": "ApproxMeanAbsSHAP",
            "CI_low": "ApproxCI_low",
            "CI_high": "ApproxCI_high",
        }
    )
    official = pd.read_csv(OFFICIAL_SHAP_IMPORTANCE)
    forest_df = forest_df.merge(
        official[["Feature", "Feature_EN", "MeanAbsSHAP"]].rename(
            columns={"Feature_EN": "OfficialFeature_EN", "MeanAbsSHAP": "OfficialMeanAbsSHAP"}
        ),
        on="Feature",
        how="left",
    )
    forest_df["Feature_EN"] = forest_df["OfficialFeature_EN"].fillna(forest_df["Feature_EN"])
    scale = forest_df["OfficialMeanAbsSHAP"] / forest_df["ApproxMeanAbsSHAP"].replace(0, np.nan)
    forest_df["MeanAbsSHAP"] = forest_df["OfficialMeanAbsSHAP"].fillna(forest_df["ApproxMeanAbsSHAP"])
    forest_df["CI_low"] = forest_df["MeanAbsSHAP"] - (
        forest_df["ApproxMeanAbsSHAP"] - forest_df["ApproxCI_low"]
    ) * scale.fillna(1.0)
    forest_df["CI_high"] = forest_df["MeanAbsSHAP"] + (
        forest_df["ApproxCI_high"] - forest_df["ApproxMeanAbsSHAP"]
    ) * scale.fillna(1.0)
    forest_df["CI_low"] = forest_df["CI_low"].clip(lower=0)
    forest_df = forest_df.sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
    forest_df["Rank"] = np.arange(1, len(forest_df) + 1)
    forest_df["Bootstrap N"] = N_BOOT
    forest_df["SHAP cohort"] = "Temporal external validation for CI; official point estimates from RESULTS/tables"
    forest_df["SHAP rows"] = len(external_x)
    forest_df["CI method"] = "Bootstrap interval width from probability-scale XGBoost TreeSHAP approximation, rescaled to official MeanAbsSHAP"

    forest_df.to_csv(OUT / "0609_shap_forest_plot_data.csv", index=False, encoding="utf-8-sig")
    forest_df.to_excel(OUT / "0609_shap_forest_plot_data.xlsx", index=False)

    top_df = forest_df.head(TOP_N).copy()
    fig = make_forest_plot(top_df)
    fig.write_html(str(OUT / "0609_shap_forest_plot_top8.html"))
    fig.write_image(str(OUT / "0609_shap_forest_plot_top8.png"), scale=3)
    fig.write_image(str(OUT / "0609_shap_forest_plot_top8.jpg"), scale=3)

    print(top_df[["Rank", "Feature", "Feature_EN", "MeanAbsSHAP", "CI_low", "CI_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
