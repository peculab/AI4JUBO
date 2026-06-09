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

BEESWARM_LABEL_OVERRIDES = {
    "BW_diff_seq": "Body Weight Change (Sequential)",
}


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


def scaled_probability_shap_values(
    prob_scale_shap_values: np.ndarray,
    feature_names: list[str],
    official: pd.DataFrame,
) -> np.ndarray:
    approx_mean = np.abs(prob_scale_shap_values).mean(axis=0)
    official_map = official.set_index("Feature")["MeanAbsSHAP"].to_dict()
    scale = np.ones(len(feature_names), dtype=float)
    for j, feature in enumerate(feature_names):
        official_mean = official_map.get(feature)
        if official_mean is not None and approx_mean[j] > 0:
            scale[j] = official_mean / approx_mean[j]
    return prob_scale_shap_values * scale[None, :]


def feature_color_values(raw_df: pd.DataFrame, feature: str) -> pd.Series:
    values = pd.to_numeric(raw_df[feature], errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.zeros(len(raw_df)), index=raw_df.index)
    return values.fillna(values.median())


def make_beeswarm_data(
    shap_values: np.ndarray,
    raw_df: pd.DataFrame,
    feature_names: list[str],
    feature_labels: dict[str, str],
    top_features: list[str],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    feature_to_idx = {feature: i for i, feature in enumerate(feature_names)}
    for rank, feature in enumerate(top_features):
        j = feature_to_idx[feature]
        values = feature_color_values(raw_df, feature).reset_index(drop=True)
        shap_col = shap_values[:, j]
        rows.append(
            pd.DataFrame(
                {
                    "Feature": feature,
                    "Feature_EN": feature_labels[feature],
                    "Rank": rank,
                    "SHAP value": shap_col,
                    "Feature value": values.to_numpy(),
                    "Y jitter": rank + rng.normal(0, 0.075, size=len(shap_col)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def make_beeswarm_plot(beeswarm_df: pd.DataFrame, feature_labels_order: list[str]) -> go.Figure:
    fig = go.Figure()
    cmin = float(np.nanpercentile(beeswarm_df["Feature value"], 1))
    cmax = float(np.nanpercentile(beeswarm_df["Feature value"], 99))
    if cmin == cmax:
        cmin = float(beeswarm_df["Feature value"].min())
        cmax = float(beeswarm_df["Feature value"].max())
    fig.add_trace(
        go.Scattergl(
            x=beeswarm_df["SHAP value"],
            y=beeswarm_df["Y jitter"],
            mode="markers",
            marker=dict(
                size=4,
                opacity=0.86,
                color=beeswarm_df["Feature value"],
                colorscale="Bluered",
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title="Feature value", len=0.88, thickness=24),
                line=dict(width=0),
            ),
            text=beeswarm_df["Feature_EN"],
            customdata=np.stack([beeswarm_df["Feature"], beeswarm_df["Feature value"]], axis=1),
            hovertemplate=(
                "%{text}<br>"
                "Original feature=%{customdata[0]}<br>"
                "SHAP value=%{x:.4f}<br>"
                "Feature value=%{customdata[1]:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_width=1.2, line_color="#4a4a4a")
    fig.update_layout(
        template="plotly_white",
        width=900,
        height=430,
        margin=dict(l=210, r=95, t=25, b=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="SHAP value (impact on model output)",
        yaxis_title="",
        font=dict(family="Arial", size=13, color="#25364a"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(feature_labels_order))),
        ticktext=feature_labels_order,
        range=[len(feature_labels_order) - 0.35, -0.65],
        showgrid=False,
        zeroline=False,
    )
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

    official = pd.read_csv(OFFICIAL_SHAP_IMPORTANCE)
    scaled_shap_values = scaled_probability_shap_values(prob_scale_shap_values, list(external_x.columns), official)

    forest_df = bootstrap_mean_abs_ci(scaled_shap_values)
    forest_df.insert(0, "Feature", external_x.columns)
    forest_df.insert(1, "Feature_EN", [rgr.FEATURE_NAME_MAP.get(c, c) for c in external_x.columns])
    forest_df = forest_df.rename(
        columns={
            "MeanAbsSHAP": "ApproxMeanAbsSHAP",
            "CI_low": "ApproxCI_low",
            "CI_high": "ApproxCI_high",
        }
    )
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

    feature_labels = {
        row.Feature: BEESWARM_LABEL_OVERRIDES.get(row.Feature, row.Feature_EN)
        for row in forest_df.itertuples(index=False)
    }
    top_features = top_df.sort_values("MeanAbsSHAP", ascending=True)["Feature"].tolist()
    beeswarm_df = make_beeswarm_data(
        scaled_shap_values,
        external_df,
        list(external_x.columns),
        feature_labels,
        top_features,
    )
    beeswarm_df.to_csv(OUT / "0609_shap_beeswarm_top8_data.csv", index=False, encoding="utf-8-sig")
    beeswarm_df.to_excel(OUT / "0609_shap_beeswarm_top8_data.xlsx", index=False)

    beeswarm_labels = [feature_labels[f] for f in top_features]
    beeswarm_fig = make_beeswarm_plot(beeswarm_df, beeswarm_labels)
    beeswarm_fig.write_html(str(OUT / "0609_shap_beeswarm_top8.html"))
    beeswarm_fig.write_image(str(OUT / "0609_shap_beeswarm_top8.png"), scale=3)
    beeswarm_fig.write_image(str(OUT / "0609_shap_beeswarm_top8.jpg"), scale=3)

    print(top_df[["Rank", "Feature", "Feature_EN", "MeanAbsSHAP", "CI_low", "CI_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
