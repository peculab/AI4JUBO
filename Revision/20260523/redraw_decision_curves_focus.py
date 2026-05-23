from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Revision" / "20260523"
SEARCH_ROOTS = [
    ROOT / "Revision" / "20260523",
    ROOT / "Revision" / "01_需蔡老師補做資料表",
    ROOT / "RESULTS" / "tables",
]

# Reviewer asked to reduce the Net Benefit Y-axis scale to the 0 to -2 area.
# We set the lower bound to -2 and keep small positive headroom because all
# model net-benefit curves sit above zero for clinically relevant thresholds.
Y_LIM = (-2.0, 0.30)
Y_TICKS = [-2.0, -1.5, -1.0, -0.5, 0, 0.1, 0.2, 0.3]

COLORS = {
    "Treat None": "#222222",
    "Treat All": "#888888",
    "HybridXGBRF (Our Approach)": "#d62728",
    "XGBClassifier": "#1f77b4",
    "RandomForestClassifier": "#2ca02c",
    "LogisticRegression (max_iter=200)": "#9467bd",
    "LogisticRegression (max_iter=1000)": "#8c564b",
    "Ridge": "#e377c2",
    "Lasso": "#7f7f7f",
    "Elastic": "#bcbd22",
}


def find_sources() -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.glob("decision_curve*.xlsx"):
            if "manifest" in path.name.lower():
                continue
            sources.setdefault(path.name, path)
    return sources


def plot_title(stem: str) -> str:
    if "external" in stem:
        return "Decision Curve Analysis: External Validation"
    if "internal" in stem:
        return "Decision Curve Analysis: Internal Cross-Validation"
    return stem.replace("_", " ").title()


def redraw(xlsx: Path, png: Path) -> dict[str, object]:
    df = pd.read_excel(xlsx)
    required = {"Model", "Threshold", "Net Benefit"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{xlsx} missing columns: {sorted(missing)}")

    df = df.sort_values(["Model", "Threshold"])
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    for model, g in df.groupby("Model", sort=False):
        lw = 2.8 if model == "HybridXGBRF (Our Approach)" else 1.8
        ls = "--" if model in {"Treat None", "Treat All"} else "-"
        alpha = 1.0 if model == "HybridXGBRF (Our Approach)" else 0.86
        ax.plot(
            g["Threshold"],
            g["Net Benefit"],
            label=model,
            color=COLORS.get(model),
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
        )

    ax.axhline(0, color="#333333", linewidth=0.9, alpha=0.6)
    ax.set_title(plot_title(xlsx.stem))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.grid(True, axis="both", alpha=0.22)
    ax.legend(loc="lower left", fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(png, dpi=220)
    plt.close(fig)

    model_range = df.groupby("Model")["Net Benefit"].agg(["min", "max"]).reset_index()
    return {
        "source": str(xlsx),
        "output": str(png),
        "n_rows": len(df),
        "models": df["Model"].nunique(),
        "overall_min": float(df["Net Benefit"].min()),
        "overall_max": float(df["Net Benefit"].max()),
        "model_range": model_range,
    }


def bk4_alias_for(png: Path) -> Path | None:
    if "external_validation" in png.stem:
        return png.with_name("BK4_decision_curve_external_validation_yaxis_minus2.png")
    if "internal_cv" in png.stem:
        return png.with_name("BK4_decision_curve_internal_cv_yaxis_minus2.png")
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = find_sources()
    rows = []
    details = []

    for name, src in sorted(sources.items()):
        dst_xlsx = OUT / name
        if src.resolve() != dst_xlsx.resolve():
            dst_xlsx.write_bytes(src.read_bytes())
        dst_png = OUT / f"{Path(name).stem}.png"
        info = redraw(dst_xlsx, dst_png)
        alias = bk4_alias_for(dst_png)
        if alias is not None:
            alias.write_bytes(dst_png.read_bytes())
            info["bk4_alias"] = str(alias)
        else:
            info["bk4_alias"] = ""
        rows.append({k: v for k, v in info.items() if k != "model_range"})
        details.append((name, info["model_range"]))
        print(f"redrew {dst_png}")

    manifest = pd.DataFrame(rows)
    manifest_path = OUT / "decision_curve_redraw_manifest_20260523.xlsx"
    with pd.ExcelWriter(manifest_path) as writer:
        manifest.to_excel(writer, index=False, sheet_name="Redrawn files")
        for name, model_range in details:
            sheet = Path(name).stem[:31]
            model_range.to_excel(writer, index=False, sheet_name=sheet)

    md = [
        "# Decision Curve Redraw Manifest",
        "",
        "Reviewer note in `01_請蔡老師協助的部分2.docx`: decision curve / BK4 decision curve; try reducing the Net Benefit Y-axis scale.",
        "",
        f"Y-axis used for all redrawn decision-curve PNGs: `{Y_LIM[0]}` to `{Y_LIM[1]}`.",
        "This keeps the requested negative range down to -2 while preserving the small positive net-benefit region where the model curves are visible.",
        "",
        "Redrawn files:",
    ]
    for row in rows:
        md.append(f"- `{Path(row['output']).name}` from `{Path(row['source']).name}`")
        if row.get("bk4_alias"):
            md.append(f"  - BK4 alias: `{Path(row['bk4_alias']).name}`")
    md.append("")
    md.append("These are supplementary decision-curve-analysis figures and do not change the FINAL manuscript model-performance point estimates.")
    (OUT / "decision_curve_redraw_manifest_20260523.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
