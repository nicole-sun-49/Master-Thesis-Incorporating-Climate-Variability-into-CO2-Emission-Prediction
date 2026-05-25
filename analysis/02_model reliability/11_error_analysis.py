"""
error_analysis.py
==================
OUTPUT STRUCTURE
----------------
  outputs/error_analysis/
  ├── prediction_vs_actual/          MAIN TEXT  12 PNGs (6 countries × 2 feature sets)
  ├── heatmap/                       MAIN TEXT  10 PNGs + 2 CSVs (5 models × 2 feature sets)
  ├── appendix/
  │   ├── heatmap_per_country/       APPENDIX   12 PNGs (6 countries × 2 feature sets)
  │   └── lineplots/                 APPENDIX   24 PNGs (6 countries × 4 models)
  └── data/                          2 CSVs

INPUTS (relative to script location in model_reliability/)
------
  ../../model result/arimax_predictions_all.csv
  ../../model result/svr_predictions_all.csv
  ../../model result/xgboost_predictions_all.csv
  ../../model result/lstcn_predictions_all.csv

DEPENDENCIES: pandas, numpy, matplotlib, seaborn
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, "../../model result")
OUT_BASE   = os.path.join(SCRIPT_DIR, "outputs", "error_analysis")

PRED_FILES = {
    "ARIMAX":  "arimax_predictions_all.csv",
    "SVR":     "svr_predictions_all.csv",
    "XGBoost": "xgboost_predictions_all.csv",
    "LSTCN":   "lstcn_predictions_all.csv",
}

COUNTRY_ORDER = [
    "Canada", "Russian Federation", "China",
    "United States", "India", "Indonesia",
]
COUNTRY_SAFE = {
    "Canada":             "Canada",
    "Russian Federation": "RussianFederation",
    "China":              "China",
    "United States":      "UnitedStates",
    "India":              "India",
    "Indonesia":          "Indonesia",
}
MODEL_COLORS = {
    "ARIMAX":  "#1f77b4",
    "SVR":     "#ff7f0e",
    "XGBoost": "#2ca02c",
    "LSTCN":   "#d62728",
}
ACTUAL_COLOR = "black"
ACTUAL_LW    = 2.2
TEST_YEARS   = list(range(2017, 2024))


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_predictions():
    dfs = {}
    for model, fname in PRED_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"  [WARN] {model}: {fname} not found — skipped.")
            continue
        df = pd.read_csv(path)
        if "Country Name" in df.columns and "country" not in df.columns:
            df = df.rename(columns={"Country Name": "country"})
        df["year"]      = df["year"].astype(int)
        df["actual"]    = pd.to_numeric(df["actual"],    errors="coerce")
        df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
        dfs[model] = df
    return dfs


def get_series(pred_dfs, model, country, feature_set):
    df = pred_dfs.get(model)
    if df is None:
        return pd.DataFrame()
    mask = (df["country"] == country) & (df["feature_set"] == feature_set)
    return df[mask][["year", "actual", "predicted"]].sort_values("year").copy()


# ============================================================
# 3. OUTPUT ①  prediction_vs_actual  (12 PNGs — main text)
# ============================================================

def plot_prediction_vs_actual(pred_dfs, out_dir):
    """
    12 PNGs: 6 countries × 2 feature sets.
    Y-axis shared within the same country so baseline/augmented
    can be directly compared when placed side by side in the thesis.
    """
    os.makedirs(out_dir, exist_ok=True)

    for country in COUNTRY_ORDER:
        # Shared y-limits across both feature sets + all models
        all_y = []
        for fset in ("baseline", "augmented"):
            for model in MODEL_COLORS:
                s = get_series(pred_dfs, model, country, fset)
                if not s.empty:
                    all_y.extend(s["actual"].dropna().tolist())
                    all_y.extend(s["predicted"].dropna().tolist())

        if all_y:
            margin = (max(all_y) - min(all_y)) * 0.05
            y_min, y_max = min(all_y) - margin, max(all_y) + margin
        else:
            y_min, y_max = None, None

        for fset in ("baseline", "augmented"):
            feat_label = "Baseline" if fset == "baseline" else "Augmented"
            fig, ax = plt.subplots(figsize=(7, 4.5))

            for model in MODEL_COLORS:
                s = get_series(pred_dfs, model, country, fset)
                if not s.empty:
                    ax.plot(s["year"], s["actual"],
                            color=ACTUAL_COLOR, linewidth=ACTUAL_LW,
                            marker="s", markersize=5, label="Actual", zorder=5)
                    break

            for model in MODEL_COLORS:
                s = get_series(pred_dfs, model, country, fset)
                if s.empty:
                    continue
                ax.plot(s["year"], s["predicted"],
                        color=MODEL_COLORS[model], linewidth=1.6,
                        linestyle="--", marker="s", markersize=4,
                        label=model, alpha=0.85)

            if y_min is not None:
                ax.set_ylim(y_min, y_max)

            ax.set_title(f"CO₂ Predictions for {country} ({feat_label})", fontsize=11)
            ax.set_xlabel("Year", fontsize=9)
            ax.set_ylabel("CO₂ Emissions (Mt CO₂e)", fontsize=9)
            ax.set_xticks(TEST_YEARS)
            ax.set_xticklabels(TEST_YEARS, rotation=45, ha="right", fontsize=8)
            ax.legend(fontsize=8, loc="best", framealpha=0.7)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            sns.despine(ax=ax)
            fig.tight_layout()

            fname = os.path.join(out_dir, f"{COUNTRY_SAFE[country]}_{fset}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: prediction_vs_actual/{COUNTRY_SAFE[country]}_{fset}.png")


# ============================================================
# 4. MAPE HELPER
# ============================================================

def compute_mape_matrix(pred_dfs, feature_set):
    """Returns {model: DataFrame(country x year, MAPE %)} plus 'Average'."""
    matrices = {}
    for model in MODEL_COLORS:
        mat = pd.DataFrame(index=COUNTRY_ORDER, columns=TEST_YEARS, dtype=float)
        for country in COUNTRY_ORDER:
            s = get_series(pred_dfs, model, country, feature_set)
            if s.empty:
                continue
            s = s[s["year"].isin(TEST_YEARS)].copy()
            s["mape"] = (s["actual"] - s["predicted"]).abs() / s["actual"].abs() * 100
            for _, row in s.iterrows():
                mat.loc[country, int(row["year"])] = row["mape"]
        matrices[model] = mat.astype(float)

    stack = np.stack([m.values for m in matrices.values()], axis=0)
    matrices["Average"] = pd.DataFrame(
        np.nanmean(stack, axis=0), index=COUNTRY_ORDER, columns=TEST_YEARS)
    return matrices


def _shared_vmax(matrices):
    all_vals = np.concatenate([m.values.flatten() for m in matrices.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    return float(np.percentile(all_vals, 95))


# ============================================================
# 5. OUTPUT ②  per-model heatmap  (10 PNGs — main text)
# ============================================================

def plot_mape_heatmap_per_model(pred_dfs, feature_set, out_dir, csv_dir):
    """
    One PNG per model + Average per feature set (5 × 2 = 10 total).
    Rows = 6 countries, Columns = 2017-2023.
    Saved to: heatmap/
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    matrices   = compute_mape_matrix(pred_dfs, feature_set)
    feat_label = "Baseline" if feature_set == "baseline" else "Augmented"
    vmax       = _shared_vmax(matrices)

    for model in ["ARIMAX", "SVR", "XGBoost", "LSTCN", "Average"]:
        mat   = matrices[model]
        annot = mat.applymap(lambda v: f"{v:.1f}" if not np.isnan(v) else "—")

        fig, ax = plt.subplots(figsize=(8, 3.8))
        sns.heatmap(mat, ax=ax, cmap="YlOrRd", vmin=0.0, vmax=vmax,
                    annot=annot, fmt="s", annot_kws={"size": 9},
                    linewidths=0.5, linecolor="white",
                    cbar=True, cbar_kws={"label": "MAPE (%)"})
        ax.set_title(
            f"MAPE Heatmap — {feat_label} Models ({model})\n"
            "(darker = larger prediction error)", fontsize=11, pad=8)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Country", fontsize=9)
        ax.set_xticklabels(TEST_YEARS, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)
        fig.tight_layout()

        fname = os.path.join(out_dir, f"mape_heatmap_{feature_set}_{model}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: heatmap/mape_heatmap_{feature_set}_{model}.png")

    # Save long-format CSV to data/
    long_rows = []
    for model, mat in matrices.items():
        for country in COUNTRY_ORDER:
            for year in TEST_YEARS:
                val = mat.loc[country, year]
                long_rows.append({
                    "feature_set": feature_set, "model": model,
                    "country": country, "year": year,
                    "MAPE_pct": round(val, 4) if not np.isnan(val) else np.nan,
                })
    pd.DataFrame(long_rows).to_csv(
        os.path.join(csv_dir, f"mape_heatmap_{feature_set}.csv"), index=False)
    print(f"  Saved: data/mape_heatmap_{feature_set}.csv")


# ============================================================
# 5b. Combined 5-panel heatmap  (2 PNGs — main text)
# ============================================================

def plot_mape_heatmap_combined(pred_dfs, feature_set, out_dir):
    """
    One figure with 5 side-by-side panels: ARIMAX | SVR | XGBoost | LSTCN | Average.
    Rows = 6 countries, Columns = 2017-2023.
    Shared colour scale across all panels.
    Only ARIMAX panel shows country labels; only Average panel shows colourbar.

    Useful for at-a-glance cross-model comparison in the main text.

    Naming: mape_heatmap_<feature_set>_combined.png
    """
    os.makedirs(out_dir, exist_ok=True)

    matrices   = compute_mape_matrix(pred_dfs, feature_set)
    feat_label = "Baseline" if feature_set == "baseline" else "Augmented"
    vmax       = _shared_vmax(matrices)
    panel_order = ["ARIMAX", "SVR", "XGBoost", "LSTCN", "Average"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.suptitle(
        f"MAPE Heatmap — {feat_label} Models (all models)\n"
        "(darker = larger prediction error)",
        fontsize=12, y=1.02,
    )

    for ax, model in zip(axes, panel_order):
        mat   = matrices[model]
        annot = mat.applymap(lambda v: f"{v:.1f}" if not np.isnan(v) else "—")

        sns.heatmap(
            mat, ax=ax, cmap="YlOrRd", vmin=0.0, vmax=vmax,
            annot=annot, fmt="s", annot_kws={"size": 7.5},
            linewidths=0.4, linecolor="white",
            cbar=(model == "Average"),
            cbar_kws={"label": "MAPE (%)"},
            yticklabels=(model == "ARIMAX"),
        )
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_xlabel("Year", fontsize=8)
        if model == "ARIMAX":
            ax.set_ylabel("Country", fontsize=8)
        ax.set_xticklabels(TEST_YEARS, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)

    fig.tight_layout()
    fname = os.path.join(out_dir, f"mape_heatmap_{feature_set}_combined.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: heatmap/mape_heatmap_{feature_set}_combined.png")


# ============================================================
# 6. OUTPUT ③a  per-country heatmap  (12 PNGs — appendix)
# ============================================================

def plot_mape_heatmap_per_country(pred_dfs, feature_set, out_dir):
    """
    One PNG per country: rows = 4 models, columns = 2017-2023.
    Shared colour scale with per-model heatmaps.
    Saved to: appendix/heatmap_per_country/
    """
    os.makedirs(out_dir, exist_ok=True)

    matrices   = compute_mape_matrix(pred_dfs, feature_set)
    feat_label = "Baseline" if feature_set == "baseline" else "Augmented"
    vmax       = _shared_vmax(matrices)
    model_order = ["ARIMAX", "SVR", "XGBoost", "LSTCN"]

    for country in COUNTRY_ORDER:
        mat = pd.DataFrame(index=model_order, columns=TEST_YEARS, dtype=float)
        for model in model_order:
            mm = matrices.get(model)
            if mm is None or country not in mm.index:
                continue
            for year in TEST_YEARS:
                mat.loc[model, year] = mm.loc[country, year]
        mat   = mat.astype(float)
        annot = mat.applymap(lambda v: f"{v:.1f}" if not np.isnan(v) else "—")

        fig, ax = plt.subplots(figsize=(8, 3.2))
        sns.heatmap(mat, ax=ax, cmap="YlOrRd", vmin=0.0, vmax=vmax,
                    annot=annot, fmt="s", annot_kws={"size": 9.5},
                    linewidths=0.5, linecolor="white",
                    cbar=True, cbar_kws={"label": "MAPE (%)"})
        ax.set_title(
            f"MAPE Heatmap — {country} ({feat_label})\n"
            "(darker = larger prediction error)", fontsize=11, pad=8)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Model", fontsize=9)
        ax.set_xticklabels(TEST_YEARS, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)
        fig.tight_layout()

        fname = os.path.join(
            out_dir, f"mape_heatmap_{feature_set}_{COUNTRY_SAFE[country]}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: appendix/heatmap_per_country/"
              f"mape_heatmap_{feature_set}_{COUNTRY_SAFE[country]}.png")


# ============================================================
# 7. OUTPUT ③b  appendix line plots  (24 PNGs — appendix)
# ============================================================

def plot_appendix_figures(pred_dfs, out_dir):
    """
    24 PNGs: 6 countries × 4 models.
    Each figure: actual (black) + baseline (blue dashed) + augmented (orange dashed).
    Y-axis shared within each figure.
    Saved to: appendix/lineplots/
    """
    os.makedirs(out_dir, exist_ok=True)

    for country in COUNTRY_ORDER:
        for model in MODEL_COLORS:
            s_base = get_series(pred_dfs, model, country, "baseline")
            s_aug  = get_series(pred_dfs, model, country, "augmented")

            if s_base.empty and s_aug.empty:
                print(f"  [SKIP] {country} | {model}: no data")
                continue

            fig, ax = plt.subplots(figsize=(8, 4.5))

            # Shared y-limits
            all_y = []
            for s in (s_base, s_aug):
                if not s.empty:
                    all_y.extend(s["actual"].dropna().tolist())
                    all_y.extend(s["predicted"].dropna().tolist())
            if all_y:
                margin = (max(all_y) - min(all_y)) * 0.05
                ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

            actual_src = s_base if not s_base.empty else s_aug
            ax.plot(actual_src["year"], actual_src["actual"],
                    color=ACTUAL_COLOR, linewidth=ACTUAL_LW,
                    marker="s", markersize=5, label="Actual", zorder=5)

            if not s_base.empty:
                ax.plot(s_base["year"], s_base["predicted"],
                        color="#1f77b4", linewidth=1.6, linestyle="--",
                        marker="o", markersize=4, label="Baseline", alpha=0.85)
            if not s_aug.empty:
                ax.plot(s_aug["year"], s_aug["predicted"],
                        color="#ff7f0e", linewidth=1.6, linestyle="--",
                        marker="o", markersize=4, label="Augmented", alpha=0.85)

            ax.set_title(
                f"{country} — {model}: Actual vs Baseline / Augmented",
                fontsize=11)
            ax.set_xlabel("Year", fontsize=9)
            ax.set_ylabel("CO₂ Emissions (Mt CO₂e)", fontsize=9)
            ax.set_xticks(TEST_YEARS)
            ax.set_xticklabels(TEST_YEARS, rotation=45, ha="right", fontsize=8)
            ax.legend(fontsize=9, framealpha=0.7)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            sns.despine(ax=ax)
            fig.tight_layout()

            fname = os.path.join(out_dir, f"{COUNTRY_SAFE[country]}_{model}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  Saved: appendix/lineplots/{COUNTRY_SAFE[country]}_*.png  (4 figures)")


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("error_analysis.py")
    print("=" * 60)

    dir_pva           = os.path.join(OUT_BASE, "prediction_vs_actual")
    dir_heatmap       = os.path.join(OUT_BASE, "heatmap")
    dir_appendix_heat = os.path.join(OUT_BASE, "appendix", "heatmap_per_country")
    dir_appendix_line = os.path.join(OUT_BASE, "appendix", "lineplots")
    dir_data          = os.path.join(OUT_BASE, "data")

    for d in (dir_pva, dir_heatmap, dir_appendix_heat,
              dir_appendix_line, dir_data):
        os.makedirs(d, exist_ok=True)

    print("\nLoading prediction CSVs …")
    pred_dfs = load_predictions()
    if not pred_dfs:
        raise RuntimeError("No prediction CSVs loaded. Check MODEL_DIR and filenames.")
    print(f"  Loaded: {list(pred_dfs.keys())}")

    print("\n[1/6] Prediction vs actual — 12 PNGs (→ prediction_vs_actual/) …")
    plot_prediction_vs_actual(pred_dfs, dir_pva)

    print("\n[2/8] MAPE heatmap per model — baseline (→ heatmap/) …")
    plot_mape_heatmap_per_model(pred_dfs, "baseline", dir_heatmap, dir_data)

    print("\n[3/8] MAPE heatmap per model — augmented (→ heatmap/) …")
    plot_mape_heatmap_per_model(pred_dfs, "augmented", dir_heatmap, dir_data)

    print("\n[4/8] MAPE heatmap combined — baseline (→ heatmap/) …")
    plot_mape_heatmap_combined(pred_dfs, "baseline", dir_heatmap)

    print("\n[5/8] MAPE heatmap combined — augmented (→ heatmap/) …")
    plot_mape_heatmap_combined(pred_dfs, "augmented", dir_heatmap)

    print("\n[6/8] MAPE heatmap per country — baseline (→ appendix/heatmap_per_country/) …")
    plot_mape_heatmap_per_country(pred_dfs, "baseline", dir_appendix_heat)

    print("\n[7/8] MAPE heatmap per country — augmented (→ appendix/heatmap_per_country/) …")
    plot_mape_heatmap_per_country(pred_dfs, "augmented", dir_appendix_heat)

    print("\n[8/8] Appendix line plots (→ appendix/lineplots/) …")
    plot_appendix_figures(pred_dfs, dir_appendix_line)

    print("\n✅  All done.")
    print(f"\n  outputs/error_analysis/")
    print(f"  ├── prediction_vs_actual/           12 PNGs  (main text)")
    print(f"  ├── heatmap/                        12 PNGs + 2 CSVs  (main text)")
    print(f"  │   ├── mape_heatmap_<fset>_<model>.png   10 PNGs (individual)")
    print(f"  │   └── mape_heatmap_<fset>_combined.png   2 PNGs (5-panel combined)")
    print(f"  ├── appendix/")
    print(f"  │   ├── heatmap_per_country/        12 PNGs")
    print(f"  │   └── lineplots/                  24 PNGs")
    print(f"  └── data/                            2 CSVs")


if __name__ == "__main__":
    main()
