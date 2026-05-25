"""
hyperparameter_sensitivity.py
==============================
PURPOSE
-------
Analyses how sensitive each model's CV performance is to individual
hyperparameter choices, using the *max median MAE gap* metric:

    max_median_gap = max(median CV MAE across parameter values)
                   - min(median CV MAE across parameter values)

A larger gap means the model is more sensitive to that hyperparameter.
One analysis is run per model per feature set (baseline / augmented).

INPUTS  (relative to this script in model_reliability/)
------
  ../../model result/arimax_tuning_results.csv
  ../../model result/svr_tuning_results.csv
  ../../model result/xgboost_tuning_results.csv
  ../../model result/lstcn_tuning_results.csv

OUTPUT STRUCTURE
----------------
  outputs/hyperparameter_sensitivity/
  ├── ARIMAX/
  │   ├── plots/
  │   │   ├── baseline_<param>_barplot.png    per-param median MAE bar
  │   │   ├── augmented_<param>_barplot.png
  │   │   ├── baseline_median_gap_barplot.png  importance ranking
  │   │   ├── augmented_median_gap_barplot.png
  │   │   └── combined_median_gap_barplot.png  baseline vs augmented
  │   └── summary/
  │       ├── cleaning_report.csv
  │       ├── <fset>_<param>_summary.csv
  │       ├── <fset>_<param>_median_table.csv
  │       ├── <fset>_median_gap_summary.csv
  │       ├── combined_median_gap_summary.csv
  │       └── all_median_gap_summary_long.csv
  ├── SVR/  (same structure)
  ├── XGBoost/
  ├── LSTCN/
  ├── all_models_median_gap_summary_long.csv
  └── all_models_median_gap_summary_wide.csv

PIPELINE POSITION
-----------------
  Model training scripts  →  *_tuning_results.csv
                                      ↓
                         [THIS SCRIPT]
                                      ↓
                         Chapter 5.5.3 — Hyperparameter Sensitivity

DEPENDENCIES
------------
  pandas, numpy, matplotlib
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

SCRIPT_DIR   = Path(__file__).resolve().parent
MODEL_DIR    = SCRIPT_DIR.parents[1] / "results" / "tuning"
OUTPUT_ROOT  = SCRIPT_DIR / "outputs" / "hyperparameter_sensitivity"

TARGET_COL      = "cv_MAE_scaled_y_mean"
FEATURE_SET_COL = "feature_set"
VALID_COL       = "valid"
MAE_THRESHOLD   = 2      # drop rows with CV MAE ≥ 2 (likely failed fits)

# ── Plot styling (consistent with overfitting_analysis.py) ───
COLOR_BASELINE  = "#A8C4D4"   # light blue
COLOR_AUGMENTED = "#2271B3"   # dark blue
COLOR_SINGLE    = "#2271B3"   # single-feature-set bar
HEADER_COLOR    = "#2C3E50"   # dark navy (matches table headers elsewhere)
DPI             = 150
GRID_ALPHA      = 0.4


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ARIMAX": {
        "file": "arimax_tuning_results.csv",
        "params": ["p", "d", "q", "trend"],
        "string_params": ["trend"],
        "category_orders": {"trend": ["n", "c"]},
        "plot_width": 8,
    },
    "SVR": {
        "file": "svr_tuning_results.csv",
        "params": ["kernel", "C", "epsilon", "gamma"],
        "string_params": ["kernel", "gamma"],
        "category_orders": {
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto", "0.1", "1", "nan"],
        },
        "plot_width": 8,
    },
    "XGBoost": {
        "file": "xgboost_tuning_results.csv",
        "params": [
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "reg_lambda", "reg_alpha", "min_child_weight",
        ],
        "string_params": [],
        "category_orders": {},
        "plot_width": 11,
    },
    "LSTCN": {
        "file": "lstcn_tuning_results.csv",
        "params": ["n_steps", "n_blocks", "function", "solver", "alpha"],
        "string_params": ["function", "solver", "alpha"],
        "category_orders": {
            "function": ["hyperbolic"],
            "solver": ["svd", "cholesky", "lsqr"],
            "alpha": ["0.0", "0.0001", "0.001", "0.01", "0.1"],
        },
        "plot_width": 8,
    },
}


# ============================================================
# 3. HELPERS
# ============================================================

def ensure_dirs(model_name: str) -> Dict[str, Path]:
    model_dir   = OUTPUT_ROOT / model_name
    summary_dir = model_dir / "summary"
    plots_dir   = model_dir / "plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return {"model": model_dir, "summary": summary_dir, "plots": plots_dir}


def normalize_bool_column(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    v = df[col]
    if v.dtype == bool:
        return v
    return v.astype(str).str.lower().isin(["true", "1", "yes"])


def load_and_clean(input_path: Path, model_name: str,
                   summary_dir: Path) -> pd.DataFrame:
    """Load tuning CSV, keep valid rows, drop MAE outliers, save cleaning report."""
    if not input_path.exists():
        raise FileNotFoundError(f"{model_name}: file not found at {input_path}")

    df = pd.read_csv(input_path)
    n0 = len(df)

    for col in (TARGET_COL, FEATURE_SET_COL):
        if col not in df.columns:
            raise ValueError(f"{col} not found in {input_path.name}")

    df = df[normalize_bool_column(df, VALID_COL)].copy()
    df = df[df[TARGET_COL].notna()].copy()
    df = df[df[TARGET_COL] < MAE_THRESHOLD].copy()
    n1 = len(df)

    pd.DataFrame([{
        "model": model_name, "mae_threshold": MAE_THRESHOLD,
        "original_rows": n0, "cleaned_rows": n1,
        "removed_rows": n0 - n1,
        "removed_ratio": (n0 - n1) / n0 if n0 else 0,
    }]).to_csv(summary_dir / "cleaning_report.csv", index=False)

    if df.empty:
        raise ValueError(f"{model_name}: no rows left after cleaning.")
    return df


def prepare_col(df: pd.DataFrame, param: str, string_params: List[str]):
    df = df.copy()
    if param in string_params:
        df[param] = df[param].astype(str)
    return df


def get_order(df: pd.DataFrame, param: str,
              category_orders: Dict[str, List[str]]) -> Optional[List[Any]]:
    unique = df[param].dropna().unique()
    if param in category_orders:
        return [x for x in category_orders[param]
                if str(x) in set(map(str, unique))]
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=str)


def sort_by_order(df: pd.DataFrame, param: str,
                  order: Optional[List[Any]],
                  string_params: List[str]) -> pd.DataFrame:
    if not order:
        try:
            return df.sort_values(param).reset_index(drop=True)
        except TypeError:
            return df.sort_values(param, key=lambda s: s.astype(str)).reset_index(drop=True)

    order_str = [str(x) for x in order]
    if param in string_params:
        df[param] = pd.Categorical(df[param].astype(str),
                                   categories=order_str, ordered=True)
    else:
        try:
            df = df.sort_values(param).reset_index(drop=True)
            return df
        except TypeError:
            df[param] = pd.Categorical(df[param].astype(str),
                                       categories=order_str, ordered=True)
    return df.sort_values(param).reset_index(drop=True)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 4. ANALYSIS FUNCTIONS
# ============================================================

def summarize_param(df, param, fset, summary_dir, string_params):
    if param not in df.columns or df[param].dropna().nunique() == 0:
        return None
    df_l = prepare_col(df, param, string_params)
    summary = (df_l.groupby(param)[TARGET_COL]
               .agg(["mean", "median", "std", "min", "max", "count"])
               .sort_values("mean").reset_index())
    summary.to_csv(summary_dir / f"{fset}_{param}_summary.csv", index=False)
    return summary


def calc_median_gap(df, params, fset, summary_dir,
                    string_params, category_orders) -> pd.DataFrame:
    """
    For each hyperparameter, compute:
      max_median_gap = max(median CV MAE) − min(median CV MAE)
    across all values of that parameter.

    Also records the best (lowest median MAE) and worst parameter values.
    """
    rows = []
    for param in params:
        if param not in df.columns:
            continue
        df_l = prepare_col(df, param, string_params)
        if df_l[param].dropna().nunique() <= 1:
            print(f"    [SKIP fixed] {param}")
            continue

        med = (df_l.groupby(param)[TARGET_COL]
               .median().reset_index()
               .rename(columns={TARGET_COL: "median_MAE"}))
        order = get_order(df_l, param, category_orders)
        med   = sort_by_order(med, param, order, string_params)
        # Ensure param column is string for CSV serialisation
        med[param] = med[param].astype(str)

        best  = med.loc[med["median_MAE"].idxmin()]
        worst = med.loc[med["median_MAE"].idxmax()]
        rows.append({
            "param":            param,
            "best_value":       str(best[param]),
            "best_median_MAE":  best["median_MAE"],
            "worst_value":      str(worst[param]),
            "worst_median_MAE": worst["median_MAE"],
            "max_median_gap":   worst["median_MAE"] - best["median_MAE"],
        })
        med.to_csv(summary_dir / f"{fset}_{param}_median_table.csv", index=False)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("max_median_gap", ascending=False).reset_index(drop=True)
    result.to_csv(summary_dir / f"{fset}_median_gap_summary.csv", index=False)
    return result


# ============================================================
# 5. PLOT FUNCTIONS
# ============================================================

def plot_param_median_bar(df, param, fset, model_name,
                          plots_dir, string_params, category_orders):
    """
    Bar chart: X = parameter values, Y = median CV MAE.
    Shows the effect of each individual hyperparameter value on performance.
    Colour = COLOR_SINGLE (#2271B3, dark blue).
    """
    if param not in df.columns:
        return
    df_l = prepare_col(df, param, string_params)
    if df_l[param].dropna().nunique() <= 1:
        return

    med = (df_l.groupby(param)[TARGET_COL]
           .median().reset_index()
           .rename(columns={TARGET_COL: "median_MAE"}))
    order = get_order(df_l, param, category_orders)
    med   = sort_by_order(med, param, order, string_params)
    med[param] = med[param].astype(str)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(med[param], med["median_MAE"],
                  color=COLOR_SINGLE, edgecolor="white", linewidth=0.6)

    y_max = med["median_MAE"].max() * 1.2 or 1
    ax.set_ylim(0, y_max)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + y_max * 0.015,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    feat_label = "Baseline" if fset == "baseline" else "Augmented"
    ax.set_title(f"{model_name} ({feat_label}): {param} vs Median CV MAE",
                 fontsize=11, pad=6)
    ax.set_xlabel(param, fontsize=9)
    ax.set_ylabel("Median CV MAE (scaled y)", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{fset}_{param}_barplot.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_median_gap_bar(summary_df, title, filename, plots_dir,
                        y_max=None):
    """
    Bar chart: X = hyperparameter names, Y = max median MAE gap.
    Bars sorted by gap (descending) to show importance ranking.
    Colour = COLOR_SINGLE.
    """
    if summary_df is None or summary_df.empty:
        return

    df_plot = summary_df.sort_values("max_median_gap", ascending=False).copy()

    if y_max is None:
        mv = df_plot["max_median_gap"].max()
        y_max = mv * 1.2 if mv > 0 else 1

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(df_plot["param"], df_plot["max_median_gap"],
                  color=COLOR_SINGLE, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, y_max)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + y_max * 0.015,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylabel("Max median MAE gap", fontsize=9)
    ax.set_xlabel("Hyperparameter", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_combined_gap_bar(baseline_df, augmented_df, params,
                          plots_dir, summary_dir, model_name,
                          plot_width=8):
    """
    Grouped bar chart: baseline (light blue) vs augmented (dark blue).
    X = hyperparameter names, Y = max median MAE gap.
    Mirrors the Sprint 3 slide format for SVR hyperparameter analysis.
    """
    if (baseline_df is None or baseline_df.empty or
            augmented_df is None or augmented_df.empty):
        return None

    base = baseline_df[["param", "max_median_gap"]].rename(
        columns={"max_median_gap": "baseline_gap"})
    aug  = augmented_df[["param", "max_median_gap"]].rename(
        columns={"max_median_gap": "augmented_gap"})
    combined = pd.merge(base, aug, on="param", how="outer")
    combined["param"] = pd.Categorical(combined["param"],
                                       categories=params, ordered=True)
    combined = combined.sort_values("param").reset_index(drop=True)
    combined.to_csv(summary_dir / "combined_median_gap_summary.csv", index=False)

    x     = np.arange(len(combined))
    width = 0.35

    fig, ax = plt.subplots(figsize=(plot_width, 4.8))

    bars_b = ax.bar(x - width / 2, combined["baseline_gap"],  width,
                    label="Baseline",  color=COLOR_BASELINE,
                    edgecolor="white", linewidth=0.6)
    bars_a = ax.bar(x + width / 2, combined["augmented_gap"], width,
                    label="Augmented", color=COLOR_AUGMENTED,
                    edgecolor="white", linewidth=0.6)

    mv    = combined[["baseline_gap", "augmented_gap"]].max().max()
    y_max = mv * 1.2 if mv > 0 else 1
    ax.set_ylim(0, y_max)

    for bars in (bars_b, bars_a):
        for bar in bars:
            h = bar.get_height()
            if pd.isna(h):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, h + y_max * 0.015,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    rotation = 20 if plot_width > 8 else 0
    ha       = "right" if plot_width > 8 else "center"
    ax.set_xticks(x)
    ax.set_xticklabels(combined["param"].astype(str), rotation=rotation, ha=ha)
    ax.set_ylabel("Max median MAE gap", fontsize=9)
    ax.set_xlabel("Hyperparameter", fontsize=9)
    ax.set_title(f"{model_name}: Hyperparameter importance by median MAE gap",
                 fontsize=11, pad=6)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "combined_median_gap_barplot.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return combined


# ============================================================
# 6. PER-MODEL RUNNER
# ============================================================

def run_model_analysis(model_name: str,
                       config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Full sensitivity analysis pipeline for one model."""
    print(f"\n{'─'*50}\nRunning {model_name} …\n{'─'*50}")

    dirs        = ensure_dirs(model_name)
    summary_dir = dirs["summary"]
    plots_dir   = dirs["plots"]

    input_path      = MODEL_DIR / config["file"]
    params          = config["params"]
    string_params   = config.get("string_params", [])
    category_orders = config.get("category_orders", {})
    plot_width      = config.get("plot_width", 8)

    df_clean = load_and_clean(input_path, model_name, summary_dir)

    all_gap_summaries   = []
    feature_gap_results = {}

    for fset in ("baseline", "augmented"):
        df_f = df_clean[df_clean[FEATURE_SET_COL] == fset].copy()
        if df_f.empty:
            print(f"  [SKIP] No data for {fset}")
            continue
        print(f"  Analysing {fset} …")

        # Per-parameter summary CSV + bar chart
        for param in params:
            summarize_param(df_f, param, fset, summary_dir, string_params)
            plot_param_median_bar(df_f, param, fset, model_name,
                                  plots_dir, string_params, category_orders)

        # Gap summary (importance ranking)
        gap = calc_median_gap(df_f, params, fset, summary_dir,
                              string_params, category_orders)
        feature_gap_results[fset] = gap

        if not gap.empty:
            g = gap.copy()
            g.insert(0, "feature_set", fset)
            g.insert(0, "model", model_name)
            all_gap_summaries.append(g)

    base_gap = feature_gap_results.get("baseline", pd.DataFrame())
    aug_gap  = feature_gap_results.get("augmented", pd.DataFrame())

    # Shared y-axis across single-fset importance bar charts
    mv_list = []
    if not base_gap.empty:
        mv_list.append(base_gap["max_median_gap"].max())
    if not aug_gap.empty:
        mv_list.append(aug_gap["max_median_gap"].max())
    common_y = max(mv_list) * 1.2 if mv_list and max(mv_list) > 0 else 1

    plot_median_gap_bar(
        base_gap,
        title=f"{model_name} (Baseline): Hyperparameter importance",
        filename="baseline_median_gap_barplot.png",
        plots_dir=plots_dir, y_max=common_y)

    plot_median_gap_bar(
        aug_gap,
        title=f"{model_name} (Augmented): Hyperparameter importance",
        filename="augmented_median_gap_barplot.png",
        plots_dir=plots_dir, y_max=common_y)

    plot_combined_gap_bar(
        base_gap, aug_gap, params=params,
        plots_dir=plots_dir, summary_dir=summary_dir,
        model_name=model_name, plot_width=plot_width)

    if all_gap_summaries:
        model_summary = pd.concat(all_gap_summaries, ignore_index=True)
        model_summary.to_csv(
            summary_dir / "all_median_gap_summary_long.csv", index=False)
        return model_summary

    return None


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("hyperparameter_sensitivity.py")
    print("=" * 60)
    print(f"Input  : {MODEL_DIR}")
    print(f"Output : {OUTPUT_ROOT}")

    all_summaries = []

    for model_name, config in MODEL_CONFIGS.items():
        try:
            summary = run_model_analysis(model_name, config)
            if summary is not None:
                all_summaries.append(summary)
        except Exception as exc:
            print(f"\n[ERROR] {model_name}: {exc}")

    if all_summaries:
        final = pd.concat(all_summaries, ignore_index=True)
        final.to_csv(OUTPUT_ROOT / "all_models_median_gap_summary_long.csv",
                     index=False)

        wide = final.pivot_table(
            index=["model", "param"],
            columns="feature_set",
            values="max_median_gap",
            aggfunc="first",
        ).reset_index()
        wide.to_csv(OUTPUT_ROOT / "all_models_median_gap_summary_wide.csv",
                    index=False)

        print("\nOverall summaries saved:")
        print(f"  {OUTPUT_ROOT / 'all_models_median_gap_summary_long.csv'}")
        print(f"  {OUTPUT_ROOT / 'all_models_median_gap_summary_wide.csv'}")

    print("\n✅  Done.")


if __name__ == "__main__":
    main()
