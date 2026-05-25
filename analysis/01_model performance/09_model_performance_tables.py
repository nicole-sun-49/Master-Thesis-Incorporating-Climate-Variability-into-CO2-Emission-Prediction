"""
model_performance_tables.py
============================
PURPOSE
-------
Reads the four model metrics CSVs (ARIMAX, SVR, XGBoost, LSTCN) and
produces all summary tables needed for Chapter 5 of the thesis.

OUTPUT STRUCTURE
----------------
  (same folder as this script)/
  ├── tables/
  │   ├── overview_table.png            5-1 opener: 4 models × avg MAE/RMSE baseline+augmented
  │   ├── baseline_MAE_table.png        5-1 detail: 4 models × 6 countries, MAE
  │   ├── baseline_RMSE_table.png       5-1 detail: 4 models × 6 countries, RMSE
  │   ├── augmented_MAE_table.png       5-2 detail: same format, augmented
  │   ├── augmented_RMSE_table.png      5-2 detail: same format, augmented
  │   ├── delta_MAE_table.png           5-3: augmented − baseline, MAE
  │   └── delta_RMSE_table.png          5-3: augmented − baseline, RMSE
  └── data/
      ├── overview_table.csv
      ├── baseline_MAE_table.csv
      ├── baseline_RMSE_table.csv
      ├── augmented_MAE_table.csv
      ├── augmented_RMSE_table.csv
      ├── delta_MAE_table.csv
      └── delta_RMSE_table.csv

TABLE DESCRIPTIONS
------------------
  overview_table
    Rows    : ARIMAX | SVR | XGBoost | LSTCN
    Columns : Baseline MAE | Augmented MAE | Baseline RMSE | Augmented RMSE
              | Baseline MAPE | Augmented MAPE
    Purpose : First impression — lets the reader see all models at a glance
              before diving into per-country details.

  baseline_MAE_table  /  augmented_MAE_table
  baseline_RMSE_table /  augmented_RMSE_table
    Rows    : ARIMAX | SVR | XGBoost | LSTCN | Model Average
    Columns : Canada | Russian Federation | China | United States
              | India | Indonesia | Average
    Purpose : Per-country breakdown of model performance within one feature set.
    Note    : MAPE tables are generated as CSV only (appendix use).

  delta_MAE_table  /  delta_RMSE_table
    Rows    : ARIMAX | SVR | XGBoost | LSTCN | Model Average
    Columns : same 7 country columns
    Values  : Augmented − Baseline  (negative = improvement, positive = worse)
    Colour  : green cell = improvement (< −0.001), red cell = degradation (> +0.001)
    Purpose : 5-3 comparison section.

PIPELINE POSITION
-----------------
  Model scripts (ARIMAX/SVR/XGBoost/LSTCN)
        ↓  (produce metrics CSVs)
  [THIS SCRIPT]
        ↓
  Chapter 5 tables (PNG + CSV)

DEPENDENCIES
------------
  pandas, numpy, matplotlib
  Install: pip install pandas numpy matplotlib
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.simplefilter("ignore", UserWarning)

# ============================================================
# 1. CONFIGURATION  ← edit paths here if filenames differ
# ============================================================

# This script lives in:
#   result interpretation/model performance/
# All model result CSVs are two levels up in "model result/".
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))   
MODEL_RESULT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results", "metrics")

# ── Model metrics CSV filenames ───────────────────────────────
MODEL_FILES = {
    "ARIMAX":   "arimax_metrics_summary.csv",
    "SVR":      "svr_metrics_summary.csv",
    "XGBoost":  "xgboost_metrics_summary.csv",
    "LSTCN":    "lstcn_metrics_summary.csv",
}

# ── Country display order and short labels ────────────────────
COUNTRY_ORDER = [
    "Canada",
    "Russian Federation",
    "China",
    "United States",
    "India",
    "Indonesia",
]
# Short labels used as column headers (keep ≤ 12 chars for readability)
COUNTRY_LABELS = {
    "Canada":             "Canada",
    "Russian Federation": "Russia",
    "China":              "China",
    "United States":      "US",
    "India":              "India",
    "Indonesia":          "Indonesia",
}

# ── Output directories ────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR  = os.path.join(SCRIPT_DIR, "tables")
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")

# ── Table styling ─────────────────────────────────────────────
HEADER_COLOR  = "#2C3E50"   # dark navy
AVG_ROW_COLOR = "#D5E8D4"   # light green — Model Average row
IMPROVE_COLOR = "#C8E6C9"   # light green — delta < −0.001 (improvement)
WORSE_COLOR   = "#FFCDD2"   # light red   — delta > +0.001 (degradation)
NEUTRAL_COLOR = "#FFFFFF"   # white        — delta ≈ 0
ODD_ROW_BG    = "#FFFFFF"
EVEN_ROW_BG   = "#F2F2F2"


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_all_models() -> dict[str, pd.DataFrame]:
    """
    Load one metrics CSV per model.

    Returns a dict  { model_name: DataFrame }  where each DataFrame
    contains per-country and summary (AVG) rows.
    """
    dfs = {}
    for model_name, fname in MODEL_FILES.items():
        fpath = os.path.join(MODEL_RESULT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] {model_name}: file not found at {fpath} — skipped.")
            continue
        df = pd.read_csv(fpath)
        # Standardise country column name (LSTCN uses 'Country Name' in some versions)
        if "Country Name" in df.columns and "country" not in df.columns:
            df = df.rename(columns={"Country Name": "country"})
        dfs[model_name] = df
    return dfs


def get_country_value(df: pd.DataFrame,
                       country: str,
                       feature_set: str,
                       metric: str) -> float:
    """
    Extract a single metric value for one country × feature set.
    Returns NaN if the row is not found.
    """
    mask = (df["country"] == country) & (df["feature_set"] == feature_set)
    row  = df[mask]
    if row.empty or metric not in row.columns:
        return np.nan
    return float(row.iloc[0][metric])


def get_avg_value(df: pd.DataFrame,
                   feature_set: str,
                   metric: str) -> float:
    """
    Extract the pre-computed 6-country average value from the AVG summary rows.

    Looks for a row where country contains 'BASELINE_AVG' or 'AUGMENTED_AVG'
    depending on the feature_set requested.
    """
    avg_label = "BASELINE_AVG" if feature_set == "baseline" else "AUGMENTED_AVG"
    mask = df["country"].str.contains(avg_label, na=False)
    row  = df[mask]
    if row.empty or metric not in row.columns:
        return np.nan
    return float(row.iloc[0][metric])


# ============================================================
# 3. TABLE BUILDERS
# ============================================================

def build_overview_df(model_dfs: dict) -> pd.DataFrame:
    """
    Build the overview DataFrame: 4 models × 9 metric columns.

    Columns: Baseline MAE | Augmented MAE | Δ MAE
             | Baseline RMSE | Augmented RMSE | Δ RMSE
             | Baseline MAPE | Augmented MAPE | Δ MAPE
    """
    rows = []
    for model in ["ARIMAX", "SVR", "XGBoost", "LSTCN"]:
        if model not in model_dfs:
            continue
        df = model_dfs[model]
        b_mae  = get_avg_value(df, "baseline",  "MAE_scaled_y")
        a_mae  = get_avg_value(df, "augmented", "MAE_scaled_y")
        b_rmse = get_avg_value(df, "baseline",  "RMSE_scaled_y")
        a_rmse = get_avg_value(df, "augmented", "RMSE_scaled_y")
        b_mape = get_avg_value(df, "baseline",  "MAPE")
        a_mape = get_avg_value(df, "augmented", "MAPE")
        rows.append({
            "Model":     model,
            "Base MAE":  round(b_mae,  4),
            "Aug MAE":   round(a_mae,  4),
            "ΔMAE":      round(a_mae  - b_mae,  4),
            "Base RMSE": round(b_rmse, 4),
            "Aug RMSE":  round(a_rmse, 4),
            "ΔRMSE":     round(a_rmse - b_rmse, 4),
            "Base MAPE": round(b_mape, 4),
            "Aug MAPE":  round(a_mape, 4),
            "ΔMAPE":     round(a_mape - b_mape, 4),
        })
    return pd.DataFrame(rows)


def build_detail_df(model_dfs: dict, feature_set: str, metric: str) -> pd.DataFrame:
    """
    Build a per-country detail DataFrame for one feature set × one metric.

    Rows    : model names + 'Model Average'
    Columns : short country labels + 'Average'
    """
    col_labels = [COUNTRY_LABELS[c] for c in COUNTRY_ORDER] + ["Average"]
    rows = []

    for model in ["ARIMAX", "SVR", "XGBoost", "LSTCN"]:
        if model not in model_dfs:
            continue
        df   = model_dfs[model]
        vals = [get_country_value(df, c, feature_set, metric) for c in COUNTRY_ORDER]
        avg  = get_avg_value(df, feature_set, metric)
        # Format as fixed 4-decimal strings so all cells have equal width
        row  = dict(zip(col_labels,
                        [f"{v:.4f}" for v in vals] + [f"{avg:.4f}"]))
        row["Model"] = model
        rows.append(row)

    detail_df = pd.DataFrame(rows, columns=["Model"] + col_labels)

    # Model Average row
    numeric_cols = col_labels
    avg_row = {"Model": "Model Avg"}
    for col in numeric_cols:
        vals_col = pd.to_numeric(detail_df[col], errors="coerce")
        avg_row[col] = f"{float(vals_col.mean()):.4f}"
    detail_df = pd.concat(
        [detail_df, pd.DataFrame([avg_row])], ignore_index=True
    )

    return detail_df


def build_delta_df(model_dfs: dict, metric: str) -> pd.DataFrame:
    """
    Build delta DataFrame: augmented − baseline for one metric.

    Positive value = augmented is WORSE.
    Negative value = augmented is BETTER (improvement).
    """
    col_labels = [COUNTRY_LABELS[c] for c in COUNTRY_ORDER] + ["Average"]
    rows = []

    for model in ["ARIMAX", "SVR", "XGBoost", "LSTCN"]:
        if model not in model_dfs:
            continue
        df = model_dfs[model]
        deltas = []
        for c in COUNTRY_ORDER:
            b = get_country_value(df, c, "baseline",  metric)
            a = get_country_value(df, c, "augmented", metric)
            deltas.append(f"{a - b:.4f}")
        avg_b = get_avg_value(df, "baseline",  metric)
        avg_a = get_avg_value(df, "augmented", metric)

        row = dict(zip(col_labels, deltas + [f"{avg_a - avg_b:.4f}"]))
        row["Model"] = model
        rows.append(row)

    delta_df = pd.DataFrame(rows, columns=["Model"] + col_labels)

    # Model Average row
    avg_row = {"Model": "Model Avg"}
    for col in col_labels:
        vals_col = pd.to_numeric(delta_df[col], errors="coerce")
        avg_row[col] = f"{float(vals_col.mean()):.4f}"
    delta_df = pd.concat(
        [delta_df, pd.DataFrame([avg_row])], ignore_index=True
    )

    return delta_df


# ============================================================
# 4. TABLE RENDERERS
# ============================================================

def _style_header(tbl, n_cols: int) -> None:
    """Apply dark navy styling to the header row (row index 0)."""
    for col_idx in range(n_cols):
        cell = tbl[0, col_idx]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold")


def _style_rows(tbl, n_data_rows: int, n_cols: int,
                avg_row_idx: int = None) -> None:
    """
    Apply alternating row backgrounds to data rows.
    If avg_row_idx is provided, that row gets the distinct average colour.
    """
    for row_idx in range(1, n_data_rows + 1):
        if avg_row_idx is not None and row_idx == avg_row_idx:
            bg = AVG_ROW_COLOR
            fw = "bold"
        else:
            bg = EVEN_ROW_BG if row_idx % 2 == 0 else ODD_ROW_BG
            fw = "normal"
        for col_idx in range(n_cols):
            tbl[row_idx, col_idx].set_facecolor(bg)
            tbl[row_idx, col_idx].set_text_props(fontweight=fw)


def render_overview_table(overview_df: pd.DataFrame,
                           png_path: str, csv_path: str) -> None:
    """
    Render the overview table PNG and save its CSV.

    Δ columns are tinted green (improvement) or red (degradation).
    """
    col_headers = list(overview_df.columns)
    cell_data   = overview_df.values.tolist()
    n_rows      = len(cell_data)
    n_cols      = len(col_headers)

    fig, ax = plt.subplots(figsize=(15, 0.55 * (n_rows + 2.5)))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    _style_header(tbl, n_cols)
    _style_rows(tbl, n_rows, n_cols)

    # Tint Δ columns
    delta_cols = [i for i, h in enumerate(col_headers) if h.startswith("Δ")]
    for row_idx in range(1, n_rows + 1):
        for col_idx in delta_cols:
            try:
                val = float(cell_data[row_idx - 1][col_idx])
                if val < -0.0005:
                    tbl[row_idx, col_idx].set_facecolor(IMPROVE_COLOR)
                elif val > 0.0005:
                    tbl[row_idx, col_idx].set_facecolor(WORSE_COLOR)
            except (ValueError, TypeError):
                pass

    ax.set_title(
        "Model Performance Overview — Baseline vs Augmented (6-Country Average)",
        fontsize=12, fontweight="bold",
        pad=4,        # 緊貼表格
    )
    fig.text(
        0.5, 0.04,    # 上移靠近表格底部
        "MAE / RMSE computed on scaled y (range −0.9 to 0.9)  │  "
        "Δ = Augmented − Baseline  │  Green = improvement, Red = degradation",
        ha="center", fontsize=8, color="#555555",
    )

    plt.subplots_adjust(top=0.88, bottom=0.12)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    overview_df.to_csv(csv_path, index=False)
    print(f"  Saved: {os.path.basename(png_path)}")


def render_detail_table(detail_df: pd.DataFrame,
                         title: str,
                         png_path: str, csv_path: str) -> None:
    """
    Render a per-country detail table PNG and save its CSV.

    Styling:
    • Header row       : dark navy, white bold text
    • Model name col   : uniform muted blue-grey background (no heatmap)
    • Country cells    : YlOrRd heatmap — light yellow (low/good) → dark red (high/bad)
                         normalised across all country × model cells in the table
    • Average column   : AVG_ROW_COLOR (green) — distinct from heatmap
    • Model Avg row    : AVG_ROW_COLOR (green) — distinct from heatmap
    • Text colour      : auto white on dark cells, dark on light cells
    • No alternating row stripes
    """
    col_headers  = list(detail_df.columns)
    cell_data    = detail_df.values.tolist()
    n_rows       = len(cell_data)          # includes Model Avg row
    n_cols       = len(col_headers)        # includes "Model" and "Average" cols

    # index layout:
    #   col 0         = "Model"   — uniform bg
    #   cols 1..n-2   = countries — heatmap
    #   col  n-1      = "Average" — avg green
    #   rows 1..n-1   = model rows
    #   row  n        = Model Avg — avg green throughout
    country_cols = list(range(1, n_cols - 1))
    model_rows   = list(range(1, n_rows))       # tbl row indices

    MODEL_COL_BG = "#DDE3EA"   # muted blue-grey for model name column

    # ── Normalise all country × model values ─────────────────────────────
    numeric_vals = []
    for ri in model_rows:
        for ci in country_cols:
            try:
                numeric_vals.append(float(cell_data[ri - 1][ci]))
            except (ValueError, TypeError):
                pass

    vmin = min(numeric_vals) if numeric_vals else 0.0
    vmax = max(numeric_vals) if numeric_vals else 1.0

    perf_cmap = plt.cm.YlOrRd   # light yellow = low/good, dark red = high/bad

    def _auto_text_color(rgba):
        """Return white for dark backgrounds, near-black for light ones."""
        r, g, b = rgba[0], rgba[1], rgba[2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return "white" if lum < 0.50 else "#1A1A1A"

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 0.50 * (n_rows + 1.8)))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.8)

    # 1. Header
    _style_header(tbl, n_cols)

    # 2. Reset all data cells to white (clears matplotlib defaults)
    for ri in range(1, n_rows + 1):
        for ci in range(n_cols):
            tbl[ri, ci].set_facecolor("#FFFFFF")
            tbl[ri, ci].set_text_props(color="#1A1A1A", fontweight="normal")

    # 3. Model name column: uniform muted bg
    for ri in range(1, n_rows + 1):
        tbl[ri, 0].set_facecolor(MODEL_COL_BG)

    # 4. Country value cells: YlOrRd heatmap (model rows only, not Model Avg)
    for ri in model_rows:
        for ci in country_cols:
            try:
                v     = float(cell_data[ri - 1][ci])
                norm  = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                norm  = max(0.0, min(1.0, norm))
                rgba  = perf_cmap(norm)
                tbl[ri, ci].set_facecolor(rgba)
                tbl[ri, ci].set_text_props(color=_auto_text_color(rgba))
            except (ValueError, TypeError):
                pass

    # 5. Average column → AVG_ROW_COLOR (green) for every data row
    avg_col = n_cols - 1
    for ri in range(1, n_rows + 1):
        tbl[ri, avg_col].set_facecolor(AVG_ROW_COLOR)
        tbl[ri, avg_col].set_text_props(color="#1A1A1A")

    # 6. Model Avg row → AVG_ROW_COLOR (green) for every column; bold text
    for ci in range(n_cols):
        tbl[n_rows, ci].set_facecolor(AVG_ROW_COLOR)
        tbl[n_rows, ci].set_text_props(color="#1A1A1A", fontweight="bold")

    # ── Title & footnote ─────────────────────────────────────────────────
    ax.set_title(title, fontsize=12, fontweight="bold", pad=4)
    fig.text(
        0.5, 0.01,
        "Metrics on scaled y (−0.9 to 0.9)  │  "
        "Cell colour: light yellow = lower/better → dark red = higher/worse  │  "
        "Green = row/column averages",
        ha="center", fontsize=8, color="#555555",
    )

    plt.subplots_adjust(top=0.93, bottom=0.05)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    detail_df.to_csv(csv_path, index=False)
    print(f"  Saved: {os.path.basename(png_path)}")


def render_delta_table(delta_df: pd.DataFrame,
                        metric_label: str,
                        png_path: str, csv_path: str) -> None:
    """
    Render the delta table PNG and save its CSV.

    Each cell is coloured:
      green  → augmented improved  (delta < −0.001)
      red    → augmented degraded  (delta > +0.001)
      white  → negligible change
    The Model Avg row follows standard alternating styling.
    """
    col_headers = list(delta_df.columns)
    cell_data   = delta_df.values.tolist()
    n_rows      = len(cell_data)
    n_cols      = len(col_headers)

    fig, ax = plt.subplots(figsize=(12, 0.55 * (n_rows + 2.5)))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.8)

    _style_header(tbl, n_cols)
    _style_rows(tbl, n_rows, n_cols, avg_row_idx=n_rows)

    # Colour-code each numeric cell based on delta direction
    n_model_rows = n_rows - 1   # exclude Model Avg row for colouring
    for row_idx in range(1, n_model_rows + 1):
        for col_idx in range(1, n_cols):   # skip Model column
            try:
                val = float(cell_data[row_idx - 1][col_idx])
                if val < -0.001:
                    tbl[row_idx, col_idx].set_facecolor(IMPROVE_COLOR)
                elif val > 0.001:
                    tbl[row_idx, col_idx].set_facecolor(WORSE_COLOR)
                else:
                    tbl[row_idx, col_idx].set_facecolor(NEUTRAL_COLOR)
            except (ValueError, TypeError):
                pass

    ax.set_title(
        f"Δ {metric_label} — Augmented vs Baseline (per country & model)",
        fontsize=12, fontweight="bold", pad=4,
    )
    fig.text(
        0.5, 0.04,
        "Δ = Augmented − Baseline  │  "
        "Green = improvement (Δ < −0.001)  │  Red = degradation (Δ > +0.001)",
        ha="center", fontsize=8, color="#555555",
    )

    plt.subplots_adjust(top=0.88, bottom=0.12)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    delta_df.to_csv(csv_path, index=False)
    print(f"  Saved: {os.path.basename(png_path)}")


def render_delta_heatmap(model_dfs: dict,
                          metric: str,
                          metric_label: str,
                          threshold: float,
                          png_path: str,
                          csv_path: str) -> None:
    """
    Render a delta heatmap (country × model) in the style of Sprint 3 Slide 7.

    Each cell shows:
      • the Δ value  (augmented − baseline)
      • a label: Improved / Worsened / No material change

    A 'Model Avg' column and a 'Country Avg' row are appended.
    Colour: green = improved (Δ < −threshold), red = worsened (Δ > +threshold),
            grey = no material change.
    """
    from matplotlib.patches import Patch

    models    = [m for m in ["ARIMAX", "SVR", "XGBoost", "LSTCN"] if m in model_dfs]
    countries = COUNTRY_ORDER

    # ── Build delta matrix  (country × model) ─────────────
    mat = {}
    for country in countries:
        mat[country] = {}
        for model in models:
            df = model_dfs[model]
            b  = get_country_value(df, country, "baseline",  metric)
            a  = get_country_value(df, country, "augmented", metric)
            mat[country][model] = round(a - b, 4)

    # Add Country Avg column (for each country row, average across models)
    for country in countries:
        vals = [mat[country][m] for m in models if not np.isnan(mat[country][m])]
        mat[country]["Country\nAvg"] = round(np.mean(vals), 4) if vals else np.nan

    all_cols = models + ["Country\nAvg"]

    # Add Model Avg row (for each model column, average across countries)
    mat["Model\nAvg"] = {}
    for col in all_cols:
        vals = [mat[c][col] for c in countries if not np.isnan(mat[c].get(col, np.nan))]
        mat["Model\nAvg"][col] = round(np.mean(vals), 4) if vals else np.nan

    all_rows = countries + ["Model\nAvg"]

    # ── Colour helpers ─────────────────────────────────────
    # Regular cells
    CELL_IMPROVE = "#C8E6C9"
    CELL_WORSE   = "#FFCDD2"
    CELL_NEUTRAL = "#EEEEEE"
    # Average cells — deeper shades of the same direction colours
    CELL_IMPROVE_AVG = "#66BB6A"
    CELL_WORSE_AVG   = "#EF9A9A"
    CELL_NEUTRAL_AVG = "#BDBDBD"

    def get_bg(val, is_avg):
        if pd.isna(val):
            return "white"
        improved = val < -threshold
        worsened = val > threshold
        if is_avg:
            return CELL_IMPROVE_AVG if improved else CELL_WORSE_AVG if worsened else CELL_NEUTRAL_AVG
        return CELL_IMPROVE if improved else CELL_WORSE if worsened else CELL_NEUTRAL

    def get_label(val):
        if pd.isna(val):
            return "—"
        if val < -threshold:
            return "Improved"
        if val > threshold:
            return "Worsened"
        return "No material\nchange"

    def _avg_text_color(val):
        """White text on the darker avg-cell greens; dark on lighter shades."""
        if pd.isna(val):
            return "#1A1A1A"
        improved = val < -threshold
        # CELL_IMPROVE_AVG (#66BB6A) is medium-dark → white reads better
        return "white" if improved else "#1A1A1A"

    # ── Figure setup ───────────────────────────────────────
    n_rows    = len(all_rows)
    n_cols    = len(all_cols)
    cell_w    = 1.75   # inches per column
    cell_h    = 0.85   # inches per row
    fig_w     = cell_w * n_cols + 1.5
    fig_h     = cell_h * n_rows + 1.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("auto")
    ax.axis("off")

    fig.subplots_adjust(left=0.18, right=0.97, top=0.82, bottom=0.10)

    # ── Draw cells ─────────────────────────────────────────
    for ri, row_name in enumerate(all_rows):
        y          = n_rows - ri - 1
        is_avg_row = (row_name == "Model\nAvg")
        for ci, col_name in enumerate(all_cols):
            val     = mat[row_name].get(col_name, np.nan)
            is_avg  = is_avg_row or (col_name == "Country\nAvg")
            bg      = get_bg(val, is_avg)
            lbl     = get_label(val)
            val_str = f"{val:+.4f}" if not pd.isna(val) else "—"
            bold    = "bold" if is_avg else "normal"
            txt_col = _avg_text_color(val) if is_avg else "#1A1A1A"
            lbl_col = "#333333" if is_avg else "#444444"

            # Background rectangle
            rect = plt.Rectangle(
                (ci, y), 1, 1,
                facecolor=bg, edgecolor="white", linewidth=1.5,
                transform=ax.transData, clip_on=False,
            )
            ax.add_patch(rect)

            # Delta value (upper text)
            ax.text(
                ci + 0.5, y + 0.60, val_str,
                ha="center", va="center",
                fontsize=9.5, fontweight=bold, color=txt_col,
                transform=ax.transData,
            )
            # Label (lower text)
            ax.text(
                ci + 0.5, y + 0.25, lbl,
                ha="center", va="center",
                fontsize=7, color=lbl_col,
                transform=ax.transData,
            )

    # ── Bold border overlays for avg column and avg row ────
    AVG_BORDER_COLOR = "#2C3E50"
    AVG_BORDER_LW    = 2.8

    # Avg column (Country Avg = last column, x = n_cols-1)
    ax.add_patch(plt.Rectangle(
        (n_cols - 1, 0), 1, n_rows,
        facecolor="none", edgecolor=AVG_BORDER_COLOR, linewidth=AVG_BORDER_LW,
        transform=ax.transData, clip_on=False, zorder=5,
    ))
    # Avg row (Model Avg = bottom row, y = 0)
    ax.add_patch(plt.Rectangle(
        (0, 0), n_cols, 1,
        facecolor="none", edgecolor=AVG_BORDER_COLOR, linewidth=AVG_BORDER_LW,
        transform=ax.transData, clip_on=False, zorder=5,
    ))

    # ── Column headers ─────────────────────────────────────
    for ci, col_name in enumerate(all_cols):
        bg = "#405A6E" if col_name == "Country\nAvg" else HEADER_COLOR
        ax.text(
            ci + 0.5, n_rows + 0.08, col_name,
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="white",
            transform=ax.transData,
            bbox=dict(facecolor=bg, edgecolor="none",
                      boxstyle="round,pad=0.28", alpha=0.95),
        )

    # ── Row labels ─────────────────────────────────────────
    for ri, row_name in enumerate(all_rows):
        y = n_rows - ri - 1
        fw = "bold" if row_name == "Model\nAvg" else "normal"
        ax.text(
            -0.07, y + 0.5, row_name,
            ha="right", va="center",
            fontsize=9.5, fontweight=fw, color=HEADER_COLOR,
            transform=ax.transData,
        )

    # ── Axis labels ────────────────────────────────────────
    ax.text(
        n_cols / 2, -0.55, "Model",
        ha="center", va="top", fontsize=10, color="#333333",
        transform=ax.transData,
    )
    ax.text(
        -0.75, n_rows / 2, "Country",
        ha="center", va="center", fontsize=10, color="#333333",
        rotation=90, transform=ax.transData,
    )

    # ── Legend ─────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor=CELL_IMPROVE, edgecolor="#aaa", label="Improved"),
        Patch(facecolor=CELL_NEUTRAL, edgecolor="#aaa", label="No material change"),
        Patch(facecolor=CELL_WORSE,   edgecolor="#aaa", label="Worsened"),
        Patch(facecolor=CELL_NEUTRAL_AVG, edgecolor=AVG_BORDER_COLOR,
              linewidth=2, label="Avg row / column\n(deeper shade + bold border)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.25),
        fontsize=8.5,
        title=f"Threshold = ±{threshold}",
        title_fontsize=8,
        framealpha=0.9,
    )

    # ── Title & footnote ───────────────────────────────────
    fig.suptitle(
        f"Temperature Impact Heatmap ({metric_label})\n"
        f"Δ = Augmented − Baseline   │   Threshold = ±{threshold}",
        fontsize=12, fontweight="bold", y=0.96,
    )
    fig.text(
        0.5, 0.02,
        f"Green = Improved (Δ < −{threshold})   │   "
        f"Red = Worsened (Δ > +{threshold})   │   "
        "Grey = No material change",
        ha="center", fontsize=8, color="#555555",
    )

    # ── Save ───────────────────────────────────────────────
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save raw CSV
    rows_data = []
    for row_name in all_rows:
        row = {"Country": row_name}
        row.update({col: mat[row_name].get(col, np.nan) for col in all_cols})
        rows_data.append(row)
    pd.DataFrame(rows_data).to_csv(csv_path, index=False)
    print(f"  Saved: {os.path.basename(png_path)}")


def build_best_model_df(model_dfs: dict) -> pd.DataFrame:
    """
    Build a table identifying the best-performing model per country,
    separately for baseline and augmented feature sets.

    Tie-breaking rule: lowest MAE first; if tied (within 0.0001), use RMSE.

    Columns
    -------
    Country | Best Baseline Model | Baseline MAE | Baseline RMSE
            | Best Augmented Model | Augmented MAE | Augmented RMSE
            | Model Changed?
    """
    models = [m for m in ["ARIMAX", "SVR", "XGBoost", "LSTCN"] if m in model_dfs]
    rows   = []

    for country in COUNTRY_ORDER:
        row = {"Country": country}

        for fset, prefix in [("baseline", "Base"), ("augmented", "Aug")]:
            best_model = None
            best_mae   = np.inf
            best_rmse  = np.inf

            for model in models:
                df   = model_dfs[model]
                mae  = get_country_value(df, country, fset, "MAE_scaled_y")
                rmse = get_country_value(df, country, fset, "RMSE_scaled_y")
                if pd.isna(mae):
                    continue
                # Tie-break: MAE first, then RMSE
                if (mae < best_mae - 0.0001) or \
                   (abs(mae - best_mae) <= 0.0001 and rmse < best_rmse):
                    best_model = model
                    best_mae   = mae
                    best_rmse  = rmse

            row[f"{prefix} Best Model"] = best_model if best_model else "—"
            row[f"{prefix} MAE"]        = f"{best_mae:.4f}"  if best_model else "—"
            row[f"{prefix} RMSE"]       = f"{best_rmse:.4f}" if best_model else "—"

        # Flag whether the best model changed between feature sets
        row["Model Changed?"] = (
            "Yes" if row["Base Best Model"] != row["Aug Best Model"] else "No"
        )
        rows.append(row)

    return pd.DataFrame(rows)


def render_best_model_table(best_df: pd.DataFrame,
                             png_path: str, csv_path: str) -> None:
    """
    Render the best-model-per-country table as a styled PNG.

    Rows    : 6 countries
    Columns : Country | Best Baseline Model | Base MAE | Base RMSE
                      | Best Augmented Model | Aug MAE | Aug RMSE
                      | Model Changed?

    Styling
    -------
    • Best model name cells are tinted blue so they stand out
    • "Model Changed? = Yes" cells are tinted gold to highlight shifts
    • "Model Changed? = No"  cells are plain white
    """
    col_headers = list(best_df.columns)
    cell_data   = best_df.values.tolist()
    n_rows      = len(cell_data)
    n_cols      = len(col_headers)

    fig, ax = plt.subplots(figsize=(14, 0.55 * (n_rows + 2.5)))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.9)

    _style_header(tbl, n_cols)

    # Column index lookup
    col_idx_map = {h: i for i, h in enumerate(col_headers)}
    base_model_col = col_idx_map.get("Base Best Model", -1)
    aug_model_col  = col_idx_map.get("Aug Best Model",  -1)
    changed_col    = col_idx_map.get("Model Changed?",  -1)

    MODEL_CELL_COLOR  = "#D0E4F7"   # light blue  — best model name
    CHANGED_YES_COLOR = "#FFF2CC"   # light gold  — model switched
    CHANGED_NO_COLOR  = "#FFFFFF"   # white       — model stayed same

    for row_idx in range(1, n_rows + 1):
        bg = EVEN_ROW_BG if row_idx % 2 == 0 else ODD_ROW_BG
        for col_idx in range(n_cols):
            tbl[row_idx, col_idx].set_facecolor(bg)

        # Highlight best model name cells
        if base_model_col >= 0:
            tbl[row_idx, base_model_col].set_facecolor(MODEL_CELL_COLOR)
            tbl[row_idx, base_model_col].set_text_props(fontweight="bold")
        if aug_model_col >= 0:
            tbl[row_idx, aug_model_col].set_facecolor(MODEL_CELL_COLOR)
            tbl[row_idx, aug_model_col].set_text_props(fontweight="bold")

        # Highlight "Model Changed?" column
        if changed_col >= 0:
            changed_val = cell_data[row_idx - 1][changed_col]
            color = CHANGED_YES_COLOR if changed_val == "Yes" else CHANGED_NO_COLOR
            tbl[row_idx, changed_col].set_facecolor(color)

    ax.set_title(
        "Best-Performing Model per Country — Baseline vs Augmented (by MAE)",
        fontsize=12, fontweight="bold", pad=4,
    )
    fig.text(
        0.5, 0.04,
        "Tie-breaking: lowest MAE first; if tied (within 0.0001), use RMSE  │  "
        "Metrics on scaled y (range −0.9 to 0.9)",
        ha="center", fontsize=8, color="#555555",
    )

    plt.subplots_adjust(top=0.88, bottom=0.12)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    best_df.to_csv(csv_path, index=False)
    print(f"  Saved: {os.path.basename(png_path)}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("model_performance_tables.py")
    print("=" * 60)

    # ── Create output directories ──────────────────────────
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    # ── Load all model CSVs ────────────────────────────────
    print("\nLoading model metrics CSVs …")
    model_dfs = load_all_models()
    if not model_dfs:
        raise RuntimeError("No model CSVs loaded. Check MODEL_RESULT_DIR and file names.")
    print(f"  Loaded: {list(model_dfs.keys())}")

    # ── Metric configuration ───────────────────────────────
    # (internal column name, display label, heatmap threshold)
    METRICS = [
        ("MAE_scaled_y",  "MAE (scaled y)", 0.01),
        ("RMSE_scaled_y", "RMSE (scaled y)", 0.01),
        ("MAPE",          "MAPE",            0.005),
    ]

    # ── 1. Overview table (one table covering all metrics) ─
    print("\n[1] Overview table …")
    overview_df = build_overview_df(model_dfs)
    render_overview_table(
        overview_df,
        png_path=os.path.join(TABLES_DIR, "overview_table.png"),
        csv_path=os.path.join(DATA_DIR,   "overview_table.csv"),
    )

    # ── 2. Baseline detail tables — one PNG per metric ─────
    print("\n[2] Baseline detail tables (MAE / RMSE / MAPE) …")
    for metric, label, _ in METRICS:
        short = metric.replace("_scaled_y", "")
        df = build_detail_df(model_dfs, "baseline", metric)
        render_detail_table(
            df,
            title=f"Baseline Model Performance — {label}",
            png_path=os.path.join(TABLES_DIR, f"baseline_{short}_table.png"),
            csv_path=os.path.join(DATA_DIR,   f"baseline_{short}_table.csv"),
        )

    # ── 3. Augmented detail tables — one PNG per metric ────
    print("\n[3] Augmented detail tables (MAE / RMSE / MAPE) …")
    for metric, label, _ in METRICS:
        short = metric.replace("_scaled_y", "")
        df = build_detail_df(model_dfs, "augmented", metric)
        render_detail_table(
            df,
            title=f"Augmented Model Performance — {label}",
            png_path=os.path.join(TABLES_DIR, f"augmented_{short}_table.png"),
            csv_path=os.path.join(DATA_DIR,   f"augmented_{short}_table.csv"),
        )

    # ── 4. Delta tables — one PNG per metric ───────────────
    print("\n[4] Delta tables (MAE / RMSE / MAPE) …")
    for metric, label, _ in METRICS:
        short = metric.replace("_scaled_y", "")
        df = build_delta_df(model_dfs, metric)
        render_delta_table(
            df,
            metric_label=label,
            png_path=os.path.join(TABLES_DIR, f"delta_{short}_table.png"),
            csv_path=os.path.join(DATA_DIR,   f"delta_{short}_table.csv"),
        )

    # ── 5. Delta heatmaps — one PNG per metric ─────────────
    print("\n[5] Delta heatmaps (MAE / RMSE / MAPE) …")
    for metric, label, threshold in METRICS:
        short = metric.replace("_scaled_y", "")
        render_delta_heatmap(
            model_dfs,
            metric=metric,
            metric_label=label,
            threshold=threshold,
            png_path=os.path.join(TABLES_DIR, f"delta_heatmap_{short}.png"),
            csv_path=os.path.join(DATA_DIR,   f"delta_heatmap_{short}.csv"),
        )

    # ── 6. Best model per country table ───────────────────
    print("\n[6] Best model per country table …")
    best_df = build_best_model_df(model_dfs)
    render_best_model_table(
        best_df,
        png_path=os.path.join(TABLES_DIR, "best_model_per_country.png"),
        csv_path=os.path.join(DATA_DIR,   "best_model_per_country.csv"),
    )

    # ── Summary ────────────────────────────────────────────
    n_pngs = 1 + 3 + 3 + 3 + 3 + 1   # overview + baseline + augmented + delta + heatmap + best
    print(f"\n✅  All done.  {n_pngs} PNGs generated.")
    print(f"\n  tables/  →  {TABLES_DIR}")
    print(f"  data/    →  {DATA_DIR}")
    print("\n  File list:")
    print("    overview_table.png")
    for prefix in ("baseline", "augmented", "delta"):
        for short in ("MAE", "RMSE", "MAPE"):
            print(f"    {prefix}_{short}_table.png")
    for short in ("MAE", "RMSE", "MAPE"):
        print(f"    delta_heatmap_{short}.png")
    print("    best_model_per_country.png")


if __name__ == "__main__":
    main()
