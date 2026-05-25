"""
SVR_train_val_test_metrics_regularized.py
==========================================
PURPOSE
-------
Re-evaluate the regularized SVR model's performance on the training,
validation (CV out-of-fold), and test sets for every country × feature set
combination, using the best hyperparameters stored in
svr_regularized_metrics_summary.csv.

This script does NOT re-run hyperparameter search. It takes the already-
selected best parameters, re-fits the model following the same
expanding-window protocol as SVR_rolling_cv.py, and produces:

  1. A CSV with Train / Val (CV OOF) / Test metrics per country.
  2. Eight table figures (baseline + augmented × 4 metrics).

The table figures include:
  • A  Gap column  (= Test MAE − Train MAE) to quantify overfitting.
  • An Average row at the bottom to give a 6-country summary.

INPUT FILES
-----------
  ../../data/dataset_recursive_3yr_avg_drop_Industry.csv
  ../../model result/svr_regularized_metrics_summary.csv

OUTPUTS  (written to the same folder as this script)
-------
  svr_regularized_train_val_test_comparison.csv
  svr_regularized_table_MAE_baseline.png
  svr_regularized_table_MAE_augmented.png
  svr_regularized_table_RMSE_baseline.png
  svr_regularized_table_RMSE_augmented.png
  svr_regularized_table_MAPE_baseline.png
  svr_regularized_table_MAPE_augmented.png
  svr_regularized_table_R_squared_baseline.png
  svr_regularized_table_R_squared_augmented.png

PIPELINE POSITION
-----------------
  SVR_regularized.py  →  svr_regularized_metrics_summary.csv
                                    ↓
                         [THIS SCRIPT]
                                    ↓
                         SHAP_SVR_augmented_only.py
                         (uses the Train/Val/Test CSV for comparison charts)

DEPENDENCIES
------------
  pandas, numpy, matplotlib, scikit-learn
  Install: pip install pandas numpy matplotlib scikit-learn
"""

import warnings
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.svm import SVR

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


# ============================================================
# 1. CONFIGURATION
# ============================================================
# This script lives in:
#   result interpretation/metrics on train val test set/
# Paths go up two levels to reach the repo root.

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed", "dataset_recursive_3yr_avg_drop_Industry.csv")
METRICS_PATH = os.path.join(SCRIPT_DIR, "..", "..", "results", "metrics", "svr_regularized_metrics_summary.csv")
OUTPUT_CSV   = os.path.join(SCRIPT_DIR, "outputs", "svr_regularized_train_val_test_comparison.csv")

TARGET          = "Total CO2 emissions"
LAG_TARGET_NAME = "CO2_emissions_lag1"
TRAIN_END_YEAR  = 2016
TEST_START_YEAR = 2017
SCALE_RANGE     = (-0.9, 0.9)

# Country display order — consistent across all result scripts
COUNTRY_ORDER = [
    "Canada",
    "Russian Federation",
    "China",
    "United States",
    "India",
    "Indonesia",
]

BASE_FEATURES = [
    "Population",
    "GDP",
    "Electric power consumption",
    "Fossil fuel energy consumption",
    "Renewable energy consumption",
    "Fertilizer consumption",
]
TEMP_FEATURES = [
    "Temperature annual mean",
    "Temperature std across months",
    "Number of frost days",
    "Number of hot days",
]
FEATURE_SETS = {
    "baseline":  BASE_FEATURES + [LAG_TARGET_NAME],
    "augmented": BASE_FEATURES + TEMP_FEATURES + [LAG_TARGET_NAME],
}

# Metric display labels used in table column headers and figure titles
METRIC_LABELS = {
    "MAE_scaled_y":  "MAE (scaled)",
    "RMSE_scaled_y": "RMSE (scaled)",
    "MAPE":          "MAPE",
    "R_squared":     "R²",
}

# Table styling
HEADER_COLOR  = "#2C3E50"   # dark navy header background
AVG_ROW_COLOR = "#D5E8D4"   # light green for the Average row
GAP_COL_ODD   = "#FFFBF2"   # subtle gold tint for Gap column (odd rows)
GAP_COL_EVEN  = "#FEF9EC"   # subtle gold tint for Gap column (even rows)


# ============================================================
# 2. HELPERS
# ============================================================

def parse_gamma(raw):
    """
    Safely parse the gamma hyperparameter.

    Handles: NaN → None, string 'scale'/'auto', float strings like '0.1'.
    """
    if pd.isna(raw):
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return str(raw)


def parse_degree(raw):
    """Safely parse the degree hyperparameter (poly kernel only)."""
    if pd.isna(raw):
        return None
    return int(raw)


def build_svr(kernel, C, epsilon, gamma=None, degree=None):
    """Construct an SVR instance from the given hyperparameters."""
    params = {"kernel": kernel, "C": C, "epsilon": epsilon}
    if kernel in {"rbf", "poly"}:
        params["gamma"] = gamma
    if kernel == "poly":
        params["degree"] = degree
    return SVR(**params)


def prepare_data(data: pd.DataFrame, country: str, feature_list: list) -> pd.DataFrame:
    """
    Extract and prepare data for one country.

    Steps:
      1. Filter to the specified country, sorted by Year.
      2. Create CO2_emissions_lag1 (previous year's CO2).
      3. Select Year + target + features, then drop any NaN rows.
    """
    cdf = data[data["Country Name"] == country].sort_values("Year").copy()
    cdf[LAG_TARGET_NAME] = cdf[TARGET].shift(1)
    cols = ["Year", TARGET] + feature_list
    return cdf[cols].dropna().copy()


# ============================================================
# 3. THREE-SPLIT METRIC COMPUTATION
# ============================================================

def compute_three_split_metrics(data: pd.DataFrame,
                                 country: str,
                                 feature_list: list,
                                 best_params: dict) -> dict:
    """
    Compute Train, Validation (CV OOF), and Test metrics for one
    country × feature set using the supplied best hyperparameters.

    Train split
    -----------
    Fit the model on the entire training set; predict on the same
    training set.  This is in-sample performance.

    Validation split  (CV out-of-fold)
    -----------------------------------
    Use TimeSeriesSplit to produce out-of-fold predictions across the
    training set.  Each fold trains on past data, predicts on the next
    block — mirrors the expanding-window approach used in rolling CV.
    This gives an unbiased estimate of generalisation within the training
    period.

    Test split  (expanding window, same as SVR_rolling_cv.py)
    ---------------------------------------------------------
    At each test time-step:
      1. Fit a fresh model on all available history (train + previous test steps).
      2. Predict the next time-step.
      3. Append the TRUE observation to history before the next step.
    This exactly replicates the evaluation protocol of SVR_rolling_cv.py.

    Returns a flat dict with keys: train_*/val_*/test_*  ×  4 metrics.
    """
    df_use = prepare_data(data, country, feature_list)

    sc_x = MinMaxScaler(feature_range=SCALE_RANGE)
    sc_y = MinMaxScaler(feature_range=SCALE_RANGE)

    df_scaled                    = df_use.copy()
    df_scaled[feature_list]      = sc_x.fit_transform(df_use[feature_list].values)
    df_scaled[[TARGET]]          = sc_y.fit_transform(df_use[[TARGET]].values)

    train_df      = df_scaled[df_scaled["Year"] <= TRAIN_END_YEAR].copy()
    test_df       = df_scaled[df_scaled["Year"] >= TEST_START_YEAR].copy()

    X_train       = train_df[feature_list].values
    y_train       = train_df[TARGET].values.reshape(-1)
    X_test        = test_df[feature_list].values
    y_test_scaled = test_df[TARGET].values.reshape(-1)

    kernel  = best_params["best_kernel"]
    C       = float(best_params["best_C"])
    epsilon = float(best_params["best_epsilon"])
    gamma   = parse_gamma(best_params["best_gamma"])
    degree  = parse_degree(best_params["best_degree"])

    def _metrics(y_true, y_pred, prefix):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        return {
            f"{prefix}MAE_scaled_y":  mean_absolute_error(y_true, y_pred),
            f"{prefix}RMSE_scaled_y": np.sqrt(mean_squared_error(y_true, y_pred)),
            f"{prefix}MAPE":          mean_absolute_percentage_error(y_true, y_pred),
            f"{prefix}R_squared":     r2_score(y_true, y_pred),
        }

    # ── Train ──────────────────────────────────────────────────────────
    model_train = build_svr(kernel, C, epsilon, gamma, degree)
    model_train.fit(X_train, y_train)
    train_metrics = _metrics(y_train, model_train.predict(X_train), "train_")

    # ── Validation (CV OOF) ────────────────────────────────────────────
    n_splits = min(5, max(2, len(X_train) // 4))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_true, oof_pred = [], []
    for tr_idx, val_idx in tscv.split(X_train):
        m = build_svr(kernel, C, epsilon, gamma, degree)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        oof_pred.extend(m.predict(X_train[val_idx]).tolist())
        oof_true.extend(y_train[val_idx].tolist())
    val_metrics = _metrics(oof_true, oof_pred, "val_")

    # ── Test (expanding window) ────────────────────────────────────────
    history_X, history_y = X_train.copy(), y_train.copy()
    preds_scaled = []
    for i in range(len(X_test)):
        m = build_svr(kernel, C, epsilon, gamma, degree)
        m.fit(history_X, history_y)
        preds_scaled.append(float(m.predict(X_test[i].reshape(1, -1))[0]))
        history_X = np.vstack([history_X, X_test[i]])
        history_y = np.append(history_y, y_test_scaled[i])
    test_metrics = _metrics(y_test_scaled, preds_scaled, "test_")

    return {**train_metrics, **val_metrics, **test_metrics}


# ============================================================
# 4. TABLE FIGURE GENERATOR
# ============================================================

def make_table_figure(result_df: pd.DataFrame,
                       feature_set: str,
                       metric: str,
                       output_path: str) -> None:
    """
    Render a styled matplotlib table for one feature set × one metric
    and save it as a high-resolution PNG.

    Table structure
    ---------------
    Rows    : one per country, ordered by COUNTRY_ORDER; final row = Average
    Columns : Country | Train | Val (CV) | Test | Gap (Test − Train)

    Styling
    -------
    • Dark navy header row with white bold text
    • Alternating white / light-grey row backgrounds
    • Gap column highlighted in subtle gold to draw the reader's eye
    • Average row highlighted in light green
    • Small footnote explaining Gap and the scaling range

    Parameters
    ----------
    result_df   : DataFrame produced by main() — one row per country
    feature_set : 'baseline' or 'augmented'
    metric      : key in METRIC_LABELS (e.g. 'MAE_scaled_y')
    output_path : full path for the PNG file
    """
    sub = result_df[result_df["feature_set"] == feature_set].copy()
    sub["country"] = pd.Categorical(sub["country"],
                                     categories=COUNTRY_ORDER, ordered=True)
    sub = sub.sort_values("country").reset_index(drop=True)

    col_label  = METRIC_LABELS[metric]
    train_vals = sub[f"train_{metric}"].round(4).values
    val_vals   = sub[f"val_{metric}"].round(4).values
    test_vals  = sub[f"test_{metric}"].round(4).values
    gap_vals   = (sub[f"test_{metric}"] - sub[f"train_{metric}"]).round(4).values

    # Build data rows (countries)
    rows = []
    for i in range(len(sub)):
        rows.append([
            sub.loc[i, "country"],
            train_vals[i],
            val_vals[i],
            test_vals[i],
            gap_vals[i],
        ])

    # Average row
    avg_row = [
        "Average",
        round(float(np.mean(train_vals)), 4),
        round(float(np.mean(val_vals)),   4),
        round(float(np.mean(test_vals)),  4),
        round(float(np.mean(gap_vals)),   4),
    ]
    rows.append(avg_row)

    col_headers = [
        "Country",
        f"Train {col_label}",
        f"Val {col_label} (CV)",
        f"Test {col_label}",
        "Gap (Test−Train)",
    ]

    n_rows = len(rows)   # includes Average row
    n_cols = len(col_headers)
    n_countries = n_rows - 1  # excludes Average row

    # ── Figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 0.55 * (n_rows + 2)))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)

    # ── Header row ────────────────────────────────────────────────────
    for col_idx in range(n_cols):
        cell = tbl[0, col_idx]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold")

    # ── Country rows ──────────────────────────────────────────────────
    for row_idx in range(1, n_countries + 1):
        bg = "#F2F2F2" if row_idx % 2 == 0 else "white"
        for col_idx in range(n_cols):
            tbl[row_idx, col_idx].set_facecolor(bg)
        # Highlight the Gap column (last column) in subtle gold
        tbl[row_idx, n_cols - 1].set_facecolor(
            GAP_COL_EVEN if row_idx % 2 == 0 else GAP_COL_ODD
        )

    # ── Average row ───────────────────────────────────────────────────
    for col_idx in range(n_cols):
        tbl[n_rows, col_idx].set_facecolor(AVG_ROW_COLOR)
        tbl[n_rows, col_idx].set_text_props(fontweight="bold")

    # ── Title (small pad so title sits close to the table) ────────────
    feature_label = "Baseline" if feature_set == "baseline" else "Augmented"
    ax.set_title(
        f"SVR Regularized – {feature_label}: Train / Val (CV) / Test {col_label}",
        fontsize=13,
        fontweight="bold",
        pad=6,          # was 14 — reduced to close the gap between title and table
    )

    # ── Footnote ──────────────────────────────────────────────────────
    fig.text(
        0.5, 0.005,
        "Gap = Test − Train  │  All metrics computed on scaled y  (range −0.9 to 0.9)",
        ha="center", fontsize=8, color="#555555",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("SVR_train_val_test_metrics_regularized.py")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────
    data       = pd.read_csv(DATA_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)

    # Remove summary average rows — keep only per-country rows
    country_rows = metrics_df[~metrics_df["country"].str.contains("AVG")].copy()

    rows = []
    for _, row in country_rows.iterrows():
        country     = row["country"]
        feature_set = row["feature_set"]
        print(f"\n  Processing: {country} | {feature_set}")

        best_params = {
            "best_kernel":  row["best_kernel"],
            "best_C":       row["best_C"],
            "best_epsilon": row["best_epsilon"],
            "best_gamma":   row["best_gamma"],
            "best_degree":  row["best_degree"],
        }

        try:
            metrics = compute_three_split_metrics(
                data, country, FEATURE_SETS[feature_set], best_params
            )
            rows.append({
                "country":     country,
                "feature_set": feature_set,
                # Keep best params in the CSV for traceability
                **{k: row[k] for k in ["best_kernel", "best_C", "best_epsilon",
                                        "best_gamma", "best_degree"]},
                **metrics,
            })
        except Exception as e:
            print(f"    ERROR ({country} | {feature_set}): {e}")

    # ── Save CSV ──────────────────────────────────────────────────────
    result_df = pd.DataFrame(rows)

    col_order = [
        "country", "feature_set",
        "best_kernel", "best_C", "best_epsilon", "best_gamma", "best_degree",
        "train_MAE_scaled_y",  "val_MAE_scaled_y",  "test_MAE_scaled_y",
        "train_RMSE_scaled_y", "val_RMSE_scaled_y", "test_RMSE_scaled_y",
        "train_MAPE",          "val_MAPE",          "test_MAPE",
        "train_R_squared",     "val_R_squared",     "test_R_squared",
    ]
    result_df = result_df[col_order]
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CSV: {OUTPUT_CSV}")

    # ── Generate table figures ────────────────────────────────────────
    # baseline + augmented × 4 metrics = 8 figures
    print("\nGenerating table figures …")
    for feature_set in ["baseline", "augmented"]:
        for metric in ["MAE_scaled_y", "RMSE_scaled_y", "MAPE", "R_squared"]:
            short = metric.replace("_scaled_y", "")
            fname = f"svr_regularized_table_{short}_{feature_set}.png"
            make_table_figure(result_df, feature_set, metric, fname)

    print("\n✅  Done.")
    print(f"   CSV  : {OUTPUT_CSV}")
    print(f"   PNGs : svr_regularized_table_*.png  (8 files)")


if __name__ == "__main__":
    main()
