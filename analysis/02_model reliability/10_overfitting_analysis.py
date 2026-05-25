"""
overfitting_analysis.py
========================
PURPOSE
-------
Re-evaluates each model's best hyperparameter configuration on the
training, validation (CV OOF), and test sets to quantify overfitting.

Pipeline is kept strictly consistent with the original training scripts:
  • Scaling      : MinMaxScaler(-0.9, 0.9), fit on full usable data before split
  • Train split  : Year ≤ 2016
  • Test split   : Year ≥ 2017
  • Val method   : TimeSeriesSplit OOF, one-step-ahead rolling within each fold
                   (model refit at every step, true observation appended to history)
  • Test method  : Expanding-window one-step-ahead (same as original model scripts)
  • Random seed  : 42
  • Lag feature  : CO2_emissions_lag1 added for SVR and XGBoost only

OUTPUTS
-------
  outputs/overfitting/
    MAE/
      ARIMAX_overfitting_baseline.png   ← main metric tables (8 total)
      ...
      all_models_gap_summary.png        ← avg gap bar chart
      gap_distribution_boxplot.png      ← 6-country gap boxplot
    RMSE/
      ...                               ← same format, RMSE metric
    MAPE/
      ...                               ← same format, MAPE metric
    data/
      summary/    ← per-model/feature-set aggregated CSV
      folds/      ← per-country per-fold train+val details

DEPENDENCIES
------------
  pandas, numpy, matplotlib, scikit-learn, statsmodels, xgboost, lstcn
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from lstcn.LSTCN import LSTCN

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed",
                             "dataset_recursive_3yr_avg_drop_Industry.csv")
MODEL_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "results", "metrics")
OUT_BASE     = os.path.join(SCRIPT_DIR, "outputs", "overfitting")

METRICS_FILES = {
    "ARIMAX":   "arimax_metrics_summary.csv",
    "SVR":      "svr_metrics_summary.csv",
    "XGBoost":  "xgboost_metrics_summary.csv",
    "LSTCN":    "lstcn_metrics_summary.csv",
}

RANDOM_SEED     = 42
TRAIN_END_YEAR  = 2016
TEST_START_YEAR = 2017
SCALE_RANGE     = (-0.9, 0.9)
TARGET          = "Total CO2 emissions"
LAG_COL         = "CO2_emissions_lag1"

np.random.seed(RANDOM_SEED)

COUNTRY_ORDER = [
    "Canada", "Russian Federation", "China",
    "United States", "India", "Indonesia",
]

BASE_FEATURES = [
    "Population", "GDP", "Electric power consumption",
    "Fossil fuel energy consumption", "Renewable energy consumption",
    "Fertilizer consumption",
]
TEMP_FEATURES = [
    "Temperature annual mean", "Temperature std across months",
    "Number of frost days", "Number of hot days",
]

# Table styling
HEADER_COLOR  = "#2C3E50"
AVG_ROW_COLOR = "#D5E8D4"
GAP_COL_ODD   = "#FFFBF2"
GAP_COL_EVEN  = "#FEF9EC"

# Metric display config: (internal_key, display_label, unit_note)
METRIC_CONFIG = [
    ("MAE",  "MAE",  "scaled y"),
    ("RMSE", "RMSE", "scaled y"),
    ("MAPE", "MAPE", "raw scale"),
]


# ============================================================
# 2. SHARED HELPERS
# ============================================================

def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_metrics(model_name: str) -> pd.DataFrame:
    path = os.path.join(MODEL_DIR, METRICS_FILES[model_name])
    df   = pd.read_csv(path)
    return df[~df["country"].str.contains("AVG", na=False)].copy()


def get_feature_list(model_name: str, feature_set: str) -> list:
    base = BASE_FEATURES + (TEMP_FEATURES if feature_set == "augmented" else [])
    return base + [LAG_COL] if model_name in ("SVR", "XGBoost") else base


def prepare_country_data(data, country, feature_list, add_lag=False):
    cdf = data[data["Country Name"] == country].sort_values("Year").copy()
    if add_lag:
        cdf[LAG_COL] = cdf[TARGET].shift(1)
    return cdf[["Year", TARGET] + feature_list].dropna().copy()


def scale_and_split(df, feature_list):
    sc_x = MinMaxScaler(feature_range=SCALE_RANGE)
    sc_y = MinMaxScaler(feature_range=SCALE_RANGE)
    df_s = df.copy()
    df_s[feature_list] = sc_x.fit_transform(df[feature_list].values)
    df_s[[TARGET]]     = sc_y.fit_transform(df[[TARGET]].values)
    train = df_s[df_s["Year"] <= TRAIN_END_YEAR].copy()
    test  = df_s[df_s["Year"] >= TEST_START_YEAR].copy()
    return train, test, sc_y


def compute_n_splits(n_train: int) -> int:
    return min(5, max(2, n_train // 4))


def _all_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, MAPE from arrays."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


# ============================================================
# 3. MODEL-SPECIFIC EVALUATORS
# ============================================================
# Each returns:
#   {
#     "train_MAE/RMSE/MAPE"  : float,
#     "val_MAE/RMSE/MAPE"    : float  (mean across folds),
#     "val_MAE/RMSE/MAPE_std": float  (std across folds),
#     "test_MAE/RMSE/MAPE"   : float,
#     "fold_rows"            : list of dicts  (one per fold)
#   }

# ── 3a. ARIMAX ───────────────────────────────────────────────

def _fit_arimax(y_train, X_train, p, d, q, trend):
    try:
        m = SARIMAX(endog=y_train, exog=X_train, order=(p, d, q),
                    seasonal_order=(0,0,0,0), trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
        return m.fit(disp=False, maxiter=200)
    except Exception:
        return None


def evaluate_arimax(data, country, feature_list, params):
    p, d, q = int(params["best_p"]), int(params["best_d"]), int(params["best_q"])
    trend   = str(params["best_trend"])

    df_use = prepare_country_data(data, country, feature_list, add_lag=False)
    if len(df_use) == 0:
        return None
    train, test, _ = scale_and_split(df_use, feature_list)
    X_tr, y_tr = train[feature_list].values, train[TARGET].values
    X_te, y_te = test[feature_list].values,  test[TARGET].values

    # Train
    res = _fit_arimax(y_tr, X_tr, p, d, q, trend)
    if res is None:
        return None
    train_m = _all_metrics(y_tr, np.asarray(res.fittedvalues).reshape(-1))

    # Val (OOF rolling)
    n_splits = compute_n_splits(len(X_tr))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    for fi, (tri, vali) in enumerate(tscv.split(X_tr)):
        h_X, h_y = list(X_tr[tri]), list(y_tr[tri])
        oof_t, oof_p = [], []
        fold_ok = True
        for j in range(len(vali)):
            r = _fit_arimax(np.array(h_y), np.array(h_X), p, d, q, trend)
            if r is None:
                fold_ok = False; break
            try:
                yhat = float(np.asarray(
                    r.forecast(steps=1, exog=X_tr[vali[j]].reshape(1,-1))
                ).reshape(-1)[0])
            except Exception:
                fold_ok = False; break
            oof_p.append(yhat); oof_t.append(y_tr[vali[j]])
            h_X.append(X_tr[vali[j]]); h_y.append(y_tr[vali[j]])
        if fold_ok and oof_t:
            fm = _all_metrics(oof_t, oof_p)
            # fold-level train on training portion
            r2 = _fit_arimax(np.array(y_tr[tri]), X_tr[tri], p, d, q, trend)
            fold_train_mae = float(mean_absolute_error(
                y_tr[tri], np.asarray(r2.fittedvalues).reshape(-1))) if r2 else np.nan
            fold_rows.append({"fold": fi,
                               "fold_train_MAE": fold_train_mae,
                               "fold_val_MAE": fm["MAE"],
                               "fold_val_RMSE": fm["RMSE"],
                               "fold_val_MAPE": fm["MAPE"]})

    # Test (expanding window)
    h_X, h_y = list(X_tr), list(y_tr)
    test_p = []
    for i in range(len(X_te)):
        r = _fit_arimax(np.array(h_y), np.array(h_X), p, d, q, trend)
        if r is None:
            return None
        yhat = float(np.asarray(
            r.forecast(steps=1, exog=X_te[i].reshape(1,-1))).reshape(-1)[0])
        test_p.append(yhat)
        h_X.append(X_te[i]); h_y.append(y_te[i])
    test_m = _all_metrics(y_te, test_p)

    return _build_result(train_m, fold_rows, test_m)


# ── 3b. SVR ──────────────────────────────────────────────────

def _parse_gamma(raw):
    if pd.isna(raw): return None
    try: return float(raw)
    except: return str(raw)

def _parse_degree(raw):
    return None if pd.isna(raw) else int(raw)

def _build_svr(params):
    k = params["best_kernel"]
    kw = {"kernel": k, "C": float(params["best_C"]),
          "epsilon": float(params["best_epsilon"])}
    if k in ("rbf","poly"): kw["gamma"]  = _parse_gamma(params["best_gamma"])
    if k == "poly":          kw["degree"] = _parse_degree(params["best_degree"])
    return SVR(**kw)

def evaluate_svr(data, country, feature_list, params):
    df_use = prepare_country_data(data, country, feature_list, add_lag=True)
    if len(df_use) == 0: return None
    train, test, _ = scale_and_split(df_use, feature_list)
    X_tr, y_tr = train[feature_list].values, train[TARGET].values
    X_te, y_te = test[feature_list].values,  test[TARGET].values

    m = _build_svr(params); m.fit(X_tr, y_tr)
    train_m = _all_metrics(y_tr, m.predict(X_tr))

    n_splits = compute_n_splits(len(X_tr))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    for fi, (tri, vali) in enumerate(tscv.split(X_tr)):
        h_X, h_y = X_tr[tri].copy(), y_tr[tri].copy()
        oof_t, oof_p = [], []
        for j in range(len(vali)):
            m2 = _build_svr(params); m2.fit(h_X, h_y)
            yhat = float(m2.predict(X_tr[vali[j]].reshape(1,-1))[0])
            oof_p.append(yhat); oof_t.append(y_tr[vali[j]])
            h_X = np.vstack([h_X, X_tr[vali[j]]]); h_y = np.append(h_y, y_tr[vali[j]])
        if oof_t:
            fm = _all_metrics(oof_t, oof_p)
            mf = _build_svr(params); mf.fit(X_tr[tri], y_tr[tri])
            fold_rows.append({"fold": fi,
                               "fold_train_MAE": float(mean_absolute_error(y_tr[tri], mf.predict(X_tr[tri]))),
                               "fold_val_MAE": fm["MAE"],
                               "fold_val_RMSE": fm["RMSE"],
                               "fold_val_MAPE": fm["MAPE"]})

    h_X, h_y = X_tr.copy(), y_tr.copy()
    test_p = []
    for i in range(len(X_te)):
        m3 = _build_svr(params); m3.fit(h_X, h_y)
        test_p.append(float(m3.predict(X_te[i].reshape(1,-1))[0]))
        h_X = np.vstack([h_X, X_te[i]]); h_y = np.append(h_y, y_te[i])
    test_m = _all_metrics(y_te, test_p)

    return _build_result(train_m, fold_rows, test_m)


# ── 3c. XGBoost ──────────────────────────────────────────────

def _build_xgb(params):
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(params["best_n_estimators"]),
        max_depth=None if pd.isna(params.get("best_max_depth")) else int(params["best_max_depth"]),
        learning_rate=float(params["best_learning_rate"]),
        subsample=float(params["best_subsample"]),
        colsample_bytree=float(params["best_colsample_bytree"]),
        reg_lambda=float(params["best_reg_lambda"]),
        reg_alpha=float(params["best_reg_alpha"]),
        min_child_weight=int(params["best_min_child_weight"]),
        random_state=RANDOM_SEED, n_jobs=-1,
    )

def evaluate_xgboost(data, country, feature_list, params):
    df_use = prepare_country_data(data, country, feature_list, add_lag=True)
    if len(df_use) == 0: return None
    train, test, _ = scale_and_split(df_use, feature_list)
    X_tr, y_tr = train[feature_list].values, train[TARGET].values
    X_te, y_te = test[feature_list].values,  test[TARGET].values

    m = _build_xgb(params); m.fit(X_tr, y_tr)
    train_m = _all_metrics(y_tr, m.predict(X_tr))

    n_splits = compute_n_splits(len(X_tr))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    for fi, (tri, vali) in enumerate(tscv.split(X_tr)):
        h_X, h_y = X_tr[tri].copy(), y_tr[tri].copy()
        oof_t, oof_p = [], []
        for j in range(len(vali)):
            m2 = _build_xgb(params); m2.fit(h_X, h_y)
            yhat = float(m2.predict(X_tr[vali[j]].reshape(1,-1))[0])
            oof_p.append(yhat); oof_t.append(y_tr[vali[j]])
            h_X = np.vstack([h_X, X_tr[vali[j]]]); h_y = np.append(h_y, y_tr[vali[j]])
        if oof_t:
            fm = _all_metrics(oof_t, oof_p)
            mf = _build_xgb(params); mf.fit(X_tr[tri], y_tr[tri])
            fold_rows.append({"fold": fi,
                               "fold_train_MAE": float(mean_absolute_error(y_tr[tri], mf.predict(X_tr[tri]))),
                               "fold_val_MAE": fm["MAE"],
                               "fold_val_RMSE": fm["RMSE"],
                               "fold_val_MAPE": fm["MAPE"]})

    h_X, h_y = list(X_tr), list(y_tr)
    test_p = []
    for i in range(len(X_te)):
        m3 = _build_xgb(params); m3.fit(np.array(h_X), np.array(h_y))
        test_p.append(float(m3.predict(X_te[i].reshape(1,-1))[0]))
        h_X.append(X_te[i]); h_y.append(y_te[i])
    test_m = _all_metrics(y_te, test_p)

    return _build_result(train_m, fold_rows, test_m)


# ── 3d. LSTCN ────────────────────────────────────────────────

def _build_lstcn(n_features, params):
    return LSTCN(n_features=n_features,
                 n_steps=int(params["best_n_steps"]),
                 n_blocks=int(params["best_n_blocks"]),
                 function=str(params["best_function"]),
                 solver=str(params["best_solver"]),
                 alpha=float(params["best_alpha"]))

def evaluate_lstcn(data, country, feature_list, params):
    df_use = prepare_country_data(data, country, feature_list, add_lag=False)
    if len(df_use) == 0: return None
    train, test, sc_y = scale_and_split(df_use, feature_list)
    X_tr = train[feature_list].values
    y_tr = train[[TARGET]].values
    X_te = test[feature_list].values
    y_te_s = test[[TARGET]].values.reshape(-1)
    y_te_raw = df_use[df_use["Year"] >= TEST_START_YEAR][[TARGET]].values.reshape(-1)

    Xy_tr = np.hstack([X_tr, y_tr])
    nf    = Xy_tr.shape[1]
    X_in  = Xy_tr[:-1]; Y_in = Xy_tr[1:]

    # Train
    try:
        m = _build_lstcn(nf, params); m.fit(X_in, Y_in)
        train_m = _all_metrics(Y_in[:,-1], m.predict(X_in)[:,-1])
    except Exception:
        return None

    # Val (OOF rolling)
    n_splits = compute_n_splits(len(X_in))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    for fi, (tri, vali) in enumerate(tscv.split(X_in)):
        hist = list(Xy_tr[: tri[-1]+2])
        oof_t, oof_p = [], []
        ok = True
        for j in range(len(vali)):
            H = np.array(hist)
            try:
                m2 = _build_lstcn(nf, params); m2.fit(H[:-1], H[1:])
                yhat = float(m2.predict(H[-1:].copy())[0,-1])
            except Exception:
                ok = False; break
            oof_p.append(yhat); oof_t.append(Y_in[vali[j],-1])
            hist.append(Xy_tr[vali[j]+1])
        if ok and oof_t:
            fm = _all_metrics(oof_t, oof_p)
            # fold train MAE on training portion of Xy
            try:
                mf = _build_lstcn(nf, params)
                mf.fit(X_in[tri], Y_in[tri])
                ftm = float(mean_absolute_error(Y_in[tri,-1], mf.predict(X_in[tri])[:,-1]))
            except Exception:
                ftm = np.nan
            fold_rows.append({"fold": fi, "fold_train_MAE": ftm,
                               "fold_val_MAE": fm["MAE"],
                               "fold_val_RMSE": fm["RMSE"],
                               "fold_val_MAPE": fm["MAPE"]})

    # Test (expanding window)
    hist = list(Xy_tr)
    test_p = []
    for i in range(len(X_te)):
        H = np.array(hist)
        try:
            m3 = _build_lstcn(nf, params); m3.fit(H[:-1], H[1:])
            yhat_s = float(m3.predict(H[-1:].copy())[0,-1])
        except Exception:
            return None
        test_p.append(yhat_s)
        true_s = float(np.clip(sc_y.transform([[float(y_te_raw[i])]])[0][0],
                               SCALE_RANGE[0], SCALE_RANGE[1]))
        hist.append(np.append(X_te[i], true_s))
    test_m = _all_metrics(y_te_s, test_p)

    return _build_result(train_m, fold_rows, test_m)


# ── Shared result builder ─────────────────────────────────────

def _build_result(train_m: dict, fold_rows: list, test_m: dict) -> dict:
    """
    Aggregate fold-level results into a single result dict.
    fold_rows: list of dicts with fold_val_MAE / fold_val_RMSE / fold_val_MAPE
    """
    result = {}
    for metric in ("MAE", "RMSE", "MAPE"):
        result[f"train_{metric}"] = train_m[metric]
        result[f"test_{metric}"]  = test_m[metric]
        if fold_rows:
            vals = [r[f"fold_val_{metric}"] for r in fold_rows]
            result[f"val_{metric}"]     = float(np.mean(vals))
            result[f"val_{metric}_std"] = float(np.std(vals))
        else:
            result[f"val_{metric}"]     = np.nan
            result[f"val_{metric}_std"] = np.nan
    result["fold_rows"] = fold_rows
    return result


# ============================================================
# 4. TABLE RENDERER
# ============================================================

def render_overfitting_table(summary_df, model_name, feature_set,
                              metric_key, metric_label, png_path):
    """
    Render Train / Val (CV) / Test + Gap table for one metric.
    Columns: Country | Train <m> | Val <m> (CV) | Val Std | Test <m> | Gap
    """
    col_headers = list(summary_df.columns)
    cell_data   = [[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
                   for row in summary_df.values.tolist()]
    n_rows, n_cols = len(cell_data), len(col_headers)

    fig, ax = plt.subplots(figsize=(12, 0.55*(n_rows+2.5)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_data, colLabels=col_headers,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    gap_col = n_cols - 1

    for ci in range(n_cols):
        c = tbl[0, ci]
        c.set_facecolor(HEADER_COLOR)
        c.set_text_props(color="white", fontweight="bold")

    for ri in range(1, n_rows+1):
        is_avg = (ri == n_rows)
        bg = AVG_ROW_COLOR if is_avg else ("#F2F2F2" if ri%2==0 else "white")
        fw = "bold" if is_avg else "normal"
        for ci in range(n_cols):
            tbl[ri,ci].set_facecolor(bg)
            tbl[ri,ci].set_text_props(fontweight=fw)
        if not is_avg:
            tbl[ri, gap_col].set_facecolor(GAP_COL_EVEN if ri%2==0 else GAP_COL_ODD)

    feat = "Baseline" if feature_set == "baseline" else "Augmented"
    ax.set_title(
        f"{model_name} — {feat}: Train / Val (CV) / Test {metric_label} (scaled y)",
        fontsize=11, fontweight="bold", pad=4)
    fig.text(0.5, 0.04,
             f"Gap = Test − Train  │  Val Std = std across CV folds  │  "
             f"All {metric_label} on scaled y (range −0.9 to 0.9)",
             ha="center", fontsize=7.5, color="#555555")
    plt.subplots_adjust(top=0.88, bottom=0.12)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.relpath(png_path, SCRIPT_DIR)}")


def render_gap_bar(gap_df, metric_key, metric_label, png_path):
    """Grouped bar: avg gap per model, baseline vs augmented."""
    models   = gap_df["Model"].tolist()
    base_gap = [float(v) for v in gap_df[f"Baseline Avg Gap_{metric_key}"]]
    aug_gap  = [float(v) for v in gap_df[f"Augmented Avg Gap_{metric_key}"]]

    x, width = np.arange(len(models)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_b = ax.bar(x-width/2, base_gap, width, label="Baseline",
                    color="#A8C4D4", edgecolor="white")
    bars_a = ax.bar(x+width/2, aug_gap,  width, label="Augmented",
                    color="#2271B3", edgecolor="white")

    all_vals = base_gap + aug_gap
    y_min = min(0, min(all_vals)) * 1.3
    y_max = max(0, max(all_vals)) * 1.3
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color="black", linewidth=0.8)

    for bars in (bars_b, bars_a):
        for bar in bars:
            h = bar.get_height()
            offset = 0.002 if h >= 0 else -0.005
            ax.text(bar.get_x()+bar.get_width()/2, h+offset,
                    f"{h:.4f}", ha="center",
                    va="bottom" if h >= 0 else "top", fontsize=8.5)

    ax.set_title(
        f"Average Generalization Gap (Test − Train {metric_label}) by Model\n"
        "(averaged over 6 countries)", fontsize=11, pad=6)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel(f"Gap (Test − Train {metric_label}, scaled y)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(0.99, 0.97, "Larger gap = more overfitting",
            transform=ax.transAxes, fontsize=7.5, color="gray",
            ha="right", va="top")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.relpath(png_path, SCRIPT_DIR)}")


def render_gap_boxplot(all_country_gaps, metric_key, metric_label, png_path):
    """
    Horizontal boxplot: ARIMAX at top, LSTCN at bottom.

    Achieved by plotting in reversed order (LSTCN at y=0, ARIMAX at y=3)
    so matplotlib's natural y-axis (low→high = bottom→top) gives the
    correct top-to-bottom reading order.
    """
    display_order = ["ARIMAX", "SVR", "XGBoost", "LSTCN"]
    plot_order    = list(reversed(display_order))   # LSTCN=0, ARIMAX=3

    fig, ax = plt.subplots(figsize=(10, 5))

    offset = 0.18
    for i, model in enumerate(plot_order):
        for fset, yo, color in [
            ("baseline",   offset, "#A8C4D4"),
            ("augmented", -offset, "#2271B3"),
        ]:
            vals = [v for v in all_country_gaps.get((model, fset, metric_key), [])
                    if not np.isnan(v)]
            if not vals:
                continue
            ax.boxplot(
                [vals], positions=[i + yo], vert=False,
                widths=0.28, patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker="o", color=color, alpha=0.5, markersize=5),
            )

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#A8C4D4", label="Baseline"),
        Patch(facecolor="#2271B3", label="Augmented"),
    ], fontsize=9, loc="lower right")

    # y-tick positions 0,1,2,3 → labels LSTCN,XGBoost,SVR,ARIMAX (bottom→top)
    ax.set_yticks(range(len(plot_order)))
    ax.set_yticklabels(plot_order, fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel(f"Gap (Test − Train {metric_label}, scaled y)", fontsize=10)
    ax.set_title(
        f"Gap Distribution Across 6 Countries — {metric_label}\n"
        "(each box = 6 country-level gaps)",
        fontsize=11, pad=6)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.relpath(png_path, SCRIPT_DIR)}")


# ============================================================
# 5. PER-MODEL RUNNER
# ============================================================

EVALUATORS = {
    "ARIMAX":  evaluate_arimax,
    "SVR":     evaluate_svr,
    "XGBoost": evaluate_xgboost,
    "LSTCN":   evaluate_lstcn,
}


def run_model(model_name, data):
    metrics_df = load_metrics(model_name)
    evaluator  = EVALUATORS[model_name]
    results    = {"baseline": {}, "augmented": {}}

    for fset in ("baseline", "augmented"):
        fset_df   = metrics_df[metrics_df["feature_set"] == fset]
        feat_list = get_feature_list(model_name, fset)
        for country in COUNTRY_ORDER:
            row = fset_df[fset_df["country"] == country]
            if row.empty:
                print(f"  [SKIP] {model_name}|{fset}|{country}")
                continue
            params = row.iloc[0].to_dict()
            print(f"  {model_name}|{fset}|{country} … ", end="", flush=True)
            try:
                res = evaluator(data, country, feat_list, params)
                if res is None:
                    print("FAILED"); continue
                results[fset][country] = res
                print(f"train={res['train_MAE']:.4f}  "
                      f"val={res['val_MAE']:.4f}±{res['val_MAE_std']:.4f}  "
                      f"test={res['test_MAE']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
    return results


def build_summary_df(results, feature_set, metric_key):
    """Build summary DataFrame for one metric."""
    rows = []
    for country in COUNTRY_ORDER:
        r = results.get(feature_set, {}).get(country)
        if r is None:
            rows.append([country] + [np.nan]*5)
        else:
            gap = r[f"test_{metric_key}"] - r[f"train_{metric_key}"]
            rows.append([country,
                         r[f"train_{metric_key}"],
                         r[f"val_{metric_key}"],
                         r[f"val_{metric_key}_std"],
                         r[f"test_{metric_key}"],
                         gap])

    df = pd.DataFrame(rows, columns=[
        "Country", f"Train {metric_key}", f"Val {metric_key} (CV)",
        f"Val {metric_key} Std", f"Test {metric_key}", "Gap (Test−Train)"])

    # Average row
    num = df.select_dtypes(include=float)
    avg = {"Country": "Average"}; avg.update(num.mean().to_dict())
    df  = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    return df


def save_fold_details(results, model_name, data_dir):
    """Save per-fold train/val details to CSV."""
    rows = []
    for fset in ("baseline", "augmented"):
        for country in COUNTRY_ORDER:
            r = results.get(fset, {}).get(country)
            if r is None: continue
            for fr in r.get("fold_rows", []):
                rows.append({"model": model_name, "feature_set": fset,
                             "country": country, **fr})
    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(data_dir, f"{model_name}_fold_details.csv")
        df.to_csv(path, index=False)
        print(f"  Saved fold details: {os.path.basename(path)}")


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("overfitting_analysis.py")
    print("="*60)

    # Create output dirs
    metric_dirs = {}
    for mk, _, _ in METRIC_CONFIG:
        d = os.path.join(OUT_BASE, mk)
        os.makedirs(d, exist_ok=True)
        metric_dirs[mk] = d

    summary_dir = os.path.join(OUT_BASE, "data", "summary")
    folds_dir   = os.path.join(OUT_BASE, "data", "folds")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(folds_dir,   exist_ok=True)

    data = load_data()

    # Storage for gap summary and boxplot
    gap_rows         = []
    all_country_gaps = {}   # key=(model, fset, metric_key) → list of 6 gaps

    for model_name in ("ARIMAX", "SVR", "XGBoost", "LSTCN"):
        print(f"\n{'─'*50}\nRunning {model_name} …\n{'─'*50}")
        results = run_model(model_name, data)
        save_fold_details(results, model_name, folds_dir)

        gap_row = {"Model": model_name}

        for fset in ("baseline", "augmented"):
            for mk, ml, _ in METRIC_CONFIG:
                summary_df = build_summary_df(results, fset, mk)

                # PNG
                png = os.path.join(metric_dirs[mk],
                                   f"{model_name}_overfitting_{fset}.png")
                render_overfitting_table(summary_df, model_name, fset, mk, ml, png)

                # CSV
                csv = os.path.join(summary_dir,
                                   f"{model_name}_{fset}_{mk}.csv")
                summary_df.to_csv(csv, index=False)

                # Collect per-country gaps for boxplot
                country_gaps = []
                for country in COUNTRY_ORDER:
                    r = results.get(fset, {}).get(country)
                    if r:
                        country_gaps.append(
                            r[f"test_{mk}"] - r[f"train_{mk}"])
                all_country_gaps[(model_name, fset, mk)] = country_gaps

                # Avg gap for bar chart
                avg_row = summary_df[summary_df["Country"] == "Average"]
                gval = float(avg_row["Gap (Test−Train)"].values[0]) \
                       if len(avg_row) > 0 else np.nan
                fset_label = "Baseline" if fset == "baseline" else "Augmented"
                gap_row[f"{fset_label} Avg Gap_{mk}"] = gval

        gap_rows.append(gap_row)

    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(os.path.join(summary_dir, "all_models_gap_summary.csv"),
                  index=False)

    # Gap bar chart + boxplot for each metric
    for mk, ml, _ in METRIC_CONFIG:
        render_gap_bar(
            gap_df, mk, ml,
            os.path.join(metric_dirs[mk], "all_models_gap_summary.png"))
        render_gap_boxplot(
            all_country_gaps, mk, ml,
            os.path.join(metric_dirs[mk], "gap_distribution_boxplot.png"))

    print("\n✅  All done.")
    print(f"\n  Outputs  →  {OUT_BASE}")
    print("  MAE/     ← main tables + bar chart + boxplot")
    print("  RMSE/    ← same format")
    print("  MAPE/    ← same format")
    print("  data/summary/   ← all CSVs")
    print("  data/folds/     ← per-fold CV details")


if __name__ == "__main__":
    main()
