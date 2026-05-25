"""
ARIMAX Rolling One-Step-Ahead Forecast for CO2 Emissions
=========================================================
This script trains an ARIMAX (AutoRegressive Integrated Moving Average with
eXogenous variables) model per country to forecast annual total CO2 emissions.

Pipeline overview
-----------------
For each country × feature_set combination:
  1. Filter rows for the target country, sort by year, and select the
   required columns. dropna() is applied as a safety guard; the upstream
   dataset is assumed to be already preprocessed.
  2. Scale all variables to [-0.9, 0.9] using the full usable dataset
  3. Split into train (≤ 2016) and test (≥ 2017) after scaling
  4. Tune ARIMAX order (p, d, q) and trend via time-series cross-validation
     on the training set only. CV uses rolling one-step-ahead evaluation
     to mirror the actual test procedure.
  5. Run a rolling one-step-ahead forecast on the test set:
       - Re-fit ARIMAX on all available history before each step
       - Append the TRUE observed value to history after each step
         (expanding window)
  6. Inverse-transform predictions and compute evaluation metrics
  7. Save a checkpoint CSV after every country completes

Key design notes
----------------
- No lag feature is added manually. ARIMAX captures autoregression
  through the AR(p) component; the p parameter in the grid search
  determines how many past values are used.
- SEASONAL_ORDER = (0,0,0,0) disables seasonality, making this a
  pure ARIMAX rather than SARIMAX.
- The scaler is fit on the full usable dataset (train + test) before
  splitting. This is intentional for scaling consistency across countries;
  all four models in the study follow the same convention.
- Tuning criterion: cv_MAE_scaled_y_mean (primary), aligned with SVR, XGBoost, and LSTCN
  for fair cross-model comparison.

Output files
------------
  arimax_metrics_summary.csv  : Best params + all metrics per country/feature_set
  arimax_predictions_all.csv  : Year-by-year actual vs predicted (original scale)
  arimax_tuning_results.csv   : CV scores for every hyperparameter combination
"""

import warnings
import pandas as pd
import numpy as np
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# Suppress expected warnings from statsmodels during iterative MLE fitting
# and from sklearn/pandas version compatibility
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


# =============================================================================
# 1. Configuration
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Path to the preprocessed dataset (annual panel data, one row per country-year)
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed", "dataset_recursive_3yr_avg_drop_Industry.csv")

# Column name of the prediction target
TARGET = "Total CO2 emissions"

# Countries to model. Each is run independently with its own scaler and tuning.
COUNTRIES = [
    "Canada",
    "China",
    "India",
    "Indonesia",
    "Russian Federation",
    "United States"
]

# Train/test temporal split: years up to and including 2016 are used for
# training; 2017 onward is the held-out test set.
TRAIN_END_YEAR = 2016
TEST_START_YEAR = 2017

# MinMaxScaler target range. Using (-0.9, 0.9) instead of (-1, 1) provides
# a small margin that prevents exact boundary values, which can destabilise
# some optimisers. All four models in this study use the same range.
TRAIN_SCALE_RANGE = (-0.9, 0.9)

# ---- Feature groups --------------------------------------------------------
# BASE_FEATURES: socioeconomic and energy-related exogenous variables
BASE_FEATURES = [
    "Population",
    "GDP",
    "Electric power consumption",
    "Fossil fuel energy consumption",
    "Renewable energy consumption",
    "Fertilizer consumption"
]

# TEMP_FEATURES: climate variables added in the augmented experiment
TEMP_FEATURES = [
    "Temperature annual mean",
    "Temperature std across months",
    "Number of frost days",
    "Number of hot days"
]

# Two experiments are run per country:
#   "baseline"  – only socioeconomic/energy features
#   "augmented" – adds climate features on top of baseline
# NOTE: Unlike SVR and XGBoost, ARIMAX does NOT include a lagged CO2 feature
# here. The AR(p) component of ARIMAX already models past CO2 values through
# the endogenous variable directly.
FEATURE_SETS = {
    "baseline": BASE_FEATURES,
    "augmented": BASE_FEATURES + TEMP_FEATURES
}

# ---- ARIMAX hyperparameter search grid ------------------------------------
# p : AR order – number of past endogenous (CO2) values included
# d : differencing order – how many times the series is differenced before
#     fitting, to reduce non-stationarity (0 = no differencing)
# q : MA order – number of past residuals (forecast errors) included
# trend: "n" = no intercept/constant; "c" = include an intercept term
# Total search space: 4 × 2 × 4 × 2 = 64 combinations
P_GRID = [0, 1, 2, 3]
D_GRID = [0, 1]
Q_GRID = [0, 1, 2, 3]
TREND_GRID = ["n", "c"]

# Seasonal order set to (0,0,0,0) to disable seasonality – annual data has no
# meaningful within-year seasonal pattern.
SEASONAL_ORDER = (0, 0, 0, 0)

# Relaxing these constraints allows the model to fit even when AR/MA roots
# fall outside the unit circle, which can happen with short annual series.
ENFORCE_STATIONARITY = False
ENFORCE_INVERTIBILITY = False

# Maximum number of MLE (Maximum Likelihood Estimation) iterations per fit
MAX_ITER = 200

# Primary metric used to rank hyperparameter combinations during CV tuning.
# Using the scaled-space MAE makes cross-country comparison consistent and
# aligns with the criterion used in SVR, XGBoost, and LSTCN.
TUNING_PRIMARY_METRIC = "MAE_scaled_y"

# ---- Output file names -----------------------------------------------------
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
METRICS_OUT = os.path.join(RESULTS_DIR, "metrics",     "arimax_metrics_summary.csv")
PRED_OUT    = os.path.join(RESULTS_DIR, "predictions", "arimax_predictions_all.csv")
TUNING_OUT  = os.path.join(RESULTS_DIR, "tuning",      "arimax_tuning_results.csv")


# =============================================================================
# 2. Utility / helper functions
# =============================================================================

def compute_metrics(y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled):
    """
    Compute a standard set of regression metrics in both the original scale
    and the scaled ([-0.9, 0.9]) space.

    Parameters
    ----------
    y_true_raw    : array-like – ground-truth values in the original CO2 unit
    y_pred_raw    : array-like – predicted values in the original CO2 unit
    y_true_scaled : array-like – ground-truth values after MinMax scaling
    y_pred_scaled : array-like – predicted values after MinMax scaling

    Returns
    -------
    dict with keys:
        RMSE_absolute  – RMSE in original units
        RMSE_scaled_y  – RMSE in scaled space (used for cross-country comparison)
        RMSE_relative  – RMSE normalised by the mean of y_true_raw
        MAE_absolute   – MAE in original units
        MAE_scaled_y   – MAE in scaled space (primary tuning criterion)
        MAE_relative   – MAE normalised by the mean of y_true_raw
        MAPE           – Mean Absolute Percentage Error (0–∞; lower is better)
        Accuracy       – 1 - MAPE (can be negative if MAPE > 1)
        R_squared      – Coefficient of determination on the original scale

    Note on CV usage
    ----------------
    When called inside evaluate_param_set_cv(), y_true_raw == y_true_scaled
    and y_pred_raw == y_pred_scaled (because the CV fold data is already in
    scaled space). In that case MAE_absolute and MAE_scaled_y are numerically
    identical, and only MAE_scaled_y is used for tuning decisions.
    """
    y_true_raw    = np.asarray(y_true_raw).reshape(-1)
    y_pred_raw    = np.asarray(y_pred_raw).reshape(-1)
    y_true_scaled = np.asarray(y_true_scaled).reshape(-1)
    y_pred_scaled = np.asarray(y_pred_scaled).reshape(-1)

    # ── Absolute-scale metrics (original CO2 units) ──────────────────────────
    rmse_abs = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    mae_abs  = mean_absolute_error(y_true_raw, y_pred_raw)

    # ── Scaled-space metrics (comparable across countries) ───────────────────
    rmse_scaled = np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))
    mae_scaled  = mean_absolute_error(y_true_scaled, y_pred_scaled)

    # ── Relative metrics (dimensionless) ─────────────────────────────────────
    # Guard against division by zero if the target mean is effectively zero
    y_mean = np.mean(y_true_raw)
    if np.isclose(y_mean, 0):
        rmse_rel = np.nan
        mae_rel  = np.nan
    else:
        rmse_rel = rmse_abs / y_mean
        mae_rel  = mae_abs  / y_mean

    # ── Other metrics ─────────────────────────────────────────────────────────
    mape     = mean_absolute_percentage_error(y_true_raw, y_pred_raw)
    accuracy = 1 - mape          # warning: can be negative when MAPE > 1
    r2       = r2_score(y_true_raw, y_pred_raw)

    return {
        "RMSE_absolute": rmse_abs,
        "RMSE_scaled_y": rmse_scaled,
        "RMSE_relative": rmse_rel,
        "MAE_absolute":  mae_abs,
        "MAE_scaled_y":  mae_scaled,
        "MAE_relative":  mae_rel,
        "MAPE":          mape,
        "Accuracy":      accuracy,
        "R_squared":     r2
    }


def safe_fit_arimax(y_train, X_train, order, trend):
    """
    Fit a SARIMAX model (used here as ARIMAX with no seasonality) and return
    the fitted results object, or None if the fit fails for any reason.

    Wrapping the fit in a try/except is necessary because:
      - Some (p, d, q) orders produce singular matrices with short series
      - MLE optimisation can fail to converge for ill-conditioned problems
    Returning None instead of raising lets the caller skip this combination
    gracefully rather than aborting the entire tuning run.

    Parameters
    ----------
    y_train : 1-D array – scaled endogenous variable (CO2 time series)
    X_train : 2-D array – scaled exogenous features, shape (T, n_features)
    order   : tuple (p, d, q)
    trend   : str "n" or "c"

    Returns
    -------
    fitted SARIMAXResults object, or None on failure
    """
    try:
        model = SARIMAX(
            endog=y_train,
            exog=X_train,
            order=order,
            seasonal_order=SEASONAL_ORDER,       # (0,0,0,0) → no seasonality
            trend=trend,
            enforce_stationarity=ENFORCE_STATIONARITY,
            enforce_invertibility=ENFORCE_INVERTIBILITY
        )
        results = model.fit(disp=False, maxiter=MAX_ITER)
        return results
    except Exception:
        return None


def one_step_forecast(results, X_next):
    """
    Produce a single one-step-ahead forecast from a fitted SARIMAX model.

    Parameters
    ----------
    results : fitted SARIMAX Results object
    X_next  : 2-D array of shape (1, n_features) – exogenous values for the
              next time step

    Returns
    -------
    float – the point forecast for the next step (still in scaled space)
    """
    pred = results.forecast(steps=1, exog=X_next)
    return float(np.asarray(pred).reshape(-1)[0])


def evaluate_param_set_cv(X_train, y_train, order, trend):
    """
    Evaluate one (p, d, q, trend) combination via rolling one-step-ahead
    time-series cross-validation on the training data.

    Why rolling one-step-ahead inside CV?
    --------------------------------------
    The final test evaluation also uses rolling one-step-ahead (re-fitting
    on expanding history before each prediction). CV must mirror this exactly
    so that the hyperparameter ranking reflects true out-of-sample behaviour.
    Using a static multi-step forecast inside CV would create a mismatch and
    potentially select the wrong hyperparameters.

    CV structure
    ------------
    TimeSeriesSplit ensures that validation years always come after training
    years within each fold (no temporal leakage).

        Fold 1: train [t0 … t_k]      val [t_{k+1} … t_{k+m}]
        Fold 2: train [t0 … t_{k+m}]  val [t_{k+m+1} … ]
        ...

    Inside each fold, for every validation step j:
      1. Re-fit ARIMAX on history (initially = fold train, grows each step)
      2. Forecast one step ahead
      3. Append the TRUE observation to history (not the forecast)

    Parameters
    ----------
    X_train : 2-D array – scaled features, shape (T_train, n_features)
    y_train : 1-D array – scaled CO2 values, shape (T_train,)
    order   : tuple (p, d, q)
    trend   : str "n" or "c"

    Returns
    -------
    dict with keys:
        valid            – bool; False if all folds failed
        reason           – str; failure message if valid is False
        n_splits         – int; actual number of CV folds used
        n_success_folds  – int
        n_failed_folds   – int
        cv_<metric>_mean – float; mean of each metric across successful folds
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    # Minimum data guard: ARIMAX needs enough observations to estimate
    # AR and MA parameters. Below 6 points we skip entirely.
    if len(y_train) < 6:
        return {
            "valid": False,
            "reason": "Not enough training rows for hyperparameter tuning."
        }

    # Adapt number of folds to the available training size.
    # min/max bounds prevent either too-few or too-many folds for short series.
    n_splits = min(5, max(2, len(X_train) // 4))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metric_rows = []
    n_failed_folds   = 0
    n_success_folds  = 0

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        # ARIMAX-specific guard: the AR and MA components require a minimum
        # number of observations. If the fold's train split is too short for
        # the current (p, d, q) order, skip this fold rather than letting it
        # produce a degenerate fit.
        # Minimum required: max(p, q) + d + 1 observations
        if len(y_tr) <= max(order[0], order[2]) + order[1] + 1:
            n_failed_folds += 1
            continue

        # Initialise the expanding history with the fold's training portion
        history_X = list(X_tr)
        history_y = list(y_tr)
        preds_val  = []
        fold_failed = False

        # Rolling one-step-ahead loop over the validation window
        for i in range(len(X_val)):
            y_hist = np.asarray(history_y, dtype=float)
            X_hist = np.asarray(history_X, dtype=float)

            # Re-fit ARIMAX on the full current history
            fitted = safe_fit_arimax(
                y_train=y_hist,
                X_train=X_hist,
                order=order,
                trend=trend
            )

            # If the fit failed (e.g. singular matrix), mark the whole fold
            # as failed and move on. We do not propagate the error upward.
            if fitted is None:
                fold_failed = True
                break

            try:
                x_next = np.asarray(X_val[i], dtype=float).reshape(1, -1)
                yhat = one_step_forecast(fitted, x_next)
                preds_val.append(yhat)
            except Exception:
                fold_failed = True
                break

            # Expand history with the TRUE observation, not the forecast.
            # This matches real-world usage where actual data becomes available
            # at each period end before the next forecast.
            history_X.append(X_val[i])
            history_y.append(y_val[i])

        if fold_failed or len(preds_val) != len(y_val):
            n_failed_folds += 1
            continue

        # Compute metrics for this fold.
        # Both raw and scaled arguments receive the same scaled values because
        # all CV data is already in the scaled space. As a result,
        # MAE_absolute == MAE_scaled_y numerically here; only MAE_scaled_y
        # is used in the ranking step downstream.
        preds_val = np.asarray(preds_val).reshape(-1)
        metrics_fold = compute_metrics(
            y_true_raw=y_val,
            y_pred_raw=preds_val,
            y_true_scaled=y_val,
            y_pred_scaled=preds_val
        )
        fold_metric_rows.append(metrics_fold)
        n_success_folds += 1

    # If no fold succeeded, this parameter combination is invalid
    if n_success_folds == 0:
        return {
            "valid": False,
            "reason": "All folds failed.",
            "n_splits": n_splits,
            "n_success_folds": 0,
            "n_failed_folds": n_failed_folds
        }

    # Aggregate metrics across successful folds (simple mean)
    fold_df = pd.DataFrame(fold_metric_rows)
    out = {
        "valid": True,
        "n_splits": n_splits,
        "n_success_folds": n_success_folds,
        "n_failed_folds": n_failed_folds
    }
    for col in fold_df.columns:
        out[f"cv_{col}_mean"] = fold_df[col].mean()

    return out


def tune_arimax_hyperparams(X_train, y_train, country_name, feature_set_name):
    """
    Grid-search over all (p, d, q, trend) combinations and select the best
    one by time-series cross-validation.

    Ranking priority (all ascending, lower is better):
        1. cv_MAE_scaled_y_mean   – primary criterion (aligns with SVR/XGBoost)
        2. cv_RMSE_scaled_y_mean  – secondary; penalises large errors more
        3. cv_MAPE_mean           – tertiary; scale-free percentage error
        4. p, d, q                – tiebreaker: prefer simpler models

    Parameters
    ----------
    X_train          : 2-D array – scaled features (training period only)
    y_train          : 1-D array – scaled CO2 (training period only)
    country_name     : str – used for labelling tuning records
    feature_set_name : str – "baseline" or "augmented"

    Returns
    -------
    best_params : dict with keys best_p, best_d, best_q, best_trend
    tuning_df   : DataFrame with one row per (p, d, q, trend) combination,
                  containing CV metrics and validity flags
    """
    tuning_rows = []

    # Exhaustive grid search: 4 × 2 × 4 × 2 = 64 combinations
    for p in P_GRID:
        for d in D_GRID:
            for q in Q_GRID:
                for trend in TREND_GRID:
                    order = (p, d, q)
                    cv_result = evaluate_param_set_cv(
                        X_train=X_train,
                        y_train=y_train,
                        order=order,
                        trend=trend
                    )

                    # Record every combination, even failures, for diagnostics
                    tuning_rows.append({
                        "country":       country_name,
                        "feature_set":   feature_set_name,
                        "p":             p,
                        "d":             d,
                        "q":             q,
                        "trend":         trend,
                        "seasonal_order": str(SEASONAL_ORDER),
                        "valid":          cv_result.get("valid", False),
                        "reason":         cv_result.get("reason", ""),
                        "n_splits":       cv_result.get("n_splits", np.nan),
                        "n_success_folds": cv_result.get("n_success_folds", 0),
                        "n_failed_folds":  cv_result.get("n_failed_folds", 0),
                        "cv_RMSE_absolute_mean": cv_result.get("cv_RMSE_absolute_mean", np.nan),
                        "cv_RMSE_scaled_y_mean": cv_result.get("cv_RMSE_scaled_y_mean", np.nan),
                        "cv_RMSE_relative_mean": cv_result.get("cv_RMSE_relative_mean", np.nan),
                        "cv_MAE_absolute_mean":  cv_result.get("cv_MAE_absolute_mean",  np.nan),
                        "cv_MAE_scaled_y_mean":  cv_result.get("cv_MAE_scaled_y_mean",  np.nan),
                        "cv_MAE_relative_mean":  cv_result.get("cv_MAE_relative_mean",  np.nan),
                        "cv_MAPE_mean":          cv_result.get("cv_MAPE_mean",           np.nan),
                        "cv_Accuracy_mean":      cv_result.get("cv_Accuracy_mean",       np.nan),
                        "cv_R_squared_mean":     cv_result.get("cv_R_squared_mean",      np.nan)
                    })

    tuning_df = pd.DataFrame(tuning_rows)

    # Keep only combinations that produced at least one valid CV score
    valid_df = tuning_df[
        tuning_df["valid"].fillna(False)
        & tuning_df["cv_MAE_scaled_y_mean"].notna()
    ].copy()

    if valid_df.empty:
        raise RuntimeError(
            f"{country_name} | {feature_set_name}: no valid ARIMAX parameter set found."
        )

    # Sort by the ranking priority described in the docstring
    valid_df = valid_df.sort_values(
        by=[
            "cv_MAE_scaled_y_mean",
            "cv_RMSE_scaled_y_mean",
            "cv_MAPE_mean",
            "p", "d", "q"          # tiebreaker: prefer simpler order
        ],
        ascending=[True, True, True, True, True, True]
    ).reset_index(drop=True)

    best_row = valid_df.iloc[0].to_dict()
    best_params = {
        "best_p":     int(best_row["p"]),
        "best_d":     int(best_row["d"]),
        "best_q":     int(best_row["q"]),
        "best_trend": best_row["trend"]
    }

    return best_params, tuning_df


def run_arimax_single_country(data, country_name, feature_set_name, feature_list):
    """
    Full pipeline for a single country × feature_set combination:
      1. Extract and clean country data
      2. Scale features and target on the full usable dataset
      3. Split into train / test
      4. Tune hyperparameters via CV on training data only
      5. Rolling one-step-ahead forecast on the test set
      6. Inverse-transform and evaluate

    Parameters
    ----------
    data             : DataFrame – the full multi-country panel dataset
    country_name     : str
    feature_set_name : str – "baseline" or "augmented"
    feature_list     : list[str] – feature column names for this experiment

    Returns
    -------
    metrics    : dict – test-set evaluation metrics
    pred_df    : DataFrame – year / actual / predicted (original scale)
    best_params: dict – best (p, d, q, trend) found by CV
    tuning_df  : DataFrame – full CV tuning log
    """
    # ── Step 1: Filter country and drop rows with any missing value ────────
    cdf = data[data["Country Name"] == country_name].sort_values("Year").copy()

    # Select only the columns needed; drop any year where a feature is missing.
    # This happens before scaling so the scaler only sees clean data.
    df_use = cdf[["Year", TARGET] + feature_list].dropna().copy()
    if len(df_use) == 0:
        raise ValueError(f"{country_name}: no usable rows after dropna().")

    # ── Step 2: Scale on the FULL usable dataset (before train/test split) ─
    # Both scalers are fit on all years, including the test period. This is a
    # deliberate design choice shared across all four models in the study:
    # it ensures the target's min/max used for inverse-transforming test
    # predictions is stable and consistent. In a strict production setting
    # one would fit the scaler on training data only.
    sc_x = MinMaxScaler(feature_range=TRAIN_SCALE_RANGE)
    sc_y = MinMaxScaler(feature_range=TRAIN_SCALE_RANGE)

    df_scaled = df_use.copy()
    df_scaled[feature_list] = sc_x.fit_transform(df_use[feature_list].values)
    df_scaled[[TARGET]]     = sc_y.fit_transform(df_use[[TARGET]].values)

    # ── Step 3: Train / test split (year-based, after scaling) ─────────────
    train_df = df_scaled[df_scaled["Year"] <= TRAIN_END_YEAR].copy()
    test_df  = df_scaled[df_scaled["Year"] >= TEST_START_YEAR].copy()

    if len(train_df) < 6:
        raise ValueError(f"{country_name}: training rows too few ({len(train_df)}).")
    if len(test_df) == 0:
        raise ValueError(f"{country_name}: no test rows after split.")

    X_train    = train_df[feature_list].values
    y_train    = train_df[[TARGET]].values.reshape(-1)
    X_test     = test_df[feature_list].values
    years_test = test_df["Year"].values

    # y_test_raw: true CO2 in the ORIGINAL (unscaled) unit, used for
    # absolute metrics (MAE_absolute, MAPE, R²) at evaluation time.
    y_test_raw = df_use[df_use["Year"] >= TEST_START_YEAR][[TARGET]].values.reshape(-1)

    # ── Step 4: Hyperparameter tuning (train set only) ─────────────────────
    best_params, tuning_df = tune_arimax_hyperparams(
        X_train=X_train,
        y_train=y_train,
        country_name=country_name,
        feature_set_name=feature_set_name
    )

    order = (best_params["best_p"], best_params["best_d"], best_params["best_q"])
    trend = best_params["best_trend"]

    # ── Step 5: Rolling one-step-ahead forecast on the test set ────────────
    # Start history with the complete training set.
    # After each prediction the TRUE observation is appended (expanding window).
    # This mirrors the real-world scenario where actual data is observed at
    # the end of each year before the next forecast is made.
    history_X  = list(X_train)
    history_y  = list(y_train)
    preds_scaled = []

    for i in range(len(X_test)):
        X_hist = np.asarray(history_X, dtype=float)
        y_hist = np.asarray(history_y, dtype=float)

        # Re-fit the model on all available history up to this point
        fitted = safe_fit_arimax(
            y_train=y_hist,
            X_train=X_hist,
            order=order,
            trend=trend
        )
        if fitted is None:
            # Unlike in CV (where we just skip the fold), a failure here during
            # the final test evaluation is a hard error – we cannot produce a
            # valid forecast for this country / feature_set combination.
            raise RuntimeError(
                f"{country_name}: ARIMAX fit failed during rolling forecast "
                f"with order={order}, trend={trend}."
            )

        # Forecast one step ahead using the next test year's exogenous values
        x_next = np.asarray(X_test[i], dtype=float).reshape(1, -1)
        yhat_scaled = one_step_forecast(fitted, x_next)
        preds_scaled.append(yhat_scaled)

        # Expand history: use the TRUE scaled y (not the forecast yhat)
        true_y_scaled = float(test_df.iloc[i][TARGET])
        history_X.append(X_test[i])
        history_y.append(true_y_scaled)

    # ── Step 6: Inverse-transform and evaluate ─────────────────────────────
    preds_scaled   = np.array(preds_scaled).reshape(-1, 1)
    y_pred_raw     = sc_y.inverse_transform(preds_scaled).reshape(-1)
    y_test_scaled  = test_df[[TARGET]].values.reshape(-1)
    y_pred_scaled  = preds_scaled.reshape(-1)

    metrics = compute_metrics(
        y_true_raw=y_test_raw,       # original scale – for absolute/MAPE/R²
        y_pred_raw=y_pred_raw,       # inverse-transformed predictions
        y_true_scaled=y_test_scaled, # scaled space – for MAE_scaled_y / RMSE_scaled_y
        y_pred_scaled=y_pred_scaled  # predictions still in scaled space
    )

    # Build a tidy predictions DataFrame for later export
    pred_df = pd.DataFrame({
        "country":     country_name,
        "feature_set": feature_set_name,
        "year":        years_test,
        "actual":      y_test_raw,
        "predicted":   y_pred_raw
    })

    return metrics, pred_df, best_params, tuning_df


def make_average_rows(metrics_df):
    """
    Append summary rows to the metrics table for convenient reporting:
      - BASELINE_AVG  : mean across all countries for the baseline feature set
      - AUGMENTED_AVG : mean across all countries for the augmented feature set
      - OVERALL_AVG   : mean across all country × feature_set combinations

    These rows are appended at the bottom of the metrics CSV so that a
    reader can quickly see aggregate model performance without additional
    processing.

    Parameters
    ----------
    metrics_df : DataFrame – one row per country × feature_set

    Returns
    -------
    DataFrame with three summary rows
    """
    metric_cols = [
        "RMSE_absolute", "RMSE_scaled_y", "RMSE_relative",
        "MAE_absolute",  "MAE_scaled_y",  "MAE_relative",
        "MAPE", "Accuracy", "R_squared"
    ]

    rows = []

    # Per-feature-set averages across countries
    for name, fset in [
        ("BASELINE_AVG",  "baseline"),
        ("AUGMENTED_AVG", "augmented")
    ]:
        sub = metrics_df[metrics_df["feature_set"] == fset].copy()
        row = {
            "country": name, "feature_set": fset,
            "best_p": np.nan, "best_d": np.nan,
            "best_q": np.nan, "best_trend": np.nan
        }
        for col in metric_cols:
            row[col] = sub[col].mean()
        rows.append(row)

    # Overall average across all country × feature_set combinations
    overall_row = {
        "country": "OVERALL_AVG", "feature_set": "all",
        "best_p": np.nan, "best_d": np.nan,
        "best_q": np.nan, "best_trend": np.nan
    }
    for col in metric_cols:
        overall_row[col] = metrics_df[col].mean()
    rows.append(overall_row)

    return pd.DataFrame(rows)


def save_checkpoint(metrics_rows, pred_frames, tuning_frames):
    """
    Write all accumulated results to disk.

    This function is called after every country completes (not only at the
    end of the script) so that partial results are preserved if the run is
    interrupted. Each call overwrites the previous checkpoint with the most
    up-to-date cumulative state.

    Three files are written:
      METRICS_OUT  – best params + test metrics per country/feature_set,
                     with BASELINE_AVG / AUGMENTED_AVG / OVERALL_AVG appended
      PRED_OUT     – year-by-year actual vs predicted values (original scale)
      TUNING_OUT   – CV scores for every hyperparameter combination tried,
                     sorted by cv_MAE_scaled_y_mean for readability

    Parameters
    ----------
    metrics_rows  : list[dict] – one dict per completed country/feature_set
    pred_frames   : list[DataFrame] – prediction DataFrames to concatenate
    tuning_frames : list[DataFrame] – tuning log DataFrames to concatenate
    """
    # ── Metrics CSV ────────────────────────────────────────────────────────
    if metrics_rows:
        metrics_df    = pd.DataFrame(metrics_rows)
        avg_df        = make_average_rows(metrics_df)
        metrics_final = pd.concat([metrics_df, avg_df], ignore_index=True)
        metrics_final = metrics_final[[
            "country", "feature_set",
            "best_p", "best_d", "best_q", "best_trend",
            "RMSE_absolute", "RMSE_scaled_y", "RMSE_relative",
            "MAE_absolute",  "MAE_scaled_y",  "MAE_relative",
            "MAPE", "Accuracy", "R_squared"
        ]]
        metrics_final.to_csv(METRICS_OUT, index=False)

    # ── Predictions CSV ────────────────────────────────────────────────────
    if pred_frames:
        preds_final = pd.concat(pred_frames, ignore_index=True)
        preds_final = preds_final[
            ["country", "feature_set", "year", "actual", "predicted"]
        ].sort_values(["country", "feature_set", "year"])
        preds_final.to_csv(PRED_OUT, index=False)

    # ── Tuning log CSV ─────────────────────────────────────────────────────
    if tuning_frames:
        tuning_final = pd.concat(tuning_frames, ignore_index=True)
        tuning_final = tuning_final[[
            "country", "feature_set",
            "p", "d", "q", "trend", "seasonal_order",
            "valid", "reason",
            "n_splits", "n_success_folds", "n_failed_folds",
            "cv_RMSE_absolute_mean", "cv_RMSE_scaled_y_mean", "cv_RMSE_relative_mean",
            "cv_MAE_absolute_mean",  "cv_MAE_scaled_y_mean",  "cv_MAE_relative_mean",
            "cv_MAPE_mean", "cv_Accuracy_mean", "cv_R_squared_mean"
        ]].sort_values(
            ["country", "feature_set", "cv_MAE_scaled_y_mean", "p", "d", "q"],
            ascending=[True, True, True, True, True, True]
        )
        tuning_final.to_csv(TUNING_OUT, index=False)


# =============================================================================
# 3. Main entry point
# =============================================================================

def main():
    """
    Orchestrate the full experiment:
      - Load the panel dataset
      - For each country × feature_set: run the complete ARIMAX pipeline
      - Save a checkpoint after each country to guard against mid-run crashes
      - Write final output files when all countries are done

    Progress is printed to stdout so long-running jobs can be monitored.
    """
    os.makedirs(os.path.join(RESULTS_DIR, "metrics"),     exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "tuning"),      exist_ok=True)
    
    data = pd.read_csv(DATA_PATH)

    # Accumulators for results across countries
    metrics_rows  = []
    pred_frames   = []
    tuning_frames = []

    for country in COUNTRIES:
        print(f"\n===== Running {country} =====")
        country_success = False

        for feature_set_name, feature_list in FEATURE_SETS.items():
            print(f"  -> feature set: {feature_set_name}")
            try:
                metrics, pred_df, best_params, tuning_df = run_arimax_single_country(
                    data=data,
                    country_name=country,
                    feature_set_name=feature_set_name,
                    feature_list=feature_list
                )

                # Merge best_params and metrics into a single flat dict for the
                # metrics summary CSV row
                metrics_rows.append({
                    "country":     country,
                    "feature_set": feature_set_name,
                    **best_params,
                    **metrics
                })
                pred_frames.append(pred_df)
                tuning_frames.append(tuning_df)
                country_success = True

                print(
                    f"     best_order=({best_params['best_p']},"
                    f"{best_params['best_d']},{best_params['best_q']}), "
                    f"best_trend={best_params['best_trend']}, "
                    f"MAE_scaled_y={metrics['MAE_scaled_y']:.6f}, "
                    f"RMSE_scaled_y={metrics['RMSE_scaled_y']:.6f}, "
                    f"Accuracy={metrics['Accuracy']:.6f}"
                )
            except Exception as e:
                print(f"     ERROR: {e}")

        # Checkpoint: write all results accumulated so far to CSV.
        # This means if a later country crashes, results for completed
        # countries are not lost. Only written when at least one feature_set
        # succeeded for this country.
        if country_success:
            save_checkpoint(metrics_rows, pred_frames, tuning_frames)
            print(f"  >> Checkpoint saved after {country}")

    if not metrics_rows:
        raise RuntimeError("No successful ARIMAX runs. Please check the data and parameter grid.")

    # Final save (ensures the last country's results are always written even
    # if save_checkpoint was already called after it)
    save_checkpoint(metrics_rows, pred_frames, tuning_frames)

    print("\nSaved files:")
    print(f"  - {METRICS_OUT}")
    print(f"  - {PRED_OUT}")
    print(f"  - {TUNING_OUT}")


if __name__ == "__main__":
    main()
