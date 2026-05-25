"""
SVR Rolling One-Step-Ahead Forecast for CO2 Emissions
======================================================
This script trains a Support Vector Regressor (SVR) per country to forecast
annual total CO2 emissions, using an expanding-window rolling evaluation
that mirrors real-world sequential prediction.

Pipeline overview
-----------------
For each country × feature_set combination:
  1. Filter rows for the target country, sort by year, and select the
     required columns. dropna() is applied as a safety guard; the upstream
     dataset is assumed to be already preprocessed.
  2. Construct a one-period lag of the target (CO2_lag1) as an autoregressive
     feature, then drop the first year which has no lag value.
  3. Scale all variables to [-0.9, 0.9] using the full usable dataset.
  4. Split into train (<=2016) and test (>=2017) after scaling.
  5. Tune SVR hyperparameters (kernel, C, epsilon, gamma, degree) via
     time-series cross-validation on the training set only.
     CV uses rolling one-step-ahead evaluation to mirror the test procedure.
  6. Run a rolling one-step-ahead forecast on the test set:
       - Re-fit SVR on all available history before each step
       - Append the TRUE observed value to history after each step
         (expanding window)
  7. Inverse-transform predictions and compute evaluation metrics.

Key design notes
----------------
- Unlike ARIMAX (which captures past CO2 values via the AR(p) component) and
  LSTCN (which passes the full [X | y] row at each time step), SVR has no
  built-in memory of past values. A one-period lag of the target
  (CO2_emissions_lag1) is therefore added as an explicit feature so the model
  knows the previous year's CO2 level. Only a single lag is used; deeper
  autoregressive structure is not captured.
- The scaler is fit on the full usable dataset (train + test) before
  splitting. This is intentional for scaling consistency across countries;
  all four models in the study (ARIMAX, SVR, XGBoost, LSTCN) follow the
  same convention.
- Tuning criterion: cv_MAE_scaled_y_mean (primary), aligned with ARIMAX,
  XGBoost, and LSTCN for fair cross-model comparison.
- No checkpoint is saved mid-run (unlike ARIMAX). Results are written once
  all countries have completed.

Output files
------------
  svr_metrics_summary.csv  : Best params + all metrics per country/feature_set
  svr_predictions_all.csv  : Year-by-year actual vs predicted (original scale)
  svr_tuning_results.csv   : CV scores for every hyperparameter combination
"""

import warnings
import os                                                          
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    
import pandas as pd
import numpy as np

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


# =============================================================================
# 1. Configuration
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Path to the preprocessed dataset (annual panel data, one row per country-year)
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed", "dataset_recursive_3yr_avg_drop_Industry.csv")

# Column name of the prediction target
TARGET = "Total CO2 emissions"

# Name of the lag feature created inside prepare_country_data().
# This column holds the previous year's CO2 value and is appended to every
# feature set so the model has at least one period of autoregressive signal.
# ARIMAX does not need this because AR(p) models past CO2 directly through
# the endogenous variable. LSTCN does not need this because it receives the
# full [X | y] row from the previous time step as its input.
LAG_TARGET_NAME = "CO2_emissions_lag1"

# Countries to model. Each is run independently with its own scaler and tuning.
COUNTRIES = [
    "Canada",
    "China",
    "India",
    "Indonesia",
    "Russian Federation",
    "United States",
]

# Train/test temporal split: years up to and including 2016 are used for
# training; 2017 onward is the held-out test set.
# Identical across all four models (ARIMAX, SVR, XGBoost, LSTCN).
TRAIN_END_YEAR  = 2016
TEST_START_YEAR = 2017

# These ratio constants are defined for reference but are not actively used
# in the splitting logic (which is year-based, not ratio-based).
TRAIN_RATIO = 0.8
TEST_RATIO  = 0.2

# MinMaxScaler target range. Using (-0.9, 0.9) instead of (-1, 1) provides
# a small margin that prevents exact boundary values, which can destabilise
# some optimisers. All four models (ARIMAX, SVR, XGBoost, LSTCN) use the
# same range for consistency.
SCALE_RANGE = (-0.9, 0.9)

# ── Feature groups ─────────────────────────────────────────────────────────
# BASE_FEATURES: socioeconomic and energy-related exogenous variables.
# Identical across all four models.
BASE_FEATURES = [
    "Population",
    "GDP",
    "Electric power consumption",
    "Fossil fuel energy consumption",
    "Renewable energy consumption",
    "Fertilizer consumption",
]

# TEMP_FEATURES: climate variables added in the augmented experiment.
# Identical across all four models.
TEMP_FEATURES = [
    "Temperature annual mean",
    "Temperature std across months",
    "Number of frost days",
    "Number of hot days",
]

# Two experiments are run per country:
#   "baseline"  - socioeconomic/energy features + CO2 lag
#   "augmented" - adds climate features on top of baseline, still with CO2 lag
#
# NOTE: LAG_TARGET_NAME is appended to both sets because SVR has no built-in
# temporal memory. XGBoost uses the same approach for the same reason.
# ARIMAX and LSTCN do NOT include LAG_TARGET_NAME:
#   ARIMAX  - models past CO2 via the AR(p) endogenous component
#   LSTCN   - receives it implicitly through the [X | y] input structure
FEATURE_SETS = {
    "baseline":  BASE_FEATURES + [LAG_TARGET_NAME],
    "augmented": BASE_FEATURES + TEMP_FEATURES + [LAG_TARGET_NAME],
}

# ── SVR hyperparameter search grid ─────────────────────────────────────────
# kernel  : the kernel function that maps inputs into a higher-dimensional space
#   "linear" - standard dot product; good for approximately linear relationships
#   "rbf"    - Radial Basis Function (Gaussian); most common choice for
#              nonlinear data; scale of influence controlled by gamma
#   "poly"   - polynomial kernel; complexity controlled by degree and gamma
#
# C       : regularisation strength; larger C = narrower margin = less tolerance
#           for errors inside the tube (higher risk of overfitting on small data)
# epsilon : half-width of the insensitive tube; predictions within +/-epsilon of
#           the true value incur zero loss, making SVR less sensitive to small
#           residuals and preventing overfitting to noise
# gamma   : kernel coefficient for "rbf" and "poly"
#   "scale" = 1 / (n_features * Var(X))  - recommended default
#   "auto"  = 1 / n_features
#   float   - manual value
# degree  : polynomial degree (only for "poly" kernel; ignored otherwise)
#
# Total combinations (non-uniform due to kernel-specific parameters):
#   linear : 5 (C) x 4 (epsilon)                          =  20
#   rbf    : 5 (C) x 4 (epsilon) x 4 (gamma)              =  80
#   poly   : 5 (C) x 4 (epsilon) x 4 (gamma) x 2 (degree) = 160
#   Total  : ~260 combinations
KERNEL_GRID  = ["linear", "rbf", "poly"]
C_GRID       = [0.1, 1, 10, 100, 500]
EPSILON_GRID = [0.001, 0.01, 0.1, 0.2]
GAMMA_GRID   = ["scale", "auto", 0.1, 1]
DEGREE_GRID  = [2, 3]   # only used when kernel == "poly"

# Primary metric used to rank hyperparameter combinations during CV tuning.
# Using the scaled-space MAE makes cross-country comparison consistent and
# aligns with the criterion used in ARIMAX, XGBoost, and LSTCN.
TUNING_PRIMARY_METRIC = "MAE_scaled_y"

# ── Output file names ───────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
METRICS_OUT = os.path.join(RESULTS_DIR, "metrics",     "svr_metrics_summary.csv")
PRED_OUT    = os.path.join(RESULTS_DIR, "predictions", "svr_predictions_all.csv")
TUNING_OUT  = os.path.join(RESULTS_DIR, "tuning",      "svr_tuning_results.csv")


# =============================================================================
# 2. Utility / helper functions
# =============================================================================

def compute_metrics(y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled):
    """
    Compute a standard set of regression metrics in both the original scale
    and the scaled ([-0.9, 0.9]) space.

    This function is identical across all four models (ARIMAX, SVR, XGBoost,
    LSTCN) to guarantee that metric definitions are consistent when comparing
    model results.

    Parameters
    ----------
    y_true_raw    : array-like - ground-truth values in the original CO2 unit
    y_pred_raw    : array-like - predicted values in the original CO2 unit
    y_true_scaled : array-like - ground-truth values after MinMax scaling
    y_pred_scaled : array-like - predicted values after MinMax scaling

    Returns
    -------
    dict with keys:
        RMSE_absolute  - RMSE in original units
        RMSE_scaled_y  - RMSE in scaled space (used for cross-country comparison)
        RMSE_relative  - RMSE normalised by the mean of y_true_raw
        MAE_absolute   - MAE in original units
        MAE_scaled_y   - MAE in scaled space (primary tuning criterion)
        MAE_relative   - MAE normalised by the mean of y_true_raw
        MAPE           - Mean Absolute Percentage Error (0 to inf; lower is better)
        Accuracy       - 1 - MAPE (can be negative if MAPE > 1)
        R_squared      - Coefficient of determination on the original scale

    Note on CV usage
    ----------------
    When called inside evaluate_param_set_cv(), y_true_raw == y_true_scaled
    and y_pred_raw == y_pred_scaled (because the CV fold data is already in
    scaled space). In that case MAE_absolute and MAE_scaled_y are numerically
    identical; only MAE_scaled_y is used for tuning decisions.
    MinMaxScaler is a linear transformation, so scaled and raw rankings are
    always equivalent within a single country -- the best hyperparameter set
    selected by MAE_scaled_y would be the same as the one selected by
    MAE_absolute.
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
    accuracy = 1 - mape     # warning: can be negative when MAPE > 1
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
        "R_squared":     r2,
    }


def build_svr_model(kernel, C, epsilon, gamma=None, degree=None):
    """
    Instantiate an SVR model from the given hyperparameters.

    Centralising model construction here means the same parameter-passing
    logic is reused in both the CV tuning loop and the final rolling forecast,
    avoiding duplication and potential inconsistency.

    Parameters
    ----------
    kernel  : str - "linear", "rbf", or "poly"
    C       : float - regularisation parameter
    epsilon : float - insensitive tube half-width
    gamma   : str or float or None - kernel coefficient; ignored for "linear"
    degree  : int or None - polynomial degree; only used when kernel == "poly"

    Returns
    -------
    sklearn SVR instance (not yet fitted)
    """
    params = {
        "kernel":  kernel,
        "C":       C,
        "epsilon": epsilon,
    }

    # gamma is only meaningful for kernels that use a distance function
    if kernel in {"rbf", "poly"}:
        params["gamma"] = gamma

    # degree is only meaningful for the polynomial kernel
    if kernel == "poly":
        params["degree"] = degree

    return SVR(**params)


def iter_param_grid():
    """
    Generate all valid hyperparameter combinations as a dictionary iterator.

    The grid is non-uniform because some parameters only apply to specific
    kernels:
      - "linear" : no gamma, no degree
      - "rbf"    : gamma required, no degree
      - "poly"   : gamma and degree both required

    Using np.nan as a placeholder for inapplicable parameters keeps the
    downstream DataFrame schema consistent (every row has the same columns).

    Yields
    ------
    dict with keys: kernel, C, epsilon, gamma, degree
    """
    for kernel in KERNEL_GRID:
        for C in C_GRID:
            for epsilon in EPSILON_GRID:
                if kernel == "linear":
                    yield {
                        "kernel":  kernel,
                        "C":       C,
                        "epsilon": epsilon,
                        "gamma":   np.nan,   # not applicable for linear kernel
                        "degree":  np.nan,   # not applicable for linear kernel
                    }
                elif kernel == "rbf":
                    for gamma in GAMMA_GRID:
                        yield {
                            "kernel":  kernel,
                            "C":       C,
                            "epsilon": epsilon,
                            "gamma":   gamma,
                            "degree":  np.nan,  # not applicable for rbf kernel
                        }
                elif kernel == "poly":
                    for gamma in GAMMA_GRID:
                        for degree in DEGREE_GRID:
                            yield {
                                "kernel":  kernel,
                                "C":       C,
                                "epsilon": epsilon,
                                "gamma":   gamma,
                                "degree":  degree,
                            }


def evaluate_param_set_cv(X_train, y_train, params):
    """
    Evaluate one hyperparameter combination via rolling one-step-ahead
    time-series cross-validation on the training data.

    Why rolling one-step-ahead inside CV?
    --------------------------------------
    The final test evaluation uses rolling one-step-ahead (re-fitting on
    expanding history before each prediction). CV must mirror this exactly so
    that the hyperparameter ranking reflects true out-of-sample behaviour.
    A static multi-step forecast inside CV would create a mismatch and could
    select the wrong hyperparameters.
    This design is shared across all four models (ARIMAX, SVR, XGBoost, LSTCN).

    CV structure (shared across all four models)
    ---------------------------------------------
    TimeSeriesSplit ensures that validation years always come after training
    years within each fold (no temporal leakage).

        Fold 1: train [t0 ... t_k]      val [t_{k+1} ... t_{k+m}]
        Fold 2: train [t0 ... t_{k+m}]  val [t_{k+m+1} ... ]
        ...

    Inside each fold, for every validation step j:
      1. Re-fit SVR on history (initially = fold train, grows each step)
      2. Predict one step ahead
      3. Append the TRUE observation to history (not the forecast)

    Parameters
    ----------
    X_train : 2-D array - scaled features including CO2 lag, shape (T_train, p)
    y_train : 1-D array - scaled CO2 values, shape (T_train,)
    params  : dict with keys kernel, C, epsilon, gamma, degree

    Returns
    -------
    dict with keys:
        valid            - bool; False if all folds failed
        reason           - str; failure message if valid is False
        n_splits         - int; actual number of CV folds used
        n_success_folds  - int
        n_failed_folds   - int
        cv_<metric>_mean - float; mean of each metric across successful folds
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1)

    # Minimum data guard: shared across all four models.
    # Below 6 training points we cannot form a meaningful CV split.
    if len(X_train) < 6:
        return {
            "valid":           False,
            "reason":          "Not enough training rows for hyperparameter tuning.",
            "n_splits":        np.nan,
            "n_success_folds": 0,
            "n_failed_folds":  0,
        }

    # Adapt number of folds to available training size.
    # Formula is identical across all four models.
    n_splits = min(5, max(2, len(X_train) // 4))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_rows       = []
    n_success_folds = 0
    n_failed_folds  = 0

    for train_idx, val_idx in tscv.split(X_train):
        X_tr  = X_train[train_idx]
        y_tr  = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        # Initialise the expanding history with the fold's training portion.
        # Unlike ARIMAX, SVR has no minimum-sample-per-order requirement,
        # so no additional guard is needed here.
        history_X_cv = X_tr.copy()
        history_y_cv = y_tr.copy()
        preds_val    = []
        fold_failed  = False

        # Rolling one-step-ahead loop over the validation window.
        # This inner loop structure is identical across SVR, XGBoost, and LSTCN.
        # ARIMAX uses the same logic but wraps the fit in safe_fit_arimax()
        # to handle MLE convergence failures gracefully.
        for j in range(len(X_val)):
            try:
                # Re-fit SVR on the full current history at each step.
                # This is computationally expensive but necessary to correctly
                # simulate the test-time procedure.
                model = build_svr_model(
                    kernel=params["kernel"],
                    C=params["C"],
                    epsilon=params["epsilon"],
                    gamma=None  if pd.isna(params["gamma"])  else params["gamma"],
                    degree=None if pd.isna(params["degree"]) else int(params["degree"]),
                )
                model.fit(history_X_cv, history_y_cv)
                yhat = float(model.predict(X_val[j].reshape(1, -1))[0])
                preds_val.append(yhat)

                # Expand history with the TRUE observation, not the forecast.
                # This matches real-world usage where the actual value becomes
                # available at each period end before the next forecast.
                # Identical behaviour in ARIMAX, XGBoost, and LSTCN.
                history_X_cv = np.vstack([history_X_cv, X_val[j]])
                history_y_cv = np.append(history_y_cv, y_val[j])
            except Exception:
                fold_failed = True
                break

        if fold_failed or len(preds_val) != len(X_val):
            n_failed_folds += 1
            continue

        pred_val = np.array(preds_val)

        # Compute metrics for this fold.
        # Both raw and scaled arguments receive the same scaled values because
        # all CV data is already in the scaled space. MAE_absolute and
        # MAE_scaled_y are therefore numerically identical here; only
        # MAE_scaled_y is used downstream for ranking.
        fold_metrics = compute_metrics(
            y_true_raw=y_val,
            y_pred_raw=pred_val,
            y_true_scaled=y_val,
            y_pred_scaled=pred_val,
        )
        fold_rows.append(fold_metrics)
        n_success_folds += 1

    if n_success_folds == 0:
        return {
            "valid":           False,
            "reason":          "All folds failed.",
            "n_splits":        n_splits,
            "n_success_folds": 0,
            "n_failed_folds":  n_failed_folds,
        }

    # Aggregate metrics across successful folds (simple mean).
    # Identical aggregation logic across all four models.
    fold_df = pd.DataFrame(fold_rows)
    out = {
        "valid":           True,
        "reason":          "",
        "n_splits":        n_splits,
        "n_success_folds": n_success_folds,
        "n_failed_folds":  n_failed_folds,
    }
    for col in fold_df.columns:
        out[f"cv_{col}_mean"] = fold_df[col].mean()

    return out


def tune_svr_hyperparams(X_train, y_train, country_name, feature_set_name):
    """
    Grid-search over all valid (kernel, C, epsilon, gamma, degree) combinations
    and select the best one by time-series cross-validation.

    Ranking priority (all ascending, lower is better):
        1. cv_MAE_scaled_y_mean   - primary criterion (aligned with all four models)
        2. cv_RMSE_scaled_y_mean  - secondary; penalises large errors more heavily
        3. cv_MAPE_mean           - tertiary; scale-free percentage error
        4. kernel, C, epsilon     - tiebreaker: prefer simpler / smaller values

    Parameters
    ----------
    X_train          : 2-D array - scaled features (training period only)
    y_train          : 1-D array - scaled CO2 (training period only)
    country_name     : str - used for labelling tuning records
    feature_set_name : str - "baseline" or "augmented"

    Returns
    -------
    best_params : dict with keys best_kernel, best_C, best_epsilon,
                  best_gamma, best_degree
    tuning_df   : DataFrame with one row per combination, containing
                  CV metrics and validity flags
    """
    tuning_rows = []

    for params in iter_param_grid():
        cv_result = evaluate_param_set_cv(X_train, y_train, params)

        # Record every combination, including failures, for diagnostics
        row = {
            "country":         country_name,
            "feature_set":     feature_set_name,
            "kernel":          params["kernel"],
            "C":               params["C"],
            "epsilon":         params["epsilon"],
            "gamma":           params["gamma"],
            "degree":          params["degree"],
            "valid":           cv_result.get("valid",           False),
            "reason":          cv_result.get("reason",          ""),
            "n_splits":        cv_result.get("n_splits",        np.nan),
            "n_success_folds": cv_result.get("n_success_folds", 0),
            "n_failed_folds":  cv_result.get("n_failed_folds",  0),
            "cv_RMSE_absolute_mean": cv_result.get("cv_RMSE_absolute_mean", np.nan),
            "cv_RMSE_scaled_y_mean": cv_result.get("cv_RMSE_scaled_y_mean", np.nan),
            "cv_RMSE_relative_mean": cv_result.get("cv_RMSE_relative_mean", np.nan),
            "cv_MAE_absolute_mean":  cv_result.get("cv_MAE_absolute_mean",  np.nan),
            "cv_MAE_scaled_y_mean":  cv_result.get("cv_MAE_scaled_y_mean",  np.nan),
            "cv_MAE_relative_mean":  cv_result.get("cv_MAE_relative_mean",  np.nan),
            "cv_MAPE_mean":          cv_result.get("cv_MAPE_mean",           np.nan),
            "cv_Accuracy_mean":      cv_result.get("cv_Accuracy_mean",       np.nan),
            "cv_R_squared_mean":     cv_result.get("cv_R_squared_mean",      np.nan),
        }
        tuning_rows.append(row)

    tuning_df = pd.DataFrame(tuning_rows)

    # Keep only combinations that produced at least one valid CV score
    valid_df = tuning_df[
        tuning_df["valid"].fillna(False)
        & tuning_df["cv_MAE_scaled_y_mean"].notna()
    ].copy()

    if valid_df.empty:
        raise RuntimeError(
            f"{country_name} | {feature_set_name}: no valid SVR parameter set found."
        )

    # Sort by the ranking priority described in the docstring.
    # Identical ranking logic across all four models.
    valid_df = valid_df.sort_values(
        by=[
            "cv_MAE_scaled_y_mean",
            "cv_RMSE_scaled_y_mean",
            "cv_MAPE_mean",
            "kernel", "C", "epsilon",   # tiebreaker: prefer simpler values
        ],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)

    best_row = valid_df.iloc[0]
    best_params = {
        "best_kernel":  best_row["kernel"],
        "best_C":       float(best_row["C"]),
        "best_epsilon": float(best_row["epsilon"]),
        "best_gamma":   best_row["gamma"],
        "best_degree":  best_row["degree"],
    }

    return best_params, tuning_df


def prepare_country_data(data, country_name, feature_list):
    """
    Filter the panel dataset to a single country, create the one-period CO2
    lag feature, and drop any rows with missing values.

    Why lag is created here (not in main):
    The lag must be computed on the full country time series before any split,
    so that the lag value for the first test year correctly reflects the last
    training year's CO2 value. Computing lag after splitting would produce
    a NaN for the first test row.

    The lag is created before dropna() so that the first year -- which has
    no lag value -- is cleanly removed. This means SVR and XGBoost datasets
    start one year later than ARIMAX and LSTCN (which do not use a lag feature).

    Parameters
    ----------
    data         : DataFrame - full multi-country panel
    country_name : str
    feature_list : list[str] - includes LAG_TARGET_NAME

    Returns
    -------
    df_use : DataFrame with columns [Year, TARGET] + feature_list, no NaNs
    """
    cdf = data[data["Country Name"] == country_name].sort_values("Year").copy()

    # Create lag feature: CO2_emissions_lag1[t] = TARGET[t-1]
    # shift(1) moves values down by one row, so row t contains t-1's value.
    # The first row becomes NaN and is removed by the subsequent dropna().
    cdf[LAG_TARGET_NAME] = cdf[TARGET].shift(1)

    cols   = ["Year", TARGET] + feature_list
    df_use = cdf[cols].dropna().copy()

    if len(df_use) == 0:
        raise ValueError(
            f"{country_name}: no usable rows after lag creation and dropna()."
        )

    return df_use


def run_svr_single_country(data, country_name, feature_set_name, feature_list):
    """
    Full pipeline for a single country x feature_set combination:
      1. Extract country data and create the CO2 lag feature
      2. Scale features and target on the full usable dataset
      3. Split into train / test
      4. Tune hyperparameters via CV on training data only
      5. Rolling one-step-ahead forecast on the test set
      6. Inverse-transform and evaluate

    Parameters
    ----------
    data             : DataFrame - the full multi-country panel dataset
    country_name     : str
    feature_set_name : str - "baseline" or "augmented"
    feature_list     : list[str] - feature column names for this experiment
                       (already includes LAG_TARGET_NAME)

    Returns
    -------
    metrics    : dict - test-set evaluation metrics
    pred_df    : DataFrame - year / actual / predicted (original scale)
    best_params: dict - best (kernel, C, epsilon, gamma, degree) found by CV
    tuning_df  : DataFrame - full CV tuning log
    """
    # ── Step 1: Filter, create lag, drop NaN ──────────────────────────────
    # prepare_country_data handles lag creation and safety dropna().
    # dropna() here acts as a safety guard; the upstream dataset is assumed
    # to be already preprocessed.
    df_use = prepare_country_data(data, country_name, feature_list)

    # ── Step 2: Scale on the FULL usable dataset (before train/test split) ─
    # Both scalers are fit on all years, including the test period. This is a
    # deliberate design choice shared across all four models (ARIMAX, SVR,
    # XGBoost, LSTCN): it ensures the target's min/max used for
    # inverse-transforming test predictions is stable and consistent.
    # In a strict production setting one would fit the scaler on training
    # data only to avoid any information leakage from test years.
    sc_x = MinMaxScaler(feature_range=SCALE_RANGE)
    sc_y = MinMaxScaler(feature_range=SCALE_RANGE)

    df_scaled = df_use.copy()
    df_scaled[feature_list] = sc_x.fit_transform(df_use[feature_list].values)
    df_scaled[[TARGET]]     = sc_y.fit_transform(df_use[[TARGET]].values)

    # ── Step 3: Train / test split (year-based, after scaling) ─────────────
    # Splitting after scaling is consistent across all four models.
    train_df = df_scaled[df_scaled["Year"] <= TRAIN_END_YEAR].copy()
    test_df  = df_scaled[df_scaled["Year"] >= TEST_START_YEAR].copy()

    if len(train_df) < 6:
        raise ValueError(
            f"{country_name}: training rows too few ({len(train_df)})."
        )
    if len(test_df) == 0:
        raise ValueError(f"{country_name}: no test rows after split.")

    X_train       = train_df[feature_list].values
    y_train       = train_df[TARGET].values.reshape(-1)
    X_test        = test_df[feature_list].values
    y_test_scaled = test_df[TARGET].values.reshape(-1)
    years_test    = test_df["Year"].values

    # y_test_raw: true CO2 in the ORIGINAL (unscaled) unit.
    # Used for absolute metrics (MAE_absolute, MAPE, R²) at evaluation time.
    # Identical pattern across all four models.
    y_test_raw = df_use[
        df_use["Year"] >= TEST_START_YEAR
    ][TARGET].values.reshape(-1)

    # ── Step 4: Hyperparameter tuning (train set only) ─────────────────────
    best_params, tuning_df = tune_svr_hyperparams(
        X_train=X_train,
        y_train=y_train,
        country_name=country_name,
        feature_set_name=feature_set_name,
    )

    # ── Step 5: Rolling one-step-ahead forecast on the test set ────────────
    # Start history with the complete training set.
    # After each prediction the TRUE observation is appended (expanding window).
    # This mirrors the real-world scenario where actual data is observed at
    # the end of each year before the next forecast is made.
    # Identical expanding-window logic across all four models.
    history_X    = X_train.copy()
    history_y    = y_train.copy()
    preds_scaled = []

    for i in range(len(X_test)):
        # Re-fit SVR on all available history up to this point.
        # SVR is a non-parametric kernel method -- it stores training points
        # as support vectors -- so re-fitting from scratch is the only way
        # to incorporate the latest observation into the model.
        model = build_svr_model(
            kernel=best_params["best_kernel"],
            C=best_params["best_C"],
            epsilon=best_params["best_epsilon"],
            gamma=None  if pd.isna(best_params["best_gamma"])  else best_params["best_gamma"],
            degree=None if pd.isna(best_params["best_degree"]) else int(best_params["best_degree"]),
        )
        model.fit(history_X, history_y)

        # Forecast one step ahead using the next test year's features
        x_next      = X_test[i].reshape(1, -1)
        yhat_scaled = float(model.predict(x_next)[0])
        preds_scaled.append(yhat_scaled)

        # Expand history: use the TRUE scaled y (not the forecast yhat).
        # Identical update rule in ARIMAX, XGBoost, and LSTCN.
        history_X = np.vstack([history_X, X_test[i]])
        history_y = np.append(history_y, y_test_scaled[i])

    # ── Step 6: Inverse-transform and evaluate ─────────────────────────────
    preds_scaled  = np.asarray(preds_scaled).reshape(-1, 1)
    y_pred_raw    = sc_y.inverse_transform(preds_scaled).reshape(-1)
    y_pred_scaled = preds_scaled.reshape(-1)

    metrics = compute_metrics(
        y_true_raw=y_test_raw,       # original scale - for absolute/MAPE/R²
        y_pred_raw=y_pred_raw,       # inverse-transformed predictions
        y_true_scaled=y_test_scaled, # scaled space - for MAE_scaled_y / RMSE_scaled_y
        y_pred_scaled=y_pred_scaled, # predictions still in scaled space
    )

    # Build a tidy predictions DataFrame for later export
    pred_df = pd.DataFrame({
        "country":     country_name,
        "feature_set": feature_set_name,
        "year":        years_test,
        "actual":      y_test_raw,
        "predicted":   y_pred_raw,
    })

    return metrics, pred_df, best_params, tuning_df


def make_average_rows(metrics_df):
    """
    Append summary rows to the metrics table for convenient reporting:
      - BASELINE_AVG  : mean across all countries for the baseline feature set
      - AUGMENTED_AVG : mean across all countries for the augmented feature set
      - OVERALL_AVG   : mean across all country x feature_set combinations

    This function is structurally identical across all four models (ARIMAX,
    SVR, XGBoost, LSTCN). The only difference is which hyperparameter
    columns are set to NaN in the summary rows (reflecting each model's
    own parameter names).

    Parameters
    ----------
    metrics_df : DataFrame - one row per country x feature_set

    Returns
    -------
    DataFrame with three summary rows
    """
    metric_cols = [
        "RMSE_absolute", "RMSE_scaled_y", "RMSE_relative",
        "MAE_absolute",  "MAE_scaled_y",  "MAE_relative",
        "MAPE", "Accuracy", "R_squared",
    ]

    rows = []

    # SVR-specific hyperparameter columns filled with NaN in summary rows
    svr_param_nans = {
        "best_kernel":  np.nan,
        "best_C":       np.nan,
        "best_epsilon": np.nan,
        "best_gamma":   np.nan,
        "best_degree":  np.nan,
    }

    baseline_df  = metrics_df[metrics_df["feature_set"] == "baseline"].copy()
    baseline_avg = {"country": "BASELINE_AVG", "feature_set": "baseline", **svr_param_nans}
    for col in metric_cols:
        baseline_avg[col] = baseline_df[col].mean()
    rows.append(baseline_avg)

    augmented_df  = metrics_df[metrics_df["feature_set"] == "augmented"].copy()
    augmented_avg = {"country": "AUGMENTED_AVG", "feature_set": "augmented", **svr_param_nans}
    for col in metric_cols:
        augmented_avg[col] = augmented_df[col].mean()
    rows.append(augmented_avg)

    overall_avg = {"country": "OVERALL_AVG", "feature_set": "all", **svr_param_nans}
    for col in metric_cols:
        overall_avg[col] = metrics_df[col].mean()
    rows.append(overall_avg)

    return pd.DataFrame(rows)


# =============================================================================
# 3. Main entry point
# =============================================================================

def main():
    """
    Orchestrate the full experiment:
      - Load the panel dataset
      - For each country x feature_set: run the complete SVR pipeline
      - Write all three output CSVs when all countries are done

    Unlike the ARIMAX script, no checkpoint is saved after each country.
    If the run is interrupted, the entire script must be re-run from scratch.
    Progress is printed to stdout so long-running jobs can be monitored.
    """
    os.makedirs(os.path.join(RESULTS_DIR, "metrics"),     exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "tuning"),      exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    # Accumulators for results across all countries
    metrics_rows  = []
    pred_frames   = []
    tuning_frames = []

    for country in COUNTRIES:
        print(f"\n===== Running {country} =====")

        for feature_set_name, feature_list in FEATURE_SETS.items():
            print(f"  -> feature set: {feature_set_name}")
            try:
                metrics, pred_df, best_params, tuning_df = run_svr_single_country(
                    data=data,
                    country_name=country,
                    feature_set_name=feature_set_name,
                    feature_list=feature_list,
                )

                # Merge best_params and metrics into a single flat dict for
                # the metrics summary CSV row
                row = {
                    "country":     country,
                    "feature_set": feature_set_name,
                    **best_params,
                    **metrics,
                }
                metrics_rows.append(row)
                pred_frames.append(pred_df)
                tuning_frames.append(tuning_df)

                print(
                    f"     best_kernel={best_params['best_kernel']}, "
                    f"best_C={best_params['best_C']}, "
                    f"best_epsilon={best_params['best_epsilon']}, "
                    f"best_gamma={best_params['best_gamma']}, "
                    f"best_degree={best_params['best_degree']}, "
                    f"MAE_scaled_y={metrics['MAE_scaled_y']:.6f}, "
                    f"RMSE_scaled_y={metrics['RMSE_scaled_y']:.6f}, "
                    f"Accuracy={metrics['Accuracy']:.6f}"
                )
            except Exception as e:
                print(f"     ERROR: {e}")

    if not metrics_rows:
        raise RuntimeError(
            "No successful SVR runs. "
            "Please check data, columns, and package versions."
        )

    # ── Build and write metrics CSV ────────────────────────────────────────
    metrics_df    = pd.DataFrame(metrics_rows)
    avg_df        = make_average_rows(metrics_df)
    metrics_final = pd.concat([metrics_df, avg_df], ignore_index=True)
    metrics_final = metrics_final[[
        "country", "feature_set",
        "best_kernel", "best_C", "best_epsilon", "best_gamma", "best_degree",
        "RMSE_absolute", "RMSE_scaled_y", "RMSE_relative",
        "MAE_absolute",  "MAE_scaled_y",  "MAE_relative",
        "MAPE", "Accuracy", "R_squared",
    ]]

    # ── Build and write predictions CSV ────────────────────────────────────
    preds_final = pd.concat(pred_frames, ignore_index=True)
    preds_final = preds_final[
        ["country", "feature_set", "year", "actual", "predicted"]
    ].sort_values(["country", "feature_set", "year"])

    # ── Build and write tuning log CSV ─────────────────────────────────────
    # Sorted by cv_MAE_scaled_y_mean so the best combinations appear first,
    # consistent with the sorting convention in ARIMAX, XGBoost, and LSTCN.
    tuning_final = pd.concat(tuning_frames, ignore_index=True)
    tuning_final = tuning_final[[
        "country", "feature_set",
        "kernel", "C", "epsilon", "gamma", "degree",
        "valid", "reason",
        "n_splits", "n_success_folds", "n_failed_folds",
        "cv_RMSE_absolute_mean", "cv_RMSE_scaled_y_mean", "cv_RMSE_relative_mean",
        "cv_MAE_absolute_mean",  "cv_MAE_scaled_y_mean",  "cv_MAE_relative_mean",
        "cv_MAPE_mean", "cv_Accuracy_mean", "cv_R_squared_mean",
    ]].sort_values(
        ["country", "feature_set", "cv_MAE_scaled_y_mean", "kernel", "C", "epsilon"],
        ascending=[True, True, True, True, True, True],
    )

    metrics_final.to_csv(METRICS_OUT, index=False)
    preds_final.to_csv(PRED_OUT,    index=False)
    tuning_final.to_csv(TUNING_OUT, index=False)

    print("\nSaved files:")
    print(f"  - {METRICS_OUT}")
    print(f"  - {PRED_OUT}")
    print(f"  - {TUNING_OUT}")

    print("\nMetrics preview:")
    print(metrics_final.round(6))


if __name__ == "__main__":
    main()
