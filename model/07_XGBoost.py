"""
XGBoost Rolling One-Step-Ahead Forecast for CO2 Emissions
==========================================================
This script trains an XGBoost regressor (XGBRegressor) per country to forecast
annual total CO2 emissions, using an expanding-window rolling evaluation
that mirrors real-world sequential prediction.

Pipeline overview
-----------------
For each country x feature_set combination:
  1. Filter rows for the target country, sort by year, and select the
     required columns. dropna() is applied as a safety guard; the upstream
     dataset is assumed to be already preprocessed.
  2. Construct a one-period lag of the target (CO2_lag1) as an autoregressive
     feature, then drop the first year which has no lag value.
  3. Scale all variables to [-0.9, 0.9] using the full usable dataset.
  4. Split into train (<=2016) and test (>=2017) after scaling.
  5. Tune XGBoost hyperparameters via time-series cross-validation on the
     training set only. CV uses rolling one-step-ahead evaluation to mirror
     the actual test procedure.
  6. Run a rolling one-step-ahead forecast on the test set:
       - Re-fit XGBoost on all available history before each step
       - Append the TRUE observed value to history after each step
         (expanding window)
  7. Inverse-transform predictions and compute evaluation metrics.

Key design notes
----------------
- Like SVR (and unlike ARIMAX and LSTCN), XGBoost has no built-in memory of
  past values. A one-period lag of the target (CO2_emissions_lag1) is added
  as an explicit feature so the model knows the previous year's CO2 level.
  Only a single lag is used; deeper autoregressive structure is not captured.
  ARIMAX models past CO2 via the AR(p) endogenous component.
  LSTCN receives it implicitly through the [X | y] sequential input structure.
- The scaler is fit on the full usable dataset (train + test) before
  splitting. This is intentional for scaling consistency across countries;
  all four models in the study (ARIMAX, SVR, XGBoost, LSTCN) follow the
  same convention.
- Tuning criterion: cv_MAE_scaled_y_mean (primary), aligned with ARIMAX,
  SVR, and LSTCN for fair cross-model comparison.
- No checkpoint is saved mid-run (unlike ARIMAX). Results are written once
  all countries have completed.

Output files
------------
  xgboost_metrics_summary_ver2.csv  : Best params + all metrics per country/feature_set
  xgboost_predictions_all_ver2.csv  : Year-by-year actual vs predicted (original scale)
  xgboost_tuning_results_ver2.csv   : CV scores for every hyperparameter combination
"""

import warnings
import os                                                          
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
import itertools
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

from xgboost import XGBRegressor

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

# Name of the lag feature constructed before scaling.
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
    "United States"
]

# Train/test temporal split: years up to and including 2016 are used for
# training; 2017 onward is the held-out test set.
# Identical across all four models (ARIMAX, SVR, XGBoost, LSTCN).
TRAIN_END_YEAR  = 2016
TEST_START_YEAR = 2017

# MinMaxScaler target range. Using (-0.9, 0.9) instead of (-1, 1) provides
# a small margin that prevents exact boundary values, which can destabilise
# some optimisers. All four models (ARIMAX, SVR, XGBoost, LSTCN) use the
# same range for consistency.
TRAIN_SCALE_RANGE = (-0.9, 0.9)

# ── Feature groups ─────────────────────────────────────────────────────────
# BASE_FEATURES: socioeconomic and energy-related exogenous variables.
# Identical across all four models.
BASE_FEATURES = [
    "Population",
    "GDP",
    "Electric power consumption",
    "Fossil fuel energy consumption",
    "Renewable energy consumption",
    "Fertilizer consumption"
]

# TEMP_FEATURES: climate variables added in the augmented experiment.
# Identical across all four models.
TEMP_FEATURES = [
    "Temperature annual mean",
    "Temperature std across months",
    "Number of frost days",
    "Number of hot days"
]

# Two experiments are run per country:
#   "baseline"  - socioeconomic/energy features + CO2 lag
#   "augmented" - adds climate features on top of baseline, still with CO2 lag
#
# NOTE: LAG_TARGET_NAME is appended to both sets because XGBoost has no
# built-in temporal memory. SVR uses the same approach for the same reason.
# ARIMAX and LSTCN do NOT include LAG_TARGET_NAME:
#   ARIMAX - models past CO2 via the AR(p) endogenous component
#   LSTCN  - receives it implicitly through the [X | y] input structure
FEATURE_SETS = {
    "baseline":  BASE_FEATURES + [LAG_TARGET_NAME],
    "augmented": BASE_FEATURES + TEMP_FEATURES + [LAG_TARGET_NAME]
}

# ── XGBoost hyperparameter search grid ─────────────────────────────────────
# n_estimators    : number of boosting rounds (trees); more trees can improve
#                   fit but risk overfitting and increase compute cost
# max_depth       : maximum depth of each tree; fixed at 10 here to constrain
#                   model complexity given the small annual dataset size
# learning_rate   : shrinkage applied to each tree's contribution; lower values
#                   require more trees but often generalise better
# subsample       : fraction of training rows sampled per tree; < 1.0 adds
#                   randomness and reduces overfitting
# colsample_bytree: fraction of features sampled per tree; < 1.0 similarly
#                   adds regularisation
# reg_lambda      : L2 regularisation on leaf weights; larger values push
#                   weights toward zero (ridge-like)
# reg_alpha       : L1 regularisation on leaf weights; encourages sparsity
#                   (lasso-like)
# min_child_weight: minimum sum of instance weights in a leaf; higher values
#                   prevent the model from learning very specific patterns
# random_state    : fixed at 42 to ensure reproducibility across all runs
#
# Total combinations:
#   3 x 1 x 3 x 2 x 2 x 4 x 3 x 2 = 864 combinations
xgb_param_grid = {
    "n_estimators":     [50, 100, 150],
    "max_depth":        [10],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda":       [0, 1.0, 5.0, 10.0],
    "reg_alpha":        [0, 0.1, 1.0],
    "min_child_weight": [1, 3],
    "random_state":     [42]
}

# Primary metric used to rank hyperparameter combinations during CV tuning.
# Using the scaled-space MAE makes cross-country comparison consistent and
# aligns with the criterion used in ARIMAX, SVR, and LSTCN.
TUNING_PRIMARY_METRIC = "MAE_scaled_y"

# ── Output file names ───────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
METRICS_OUT = os.path.join(RESULTS_DIR, "metrics",     "xgboost_metrics_summary.csv")
PRED_OUT    = os.path.join(RESULTS_DIR, "predictions", "xgboost_predictions_all.csv")
TUNING_OUT  = os.path.join(RESULTS_DIR, "tuning",      "xgboost_tuning_results.csv")


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
        "R_squared":     r2
    }


def build_xgb_model(params):
    """
    Instantiate an XGBRegressor from the given hyperparameter dictionary.

    Centralising model construction here means the same parameter-passing
    logic is reused in both the CV tuning loop and the final rolling forecast,
    avoiding duplication and potential inconsistency.

    The objective is fixed to "reg:squarederror" (MSE loss) throughout;
    it is not part of the hyperparameter search because all regression
    targets in this study use the same loss.

    Parameters
    ----------
    params : dict - must contain the keys defined in xgb_param_grid plus
             any additional fields (e.g. random_state); extra keys are
             passed through transparently via **params

    Returns
    -------
    XGBRegressor instance (not yet fitted)
    """
    return XGBRegressor(
        objective="reg:squarederror",   # fixed MSE loss; not tuned
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        min_child_weight=params["min_child_weight"],
        random_state=params["random_state"],
        n_jobs=-1   # use all available CPU cores for faster tree building
    )


def generate_param_combinations(param_grid):
    """
    Expand a hyperparameter grid dictionary into a flat list of all
    combination dictionaries using itertools.product.

    This is the XGBoost equivalent of SVR's iter_param_grid(). Unlike SVR,
    XGBoost's grid is uniform (all parameters apply to every combination),
    so a simple Cartesian product suffices with no conditional branching.

    Parameters
    ----------
    param_grid : dict - keys are parameter names, values are lists of
                 candidate values

    Returns
    -------
    list[dict] - one dict per combination
    """
    keys   = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


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
      1. Re-fit XGBoost on history (initially = fold train, grows each step)
      2. Predict one step ahead
      3. Append the TRUE observation to history (not the forecast)

    Parameters
    ----------
    X_train : 2-D array - scaled features including CO2 lag, shape (T_train, p)
    y_train : 1-D array - scaled CO2 values, shape (T_train,)
    params  : dict - one combination from xgb_param_grid

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
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    # Minimum data guard: shared across all four models.
    # Below 6 training points we cannot form a meaningful CV split.
    if len(X_train) < 6:
        return {
            "valid":           False,
            "reason":          "Not enough training rows for hyperparameter tuning.",
            "n_splits":        np.nan,
            "n_success_folds": 0,
            "n_failed_folds":  0
        }

    # Adapt number of folds to available training size.
    # Formula is identical across all four models.
    n_splits = min(5, max(2, len(X_train) // 4))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metric_rows = []
    n_success_folds  = 0
    n_failed_folds   = 0

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Initialise the expanding history with the fold's training portion.
        # XGBoost, like SVR, has no minimum-sample-per-order requirement
        # (unlike ARIMAX), so no additional guard is needed here.
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
                # Re-fit XGBoost on the full current history at each step.
                # Because XGBoost builds an ensemble of trees from scratch,
                # re-fitting is necessary to incorporate the latest observation.
                model = build_xgb_model(params)
                model.fit(history_X_cv, history_y_cv)
                yhat = float(model.predict(X_val[j].reshape(1, -1))[0])
                preds_val.append(yhat)

                # Expand history with the TRUE observation, not the forecast.
                # This matches real-world usage where the actual value becomes
                # available at each period end before the next forecast.
                # Identical behaviour in ARIMAX, SVR, and LSTCN.
                history_X_cv = np.vstack([history_X_cv, X_val[j]])
                history_y_cv = np.append(history_y_cv, y_val[j])
            except Exception:
                fold_failed = True
                break

        if fold_failed or len(preds_val) != len(X_val):
            n_failed_folds += 1
            continue

        y_val_pred = np.array(preds_val)

        # Compute metrics for this fold.
        # Both raw and scaled arguments receive the same scaled values because
        # all CV data is already in the scaled space. MAE_absolute and
        # MAE_scaled_y are therefore numerically identical here; only
        # MAE_scaled_y is used downstream for ranking.
        metrics_fold = compute_metrics(
            y_true_raw=y_val,
            y_pred_raw=y_val_pred,
            y_true_scaled=y_val,
            y_pred_scaled=y_val_pred
        )
        fold_metric_rows.append(metrics_fold)
        n_success_folds += 1

    if n_success_folds == 0:
        return {
            "valid":           False,
            "reason":          "All folds failed.",
            "n_splits":        n_splits,
            "n_success_folds": 0,
            "n_failed_folds":  n_failed_folds
        }

    # Aggregate metrics across successful folds (simple mean).
    # Identical aggregation logic across all four models.
    fold_df = pd.DataFrame(fold_metric_rows)
    out = {
        "valid":           True,
        "reason":          "",
        "n_splits":        n_splits,
        "n_success_folds": n_success_folds,
        "n_failed_folds":  n_failed_folds
    }
    for col in fold_df.columns:
        out[f"cv_{col}_mean"] = fold_df[col].mean()

    return out


def tune_xgboost_hyperparams(X_train, y_train, country_name, feature_set_name):
    """
    Grid-search over all 864 hyperparameter combinations and select the best
    one by time-series cross-validation.

    Ranking priority (all ascending, lower is better):
        1. cv_MAE_scaled_y_mean   - primary criterion (aligned with all four models)
        2. cv_RMSE_scaled_y_mean  - secondary; penalises large errors more heavily
        3. cv_MAPE_mean           - tertiary; scale-free percentage error
        4. n_estimators, max_depth, learning_rate  - tiebreaker: prefer simpler models

    Parameters
    ----------
    X_train          : 2-D array - scaled features (training period only)
    y_train          : 1-D array - scaled CO2 (training period only)
    country_name     : str - used for labelling tuning records
    feature_set_name : str - "baseline" or "augmented"

    Returns
    -------
    best_params : dict with keys best_n_estimators, best_max_depth,
                  best_learning_rate, best_subsample, best_colsample_bytree,
                  best_reg_lambda, best_reg_alpha, best_min_child_weight,
                  best_random_state
    tuning_df   : DataFrame with one row per combination, containing
                  CV metrics and validity flags
    """
    tuning_rows   = []
    param_combos  = generate_param_combinations(xgb_param_grid)

    for params in param_combos:
        cv_result = evaluate_param_set_cv(
            X_train=X_train,
            y_train=y_train,
            params=params
        )

        # Record every combination, including failures, for diagnostics
        row = {
            "country":     country_name,
            "feature_set": feature_set_name,
            **params,                              # unpack all grid params as columns
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
            "cv_R_squared_mean":     cv_result.get("cv_R_squared_mean",      np.nan)
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
            f"{country_name} | {feature_set_name}: no valid XGBoost parameter set found."
        )

    # Sort by the ranking priority described in the docstring.
    # Identical ranking logic across all four models.
    valid_df = valid_df.sort_values(
        by=[
            "cv_MAE_scaled_y_mean",
            "cv_RMSE_scaled_y_mean",
            "cv_MAPE_mean",
            "n_estimators", "max_depth", "learning_rate"  # tiebreaker
        ],
        ascending=[True, True, True, True, True, True]
    ).reset_index(drop=True)

    best_row = valid_df.iloc[0].to_dict()

    best_params = {
        "best_n_estimators":     int(best_row["n_estimators"]),
        # max_depth stored as NaN-safe cast: handles edge cases where the
        # value may be float NaN after DataFrame operations
        "best_max_depth":        None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
        "best_learning_rate":    float(best_row["learning_rate"]),
        "best_subsample":        float(best_row["subsample"]),
        "best_colsample_bytree": float(best_row["colsample_bytree"]),
        "best_reg_lambda":       float(best_row["reg_lambda"]),
        "best_reg_alpha":        float(best_row["reg_alpha"]),
        "best_min_child_weight": int(best_row["min_child_weight"]),
        "best_random_state":     int(best_row["random_state"])
    }

    return best_params, tuning_df


def run_xgboost_single_country(data, country_name, feature_set_name, feature_list):
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
    best_params: dict - best XGBoost hyperparameters found by CV
    tuning_df  : DataFrame - full CV tuning log
    """
    # ── Step 1: Filter, create lag, drop NaN ──────────────────────────────
    cdf = data[data["Country Name"] == country_name].sort_values("Year").copy()

    # Create lag feature: CO2_emissions_lag1[t] = TARGET[t-1].
    # shift(1) moves values down by one row, so row t contains t-1's value.
    # The first row becomes NaN and is removed by the subsequent dropna(),
    # meaning XGBoost and SVR datasets start one year later than ARIMAX and
    # LSTCN (which do not require a lag feature).
    cdf[LAG_TARGET_NAME] = cdf[TARGET].shift(1)

    # dropna() is a safety guard; the upstream dataset is assumed to be
    # already preprocessed.
    df_use = cdf[["Year", TARGET] + feature_list].dropna().copy()

    if len(df_use) == 0:
        raise ValueError(f"{country_name}: no usable rows after dropna().")

    # ── Step 2: Scale on the FULL usable dataset (before train/test split) ─
    # Both scalers are fit on all years, including the test period. This is a
    # deliberate design choice shared across all four models (ARIMAX, SVR,
    # XGBoost, LSTCN): it ensures the target's min/max used for
    # inverse-transforming test predictions is stable and consistent.
    # In a strict production setting one would fit the scaler on training
    # data only to avoid any information leakage from test years.
    sc_x = MinMaxScaler(feature_range=TRAIN_SCALE_RANGE)
    sc_y = MinMaxScaler(feature_range=TRAIN_SCALE_RANGE)

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
    y_train       = train_df[[TARGET]].values.reshape(-1)
    X_test        = test_df[feature_list].values
    years_test    = test_df["Year"].values
    y_test_scaled = test_df[[TARGET]].values.reshape(-1)

    # y_test_raw: true CO2 in the ORIGINAL (unscaled) unit.
    # Used for absolute metrics (MAE_absolute, MAPE, R²) at evaluation time.
    # Identical pattern across all four models.
    y_test_raw = df_use[
        df_use["Year"] >= TEST_START_YEAR
    ][[TARGET]].values.reshape(-1)

    # ── Step 4: Hyperparameter tuning (train set only) ─────────────────────
    best_params, tuning_df = tune_xgboost_hyperparams(
        X_train=X_train,
        y_train=y_train,
        country_name=country_name,
        feature_set_name=feature_set_name
    )

    # Reconstruct a flat params dict compatible with build_xgb_model()
    model_params = {
        "n_estimators":     best_params["best_n_estimators"],
        "max_depth":        best_params["best_max_depth"],
        "learning_rate":    best_params["best_learning_rate"],
        "subsample":        best_params["best_subsample"],
        "colsample_bytree": best_params["best_colsample_bytree"],
        "reg_lambda":       best_params["best_reg_lambda"],
        "reg_alpha":        best_params["best_reg_alpha"],
        "min_child_weight": best_params["best_min_child_weight"],
        "random_state":     best_params["best_random_state"]
    }

    # ── Step 5: Rolling one-step-ahead forecast on the test set ────────────
    # Start history with the complete training set.
    # After each prediction the TRUE observation is appended (expanding window).
    # This mirrors the real-world scenario where actual data is observed at
    # the end of each year before the next forecast is made.
    # Identical expanding-window logic across all four models.
    history_X    = list(X_train)
    history_y    = list(y_train)
    preds_scaled = []

    for i in range(len(X_test)):
        X_hist = np.asarray(history_X, dtype=float)
        y_hist = np.asarray(history_y, dtype=float).reshape(-1)

        # Re-fit XGBoost on all available history up to this point.
        # XGBoost builds an ensemble from scratch each time; there is no
        # incremental update mechanism for new observations.
        model = build_xgb_model(model_params)
        model.fit(X_hist, y_hist)

        # Forecast one step ahead using the next test year's features
        x_next      = np.asarray(X_test[i], dtype=float).reshape(1, -1)
        yhat_scaled = float(model.predict(x_next)[0])
        preds_scaled.append(yhat_scaled)

        # Expand history: use the TRUE scaled y (not the forecast yhat).
        # Identical update rule in ARIMAX, SVR, and LSTCN.
        history_X.append(X_test[i])
        history_y.append(y_test_scaled[i])

    # ── Step 6: Inverse-transform and evaluate ─────────────────────────────
    preds_scaled  = np.array(preds_scaled).reshape(-1, 1)
    y_pred_raw    = sc_y.inverse_transform(preds_scaled).reshape(-1)
    y_pred_scaled = preds_scaled.reshape(-1)

    metrics = compute_metrics(
        y_true_raw=y_test_raw,       # original scale - for absolute/MAPE/R²
        y_pred_raw=y_pred_raw,       # inverse-transformed predictions
        y_true_scaled=y_test_scaled, # scaled space - for MAE_scaled_y / RMSE_scaled_y
        y_pred_scaled=y_pred_scaled  # predictions still in scaled space
    )

    # Build a tidy predictions DataFrame for later export
    pred_df = pd.DataFrame({
        "country":   country_name,
        "year":      years_test,
        "actual":    y_test_raw,
        "predicted": y_pred_raw
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
        "MAPE", "Accuracy", "R_squared"
    ]

    rows = []

    # XGBoost-specific hyperparameter columns filled with NaN in summary rows
    xgb_param_nans = {
        "best_n_estimators":     np.nan,
        "best_max_depth":        np.nan,
        "best_learning_rate":    np.nan,
        "best_subsample":        np.nan,
        "best_colsample_bytree": np.nan,
        "best_reg_lambda":       np.nan,
        "best_reg_alpha":        np.nan,
        "best_min_child_weight": np.nan,
        "best_random_state":     np.nan
    }

    baseline_df  = metrics_df[metrics_df["feature_set"] == "baseline"].copy()
    baseline_avg = {"country": "BASELINE_AVG", "feature_set": "baseline", **xgb_param_nans}
    for col in metric_cols:
        baseline_avg[col] = baseline_df[col].mean()
    rows.append(baseline_avg)

    augmented_df  = metrics_df[metrics_df["feature_set"] == "augmented"].copy()
    augmented_avg = {"country": "AUGMENTED_AVG", "feature_set": "augmented", **xgb_param_nans}
    for col in metric_cols:
        augmented_avg[col] = augmented_df[col].mean()
    rows.append(augmented_avg)

    overall_avg = {"country": "OVERALL_AVG", "feature_set": "all", **xgb_param_nans}
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
      - For each country x feature_set: run the complete XGBoost pipeline
      - Write all three output CSVs when all countries are done

    Unlike the ARIMAX script, no checkpoint is saved after each country.
    If the run is interrupted, the entire script must be re-run from scratch.
    Progress is printed to stdout so long-running jobs can be monitored.

    Note on compute time: the XGBoost grid has 864 combinations. With rolling
    one-step-ahead CV inside each combination, this is the most compute-
    intensive tuning step among the four models. Expect significantly longer
    runtime compared to ARIMAX (64 combinations) or LSTCN (75 combinations).
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
                metrics, pred_df, best_params, tuning_df = run_xgboost_single_country(
                    data=data,
                    country_name=country,
                    feature_set_name=feature_set_name,
                    feature_list=feature_list
                )

                row = {
                    "country":     country,
                    "feature_set": feature_set_name,
                    **best_params,
                    **metrics
                }
                metrics_rows.append(row)

                # feature_set label is added here (not inside the function)
                # so the function itself stays agnostic to the column name
                pred_df["feature_set"] = feature_set_name
                pred_frames.append(pred_df)
                tuning_frames.append(tuning_df)

                print(
                    f"     best=({best_params['best_n_estimators']}, "
                    f"{best_params['best_max_depth']}, "
                    f"{best_params['best_learning_rate']}, "
                    f"{best_params['best_subsample']}, "
                    f"{best_params['best_colsample_bytree']}, "
                    f"{best_params['best_reg_lambda']}, "
                    f"{best_params['best_reg_alpha']}, "
                    f"{best_params['best_min_child_weight']}), "
                    f"RMSE_abs={metrics['RMSE_absolute']:.4f}, "
                    f"MAE_scaled_y={metrics['MAE_scaled_y']:.6f}, "
                    f"Accuracy={metrics['Accuracy']:.6f}"
                )

            except Exception as e:
                print(f"     ERROR: {e}")

    if not metrics_rows:
        raise RuntimeError(
            "No successful runs. "
            "Please check data, package version, or parameter compatibility."
        )

    # ── Build and write metrics CSV ────────────────────────────────────────
    metrics_df    = pd.DataFrame(metrics_rows)
    avg_df        = make_average_rows(metrics_df)
    metrics_final = pd.concat([metrics_df, avg_df], ignore_index=True)
    metrics_final = metrics_final[[
        "country", "feature_set",
        "best_n_estimators", "best_max_depth", "best_learning_rate",
        "best_subsample", "best_colsample_bytree",
        "best_reg_lambda", "best_reg_alpha", "best_min_child_weight",
        "best_random_state",
        "RMSE_absolute", "RMSE_scaled_y", "RMSE_relative",
        "MAE_absolute",  "MAE_scaled_y",  "MAE_relative",
        "MAPE", "Accuracy", "R_squared"
    ]]

    # ── Build and write predictions CSV ────────────────────────────────────
    preds_final = pd.concat(pred_frames, ignore_index=True)
    preds_final = preds_final[
        ["country", "feature_set", "year", "actual", "predicted"]
    ].sort_values(["country", "feature_set", "year"])

    # ── Build and write tuning log CSV ─────────────────────────────────────
    # Sorted by cv_MAE_scaled_y_mean so the best combinations appear first,
    # consistent with the sorting convention in ARIMAX, SVR, and LSTCN.
    tuning_final = pd.concat(tuning_frames, ignore_index=True)
    tuning_final = tuning_final[[
        "country", "feature_set",
        "n_estimators", "max_depth", "learning_rate",
        "subsample", "colsample_bytree",
        "reg_lambda", "reg_alpha", "min_child_weight", "random_state",
        "valid", "reason",
        "n_splits", "n_success_folds", "n_failed_folds",
        "cv_RMSE_absolute_mean", "cv_RMSE_scaled_y_mean", "cv_RMSE_relative_mean",
        "cv_MAE_absolute_mean",  "cv_MAE_scaled_y_mean",  "cv_MAE_relative_mean",
        "cv_MAPE_mean", "cv_Accuracy_mean", "cv_R_squared_mean"
    ]].sort_values(
        [
            "country", "feature_set",
            "cv_MAE_scaled_y_mean", "cv_RMSE_scaled_y_mean",
            "n_estimators", "max_depth"
        ],
        ascending=[True, True, True, True, True, True]
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
