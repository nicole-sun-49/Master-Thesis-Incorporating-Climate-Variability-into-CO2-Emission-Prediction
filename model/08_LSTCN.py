"""
LSTCN Rolling One-Step-Ahead Forecast for CO2 Emissions
========================================================
This script trains a Long Short-Term Cognitive Network (LSTCN) per country
to forecast annual total CO2 emissions, using an expanding-window rolling
evaluation that mirrors real-world sequential prediction.

Pipeline overview
-----------------
For each country x feature_set combination:
  1. Filter rows for the target country, sort by year, and select the
     required columns. dropna() is applied as a safety guard; the upstream
     dataset is assumed to be already preprocessed.
  2. Scale all variables to [-0.9, 0.9] using the full usable dataset.
  3. Split into train (<=2016) and test (>=2017) after scaling.
  4. Tune LSTCN hyperparameters (n_blocks, solver, alpha) via time-series
     cross-validation on the training set only. CV uses rolling one-step-ahead
     evaluation to mirror the actual test procedure.
  5. Run a rolling one-step-ahead forecast on the test set:
       - Re-fit LSTCN on all available history before each step
       - Append the TRUE observed value to history after each step
         (expanding window)
  6. Inverse-transform predictions and compute evaluation metrics.

Key design notes
----------------
- LSTCN does NOT use a lag feature. Instead, features and target are
  horizontally concatenated into a single Xy matrix before training:
      Xy[t] = [X[t] | y[t]]   shape: (T, n_features + 1)
  The model is then trained to predict Xy[t+1] from Xy[t], so the previous
  year's CO2 value is naturally included as the last element of the input
  row at every step. This is LSTCN's equivalent of the lag feature used by
  SVR and XGBoost, and the AR(p) component used by ARIMAX.
- n_steps is fixed at 1 throughout (only one-step-ahead transition modelled
  per block). Only n_blocks, solver, and alpha are varied in the grid search.
- The scaler is fit on the full usable dataset (train + test) before
  splitting. This is intentional for scaling consistency across countries;
  all four models in the study (ARIMAX, SVR, XGBoost, LSTCN) follow the
  same convention.
- Tuning criterion: cv_MAE_scaled_y_mean (primary), aligned with ARIMAX,
  SVR, and XGBoost for fair cross-model comparison.
- No checkpoint is saved mid-run (unlike ARIMAX). Results are written once
  all countries have completed.

Output files
------------
  lstcn_metrics_summary.csv  : Best params + all metrics per country/feature_set
  lstcn_predictions_all.csv  : Year-by-year actual vs predicted (original scale)
  lstcn_tuning_results.csv   : CV scores for every hyperparameter combination
"""
import os                                                          
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

from lstcn.LSTCN import LSTCN


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
# Identical across all four models (ARIMAX, SVR, XGBoost, LSTCN).
TRAIN_END_YEAR  = 2016
TEST_START_YEAR = 2017

# MinMaxScaler target range. Using (-0.9, 0.9) instead of (-1, 1) provides
# a small margin that prevents exact boundary values. This is particularly
# relevant for LSTCN's hyperbolic activation, which is defined on (-1, 1)
# and can become numerically unstable near the boundaries.
# All four models (ARIMAX, SVR, XGBoost, LSTCN) use the same range.
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
#   "baseline"  - only socioeconomic/energy features
#   "augmented" - adds climate features on top of baseline
#
# NOTE: Unlike SVR and XGBoost, LSTCN does NOT include a lag target feature.
# The lag is embedded implicitly: features and target are concatenated into
# Xy = [X | y] before training. Predicting Xy[t+1] from Xy[t] means the
# model always receives the previous year's CO2 as part of its input.
# ARIMAX also excludes the lag for the same conceptual reason (AR(p) handles it).
FEATURE_SETS = {
    "baseline":  BASE_FEATURES,
    "augmented": BASE_FEATURES + TEMP_FEATURES
}

# ── LSTCN hyperparameter search grid ───────────────────────────────────────
# n_steps   : number of time steps per LSTCN block; fixed at 1 because the
#             dataset is annual and we only model one-step-ahead transitions.
#             Varying this would require multi-step sequences, which is not
#             applicable here.
# n_blocks  : number of stacked LSTCN blocks; more blocks increase the model's
#             capacity to learn nonlinear temporal patterns, but also increase
#             risk of overfitting on short annual series
# function  : activation function applied inside each block; fixed to
#             "hyperbolic" (tanh-like) as it is the standard LSTCN activation
#             and suits the [-0.9, 0.9] scaled input range
# solver    : linear solver used to compute block weights via ridge regression
#   "svd"      - Singular Value Decomposition; most numerically stable,
#                recommended when n_samples is close to n_features
#   "cholesky" - Cholesky decomposition; faster than SVD when the system
#                is well-conditioned
#   "lsqr"     - iterative least-squares solver; efficient for sparse or
#                large systems
# alpha     : L2 regularisation strength applied in each block's ridge solve;
#             higher values suppress overfitting but may underfit on
#             informative features; alpha=0 means no regularisation
#
# Total combinations: 1 x 5 x 1 x 3 x 5 = 75
PARAM_GRID = {
    "n_steps":  [1],
    "n_blocks": [1, 2, 3, 4, 5],
    "function": ["hyperbolic"],
    "solver":   ["svd", "cholesky", "lsqr"],
    "alpha":    [1e-4, 1e-3, 1e-2, 1e-1, 0]
}

# Primary metric used to sort and select the best hyperparameter combination.
# Using the scaled-space MAE makes cross-country comparison consistent and
# aligns with the criterion used in ARIMAX, SVR, and XGBoost.
TUNING_SORT_METRIC = "cv_MAE_scaled_y_mean"

# ── Output file names ───────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
METRICS_OUT = os.path.join(RESULTS_DIR, "metrics",     "lstcn_metrics_summary.csv")
PRED_OUT    = os.path.join(RESULTS_DIR, "predictions", "lstcn_predictions_all.csv")
TUNING_OUT  = os.path.join(RESULTS_DIR, "tuning",      "lstcn_tuning_results.csv")


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
    When called inside tune_lstcn_hyperparams_manual(), y_true_raw ==
    y_true_scaled and y_pred_raw == y_pred_scaled (because the CV fold data
    is already in scaled space). In that case MAE_absolute and MAE_scaled_y
    are numerically identical; only MAE_scaled_y is used for tuning decisions.
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


def tune_lstcn_hyperparams_manual(X_train, y_train, country_name, feature_set_name):
    """
    Grid-search over all PARAM_GRID combinations and select the best one via
    rolling one-step-ahead time-series cross-validation on the training data.

    Why LSTCN uses a combined Xy matrix
    ------------------------------------
    LSTCN is trained on transitions between consecutive time steps:
        input  : Xy[t]   = [X[t]  | y[t]]    shape: (1, n_features + 1)
        target : Xy[t+1] = [X[t+1]| y[t+1]]  shape: (1, n_features + 1)

    The model predicts the ENTIRE next state vector, not just y. The CO2
    forecast is extracted as the LAST element of the predicted vector
    (Y_pred[0, -1]), since y is appended as the last column of Xy.

    This design means y[t] is always part of the input at step t, giving
    LSTCN implicit access to the lag target without a separate lag feature.
    SVR and XGBoost achieve the same thing via the explicit CO2_emissions_lag1
    column.

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
      1. Re-fit LSTCN on history (the Xy matrix built from expanding rows)
      2. Predict next Xy row from the last row in history
      3. Extract the last element of the predicted row as yhat
      4. Append the TRUE Xy row to history (not the predicted one)

    LSTCN-specific indexing note
    ----------------------------
    The Xy matrix is indexed differently from X_train/y_train in other models:
        X_in = Xy[:-1]   (all rows except the last -- these are the "from" states)
        Y_in = Xy[1:]    (all rows except the first -- these are the "to" states)
    So a history of length T produces T-1 training pairs.
    The last row of history (H[-1]) serves as the input for the next forecast.

    Parameters
    ----------
    X_train          : 2-D array - scaled features, shape (T_train, n_features)
    y_train          : 2-D array - scaled CO2, shape (T_train, 1)
    country_name     : str - used for labelling tuning records
    feature_set_name : str - "baseline" or "augmented"

    Returns
    -------
    best_params : dict with keys best_n_steps, best_n_blocks, best_function,
                  best_solver, best_alpha
    tuning_df   : DataFrame with one row per combination, containing
                  CV metrics and validity flags
    """
    # Concatenate features and target horizontally to form the Xy state matrix.
    # Shape: (T_train, n_features + 1), where the last column is scaled CO2.
    Xy_train = np.hstack([X_train, y_train])

    # Build transition pairs for LSTCN training:
    #   X_in[t] = Xy_train[t]     (current state)
    #   Y_in[t] = Xy_train[t+1]   (next state to predict)
    # This produces T_train - 1 training pairs.
    X_in = Xy_train[:-1]
    Y_in = Xy_train[1:]

    # Minimum data guard: need at least 6 transition pairs to form a CV split.
    # Shared requirement across all four models.
    if len(X_in) < 6:
        raise ValueError(
            "Not enough training sequences for hyperparameter tuning."
        )

    # Adapt number of folds to available training size.
    # Formula is identical across all four models.
    n_splits = min(5, max(2, len(X_in) // 4))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    metric_names = [
        "RMSE_absolute", "RMSE_scaled_y", "RMSE_relative",
        "MAE_absolute",  "MAE_scaled_y",  "MAE_relative",
        "MAPE", "Accuracy", "R_squared"
    ]

    rows = []

    # ParameterGrid expands PARAM_GRID into a list of dicts, one per combination.
    # This is sklearn's equivalent of itertools.product used in XGBoost.
    for params in ParameterGrid(PARAM_GRID):
        fold_metric_rows  = []
        n_failed_folds    = 0
        failure_messages  = []

        for train_idx, val_idx in tscv.split(X_in):
            # Note: train_idx and val_idx index into X_in / Y_in (transition pairs),
            # not directly into Xy_train rows. The mapping is:
            #   transition pair t corresponds to Xy_train rows t and t+1.

            # Rolling one-step-ahead loop inside the CV fold.
            # History is initialised to cover Xy_train rows 0 through
            # train_idx[-1] + 1 (inclusive), which is the full "from" state
            # for the last training transition plus one extra row so that
            # H[-1] is always a valid input for the next forecast.
            history_Xy_cv = list(Xy_train[: train_idx[-1] + 2])
            preds_val_cv  = []
            fold_failed   = False

            for j in range(len(val_idx)):
                H     = np.array(history_Xy_cv)
                X_in_h = H[:-1]   # "from" states
                Y_in_h = H[1:]    # "to" states

                try:
                    model = LSTCN(
                        n_features=Xy_train.shape[1],  # n_features + 1 (includes y)
                        n_steps=params["n_steps"],
                        n_blocks=params["n_blocks"],
                        function=params["function"],
                        solver=params["solver"],
                        alpha=params["alpha"],
                    )
                    model.fit(X_in_h, Y_in_h)

                    # Predict the next full state vector from the last known state
                    last_row = H[-1:].copy()   # shape: (1, n_features + 1)
                    Y_pred   = model.predict(last_row)

                    # Extract CO2 forecast: last column of the predicted row
                    # (because y was appended as the last column of Xy)
                    yhat = float(Y_pred[0, -1])
                    preds_val_cv.append(yhat)

                except Exception as e:
                    fold_failed = True
                    failure_messages.append(type(e).__name__)
                    break

                # Expand history: append the TRUE next Xy row (not the predicted one).
                # val_idx[j] + 1 is the index into Xy_train for the next "to" state.
                # Identical principle to ARIMAX, SVR, and XGBoost: always use true
                # observations to build history, not the model's own predictions.
                history_Xy_cv.append(Xy_train[val_idx[j] + 1])

            if fold_failed or len(preds_val_cv) != len(val_idx):
                n_failed_folds += 1
                continue

            # Ground truth for this fold: last column of Y_in for validation indices
            y_true_scaled = Y_in[val_idx][:, -1]
            y_pred_scaled = np.array(preds_val_cv)

            # Compute metrics for this fold.
            # Both raw and scaled arguments receive scaled values because all
            # CV data is in the scaled space (identical to ARIMAX, SVR, XGBoost).
            fold_metric_rows.append(
                compute_metrics(
                    y_true_raw=y_true_scaled,
                    y_pred_raw=y_pred_scaled,
                    y_true_scaled=y_true_scaled,
                    y_pred_scaled=y_pred_scaled,
                )
            )

        n_success_folds = len(fold_metric_rows)
        valid = n_success_folds > 0

        # Collect error type names for diagnostic logging when some folds fail
        if n_failed_folds == 0:
            reason = np.nan
        else:
            unique_errors = sorted(set(failure_messages))
            reason = "one_or_more_folds_failed:" + "|".join(unique_errors)

        # Record this combination, including failures, for diagnostics.
        # Identical recording pattern across all four models.
        row = {
            "country":         country_name,
            "feature_set":     feature_set_name,
            "n_steps":         params["n_steps"],
            "n_blocks":        params["n_blocks"],
            "function":        params["function"],
            "solver":          params["solver"],
            "alpha":           params["alpha"],
            "valid":           valid,
            "reason":          reason,
            "n_splits":        n_splits,
            "n_success_folds": n_success_folds,
            "n_failed_folds":  n_failed_folds,
        }

        if valid:
            # Aggregate metrics across successful folds (simple mean).
            # Identical aggregation logic across all four models.
            for metric_name in metric_names:
                row[f"cv_{metric_name}_mean"] = np.nanmean(
                    [m[metric_name] for m in fold_metric_rows]
                )
        else:
            for metric_name in metric_names:
                row[f"cv_{metric_name}_mean"] = np.nan

        rows.append(row)

    tuning_df = pd.DataFrame(rows)

    # Keep only combinations that produced at least one valid CV score
    valid_df = tuning_df[tuning_df["valid"]].copy()
    if valid_df.empty:
        raise RuntimeError(
            f"{country_name}-{feature_set_name}: "
            "no valid hyperparameter combination found."
        )

    # Sort by the ranking priority (aligned with all four models):
    #   1. cv_MAE_scaled_y_mean  - primary criterion
    #   2. alpha                 - prefer stronger regularisation as tiebreaker
    #   3. n_blocks              - prefer fewer blocks (simpler model)
    #   4. solver, function      - alphabetical tiebreaker
    valid_df = valid_df.sort_values(
        [TUNING_SORT_METRIC, "alpha", "n_blocks", "solver", "function"],
        ascending=[True, True, True, True, True],
        na_position="last"
    ).reset_index(drop=True)

    best_row    = valid_df.iloc[0]
    best_params = {
        "best_n_steps":  int(best_row["n_steps"]),
        "best_n_blocks": int(best_row["n_blocks"]),
        "best_function": best_row["function"],
        "best_solver":   best_row["solver"],
        "best_alpha":    float(best_row["alpha"]),
    }

    # Re-sort the full tuning_df (including invalid rows) for export
    tuning_df = tuning_df.sort_values(
        ["country", "feature_set",
         TUNING_SORT_METRIC, "alpha", "n_blocks", "solver", "function"],
        ascending=[True, True, True, True, True, True, True],
        na_position="last"
    ).reset_index(drop=True)

    return best_params, tuning_df


def run_lstcn_single_country(data, country_name, feature_set_name, feature_list):
    """
    Full pipeline for a single country x feature_set combination:
      1. Extract and clean country data
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
                       (does NOT include LAG_TARGET_NAME; LSTCN handles
                       temporal context through the Xy structure)

    Returns
    -------
    metrics    : dict - test-set evaluation metrics
    pred_df    : DataFrame - year / actual / predicted (original scale)
    best_params: dict - best LSTCN hyperparameters found by CV
    tuning_df  : DataFrame - full CV tuning log
    """
    # ── Step 1: Filter country and drop rows with any missing value ────────
    cdf    = data[data["Country Name"] == country_name].sort_values("Year").copy()
    df_use = cdf[["Year", TARGET] + feature_list].dropna().copy()
    # dropna() is a safety guard; the upstream dataset is assumed to be
    # already preprocessed. Unlike SVR and XGBoost, no lag shift is applied
    # here, so no additional year is lost.

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

    if len(train_df) < 2:
        # LSTCN requires at least 2 rows to form one transition pair (Xy[t] -> Xy[t+1])
        raise ValueError(f"{country_name}: training rows < 2.")
    if len(test_df) == 0:
        raise ValueError(f"{country_name}: no test rows after split.")

    X_train    = train_df[feature_list].values
    y_train    = train_df[[TARGET]].values   # keep as 2-D (T, 1) for hstack
    X_test     = test_df[feature_list].values
    years_test = test_df["Year"].values

    # y_test_raw: true CO2 in the ORIGINAL (unscaled) unit.
    # Used for absolute metrics (MAE_absolute, MAPE, R²) at evaluation time.
    # Identical pattern across all four models.
    y_test_raw = df_use[
        df_use["Year"] >= TEST_START_YEAR
    ][[TARGET]].values.reshape(-1)

    # ── Step 4: Hyperparameter tuning (train set only) ─────────────────────
    best_params, tuning_df = tune_lstcn_hyperparams_manual(
        X_train=X_train,
        y_train=y_train,
        country_name=country_name,
        feature_set_name=feature_set_name,
    )

    # ── Step 5: Rolling one-step-ahead forecast on the test set ────────────
    # Build the combined Xy state matrix from the full training set.
    # history_Xy accumulates rows of [X | y] as the window expands.
    Xy_train    = np.hstack([X_train, y_train])
    history_Xy  = list(Xy_train)
    preds_scaled = []

    for i in range(len(X_test)):
        H = np.array(history_Xy)   # shape: (t, n_features + 1)

        # Build transition pairs from the current history
        X_in = H[:-1]   # "from" states: rows 0 to t-1
        Y_in = H[1:]    # "to" states:   rows 1 to t

        # Re-fit LSTCN on all available history up to this point.
        # Re-fitting from scratch is the only way to incorporate the latest
        # observation, identical to SVR, XGBoost, and ARIMAX.
        model = LSTCN(
            n_features=H.shape[1],              # n_features + 1 (includes y)
            n_steps=best_params["best_n_steps"],
            alpha=best_params["best_alpha"],
            n_blocks=best_params["best_n_blocks"],
            solver=best_params["best_solver"],
            function=best_params["best_function"],
        )
        model.fit(X_in, Y_in)

        # Forecast next state from the last known state
        last_row = H[-1:].copy()       # shape: (1, n_features + 1)
        Y_pred   = model.predict(last_row)

        # Extract the CO2 forecast: last element of the predicted state vector
        yhat_scaled = float(Y_pred[0, -1])
        preds_scaled.append(yhat_scaled)

        # Expand history: append the TRUE next Xy row, not the predicted one.
        # true_y_scaled is obtained by re-applying sc_y to the raw test value.
        # np.clip prevents any out-of-range values (due to the scaler being fit
        # on the full dataset, test values should always be in range, but clipping
        # is a safe guard).
        # This is the LSTCN-specific way of computing the scaled true y;
        # ARIMAX, SVR, and XGBoost read it directly from the pre-scaled test_df.
        true_y_scaled = float(sc_y.transform([[float(y_test_raw[i])]])[0][0])
        true_y_scaled = np.clip(
            true_y_scaled,
            TRAIN_SCALE_RANGE[0],
            TRAIN_SCALE_RANGE[1]
        )

        # Concatenate the next test year's X with the true scaled y to form
        # the new Xy row to append to history
        new_row = np.append(X_test[i], true_y_scaled)
        history_Xy.append(new_row)

    # ── Step 6: Inverse-transform and evaluate ─────────────────────────────
    preds_scaled  = np.array(preds_scaled).reshape(-1, 1)
    y_pred_raw    = sc_y.inverse_transform(preds_scaled).reshape(-1)
    y_test_scaled = test_df[[TARGET]].values.reshape(-1)
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

    # LSTCN-specific hyperparameter columns filled with NaN in summary rows
    lstcn_param_nans = {
        "best_alpha":    np.nan,
        "best_n_blocks": np.nan,
        "best_n_steps":  np.nan,
        "best_solver":   np.nan,
        "best_function": np.nan,
    }

    baseline_df  = metrics_df[metrics_df["feature_set"] == "baseline"].copy()
    baseline_avg = {"country": "BASELINE_AVG", "feature_set": "baseline", **lstcn_param_nans}
    for col in metric_cols:
        baseline_avg[col] = baseline_df[col].mean()
    rows.append(baseline_avg)

    augmented_df  = metrics_df[metrics_df["feature_set"] == "augmented"].copy()
    augmented_avg = {"country": "AUGMENTED_AVG", "feature_set": "augmented", **lstcn_param_nans}
    for col in metric_cols:
        augmented_avg[col] = augmented_df[col].mean()
    rows.append(augmented_avg)

    overall_avg = {"country": "OVERALL_AVG", "feature_set": "all", **lstcn_param_nans}
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
      - For each country x feature_set: run the complete LSTCN pipeline
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
                metrics, pred_df, best_params, tuning_df = run_lstcn_single_country(
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
                    f"     best_alpha={best_params['best_alpha']}, "
                    f"best_n_blocks={best_params['best_n_blocks']}, "
                    f"best_solver={best_params['best_solver']}, "
                    f"best_function={best_params['best_function']}, "
                    f"RMSE_abs={metrics['RMSE_absolute']:.4f}, "
                    f"RMSE_scaled_y={metrics['RMSE_scaled_y']:.6f}, "
                    f"RMSE_rel={metrics['RMSE_relative']:.6f}, "
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
        "best_n_steps", "best_n_blocks", "best_function",
        "best_solver", "best_alpha",
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
    # consistent with the sorting convention in ARIMAX, SVR, and XGBoost.
    tuning_final = pd.concat(tuning_frames, ignore_index=True)
    tuning_final = tuning_final[[
        "country", "feature_set",
        "n_steps", "n_blocks", "function", "solver", "alpha",
        "valid", "reason",
        "n_splits", "n_success_folds", "n_failed_folds",
        "cv_RMSE_absolute_mean", "cv_RMSE_scaled_y_mean", "cv_RMSE_relative_mean",
        "cv_MAE_absolute_mean",  "cv_MAE_scaled_y_mean",  "cv_MAE_relative_mean",
        "cv_MAPE_mean", "cv_Accuracy_mean", "cv_R_squared_mean"
    ]].sort_values(
        ["country", "feature_set",
         TUNING_SORT_METRIC, "alpha", "n_blocks", "solver", "function"],
        ascending=[True, True, True, True, True, True, True],
        na_position="last"
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
