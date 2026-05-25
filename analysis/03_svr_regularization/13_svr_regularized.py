"""
svr_regularized.py
==================
PURPOSE
-------
Re-runs SVR hyperparameter search with a constrained grid to address the
overfitting identified in 02_model_reliability/overfitting_analysis.py.

Three regularization strategies are applied relative to the original SVR
(06_svr.py):
  1. Poly kernel removed — poly increases model flexibility and was found
     to worsen the train/test gap; only linear and rbf are searched.
  2. Narrower C grid (0.001–10 vs 0.1–500) — smaller C values enforce a
     wider margin and stronger regularization.
  3. Wider epsilon grid (0.01–0.3 vs 0.001–0.2) — larger epsilon reduces
     sensitivity to small residuals, acting as implicit regularization.
  4. Finer gamma grid (adds 0.001, 0.01, 0.05) — allows the search to find
     smoother RBF kernels that generalise better on short annual series.

Pipeline overview
-----------------
For each country × feature_set combination:
  1. Filter, sort, and create a one-period lag of the target (CO2_lag1).
  2. Scale all variables to [-0.9, 0.9] on the full usable dataset.
  3. Split into train (≤ 2016) and test (≥ 2017) after scaling.
  4. Tune SVR hyperparameters via time-series cross-validation on the
     training set only, using the constrained grid above.
  5. Run a rolling one-step-ahead forecast on the test set with an
     expanding window (TRUE observation appended after each step).
  6. Inverse-transform predictions and compute evaluation metrics.

Key design notes
----------------
- All design choices (scaling convention, train/test split, rolling
  evaluation, lag feature) are identical to 06_svr.py so that results
  are directly comparable.
- Total grid combinations:
    linear : 8 (C) × 5 (epsilon)               =  40
    rbf    : 8 (C) × 5 (epsilon) × 6 (gamma)   = 240
    Total  : 280 combinations (vs ~260 in original SVR)

PIPELINE POSITION
-----------------
  06_svr.py  →  results/metrics/svr_metrics_summary.csv
                        ↓
  02_model_reliability/overfitting_analysis.py  (identifies overfitting)
                        ↓
  [THIS SCRIPT]  →  results/metrics/svr_regularized_metrics_summary.csv
                        ↓
  svr_train_val_test_regularized.py
                        ↓
  04_shap/shap_svr.py

Output files
------------
  results/metrics/svr_regularized_metrics_summary.csv
  results/predictions/svr_regularized_predictions_all.csv
  results/tuning/svr_regularized_tuning_results.csv

DEPENDENCIES
------------
  pandas, numpy, scikit-learn
  Install: pip install pandas numpy scikit-learn
"""

import warnings
import os
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# 1. Config
# =========================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed",
                         "dataset_recursive_3yr_avg_drop_Industry.csv")
TARGET = "Total CO2 emissions"
LAG_TARGET_NAME = "CO2_emissions_lag1"

COUNTRIES = [
    "Canada",
    "China",
    "India",
    "Indonesia",
    "Russian Federation",
    "United States",
]

# Unified split used across models
TRAIN_END_YEAR = 2016
TEST_START_YEAR = 2017
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2

SCALE_RANGE = (-0.9, 0.9)

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
    "baseline": BASE_FEATURES + [LAG_TARGET_NAME],
    "augmented": BASE_FEATURES + TEMP_FEATURES + [LAG_TARGET_NAME],
}

# Regularized SVR hyperparameter grid
# Poly kernel is removed because it can make the model more flexible and may worsen overfitting.
# Smaller C values strengthen regularization; larger epsilon values make the model less sensitive to small deviations.
KERNEL_GRID = ["linear", "rbf"]
C_GRID = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
EPSILON_GRID = [0.01, 0.05, 0.1, 0.2, 0.3]
GAMMA_GRID = ["scale", "auto", 0.001, 0.01, 0.05, 0.1]

TUNING_PRIMARY_METRIC = "MAE_scaled_y"

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results")
METRICS_OUT = os.path.join(RESULTS_DIR, "metrics",     "svr_regularized_metrics_summary.csv")
PRED_OUT    = os.path.join(RESULTS_DIR, "predictions", "svr_regularized_predictions_all.csv")
TUNING_OUT  = os.path.join(RESULTS_DIR, "tuning",      "svr_regularized_tuning_results.csv")


# =========================
# 2. Utility functions
# =========================
def compute_metrics(y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled):
    y_true_raw = np.asarray(y_true_raw).reshape(-1)
    y_pred_raw = np.asarray(y_pred_raw).reshape(-1)
    y_true_scaled = np.asarray(y_true_scaled).reshape(-1)
    y_pred_scaled = np.asarray(y_pred_scaled).reshape(-1)

    rmse_abs = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    mae_abs = mean_absolute_error(y_true_raw, y_pred_raw)

    rmse_scaled = np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))
    mae_scaled = mean_absolute_error(y_true_scaled, y_pred_scaled)

    y_mean = np.mean(y_true_raw)
    if np.isclose(y_mean, 0):
        rmse_rel = np.nan
        mae_rel = np.nan
    else:
        rmse_rel = rmse_abs / y_mean
        mae_rel = mae_abs / y_mean

    mape = mean_absolute_percentage_error(y_true_raw, y_pred_raw)
    accuracy = 1 - mape
    r2 = r2_score(y_true_raw, y_pred_raw)

    return {
        "RMSE_absolute": rmse_abs,
        "RMSE_scaled_y": rmse_scaled,
        "RMSE_relative": rmse_rel,
        "MAE_absolute": mae_abs,
        "MAE_scaled_y": mae_scaled,
        "MAE_relative": mae_rel,
        "MAPE": mape,
        "Accuracy": accuracy,
        "R_squared": r2,
    }


def build_svr_model(kernel, C, epsilon, gamma=None, degree=None):
    params = {
        "kernel": kernel,
        "C": C,
        "epsilon": epsilon,
    }

    if kernel in {"rbf", "poly"}:
        params["gamma"] = gamma

    if kernel == "poly":
        params["degree"] = degree

    return SVR(**params)


def iter_param_grid():
    for kernel in KERNEL_GRID:
        for C in C_GRID:
            for epsilon in EPSILON_GRID:
                if kernel == "linear":
                    yield {
                        "kernel": kernel,
                        "C": C,
                        "epsilon": epsilon,
                        "gamma": np.nan,
                        "degree": np.nan,
                    }
                elif kernel == "rbf":
                    for gamma in GAMMA_GRID:
                        yield {
                            "kernel": kernel,
                            "C": C,
                            "epsilon": epsilon,
                            "gamma": gamma,
                            "degree": np.nan,
                        }
                elif kernel == "poly":
                    for gamma in GAMMA_GRID:
                        for degree in DEGREE_GRID:
                            yield {
                                "kernel": kernel,
                                "C": C,
                                "epsilon": epsilon,
                                "gamma": gamma,
                                "degree": degree,
                            }


def evaluate_param_set_cv(X_train, y_train, params):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1)

    if len(X_train) < 6:
        return {
            "valid": False,
            "reason": "Not enough training rows for hyperparameter tuning.",
            "n_splits": np.nan,
            "n_success_folds": 0,
            "n_failed_folds": 0,
        }

    n_splits = min(5, max(2, len(X_train) // 4))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_rows = []
    n_success_folds = 0
    n_failed_folds = 0

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        try:
            model = build_svr_model(
                kernel=params["kernel"],
                C=params["C"],
                epsilon=params["epsilon"],
                gamma=None if pd.isna(params["gamma"]) else params["gamma"],
                degree=None if pd.isna(params["degree"]) else int(params["degree"]),
            )
            model.fit(X_tr, y_tr)
            pred_val = model.predict(X_val)

            fold_metrics = compute_metrics(
                y_true_raw=y_val,
                y_pred_raw=pred_val,
                y_true_scaled=y_val,
                y_pred_scaled=pred_val,
            )
            fold_rows.append(fold_metrics)
            n_success_folds += 1
        except Exception:
            n_failed_folds += 1

    if n_success_folds == 0:
        return {
            "valid": False,
            "reason": "All folds failed.",
            "n_splits": n_splits,
            "n_success_folds": 0,
            "n_failed_folds": n_failed_folds,
        }

    fold_df = pd.DataFrame(fold_rows)
    out = {
        "valid": True,
        "reason": "",
        "n_splits": n_splits,
        "n_success_folds": n_success_folds,
        "n_failed_folds": n_failed_folds,
    }
    for col in fold_df.columns:
        out[f"cv_{col}_mean"] = fold_df[col].mean()

    return out


def tune_svr_hyperparams(X_train, y_train, country_name, feature_set_name):
    tuning_rows = []

    for params in iter_param_grid():
        cv_result = evaluate_param_set_cv(X_train, y_train, params)
        row = {
            "country": country_name,
            "feature_set": feature_set_name,
            "kernel": params["kernel"],
            "C": params["C"],
            "epsilon": params["epsilon"],
            "gamma": params["gamma"],
            "degree": params["degree"],
            "valid": cv_result.get("valid", False),
            "reason": cv_result.get("reason", ""),
            "n_splits": cv_result.get("n_splits", np.nan),
            "n_success_folds": cv_result.get("n_success_folds", 0),
            "n_failed_folds": cv_result.get("n_failed_folds", 0),
            "cv_RMSE_absolute_mean": cv_result.get("cv_RMSE_absolute_mean", np.nan),
            "cv_RMSE_scaled_y_mean": cv_result.get("cv_RMSE_scaled_y_mean", np.nan),
            "cv_RMSE_relative_mean": cv_result.get("cv_RMSE_relative_mean", np.nan),
            "cv_MAE_absolute_mean": cv_result.get("cv_MAE_absolute_mean", np.nan),
            "cv_MAE_scaled_y_mean": cv_result.get("cv_MAE_scaled_y_mean", np.nan),
            "cv_MAE_relative_mean": cv_result.get("cv_MAE_relative_mean", np.nan),
            "cv_MAPE_mean": cv_result.get("cv_MAPE_mean", np.nan),
            "cv_Accuracy_mean": cv_result.get("cv_Accuracy_mean", np.nan),
            "cv_R_squared_mean": cv_result.get("cv_R_squared_mean", np.nan),
        }
        tuning_rows.append(row)

    tuning_df = pd.DataFrame(tuning_rows)
    valid_df = tuning_df[
        tuning_df["valid"].fillna(False) & tuning_df["cv_MAE_scaled_y_mean"].notna()
    ].copy()

    if valid_df.empty:
        raise RuntimeError(f"{country_name} | {feature_set_name}: no valid SVR parameter set found.")

    valid_df = valid_df.sort_values(
        by=[
            "cv_MAE_scaled_y_mean",
            "cv_RMSE_scaled_y_mean",
            "cv_MAPE_mean",
            "kernel",
            "C",
            "epsilon",
        ],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)

    best_row = valid_df.iloc[0]
    best_params = {
        "best_kernel": best_row["kernel"],
        "best_C": float(best_row["C"]),
        "best_epsilon": float(best_row["epsilon"]),
        "best_gamma": best_row["gamma"],
        "best_degree": best_row["degree"],
    }

    return best_params, tuning_df


def prepare_country_data(data, country_name, feature_list):
    cdf = data[data["Country Name"] == country_name].sort_values("Year").copy()
    cdf[LAG_TARGET_NAME] = cdf[TARGET].shift(1)

    cols = ["Year", TARGET] + feature_list
    df_use = cdf[cols].dropna().copy()

    if len(df_use) == 0:
        raise ValueError(f"{country_name}: no usable rows after lag creation and dropna().")

    return df_use


def run_svr_single_country(data, country_name, feature_set_name, feature_list):
    df_use = prepare_country_data(data, country_name, feature_list)

    sc_x = MinMaxScaler(feature_range=SCALE_RANGE)
    sc_y = MinMaxScaler(feature_range=SCALE_RANGE)

    df_scaled = df_use.copy()
    df_scaled[feature_list] = sc_x.fit_transform(df_use[feature_list].values)
    df_scaled[[TARGET]] = sc_y.fit_transform(df_use[[TARGET]].values)

    train_df = df_scaled[df_scaled["Year"] <= TRAIN_END_YEAR].copy()
    test_df = df_scaled[df_scaled["Year"] >= TEST_START_YEAR].copy()

    if len(train_df) < 6:
        raise ValueError(f"{country_name}: training rows too few ({len(train_df)}).")
    if len(test_df) == 0:
        raise ValueError(f"{country_name}: no test rows after split.")

    X_train = train_df[feature_list].values
    y_train = train_df[TARGET].values.reshape(-1)

    X_test = test_df[feature_list].values
    y_test_scaled = test_df[TARGET].values.reshape(-1)
    years_test = test_df["Year"].values

    y_test_raw = df_use[df_use["Year"] >= TEST_START_YEAR][TARGET].values.reshape(-1)

    best_params, tuning_df = tune_svr_hyperparams(
        X_train=X_train,
        y_train=y_train,
        country_name=country_name,
        feature_set_name=feature_set_name,
    )

    history_X = X_train.copy()
    history_y = y_train.copy()
    preds_scaled = []

    for i in range(len(X_test)):
        model = build_svr_model(
            kernel=best_params["best_kernel"],
            C=best_params["best_C"],
            epsilon=best_params["best_epsilon"],
            gamma=None if pd.isna(best_params["best_gamma"]) else best_params["best_gamma"],
            degree=None if pd.isna(best_params["best_degree"]) else int(best_params["best_degree"]),
        )
        model.fit(history_X, history_y)

        x_next = X_test[i].reshape(1, -1)
        yhat_scaled = float(model.predict(x_next)[0])
        preds_scaled.append(yhat_scaled)

        # expanding-window update with TRUE next observation
        history_X = np.vstack([history_X, X_test[i]])
        history_y = np.append(history_y, y_test_scaled[i])

    preds_scaled = np.asarray(preds_scaled).reshape(-1, 1)
    y_pred_raw = sc_y.inverse_transform(preds_scaled).reshape(-1)
    y_pred_scaled = preds_scaled.reshape(-1)

    metrics = compute_metrics(
        y_true_raw=y_test_raw,
        y_pred_raw=y_pred_raw,
        y_true_scaled=y_test_scaled,
        y_pred_scaled=y_pred_scaled,
    )

    pred_df = pd.DataFrame(
        {
            "country": country_name,
            "feature_set": feature_set_name,
            "year": years_test,
            "actual": y_test_raw,
            "predicted": y_pred_raw,
        }
    )

    return metrics, pred_df, best_params, tuning_df


def make_average_rows(metrics_df):
    metric_cols = [
        "RMSE_absolute",
        "RMSE_scaled_y",
        "RMSE_relative",
        "MAE_absolute",
        "MAE_scaled_y",
        "MAE_relative",
        "MAPE",
        "Accuracy",
        "R_squared",
    ]

    rows = []

    baseline_df = metrics_df[metrics_df["feature_set"] == "baseline"].copy()
    baseline_avg = {
        "country": "BASELINE_AVG",
        "feature_set": "baseline",
        "best_kernel": np.nan,
        "best_C": np.nan,
        "best_epsilon": np.nan,
        "best_gamma": np.nan,
        "best_degree": np.nan,
    }
    for col in metric_cols:
        baseline_avg[col] = baseline_df[col].mean()
    rows.append(baseline_avg)

    augmented_df = metrics_df[metrics_df["feature_set"] == "augmented"].copy()
    augmented_avg = {
        "country": "AUGMENTED_AVG",
        "feature_set": "augmented",
        "best_kernel": np.nan,
        "best_C": np.nan,
        "best_epsilon": np.nan,
        "best_gamma": np.nan,
        "best_degree": np.nan,
    }
    for col in metric_cols:
        augmented_avg[col] = augmented_df[col].mean()
    rows.append(augmented_avg)

    overall_avg = {
        "country": "OVERALL_AVG",
        "feature_set": "all",
        "best_kernel": np.nan,
        "best_C": np.nan,
        "best_epsilon": np.nan,
        "best_gamma": np.nan,
        "best_degree": np.nan,
    }
    for col in metric_cols:
        overall_avg[col] = metrics_df[col].mean()
    rows.append(overall_avg)

    return pd.DataFrame(rows)


# =========================
# 3. Main
# =========================
def main():
    os.makedirs(os.path.join(RESULTS_DIR, "metrics"),     exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "tuning"),      exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    metrics_rows = []
    pred_frames = []
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

                row = {
                    "country": country,
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
        raise RuntimeError("No successful SVR runs. Please check data, columns, and package versions.")

    metrics_df = pd.DataFrame(metrics_rows)
    avg_df = make_average_rows(metrics_df)
    metrics_final = pd.concat([metrics_df, avg_df], ignore_index=True)
    metrics_final = metrics_final[
        [
            "country",
            "feature_set",
            "best_kernel",
            "best_C",
            "best_epsilon",
            "best_gamma",
            "best_degree",
            "RMSE_absolute",
            "RMSE_scaled_y",
            "RMSE_relative",
            "MAE_absolute",
            "MAE_scaled_y",
            "MAE_relative",
            "MAPE",
            "Accuracy",
            "R_squared",
        ]
    ]

    preds_final = pd.concat(pred_frames, ignore_index=True)
    preds_final = preds_final[
        ["country", "feature_set", "year", "actual", "predicted"]
    ].sort_values(["country", "feature_set", "year"])

    tuning_final = pd.concat(tuning_frames, ignore_index=True)
    tuning_final = tuning_final[
        [
            "country",
            "feature_set",
            "kernel",
            "C",
            "epsilon",
            "gamma",
            "degree",
            "valid",
            "reason",
            "n_splits",
            "n_success_folds",
            "n_failed_folds",
            "cv_RMSE_absolute_mean",
            "cv_RMSE_scaled_y_mean",
            "cv_RMSE_relative_mean",
            "cv_MAE_absolute_mean",
            "cv_MAE_scaled_y_mean",
            "cv_MAE_relative_mean",
            "cv_MAPE_mean",
            "cv_Accuracy_mean",
            "cv_R_squared_mean",
        ]
    ].sort_values(
        ["country", "feature_set", "cv_MAE_scaled_y_mean", "kernel", "C", "epsilon"],
        ascending=[True, True, True, True, True, True],
    )

    metrics_final.to_csv(METRICS_OUT, index=False)
    preds_final.to_csv(PRED_OUT, index=False)
    tuning_final.to_csv(TUNING_OUT, index=False)

    print("\nSaved files:")
    print(f"  - {METRICS_OUT}")
    print(f"  - {PRED_OUT}")
    print(f"  - {TUNING_OUT}")

    print("\nMetrics preview:")
    print(metrics_final.round(6))


if __name__ == "__main__":
    main()
