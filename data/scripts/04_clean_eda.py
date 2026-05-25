"""
04_clean_eda.py
===============
PURPOSE
-------
Phase 2 EDA on the cleaned dataset (after imputation and anomaly treatment).
All analysis uses the model-ready data produced by 03_clean_data.py.

This script generates the evidence that justifies the modelling choices
made in the thesis (feature selection, per-country modelling, ARIMAX
lag structure, augmented feature set).

INPUT
-----
  dataset_recursive_3yr_avg_drop_Industry.csv   (output of 03_clean_data.py)

OUTPUTS  (written to ./04_outputs/)
-------
  01_correlation/
    correlation_by_latitude_group.png           – 3-panel combined bar chart
    correlation_by_latitude_group.csv           – numeric values behind the chart
    correlation_per_latitude_group/
      correlation_HighLatitude.png              – individual bar chart per group
      correlation_MiddleLatitude.png
      correlation_LowLatitude.png
    correlation_per_country/
      correlation_Canada.png                    – per-country full heatmap (appendix)
      correlation_China.png
      correlation_India.png
      correlation_Indonesia.png
      correlation_RussianFederation.png
      correlation_UnitedStates.png
      correlation_per_country.csv

  02_acf/
    acf_summary.csv                             – lag-1/2/3 ACF per country (main text)
    acf_per_country/
      acf_Canada.png  …  acf_UnitedStates.png

  03_adf/
    adf_co2_results.csv

  04_temperature/
    temperature_by_latitude_group.csv           – grouped stats (main text, 3 rows)
    temperature_per_country_<feature>.csv       – per-country per-variable (appendix)
    trends/
      trend_Temperature_annual_mean.png         – trend plots moved from 02_raw_eda.py
      trend_Temperature_std_across_months.png   – uses cleaned dataset
      trend_Number_of_frost_days.png
      trend_Number_of_hot_days.png

PIPELINE POSITION
-----------------
  03_clean_data.py  →  [THIS SCRIPT]  →  04_outputs/
                                               ↓
                                         Model scripts

LATITUDE GROUPING
-----------------
  High Latitude   : Canada, Russian Federation
  Middle Latitude : United States, China
  Low Latitude    : India, Indonesia

  Pooled correlation: both countries' rows are stacked (68 rows per group)
  and Pearson correlation is computed on the combined data — NOT the average
  of two separate correlations.

  Pooled descriptive stats: mean and std are computed on the combined rows
  (up to 68 valid observations per metric, fewer if NaN present).

DEPENDENCIES
------------
  pandas, numpy, matplotlib, seaborn, scipy, statsmodels
  Install: pip install pandas numpy matplotlib seaborn scipy statsmodels
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.linalg import lstsq as scipy_lstsq


# ── ACF (no statsmodels) ──────────────────────────────────────
def _acf_manual(series: np.ndarray, nlags: int) -> np.ndarray:
    """
    Compute sample autocorrelation function up to `nlags` lags.
    Normalised so that lag-0 = 1.0.
    """
    x   = series - series.mean()
    n   = len(x)
    c0  = np.dot(x, x) / n
    out = [1.0]
    for k in range(1, nlags + 1):
        ck = np.dot(x[:-k], x[k:]) / n
        out.append(ck / c0)
    return np.array(out)


# ── ADF (no statsmodels) ──────────────────────────────────────
def _adfuller_manual(series: np.ndarray, maxlag: int = None):
    """
    Simplified Augmented Dickey-Fuller test.

    Regression:  Δy_t = α + β·y_{t-1} + Σγ_i·Δy_{t-i} + ε
    H₀: β = 0  (unit root, non-stationary)

    Lag order is selected by AIC up to maxlag (default: int(12*(n/100)^0.25)).
    Returns (adf_stat, p_value, used_lag, n_obs, critical_values_dict).

    Critical values are approximated from MacKinnon (1994) response surface
    for the constant-only case (regression type 'c').
    """
    y  = np.asarray(series, dtype=float)
    n  = len(y)

    if maxlag is None:
        maxlag = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    maxlag = min(maxlag, n // 3)

    dy = np.diff(y)                            # first differences Δy_t

    best_lag, best_aic, best_result = 0, np.inf, None

    for lag in range(0, maxlag + 1):
        # Build regressor matrix
        nobs = len(dy) - lag
        if nobs < lag + 3:
            continue

        y_lag1 = y[lag: lag + nobs]            # y_{t-1}
        dy_t   = dy[lag: lag + nobs]            # Δy_t  (dependent)

        X = np.column_stack([
            np.ones(nobs),                     # constant
            y_lag1,                            # β·y_{t-1}
        ])
        if lag > 0:
            lagged_dy = np.column_stack(
                [dy[lag - i - 1: lag - i - 1 + nobs] for i in range(lag)]
            )
            X = np.column_stack([X, lagged_dy])

        coef, resid, rank, _ = scipy_lstsq(X, dy_t)
        rss = float(np.dot(resid, resid)) if resid.size else float(np.sum((dy_t - X @ coef) ** 2))
        k   = X.shape[1]
        aic = nobs * np.log(rss / nobs) + 2 * k

        if aic < best_aic:
            best_aic    = aic
            best_lag    = lag
            best_result = (coef, X, dy_t, nobs, k)

    coef, X, dy_t, nobs, k = best_result
    fitted  = X @ coef
    e       = dy_t - fitted
    s2      = np.sum(e ** 2) / (nobs - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se      = np.sqrt(np.diag(s2 * XtX_inv))

    # β is the coefficient on y_{t-1}, which is column index 1
    adf_stat = coef[1] / se[1]

    # MacKinnon (1994) approximate p-value for constant-only ADF (table 4.2)
    # Response surface: p = Φ(a0 + a1/n + a2/n²) fitted to tabulated values
    # Using the "c" (constant only) case coefficients
    mackinnon_c = np.array([
        # [tau_crit, beta_inf, beta_1, beta_2]  for 1%, 5%, 10%
        [-3.43035, -6.5393, -16.786, -79.433],
        [-2.86154, -2.8621, -13.786, -32.414],
        [-2.56677, -1.5384,  -9.293, -19.329],
    ])
    # Approximate p-value via interpolation between critical values
    crits = {
        "1%":  mackinnon_c[0, 0] + mackinnon_c[0, 1] / nobs,
        "5%":  mackinnon_c[1, 0] + mackinnon_c[1, 1] / nobs,
        "10%": mackinnon_c[2, 0] + mackinnon_c[2, 1] / nobs,
    }

    # Rough p-value: interpolate linearly in the ADF stat ↔ significance space
    if   adf_stat <= crits["1%"]:
        p_val = 0.005
    elif adf_stat <= crits["5%"]:
        t0, t1 = crits["1%"],  crits["5%"]
        p_val  = 0.01 + (adf_stat - t0) / (t1 - t0) * (0.05 - 0.01)
    elif adf_stat <= crits["10%"]:
        t0, t1 = crits["5%"],  crits["10%"]
        p_val  = 0.05 + (adf_stat - t0) / (t1 - t0) * (0.10 - 0.05)
    else:
        # Beyond 10% critical: rough linear extrapolation, capped at 0.99
        slope = (0.10 - 0.05) / (crits["10%"] - crits["5%"])
        p_val = min(0.10 + slope * (adf_stat - crits["10%"]), 0.99)

    return adf_stat, p_val, best_lag, nobs, crits

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "processed", "dataset_recursive_3yr_avg_drop_Industry.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "eda", "04_clean_eda")

# ── Latitude grouping ─────────────────────────────────────────
LATITUDE_GROUPS = {
    "High Latitude":   ["Canada", "Russian Federation"],
    "Middle Latitude": ["United States", "China"],
    "Low Latitude":    ["India", "Indonesia"],
}

# ── Feature sets ──────────────────────────────────────────────
# All features (excluding identifiers and target)
ALL_FEATURES = [
    "Population",
    "GDP",
    "Electric power consumption",
    "Fossil fuel energy consumption",
    "Renewable energy consumption",
    "Fertilizer consumption",
    "Temperature annual mean",
    "Temperature std across months",
    "Number of frost days",
    "Number of hot days",
]
TARGET = "Total CO2 emissions"

TEMP_FEATURES = [
    "Temperature annual mean",
    "Temperature std across months",
    "Number of frost days",
    "Number of hot days",
]

# ── Consistent country colour palette ─────────────────────────
COUNTRY_COLORS = {
    "Canada":             "#1f77b4",
    "China":              "#d62728",
    "India":              "#ff7f0e",
    "Indonesia":          "#2ca02c",
    "Russian Federation": "#9467bd",
    "United States":      "#8c564b",
}

# ── ACF settings ──────────────────────────────────────────────
ACF_NLAGS    = 10
ACF_SUMMARY_LAGS = [1, 2, 3]   # lags shown in the summary CSV (main text)


# ============================================================
# 2. SETUP
# ============================================================

def setup_output_dirs() -> dict:
    """Create all output sub-directories and return their paths."""
    dirs = {
        "corr":         os.path.join(OUT_DIR, "01_correlation"),
        "corr_pg":      os.path.join(OUT_DIR, "01_correlation", "correlation_per_latitude_group"),
        "corr_pc":      os.path.join(OUT_DIR, "01_correlation", "correlation_per_country"),
        "acf":          os.path.join(OUT_DIR, "02_acf"),
        "acf_pc":       os.path.join(OUT_DIR, "02_acf", "acf_per_country"),
        "adf":          os.path.join(OUT_DIR, "03_adf"),
        "temp":         os.path.join(OUT_DIR, "04_temperature"),
        "temp_trends":  os.path.join(OUT_DIR, "04_temperature", "trends"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def load_data() -> pd.DataFrame:
    """Load the cleaned panel dataset."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            f"Run 03_clean_data.py first."
        )
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(["Country Name", "Year"]).reset_index(drop=True)
    return df


def safe_filename(name: str) -> str:
    """Convert a country name to a safe filename string."""
    return name.replace(" ", "").replace(".", "")


# ============================================================
# 3. CORRELATION ANALYSIS
# ============================================================

def compute_feature_co2_corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation of each feature with Total CO2 emissions.

    For each (feature, group/country) pair, the correlation is calculated
    on the pooled rows (both countries stacked for group-level analysis).
    Returns a DataFrame with features as rows and groups/countries as columns.
    """
    rows = []
    for feat in ALL_FEATURES:
        row = {"Feature": feat}
        for group, countries in LATITUDE_GROUPS.items():
            sub = df[df["Country Name"].isin(countries)][[feat, TARGET]].dropna()
            r, _ = stats.pearsonr(sub[feat], sub[TARGET]) if len(sub) > 2 else (np.nan, np.nan)
            row[group] = round(r, 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Feature")


def _highlight_temp_yticks(ax) -> None:
    """
    Apply an orange bounding box to any y-axis tick label whose text
    matches a temperature-related feature, consistent with the SHAP bar
    chart styling used in SHAP_SVR_augmented_only.py.

    Visual spec
    -----------
    Box face colour : #FFE4CC  (very light orange)
    Box edge colour : #E8762C  (warm orange)
    Text colour     : #C45A00  (darker orange for legibility)
    """
    temp_keywords = ["temperature", "frost", "hot days"]
    for label in ax.get_yticklabels():
        if any(kw in label.get_text().lower() for kw in temp_keywords):
            label.set_color("#C45A00")
            label.set_bbox({
                "facecolor": "#FFE4CC",
                "edgecolor": "#E8762C",
                "boxstyle":  "round,pad=0.25",
                "linewidth": 0.8,
            })


def _highlight_temp_axis_labels(ax, axis: str = "both") -> None:
    """
    Apply the same orange highlight to tick labels on x-axis, y-axis, or both.
    Used for correlation heatmaps where temperature features appear on both axes.
    """
    temp_keywords = ["temperature", "frost", "hot days"]

    def _apply(labels):
        for label in labels:
            if any(kw in label.get_text().lower() for kw in temp_keywords):
                label.set_color("#C45A00")
                label.set_bbox({
                    "facecolor": "#FFE4CC",
                    "edgecolor": "#E8762C",
                    "boxstyle":  "round,pad=0.25",
                    "linewidth": 0.8,
                })

    if axis in ("y", "both"):
        _apply(ax.get_yticklabels())
    if axis in ("x", "both"):
        _apply(ax.get_xticklabels())


def plot_correlation_by_latitude(corr_df: pd.DataFrame, dirs: dict) -> None:
    """
    Horizontal bar chart with 3 panels (one per latitude group),
    features sorted by absolute correlation within each group.
    Temperature-related features highlighted in a distinct colour.
    Mirrors the style of Image 3 in the thesis slides.

    Labels are placed INSIDE bars (near the bar tip) to avoid overlapping
    with the colourbar on the right panel.
    """
    groups   = list(LATITUDE_GROUPS.keys())
    cmap     = plt.cm.RdBu_r
    norm     = plt.Normalize(vmin=-1, vmax=1)

    # Wider figure + leave room for colourbar on the right
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
    fig.subplots_adjust(left=0.08, right=0.88, wspace=0.45)
    fig.suptitle("Feature Correlation with CO₂ Emissions by Latitude Group",
                 fontsize=13, y=1.01)

    for ax, group in zip(axes, groups):
        series = corr_df[group].dropna().sort_values()
        colors = [
            "#d62728" if feat in TEMP_FEATURES else cmap(norm(v))
            for feat, v in series.items()
        ]
        bars = ax.barh(series.index, series.values, color=colors,
                       edgecolor="white", linewidth=0.5, height=0.65)

        # Labels placed INSIDE the bar, near the tip — avoids colourbar clash
        for bar, val in zip(bars, series.values):
            abs_val = abs(val)
            # For very short bars (|val| < 0.12) place label outside instead
            if abs_val >= 0.12:
                x_pos = val * 0.88          # 88% along the bar = inside tip
                ha    = "right" if val >= 0 else "left"
            else:
                x_pos = val + (0.03 if val >= 0 else -0.03)
                ha    = "left" if val >= 0 else "right"

            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center", ha=ha,
                fontsize=7.5,
                color="white" if abs_val >= 0.12 else "black",
                fontweight="bold" if abs_val >= 0.12 else "normal",
            )

        ax.set_title(group, fontsize=10, fontweight="bold")
        ax.set_xlabel("Pearson r  with  Total CO₂ emissions", fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim(-1.15, 1.15)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=7)
        _highlight_temp_yticks(ax)   # orange bbox on temperature y-labels
        sns.despine(ax=ax)

    # Single shared colourbar — placed in its own dedicated axes on the right
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Correlation")
    cbar_ax.tick_params(labelsize=8)

    out = os.path.join(dirs["corr"], "correlation_by_latitude_group.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 01_correlation/correlation_by_latitude_group.png")


def plot_correlation_per_latitude_group(corr_df: pd.DataFrame, dirs: dict) -> None:
    """
    Save one independent PNG per latitude group — same format as each panel
    in the combined chart, but as standalone files so they can be placed
    individually in the thesis.

    Naming:
      correlation_per_latitude_group/correlation_HighLatitude.png
      correlation_per_latitude_group/correlation_MiddleLatitude.png
      correlation_per_latitude_group/correlation_LowLatitude.png
    """
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-1, vmax=1)

    safe_group = {
        "High Latitude":   "HighLatitude",
        "Middle Latitude": "MiddleLatitude",
        "Low Latitude":    "LowLatitude",
    }

    for group in LATITUDE_GROUPS:
        series = corr_df[group].dropna().sort_values()
        colors = [
            "#d62728" if feat in TEMP_FEATURES else cmap(norm(v))
            for feat, v in series.items()
        ]

        fig, ax = plt.subplots(figsize=(7, 6))
        bars = ax.barh(series.index, series.values, color=colors,
                       edgecolor="white", linewidth=0.5, height=0.65)

        for bar, val in zip(bars, series.values):
            abs_val = abs(val)
            if abs_val >= 0.12:
                x_pos = val * 0.88
                ha    = "right" if val >= 0 else "left"
            else:
                x_pos = val + (0.03 if val >= 0 else -0.03)
                ha    = "left" if val >= 0 else "right"
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center", ha=ha, fontsize=8,
                color="white" if abs_val >= 0.12 else "black",
                fontweight="bold" if abs_val >= 0.12 else "normal",
            )

        countries_label = " / ".join(LATITUDE_GROUPS[group])
        ax.set_title(f"{group}\n({countries_label})",
                     fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Pearson r  with  Total CO₂ emissions", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim(-1.15, 1.15)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
        _highlight_temp_yticks(ax)   # orange bbox on temperature y-labels
        sns.despine(ax=ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Correlation", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        fig.tight_layout()
        fname = os.path.join(
            dirs["corr_pg"],
            f"correlation_{safe_group[group]}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: 01_correlation/correlation_per_latitude_group/"
              f"correlation_{safe_group[group]}.png")


def plot_correlation_per_country(df: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    """
    Full feature × feature correlation heatmap for each country individually.
    Saved to 01_correlation/correlation_per_country/ (appendix figures).

    Why some features are excluded per country
    ------------------------------------------
    Pearson correlation requires non-zero variance in both variables.
    After cleaning, some features are constant for a specific country:
      - Indonesia: Number of frost days = 0 for all years (tropical country,
        no frost physically possible → imputed as 0 in 03_clean_data.py).
      - Canada:    Number of hot days may also be near-constant.
    A constant column produces std = 0, making Pearson r undefined (NaN).
    pandas .corr() silently returns NaN for such columns, and seaborn renders
    them as blank rows/columns in the heatmap.

    Fix: drop any column whose std == 0 (after dropping NaN rows) before
    computing the correlation matrix.  A subtitle notes excluded features.
    """
    countries = sorted(df["Country Name"].unique())
    csv_rows  = []

    for country in countries:
        sub_raw = df[df["Country Name"] == country][ALL_FEATURES + [TARGET]].dropna()

        # Identify constant columns (std == 0 → Pearson r is undefined)
        zero_var_cols = [c for c in sub_raw.columns
                         if sub_raw[c].std() == 0]
        sub  = sub_raw.drop(columns=zero_var_cols)
        corr = sub.corr(method="pearson")

        # Build subtitle noting any excluded features
        subtitle = ""
        if zero_var_cols:
            subtitle = f"Excluded (constant, std = 0): {', '.join(zero_var_cols)}"

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            corr,
            ax=ax,
            cmap="RdBu_r",
            vmin=-1, vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            linewidths=0.3,
            linecolor="white",
            square=True,
            cbar_kws={"label": "Pearson r", "shrink": 0.7},
        )
        ax.set_title(f"Correlation Matrix: {country}", fontsize=11, pad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
        _highlight_temp_axis_labels(ax, axis="both")  # orange bbox on temp labels
        fig.tight_layout()

        if subtitle:
            # Place note at the bottom of the figure, below the heatmap
            fig.text(0.5, -0.02, subtitle,
                     ha="center", va="top", fontsize=7.5,
                     color="#888888",
                     style="italic")

        fname = os.path.join(dirs["corr_pc"],
                             f"correlation_{safe_filename(country)}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: 01_correlation/correlation_per_country/"
              f"correlation_{safe_filename(country)}.png")

        # Collect feature–CO2 row for CSV
        for feat in ALL_FEATURES:
            if feat in corr.index:
                csv_rows.append({
                    "Country": country,
                    "Feature": feat,
                    "Pearson_r_with_CO2": round(corr.loc[feat, TARGET], 3),
                })
            else:
                # Excluded because constant (std = 0) — correlation undefined
                csv_rows.append({
                    "Country": country,
                    "Feature": feat,
                    "Pearson_r_with_CO2": np.nan,
                })

    pc_df = pd.DataFrame(csv_rows)
    return pc_df


def run_correlation(df: pd.DataFrame, dirs: dict) -> None:
    """Orchestrate all correlation outputs."""
    print("\n[1/5] Correlation analysis …")

    # Group-level: combined 3-panel chart
    corr_group = compute_feature_co2_corr(df)
    plot_correlation_by_latitude(corr_group, dirs)

    # Group-level: individual chart per latitude group (3 PNGs)
    plot_correlation_per_latitude_group(corr_group, dirs)

    # Save group-level CSV
    corr_group.reset_index().to_csv(
        os.path.join(dirs["corr"], "correlation_by_latitude_group.csv"), index=False)
    print("  Saved: 01_correlation/correlation_by_latitude_group.csv")

    # Per-country heatmaps + CSV
    pc_df = plot_correlation_per_country(df, dirs)
    pc_df.to_csv(os.path.join(dirs["corr_pc"], "correlation_per_country.csv"), index=False)
    print("  Saved: 01_correlation/correlation_per_country/correlation_per_country.csv")


# ============================================================
# 4. ACF ANALYSIS
# ============================================================

def plot_acf_per_country(df: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    """
    Plot ACF of Total CO₂ emissions for each country (appendix figures).
    Also builds and returns a summary DataFrame of ACF values at key lags.

    ACF is computed on the raw (non-differenced) CO₂ series.
    Dashed confidence bands at ±1.96/√T are drawn for visual reference.
    """
    countries   = sorted(df["Country Name"].unique())
    summary_rows = []

    for country in countries:
        series = (df[df["Country Name"] == country]
                  .sort_values("Year")["Total CO2 emissions"]
                  .dropna()
                  .values)

        n        = len(series)
        acf_vals = _acf_manual(series, nlags=ACF_NLAGS)
        conf_int = 1.96 / np.sqrt(n)           # approximate 95% CI half-width

        # ── ACF bar chart ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))

        lags = np.arange(len(acf_vals))
        ax.bar(lags, acf_vals,
               color=[COUNTRY_COLORS[country] if v > 0 else "#aaaaaa" for v in acf_vals],
               alpha=0.85, width=0.4)
        ax.axhline(0,          color="black",  linewidth=0.8)
        ax.axhline( conf_int,  color="#e15759", linewidth=1.2,
                   linestyle="--", label=f"95% CI (±{conf_int:.3f})")
        ax.axhline(-conf_int,  color="#e15759", linewidth=1.2, linestyle="--")

        # Annotate lag-0 explicitly
        ax.set_xticks(lags)
        ax.set_xticklabels([str(l) for l in lags], fontsize=9)
        ax.set_xlabel("Lag (years)", fontsize=10)
        ax.set_ylabel("Autocorrelation", fontsize=10)
        ax.set_title(f"ACF of Total CO₂ Emissions — {country}", fontsize=11)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        sns.despine(ax=ax)
        fig.tight_layout()

        fname = os.path.join(dirs["acf_pc"],
                             f"acf_{safe_filename(country)}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: 02_acf/acf_per_country/acf_{safe_filename(country)}.png")

        # Collect summary row
        row = {"Country": country, "n_obs": n, "CI_95pct": round(conf_int, 4)}
        for lag in ACF_SUMMARY_LAGS:
            row[f"ACF_lag{lag}"] = round(acf_vals[lag], 4) if lag < len(acf_vals) else np.nan
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def run_acf(df: pd.DataFrame, dirs: dict) -> None:
    """Orchestrate ACF outputs."""
    print("\n[2/5] ACF analysis …")
    summary_df = plot_acf_per_country(df, dirs)
    out = os.path.join(dirs["acf"], "acf_summary.csv")
    summary_df.to_csv(out, index=False)
    print("  Saved: 02_acf/acf_summary.csv")
    print("\n  ACF summary (lag 1–3):")
    print(summary_df.to_string(index=False))


# ============================================================
# 5. ADF TEST
# ============================================================

def run_adf(df: pd.DataFrame, dirs: dict) -> None:
    """
    Run Augmented Dickey-Fuller test on Total CO₂ emissions for each country.

    H₀: the series has a unit root (non-stationary).
    If p-value < 0.05, we reject H₀ → series is stationary.

    Results are saved as a single CSV table (one row per country).
    """
    print("\n[3/5] ADF stationarity test …")
    rows = []

    for country in sorted(df["Country Name"].unique()):
        series = (df[df["Country Name"] == country]
                  .sort_values("Year")["Total CO2 emissions"]
                  .dropna()
                  .values)

        adf_stat, p_val, used_lag, n_obs, crit_vals = _adfuller_manual(series)

        rows.append({
            "Country":            country,
            "ADF_statistic":      round(adf_stat, 4),
            "p_value":            round(p_val, 4),
            "used_lags":          used_lag,
            "n_obs":              n_obs,
            "Critical_1pct":      round(crit_vals["1%"],  4),
            "Critical_5pct":      round(crit_vals["5%"],  4),
            "Critical_10pct":     round(crit_vals["10%"], 4),
            "Stationary_5pct":    "Yes" if p_val < 0.05 else "No",
        })

    adf_df = pd.DataFrame(rows)
    out    = os.path.join(dirs["adf"], "adf_co2_results.csv")
    adf_df.to_csv(out, index=False)
    print("  Saved: 03_adf/adf_co2_results.csv")
    print("\n  ADF results:")
    print(adf_df[["Country","ADF_statistic","p_value","Stationary_5pct"]].to_string(index=False))


# ============================================================
# 6. TEMPERATURE DESCRIPTIVE STATISTICS
# ============================================================

def compute_temp_stats_per_country(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute per-country descriptive statistics for the four temperature features.
    Returns a dict { feature_name: DataFrame } — one DataFrame per variable.

    Each DataFrame has 6 rows (one per country) and columns:
      Country, mean, std, min, max

    Column names use the original feature names from the dataset.
    These are saved as separate CSVs (appendix tables).
    """
    result = {}
    for feat in TEMP_FEATURES:
        rows = []
        for country in sorted(df["Country Name"].unique()):
            vals = df[df["Country Name"] == country][feat].dropna()
            rows.append({
                "Country":    country,
                f"mean":      round(vals.mean(), 4) if len(vals) > 0 else np.nan,
                f"std":       round(vals.std(),  4) if len(vals) > 0 else np.nan,
                f"min":       round(vals.min(),  4) if len(vals) > 0 else np.nan,
                f"max":       round(vals.max(),  4) if len(vals) > 0 else np.nan,
                f"n_obs":     int(vals.count()),
            })
        result[feat] = pd.DataFrame(rows)
    return result


def compute_temp_stats_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute latitude-group descriptive statistics for the four temperature features.
    Returns one row per latitude group (3 rows total) — for the main text table.

    Groups are pooled (both countries' rows stacked) before computing stats,
    so mean/std reflect the combined distribution, not the average of two means.
    """
    rows = []
    for group, countries in LATITUDE_GROUPS.items():
        sub = df[df["Country Name"].isin(countries)]
        row = {"Latitude Group": group,
               "Countries": " / ".join(countries)}
        for feat in TEMP_FEATURES:
            vals = sub[feat].dropna()
            row[f"{feat}_mean"] = round(vals.mean(), 4) if len(vals) > 0 else np.nan
            row[f"{feat}_std"]  = round(vals.std(),  4) if len(vals) > 0 else np.nan
            row[f"{feat}_min"]  = round(vals.min(),  4) if len(vals) > 0 else np.nan
            row[f"{feat}_max"]  = round(vals.max(),  4) if len(vals) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_temp_feature_trends(df: pd.DataFrame, dirs: dict) -> None:
    """
    Trend line chart for each of the four temperature features.
    Moved here from 02_raw_eda.py so that temperature analysis is performed
    on the CLEANED dataset (post-imputation, post-anomaly treatment).

    One PNG per feature, all 6 countries overlaid with consistent colours.
    Saved to 04_temperature/trends/.
    """
    countries = sorted(df["Country Name"].unique())
    years     = sorted(df["Year"].unique())

    for feat in TEMP_FEATURES:
        fig, ax = plt.subplots(figsize=(10, 5))

        for country in countries:
            sub  = df[df["Country Name"] == country].sort_values("Year")
            ax.plot(
                sub["Year"].values,
                sub[feat].values,
                label=country,
                color=COUNTRY_COLORS[country],
                linewidth=1.8,
                marker="o",
                markersize=3,
            )

        ax.set_title(f"{feat}  –  Trend by Country (cleaned, 1990–2023)", fontsize=12)
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel(feat, fontsize=10)
        ax.set_xticks(years[::2])
        ax.set_xticklabels(years[::2], rotation=45, ha="right", fontsize=8)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7, ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        sns.despine(ax=ax)

        fig.tight_layout()
        safe_name = feat.replace(" ", "_").replace("/", "_")
        fname     = os.path.join(dirs["temp_trends"], f"trend_{safe_name}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: 04_temperature/trends/trend_{safe_name}.png")


def run_temperature(df: pd.DataFrame, dirs: dict) -> None:
    """Orchestrate temperature descriptive statistics and trend outputs."""
    print("\n[5/5] Temperature analysis …")

    # Trend plots (moved from 02_raw_eda.py)
    plot_temp_feature_trends(df, dirs)

    # Table 1: by latitude group — 3 rows, main text
    group_df  = compute_temp_stats_by_group(df)
    out_group = os.path.join(dirs["temp"], "temperature_by_latitude_group.csv")
    group_df.to_csv(out_group, index=False)
    print("  Saved: 04_temperature/temperature_by_latitude_group.csv")

    # Table 2–5: per country, one CSV per temperature variable — appendix
    per_country_dict = compute_temp_stats_per_country(df)
    for feat, feat_df in per_country_dict.items():
        safe = feat.replace(" ", "_")
        out  = os.path.join(dirs["temp"], f"temperature_per_country_{safe}.csv")
        feat_df.to_csv(out, index=False)
        print(f"  Saved: 04_temperature/temperature_per_country_{safe}.csv")

    # Console preview of group table
    print("\n  Temperature by latitude group (mean values):")
    preview_cols = (["Latitude Group", "Countries"] +
                    [f"{f}_mean" for f in TEMP_FEATURES])
    print(group_df[preview_cols].to_string(index=False))


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("04_clean_eda.py  –  Phase 2 EDA (clean data)")
    print("=" * 60)

    dirs = setup_output_dirs()

    df = load_data()
    print(f"\nLoaded: {INPUT_FILE}")
    print(f"  Shape    : {df.shape}")
    print(f"  Countries: {sorted(df['Country Name'].unique())}")
    print(f"  Years    : {df['Year'].min()} – {df['Year'].max()}")

    run_correlation(df, dirs)
    run_acf(df, dirs)
    run_adf(df, dirs)
    run_temperature(df, dirs)

    print(f"\n✅  All outputs saved to: {OUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
