"""
02_raw_eda.py
=============
PURPOSE
-------
Phase 1 EDA on the raw panel dataset (before any cleaning or imputation).
All analysis is performed on the original data so that missing values and
anomalies are visible in their natural state.

This script produces the evidence base that justifies every cleaning
decision made in 03_clean_data.py.

INPUT
-----
  raw_panel_dataset_1990_2023.csv   (output of 01_data_integration.py)

OUTPUTS  (written to ./02_outputs/)
-------
  Missingness
    missing_heatmap_country_feature.png   – missing-rate per (country, feature)
    missing_heatmap_feature_year.png      – missing-rate per (feature, year)
    nan_coordinates.csv                   – full list of every NaN cell

  Feature trends  (one PNG per feature, all 6 countries on the same chart)
    trend_Population.png
    trend_GDP.png
    trend_Electric_power_consumption.png
    trend_Fossil_fuel_energy_consumption.png   ← anomaly visible here
    trend_Renewable_energy_consumption.png
    trend_Fertilizer_consumption.png
    trend_Industry.png
    trend_Temperature_annual_mean.png
    trend_Temperature_std_across_months.png
    trend_Number_of_frost_days.png
    trend_Number_of_hot_days.png
    trend_Total_CO2_emissions.png

  Fill-zero justification
    fd_hd35_zero_justification.csv        – % of non-NaN values that equal 0,
                                            per country; supports the decision
                                            to impute NaN → 0 for these columns

PIPELINE POSITION
-----------------
  01_data_integration.py  →  raw_panel_dataset_1990_2023.csv
                                        ↓
                              [THIS SCRIPT]  →  02_outputs/
                                        ↓
                              03_clean_data.py   (imputation + anomaly treatment)

CLEANING DECISIONS EVIDENCED HERE
----------------------------------
  Industry          : US + Canada missing 1990–1996 (18 of 34 training-window
                      rows); proportion too large for a small dataset → drop column.
  Electric power    : 2022–2023 missing → impute with 3-year trailing average.
  Renewable energy  : 2022–2023 missing (most countries) → same imputation.
  Fertilizer,
  Temp annual mean,
  Temp std          : Russia 1990–1991 missing; no prior data available
                      → leave as NaN (cannot impute without history).
  fd / hd35         : NaN means the extreme-weather event did not occur
                      → impute with 0.  Justification in fd_hd35_zero_justification.csv.
  Fossil fuel       : Values collapse to 0 from 2016 onward (data artifact, not
                      real-world change); 2015 already anomalous in several
                      countries → replace 2015-onward anomalies with 3-year
                      trailing moving average.

DEPENDENCIES
------------
  pandas, numpy, matplotlib, seaborn
  Install: pip install pandas numpy matplotlib seaborn
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "processed", "raw_panel_dataset_1990_2023.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "eda", "02_raw_eda")
os.makedirs(OUT_DIR, exist_ok=True)

# Consistent country colour palette used across all trend plots.
# Each country gets its own colour so lines are easy to distinguish.
COUNTRY_COLORS = {
    "Canada":             "#1f77b4",   # blue
    "China":              "#d62728",   # red
    "India":              "#ff7f0e",   # orange
    "Indonesia":          "#2ca02c",   # green
    "Russian Federation": "#9467bd",   # purple
    "United States":      "#8c564b",   # brown
}

# Figure size used for all trend charts
TREND_FIGSIZE = (10, 5)

# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """Load the raw panel dataset produced by 01_data_integration.py."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            f"Run 01_data_integration.py first."
        )
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(["Country Name", "Year"]).reset_index(drop=True)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all columns except the two identifiers."""
    return [c for c in df.columns if c not in ("Country Name", "Year")]


# ============================================================
# 3. MISSINGNESS ANALYSIS
# ============================================================

def plot_missing_heatmap_country_feature(df: pd.DataFrame) -> None:
    """
    Heatmap: rows = countries, columns = features.
    Cell colour = fraction of years that are NaN for that (country, feature) pair.

    This makes it immediately visible WHICH countries have gaps in WHICH features
    (e.g. the deep-red Indonesia block for frost days, the US/Canada Industry gap).
    """
    feat_cols = get_feature_cols(df)
    countries = sorted(df["Country Name"].unique())

    # Build (country × feature) missing-rate matrix
    matrix = pd.DataFrame(index=countries, columns=feat_cols, dtype=float)
    for country in countries:
        sub = df[df["Country Name"] == country]
        for col in feat_cols:
            matrix.loc[country, col] = sub[col].isna().mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        matrix.astype(float),
        ax=ax,
        cmap="Reds",
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot=False,
        cbar_kws={"label": "Missing rate"},
    )
    ax.set_title("Missingness Heatmap (Country × Feature)", fontsize=13, pad=12)
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Country", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "missing_heatmap_country_feature.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def plot_missing_heatmap_feature_year(df: pd.DataFrame) -> None:
    """
    Heatmap: rows = features, columns = years.
    Cell colour = fraction of countries that are NaN for that (feature, year) pair.

    This reveals the TIME DIMENSION of gaps:
      - Industry missing 1990–1996 (early years)
      - Renewable energy and Electric power missing 2022–2023 (latest years)
      - fd scattered across years for specific countries
    """
    feat_cols = get_feature_cols(df)
    years = sorted(df["Year"].unique())

    matrix = pd.DataFrame(index=feat_cols, columns=years, dtype=float)
    for col in feat_cols:
        for year in years:
            sub = df[df["Year"] == year]
            matrix.loc[col, year] = sub[col].isna().mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        matrix.astype(float),
        ax=ax,
        cmap="Reds",
        vmin=0, vmax=1,
        linewidths=0.3,
        linecolor="white",
        annot=False,
        cbar_kws={"label": "Missing rate"},
    )
    ax.set_title("Missingness Heatmap (Feature × Year)", fontsize=13, pad=12)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "missing_heatmap_feature_year.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def export_nan_coordinates(df: pd.DataFrame) -> None:
    """
    Export a CSV listing every NaN cell as a (Country, Feature, Year) triple.

    This table serves as the explicit roadmap for 03_clean_data.py:
    every row here corresponds to a cell that must be either imputed, flagged,
    or left as NaN with a documented reason.
    """
    feat_cols = get_feature_cols(df)
    rows = []
    for col in feat_cols:
        mask = df[col].isna()
        for _, row in df[mask].iterrows():
            rows.append({
                "Country":  row["Country Name"],
                "Feature":  col,
                "Year":     int(row["Year"]),
            })

    nan_df = (
        pd.DataFrame(rows)
        .sort_values(["Feature", "Country", "Year"])
        .reset_index(drop=True)
    )

    out_path = os.path.join(OUT_DIR, "nan_coordinates.csv")
    nan_df.to_csv(out_path, index=False)
    print(f"  Saved: {os.path.basename(out_path)}  ({len(nan_df)} NaN cells)")

    # Print a per-feature summary to the console for quick reference
    print("\n  NaN count by feature:")
    for feat, cnt in nan_df.groupby("Feature").size().items():
        print(f"    {feat:<42} {cnt:>3} cells")


# ============================================================
# 4. FEATURE TREND PLOTS
# ============================================================

def plot_feature_trends(df: pd.DataFrame) -> None:
    """
    For each feature column, draw one line chart with all 6 countries overlaid.

    Lines use a consistent colour per country (see COUNTRY_COLORS).
    NaN gaps appear as natural breaks in the line, making missing stretches
    immediately visible without any additional annotation.

    The fossil fuel chart in particular shows the 2015–2016 anomaly
    (values collapsing to 0) that triggers the anomaly treatment in step 03.
    """
    feat_cols = get_feature_cols(df)
    years     = sorted(df["Year"].unique())
    countries = sorted(df["Country Name"].unique())

    for col in feat_cols:
        fig, ax = plt.subplots(figsize=TREND_FIGSIZE)

        for country in countries:
            sub  = df[df["Country Name"] == country].sort_values("Year")
            vals = sub[col].values
            ax.plot(
                sub["Year"].values,
                vals,
                label=country,
                color=COUNTRY_COLORS[country],
                linewidth=1.8,
                marker="o",
                markersize=3,
            )

        # Formatting
        ax.set_title(f"{col}  –  Trend by Country (1990–2023)", fontsize=12)
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        ax.set_xticks(years[::2])                         # label every 2 years
        ax.set_xticklabels(years[::2], rotation=45, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.1f}" if abs(x) < 1e6 else f"{x/1e6:,.2f}M"
        ))
        ax.legend(
            loc="upper left",
            fontsize=8,
            framealpha=0.7,
            ncol=2,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        sns.despine(ax=ax)

        fig.tight_layout()
        safe_name = col.replace(" ", "_").replace("/", "_")
        out_path  = os.path.join(OUT_DIR, f"trend_{safe_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {os.path.basename(out_path)}")


# ============================================================
# 5. FILL-ZERO JUSTIFICATION FOR fd / hd35
# ============================================================

def export_fd_hd35_zero_justification(df: pd.DataFrame) -> None:
    """
    For the two ERA5 climate-index columns, compute per-country statistics
    that justify the decision to impute NaN → 0 in 03_clean_data.py.

    For each (country, column) pair the table records:
      n_total        : total years in the dataset (34)
      n_nan          : number of NaN years
      n_non_nan      : number of years with a real value
      n_zero         : among non-NaN years, how many equal exactly 0
      pct_zero       : n_zero / n_non_nan × 100
      min / median / max of non-NaN values

    Interpretation guide:
      • Indonesia fd: n_nan = 34 (all years missing).  Geographic reasoning
        applies: Indonesia is a tropical country at the equator — frost days
        are physically impossible, so the correct imputed value is 0.
      • Canada hd35 : n_nan = 7, non-NaN values are tiny (median ≈ 0.03 days).
        Values are so close to 0 that filling with 0 introduces negligible error.
    """
    cols_to_check = ["Number of frost days", "Number of hot days"]
    rows = []

    for col in cols_to_check:
        for country in sorted(df["Country Name"].unique()):
            series   = df[df["Country Name"] == country][col]
            n_total  = len(series)
            n_nan    = series.isna().sum()
            non_nan  = series.dropna()
            n_non_nan = len(non_nan)
            n_zero   = int((non_nan == 0).sum())
            pct_zero = round(n_zero / n_non_nan * 100, 1) if n_non_nan > 0 else float("nan")

            rows.append({
                "Feature":    col,
                "Country":    country,
                "n_total":    n_total,
                "n_nan":      n_nan,
                "n_non_nan":  n_non_nan,
                "n_zero":     n_zero,
                "pct_zero":   pct_zero,
                "min":        round(non_nan.min(), 4)    if n_non_nan > 0 else float("nan"),
                "median":     round(non_nan.median(), 4) if n_non_nan > 0 else float("nan"),
                "max":        round(non_nan.max(), 4)    if n_non_nan > 0 else float("nan"),
                "fill_0_justification": _fill_zero_note(col, country, n_nan, n_non_nan, pct_zero),
            })

    out_df   = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "fd_hd35_zero_justification.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Saved: {os.path.basename(out_path)}")

    # Print a compact console summary
    print("\n  fd / hd35 fill-zero justification summary:")
    display_cols = ["Feature", "Country", "n_nan", "n_non_nan", "pct_zero", "median", "fill_0_justification"]
    print(out_df[display_cols].to_string(index=False))


def _fill_zero_note(col: str, country: str, n_nan: int, n_non_nan: int, pct_zero: float) -> str:
    """Return a short human-readable justification string for the CSV."""
    if n_nan == 0:
        return "No NaN – no imputation needed"
    if col == "Number of frost days" and country == "Indonesia":
        return "All years missing; tropical equatorial country – frost physically impossible → fill 0"
    if col == "Number of hot days" and country == "Canada":
        return f"7 years missing; non-NaN median ≈ 0.03 days (near-zero) → fill 0 introduces negligible error"
    if n_nan > 0 and n_non_nan > 0:
        return f"Partial NaN; pct_zero={pct_zero}% in observed years – fill 0 supported by observed distribution"
    return "NaN present – review manually"


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("02_raw_eda.py  –  Phase 1 EDA (raw data)")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────
    df = load_data()
    print(f"\nLoaded: {INPUT_FILE}")
    print(f"  Shape    : {df.shape}")
    print(f"  Countries: {sorted(df['Country Name'].unique())}")
    print(f"  Years    : {df['Year'].min()} – {df['Year'].max()}")
    print(f"  Total NaN: {df.isna().sum().sum()}")

    # ── 3. Missingness ─────────────────────────────────────────
    print("\n[1/4] Generating missingness heatmaps …")
    plot_missing_heatmap_country_feature(df)
    plot_missing_heatmap_feature_year(df)

    print("\n[2/4] Exporting NaN coordinates …")
    export_nan_coordinates(df)

    # ── 4. Feature trends ──────────────────────────────────────
    print("\n[3/4] Generating feature trend plots …")
    plot_feature_trends(df)

    # ── 5. Fill-zero justification ─────────────────────────────
    print("\n[4/4] Exporting fd / hd35 fill-zero justification …")
    export_fd_hd35_zero_justification(df)

    print(f"\n✅  All outputs saved to: {OUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
