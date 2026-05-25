"""
03_clean_data.py
================
PURPOSE
-------
Applies all cleaning and imputation decisions that were documented and
evidenced in 02_raw_eda.py.  Produces the final model-ready dataset.

INPUT
-----
  raw_panel_dataset_1990_2023.csv       (output of 01_data_integration.py)

OUTPUTS
-------
  dataset_recursive_3yr_avg_drop_Industry.csv   ← used by all model scripts
  cleaning_log.csv                              ← audit trail of every change

CLEANING STEPS (in order)
--------------------------
  1. Drop Industry
       Entire column removed.  US and Canada are missing 1990–1996 (18 / 34
       rows in the training window), which is too large a gap to impute
       meaningfully given the limited dataset size.

  2. Fossil fuel energy consumption  –  anomaly treatment
       The World Bank data for this indicator collapses to 0 from 2016 onward
       (data artifact, not a real-world change).  2015 is also anomalous in
       three countries (values drop by > 30 pp in a single year).

       Flagged as NaN first, then filled with a RECURSIVE 3-year trailing
       moving average:
         • Canada, China, United States  : 2016–2023 → NaN → fill
         • India, Indonesia, Russia       : 2015–2023 → NaN → fill

       "Recursive" means already-imputed values feed into subsequent years:
         fill(2015) = mean(2012, 2013, 2014)           [India/Indonesia/Russia only]
         fill(2016) = mean(2013, 2014, filled_2015)
         fill(2017) = mean(2014, filled_2015, filled_2016)
         ...and so on through 2023.

  3. Renewable energy consumption
       All 6 countries are missing 2022 and 2023.
       Recursive 3-year trailing fill:
         fill(2022) = mean(2019, 2020, 2021)
         fill(2023) = mean(2020, 2021, filled_2022)

  4. Electric power consumption
       Only Russian Federation 2023 is missing.
         fill(2023) = mean(2020, 2021, 2022)

  5. Number of frost days  /  Number of hot days  →  fill NaN with 0
       Indonesia  fd  : all 34 years are NaN; geographically a tropical
                        equatorial country where frost is physically impossible.
       Canada     hd35: 7 years NaN; non-NaN values have median ≈ 0.03 days
                        (near-zero), so imputing 0 introduces negligible error.

  6. Intentionally NOT imputed  (left as NaN)
       Russian Federation, 1990–1991:
         Fertilizer consumption, Temperature annual mean,
         Temperature std across months.
       Reason: no prior observations exist to compute a meaningful average.

PIPELINE POSITION
-----------------
  02_raw_eda.py  →  [THIS SCRIPT]  →  dataset_recursive_3yr_avg_drop_Industry.csv
                                                    ↓
                                          04_clean_eda.py
                                          Model scripts (ARIMAX, SVR, XGBoost, LSTCN)

DEPENDENCIES
------------
  pandas, numpy
  Install: pip install pandas numpy
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.simplefilter("ignore", UserWarning)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(SCRIPT_DIR, "..", "processed", "raw_panel_dataset_1990_2023.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "processed", "dataset_recursive_3yr_avg_drop_Industry.csv")
LOG_FILE    = os.path.join(SCRIPT_DIR, "..", "processed", "cleaning_log.csv")

# ── Fossil fuel anomaly configuration ────────────────────────
# Countries where ONLY 2016–2023 are zeroed out (2015 is kept as-is)
FF_ZERO_ONLY = ["Canada", "China", "United States"]

# Countries where 2015 is ALSO anomalous and removed alongside 2016–2023
FF_ZERO_AND_2015 = ["India", "Indonesia", "Russian Federation"]

# ── Columns intentionally left as NaN ─────────────────────────
# These cells have no prior observations to draw from.
NO_IMPUTE = [
    ("Russian Federation", "Fertilizer consumption",         [1990, 1991]),
    ("Russian Federation", "Temperature annual mean",        [1990, 1991]),
    ("Russian Federation", "Temperature std across months",  [1990, 1991]),
]


# ============================================================
# 2. AUDIT LOG HELPER
# ============================================================

class CleaningLog:
    """
    Collects a row-level record of every value that is changed.

    Each entry records:
      country, feature, year, original_value, new_value, method
    """
    def __init__(self):
        self._rows = []

    def record(self, country: str, feature: str, year: int,
               original, new_value, method: str) -> None:
        self._rows.append({
            "country":        country,
            "feature":        feature,
            "year":           year,
            "original_value": original,
            "new_value":      round(float(new_value), 6) if pd.notna(new_value) else np.nan,
            "method":         method,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def save(self, path: str) -> None:
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"  Saved cleaning log: {os.path.basename(path)}  ({len(df)} changes recorded)")


# ============================================================
# 3. CORE IMPUTATION FUNCTION
# ============================================================

def recursive_3yr_fill(series: pd.Series, years_to_fill: list[int],
                       log: CleaningLog, country: str, feature: str) -> pd.Series:
    """
    Fill specified years using a recursive 3-year trailing moving average.

    "Recursive" means previously imputed values immediately feed into the
    next year's average, so the imputation chain is:

      fill(t)   = mean( series[t-3], series[t-2], series[t-1] )
      fill(t+1) = mean( series[t-2], series[t-1], fill(t)     )   ← uses imputed value
      fill(t+2) = mean( series[t-1], fill(t),     fill(t+1)   )
      ...

    Parameters
    ----------
    series       : pd.Series indexed by Year (sorted ascending)
    years_to_fill: list of years to impute, in chronological order
    log          : CleaningLog instance for audit trail
    country      : country name (for log only)
    feature      : feature name (for log only)

    Returns
    -------
    pd.Series with imputed values filled in.
    """
    s = series.copy()

    for year in sorted(years_to_fill):
        # Identify the three years immediately before `year`
        prev_years = [year - 3, year - 2, year - 1]
        prev_vals  = [s.get(y, np.nan) for y in prev_years]
        valid_vals = [v for v in prev_vals if pd.notna(v)]

        if len(valid_vals) == 0:
            # No history available – leave as NaN and warn
            print(f"    ⚠  {country} | {feature} | {year}: "
                  f"no prior values available, leaving as NaN")
            continue

        original  = s.get(year, np.nan)
        imputed   = float(np.mean(valid_vals))
        s[year]   = imputed

        log.record(
            country=country,
            feature=feature,
            year=year,
            original=original,
            new_value=imputed,
            method=f"recursive_3yr_avg (used years {[y for y,v in zip(prev_years, prev_vals) if pd.notna(v)]})"
        )

    return s


def set_to_nan(df: pd.DataFrame, country: str, feature: str,
               years: list[int], log: CleaningLog) -> pd.DataFrame:
    """
    Set specific (country, feature, year) cells to NaN and record in log.
    Used as the first step of anomaly treatment before imputation.
    """
    mask = (df["Country Name"] == country) & (df["Year"].isin(years))
    for _, row in df[mask].iterrows():
        log.record(
            country=country,
            feature=feature,
            year=int(row["Year"]),
            original=row[feature],
            new_value=np.nan,
            method="flagged_as_anomaly → set to NaN before imputation"
        )
    df.loc[mask, feature] = np.nan
    return df


# ============================================================
# 4. CLEANING STEPS
# ============================================================

def step1_drop_industry(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """
    Step 1: Remove the Industry column.

    Rationale: US and Canada are missing 1990–1996 (18 / 34 rows each).
    Imputing this proportion of data in a small panel would introduce too
    much noise.  The column is dropped entirely so the remaining 33-year
    window (1991–2023) is not lost due to listwise deletion.
    """
    print("  Step 1 | Dropping 'Industry' column …")

    for _, row in df[df["Industry"].notna()].iterrows():
        log.record(
            country=row["Country Name"],
            feature="Industry",
            year=int(row["Year"]),
            original=row["Industry"],
            new_value=np.nan,
            method="column_dropped"
        )

    df = df.drop(columns=["Industry"])
    print(f"          Dropped.  Remaining columns: {list(df.columns)}")
    return df


def step2_fossil_fuel_anomaly(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """
    Step 2: Fix the fossil fuel energy consumption data artifact.

    The World Bank series for this indicator shows values collapsing to 0
    from 2016 onward for all 6 countries (clear data artifact – see
    trend_Fossil_fuel_energy_consumption.png from 02_raw_eda.py).
    Additionally, India, Indonesia, and Russian Federation show an extreme
    drop already in 2015, so that year is also removed.

    Treatment:
      (a) Flag affected cells as NaN.
      (b) Recursively fill each NaN year with the 3-year trailing average,
          working forward in time so each imputed value feeds the next.
    """
    feat = "Fossil fuel energy consumption"
    print(f"  Step 2 | Fixing '{feat}' anomalies …")

    # (a) Flag anomalous values as NaN
    for country in FF_ZERO_ONLY:
        years_to_flag = list(range(2016, 2024))
        df = set_to_nan(df, country, feat, years_to_flag, log)
        print(f"          {country}: flagged 2016–2023 as NaN")

    for country in FF_ZERO_AND_2015:
        years_to_flag = list(range(2015, 2024))
        df = set_to_nan(df, country, feat, years_to_flag, log)
        print(f"          {country}: flagged 2015–2023 as NaN")

    # (b) Recursive 3-year fill for each affected country
    for country in FF_ZERO_ONLY + FF_ZERO_AND_2015:
        # Determine which years still need filling after step (a)
        sub    = df[df["Country Name"] == country].set_index("Year")[feat]
        to_fill = sorted(sub[sub.isna()].index.tolist())

        if not to_fill:
            continue

        filled = recursive_3yr_fill(sub, to_fill, log, country, feat)

        # Write imputed values back to the main DataFrame
        for year, val in filled.items():
            if pd.notna(val):
                df.loc[(df["Country Name"] == country) & (df["Year"] == year), feat] = val

        print(f"          {country}: filled years {to_fill[0]}–{to_fill[-1]} with recursive 3yr avg")

    return df


def step3_renewable_energy(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """
    Step 3: Impute Renewable energy consumption for 2022–2023.

    All 6 countries are missing these two years.  Recursive 3-year fill:
      fill(2022) = mean(2019, 2020, 2021)
      fill(2023) = mean(2020, 2021, filled_2022)
    """
    feat      = "Renewable energy consumption"
    to_fill   = [2022, 2023]
    countries = sorted(df["Country Name"].unique())

    print(f"  Step 3 | Imputing '{feat}' for 2022–2023 …")

    for country in countries:
        sub    = df[df["Country Name"] == country].set_index("Year")[feat]
        filled = recursive_3yr_fill(sub, to_fill, log, country, feat)

        for year in to_fill:
            val = filled.get(year)
            if pd.notna(val):
                df.loc[(df["Country Name"] == country) & (df["Year"] == year), feat] = val

    print(f"          Done ({len(countries)} countries × 2 years)")
    return df


def step4_electric_power(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """
    Step 4: Impute Electric power consumption for Russian Federation 2023.

      fill(2023) = mean(2020, 2021, 2022)
    """
    feat    = "Electric power consumption"
    country = "Russian Federation"
    to_fill = [2023]

    print(f"  Step 4 | Imputing '{feat}' for {country} 2023 …")

    sub    = df[df["Country Name"] == country].set_index("Year")[feat]
    filled = recursive_3yr_fill(sub, to_fill, log, country, feat)

    val = filled.get(2023)
    if pd.notna(val):
        df.loc[(df["Country Name"] == country) & (df["Year"] == 2023), feat] = val
        print(f"          {country} 2023 → {val:.4f}")

    return df


def step5_fill_zero_climate(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """
    Step 5: Impute NaN → 0 for frost days and hot days.

    Number of frost days  –  Indonesia (all 34 years NaN):
      Indonesia is a tropical equatorial country.  Frost days are
      physically impossible, so 0 is the correct value, not a guess.

    Number of hot days  –  Canada (7 scattered years NaN):
      Canada's non-NaN values have median ≈ 0.03 days/year.  The values
      are so close to 0 that imputing 0 introduces negligible error.
      (See fd_hd35_zero_justification.csv from 02_raw_eda.py.)
    """
    fill_specs = [
        ("Indonesia", "Number of frost days"),
        ("Canada",    "Number of hot days"),
    ]

    print("  Step 5 | Filling NaN → 0 for fd / hd35 …")

    for country, feat in fill_specs:
        mask = (df["Country Name"] == country) & df[feat].isna()
        affected = df[mask]
        for _, row in affected.iterrows():
            log.record(
                country=country,
                feature=feat,
                year=int(row["Year"]),
                original=np.nan,
                new_value=0.0,
                method="fill_zero (event physically absent or near-zero observed distribution)"
            )
        df.loc[mask, feat] = 0.0
        print(f"          {country} | {feat}: {len(affected)} cells → 0")

    return df


def step6_verify_no_impute(df: pd.DataFrame) -> None:
    """
    Step 6: Verify that the intentionally un-imputed cells are still NaN.

    Russian Federation 1990–1991 for Fertilizer consumption,
    Temperature annual mean, and Temperature std across months are left
    as NaN because no prior observations exist to build an estimate.

    This step just prints a confirmation so the final dataset can be
    audited against the documented decisions.
    """
    print("  Step 6 | Verifying intentional NaN cells (Russian Federation 1990–1991) …")
    all_ok = True
    for country, feat, years in NO_IMPUTE:
        for year in years:
            val = df.loc[(df["Country Name"] == country) & (df["Year"] == year), feat]
            is_nan = val.isna().all()
            status = "✅ NaN" if is_nan else f"❌ UNEXPECTED VALUE: {val.values}"
            print(f"          {country} | {feat} | {year}: {status}")
            if not is_nan:
                all_ok = False
    if all_ok:
        print("          All intentional NaN cells confirmed.")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("03_clean_data.py  –  Cleaning & Imputation")
    print("=" * 60)

    # ── Load raw data ──────────────────────────────────────────
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            f"Run 01_data_integration.py first."
        )
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(["Country Name", "Year"]).reset_index(drop=True)
    print(f"\nLoaded: {INPUT_FILE}")
    print(f"  Shape before cleaning : {df.shape}")
    print(f"  Total NaN before      : {df.isna().sum().sum()}")

    log = CleaningLog()

    # ── Apply cleaning steps ───────────────────────────────────
    print()
    df = step1_drop_industry(df, log)
    print()
    df = step2_fossil_fuel_anomaly(df, log)
    print()
    df = step3_renewable_energy(df, log)
    print()
    df = step4_electric_power(df, log)
    print()
    df = step5_fill_zero_climate(df, log)
    print()
    step6_verify_no_impute(df)

    # ── Final summary ──────────────────────────────────────────
    print(f"\n  Shape after cleaning  : {df.shape}")
    remaining_nan = df.isna().sum()
    remaining_nan = remaining_nan[remaining_nan > 0]
    if len(remaining_nan) > 0:
        print("  Remaining NaN per column (expected – intentional):")
        for col, cnt in remaining_nan.items():
            print(f"    {col:<42} {cnt:>2} cells")
    else:
        print("  No remaining NaN.")

    # ── Save outputs ───────────────────────────────────────────
    print()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved dataset : {os.path.basename(OUTPUT_FILE)}")
    log.save(LOG_FILE)

    print(f"\n✅  Cleaning complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
