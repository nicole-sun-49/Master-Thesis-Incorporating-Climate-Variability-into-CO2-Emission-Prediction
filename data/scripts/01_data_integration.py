"""
01_data_integration.py
======================
PURPOSE
-------
Consolidates all raw source files into a single flat panel CSV that covers
6 countries × 34 years (1990–2023).  This is the *only* script that touches
the raw xlsx files; every downstream script reads the output CSV instead.

OUTPUT
------
  raw_panel_dataset_1990_2023.csv
    - One row per (Country Name, Year)  →  204 rows total
    - 14 columns: 2 identifiers + 11 features + 1 target
    - All values are as-downloaded: NO cleaning, NO imputation
    - NaN is preserved wherever the source data has a gap
    - This file feeds directly into 02_raw_eda.py (Phase 1 EDA)

PIPELINE POSITION
-----------------
  Raw xlsx files  →  [THIS SCRIPT]  →  raw_panel_dataset_1990_2023.csv
                                              ↓
                                       02_raw_eda.py          (missingness / anomaly check)
                                              ↓
                                       03_clean_data.py       (imputation, anomaly treatment)
                                              ↓
                                       04_clean_eda.py        (correlation, ACF/ADF, temp analysis)
                                              ↓
                                       Model scripts          (ARIMAX, SVR, XGBoost, LSTCN)

DATA SOURCES INTEGRATED
-----------------------
  A. World Bank WDI  –  8 indicators (separate xlsx per indicator)
       Population, GDP, Electric power consumption,
       Fossil fuel energy consumption, Renewable energy consumption,
       Fertilizer consumption, Industry value-added (% GDP),
       Total CO2 emissions  (← target variable)

  B. FAOSTAT Surface Temperature  –  monthly temperature-change anomalies
       Aggregated here to two annual features:
         • Temperature annual mean       (mean of 12 monthly values)
         • Temperature std across months (population std of 12 monthly values)

  C. ERA5 Reanalysis  –  annual climate extreme indices
       • Number of frost days  (sheet "fd"   in the ERA5 xlsx)
       • Number of hot days    (sheet "hd35" in the ERA5 xlsx)

HOW TO RUN
----------
  1. Place this script and all raw xlsx files in the same folder.
  2. Run:  python 01_data_integration.py
  3. Output CSV is written to the same folder.

DEPENDENCIES
------------
  pandas, numpy, openpyxl
  Install:  pip install pandas numpy openpyxl
"""

import os
import re
import glob
import warnings
import pandas as pd
import numpy as np

warnings.simplefilter("ignore", UserWarning)


# ============================================================
# 1. CONFIGURATION
# ============================================================
# All settings that might need to change are gathered here so
# you never have to hunt through function bodies to adjust paths,
# years, or country lists.

# Root folder for all raw xlsx files.
# Defaults to the directory that contains this script, so the
# script works regardless of where it is called from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "raw")

# ── A. World Bank indicator files ────────────────────────────
# Keys   = canonical column names used in the output CSV.
# Values = glob patterns; the script picks the first match so
#          the exact version-number suffix in the filename does
#          not matter (e.g. "…v2_174095.xlsx" vs "…v2_999.xlsx").
#
# NOTE: "Total CO2 emissions" is deliberately placed last so it
#       appears as the final column (= target variable) in the CSV.
WB_FILES = {
    "Population":                     "API_SP.POP.TOTL_DS2_en_excel_v2_*.xlsx",
    "GDP":                            "API_NY.GDP.MKTP.CD_DS2_en_excel_v2_*.xlsx",
    "Electric power consumption":     "API_EG.USE.ELEC.KH.PC_DS2_en_excel_v2_*.xlsx",
    "Industry":                       "API_NV.IND.TOTL.ZS_DS2_en_excel_v2_*.xlsx",
    "Fossil fuel energy consumption": "API_EG.USE.COMM.FO.ZS_DS2_en_excel_v2_*.xlsx",
    "Renewable energy consumption":   "API_EG.FEC.RNEW.ZS_DS2_en_excel_v2_*.xlsx",
    "Fertilizer consumption":         "API_AG.CON.FERT.ZS_DS2_en_excel_v2_*.xlsx",
    "Total CO2 emissions":            "API_EN.GHG.CO2.MT.CE.AR5_DS2_en_excel_v2_*.xlsx"
}

# ── B. FAOSTAT temperature file ───────────────────────────────
# Monthly surface-temperature anomalies (°C relative to a baseline period).
# The file contains data for many countries and both "Temperature change"
# and "Standard Deviation" elements; only "Temperature change" is used here.
FAO_FILE_PATTERN = "FAOSTAT_data_en_1-19-2026.xlsx"

# ── C. ERA5 climate-index file ────────────────────────────────
# One Excel file with two sheets:
#   "fd"   → annual count of frost days  (Tmin < 0 °C)
#   "hd35" → annual count of hot days    (Tmax > 35 °C)
# Dict maps sheet name → output column name.
ERA5_FILE_PATTERN = "era5-x0.25_timeseries_fd,hd35_timeseries_annual_1950-2023_mean_historical_era5_x0.25_mean.xlsx"
ERA5_SHEETS = {
    "fd":   "Number of frost days",
    "hd35": "Number of hot days",
}

# ── Study scope ───────────────────────────────────────────────
YEAR_START = 1990
YEAR_END   = 2023
YEARS      = list(range(YEAR_START, YEAR_END + 1))

# The six countries studied in the thesis.
COUNTRIES = [
    "Canada",
    "China",
    "India",
    "Indonesia",
    "Russian Federation",
    "United States",
]

# ── Country-name normalisation map ────────────────────────────
# Each raw source uses slightly different country-name strings.
# This dict maps every known variant → the canonical name used
# in the COUNTRIES list above, so all three sources join cleanly.
#
# Known variants per source:
#   World Bank  – already uses canonical names; no mapping needed.
#   FAO         – uses "China, mainland" and "United States of America".
#   ERA5        – uses "United States" (same as canonical; listed for
#                 explicitness), all others already match.
COUNTRY_NAME_MAP = {
    # FAO variants → canonical
    "China, mainland":          "China",
    "United States of America": "United States",
    # ERA5 variants → canonical  (identity entries make the mapping self-documenting)
    "Russian Federation":       "Russian Federation",
    "Canada":                   "Canada",
    "United States":            "United States",
    "China":                    "China",
    "India":                    "India",
    "Indonesia":                "Indonesia",
}

# Path for the output CSV (same folder as this script).
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "processed", "raw_panel_dataset_1990_2023.csv")


# ============================================================
# 2. SHARED HELPER FUNCTIONS
# ============================================================

def find_file(pattern: str) -> str:
    """
    Locate a raw data file by glob pattern inside DATA_DIR.

    Uses glob so the script is robust to minor filename changes
    (e.g. World Bank version-suffix numbers).  Raises a clear
    FileNotFoundError if nothing is found so the user knows
    exactly which file is missing.
    """
    hits = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not hits:
        raise FileNotFoundError(
            f"No file found matching pattern: {pattern}\n"
            f"Searched in: {DATA_DIR}\n"
            f"Make sure all raw xlsx files are in the same folder as this script."
        )
    return hits[0]   # use the first (alphabetically earliest) match


def normalize_country(name) -> str:
    """
    Convert a raw-file country name to the canonical form used in COUNTRIES.

    Returns the input unchanged if it is not in COUNTRY_NAME_MAP, so the
    function is safe to apply to any column without accidentally mangling
    unrelated rows.
    """
    if pd.isna(name):
        return name
    name = str(name).strip()
    return COUNTRY_NAME_MAP.get(name, name)


def make_country_year_scaffold() -> pd.DataFrame:
    """
    Build a complete (Country Name × Year) grid for the study scope.

    All three data sources are left-joined onto this scaffold so the
    final panel always has exactly one row per country-year combination,
    even when a source is missing an observation (NaN is inserted).

    Returns a DataFrame with columns: Country Name, Year
    """
    rows = [(country, year) for country in COUNTRIES for year in YEARS]
    return pd.DataFrame(rows, columns=["Country Name", "Year"])


# ============================================================
# 3. SOURCE-SPECIFIC READER FUNCTIONS
# ============================================================
# Each function is responsible for exactly one data source.
# They all return a long-format DataFrame keyed on
# (Country Name, Year) so they can be left-joined onto the scaffold.

# ── 3a. World Bank WDI ──────────────────────────────────────
def read_worldbank_indicator(filepath: str, indicator_name: str) -> pd.DataFrame:
    """
    Read one World Bank WDI Excel file and return it in long format.

    World Bank file structure:
      - Sheet name : "Data"
      - Real header: Excel row 4  →  pandas header=3
      - Columns    : Country Name | Country Code | Indicator Name |
                     Indicator Code | 1960 | 1961 | … | 2023
      - Year columns are numeric (int) or occasionally string integers.

    Steps:
      1. Read the "Data" sheet with header=3.
      2. Identify year columns that fall within [YEAR_START, YEAR_END].
      3. Melt from wide to long  →  one row per (country, year).
      4. Normalise country names and filter to the 6 study countries.

    Returns: DataFrame with columns [Country Name, Year, <indicator_name>]
    """
    df = pd.read_excel(filepath, sheet_name="Data", engine="openpyxl", header=3)
    df.columns = [str(c).strip() for c in df.columns]

    if "Country Name" not in df.columns:
        raise KeyError(
            f"'Country Name' column not found in {filepath}.\n"
            f"First 20 columns: {list(df.columns)[:20]}"
        )

    # Identify year columns: accept both int 1990 and string "1990"
    year_cols = []
    for c in df.columns:
        try:
            y = int(str(c).strip())
            if YEAR_START <= y <= YEAR_END:
                year_cols.append(c)
        except ValueError:
            pass   # skip non-numeric column names (e.g. "Country Code")

    if not year_cols:
        raise KeyError(
            f"No year columns found in {filepath}. "
            f"Expected integer column headers between {YEAR_START} and {YEAR_END}."
        )

    # Melt: wide → long
    long = (
        df[["Country Name"] + year_cols]
        .melt(id_vars="Country Name", var_name="Year", value_name=indicator_name)
    )
    long["Year"] = long["Year"].astype(str).str.strip().astype(int)
    long["Country Name"] = long["Country Name"].apply(normalize_country)

    # Keep only the 6 study countries
    long = long[long["Country Name"].isin(COUNTRIES)].copy()
    return long[["Country Name", "Year", indicator_name]]


def load_all_worldbank() -> pd.DataFrame:
    """
    Read every World Bank indicator file and merge them into one wide panel.

    Iterates over WB_FILES, reads each file with read_worldbank_indicator(),
    and left-joins each result onto the country-year scaffold so every
    country-year row exists even if an indicator has no data for that cell.

    Returns: wide DataFrame with columns
             [Country Name, Year, Population, GDP, …, Total CO2 emissions]
    """
    panel = make_country_year_scaffold()
    for indicator, pattern in WB_FILES.items():
        fpath = find_file(pattern)
        print(f"  [WB] {indicator}  ←  {os.path.basename(fpath)}")
        ind_df = read_worldbank_indicator(fpath, indicator)
        panel = panel.merge(ind_df, on=["Country Name", "Year"], how="left")
    return panel


# ── 3b. FAOSTAT Surface Temperature ─────────────────────────
def load_faostat_temperature() -> pd.DataFrame:
    """
    Read monthly FAOSTAT temperature-anomaly data and aggregate to annual.

    FAOSTAT file structure:
      - Columns: Area | Year | Months | Element | Value  (+ metadata cols)
      - Element "Temperature change": monthly surface-temperature anomaly
        in °C relative to the 1951–1980 baseline period.
      - Each (Area, Year) has up to 12 rows, one per calendar month.

    Aggregation logic (monthly → annual):
      • Temperature annual mean
            = arithmetic mean of the 12 monthly anomaly values.
            Captures the overall warming magnitude for that year.

      • Temperature std across months
            = population std (ddof=0) of the 12 monthly anomaly values.
            Captures intra-annual variability / seasonal asymmetry.
            ddof=0 is used because the 12 months are the complete annual
            population, not a sample drawn from a larger set.

    Steps:
      1. Load the file and identify the data sheet.
      2. Filter to Element="Temperature change", study years, study countries.
      3. groupby(Area, Year).mean()      → Temperature annual mean
      4. groupby(Area, Year).std(ddof=0) → Temperature std across months
      5. Merge both aggregations and return.

    Returns: DataFrame with columns
             [Country Name, Year, Temperature annual mean,
              Temperature std across months]
    """
    fpath = find_file(FAO_FILE_PATTERN)
    print(f"  [FAO] Temperature  ←  {os.path.basename(fpath)}")

    # FAOSTAT exports sometimes name the sheet "Data" or similar; fall back to sheet 0
    xls = pd.ExcelFile(fpath, engine="openpyxl")
    sheet = next(
        (s for s in xls.sheet_names if "data" in s.lower()),
        xls.sheet_names[0]
    )
    df = pd.read_excel(fpath, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    # Validate expected columns exist
    required = {"Area", "Year", "Months", "Element", "Value"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"FAO file is missing expected columns: {missing}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    # Type coercion: Value and Year may be read as object dtype
    df["Value"]   = pd.to_numeric(df["Value"],   errors="coerce")
    df["Year"]    = pd.to_numeric(df["Year"],     errors="coerce")
    df["Area"]    = df["Area"].astype(str).str.strip().apply(normalize_country)
    df["Element"] = df["Element"].astype(str).str.strip()

    # Keep only the temperature-change element for the study scope
    df = df[
        df["Year"].between(YEAR_START, YEAR_END) &
        df["Area"].isin(COUNTRIES) &
        (df["Element"] == "Temperature change")
    ].copy()

    if df.empty:
        raise ValueError(
            "FAO file: no rows remain after filtering.\n"
            "Possible causes:\n"
            "  • Element column does not contain 'Temperature change' "
            "(check exact spelling / casing in the file).\n"
            "  • Country names do not match after normalization "
            "(check COUNTRY_NAME_MAP)."
        )

    # ── Annual aggregation ──────────────────────────────────
    # Mean: average warming level across all 12 months
    annual_mean = (
        df.groupby(["Area", "Year"])["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Area": "Country Name", "Value": "Temperature annual mean"})
    )

    # Std (population): seasonal spread of monthly anomalies within the year
    annual_std = (
        df.groupby(["Area", "Year"])["Value"]
        .std(ddof=0)
        .reset_index()
        .rename(columns={"Area": "Country Name", "Value": "Temperature std across months"})
    )

    out = annual_mean.merge(annual_std, on=["Country Name", "Year"], how="left")
    return out[["Country Name", "Year", "Temperature annual mean", "Temperature std across months"]]


# ── 3c. ERA5 Reanalysis – frost days and hot days ───────────
def _year_from_col(col) -> int | None:
    """
    Parse a 4-digit year from an ERA5 column header.

    ERA5 exports use column names like "1990-07" (year-month) rather
    than plain integers.  This function extracts just the year so all
    columns belonging to the same year can be identified.

    Returns the integer year, or None if the column name does not start
    with four digits (e.g. "code", "name").
    """
    m = re.match(r"^(\d{4})", str(col).strip())
    return int(m.group(1)) if m else None


def load_era5_climate() -> pd.DataFrame:
    """
    Read the ERA5 Excel file and return annual frost-day and hot-day counts.

    ERA5 file structure:
      - Two sheets: "fd" (frost days) and "hd35" (hot days).
      - Each sheet is wide: rows = countries, columns = year-month labels
        (e.g. "1990-07", "1991-07", …).
      - Country identity is stored in a column called "name".

    Processing steps:
      1. For each sheet, identify year columns within [YEAR_START, YEAR_END].
      2. Melt wide → long on (name, year).
      3. Normalise country names; filter to study countries.
      4. Outer-join the two sheets on (Country Name, Year).

    Note: ERA5 column headers are "YYYY-MM" but only one value per year
    is present in these annual aggregation files, so collapsing to year
    is unambiguous.

    Returns: DataFrame with columns
             [Country Name, Year, Number of frost days, Number of hot days]
    """
    fpath = find_file(ERA5_FILE_PATTERN)
    print(f"  [ERA5] fd + hd35  ←  {os.path.basename(fpath)}")

    frames = {}
    for sheet, col_name in ERA5_SHEETS.items():
        df = pd.read_excel(fpath, sheet_name=sheet, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]

        if "name" not in df.columns:
            raise KeyError(
                f"ERA5 sheet '{sheet}': expected a column named 'name' "
                f"(country names).  Columns found: {df.columns.tolist()[:15]}"
            )

        # Identify year columns within the study range
        year_cols = []
        year_map  = {}   # original column label → integer year
        for c in df.columns:
            y = _year_from_col(c)
            if y is not None and YEAR_START <= y <= YEAR_END:
                year_cols.append(c)
                year_map[c] = y

        if not year_cols:
            raise KeyError(
                f"ERA5 sheet '{sheet}': no year columns found between "
                f"{YEAR_START} and {YEAR_END}.  "
                f"Expected labels like '1990-07'."
            )

        # Melt: wide (one col per year) → long (one row per country-year)
        long = df[["name"] + year_cols].melt(
            id_vars="name", var_name="YearRaw", value_name=col_name
        )
        long["Year"] = long["YearRaw"].map(year_map).astype(int)
        long = (
            long
            .drop(columns="YearRaw")
            .rename(columns={"name": "Country Name"})
        )
        long["Country Name"] = long["Country Name"].apply(normalize_country)
        long = long[long["Country Name"].isin(COUNTRIES)].copy()

        frames[col_name] = long[["Country Name", "Year", col_name]]

    # Outer join: both sheets should cover the same countries/years, but
    # outer is safer to avoid silently dropping any rows.
    frost_df, hot_df = list(frames.values())
    out = frost_df.merge(hot_df, on=["Country Name", "Year"], how="outer")
    return out


# ============================================================
# 4. MAIN – MERGE ALL SOURCES AND SAVE
# ============================================================

def main():
    """
    Orchestrates the full data-integration pipeline:

      1. Build a complete country × year scaffold (204 rows).
      2. Load and left-join World Bank indicators onto the scaffold.
      3. Load and left-join FAO temperature features.
      4. Load and left-join ERA5 climate indices.
      5. Enforce final column order (target variable "Total CO2 emissions" last).
      6. Print a missingness summary to the console for quick inspection.
      7. Write the result to OUTPUT_FILE (raw_panel_dataset_1990_2023.csv).
    """
    print("\n" + "=" * 60)
    print("01_data_integration.py")
    print("=" * 60)

    # ── Step 1: scaffold ───────────────────────────────────────
    # Start with a complete grid so every (country, year) pair is
    # guaranteed to exist in the output, regardless of source coverage.
    panel = make_country_year_scaffold()
    print(f"\nScaffold: {len(COUNTRIES)} countries × {len(YEARS)} years = {len(panel)} rows\n")

    # ── Step 2: World Bank indicators ─────────────────────────
    # Adds 8 socioeconomic / energy columns.  Any missing cell retains NaN.
    print("[1/3] Loading World Bank indicators …")
    wb_wide = load_all_worldbank()
    panel = panel.merge(wb_wide, on=["Country Name", "Year"], how="left")

    # ── Step 3: FAO temperature features ──────────────────────
    # Adds 2 annual temperature-anomaly columns derived from monthly data.
    print("\n[2/3] Loading FAO temperature data …")
    fao = load_faostat_temperature()
    panel = panel.merge(fao, on=["Country Name", "Year"], how="left")

    # ── Step 4: ERA5 climate indices ───────────────────────────
    # Adds 2 extreme-weather count columns (frost days, hot days).
    print("\n[3/3] Loading ERA5 climate indices …")
    era5 = load_era5_climate()
    panel = panel.merge(era5, on=["Country Name", "Year"], how="left")

    # ── Step 5: enforce column order ──────────────────────────
    # Final column layout:
    #   Country Name, Year,
    #   Population, GDP, Electric power consumption,
    #   Fossil fuel energy consumption, Renewable energy consumption,
    #   Fertilizer consumption, Industry,
    #   Temperature annual mean, Temperature std across months,
    #   Number of frost days, Number of hot days,
    #   Total CO2 emissions   ← target variable is always last
    col_order = (
        ["Country Name", "Year"]
        + [k for k in WB_FILES.keys() if k != "Total CO2 emissions"]  # WB excl. target
        + ["Temperature annual mean", "Temperature std across months"]  # FAO
        + list(ERA5_SHEETS.values())                                    # ERA5
        + ["Total CO2 emissions"]                                       # target
    )
    # Guard against any column that failed to load (shouldn't happen, but safe)
    col_order = [c for c in col_order if c in panel.columns]
    panel = panel[col_order].sort_values(["Country Name", "Year"]).reset_index(drop=True)

    # ── Step 6: console summary ────────────────────────────────
    # Quick missingness report — useful for spotting gaps before running EDA.
    print("\n" + "-" * 60)
    print(f"Final panel shape : {panel.shape}")
    print(f"Countries         : {sorted(panel['Country Name'].unique())}")
    print(f"Year range        : {panel['Year'].min()} – {panel['Year'].max()}")
    print("\nMissingness per column (% of total rows):")
    miss = (panel.isna().sum() / len(panel) * 100).round(2)
    for col, pct in miss[miss > 0].items():
        print(f"  {col:<42} {pct:>6.2f}%")
    if (miss == 0).all():
        print("  (no missing values detected)")

    # ── Step 7: save ───────────────────────────────────────────
    panel.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅  Saved: {OUTPUT_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
