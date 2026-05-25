# CO₂ Emissions Forecasting — A Machine Learning Model Comparison

This repository contains the full research pipeline for a thesis studying **whether incorporating temperature-related variables improves CO₂ emission forecasting** across six major-emitting countries, using four forecasting models.

---

## Research Overview

**Main Research Question**
How do different machine learning models perform in predicting CO₂ emissions across six countries — Canada, China, India, Indonesia, Russian Federation, and the United States — over the period 1990–2023?

**Models compared:** ARIMAX · SVR · XGBoost · LSTCN

**Feature sets:**
- *Baseline* — Population, GDP, Electric power consumption, Fossil fuel energy consumption, Renewable energy consumption, Fertilizer consumption
- *Augmented* — Baseline + Temperature annual mean, Temperature std across months, Number of frost days, Number of hot days

**Data sources:** World Bank WDI · FAOSTAT · World Bank CCKP (ERA5)

**Study period:** 1990–2023 · Train: ≤ 2016 · Test: 2017–2023

**Primary metric:** Scaled MAE (MinMax scaled to −0.9 to 0.9)

**Key findings:**
- ARIMAX achieves the best generalization performance among the four models, especially in this small-sample annual time-series setting
- The effect of temperature variables is highly context-dependent and does not produce consistent predictive improvements across models, countries, or latitudinal regions
- Cross-country forecasting difficulty far exceeds cross-model differences, suggesting data characteristics are more decisive than model selection
- SHAP analysis shows temperature variables have limited marginal contribution, likely because their effects are already proxied by socioeconomic variables such as GDP and energy consumption

---

## Repository Structure

```
├── data/
│   ├── raw/                         ← Original source files (xlsx); not tracked by Git
│   ├── processed/                   ← Model-ready CSV produced by 03_clean_data.py
│   ├── eda/
│   │   ├── raw_eda/                 ← Figures and tables from 02_raw_eda.py
│   │   └── clean_eda/              ← Figures and tables from 04_clean_eda.py
│   └── scripts/
│       ├── 01_data_integration.py
│       ├── 02_raw_eda.py
│       ├── 03_clean_data.py
│       └── 04_clean_eda.py
│
├── model/
│   ├── 05_ARIMAX.py
│   ├── 06_SVR.py
│   ├── 07_XGBoost.py
│   └── 08_LSTCN.py
│
├── results/
│   ├── metrics/                     ← *_metrics_summary.csv per model
│   ├── predictions/                 ← *_predictions_all.csv per model
│   └── tuning/                      ← *_tuning_results.csv per model
│
└── analysis/
    ├── 01_model_performance/
    │   └── 09_model_performance_tables.py
    ├── 02_model_reliability/
    │   ├── 10_error_analysis.py
    │   ├── 11_hyperparameter_sensitivity.py
    │   └── 12_overfitting_analysis.py
    ├── 03_svr_regularization/
    │   ├── 13_svr_regularized.py
    │   └── 14_svr_train_val_test_regularized.py
    └── 04_shap/
        └── 15_shap_svr.py
```

---

## Execution Order

Scripts are numbered 01–15 to reflect the full pipeline from raw data to final interpretation.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 01 | `01_data_integration.py` | Raw xlsx files in `data/raw/` | `data/processed/raw_panel_dataset_1990_2023.csv` |
| 02 | `02_raw_eda.py` | raw_panel_dataset_1990_2023.csv | `data/eda/raw_eda/` |
| 03 | `03_clean_data.py` | raw_panel_dataset_1990_2023.csv | `data/processed/dataset_recursive_3yr_avg_drop_Industry.csv` |
| 04 | `04_clean_eda.py` | dataset_recursive_...csv | `data/eda/clean_eda/` |
| 05–08 | Model scripts | dataset_recursive_...csv | `results/metrics/`, `results/predictions/`, `results/tuning/` |
| 09 | `09_model_performance_tables.py` | `results/metrics/` | `analysis/01_model_performance/tables/` |
| 10–12 | Reliability scripts | `results/` | `analysis/02_model_reliability/outputs/` |
| 13 | `13_svr_regularized.py` | dataset_recursive_...csv | `results/metrics/svr_regularized_*.csv` |
| 14 | `14_svr_train_val_test_regularized.py` | svr_regularized_metrics_summary.csv | `analysis/03_svr_regularization/outputs/` |
| 15 | `15_shap_svr.py` | dataset_recursive_...csv + results/metrics/ | `analysis/04_shap/svr_shap_outputs/` |

> Scripts 10, 11, and 12 are independent of each other and can be run in any order.

---

## Setup

**Python version:** 3.9 or above recommended

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn xgboost shap lstcn openpyxl
```

---

## Data Availability

Raw source files (`data/raw/`) are not included in this repository due to file size. They can be downloaded from:

- **World Bank WDI:** https://databank.worldbank.org/source/world-development-indicators
- **FAOSTAT Surface Temperature:** https://www.fao.org/faostat/en/#data/ET
- **World Bank CCKP (ERA5):** https://climateknowledgeportal.worldbank.org

Place all downloaded xlsx files in `data/raw/` before running `01_data_integration.py`.

---

## Citation

If you use this code or findings, please cite the original thesis (citation details to be added upon publication).
