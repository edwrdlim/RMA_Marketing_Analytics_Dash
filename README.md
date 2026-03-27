# Marketing Analytics Dashboard

A set of interactive Jupyter notebooks and a marketing analytics dashboard designed to showcase the retail demand forecasting and promotion effectiveness analysis across 44 SKUs over 100 weeks of sales data.

---

## Prerequisites

- **Python 3.10+** (tested on 3.12)
- **Jupyter Notebook** or **JupyterLab**

## Installation

1. Clone or download this repository.
2. Install the required Python packages:

```bash
pip install pandas numpy plotly scikit-learn statsmodels ipywidgets
```

1. (JupyterLab only) Enable the widgets extension:

```bash
pip install jupyterlab-widgets
```

## Running the Dashboards

1. Open a terminal in the project folder and start Jupyter:

```bash
jupyter notebook
```

1. Open either notebook and run all cells top-to-bottom (**Cell → Run All**).
2. Use the interactive dropdown menus, sliders, and buttons inside the notebooks to explore different SKUs and settings.

> Both `data_raw.csv` and `data_processed.csv` must be in the same folder as the notebooks.

---

## Project Structure

```
├── README.md
├── data_raw.csv                    # Raw weekly sales (4,400 rows: 44 SKUs × 100 weeks)
├── data_processed.csv              # Feature-engineered dataset (lagged prices, trend, dummies)
├── demand_forecasting.ipynb        # Module 1 — Demand forecasting
└── promotion_effectiveness.ipynb   # Module 2 — Promotion effectiveness
```

## Data Description


| File                 | Rows  | Description                                                                                                          |
| -------------------- | ----- | -------------------------------------------------------------------------------------------------------------------- |
| `data_raw.csv`       | 4,400 | Weekly sales by SKU with price, promo flag, color, vendor, and product category                                      |
| `data_processed.csv` | 4,312 | Enriched version with lagged prices (`price-1`, `price-2`), `trend`, month dummies, and one-hot encoded categoricals |


**Key columns in `data_raw.csv`:** `week`, `sku`, `weekly_sales`, `price`, `feat_main_page` (promoted yes/no), `functionality` (product category), `color`, `vendor`

---

## Module 1: Demand Forecasting (`demand_forecasting.ipynb`)

Interactive demand forecasting dashboard. Select any SKU, configure the forecast horizon and test set size, then click **Run forecast**.

**Models compared:**

- Linear Regression (OLS baseline)
- Random Forest (200 trees, max depth 8)
- MLP Neural Network (64→32 hidden layers, scaled inputs)

**How it works:**

1. Splits data chronologically — the last N weeks become the test set (no data leakage)
2. Trains all three models and picks the best by lowest MAE
3. Projects demand forward with widening 95% confidence intervals

**Sections:** KPI cards → forecast chart with 95% CI → forecast detail table → model comparison bars → residual diagnostics → feature importance → all-SKU heatmap

---

## Module 2: Promotion Effectiveness (`promotion_effectiveness.ipynb`)

SCAN*PRO log-log OLS model measuring how much being featured on the main page boosts sales for each SKU.

**Model:**

`ln(sales) = α + β₁·ln(price) + β₂·feat + β₃·trend + β₄·ln(price_lag1) + β₅·ln(price_lag2) + Σ γ_k·month_k + ε`

- **β₂** → Promotion coefficient; **exp(β₂) − 1** = % sales lift when featured
- **Incremental sales** per promoted week = baseline sales × lift

**Sections:** Portfolio KPIs → lift ranking with CI → incremental sales chart → lift vs price-elasticity scatter → all-SKU results table → interactive SKU deep-dive (time series, box plot, coefficients, full regression output)

**Key results:** 15/44 SKUs significant at p < 0.05, median lift 52.5%, ~3,629 total incremental units.

---

## Reference

Cohen, M. C., Gras, P.E., Pentecoste, A., & Zhang, R. (2022). *Demand Prediction in Retail — A Practical Guide to Leverage Data and Predictive Analytics.* Springer Series in Supply Chain Management 14, 1–155.