# Marketing Analytics Dashboard

A set of interactive Jupyter notebooks and HTML dashboards for retail demand forecasting, promotion effectiveness analysis, and price elasticity modelling across 44 SKUs over 100 weeks of sales data.

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

3. (JupyterLab only) Enable the widgets extension:

```bash
pip install jupyterlab-widgets
```

## Running the Notebooks

1. Open a terminal in the project folder and start Jupyter:

```bash
jupyter notebook
```

2. Open any notebook from the `notebooks/` folder and run all cells top-to-bottom (**Cell → Run All**).
3. Use the interactive dropdown menus, sliders, and buttons inside the notebooks to explore different SKUs and settings.

## Regenerating the HTML Dashboards

The `scripts/` folder contains Python generators that produce self-contained HTML dashboards. Run them from the `scripts/` directory:

```bash
cd scripts
python generate_marketing_analytics_dashboard.py   # → dashboards/marketing_analytics_dashboard.html
python generate_demand_dashboard.py                 # → dashboards/demand_forecasting_dashboard.html
python generate_price_elasticity_dashboard.py       # → dashboards/price_elasticity_dashboard.html
```

Then open the HTML files in `dashboards/` directly in a browser — no Python backend required.

---

## Project Structure

```
├── README.md
├── Imperial Fonts/                                # Custom Imperial Sans Display typeface
│   ├── ImperialSansDisplay-Regular.woff2
│   ├── ImperialSansDisplay-Medium.woff2
│   └── ... (.ttf versions of each)
├── data/
│   ├── data_raw.csv                               # Raw weekly sales (4,400 rows: 44 SKUs × 100 weeks)
│   └── data_processed.csv                         # Feature-engineered dataset (lagged prices, trend, dummies)
├── notebooks/
│   ├── demand_forecasting.ipynb                   # Module 1 — Demand forecasting (interactive)
│   ├── promotion_effectiveness.ipynb              # Module 2 — Promotion effectiveness (interactive)
│   └── price_elasticity.ipynb                     # Module 3 — Price elasticity (interactive)
├── scripts/
│   ├── generate_marketing_analytics_dashboard.py  # Generates the combined 4-tab HTML dashboard
│   ├── generate_demand_dashboard.py               # Generates standalone demand forecasting HTML
│   └── generate_price_elasticity_dashboard.py     # Generates standalone price elasticity HTML
└── dashboards/
    ├── marketing_analytics_dashboard.html         # Combined dashboard (Interpretation Guide + 3 tabs)
    ├── demand_forecasting_dashboard.html           # Standalone demand forecasting dashboard
    └── price_elasticity_dashboard.html             # Standalone price elasticity dashboard
```

---

## Data Description

| File | Rows | Description |
| --- | --- | --- |
| `data/data_raw.csv` | 4,400 | Weekly sales by SKU with price, promo flag, color, vendor, and product category |
| `data/data_processed.csv` | 4,312 | Enriched version with lagged prices (`price-1`, `price-2`), `trend`, month dummies, and one-hot encoded categoricals |

**Key columns in `data_raw.csv`:** `week`, `sku`, `weekly_sales`, `price`, `feat_main_page` (promoted yes/no), `functionality` (product category), `color`, `vendor`

---

## Module 1: Demand Forecasting

**Notebook:** `notebooks/demand_forecasting.ipynb`
**Dashboard:** `dashboards/marketing_analytics_dashboard.html` → Demand Forecasting tab

Interactive demand forecasting dashboard. Select any SKU, configure the forecast horizon and test set size, then click **Run Forecast**.

**Models compared:**

- Linear Regression (OLS baseline)
- Random Forest (200 trees, max depth 8)
- MLP Neural Network (64→32 hidden layers, scaled inputs)

**How it works:**

1. Splits data chronologically — the last N weeks become the test set (no data leakage)
2. Trains all three models and picks the best by lowest MAE
3. Projects demand forward with widening 95% confidence intervals

**Sections:** Interpretation box → KPI cards → forecast chart with 95% CI → forecast detail table → all-SKU heatmap → model comparison → residual diagnostics → feature importance

---

## Module 2: Promotion Effectiveness

**Notebook:** `notebooks/promotion_effectiveness.ipynb`
**Dashboard:** `dashboards/marketing_analytics_dashboard.html` → Promotion Effectiveness tab

Scan*Pro log-log OLS model measuring how much being featured on the main page boosts sales for each SKU.

**Model:**

`ln(sales) = α + β₁·ln(price) + β₂·feat + β₃·trend + β₄·ln(price_lag1) + β₅·ln(price_lag2) + Σ γ_k·month_k + ε`

- **β₂** → Promotion coefficient; **exp(β₂) − 1** = % sales lift when featured
- **Incremental sales** per promoted week = baseline sales × lift

**Sections:** Portfolio KPIs → per-SKU detail with interpretation → incremental sales lift ranking → promotion effectiveness summary table

---

## Module 3: Price Elasticity

**Notebook:** `notebooks/price_elasticity.ipynb`
**Dashboard:** `dashboards/marketing_analytics_dashboard.html` → Price Elasticity tab

Scan*Pro log-log OLS model estimating price sensitivity per SKU.

**Sections:** Per-SKU elasticity detail → demand & revenue curves → model fit → waterfall chart → scenario table → portfolio overview → elasticity ranking → revenue impact heatmap → strategy summary

---

## Reference

Cohen, M. C., Gras, P.E., Pentecoste, A., & Zhang, R. (2022). *Demand Prediction in Retail — A Practical Guide to Leverage Data and Predictive Analytics.* Springer Series in Supply Chain Management 14, 1–155.
