# Marketing Analytics Dashboard

## Setup

```bash
pip install plotly scikit-learn pandas numpy statsmodels ipywidgets
jupyter notebook
```

We placed both `data_raw.csv` and `data_processed.csv` in the same directory as the notebooks.

---

## Module 1: demand_forecasting.ipynb

Interactive demand forecasting with ipywidgets. Select any SKU, configure forecast horizon and test size, then run.

Models: Linear Regression, Random Forest (200 trees), MLP Neural Network (64→32)

Sections: KPI cards → forecast chart with 95% CI → forecast table → model comparison → residuals + feature importance → all-SKU heatmap

---

## Module 2: promotion_effectiveness.ipynb

SCAN*PRO log-log OLS model measuring feature-promotion lift and incremental sales per SKU.

Model: ln(sales) = α + β₁·ln(price) + β₂·feat + β₃·trend + lags + months

Sections: Portfolio KPIs → lift ranking with CI → incremental sales → lift vs elasticity scatter → all-SKU table → interactive SKU deep-dive

Key results: 15/44 SKUs significant at p<0.05, median lift 52.5%, 3,627 total incremental units.
