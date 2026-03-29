"""
Generate a self-contained interactive Marketing Analytics Dashboard.
Merges demand forecasting (3 ML models) and price elasticity (Scan*Pro)
into one tabbed HTML file — no Python backend required.

Usage:  python generate_marketing_analytics_dashboard.py
Output: marketing_analytics_dashboard.html
"""

import json, time, warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
MAX_HORIZON = 12                                    # Max forecast weeks ahead
TEST_SIZES = list(range(4, 17, 2))                  # Temporal test-set sizes to evaluate: [4,6,8,10,12,14,16]
MODEL_NAMES = ["Naïve (Moving Avg)", "Linear Regression", "Random Forest", "Neural Network (MLP)"]
SCENARIOS = [-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30]  # Price change scenarios for elasticity
CURVE_POINTS = 300                                  # Resolution of demand/revenue curves
MONTH_COLS = [f"month_{i}" for i in range(2, 13)]   # Seasonal dummy column names

t0 = time.time()

# ─── Load data ────────────────────────────────────────────────────────────────
raw = pd.read_csv("../data/data_raw.csv")
proc = pd.read_csv("../data/data_processed.csv")
raw["week"] = pd.to_datetime(raw["week"])
proc["week"] = pd.to_datetime(proc["week"])
raw["feat_main_page"] = raw["feat_main_page"].astype(str).str.lower().eq("true").astype(int)

# Build per-SKU metadata (category, colour, vendor, averages) from raw data
sku_meta = raw.groupby("sku").agg(
    functionality=("functionality", "first"),
    color=("color", "first"),
    vendor=("vendor", "first"),
    avg_price=("price", "mean"),
    avg_sales=("weekly_sales", "mean"),
    promo_rate=("feat_main_page", "mean"),
).reset_index()
sku_meta["avg_price"] = sku_meta["avg_price"].round(2)
sku_meta["avg_sales"] = sku_meta["avg_sales"].round(1)
sku_meta["feat_rate"] = (sku_meta["promo_rate"] * 100).round(1)
sku_meta["category"] = sku_meta["functionality"].str.replace(r"^\d+\.", "", regex=True).str.strip()
sku_meta["display_name"] = sku_meta.apply(lambda r: f"{r['category']} ({r['color'].title()})", axis=1)
sku_meta["label"] = sku_meta.apply(lambda r: f"SKU {r['sku']} — {r['display_name']}", axis=1)

def meta(sku_id):
    """Return the metadata row for a given SKU, or None if not found."""
    row = sku_meta[sku_meta["sku"] == sku_id]
    return row.iloc[0] if len(row) > 0 else None


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — DEMAND FORECASTING
# ═════════════════════════════════════════════════════════════════════════════
print("━" * 60)
print("PART 1: Demand Forecasting (naïve baseline + 3 ML models)")
print("━" * 60)

# ─── Feature builder ─────────────────────────────────────────────────────────
# Instead of 20+ features (month dummies, lagged prices, one-hot categoricals)
# on ~90 training rows, we use 5 economically motivated features:
#
#   price          - the key demand driver
#   feat_main_page - whether the SKU is being promoted
#   trend          - long-run growth/decline
#   quarter        - seasonal effect (1 variable vs 11 month dummies)
#
# Why not lagged prices (price-1, price-2)?
#   They are ~95% correlated with current price (VIF > 50). They exist for the
#   SCAN*PRO model (reference-price formation, stockpiling) but for demand
#   forecasting, current price already carries the price signal.
#
# Why not 11 month dummies?
#   With 90 rows, 11 binary columns mostly read as noise. A single quarter
#   variable (1–4) captures the main seasonal rhythm with 1 feature instead of 11.

DEMAND_FEATURES = ["price", "feat_main_page", "trend", "quarter", "price_change"]

def build_demand_features(proc, sku_id):
    """Build a minimal, economically motivated feature set for one SKU.

    5 features on ~90 rows gives an 18:1 sample-to-feature ratio.
    price_change = price - price_lag1 captures the reference-price effect
    (did price just go up or down?) without the multicollinearity of raw lags.
    """
    df = proc[proc["sku"] == sku_id].sort_values("week").copy()

    X = pd.DataFrame(index=df.index)
    X["price"] = df["price"].values
    X["feat_main_page"] = df["feat_main_page"].values
    X["trend"] = df["trend"].values
    X["quarter"] = pd.to_datetime(df["week"]).dt.quarter.values

    # Price change: current price minus last week's price
    # Captures the dynamic pricing signal (reference-price effect) without
    # the multicollinearity of including raw lagged prices (VIF > 50).
    if "price-1" in df.columns:
        lag1 = df["price-1"].values.copy()
        lag1 = np.where(lag1 > 0, lag1, df["price"].values)  # fill missing with current
        X["price_change"] = df["price"].values - lag1
    else:
        X["price_change"] = 0.0

    return X.values, df["weekly_sales"].values, df["week"].values, list(X.columns)


# ─── Hyperparameter tuning (on per-SKU, 5-feature data) ─────────────────────
# Tuning runs on the SAME feature space the models will actually use.
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

print("  Tuning hyperparameters on per-SKU data (5 features)...")

sku_sales = proc.groupby("sku")["weekly_sales"].mean().sort_values()
sample_skus = [sku_sales.index[i] for i in np.linspace(0, len(sku_sales) - 1, 6, dtype=int)]

X_pool_tune, y_pool_tune = [], []
for s in sample_skus:
    X_s, y_s, _, _ = build_demand_features(proc, s)
    X_pool_tune.append(X_s[:-8])
    y_pool_tune.append(y_s[:-8])
X_pool_tune = np.vstack(X_pool_tune)
y_pool_tune = np.concatenate(y_pool_tune)

tscv = TimeSeriesSplit(n_splits=3)

rf_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {"n_estimators": [100, 200], "max_depth": [4, 6, 8], "min_samples_leaf": [2, 4, 8]},
    cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1,
)
rf_search.fit(X_pool_tune, y_pool_tune)
TUNED_RF = rf_search.best_params_
print(f"    RF best:  {TUNED_RF}  (CV MAE: {-rf_search.best_score_:.2f})")

scaler_tune = StandardScaler()
X_pool_s = scaler_tune.fit_transform(X_pool_tune)

mlp_search = GridSearchCV(
    MLPRegressor(max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=42),
    {"hidden_layer_sizes": [(32,), (64, 32)], "learning_rate_init": [0.001, 0.005, 0.01]},
    cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1,
)
mlp_search.fit(X_pool_s, y_pool_tune)
TUNED_MLP = mlp_search.best_params_
print(f"    MLP best: {TUNED_MLP}  (CV MAE: {-mlp_search.best_score_:.2f})")
print(f"    Tuned on {X_pool_tune.shape[0]} rows × {X_pool_tune.shape[1]} features")


# ─── Training function ───────────────────────────────────────────────────────

def train_one_config(X_sku, y_sku, weeks, feature_cols, test_size):
    """Train naïve baseline + 3 ML models for one SKU using per-SKU data.

    With 5 features and ~90 rows (18:1 ratio), per-SKU training works well.
    Category pooling was removed because SKUs within the same category have
    different demand levels and patterns — the pooled model learned the
    category average, not this SKU's behaviour.
    """
    if len(X_sku) < test_size + 10:
        return None

    # Temporal train/test split
    X_train = X_sku[:-test_size]
    X_test = X_sku[-test_size:]
    y_train = y_sku[:-test_size]
    y_test = y_sku[-test_size:]

    # ── Naïve baseline: average of last 8 training weeks ──
    lookback = min(8, len(y_train))
    naive_level = float(np.mean(y_train[-lookback:]))
    naive_std = float(np.std(y_train[-lookback:])) if lookback > 1 else float(np.std(y_train))

    naive_test = np.full(test_size, naive_level)
    naive_forecast = [round(naive_level, 2)] * MAX_HORIZON
    naive_ci_lo = [round(max(naive_level - naive_std * 1.96 * np.sqrt(s), 0), 2) for s in range(1, MAX_HORIZON + 1)]
    naive_ci_hi = [round(naive_level + naive_std * 1.96 * np.sqrt(s), 2) for s in range(1, MAX_HORIZON + 1)]

    results = {}
    results["Naïve (Moving Avg)"] = {
        "y_pred_test": [round(float(v), 2) for v in naive_test],
        "forecast": naive_forecast,
        "ci_lower": naive_ci_lo,
        "ci_upper": naive_ci_hi,
        "mae": round(float(mean_absolute_error(y_test, naive_test)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, naive_test))), 3),
        "r2": round(float(r2_score(y_test, naive_test)), 4),
        "mape": round(float(np.mean(np.abs((y_test - naive_test) / np.maximum(y_test, 1))) * 100), 2),
    }

    # ── ML models: trained on this SKU's training data ──
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ml_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            **TUNED_RF, random_state=42, n_jobs=-1),
        "Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=TUNED_MLP["hidden_layer_sizes"],
            learning_rate_init=TUNED_MLP["learning_rate_init"],
            max_iter=500, early_stopping=True,
            validation_fraction=0.15, random_state=42),
    }

    for name, model in ml_models.items():
        use_scaled = "Neural" in name
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s if use_scaled else X_test
        model.fit(Xtr, y_train)
        y_pred_train = np.maximum(model.predict(Xtr), 0)
        y_pred_test = np.maximum(model.predict(Xte), 0)
        resid_std = float(np.std(y_train - y_pred_train))

        # Forecast from last known feature row
        last_row = X_sku[-1].copy().reshape(1, -1)
        preds, ci_lo, ci_hi = [], [], []
        for step in range(1, MAX_HORIZON + 1):
            row = last_row.copy()
            if "trend" in feature_cols:
                row[0, feature_cols.index("trend")] = X_sku[-1, feature_cols.index("trend")] + step / len(X_sku)
            Xf = scaler.transform(row) if use_scaled else row
            pred = max(float(model.predict(Xf)[0]), 0)
            ci_w = resid_std * 1.96 * np.sqrt(step)
            preds.append(round(pred, 2))
            ci_lo.append(round(max(pred - ci_w, 0), 2))
            ci_hi.append(round(pred + ci_w, 2))

        results[name] = {
            "y_pred_test": [round(float(v), 2) for v in y_pred_test],
            "forecast": preds,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "mae": round(float(mean_absolute_error(y_test, y_pred_test)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 3),
            "r2": round(float(r2_score(y_test, y_pred_test)), 4),
            "mape": round(float(np.mean(np.abs((y_test - y_pred_test) / np.maximum(y_test, 1))) * 100), 2),
        }

    best_model = min(results, key=lambda k: results[k]["mae"])
    rf_model = ml_models["Random Forest"]
    imp = rf_model.feature_importances_
    feat_imp = {feature_cols[i]: round(float(imp[i]), 5) for i in range(len(feature_cols))}

    return {"results": results, "best_model": best_model, "feature_importance": feat_imp}


# ─── Train all SKUs ──────────────────────────────────────────────────────────
demand_sku_data = {}
heatmap_preds = {}
total_configs = 0

for sku_id in sorted(proc["sku"].unique()):
    sm_row = sku_meta[sku_meta.sku == sku_id].iloc[0]

    X_full, y_full, weeks, feature_cols = build_demand_features(proc, sku_id)
    week_strs = [pd.Timestamp(w).strftime("%Y-%m-%d") for w in weeks]
    sales_list = [round(float(v), 2) for v in y_full]

    by_test = {}
    for ts in TEST_SIZES:
        cfg = train_one_config(X_full, y_full, weeks, feature_cols, ts)
        if cfg:
            by_test[str(ts)] = cfg
            total_configs += 1

    if not by_test:
        continue

    demand_sku_data[int(sku_id)] = {
        "meta": {
            "functionality": sm_row.functionality,
            "color": sm_row.color,
            "avg_price": float(sm_row.avg_price),
            "avg_sales": float(sm_row.avg_sales),
            "feat_rate": float(sm_row.feat_rate),
        },
        "weeks": week_strs,
        "sales": sales_list,
        "by_test": by_test,
    }

    # Heatmap: RF forecast 4 weeks ahead
    hm_model = RandomForestRegressor(**TUNED_RF, random_state=42, n_jobs=-1)
    hm_model.fit(X_full, y_full)
    hm_row = X_full[-1].copy().reshape(1, -1)
    hm_preds = []
    for step in range(1, 5):
        r = hm_row.copy()
        r[0, feature_cols.index("trend")] = X_full[-1, feature_cols.index("trend")] + step / len(X_full)
        hm_preds.append(round(max(float(hm_model.predict(r)[0]), 0), 1))
    heatmap_preds[int(sku_id)] = hm_preds
    print(f"  SKU {sku_id:2d} — configs: {len(by_test)}")

heatmap_week_labels = [
    (pd.Timestamp(proc["week"].max()) + pd.Timedelta(weeks=i)).strftime("%d %b")
    for i in range(1, 5)
]

demand_data = {
    "skus": demand_sku_data,
    "heatmap": heatmap_preds,
    "heatmap_weeks": heatmap_week_labels,
    "test_sizes": TEST_SIZES,
    "max_horizon": MAX_HORIZON,
}

print(f"\n  {len(demand_sku_data)} SKUs × {len(TEST_SIZES)} test sizes = {total_configs} configs")


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — PRICE ELASTICITY (Scan*Pro)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "━" * 60)
print("PART 2: Price Elasticity (Scan*Pro OLS)")
print("━" * 60)


def fit_scanpro(sku_id):
    """Fit Scan*Pro log-log OLS for one SKU.

    Model: log(sales) = β₀ + β₁·log(price) + β₂·feat + β₃·trend + Σγ·month + ε
    Returns elasticity (β₁), promotion coefficient (β₂), demand/revenue curves,
    optimal price, and what-if scenario impacts. Returns None if insufficient data.
    """
    df = proc[proc["sku"] == sku_id].sort_values("week").copy()
    df = df[df["weekly_sales"] > 0]
    if len(df) < 15 or df["feat_main_page"].nunique() < 2:
        return None

    # Build the regression matrix: log-price, promo flag, trend, month dummies
    y = np.log(df["weekly_sales"].values)
    X = pd.DataFrame(index=df.index)
    X["log_price"] = np.log(df["price"].clip(lower=0.01))
    X["feat_main_page"] = df["feat_main_page"]
    X["trend"] = df["trend"]
    for m in MONTH_COLS:
        if m in df.columns:
            X[m] = df[m]
    X = sm.add_constant(X, has_constant="add")

    try:
        model = sm.OLS(y, X).fit(cov_type='HC1')  # Robust SE for heteroscedasticity
    except Exception:
        return None

    elast = float(model.params.get("log_price", np.nan))
    elast_pval = float(model.pvalues.get("log_price", np.nan))
    promo_coef = float(model.params.get("feat_main_page", np.nan))
    promo_pval = float(model.pvalues.get("feat_main_page", np.nan))
    if np.isnan(elast):
        return None

    m = meta(sku_id)
    P0 = float(df["price"].mean())    # Baseline price
    Q0 = float(df["weekly_sales"].mean())  # Baseline demand
    R0 = P0 * Q0                      # Baseline revenue

    # Constant-elasticity demand/revenue curves: Q = Q₀·(P/P₀)^ε
    price_range = np.linspace(P0 * 0.20, P0 * 3.0, CURVE_POINTS)
    demand_curve = np.clip(Q0 * (price_range / P0) ** elast, 0, None)
    revenue_curve = price_range * demand_curve
    opt_idx = int(np.argmax(revenue_curve))  # Revenue-maximising price point

    X_full = X.copy()
    fitted = np.exp(model.predict(X_full)).tolist()

    # What-if scenarios: simulate demand/revenue at each price change %
    scenarios = []
    for pct in SCENARIOS:
        P1 = P0 * (1 + pct)
        Q1 = max(Q0 * ((P1 / P0) ** elast), 0)
        R1 = P1 * Q1
        scenarios.append({
            "label": f"{'↑' if pct > 0 else '↓'} {abs(int(pct * 100))}%",
            "pct": round(pct, 2),
            "new_price": round(P1, 2),
            "d_demand": round(Q1 - Q0, 1),
            "d_demand_pct": round((Q1 - Q0) / Q0 * 100, 1) if Q0 > 0 else 0,
            "new_rev": round(R1, 2),
            "d_rev": round(R1 - R0, 2),
            "d_rev_pct": round((R1 - R0) / R0 * 100, 1) if R0 > 0 else 0,
        })

    return {
        "sku": int(sku_id),
        "label": str(m["label"]) if m is not None else f"SKU {sku_id}",
        "category": str(m["category"]) if m is not None else "Unknown",
        "color": str(m["color"]) if m is not None else "Unknown",
        "avg_price": round(P0, 2),
        "avg_sales": round(Q0, 1),
        "avg_revenue": round(R0, 2),
        "promo_rate": round(float(m["promo_rate"]) * 100, 1) if m is not None else 0,
        "elasticity": round(elast, 4),
        "elast_pval": round(elast_pval, 4),
        "elast_sig": bool(elast_pval < 0.05),
        "promo_coef": round(promo_coef, 4),
        "promo_pval": round(promo_pval, 4),
        "r2": round(float(model.rsquared), 4),
        "optimal_price": round(float(price_range[opt_idx]), 2),
        "optimal_revenue": round(float(revenue_curve[opt_idx]), 2),
        "price_range": [round(p, 2) for p in price_range.tolist()],
        "demand_curve": [round(q, 1) for q in demand_curve.tolist()],
        "revenue_curve": [round(r, 2) for r in revenue_curve.tolist()],
        "weeks": [pd.Timestamp(w).strftime("%Y-%m-%d") for w in df["week"].values],
        "actual": [round(float(v), 1) for v in df["weekly_sales"].values],
        "fitted": [round(float(v), 1) for v in fitted],
        "scenarios": scenarios,
    }


all_elast_data = {}
for sku_id in sorted(proc["sku"].unique()):
    r = fit_scanpro(sku_id)
    if r:
        all_elast_data[int(sku_id)] = r
        print(f"  SKU {sku_id:2d} — ε={r['elasticity']:+.3f}  "
              f"p={r['elast_pval']:.3f}  "
              f"{'★' if r['elast_sig'] else ' '}  R²={r['r2']:.3f}")

# Portfolio-level elasticity summary statistics
elasticities = [r["elasticity"] for r in all_elast_data.values()]
n_elastic = sum(1 for e in elasticities if e < -1)
n_inelastic = sum(1 for e in elasticities if -1 <= e < 0)
n_sig = sum(1 for r in all_elast_data.values() if r["elast_sig"])
avg_elast = round(float(np.mean(elasticities)), 3)
most_sens = min(all_elast_data.values(), key=lambda r: r["elasticity"])
least_sens = max(all_elast_data.values(), key=lambda r: r["elasticity"])

# Build revenue-impact heatmap data (all SKUs × all price scenarios)
heatmap_skus = sorted(all_elast_data.keys())
heatmap_labels = [f"SKU {s}" for s in heatmap_skus]
scenario_labels = [f"{'↑' if p > 0 else '↓'}{abs(int(p * 100))}%" for p in SCENARIOS]
heatmap_z, heatmap_text = [], []
for sku_id in heatmap_skus:
    r = all_elast_data[sku_id]
    row_z, row_t = [], []
    for sc in r["scenarios"]:
        row_z.append(sc["d_rev_pct"])
        row_t.append(f"{sc['d_rev_pct']:+.1f}%")
    heatmap_z.append(row_z)
    heatmap_text.append(row_t)

elasticity_data = {
    "skus": all_elast_data,
    "portfolio": {
        "n_skus": len(all_elast_data),
        "n_elastic": n_elastic,
        "n_inelastic": n_inelastic,
        "n_sig": n_sig,
        "avg_elast": avg_elast,
        "most_sens": {"sku": most_sens["sku"], "label": most_sens["label"],
                      "elasticity": most_sens["elasticity"]},
        "least_sens": {"sku": least_sens["sku"], "label": least_sens["label"],
                       "elasticity": least_sens["elasticity"]},
    },
    "heatmap": {
        "z": heatmap_z,
        "text": heatmap_text,
        "sku_labels": heatmap_labels,
        "scenario_labels": scenario_labels,
    },
}

print(f"\n  {len(all_elast_data)} SKUs processed")


# ═════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE — Tabbed Marketing Analytics Dashboard
# ═════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Marketing Analytics Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  @font-face { font-family:'Imperial Sans Display'; font-weight:200; font-style:normal;
    src:url('../Imperial Fonts/ImperialSansDisplay-Extralight.woff2') format('woff2'),
        url('../Imperial Fonts/ImperialSansDisplay-Extralight.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:300; font-style:normal;
    src:url('../Imperial Fonts/ImperialSansDisplay-Light.woff2') format('woff2'),
        url('../Imperial Fonts/ImperialSansDisplay-Light.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:400; font-style:normal;
    src:url('../Imperial Fonts/ImperialSansDisplay-Regular.woff2') format('woff2'),
        url('../Imperial Fonts/ImperialSansDisplay-Regular.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:500; font-style:normal;
    src:url('../Imperial Fonts/ImperialSansDisplay-Medium.woff2') format('woff2'),
        url('../Imperial Fonts/ImperialSansDisplay-Medium.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:600; font-style:normal;
    src:url('../Imperial Fonts/ImperialSansDisplay-Semibold.woff2') format('woff2'),
        url('../Imperial Fonts/ImperialSansDisplay-Semibold.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:700; font-style:normal;
    src:url('../Imperial Fonts/ImperialSansDisplay-Bold.woff2') format('woff2'),
        url('../Imperial Fonts/ImperialSansDisplay-Bold.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:800; font-style:normal;
    src:url('../Imperial Fonts/ImperialSansDisplay-Extrabold.woff2') format('woff2'),
        url('../Imperial Fonts/ImperialSansDisplay-Extrabold.ttf') format('truetype'); }

  :root {
    --slate-50:#f8fafc; --slate-100:#f1f5f9; --slate-200:#e2e8f0;
    --slate-300:#cbd5e1; --slate-400:#94a3b8; --slate-500:#64748b;
    --slate-600:#475569; --slate-700:#334155; --slate-800:#1e293b;
    --slate-900:#0f172a;
    --blue:#2563EB; --blue-lt:rgba(37,99,235,0.10);
    --indigo:#6366f1; --green:#059669; --amber:#f59e0b;
    --red:#DC2626; --green-dk:#16a34a;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Imperial Sans Display',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    background:linear-gradient(180deg,#e8effa 0%,#f0f4fb 15%,#f6f8fc 40%,#faf9f7 100%);
    background-attachment:fixed; color:var(--slate-900); line-height:1.5; min-height:100vh; }
  .container { max-width:1200px; margin:0 auto; padding:1.5rem; }

  /* Header */
  header { background:#0000CD; border-bottom:1px solid #0000CD;
    padding:1.2rem 0; }
  header .container { display:flex; flex-direction:column; gap:.8rem; }
  .header-top { display:flex; align-items:center;
    justify-content:space-between; flex-wrap:wrap; gap:.75rem; }
  @keyframes glowPulse {
    0%, 100% { text-shadow:0 0 10px rgba(255,255,255,0.4), 0 0 20px rgba(255,255,255,0.2); }
    50% { text-shadow:0 0 20px rgba(255,255,255,0.7), 0 0 40px rgba(255,255,255,0.4), 0 0 60px rgba(255,255,255,0.15); }
  }
  h1 { font-size:58px; font-weight:700; color:white;
    animation:glowPulse 3s ease-in-out infinite; }
  .header-meta { font-size:.88rem; color:rgba(255,255,255,0.7); line-height:1.6; }
  .header-authors { font-size:.82rem; color:rgba(255,255,255,0.85); font-weight:500; }

  /* Tabs */
  .tab-bar { display:flex; gap:0; border-bottom:2px solid var(--slate-200);
    margin-bottom:.75rem; background:white; padding:0 1.5rem; }
  .tab-btn { padding:.85rem 1.6rem; font-size:.9rem; font-weight:600;
    background:none; border:none; border-bottom:3px solid transparent;
    color:var(--slate-500); cursor:pointer; transition:all .15s;
    font-family:inherit; margin-bottom:-2px; }
  .tab-btn:hover { color:var(--slate-700); background:var(--slate-50); }
  .tab-btn.active { color:var(--blue); border-bottom-color:var(--blue); }
  .tab-panel { display:none; padding-top:.5rem; }
  .tab-panel.active { display:block; }
  .tab-panel > .section-title:first-child { margin-top:.25rem; }

  /* Controls */
  .controls { display:flex; gap:1rem; align-items:center; flex-wrap:wrap;
    margin-bottom:1rem; }
  select { font-family:inherit; font-size:.95rem; padding:.7rem 2.4rem .7rem 1rem;
    border-radius:10px; border:1px solid rgba(0,0,205,0.2); background:white;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%230000CD' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
    background-repeat:no-repeat; background-position:right .9rem center;
    -webkit-appearance:none; -moz-appearance:none; appearance:none;
    min-width:320px; color:var(--slate-800); cursor:pointer;
    box-shadow:0 1px 3px rgba(0,0,205,0.06);
    transition:border-color .15s, box-shadow .15s; }
  select:hover { border-color:rgba(0,0,205,0.4); box-shadow:0 2px 8px rgba(0,0,205,0.1); }
  select:focus { outline:none; border-color:#0000CD;
    box-shadow:0 0 0 3px rgba(0,0,205,0.15), 0 2px 8px rgba(0,0,205,0.1); }
  button { font-family:inherit; font-size:1.05rem; padding:.65rem 1.4rem;
    border-radius:8px; border:none; background:#00008B;
    color:white; font-weight:600; cursor:pointer; transition:background .15s; }
  button:hover { background:#000066; }

  .slider-group { display:flex; align-items:center; gap:.5rem; }
  .slider-group label { font-size:1rem; color:var(--slate-600); white-space:nowrap; }
  .slider-group input[type=range] { width:130px; accent-color:#0000CD; cursor:pointer; }
  .slider-group .slider-val { font-size:1.05rem; font-weight:700; color:var(--slate-800);
    min-width:32px; text-align:center; font-family:'SF Mono',SFMono-Regular,Consolas,monospace; }

  /* KPI cards */
  .kpi-row { display:flex; gap:.75rem; margin:1rem 0 1.5rem; flex-wrap:wrap; }
  .kpi { flex:1; min-width:155px; background:linear-gradient(135deg,#f0f5ff 0%,#e8f0fe 100%);
    border:1px solid #7BB8F5; border-radius:12px; padding:1rem 1.2rem;
    transition:transform .15s, box-shadow .15s; }
  .kpi:hover { transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,205,0.1); }
  .kpi-label { font-size:.7rem; color:var(--slate-500); text-transform:uppercase;
    letter-spacing:.05em; font-weight:500; }
  .kpi-value { font-size:1.35rem; font-weight:700; margin-top:.15rem;
    font-family:'SF Mono',SFMono-Regular,Consolas,monospace; }
  .kpi-sub { font-size:.75rem; color:var(--slate-500); margin-top:.1rem; }

  /* SKU header */
  .sku-header { margin-bottom:.5rem; }
  .sku-header h2 { font-size:1.38rem; font-weight:700; color:#0000CD; }
  .sku-header p  { font-size:.85rem; color:var(--slate-500); margin-top:.15rem; }

  /* Interpretation badge */
  .interp { border-left:3px solid #0000CD; border:1px solid #7BB8F5; border-left:3px solid #0000CD;
    background:#eff6ff; padding:.55rem 1rem; border-radius:0 6px 6px 0;
    font-size:.84rem; color:var(--slate-700);
    margin:.6rem 0 1rem; line-height:1.5; }

  /* Charts */
  .chart-card { background:linear-gradient(180deg,#fafcff 0%,#ffffff 100%); border:1px solid #7BB8F5;
    border-radius:12px; margin-bottom:1rem; overflow:hidden;
    transition:box-shadow .15s; }
  .chart-card:hover { box-shadow:0 4px 14px rgba(0,0,205,0.08); }
  .section-title { font-size:1.2rem; font-weight:600;
    margin:1.5rem 0 .5rem; color:#0000CD; }

  /* Tables */
  table { width:100%; border-collapse:collapse; font-size:.82rem;
    background:linear-gradient(180deg,#fafcff 0%,#ffffff 100%); border:1px solid #7BB8F5;
    border-radius:12px; overflow:hidden; margin-bottom:1rem; }
  th { background:#0000CD; color:white; font-weight:500;
    text-transform:uppercase; font-size:.7rem; letter-spacing:.04em; }
  th.lt { background:#e8f0fe; color:var(--slate-600); font-weight:600; }
  th, td { padding:.6rem .9rem; text-align:center;
    border-bottom:1px solid var(--slate-100); }
  td { font-family:'SF Mono',SFMono-Regular,Consolas,monospace; font-size:.8rem; }
  td.label-col { text-align:left; font-family:inherit; font-size:.82rem; }
  .pos { color:var(--green-dk); font-weight:600; }
  .neg { color:var(--red); font-weight:600; }
  .sig { color:var(--blue); font-weight:700; }
  tr.row-up { background:#f0fdf4; }
  tr.row-dn { background:#fef9f9; }
  tr.best td { background:#f0fdf4; font-weight:600; }

  /* Grid */
  .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
  @media(max-width:768px) {
    .grid-2 { grid-template-columns:1fr; }
    .controls { flex-direction:column; align-items:stretch; }
    select { min-width:100%; }
    .tab-btn { padding:.7rem 1rem; font-size:.82rem; }
  }

  /* Methodology */
  .methodology { background:linear-gradient(135deg,#f5f8ff 0%,#eef3fc 100%); border:1px solid #7BB8F5;
    border-radius:12px; padding:1.5rem; margin-top:1.5rem;
    font-size:.85rem; color:var(--slate-600); line-height:1.7; }
  .methodology h3 { font-size:1.32rem; color:#0000CD; margin-bottom:.5rem; }
  .methodology ul { padding-left:1.2rem; }
  footer { text-align:center; padding:2rem 0;
    font-size:.78rem; color:var(--slate-400); }
  .elast-sub { padding-top:.5rem; }
</style>
</head>
<body>

<!-- ═════════════════════════════════════ HEADER ═══════════════════════════ -->
<header>
  <div class="container">
    <div class="header-top">
      <h1>Marketing Analytics Dashboard</h1>
    </div>
    <div class="header-meta">This dashboard is created using the weekly sales of 44 stock-keeping units (SKUs) from a tech-gadget e-commerce retailer over a period of 100 weeks, from October 2016 to September 2018.<br><b>Tips:</b> Click and drag the portion of some graphs of your choice to zoom in and out. Double-click to reset. Select different SKUs from the dropdown to show different forecast.</div>
    <div class="header-authors">Edward Lim, Fabricio Rodriguez Pe&ntilde;a, Rafia Al-Jassim, Rathes Waran, Wenyuan Yun</div>
  </div>
</header>

<!-- ═══════════════════════════════ TAB BAR ═══════════════════════════════ -->
<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('guide')">Interpretation Guide</button>
  <button class="tab-btn" onclick="switchTab('demand')">Demand Forecasting</button>
  <button class="tab-btn" onclick="switchTab('promo')">Promotion Effectiveness</button>
  <button class="tab-btn" onclick="switchTab('elasticity')">Price Elasticity</button>
</div>

<div class="container">

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- TAB 0: INTERPRETATION GUIDE                                           -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div id="tab-guide" class="tab-panel active">
  <div class="methodology" style="margin-top:0;">
    <h3>Welcome to the Marketing Analytics Dashboard</h3>
    <p>This dashboard analyses 44 SKUs from a tech-gadget e-commerce retailer across 100 weeks of sales data.
      It is organised into three analytical tabs. Here is how to navigate and interpret each one:</p>

    <h3 style="margin-top:1.2rem;">1. Demand Forecasting</h3>
    <p>Select any SKU from the dropdown to see its historical sales and predicted future demand.
      Four models compete: a na&iuml;ve moving-average baseline plus three ML models (Linear Regression, Random Forest, Neural Network)
      &mdash; the best-performing model (lowest MAE on the held-out test set) is automatically highlighted.</p>
    <ul>
      <li><b>Interpretation box</b> &mdash; Plain-English summary of expected demand, trend direction, and the 95% confidence range.</li>
      <li><b>Forecast chart</b> &mdash; The shaded band around the best model&rsquo;s forecast is the 95% CI: it widens with horizon, reflecting increasing uncertainty.</li>
      <li><b>Forecast weeks slider</b> &mdash; Adjust to look 2&ndash;12 weeks ahead.</li>
      <li><b>Test size slider</b> &mdash; Controls how many recent weeks are held out for evaluation. Smaller values give more training data; larger values give a longer comparison window. Consistent performance across test sizes builds confidence in the forecast.</li>
      <li><b>Residuals &amp; Feature Importance</b> &mdash; Diagnostics panel showing prediction errors and which features (price, promotion, trend, quarter, price change) drive each SKU&rsquo;s demand most.</li>
      <li><b>All-SKU Heatmap</b> &mdash; Portfolio-level view of Random Forest demand forecasts for all 44 SKUs across the next 4 weeks.</li>
    </ul>

    <h3 style="margin-top:1.2rem;">2. Promotion Effectiveness</h3>
    <p>This tab answers: <i>&ldquo;Do feature promotions actually boost sales, and by how much?&rdquo;</i></p>
    <p>We use the <b>SCAN*PRO model</b> (Van Heerde et al., 2002) &mdash; the standard marketing science framework
      for measuring promotion effectiveness from observational data. A simple comparison of promoted vs non-promoted
      weeks would be confounded by simultaneous price discounts, seasonal patterns, and demand trends.
      SCAN*PRO controls for all three, isolating the <i>pure</i> promotion effect.</p>
    <ul>
      <li><b>Portfolio KPIs</b> &mdash; Show how many SKUs have a statistically significant promotion effect.</li>
      <li><b>Per-SKU detail</b> &mdash; Select a SKU to see its promotion coefficient (&beta;&#8322;), the estimated sales lift, and incremental units generated per promo week.</li>
      <li><b>Lift chart</b> &mdash; Ranks all SKUs by incremental sales. Blue bars = significant positive lift; red = significant negative; grey = not significant.</li>
      <li><b>&#9733; symbol</b> &mdash; Indicates statistical significance (p &lt; 0.05), meaning we can be confident the effect is real.</li>
    </ul>

    <h3 style="margin-top:1.2rem;">3. Price Elasticity</h3>
    <p>This tab answers: <i>&ldquo;How sensitive is each SKU&rsquo;s demand to price changes, and what happens to revenue if we change price?&rdquo;</i></p>
    <p>Elasticity is estimated using a <b>Scan*Pro log-log OLS model</b> per SKU, controlling for promotions, trend, and seasonality. The coefficient &epsilon; directly gives price elasticity.</p>
    <ul>
      <li><b>Elasticity (&epsilon;)</b> &mdash; A value of &minus;2 means a 10% price rise causes a ~20% demand drop. &epsilon; below &minus;1 = elastic (price-sensitive); between &minus;1 and 0 = inelastic. A &#9733; means the result is statistically significant (p &lt; 0.05).</li>
      <li><b>Overview sub-tab</b> &mdash; Portfolio-level bar chart, distribution histogram, and pie chart showing how many SKUs fall into each elasticity category. Full results table with p-values and R&sup2;.</li>
      <li><b>SKU Deep Dive</b> &mdash; Per-SKU demand and revenue curves, a waterfall chart of revenue impact at standard scenarios, and a full what-if scenario table. The &#9733; on the revenue curve marks the revenue-maximising price.</li>
      <li><b>Scenario Testing</b> &mdash; Standard &plusmn;5%, 10%, 20%, 30% impact table with demand and revenue bar charts. Use the <b>custom slider</b> (&minus;30% to +30%) to simulate any price change and see demand and revenue impact instantly.</li>
      <li><b>Portfolio Analysis</b> &mdash; Category-level box plots, a ranking chart of all SKUs by elasticity with 95% confidence-based colour coding, a revenue impact heatmap across all SKUs and scenarios, and a pricing strategy recommendation table.</li>
      <li><b>Pricing rule of thumb:</b> Elastic SKUs (&epsilon; &lt; &minus;1) grow revenue by cutting price; inelastic SKUs (&epsilon; between &minus;1 and 0) grow revenue by raising price.</li>
    </ul>

    <h3 style="margin-top:1.2rem;">General Tips</h3>
    <ul>
      <li>Click and drag on any Plotly chart to zoom in. Double-click to reset.</li>
      <li>Hover over data points for detailed tooltips.</li>
      <li>Use the SKU dropdowns on each tab to switch between products.</li>
      <li>Green numbers indicate positive outcomes; red indicates negative.</li>
    </ul>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- TAB 1: DEMAND FORECASTING                                             -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div id="tab-demand" class="tab-panel">

  <div class="controls">
    <select id="demandSkuSelect"></select>
    <div class="slider-group">
      <label>Forecast weeks:</label>
      <input type="range" id="horizonSlider" min="2" max="12" value="4" step="1">
      <span class="slider-val" id="horizonVal">4</span>
    </div>
    <div class="slider-group">
      <label>Test size:</label>
      <input type="range" id="testSlider" min="__TEST_MIN__" max="__TEST_MAX__" value="8" step="__TEST_STEP__">
      <span class="slider-val" id="testVal">8</span>
    </div>
    <button onclick="renderDemand()">Run Forecast</button>
  </div>

  <div id="demandHeader" class="sku-header"></div>
  <div id="demandInterpBox" class="interp" style="display:none"></div>
  <div id="demandKpiRow" class="kpi-row"></div>
  <div class="chart-card"><div id="forecastChart"></div></div>

  <h3 class="section-title">Forecast Detail</h3>
  <div id="forecastTable"></div>

  <h3 class="section-title">Diagnostics</h3>
  <div class="grid-2">
    <div class="chart-card"><div id="residChart"></div></div>
    <div class="chart-card"><div id="featChart"></div></div>
  </div>

  <div style="margin-top:2.5rem;">
    <h3 class="section-title">All-SKU Forecast Heatmap (Random Forest)</h3>
    <div class="chart-card"><div id="demandHeatmapChart"></div></div>
  </div>

  <div class="methodology">
    <h3>Our Methodology &mdash; Demand Forecasting</h3>
    <p><b>Data:</b> 44 SKUs &times; 98 weeks from the processed dataset.</p>

    <p><b>Feature engineering &mdash; economically motivated, minimal set:</b></p>
    <ul>
      <li><b>price</b> &mdash; the primary demand driver.</li>
      <li><b>feat_main_page</b> &mdash; binary promotion indicator.</li>
      <li><b>trend</b> &mdash; linear time index capturing long-run growth/decline.</li>
      <li><b>quarter</b> &mdash; captures seasonal rhythm with 1 variable instead of 11 month dummies.
        With ~90 rows per SKU, 11 binary dummies mostly read as noise; a single quarter variable is
        more sample-efficient.</li>
      <li><b>price_change</b> &mdash; current price minus last week&rsquo;s price. Captures the
        reference-price effect (&ldquo;did price just go up or down?&rdquo;) without the
        multicollinearity of including raw lagged prices. Raw price levels (price, price-1, price-2)
        are ~95% correlated (VIF &gt; 50), but the current level and the <i>change</i> from last
        week are nearly uncorrelated.</li>
    </ul>
    <p><i>Why no lagged prices?</i> price-1 and price-2 are ~95% correlated with current price
      (VIF &gt; 50). They exist for the SCAN*PRO model (reference-price formation, stockpiling)
      but for demand forecasting, current price already carries the price signal.
      Including them causes multicollinearity without adding predictive information.</p>

    <p><b>Per-SKU training:</b> With only 5 features, each SKU&rsquo;s ~90 training rows give a
      18:1 sample-to-feature ratio &mdash; healthy for all three model types without pooling.
      Category pooling was tested but removed: SKUs within the same category have different
      demand levels and dynamics, so the pooled model learned the category average rather than
      each SKU&rsquo;s individual behaviour.</p>

    <p><b>Na&iuml;ve baseline (Moving Average):</b> Predicts the average of the last 8 training weeks.
      This is the standard retail forecasting benchmark. If an ML model cannot beat the na&iuml;ve
      baseline, it is overfitting or the data has no learnable structure beyond recent history.
      Including it makes model comparison honest.</p>

    <p><b>Hyperparameter tuning:</b> Random Forest and MLP hyperparameters selected via
      <code>GridSearchCV</code> with <code>TimeSeriesSplit</code> (3 folds) on category-pooled data
      with the same 5-feature set the models will use in production.</p>

    <p><b>Models:</b></p>
    <ul>
      <li><b>Na&iuml;ve (Moving Avg)</b> &mdash; No features. Benchmark baseline.</li>
      <li><b>Linear Regression</b> &mdash; OLS on raw features. Scale-invariant.</li>
      <li><b>Random Forest</b> &mdash; Tuned via CV on category-pooled data.</li>
      <li><b>Neural Network (MLP)</b> &mdash; Tuned via CV, StandardScaler-normalised inputs.</li>
    </ul>

    <p><b>Confidence intervals:</b> &sigma;<sub>residual</sub> &times; 1.96 &times; &radic;(step) &mdash;
      widens with forecast horizon (Hyndman &amp; Athanasopoulos, 2021).</p>
    <p><b>Feature importance (RF-based):</b> The diagnostics section shows Random Forest&rsquo;s
      importance ranking. RF importance is scale-invariant, captures non-linear effects, and is
      robust to multicollinearity &mdash; unlike LR coefficients (scale-dependent) or MLP weights (opaque).
      This tells the marketing manager which levers most influence each SKU&rsquo;s demand.</p>

    <p><b>All-SKU heatmap (RF-based):</b> The portfolio heatmap uses RF because it offers non-linear
      flexibility, requires no preprocessing (unlike MLP), and averages across many trees for robustness.
      For the per-SKU analysis, all four models are shown for honest comparison.</p>

    <p><b>Best model selection:</b> Lowest MAE on the held-out test set &mdash; na&iuml;ve baseline included.</p>
    <p style="margin-top:.75rem;font-size:.78rem;color:var(--slate-400);">
      References: Cohen, M.C. et al. (2022). <i>Demand Prediction in Retail.</i> Springer SSCM 14.
      &middot; Hyndman, R.J. &amp; Athanasopoulos, G. (2021). <i>Forecasting: Principles and Practice</i>, 3rd ed. OTexts.</p>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- TAB 2: PROMOTION EFFECTIVENESS                                        -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div id="tab-promo" class="tab-panel">

  <h3 class="section-title">Portfolio Promotion Overview</h3>
  <div id="promoPortfolioKpis" class="kpi-row"></div>

  <div class="controls">
    <select id="promoSkuSelect" onchange="renderPromoSku()"></select>
  </div>

  <div id="promoSkuHeader" class="sku-header"></div>
  <div id="promoSkuInterpBox" class="interp" style="display:none"></div>
  <div id="promoSkuKpiRow" class="kpi-row"></div>

  <h3 class="section-title">Incremental Sales Lift &mdash; All SKUs</h3>
  <div class="chart-card"><div id="promoLiftChart"></div></div>

  <h3 class="section-title">Promotion Effectiveness Summary</h3>
  <div id="promoSummaryTable"></div>

  <div class="methodology">
    <h3>Our Methodology &mdash; Feature Promotion Effectiveness</h3>

    <p><b>The question:</b> Does featuring a product on the platform&rsquo;s main page actually boost sales?
      And if so, by how much?</p>

    <p><b>Why not just compare promoted vs non-promoted weeks?</b> Because the comparison is confounded:
      retailers often discount <i>and</i> feature products simultaneously (price confounding),
      promotions cluster around high-demand periods like holidays (seasonal confounding),
      and if demand is growing over time, later promotional weeks look artificially good (trend confounding).
      We need a model that controls for all three.</p>

    <p><b>Why SCAN*PRO?</b> The SCAN*PRO model (Van Heerde et al., 2002) was specifically designed
      for measuring promotion effectiveness from observational retail data. It is the standard
      in marketing science for this application. We use <b>OLS</b> because the goal is
      <b>coefficient estimation</b> (measuring &beta;&#8322;, the promotion effect), not prediction.
      OLS gives unbiased, interpretable estimates with valid standard errors for hypothesis testing.</p>

    <p><b>Model:</b></p>
    <p style="margin:.4rem 0 .4rem 1.2rem;font-family:monospace;font-size:.82rem;">
      ln(sales<sub>it</sub>) = &alpha; + &beta;&#8321;&middot;ln(price<sub>it</sub>)
      + &beta;&#8322;&middot;feat<sub>it</sub>
      + &beta;&#8323;&middot;trend<sub>t</sub>
      + &beta;&#8324;&middot;ln(price<sub>i,t-1</sub>)
      + &beta;&#8325;&middot;ln(price<sub>i,t-2</sub>)
      + &Sigma;&gamma;<sub>k</sub>&middot;month<sub>k</sub> + &epsilon;<sub>it</sub></p>

    <p><b>Why each feature is included:</b></p>
    <ul>
      <li><b>ln(price<sub>t</sub>)</b> &mdash; Separates the price discount effect from the promotion effect.
        Without it, &beta;&#8322; captures both, overstating the promotion&rsquo;s own contribution.</li>
      <li><b>feat_main_page</b> &mdash; The variable of interest.
        Binary (0/1), so it enters linearly &mdash; exp(&beta;&#8322;) &minus; 1 = proportional sales lift.</li>
      <li><b>trend</b> &mdash; Prevents attributing demand growth to promotions if they cluster in later weeks.</li>
      <li><b>ln(price<sub>t-1</sub>), ln(price<sub>t-2</sub>)</b> &mdash; SCAN*PRO&rsquo;s defining feature.
        Captures reference-price formation and stockpiling.
        Van Heerde et al. (2002) showed that omitting price lags biases both &beta;&#8321; and &beta;&#8322;.</li>
      <li><b>month dummies</b> &mdash; Removes seasonal demand variation so &beta;&#8322;
        reflects genuine promotion response, not holiday peaks.</li>
    </ul>

    <p><b>Incremental sales:</b> Lift = e<sup>&beta;&#8322;</sup> &minus; 1;
      incremental/week = Q&#8320; &times; lift;
      total = incremental/week &times; promoted weeks.</p>
    <p><b>Significance:</b> OLS t-test on &beta;&#8322; (p &lt; 0.05).
      CI: exp(&beta;&#8322; &plusmn; 1.96&middot;SE) &minus; 1.</p>
    <p style="margin-top:.75rem;font-size:.78rem;color:var(--slate-400);">
      References: Van Heerde et al. (2002). <i>Schmalenbach Business Review</i>, 54, 198&ndash;220.
      &middot; Hanssens et al. (2001). <i>Market Response Models</i>, 2nd ed. Kluwer.
      &middot; Van Heerde et al. (2004). <i>Marketing Science</i>, 23(3), 317&ndash;334.</p>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- TAB 3: PRICE ELASTICITY                                               -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div id="tab-elasticity" class="tab-panel">

  <!-- Sub-tab bar -->
  <div style="display:flex;gap:0;border-bottom:2px solid var(--slate-200);margin-bottom:.75rem;">
    <button class="tab-btn active" onclick="switchElastSub('overview')" id="esub-overview">Overview</button>
    <button class="tab-btn" onclick="switchElastSub('skuDive')" id="esub-skuDive">SKU Deep Dive</button>
    <button class="tab-btn" onclick="switchElastSub('scenarios')" id="esub-scenarios">Scenario Testing</button>
    <button class="tab-btn" onclick="switchElastSub('portfolio')" id="esub-portfolio">Portfolio Analysis</button>
  </div>

  <!-- ═══════════════════ SUB-TAB 1: OVERVIEW ═══════════════════ -->
  <div id="elast-overview" class="elast-sub" style="display:block;">
    <div id="overviewKpis" class="kpi-row"></div>

    <h3 class="section-title">Price Elasticity by SKU</h3>
    <div class="chart-card"><div id="overviewBarChart"></div></div>

    <div class="grid-2">
      <div>
        <h3 class="section-title">Elasticity Distribution</h3>
        <div class="chart-card"><div id="overviewPie"></div></div>
      </div>
      <div>
        <h3 class="section-title">Elasticity Histogram</h3>
        <div class="chart-card"><div id="overviewHist"></div></div>
      </div>
    </div>

    <h3 class="section-title">Full Elasticity Results</h3>
    <div id="overviewTable" style="overflow-x:auto"></div>
  </div>

  <!-- ═══════════════════ SUB-TAB 2: SKU DEEP DIVE ═══════════════════ -->
  <div id="elast-skuDive" class="elast-sub" style="display:none;">
    <div class="controls">
      <select id="elastSkuSelect" onchange="renderSKUDive()"></select>
    </div>
    <div id="diveHeader" class="sku-header"></div>
    <div id="diveInterp" class="interp" style="display:none"></div>
    <div id="diveKpis" class="kpi-row"></div>

    <div class="grid-2">
      <div class="chart-card"><div id="diveDemandCurve"></div></div>
      <div class="chart-card"><div id="diveRevenueCurve"></div></div>
    </div>

    <h3 class="section-title">Revenue Impact by Price Scenario</h3>
    <div class="chart-card"><div id="diveWaterfall"></div></div>

    <h3 class="section-title">What-If Scenario Table</h3>
    <div id="diveScenarioTable"></div>
  </div>

  <!-- ═══════════════════ SUB-TAB 3: SCENARIO TESTING ═══════════════════ -->
  <div id="elast-scenarios" class="elast-sub" style="display:none;">
    <div class="controls">
      <select id="skuSelectScenario" onchange="renderScenarioTab()"></select>
    </div>
    <div id="scenarioInfo"></div>

    <h3 class="section-title">Standard Scenarios (&plusmn;5%, 10%, 20%, 30%)</h3>
    <div id="scenarioStdTable" style="overflow-x:auto"></div>

    <div class="grid-2">
      <div class="chart-card"><div id="scenarioDemandBar"></div></div>
      <div class="chart-card"><div id="scenarioRevenueBar"></div></div>
    </div>

    <h3 class="section-title">Custom Price Change</h3>
    <div style="width:100%;margin-bottom:.75rem;">
      <input type="range" id="customSlider" min="-30" max="30" value="0" step="1" oninput="updateCustomScenario()" style="width:100%;accent-color:#0000CD;cursor:pointer;display:block;">
      <div style="display:flex;justify-content:space-between;font-size:.8rem;color:var(--slate-400);margin-top:.25rem;"><span>-30%</span><span>0%</span><span>+30%</span></div>
      <div style="text-align:center;font-weight:700;font-size:1.1rem;margin:.4rem 0;color:#0000CD;" id="sliderLabel">0%</div>
    </div>
    <div id="customResult"></div>
  </div>

  <!-- ═══════════════════ SUB-TAB 4: PORTFOLIO ANALYSIS ═══════════════════ -->
  <div id="elast-portfolio" class="elast-sub" style="display:none;">
    <div id="portfolioKpis" class="kpi-row"></div>

    <h3 class="section-title">Elasticity by Product Category</h3>
    <div class="chart-card"><div id="portfolioByFunc"></div></div>

    <h3 class="section-title">Elasticity Ranking (with 95% CI)</h3>
    <div class="chart-card"><div id="rankingChart"></div></div>

    <h3 class="section-title">Revenue Impact Heatmap &mdash; All SKUs &times; All Scenarios</h3>
    <div class="chart-card"><div id="elastHeatmapChart"></div></div>

    <h3 class="section-title">Price Strategy Summary</h3>
    <div id="strategyTable"></div>
  </div>

  <div class="methodology">
    <h3>Our Methodology &mdash; Price Elasticity &amp; Scenario Testing</h3>
    <p><b>The question:</b> How sensitive is each SKU&rsquo;s demand to price changes?</p>
    <p><b>Model:</b> Scan*Pro log-log OLS with HC1 robust standard errors, fitted per SKU.</p>
    <p style="margin:.4rem 0 .4rem 1.2rem;font-family:monospace;font-size:.82rem;">
      log(sales) = &beta;&#8320; + &beta;&#8321;&middot;log(price) + &beta;&#8322;&middot;feat + &beta;&#8323;&middot;trend + &Sigma;&gamma;&middot;month + &epsilon;</p>
    <p>&beta;&#8321; = price elasticity directly. No lagged prices &mdash; we estimate steady-state
      elasticity for what-if scenarios.</p>
    <p><b>Scenario:</b> Q<sub>new</sub> = Q&#8320;&times;(P<sub>new</sub>/P&#8320;)<sup>&epsilon;</sup>,
      R<sub>new</sub> = P<sub>new</sub>&times;Q<sub>new</sub>.
      Elastic (|&epsilon;|&gt;1): cut price &rarr; grow revenue.
      Inelastic (|&epsilon;|&lt;1): raise price &rarr; grow revenue.</p>
    <p><b>Classification:</b> Inelastic (|&epsilon;|&lt;1), Moderately Elastic (1&le;|&epsilon;|&lt;2), Highly Elastic (|&epsilon;|&ge;2).</p>
    <p style="font-size:.78rem;color:var(--slate-400);margin-top:.5rem;">
      References: Van Heerde et al. (2002). <i>Schmalenbach Business Review</i>, 54.
      &middot; Hanssens et al. (2001). <i>Market Response Models</i>, 2nd ed.</p>
  </div>
</div>

</div><!-- /.container -->

<footer>Marketing Analytics Dashboard &middot; Demand Forecasting (scikit-learn) + Price Elasticity (Scan*Pro OLS)</footer>

<script>
/* ═══════════════════════════════════════════════════════════════════════════
   DATA
   ═══════════════════════════════════════════════════════════════════════════ */
const DEMAND_DATA = __DEMAND_DATA__;
const ELAST_DATA  = __ELAST_DATA__;

const plotCfg = {responsive:true, displayModeBar:false};

/* Colors */
const DM_COLORS = {"Naïve (Moving Avg)":"#64748b","Linear Regression":"#7C3AED","Random Forest":"#E11D48","Neural Network (MLP)":"#EA580C"};
const DM_MODEL_NAMES = ["Naïve (Moving Avg)","Linear Regression","Random Forest","Neural Network (MLP)"];
const BLUE   = "#2563EB", SLATE = "#0f172a", GREY_LG = "#cbd5e1",
      GREY_MD = "#94a3b8", RED = "#DC2626", GREEN = "#16a34a";

/* ═══════════════════════════════════════════════════════════════════════════
   TAB SWITCHING
   ═══════════════════════════════════════════════════════════════════════════ */
let activeTab = "guide";
let demandRendered = false, promoRendered = false, elastRendered = false;

/** Switch the active tab panel and lazy-render its content on first visit. */
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelector(`.tab-btn[onclick="switchTab('${tab}')"]`).classList.add("active");
  document.getElementById(`tab-${tab}`).classList.add("active");

  if (tab === "promo" && !promoRendered) {
    promoRendered = true;
    renderPromoPortfolio();
    renderPromoSku();
    renderPromoLiftChart();
    renderPromoSummaryTable();
  }
  if (tab === "elasticity" && !elastRendered) {
    elastRendered = true;
    ePopulateSelects();
    renderElastOverview();  // First sub-tab renders immediately
  }
  if (tab === "demand" && !demandRendered) {
    demandRendered = true;
    renderDemand();
    renderDemandHeatmap();
  }

  // Relayout visible plotly charts so they resize properly
  setTimeout(() => {
    document.querySelectorAll(`#tab-${tab} .js-plotly-plot`).forEach(el => {
      Plotly.Plots.resize(el);
    });
  }, 50);
}

/* ═══════════════════════════════════════════════════════════════════════════
   SHARED HELPERS
   ═══════════════════════════════════════════════════════════════════════════ */
/** Generate HTML for a single KPI card. */
function kpi(label, value, sub) {
  return `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-sub">${sub}</div></div>`;
}

/** Add N days to a YYYY-MM-DD date string; used to generate future forecast dates. */
function addDays(dateStr, days) {
  const d = new Date(dateStr); d.setDate(d.getDate() + days);
  return d.toISOString().split("T")[0];
}


/* ═══════════════════════════════════════════════════════════════════════════
   DEMAND FORECASTING
   ═══════════════════════════════════════════════════════════════════════════ */
const demandSel       = document.getElementById("demandSkuSelect");
const horizonSlider   = document.getElementById("horizonSlider");
const horizonVal      = document.getElementById("horizonVal");
const testSlider      = document.getElementById("testSlider");
const testVal         = document.getElementById("testVal");

horizonSlider.addEventListener("input", () => { horizonVal.textContent = horizonSlider.value; });
testSlider.addEventListener("input", () => { testVal.textContent = testSlider.value; });

Object.keys(DEMAND_DATA.skus).sort((a,b)=>+a-+b).forEach(id => {
  const opt = document.createElement("option");
  opt.value = id;
  opt.textContent = `SKU ${id} — ${DEMAND_DATA.skus[id].meta.functionality}`;
  demandSel.appendChild(opt);
});

/** Render the full demand forecasting view for the selected SKU, test size, and horizon. */
function renderDemand() {
  const id = demandSel.value;
  const s  = DEMAND_DATA.skus[id];
  const m  = s.meta;
  const testSize = parseInt(testSlider.value);
  const horizon  = parseInt(horizonSlider.value);

  const cfg = s.by_test[String(testSize)];
  if (!cfg) {
    document.getElementById("demandHeader").innerHTML =
      `<h2>SKU ${id} — ${m.functionality}</h2><p style="color:#ef4444;">Insufficient data for test size ${testSize}.</p>`;
    return;
  }

  const nWeeks = s.weeks.length;
  const wTrain = s.weeks.slice(0, nWeeks-testSize);
  const wTest  = s.weeks.slice(nWeeks-testSize);
  const yTrain = s.sales.slice(0, nWeeks-testSize);
  const yTest  = s.sales.slice(nWeeks-testSize);
  const best   = cfg.best_model;
  const bRes   = cfg.results[best];

  const lastTestDate = wTest[wTest.length-1];
  const futureWeeks = [];
  for (let i=1;i<=horizon;i++) futureWeeks.push(addDays(lastTestDate, i*7));

  document.getElementById("demandHeader").innerHTML =
    `<h2>SKU ${id} — ${m.functionality}</h2>
     <p>Color: ${m.color} · Avg price: $${m.avg_price} · Avg sales: ${m.avg_sales}/wk · Promo rate: ${m.feat_rate}%</p>`;

  const avgHist = s.sales.slice(-12).reduce((a,b)=>a+b,0)/12;
  const fcSlice = bRes.forecast.slice(0,horizon);
  const avgFc   = fcSlice.reduce((a,b)=>a+b,0)/horizon;
  const delta   = ((avgFc-avgHist)/Math.max(avgHist,1))*100;
  const sym     = delta>=0 ? "▲" : "▼";
  const r2q     = bRes.r2>0.5 ? "Good ✓" : "Moderate";
  const ciLo    = bRes.ci_lower.slice(0,horizon);
  const ciHi    = bRes.ci_upper.slice(0,horizon);
  const avgCiLo = Math.round(ciLo.reduce((a,b)=>a+b,0)/horizon);
  const avgCiHi = Math.round(ciHi.reduce((a,b)=>a+b,0)/horizon);
  const trendWord = delta > 5 ? "growing demand" : delta < -5 ? "declining demand" : "stable demand";
  const trendAdv  = delta > 5 ? "an upward trend" : delta < -5 ? "a downward trend" : "a relatively flat trend";

  const interpBox = document.getElementById("demandInterpBox");
  interpBox.innerHTML = `<b>Expected demand:</b> SKU ${id} is forecast to sell approximately <b>${Math.round(avgFc)} units/week</b> over the next <b>${horizon} weeks</b> (95% CI: ${avgCiLo}–${avgCiHi} units). This represents a <b>${Math.abs(delta).toFixed(1)}% ${delta>=0?"increase":"decrease"}</b> compared to the recent 12-week average of ${Math.round(avgHist)} units/wk, suggesting <b>${trendWord}</b>. The best-performing model is <b>${best}</b> (MAE: ${bRes.mae.toFixed(1)}, R²: ${bRes.r2.toFixed(3)}).`;
  interpBox.style.display = "block";

  document.getElementById("demandKpiRow").innerHTML = [
    kpi(`Avg Forecast (${horizon}w)`, `${Math.round(avgFc)} units/wk`, `${sym} ${Math.abs(delta).toFixed(1)}% vs recent · ${trendWord}`),
    kpi("95% CI Range", `${avgCiLo} – ${avgCiHi}`, "units/wk (best model avg)"),
    kpi("Trend", `${sym} ${Math.abs(delta).toFixed(1)}%`, `${trendAdv} vs last 12 weeks`),
    kpi("Best Model", best.split("(")[0].trim(), "Lowest MAE on test"),
    kpi("Test MAE", bRes.mae.toFixed(1), "units/week error"),
    kpi("Test R²", bRes.r2.toFixed(3), r2q),
  ].join("");

  /* Forecast chart */
  const traces = [];
  traces.push({x:wTrain,y:yTrain,mode:"lines",name:"Historical",line:{color:"#00008B",width:1.8},
    hovertemplate:"<b>%{x|%d %b %Y}</b><br>Sales: %{y:.0f}<extra>Historical</extra>"});
  traces.push({x:wTest,y:yTest,mode:"lines+markers",name:"Actual (test)",line:{color:"#16a34a",width:2.5},marker:{size:6,color:"#16a34a"},
    hovertemplate:"<b>%{x|%d %b %Y}</b><br>Actual: %{y:.0f}<extra>Actual</extra>"});

  DM_MODEL_NAMES.forEach(name => {
    const r=cfg.results[name]; const isBest=name===best;
    traces.push({x:wTest,y:r.y_pred_test,mode:"lines",name:name+" (test)",
      line:{color:DM_COLORS[name],width:isBest?3:2,dash:isBest?null:"dot"},opacity:isBest?1:0.7});
    traces.push({x:futureWeeks,y:r.forecast.slice(0,horizon),mode:"lines+markers",name:name+" (forecast)",
      line:{color:DM_COLORS[name],width:isBest?3:2},marker:{size:isBest?8:5,symbol:"diamond"},opacity:isBest?1:0.7});
    if(isBest){
      const ciUp=r.ci_upper.slice(0,horizon),ciLo=r.ci_lower.slice(0,horizon);
      traces.push({x:futureWeeks.concat([...futureWeeks].reverse()),
        y:ciUp.concat([...ciLo].reverse()),fill:"toself",fillcolor:"rgba(99,102,241,0.12)",
        line:{color:"rgba(0,0,0,0)"},name:"95% CI",showlegend:true,hoverinfo:"skip"});
    }
  });

  const allVals = [...s.sales, ...bRes.y_pred_test, ...bRes.forecast.slice(0,horizon), ...bRes.ci_upper.slice(0,horizon)];
  const yMax = Math.max(...allVals) * 1.15;

  Plotly.newPlot("forecastChart",traces,{
    title:`SKU ${id} — Demand Forecast`,height:460,hovermode:"closest",
    legend:{orientation:"h",y:-0.22,x:0.5,xanchor:"center",font:{size:11}},
    margin:{l:50,r:30,t:55,b:75},
    xaxis:{title:"Week",type:"date",tickformat:"%d %b %Y"},yaxis:{title:"Weekly sales (units)",range:[0,yMax],dtick:25},
    shapes:[{type:"line",x0:lastTestDate,x1:lastTestDate,y0:0,y1:1,yref:"paper",
      line:{width:1.5,dash:"dash",color:"#94a3b8"}}],
    annotations:[{x:lastTestDate,y:1,yref:"paper",text:"Forecast →",showarrow:false,
      font:{size:11,color:"#64748b"},xanchor:"left",yanchor:"top"}],
    hoverlabel:{bgcolor:"white",bordercolor:"#cbd5e1",font:{size:12}},
  },plotCfg);

  /* Forecast table */
  let html="<table><tr><th class='lt'>Week</th>";
  DM_MODEL_NAMES.forEach(n=>html+=`<th class='lt'>${n}</th>`);
  html+="<th class='lt'>95% CI (best)</th></tr>";
  for(let i=0;i<horizon;i++){
    html+=`<tr><td>${futureWeeks[i]}</td>`;
    DM_MODEL_NAMES.forEach(n=>html+=`<td>${Math.round(cfg.results[n].forecast[i])}</td>`);
    html+=`<td>[${Math.round(bRes.ci_lower[i])}, ${Math.round(bRes.ci_upper[i])}]</td></tr>`;
  }
  html+="</table>";
  document.getElementById("forecastTable").innerHTML=html;

  /* Residuals */
  const residuals=yTest.map((v,i)=>+(v-bRes.y_pred_test[i]).toFixed(2));
  Plotly.newPlot("residChart",[{x:wTest,y:residuals,type:"bar",
    marker:{color:residuals.map(r=>r>=0?"#6366f1":"#ef4444")},
    hovertemplate:"<b>%{x|%d %b %Y}</b><br>Residual: %{y:.1f}<extra></extra>"}],{
    title:`Residuals — ${best}`,height:300,margin:{l:45,r:20,t:40,b:30},
    xaxis:{title:"Week",type:"date",tickformat:"%d %b %Y"},yaxis:{title:"Actual − Predicted"},
    shapes:[{type:"line",x0:wTest[0],x1:wTest[wTest.length-1],y0:0,y1:0,line:{color:"#94a3b8",width:1}}],
  },plotCfg);

  /* Feature importance */
  const fi=cfg.feature_importance;
  const fiSorted=Object.entries(fi).sort((a,b)=>a[1]-b[1]);
  Plotly.newPlot("featChart",[{y:fiSorted.map(e=>e[0]),x:fiSorted.map(e=>e[1]),
    type:"bar",orientation:"h",marker:{color:"#6366f1"},
    text:fiSorted.map(e=>e[1].toFixed(3)),textposition:"outside",textfont:{size:10,family:"monospace"}}],{
    title:"Feature Importance (RF)",height:300,margin:{l:150,r:60,t:40,b:30},xaxis:{title:"Importance"},
  },plotCfg);
}

/** Render the all-SKU demand forecast heatmap (Random Forest, 4 weeks ahead). */
function renderDemandHeatmap() {
  const ids=Object.keys(DEMAND_DATA.heatmap).sort((a,b)=>+a-+b);
  const yLabels=ids.map(id=>`SKU ${id}`);
  const z=ids.map(id=>DEMAND_DATA.heatmap[id]);
  Plotly.newPlot("demandHeatmapChart",[{
    z:z,x:DEMAND_DATA.heatmap_weeks,y:yLabels,type:"heatmap",colorscale:"Blues",
    text:z.map(row=>row.map(v=>Math.round(v))),texttemplate:"%{text}",textfont:{size:9},
    hovertemplate:"<b>%{y}</b><br>Week: %{x}<br>Forecast: %{z:.0f} units<extra></extra>",
  }],{
    title:"Forecasted Demand — All SKUs (Random Forest)",
    height:Math.max(500,ids.length*22),margin:{l:70,r:20,t:50,b:30},
    yaxis:{dtick:1,autorange:"reversed"},
  },plotCfg);
}


/* ═══════════════════════════════════════════════════════════════════════════
   PRICE ELASTICITY — 4 sub-tabs (Overview, SKU Deep Dive, Scenario Testing, Portfolio)
   ═══════════════════════════════════════════════════════════════════════════ */
const E = ELAST_DATA;
const ELAST_SCENARIOS = [-30,-20,-10,-5,5,10,20,30];
const E_COLORS = {inelastic:'#22c55e', moderate:'#f59e0b', elastic:'#ef4444'};
const E_CAT_COLORS = {'Inelastic':E_COLORS.inelastic,'Moderately Elastic':E_COLORS.moderate,'Highly Elastic':E_COLORS.elastic};

// Build flat data array from ELAST_DATA.skus for convenience
const E_LIST = Object.values(E.skus).sort((a,b)=>a.sku-b.sku);

function eClassify(e) {
  return Math.abs(e)<1?'Inelastic':Math.abs(e)<2?'Moderately Elastic':'Highly Elastic';
}
function eScenario(P0,Q0,e,pct) {
  const Pn=P0*(1+pct/100), Qn=Math.max(Q0*Math.pow(Pn/P0,e),0), R0=P0*Q0, Rn=Pn*Qn;
  return {Pn,Qn,Rn,dDem:((Qn-Q0)/Q0)*100,dRev:((Rn-R0)/R0)*100};
}
function eFmtPct(v){return (v>0?'+':'')+v.toFixed(1)+'%';}
function eFmtMoney(v){return '£'+v.toLocaleString('en-GB',{minimumFractionDigits:2,maximumFractionDigits:2});}
function ePctCls(v){return v>0?'pos':v<0?'neg':'';}

// Populate all SKU dropdowns
function ePopulateSelects() {
  ['elastSkuSelect','skuSelectScenario'].forEach(selId => {
    const sel = document.getElementById(selId);
    if(!sel) return;
    E_LIST.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.sku; opt.textContent = d.label;
      sel.appendChild(opt);
    });
  });
}

// Sub-tab switching
let elastSubRendered = {overview:false, skuDive:false, scenarios:false, portfolio:false};

function switchElastSub(sub) {
  document.querySelectorAll('.elast-sub').forEach(p=>p.style.display='none');
  document.querySelectorAll('[id^="esub-"]').forEach(b=>b.classList.remove('active'));
  document.getElementById('elast-'+sub).style.display='block';
  document.getElementById('esub-'+sub).classList.add('active');

  if(!elastSubRendered[sub]) {
    elastSubRendered[sub] = true;
    if(sub==='overview') renderElastOverview();
    if(sub==='skuDive') renderSKUDive();
    if(sub==='scenarios') renderScenarioTab();
    if(sub==='portfolio') renderPortfolioAnalysis();
  }
  setTimeout(()=>{
    document.querySelectorAll('#elast-'+sub+' .js-plotly-plot').forEach(el=>Plotly.Plots.resize(el));
  },50);
}

/* ═══════════════════ OVERVIEW TAB ═══════════════════ */
function renderElastOverview() {
  const meanE = E_LIST.reduce((s,d)=>s+d.elasticity,0)/E_LIST.length;
  const mostEl = E_LIST.reduce((a,b)=>a.elasticity<b.elasticity?a:b);
  const leastEl = E_LIST.reduce((a,b)=>Math.abs(a.elasticity)<Math.abs(b.elasticity)?a:b);
  const nSig = E_LIST.filter(d=>d.elast_sig).length;

  document.getElementById('overviewKpis').innerHTML=[
    kpi('Total SKUs',''+E_LIST.length,''),
    kpi('Mean Elasticity',meanE.toFixed(3),''),
    kpi('Most Elastic',`SKU ${mostEl.sku}`,`ε = ${mostEl.elasticity.toFixed(3)}`),
    kpi('Least Elastic',`SKU ${leastEl.sku}`,`ε = ${leastEl.elasticity.toFixed(3)}`),
    kpi('Significant (p<0.05)',`${nSig} / ${E_LIST.length}`,''),
  ].join('');

  // Bar chart with CIs
  const sorted=[...E_LIST].sort((a,b)=>a.elasticity-b.elasticity);
  Plotly.newPlot('overviewBarChart',[{
    type:'bar',orientation:'h',
    y:sorted.map(d=>'SKU '+d.sku),x:sorted.map(d=>d.elasticity),
    marker:{color:sorted.map(d=>d.elast_sig?E_CAT_COLORS[eClassify(d.elasticity)]:'#cbd5e1')},
    hovertemplate:'<b>SKU %{y}</b><br>ε = %{x:.3f}<br>p = %{customdata[0]:.4f} | R² = %{customdata[1]:.3f}<extra></extra>',
    customdata:sorted.map(d=>[d.elast_pval,d.r2]),
  }],{
    height:900,margin:{l:70,r:30,t:30,b:40},
    xaxis:{title:'Price Elasticity',zeroline:true,zerolinecolor:'black'},
    shapes:[
      {type:'line',x0:-1,x1:-1,y0:0,y1:1,yref:'paper',line:{dash:'dash',color:'gray',width:1}},
      {type:'line',x0:-2,x1:-2,y0:0,y1:1,yref:'paper',line:{dash:'dot',color:'gray',width:1}}
    ],
    annotations:[
      {x:-1,y:1,yref:'paper',text:'|ε|=1',showarrow:false,yanchor:'bottom',font:{size:10,color:'gray'}},
      {x:-2,y:1,yref:'paper',text:'|ε|=2',showarrow:false,yanchor:'bottom',font:{size:10,color:'gray'}}
    ]
  },plotCfg);

  // Pie chart
  const cats={};
  E_LIST.forEach(d=>{const c=eClassify(d.elasticity);cats[c]=(cats[c]||0)+1;});
  Plotly.newPlot('overviewPie',[{
    type:'pie',labels:Object.keys(cats),values:Object.values(cats),
    marker:{colors:Object.keys(cats).map(c=>E_CAT_COLORS[c])},
    textinfo:'label+value+percent',hole:0.35
  }],{height:350,margin:{t:20,b:20}},plotCfg);

  // Histogram
  Plotly.newPlot('overviewHist',[{
    type:'histogram',x:E_LIST.map(d=>d.elasticity),nbinsx:20,
    marker:{color:BLUE,opacity:0.7}
  }],{
    height:350,margin:{t:20,b:40},xaxis:{title:'Price Elasticity'},
    shapes:[
      {type:'line',x0:-1,x1:-1,y0:0,y1:1,yref:'paper',line:{dash:'dash',color:'gray'}},
      {type:'line',x0:-2,x1:-2,y0:0,y1:1,yref:'paper',line:{dash:'dot',color:'gray'}}
    ]
  },plotCfg);

  // Results table
  let html='<table><tr><th>SKU</th><th>Product</th><th>Elasticity</th><th>P-value</th><th>R²</th><th>Avg Price</th><th>Category</th><th>Sig.</th></tr>';
  E_LIST.forEach(d=>{
    const pFmt = d.elast_pval<0.001 ? d.elast_pval.toExponential(2) : d.elast_pval.toFixed(4);
    html+=`<tr><td>${d.sku}</td><td class="label-col">${d.label.split('—')[1]?.trim()||''}</td><td class="${d.elasticity<-1?'neg':''}">${d.elasticity.toFixed(3)}</td><td>${pFmt}</td><td>${d.r2.toFixed(3)}</td><td>£${d.avg_price.toFixed(2)}</td><td>${eClassify(d.elasticity)}</td><td class="${d.elast_sig?'sig':''}">${d.elast_sig?'★':''}</td></tr>`;
  });
  document.getElementById('overviewTable').innerHTML=html+'</table>';
}

/* ═══════════════════ SKU DEEP DIVE TAB ═══════════════════ */
function renderSKUDive() {
  const sel=document.getElementById('elastSkuSelect');
  const id=sel.value;
  const r=E.skus[id];
  if(!r) return;
  const e=r.elasticity, P0=r.avg_price, Q0=r.avg_sales, R0=r.avg_revenue;

  document.getElementById('diveHeader').innerHTML=
    `<h2>${r.label}</h2><p>Category: ${r.category} · Avg price: £${P0.toFixed(2)} · Avg sales: ${Q0.toFixed(0)} units/wk · Promo rate: ${r.promo_rate}%</p>`;

  let interp;
  if(e<-1) interp=`Elastic — |ε|=${Math.abs(e).toFixed(2)}>1. A 10% price rise → ~${(Math.abs(e)*10).toFixed(0)}% demand fall. <b>Lowering price grows revenue.</b>`;
  else if(e<0) interp=`Inelastic — |ε|=${Math.abs(e).toFixed(2)}<1. A 10% price rise → only ~${(Math.abs(e)*10).toFixed(0)}% demand fall. <b>Raising price grows revenue.</b>`;
  else interp=`Unusual positive elasticity (ε=${e.toFixed(2)}). Check significance (p=${r.elast_pval.toFixed(4)}).`;
  const box=document.getElementById('diveInterp');
  box.innerHTML=interp;box.style.display='block';

  document.getElementById('diveKpis').innerHTML=[
    kpi('Elasticity',e.toFixed(4),`p=${r.elast_pval.toFixed(4)} · ${r.elast_sig?'★ sig':'not sig'}`),
    kpi('Model R²',r.r2.toFixed(4),'Scan*Pro fit'),
    kpi('Avg Price',`£${P0.toFixed(2)}`,`Sales: ${Q0.toFixed(0)}/wk`),
    kpi('Optimal Price',`£${r.optimal_price.toFixed(2)}`,`Peak £${r.optimal_revenue.toLocaleString('en-GB',{maximumFractionDigits:0})}/wk`),
  ].join('');

  // Demand curve
  Plotly.newPlot('diveDemandCurve',[
    {x:r.price_range,y:r.demand_curve,mode:'lines',line:{color:BLUE,width:2},hovertemplate:'Price: £%{x:.2f}<br>Demand: %{y:.0f}<extra></extra>'},
    {x:[P0],y:[Q0],mode:'markers',marker:{size:11,color:SLATE,line:{color:'white',width:2}},hovertemplate:`Current: £${P0.toFixed(2)} → ${Q0.toFixed(0)}<extra></extra>`}
  ],{height:360,margin:{t:30,b:40,l:60,r:20},title:`Demand Curve (ε=${e.toFixed(3)})`,xaxis:{title:'Price (£)'},yaxis:{title:'Weekly demand'}},plotCfg);

  // Revenue curve
  Plotly.newPlot('diveRevenueCurve',[
    {x:r.price_range,y:r.revenue_curve,mode:'lines',line:{color:GREY_LG,width:2},hovertemplate:'Price: £%{x:.2f}<br>Revenue: £%{y:,.0f}<extra></extra>'},
    {x:[P0],y:[R0],mode:'markers',marker:{size:11,color:SLATE,line:{color:'white',width:2}},hovertemplate:`Current: £${P0.toFixed(2)} → £${R0.toLocaleString()}<extra></extra>`},
    {x:[r.optimal_price],y:[r.optimal_revenue],mode:'markers',marker:{size:14,color:BLUE,symbol:'star',line:{color:'white',width:1.5}},hovertemplate:`Optimal: £${r.optimal_price.toFixed(2)} → £${r.optimal_revenue.toLocaleString()}<extra></extra>`}
  ],{height:360,margin:{t:30,b:40,l:60,r:20},title:`Revenue Curve — Optimal: £${r.optimal_price.toFixed(2)}`,xaxis:{title:'Price (£)'},yaxis:{title:'Weekly revenue (£)'}},plotCfg);

  // Waterfall
  const sc=r.scenarios;
  Plotly.newPlot('diveWaterfall',[{
    x:sc.map(s=>s.label),y:sc.map(s=>s.d_rev),type:'bar',
    marker:{color:sc.map(s=>s.d_rev>=0?GREEN:RED)},
    text:sc.map(s=>`£${s.d_rev>=0?'+':''}${s.d_rev.toFixed(0)}`),textposition:'outside',textfont:{size:10,color:GREY_MD},
  }],{height:340,margin:{t:30,b:40,l:60,r:40},title:`Revenue Impact (ε=${e.toFixed(3)})`,
    xaxis:{title:'Price change'},yaxis:{title:'Δ Revenue (£/wk)'},
    shapes:[{type:'line',x0:sc[0].label,x1:sc[sc.length-1].label,y0:0,y1:0,line:{color:GREY_LG}}]
  },plotCfg);

  // Scenario table
  let tbl='<table><tr><th>Scenario</th><th>New Price</th><th>Δ Demand</th><th>Δ Demand %</th><th>New Revenue</th><th>Δ Revenue</th><th>Δ Revenue %</th></tr>';
  sc.forEach(s=>{
    const dCls=s.d_rev>=0?'pos':'neg';
    tbl+=`<tr><td><b>${s.label}</b></td><td>£${s.new_price.toFixed(2)}</td><td class="${s.d_demand>=0?'pos':'neg'}">${s.d_demand>=0?'+':''}${s.d_demand.toFixed(1)}</td><td class="${s.d_demand_pct>=0?'pos':'neg'}">${s.d_demand_pct>=0?'+':''}${s.d_demand_pct.toFixed(1)}%</td><td>£${s.new_rev.toFixed(0)}</td><td class="${dCls}">${s.d_rev>=0?'+':''}£${s.d_rev.toFixed(0)}</td><td class="${dCls}">${s.d_rev_pct>=0?'+':''}${s.d_rev_pct.toFixed(1)}%</td></tr>`;
  });
  document.getElementById('diveScenarioTable').innerHTML=tbl+'</table>';
}

/* ═══════════════════ SCENARIO TESTING TAB ═══════════════════ */
function renderScenarioTab() {
  const sel=document.getElementById('skuSelectScenario');
  const id=+sel.value;
  const d=E.skus[id];
  if(!d) return;
  const P0=d.avg_price,Q0=d.avg_sales,e=d.elasticity;

  document.getElementById('scenarioInfo').innerHTML=
    `<div class="kpi-row">${[
      kpi('SKU',`${d.sku}`,''),
      kpi('Elasticity',e.toFixed(3),eClassify(e)),
      kpi('Avg Price',eFmtMoney(P0),''),
      kpi('Avg Demand',Q0.toFixed(0)+' /wk',''),
    ].join('')}</div>`;

  // Standard scenarios table
  let tbl='<table><tr><th>Scenario</th><th>New Price</th><th>New Demand</th><th>Demand Δ</th><th>New Revenue</th><th>Revenue Δ</th></tr>';
  const rows=ELAST_SCENARIOS.map(pct=>{
    const r=eScenario(P0,Q0,e,pct);
    return {pct,...r};
  });
  rows.forEach(r=>{
    tbl+=`<tr><td><b>${r.pct>0?'+':''}${r.pct}%</b></td><td>${eFmtMoney(r.Pn)}</td><td>${Math.round(r.Qn)}</td><td class="${ePctCls(r.dDem)}">${eFmtPct(r.dDem)}</td><td>${eFmtMoney(r.Rn)}</td><td class="${ePctCls(r.dRev)}">${eFmtPct(r.dRev)}</td></tr>`;
  });
  document.getElementById('scenarioStdTable').innerHTML=tbl+'</table>';

  // Demand bar
  Plotly.newPlot('scenarioDemandBar',[{
    x:rows.map(r=>(r.pct>0?'+':'')+r.pct+'%'),y:rows.map(r=>r.dDem),type:'bar',
    marker:{color:rows.map(r=>r.dDem>0?GREEN:RED)},
    text:rows.map(r=>eFmtPct(r.dDem)),textposition:'outside',textfont:{size:9}
  }],{height:320,margin:{t:30,b:40},title:'Demand Change (%)',yaxis:{title:'%'}},plotCfg);

  // Revenue bar
  Plotly.newPlot('scenarioRevenueBar',[{
    x:rows.map(r=>(r.pct>0?'+':'')+r.pct+'%'),y:rows.map(r=>r.dRev),type:'bar',
    marker:{color:rows.map(r=>r.dRev>0?GREEN:RED)},
    text:rows.map(r=>eFmtPct(r.dRev)),textposition:'outside',textfont:{size:9}
  }],{height:320,margin:{t:30,b:40},title:'Revenue Change (%)',yaxis:{title:'%'}},plotCfg);

  // Reset custom slider
  document.getElementById('customSlider').value=0;
  document.getElementById('sliderLabel').textContent='0%';
  document.getElementById('customResult').innerHTML='<div class="interp">Move the slider to simulate a custom price change.</div>';
}

function updateCustomScenario() {
  const pct=+document.getElementById('customSlider').value;
  document.getElementById('sliderLabel').textContent=(pct>0?'+':'')+pct+'%';
  if(pct===0){document.getElementById('customResult').innerHTML='<div class="interp">Move the slider to simulate a custom price change.</div>';return;}
  const id=+document.getElementById('skuSelectScenario').value;
  const d=E.skus[id];
  const r=eScenario(d.avg_price,d.avg_sales,d.elasticity,pct);
  document.getElementById('customResult').innerHTML=`
    <div class="kpi-row">
      ${kpi('New Price',eFmtMoney(r.Pn),eFmtPct(pct)+' from '+eFmtMoney(d.avg_price))}
      ${kpi('New Demand',Math.round(r.Qn)+'',`<span class="${ePctCls(r.dDem)}">${eFmtPct(r.dDem)}</span>`)}
      ${kpi('New Revenue',eFmtMoney(r.Rn),`<span class="${ePctCls(r.dRev)}">${eFmtPct(r.dRev)}</span>`)}
    </div>
    <div class="interp">A <b>${pct>0?'+':''}${pct}%</b> price change would ${r.dDem<0?'decrease':'increase'} demand by <b>${Math.abs(r.dDem).toFixed(1)}%</b> and ${r.dRev<0?'decrease':'increase'} revenue by <b>${Math.abs(r.dRev).toFixed(1)}%</b>.</div>`;
}

/* ═══════════════════ PORTFOLIO ANALYSIS TAB ═══════════════════ */
function renderPortfolioAnalysis() {
  const cats={};
  E_LIST.forEach(d=>{const c=eClassify(d.elasticity);cats[c]=(cats[c]||0)+1;});

  // +10% impact
  const res10=E_LIST.map(d=>({...d,...eScenario(d.avg_price,d.avg_sales,d.elasticity,10)}));
  const raiseCount=res10.filter(r=>r.dRev>0).length;

  document.getElementById('portfolioKpis').innerHTML=[
    kpi('Total SKUs',''+E_LIST.length,''),
    kpi('Inelastic',''+(cats['Inelastic']||0),'|ε| < 1'),
    kpi('Moderate',''+(cats['Moderately Elastic']||0),'1 ≤ |ε| < 2'),
    kpi('Highly Elastic',''+(cats['Highly Elastic']||0),'|ε| ≥ 2'),
    kpi('+10% Rev Gainers',`${raiseCount} / ${E_LIST.length}`,'inelastic — can raise price'),
  ].join('');

  // Category box plot
  const funcGroups={};
  E_LIST.forEach(d=>{
    const fname=d.category||d.label.split('—')[1]?.trim()||'Unknown';
    if(!funcGroups[fname]) funcGroups[fname]=[];
    funcGroups[fname].push(d.elasticity);
  });
  const funcNames=Object.keys(funcGroups).sort((a,b)=>{
    const ma=funcGroups[a].reduce((s,v)=>s+v,0)/funcGroups[a].length;
    const mb=funcGroups[b].reduce((s,v)=>s+v,0)/funcGroups[b].length;
    return ma-mb;
  });
  Plotly.newPlot('portfolioByFunc',funcNames.map(name=>({
    type:'box',y:funcGroups[name],name:name,boxpoints:'all',jitter:0.4,pointpos:0,
    marker:{size:5,opacity:0.6},
    hovertemplate:'<b>%{x}</b><br>ε = %{y:.3f}<extra></extra>'
  })),{
    height:400,margin:{t:20,b:80,l:60,r:20},
    yaxis:{title:'Price Elasticity (ε)',zeroline:true,zerolinecolor:'black'},
    xaxis:{tickangle:-30},showlegend:false,
    shapes:[
      {type:'line',x0:0,x1:1,xref:'paper',y0:-1,y1:-1,line:{dash:'dash',color:'gray',width:1}},
      {type:'line',x0:0,x1:1,xref:'paper',y0:-2,y1:-2,line:{dash:'dot',color:'gray',width:1}}
    ]
  },plotCfg);

  // Ranking with CIs
  const sorted=[...E_LIST].sort((a,b)=>a.elasticity-b.elasticity);
  Plotly.newPlot('rankingChart',[{
    type:'bar',orientation:'h',
    y:sorted.map(d=>'SKU '+d.sku+' — '+d.label.split('—')[1]?.trim().substring(0,30)+(d.elast_sig?' ★':'')),
    x:sorted.map(d=>d.elasticity),
    marker:{color:sorted.map(d=>d.elast_sig?E_CAT_COLORS[eClassify(d.elasticity)]:'#cbd5e1')},
    hovertemplate:'<b>%{y}</b><br>ε = %{x:.3f}<extra></extra>',
  }],{
    height:900,margin:{l:320,r:30,t:20,b:40},
    xaxis:{title:'Price Elasticity (ε)',zeroline:true,zerolinecolor:'black'},
    yaxis:{tickfont:{size:10},automargin:true},
    shapes:[{type:'line',x0:-1,x1:-1,y0:0,y1:1,yref:'paper',line:{dash:'dash',color:'gray'}}],
    annotations:[{x:0.99,y:0.01,xref:'paper',yref:'paper',text:'★ = significant (p<0.05)',
      showarrow:false,font:{size:9,color:'#94a3b8'},bgcolor:'white'}]
  },plotCfg);

  // Heatmap
  const hm=E.heatmap;
  Plotly.newPlot('elastHeatmapChart',[{
    type:'heatmap',z:hm.z,x:hm.scenario_labels,y:hm.sku_labels,
    text:hm.text,texttemplate:'%{text}',textfont:{size:8,color:'#334155'},
    colorscale:[[0,'#DC2626'],[0.4,'#FCA5A5'],[0.5,'#F8FAFC'],[0.6,'#86EFAC'],[1,'#16a34a']],
    zmid:0,colorbar:{title:'ΔRev (%)',ticksuffix:'%',len:0.7},
    hovertemplate:'<b>%{y}</b><br>Scenario: %{x}<br>Revenue: <b>%{z:+.1f}%</b><extra></extra>'
  }],{
    height:1000,margin:{l:220,r:110,t:40,b:40},
    xaxis:{title:'Price Change',side:'top'},
    yaxis:{tickfont:{size:9},autorange:'reversed',automargin:true}
  },plotCfg);

  // Strategy table
  const strats=E_LIST.map(d=>{
    const up=eScenario(d.avg_price,d.avg_sales,d.elasticity,10);
    const dn=eScenario(d.avg_price,d.avg_sales,d.elasticity,-10);
    let strategy=up.dRev>0?'Raise Price':dn.dRev>up.dRev?'Lower Price':'Maintain';
    return {...d,up,dn,strategy};
  }).sort((a,b)=>b.up.dRev-a.up.dRev);

  let tbl='<table><tr><th>SKU</th><th>Product</th><th>Elasticity</th><th>Category</th><th>+10% Rev</th><th>-10% Rev</th><th>Strategy</th></tr>';
  strats.forEach(s=>{
    tbl+=`<tr><td>${s.sku}</td><td class="label-col">${s.label.split('—')[1]?.trim()||''}</td><td>${s.elasticity.toFixed(3)}</td><td>${eClassify(s.elasticity)}</td><td class="${ePctCls(s.up.dRev)}">${eFmtPct(s.up.dRev)}</td><td class="${ePctCls(s.dn.dRev)}">${eFmtPct(s.dn.dRev)}</td><td><b>${s.strategy}</b></td></tr>`;
  });
  document.getElementById('strategyTable').innerHTML=tbl+'</table>';
}

/* ═══════════════════════════════════════════════════════════════════════════
   PROMOTION EFFECTIVENESS
   ═══════════════════════════════════════════════════════════════════════════ */
/* ═══════════════════════════════════════════════════════════════════════════
   PROMOTION EFFECTIVENESS
   ═══════════════════════════════════════════════════════════════════════════ */
const promoSkuSel = document.getElementById("promoSkuSelect");
Object.keys(ELAST_DATA.skus).sort((a,b)=>+a-+b).forEach(id => {
  const opt = document.createElement("option");
  opt.value = id;
  opt.textContent = ELAST_DATA.skus[id].label;
  promoSkuSel.appendChild(opt);
});

/** Calculate promotion lift metrics from a SKU's Scan*Pro promo coefficient.
 *  Lift factor = e^β₂; incremental units = Q₀ × (e^β₂ − 1). */
function promoLift(r) {
  const beta = r.promo_coef;
  const liftFactor = Math.exp(beta);
  const liftPct = (liftFactor - 1) * 100;
  const incremental = r.avg_sales * (liftFactor - 1);
  const sig = r.promo_pval < 0.05;
  return { beta, liftFactor, liftPct, incremental, sig, pval: r.promo_pval };
}

/** Render portfolio-level promotion effectiveness KPI cards. */
function renderPromoPortfolio() {
  const skus = Object.values(ELAST_DATA.skus);
  const lifts = skus.map(r => promoLift(r));
  const nSig = lifts.filter(l => l.sig).length;
  const nPos = lifts.filter(l => l.beta > 0).length;
  const avgLiftPct = lifts.reduce((s,l) => s + l.liftPct, 0) / lifts.length;
  const avgIncr = lifts.reduce((s,l) => s + l.incremental, 0) / lifts.length;
  const bestIdx = lifts.reduce((bi,l,i,a) => l.incremental > a[bi].incremental ? i : bi, 0);
  const best = skus[bestIdx];

  document.getElementById("promoPortfolioKpis").innerHTML = [
    kpi("SKUs Analysed", skus.length, "with Scan*Pro promo coefficient"),
    kpi("Significant (p<0.05)", `${nSig} / ${skus.length}`, "reliable promotion effect"),
    kpi("Positive Lift", `${nPos} / ${skus.length}`, "promotions boost demand"),
    kpi("Avg Lift", `${avgLiftPct>=0?"+":""}${avgLiftPct.toFixed(1)}%`, "demand increase during promo"),
    kpi("Avg Incremental", `${avgIncr.toFixed(1)} units/wk`, "additional sales from promo"),
    kpi("Top Responder", `SKU ${best.sku}`, `+${promoLift(best).incremental.toFixed(1)} units/wk`),
  ].join("");
}

/** Render per-SKU promotion detail: header, interpretation, KPI cards. */
function renderPromoSku() {
  const id = promoSkuSel.value;
  const r = ELAST_DATA.skus[id];
  const pl = promoLift(r);

  document.getElementById("promoSkuHeader").innerHTML =
    `<h2>${r.label}</h2>
     <p>Category: ${r.category} · Colour: ${r.color} · Avg price: £${r.avg_price.toFixed(2)} · Avg sales: ${r.avg_sales.toFixed(0)} units/wk · Promo rate: ${r.promo_rate.toFixed(1)}%</p>`;

  const box = document.getElementById("promoSkuInterpBox");
  let interp;
  if (!pl.sig) {
    interp = `Promotion effect is <b>not statistically significant</b> (p = ${pl.pval.toFixed(4)}). The estimated lift of ${pl.liftPct>=0?"+":""}${pl.liftPct.toFixed(1)}% cannot be reliably distinguished from zero.`;
  } else if (pl.beta > 0) {
    interp = `Feature promotions <b>significantly boost demand</b> by <b>${pl.liftPct.toFixed(1)}%</b> (p = ${pl.pval.toFixed(4)}). Each promo week generates ~<b>${pl.incremental.toFixed(1)} extra units</b> above the baseline of ${r.avg_sales.toFixed(0)} units/wk.`;
  } else {
    interp = `Unusual: promotions appear to <b>reduce demand</b> by ${Math.abs(pl.liftPct).toFixed(1)}% (p = ${pl.pval.toFixed(4)}). This may indicate cannibalisation or data quality issues.`;
  }
  box.innerHTML = interp; box.style.display = "block";

  const salesDuringPromo = r.avg_sales * pl.liftFactor;
  document.getElementById("promoSkuKpiRow").innerHTML = [
    kpi("Promo Coefficient (β₂)", pl.beta.toFixed(4), `p = ${pl.pval.toFixed(4)} · ${pl.sig?"★ significant":"not significant"}`),
    kpi("Lift Factor", `×${pl.liftFactor.toFixed(3)}`, `${pl.liftPct>=0?"+":""}${pl.liftPct.toFixed(1)}% demand change`),
    kpi("Baseline Sales", `${r.avg_sales.toFixed(0)} units/wk`, "without promotion"),
    kpi("Promoted Sales", `${salesDuringPromo.toFixed(0)} units/wk`, "estimated during promo"),
    kpi("Incremental Sales", `${pl.incremental>=0?"+":""}${pl.incremental.toFixed(1)} units/wk`, "additional from promotion"),
    kpi("Promo Frequency", `${r.promo_rate.toFixed(1)}%`, "of weeks featured"),
  ].join("");
}

/** Render horizontal bar chart ranking all SKUs by incremental promo sales. */
function renderPromoLiftChart() {
  const skus = Object.values(ELAST_DATA.skus);
  const rows = skus.map(r => ({...r, pl: promoLift(r)})).sort((a,b) => a.pl.incremental - b.pl.incremental);
  const yLabels = rows.map(r => r.label.replace("SKU ","") + (r.pl.sig?" ★":""));
  const xVals = rows.map(r => r.pl.incremental);
  const cols = rows.map(r => r.pl.sig ? (r.pl.incremental>=0?"#0000CD":"#DC2626") : GREY_MD);

  Plotly.newPlot("promoLiftChart",[{
    type:"bar", orientation:"h", y:yLabels, x:xVals,
    marker:{color:cols},
    text:xVals.map(v=>`${v>=0?"+":""}${v.toFixed(1)}`),
    textposition:"outside", textfont:{size:9,color:GREY_MD},
    customdata:rows.map(r=>[r.pl.beta,r.pl.pval,r.pl.liftPct,r.avg_sales,r.promo_rate,r.pl.sig?"Yes":"No"]),
    hovertemplate:"<b>%{y}</b><br>Incremental: <b>%{x:+.1f} units/wk</b><br>β₂: %{customdata[0]:.4f}  p: %{customdata[1]:.4f}  Sig: %{customdata[5]}<br>Lift: %{customdata[2]:+.1f}%  Base sales: %{customdata[3]:.0f}/wk<br>Promo rate: %{customdata[4]:.1f}%<extra></extra>",
  }],{
    title:{text:"Incremental Sales from Feature Promotions (units/wk)",font:{size:13,color:SLATE}},
    height:Math.max(550,rows.length*22), margin:{l:200,r:80,t:40,b:40},
    xaxis:{title:"Incremental units/wk during promo",zeroline:true,zerolinecolor:GREY_LG,zerolinewidth:1,tickfont:{color:GREY_MD}},
    yaxis:{tickfont:{size:9,color:SLATE}},showlegend:false,plot_bgcolor:"white",paper_bgcolor:"white",
    annotations:[
      {x:.99,y:.01,xref:"paper",yref:"paper",text:"★ = significant (p<0.05)  ·  Blue = positive & sig  ·  Red = negative & sig  ·  Grey = not significant",
       showarrow:false,font:{color:GREY_MD,size:9},bgcolor:"white",align:"right"},
    ],
  },plotCfg);
}

/** Render all-SKU promotion summary table with gradient effectiveness column. */
function renderPromoSummaryTable() {
  const skus = Object.values(ELAST_DATA.skus);
  const rows = skus.map(r => ({...r, pl: promoLift(r)})).sort((a,b) => b.pl.incremental - a.pl.incremental);
  const n = rows.length;
  let html = `<table><tr><th>SKU</th><th>Product</th><th>Promo Rate</th><th>β₂ (Coef)</th><th>p-value</th><th>Sig.</th><th>Lift %</th><th>Base Sales</th><th>Incremental</th><th>Effectiveness</th></tr>`;
  rows.forEach((r,i) => {
    const pl = r.pl;
    const eff = !pl.sig ? "Inconclusive" : pl.incremental > 5 ? "Highly effective" : pl.incremental > 0 ? "Moderately effective" : "Ineffective";
    const t = n>1 ? i/(n-1) : 0;
    const eR=Math.round(220-220*t), eG=Math.round(232-232*t), eB=Math.round(255-50*t);
    const eTxt = t>0.55 ? "white" : "#1e293b";
    const liftCls = pl.liftPct>=0 ? "pos" : "neg";
    const incrCls = pl.incremental>=0 ? "pos" : "neg";
    html += `<tr>
      <td class="label-col"><b>SKU ${r.sku}</b></td>
      <td class="label-col" style="font-size:.78rem">${r.label.split("—")[1]?.trim()||""}</td>
      <td>${r.promo_rate.toFixed(1)}%</td>
      <td>${pl.beta.toFixed(4)}</td>
      <td>${pl.pval.toFixed(4)}</td>
      <td>${pl.sig?"★":""}</td>
      <td class="${liftCls}">${pl.liftPct>=0?"+":""}${pl.liftPct.toFixed(1)}%</td>
      <td>${r.avg_sales.toFixed(0)}</td>
      <td class="${incrCls}">${pl.incremental>=0?"+":""}${pl.incremental.toFixed(1)} units/wk</td>
      <td class="label-col" style="font-size:.78rem;background:rgb(${eR},${eG},${eB});color:${eTxt};font-weight:600">${eff}</td></tr>`;
  });
  html += "</table>";
  document.getElementById("promoSummaryTable").innerHTML = html;
}


/* ═══════════════════════════════════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════════════════════════════════ */
// Guide tab is default — demand renders on first switch
</script>
</body>
</html>"""


# ─── Write output ─────────────────────────────────────────────────────────────
test_step = TEST_SIZES[1] - TEST_SIZES[0]
html_out = HTML_TEMPLATE.replace("__DEMAND_DATA__", json.dumps(demand_data))
html_out = html_out.replace("__ELAST_DATA__", json.dumps(elasticity_data))
html_out = html_out.replace("__TEST_MIN__", str(TEST_SIZES[0]))
html_out = html_out.replace("__TEST_MAX__", str(TEST_SIZES[-1]))
html_out = html_out.replace("__TEST_STEP__", str(test_step))

with open("../dashboards/marketing_analytics_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_out)

elapsed = time.time() - t0
print(f"\n{'═' * 60}")
print(f"  Marketing Analytics Dashboard written → dashboards/marketing_analytics_dashboard.html")
print(f"  {len(html_out):,} bytes / {len(html_out) // 1024} KB  ·  {elapsed:.1f}s total")
print(f"{'═' * 60}")
