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

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
MAX_HORIZON = 12
TEST_SIZES = list(range(4, 17, 2))
MODEL_NAMES = ["Linear Regression", "Random Forest", "Neural Network (MLP)"]
SCENARIOS = [-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30]
CURVE_POINTS = 300
MONTH_COLS = [f"month_{i}" for i in range(2, 13)]

t0 = time.time()

# ─── Load data ────────────────────────────────────────────────────────────────
raw = pd.read_csv("data_raw.csv")
proc = pd.read_csv("data_processed.csv")
raw["week"] = pd.to_datetime(raw["week"])
proc["week"] = pd.to_datetime(proc["week"])
raw["feat_main_page"] = raw["feat_main_page"].astype(str).str.lower().eq("true").astype(int)

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
    row = sku_meta[sku_meta["sku"] == sku_id]
    return row.iloc[0] if len(row) > 0 else None


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — DEMAND FORECASTING
# ═════════════════════════════════════════════════════════════════════════════
print("━" * 60)
print("PART 1: Demand Forecasting (3 ML models)")
print("━" * 60)


def get_features_target(proc, sku_id):
    df = proc[proc["sku"] == sku_id].sort_values("week").copy()
    feature_cols = [c for c in df.columns if c not in ["week", "sku", "weekly_sales"]]
    return df[feature_cols].values, df["weekly_sales"].values, df["week"].values, feature_cols


def train_one_config(X, y, weeks, feature_cols, test_size):
    if len(X) < test_size + 10:
        return None

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=4, random_state=42, n_jobs=-1),
        "Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
            validation_fraction=0.15, random_state=42, learning_rate_init=0.005),
    }

    results = {}
    for name, model in models.items():
        use_scaled = "Neural" in name
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s if use_scaled else X_test
        model.fit(Xtr, y_train)
        y_pred_train = np.maximum(model.predict(Xtr), 0)
        y_pred_test = np.maximum(model.predict(Xte), 0)
        resid_std = float(np.std(y_train - y_pred_train))

        last_row = X[-1].copy().reshape(1, -1)
        preds, ci_lo, ci_hi = [], [], []
        for step in range(1, MAX_HORIZON + 1):
            row = last_row.copy()
            if "trend" in feature_cols:
                row[0, feature_cols.index("trend")] = X[-1, feature_cols.index("trend")] + step / len(X)
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
    rf_model = models["Random Forest"]
    imp = rf_model.feature_importances_
    top_idx = np.argsort(imp)[-12:]
    feat_imp = {feature_cols[i]: round(float(imp[i]), 5) for i in top_idx}

    return {"results": results, "best_model": best_model, "feature_importance": feat_imp}


demand_sku_data = {}
heatmap_preds = {}
total_configs = 0

for sku_id in sorted(proc["sku"].unique()):
    X, y, weeks, feature_cols = get_features_target(proc, sku_id)
    sm_row = sku_meta[sku_meta.sku == sku_id].iloc[0]

    week_strs = [pd.Timestamp(w).strftime("%Y-%m-%d") for w in weeks]
    sales_list = [round(float(v), 2) for v in y]

    by_test = {}
    for ts in TEST_SIZES:
        cfg = train_one_config(X, y, weeks, feature_cols, ts)
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

    hm_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    hm_model.fit(X, y)
    hm_row = X[-1].copy().reshape(1, -1)
    hm_preds = []
    for step in range(1, 5):
        r = hm_row.copy()
        if "trend" in feature_cols:
            r[0, feature_cols.index("trend")] = X[-1, feature_cols.index("trend")] + step / len(X)
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
    df = proc[proc["sku"] == sku_id].sort_values("week").copy()
    df = df[df["weekly_sales"] > 0]
    if len(df) < 15 or df["feat_main_page"].nunique() < 2:
        return None

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
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    elast = float(model.params.get("log_price", np.nan))
    elast_pval = float(model.pvalues.get("log_price", np.nan))
    promo_coef = float(model.params.get("feat_main_page", np.nan))
    promo_pval = float(model.pvalues.get("feat_main_page", np.nan))
    if np.isnan(elast):
        return None

    m = meta(sku_id)
    P0 = float(df["price"].mean())
    Q0 = float(df["weekly_sales"].mean())
    R0 = P0 * Q0

    price_range = np.linspace(P0 * 0.20, P0 * 3.0, CURVE_POINTS)
    demand_curve = np.clip(Q0 * (price_range / P0) ** elast, 0, None)
    revenue_curve = price_range * demand_curve
    opt_idx = int(np.argmax(revenue_curve))

    X_full = X.copy()
    fitted = np.exp(model.predict(X_full)).tolist()

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

elasticities = [r["elasticity"] for r in all_elast_data.values()]
n_elastic = sum(1 for e in elasticities if e < -1)
n_inelastic = sum(1 for e in elasticities if -1 <= e < 0)
n_sig = sum(1 for r in all_elast_data.values() if r["elast_sig"])
avg_elast = round(float(np.mean(elasticities)), 3)
most_sens = min(all_elast_data.values(), key=lambda r: r["elasticity"])
least_sens = max(all_elast_data.values(), key=lambda r: r["elasticity"])

heatmap_skus = sorted(all_elast_data.keys())
heatmap_labels = [all_elast_data[s]["label"].replace("SKU ", "") for s in heatmap_skus]
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
    src:url('Imperial Fonts/ImperialSansDisplay-Extralight.woff2') format('woff2'),
        url('Imperial Fonts/ImperialSansDisplay-Extralight.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:300; font-style:normal;
    src:url('Imperial Fonts/ImperialSansDisplay-Light.woff2') format('woff2'),
        url('Imperial Fonts/ImperialSansDisplay-Light.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:400; font-style:normal;
    src:url('Imperial Fonts/ImperialSansDisplay-Regular.woff2') format('woff2'),
        url('Imperial Fonts/ImperialSansDisplay-Regular.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:500; font-style:normal;
    src:url('Imperial Fonts/ImperialSansDisplay-Medium.woff2') format('woff2'),
        url('Imperial Fonts/ImperialSansDisplay-Medium.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:600; font-style:normal;
    src:url('Imperial Fonts/ImperialSansDisplay-Semibold.woff2') format('woff2'),
        url('Imperial Fonts/ImperialSansDisplay-Semibold.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:700; font-style:normal;
    src:url('Imperial Fonts/ImperialSansDisplay-Bold.woff2') format('woff2'),
        url('Imperial Fonts/ImperialSansDisplay-Bold.ttf') format('truetype'); }
  @font-face { font-family:'Imperial Sans Display'; font-weight:800; font-style:normal;
    src:url('Imperial Fonts/ImperialSansDisplay-Extrabold.woff2') format('woff2'),
        url('Imperial Fonts/ImperialSansDisplay-Extrabold.ttf') format('truetype'); }

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
  .tab-panel { display:none; }
  .tab-panel.active { display:block; }

  /* Controls */
  .controls { display:flex; gap:1rem; align-items:center; flex-wrap:wrap;
    margin-bottom:1rem; }
  select { font-family:inherit; font-size:1.05rem; padding:.65rem 1.1rem;
    border-radius:8px; border:1px solid var(--slate-200); background:white;
    min-width:300px; color:var(--slate-700); cursor:pointer; }
  select:focus { outline:none; border-color:#0000CD;
    box-shadow:0 0 0 3px rgba(0,0,205,0.12); }
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
  <button class="tab-btn active" onclick="switchTab('demand')">Demand Forecasting</button>
  <button class="tab-btn" onclick="switchTab('promo')">Promotion Effectiveness</button>
  <button class="tab-btn" onclick="switchTab('elasticity')">Price Elasticity</button>
</div>

<div class="container">

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- TAB 1: DEMAND FORECASTING                                             -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div id="tab-demand" class="tab-panel active">

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
  <div id="demandKpiRow" class="kpi-row"></div>
  <div class="chart-card"><div id="forecastChart"></div></div>

  <h3 class="section-title">Forecast Detail</h3>
  <div id="forecastTable"></div>

  <h3 class="section-title">Model Comparison</h3>
  <div class="chart-card"><div id="compChart"></div></div>
  <div id="metricsTable"></div>

  <h3 class="section-title">Diagnostics</h3>
  <div class="grid-2">
    <div class="chart-card"><div id="residChart"></div></div>
    <div class="chart-card"><div id="featChart"></div></div>
  </div>

  <div style="margin-top:2rem;">
    <h3 class="section-title">All-SKU Forecast Heatmap (Random Forest)</h3>
    <div class="chart-card"><div id="demandHeatmapChart"></div></div>
  </div>

  <div class="methodology">
    <h3>Our Methodology — Demand Forecasting</h3>
    <p><b>Data:</b> 44 SKUs &times; 98 weeks from the processed dataset with lagged prices, trend, month dummies, and one-hot categoricals.</p>
    <p><b>Train/test split:</b> Last N weeks held out (temporal split &mdash; no data leakage).</p>
    <p><b>Models:</b></p>
    <ul>
      <li><b>Linear Regression</b> &mdash; OLS baseline on raw features</li>
      <li><b>Random Forest</b> &mdash; 200 trees, max_depth=8, min_samples_leaf=4</li>
      <li><b>Neural Network (MLP)</b> &mdash; 2 hidden layers (64&rarr;32), scaled inputs, early stopping</li>
    </ul>
    <p><b>Confidence intervals:</b> Training residual &sigma; &times; 1.96 &times; &radic;(step) &mdash; widens with forecast horizon. 95% level.</p>
    <p><b>Best model selection:</b> Lowest MAE on the held-out test set.</p>
    <p style="margin-top:.75rem;font-size:.78rem;color:var(--slate-400);">
      Reference: Cohen, M.C., Gras, P.E., Pentecoste, A., &amp; Zhang, R. (2022).
      <i>Demand Prediction in Retail.</i> Springer SSCM 14.</p>
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
    <p><b>Model:</b> The Scan*Pro log-log OLS model (fitted per SKU) includes a binary
      <code>feat_main_page</code> indicator. The estimated coefficient &beta;&#8322;
      captures the multiplicative lift in demand when a feature promotion is active.</p>
    <p><b>Incremental sales:</b> Since the model is semi-log in the promotion variable,
      the lift factor is <code>e<sup>&beta;&#8322;</sup></code>, meaning weekly sales
      increase by a factor of <code>e<sup>&beta;&#8322;</sup></code> during promotion weeks.
      Incremental units per week = Q&#8320; &times; (e<sup>&beta;&#8322;</sup> &minus; 1).</p>
    <p><b>Significance:</b> A p-value &lt; 0.05 on &beta;&#8322; indicates the promotion
      effect is statistically distinguishable from zero at the 95% confidence level.</p>
    <p style="margin-top:.75rem;font-size:.78rem;color:var(--slate-400);">
      Reference: Van Heerde, H.J., Leeflang, P.S.H., &amp; Wittink, D.R. (2004).
      <i>Decomposing the Sales Promotion Bump.</i> Marketing Science, 23(3).</p>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- TAB 3: PRICE ELASTICITY                                               -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div id="tab-elasticity" class="tab-panel">

  <div class="controls">
    <select id="elastSkuSelect" onchange="renderElasticity()"></select>
  </div>

  <div id="elastHeader" class="sku-header"></div>
  <div id="elastInterpBox" class="interp" style="display:none"></div>
  <div id="elastKpiRow" class="kpi-row"></div>

  <div class="grid-2">
    <div class="chart-card"><div id="demandCurveChart"></div></div>
    <div class="chart-card"><div id="revenueCurveChart"></div></div>
  </div>

  <h3 class="section-title">Scan*Pro Model Fit &mdash; Actual vs Fitted</h3>
  <div class="chart-card"><div id="fitChart"></div></div>

  <h3 class="section-title">Revenue Impact by Price Scenario</h3>
  <div class="chart-card"><div id="waterfallChart"></div></div>

  <h3 class="section-title">What-If Scenario Table</h3>
  <div id="scenarioTable"></div>

  <div style="margin-top:2.5rem;">
    <h3 class="section-title">Portfolio Overview &mdash; All SKUs</h3>
    <div id="portfolioKpis" class="kpi-row"></div>

    <h3 class="section-title">Elasticity Ranking</h3>
    <div class="chart-card"><div id="rankingChart"></div></div>

    <h3 class="section-title">Revenue Impact Heatmap &mdash; All SKUs &times; All Scenarios</h3>
    <div class="chart-card"><div id="elastHeatmapChart"></div></div>

    <h3 class="section-title">All-SKU Price Strategy Summary</h3>
    <div id="strategyTable"></div>
  </div>

  <div class="methodology">
    <h3>Our Methodology — Price Elasticity</h3>
    <p><b>Data:</b> 44 SKUs &times; 98 weeks from <code>data_processed.csv</code>
      (lagged prices, trend, month dummies, one-hot categoricals).
      SKU names from <code>data_raw.csv</code>.</p>
    <p><b>Model:</b> Scan*Pro log-log OLS fitted per SKU:</p>
    <p style="margin:.4rem 0 .4rem 1.2rem;font-family:monospace;font-size:.82rem;">
      log(sales) = &beta;&#8320; + &beta;&#8321;&middot;log(price) + &beta;&#8322;&middot;feat_main_page + &beta;&#8323;&middot;trend + &Sigma;&gamma;&#8344;&middot;month&#8344; + &epsilon;</p>
    <p>The price coefficient <b>&beta;&#8321; is the price elasticity directly</b>.</p>
    <p><b>Scenario simulation:</b>
      Q_new = Q&#8320; &times; (P_new/P&#8320;)^&epsilon; &nbsp;&middot;&nbsp; R_new = P_new &times; Q_new</p>
    <ul style="margin:.4rem 0 0 1.2rem;">
      <li><b>Elastic (|&epsilon;| &gt; 1):</b> Lowering price grows revenue</li>
      <li><b>Inelastic (|&epsilon;| &lt; 1):</b> Raising price grows revenue</li>
    </ul>
    <p style="margin-top:.75rem;font-size:.78rem;color:var(--slate-400);">
      Reference: Cohen, M.C., Gras, P.E., Pentecoste, A., &amp; Zhang, R. (2022).
      <i>Demand Prediction in Retail.</i> Springer SSCM 14.</p>
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
const DM_COLORS = {"Linear Regression":"#7C3AED","Random Forest":"#E11D48","Neural Network (MLP)":"#EA580C"};
const DM_MODEL_NAMES = ["Linear Regression","Random Forest","Neural Network (MLP)"];
const BLUE   = "#2563EB", SLATE = "#0f172a", GREY_LG = "#cbd5e1",
      GREY_MD = "#94a3b8", RED = "#DC2626", GREEN = "#16a34a";

/* ═══════════════════════════════════════════════════════════════════════════
   TAB SWITCHING
   ═══════════════════════════════════════════════════════════════════════════ */
let activeTab = "demand";
let demandRendered = true, promoRendered = false, elastRendered = false;

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
    renderElasticity();
    renderPortfolioKpis();
    renderRanking();
    renderElastHeatmap();
    renderStrategyTable();
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
function kpi(label, value, sub) {
  return `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-sub">${sub}</div></div>`;
}

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
  document.getElementById("demandKpiRow").innerHTML = [
    kpi("Best Model", best.split("(")[0].trim(), "Lowest MAE on test"),
    kpi("Test MAE", bRes.mae.toFixed(1), "units/week"),
    kpi("Test R²", bRes.r2.toFixed(3), r2q),
    kpi(`Avg Forecast (${horizon}w)`, Math.round(avgFc), `${sym} ${Math.abs(delta).toFixed(1)}% vs recent`),
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

  /* Comparison bars */
  const metrics=["mae","rmse","r2"], titles=["MAE ↓","RMSE ↓","R² ↑"];
  const compTraces=metrics.map((met,i)=>({
    x:DM_MODEL_NAMES,y:DM_MODEL_NAMES.map(n=>cfg.results[n][met]),
    type:"bar",marker:{color:DM_MODEL_NAMES.map(n=>DM_COLORS[n])},showlegend:false,
    text:DM_MODEL_NAMES.map(n=>cfg.results[n][met].toFixed(2)),textposition:"outside",
    textfont:{size:11,family:"monospace"},xaxis:`x${i+1}`,yaxis:`y${i+1}`,
  }));
  Plotly.newPlot("compChart",compTraces,{
    height:380,margin:{l:55,r:20,t:45,b:40},grid:{rows:1,columns:3,pattern:"independent"},
    annotations:titles.map((t,i)=>({text:`<b>${t}</b>`,x:(i+0.5)/3,y:1.08,xref:"paper",yref:"paper",showarrow:false,font:{size:12}})),
  },plotCfg);

  /* Metrics table */
  let mt="<table><tr><th class='lt'>Model</th><th class='lt'>MAE</th><th class='lt'>RMSE</th><th class='lt'>MAPE %</th><th class='lt'>R²</th><th class='lt'></th></tr>";
  DM_MODEL_NAMES.forEach(n=>{const r=cfg.results[n];const cls=n===best?' class="best"':'';
    mt+=`<tr${cls}><td style="text-align:left">${n}</td><td>${r.mae.toFixed(2)}</td><td>${r.rmse.toFixed(2)}</td><td>${r.mape.toFixed(1)}</td><td>${r.r2.toFixed(3)}</td><td>${n===best?"★":""}</td></tr>`;});
  mt+="</table>";
  document.getElementById("metricsTable").innerHTML=mt;

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
   PRICE ELASTICITY
   ═══════════════════════════════════════════════════════════════════════════ */
const elastSel = document.getElementById("elastSkuSelect");
Object.keys(ELAST_DATA.skus).sort((a,b)=>+a-+b).forEach(id => {
  const opt = document.createElement("option");
  opt.value = id;
  opt.textContent = ELAST_DATA.skus[id].label;
  elastSel.appendChild(opt);
});

function renderElasticity() {
  const id = elastSel.value;
  const r  = ELAST_DATA.skus[id];
  const e  = r.elasticity;
  const P0 = r.avg_price, Q0 = r.avg_sales, R0 = r.avg_revenue;

  document.getElementById("elastHeader").innerHTML =
    `<h2>${r.label}</h2>
     <p>Category: ${r.category} · Colour: ${r.color} · Avg price: £${P0.toFixed(2)} · Avg sales: ${Q0.toFixed(0)} units/wk · Avg revenue: £${R0.toLocaleString("en-GB",{maximumFractionDigits:0})}/wk · Promo rate: ${r.promo_rate}%</p>`;

  let interp;
  if(e<-1) interp=`Elastic — |ε| = ${Math.abs(e).toFixed(2)} > 1. Demand is price-sensitive. A 10% price rise causes ~${(Math.abs(e)*10).toFixed(0)}% demand fall. <b>Lowering price is likely to grow revenue.</b>`;
  else if(e<0) interp=`Inelastic — |ε| = ${Math.abs(e).toFixed(2)} < 1. Demand is price-insensitive. A 10% price rise causes only ~${(Math.abs(e)*10).toFixed(0)}% demand fall. <b>Raising price is likely to grow revenue.</b>`;
  else interp=`Unusual positive elasticity (ε = ${e.toFixed(2)}). Check model significance (p = ${r.elast_pval.toFixed(4)}).`;
  const box=document.getElementById("elastInterpBox");
  box.innerHTML=interp; box.style.display="block";

  const sigStr=r.elast_sig?"★ significant":"not significant";
  document.getElementById("elastKpiRow").innerHTML=[
    kpi("Price elasticity",e.toFixed(4),`p = ${r.elast_pval.toFixed(4)} · ${sigStr}`),
    kpi("Model R²",r.r2.toFixed(4),"Scan*Pro goodness of fit"),
    kpi("Avg price",`£${P0.toFixed(2)}`,`Avg sales: ${Q0.toFixed(0)} units/wk`),
    kpi("Avg revenue",`£${R0.toLocaleString("en-GB",{maximumFractionDigits:0})}`,"per week at current price"),
    kpi("Revenue-max price",`£${r.optimal_price.toFixed(2)}`,`Peak £${r.optimal_revenue.toLocaleString("en-GB",{maximumFractionDigits:0})}/wk`),
  ].join("");

  /* Demand curve */
  Plotly.newPlot("demandCurveChart",[
    {x:r.price_range,y:r.demand_curve,mode:"lines",line:{color:BLUE,width:2},name:"Demand",
     hovertemplate:"Price: £%{x:.2f}<br>Demand: %{y:.0f} units<extra></extra>"},
    {x:[P0],y:[Q0],mode:"markers",marker:{size:11,color:SLATE,line:{color:"white",width:2}},
     name:"Current price",hovertemplate:`Current: £${P0.toFixed(2)} → ${Q0.toFixed(0)} units<extra></extra>`},
  ],{
    title:{text:`Demand Curve  (ε = ${e.toFixed(3)})`,font:{size:13,color:SLATE}},
    xaxis:{title:"Price (£)",tickfont:{color:GREY_MD}},yaxis:{title:"Weekly demand (units)",tickfont:{color:GREY_MD}},
    legend:{orientation:"h",y:-0.22},plot_bgcolor:"white",paper_bgcolor:"white",
    margin:{t:50,b:50,l:60,r:20},height:340,
  },plotCfg);

  /* Revenue curve */
  Plotly.newPlot("revenueCurveChart",[
    {x:r.price_range,y:r.revenue_curve,mode:"lines",line:{color:GREY_LG,width:2},name:"Revenue",
     hovertemplate:"Price: £%{x:.2f}<br>Revenue: £%{y:,.0f}<extra></extra>"},
    {x:[P0],y:[R0],mode:"markers",marker:{size:11,color:SLATE,line:{color:"white",width:2}},name:"Current"},
    {x:[r.optimal_price],y:[r.optimal_revenue],mode:"markers",
     marker:{size:14,color:BLUE,symbol:"star",line:{color:"white",width:1.5}},
     name:`Optimal £${r.optimal_price.toFixed(2)}`},
  ],{
    title:{text:`Revenue Curve — Optimal: £${r.optimal_price.toFixed(2)}`,font:{size:13,color:SLATE}},
    xaxis:{title:"Price (£)",tickfont:{color:GREY_MD}},yaxis:{title:"Weekly revenue (£)",tickfont:{color:GREY_MD}},
    legend:{orientation:"h",y:-0.22},plot_bgcolor:"white",paper_bgcolor:"white",
    margin:{t:50,b:50,l:60,r:20},height:340,
  },plotCfg);

  /* Scan*Pro fit */
  Plotly.newPlot("fitChart",[
    {x:r.weeks,y:r.actual,mode:"lines",name:"Actual sales",line:{color:SLATE,width:1.8}},
    {x:r.weeks,y:r.fitted,mode:"lines",name:"Scan*Pro fitted",line:{color:BLUE,width:1.5,dash:"dash"}},
  ],{
    title:{text:`Scan*Pro Model Fit  (R² = ${r.r2.toFixed(3)})`,font:{size:13,color:SLATE}},
    xaxis:{title:"Week",type:"date",tickformat:"%b %Y",tickfont:{color:GREY_MD}},
    yaxis:{title:"Weekly sales (units)",tickfont:{color:GREY_MD}},
    legend:{orientation:"h",y:-0.2},plot_bgcolor:"white",paper_bgcolor:"white",
    height:320,margin:{t:50,b:60,l:60,r:20},
  },plotCfg);

  /* Waterfall */
  const sc=r.scenarios; const yVals=sc.map(s=>s.d_rev);
  Plotly.newPlot("waterfallChart",[{type:"bar",x:sc.map(s=>s.label),y:yVals,
    marker:{color:yVals.map(v=>v<0?RED:GREEN),line:{width:0}},
    text:yVals.map(v=>`£${v>=0?"+":""}${v.toLocaleString("en-GB",{maximumFractionDigits:0})}`),
    textposition:"outside",textfont:{size:10,color:GREY_MD},
    customdata:sc.map(s=>[s.new_price,s.d_demand_pct,s.new_rev,s.d_rev_pct]),
    hovertemplate:"<b>%{x}</b><br>New price: £%{customdata[0]:.2f}<br>Δ Demand: %{customdata[1]:+.1f}%<br>New revenue: £%{customdata[2]:,.0f}<br>Δ Revenue: %{customdata[3]:+.1f}%<extra></extra>",
  }],{
    title:{text:`Revenue Change by Price Scenario  (ε = ${e.toFixed(3)})`,font:{size:13,color:SLATE}},
    xaxis:{title:"Price scenario",tickfont:{color:GREY_MD}},
    yaxis:{title:"Δ Revenue vs baseline (£/week)",tickfont:{color:GREY_MD}},
    shapes:[{type:"line",x0:0,x1:1,y0:0,y1:0,xref:"paper",yref:"y",line:{color:GREY_LG,width:1}}],
    plot_bgcolor:"white",paper_bgcolor:"white",height:340,margin:{t:50,b:50,l:60,r:20},
  },plotCfg);

  /* Scenario table */
  let stHtml=`<table><tr><th>Scenario</th><th>New Price</th><th>Δ Demand (units)</th><th>Δ Demand (%)</th><th>New Revenue</th><th>Δ Revenue (£)</th><th>Δ Revenue (%)</th></tr>`;
  sc.forEach(s=>{
    const rowCls=s.pct>0?"row-dn":"row-up";
    const dc=s.d_demand>=0?"pos":"neg"; const rc=s.d_rev>=0?"pos":"neg";
    stHtml+=`<tr class="${rowCls}"><td><b>${s.label}</b></td><td>£${s.new_price.toFixed(2)}</td>
      <td class="${dc}">${s.d_demand>=0?"+":""}${s.d_demand}</td>
      <td class="${dc}">${s.d_demand_pct>=0?"+":""}${s.d_demand_pct}%</td>
      <td>£${s.new_rev.toLocaleString("en-GB",{maximumFractionDigits:0})}</td>
      <td class="${rc}">${s.d_rev>=0?"£+":"£"}${s.d_rev.toLocaleString("en-GB",{maximumFractionDigits:0})}</td>
      <td class="${rc}">${s.d_rev_pct>=0?"+":""}${s.d_rev_pct}%</td></tr>`;
  });
  stHtml+="</table>";
  document.getElementById("scenarioTable").innerHTML=stHtml;
}

/* Portfolio KPIs */
function renderPortfolioKpis() {
  const p=ELAST_DATA.portfolio;
  document.getElementById("portfolioKpis").innerHTML=[
    kpi("SKUs analysed",p.n_skus,"with sufficient data"),
    kpi("Elastic |ε|>1",`${p.n_elastic} / ${p.n_skus}`,"price-sensitive demand"),
    kpi("Inelastic |ε|<1",`${p.n_inelastic} / ${p.n_skus}`,"price-insensitive demand"),
    kpi("Significant p<0.05",`${p.n_sig} / ${p.n_skus}`,"statistically reliable ε"),
    kpi("Avg elasticity",p.avg_elast,"across all SKUs"),
    kpi("Most sensitive",`SKU ${p.most_sens.sku}`,`ε = ${p.most_sens.elasticity.toFixed(3)}`),
    kpi("Least sensitive",`SKU ${p.least_sens.sku}`,`ε = ${p.least_sens.elasticity.toFixed(3)}`),
  ].join("");
}

/* Ranking chart */
function renderRanking() {
  const rows=Object.values(ELAST_DATA.skus).sort((a,b)=>a.elasticity-b.elasticity);
  const yLbls=rows.map(r=>r.label.replace("SKU ","")+( r.elast_sig?" ★":""));
  const xVals=rows.map(r=>r.elasticity);
  const cols=rows.map(r=>r.elast_sig?BLUE:GREY_MD);
  Plotly.newPlot("rankingChart",[{type:"bar",orientation:"h",y:yLbls,x:xVals,marker:{color:cols},
    text:xVals.map(v=>v.toFixed(2)),textposition:"outside",textfont:{size:9,color:GREY_MD},
    customdata:rows.map(r=>[r.elast_pval,r.avg_price,r.avg_sales,r.r2,r.elast_sig?"Yes":"No"]),
    hovertemplate:"<b>%{y}</b><br>ε = <b>%{x:.4f}</b><br>p-value: %{customdata[0]:.4f}  Sig: %{customdata[4]}<br>Avg price: £%{customdata[1]:.2f}  Avg sales: %{customdata[2]:.0f}/wk<br>R²: %{customdata[3]:.3f}<extra></extra>",
  }],{
    height:1000,margin:{l:10,r:80,t:30,b:30},
    xaxis:{title:"Price Elasticity (ε)",zeroline:true,zerolinecolor:GREY_LG,zerolinewidth:1,tickfont:{color:GREY_MD}},
    yaxis:{tickfont:{size:10,color:SLATE}},showlegend:false,plot_bgcolor:"white",paper_bgcolor:"white",
    shapes:[{type:"line",x0:-1,x1:-1,y0:0,y1:1,xref:"x",yref:"paper",line:{dash:"dash",color:GREY_MD,width:1.2}}],
    annotations:[
      {x:-1,y:1,xref:"x",yref:"paper",text:"Unit elastic ε=−1",showarrow:false,font:{color:GREY_MD,size:10},yanchor:"bottom",xanchor:"left"},
      {x:.99,y:.01,xref:"paper",yref:"paper",text:"★ = significant (p<0.05)  ·  Blue = significant  ·  Grey = not significant",showarrow:false,font:{color:GREY_MD,size:9},bgcolor:"white",align:"right"},
    ],
  },plotCfg);
}

/* Elasticity heatmap */
function renderElastHeatmap() {
  const hm=ELAST_DATA.heatmap;
  Plotly.newPlot("elastHeatmapChart",[{type:"heatmap",z:hm.z,x:hm.scenario_labels,y:hm.sku_labels,
    text:hm.text,texttemplate:"%{text}",textfont:{size:8,color:"#334155"},
    colorscale:[[0,RED],[0.4,"#FCA5A5"],[0.5,"#F8FAFC"],[0.6,"#86EFAC"],[1,GREEN]],zmid:0,
    colorbar:{title:"ΔRevenue (%)",ticksuffix:"%",len:.7,tickfont:{size:10}},
    hovertemplate:"<b>%{y}</b><br>Scenario: %{x}<br>Revenue change: <b>%{z:+.1f}%</b><extra></extra>",
  }],{
    height:1050,margin:{l:10,r:110,t:30,b:60},
    xaxis:{title:"Price Change Scenario",side:"top",tickfont:{size:11}},
    yaxis:{tickfont:{size:9},autorange:"reversed"},paper_bgcolor:"white",
  },plotCfg);
}

/* Strategy table */
function renderStrategyTable() {
  const rows=Object.values(ELAST_DATA.skus).sort((a,b)=>a.elasticity-b.elasticity);
  const n=rows.length;
  let html=`<table><tr><th>SKU</th><th>Product</th><th>Elasticity</th><th>Sig.</th><th>Avg Price</th><th>ΔRev +10%</th><th>ΔRev −10%</th><th>Optimal Price</th><th>Strategy</th></tr>`;
  rows.forEach((r,i)=>{
    const e=r.elasticity,P0=r.avg_price,Q0=r.avg_sales,R0=r.avg_revenue;
    const Pup=P0*1.10,Pdn=P0*0.90;
    const Rup=Pup*Math.max(Q0*Math.pow(Pup/P0,e),0);
    const Rdn=Pdn*Math.max(Q0*Math.pow(Pdn/P0,e),0);
    const dUp=R0>0?((Rup-R0)/R0*100).toFixed(1):"0.0";
    const dDn=R0>0?((Rdn-R0)/R0*100).toFixed(1):"0.0";
    const strat=e<-1?"Consider lower price":e<0?"Consider higher price":"Review data";
    const eCol=r.elast_sig?"sig":"";
    const t=n>1?i/(n-1):0;
    const sR=Math.round(220-220*t), sG=Math.round(232-232*t), sB=Math.round(255-50*t);
    const sTxt=t>0.55?"white":"#1e293b";
    html+=`<tr><td class="label-col"><b>SKU ${r.sku}</b></td>
      <td class="label-col" style="font-size:.78rem">${r.label.split("—")[1]?.trim()||""}</td>
      <td class="${eCol}">${e.toFixed(4)}</td><td>${r.elast_sig?"★":""}</td>
      <td>£${P0.toFixed(2)}</td>
      <td class="${parseFloat(dUp)>=0?'pos':'neg'}">${parseFloat(dUp)>=0?'+':''}${dUp}%</td>
      <td class="${parseFloat(dDn)>=0?'pos':'neg'}">${parseFloat(dDn)>=0?'+':''}${dDn}%</td>
      <td>£${r.optimal_price.toFixed(2)}</td>
      <td class="label-col" style="font-size:.78rem;background:rgb(${sR},${sG},${sB});color:${sTxt};font-weight:600">${strat}</td></tr>`;
  });
  html+="</table>";
  document.getElementById("strategyTable").innerHTML=html;
}


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

function promoLift(r) {
  const beta = r.promo_coef;
  const liftFactor = Math.exp(beta);
  const liftPct = (liftFactor - 1) * 100;
  const incremental = r.avg_sales * (liftFactor - 1);
  const sig = r.promo_pval < 0.05;
  return { beta, liftFactor, liftPct, incremental, sig, pval: r.promo_pval };
}

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
    height:Math.max(550,rows.length*22), margin:{l:10,r:80,t:40,b:40},
    xaxis:{title:"Incremental units/wk during promo",zeroline:true,zerolinecolor:GREY_LG,zerolinewidth:1,tickfont:{color:GREY_MD}},
    yaxis:{tickfont:{size:10,color:SLATE}},showlegend:false,plot_bgcolor:"white",paper_bgcolor:"white",
    annotations:[
      {x:.99,y:.01,xref:"paper",yref:"paper",text:"★ = significant (p<0.05)  ·  Blue = positive & sig  ·  Red = negative & sig  ·  Grey = not significant",
       showarrow:false,font:{color:GREY_MD,size:9},bgcolor:"white",align:"right"},
    ],
  },plotCfg);
}

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
renderDemand();
renderDemandHeatmap();
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

with open("marketing_analytics_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_out)

elapsed = time.time() - t0
print(f"\n{'═' * 60}")
print(f"  Marketing Analytics Dashboard written → marketing_analytics_dashboard.html")
print(f"  {len(html_out):,} bytes / {len(html_out) // 1024} KB  ·  {elapsed:.1f}s total")
print(f"{'═' * 60}")
