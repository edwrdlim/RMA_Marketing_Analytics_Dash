"""
Generate a self-contained interactive HTML demand forecasting dashboard.
Pre-computes all SKU × test_size combinations so the HTML sliders work live.
Usage:  python generate_demand_dashboard.py
Output: demand_forecasting_dashboard.html
"""

import json, time, warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

MAX_HORIZON = 12
TEST_SIZES = list(range(4, 17, 2))  # [4, 6, 8, 10, 12, 14, 16]
MODEL_NAMES = ["Linear Regression", "Random Forest", "Neural Network (MLP)"]

t0 = time.time()

raw = pd.read_csv("../data/data_raw.csv")
proc = pd.read_csv("../data/data_processed.csv")
raw["week"] = pd.to_datetime(raw["week"])
proc["week"] = pd.to_datetime(proc["week"])
raw["feat_main_page"] = raw["feat_main_page"].astype(str).str.lower().eq("true").astype(int)

sku_meta = raw.groupby("sku").agg(
    functionality=("functionality", "first"),
    color=("color", "first"),
    avg_price=("price", "mean"),
    avg_sales=("weekly_sales", "mean"),
    feat_rate=("feat_main_page", "mean"),
).reset_index()
sku_meta["avg_price"] = sku_meta["avg_price"].round(2)
sku_meta["avg_sales"] = sku_meta["avg_sales"].round(1)
sku_meta["feat_rate"] = (sku_meta["feat_rate"] * 100).round(1)


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


all_sku_data = {}
heatmap_preds = {}
total_configs = 0

for sku_id in sorted(proc["sku"].unique()):
    X, y, weeks, feature_cols = get_features_target(proc, sku_id)
    sm = sku_meta[sku_meta.sku == sku_id].iloc[0]

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

    all_sku_data[int(sku_id)] = {
        "meta": {
            "functionality": sm.functionality,
            "color": sm.color,
            "avg_price": float(sm.avg_price),
            "avg_sales": float(sm.avg_sales),
            "feat_rate": float(sm.feat_rate),
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

heatmap_week_labels = [
    (pd.Timestamp(proc["week"].max()) + pd.Timedelta(weeks=i)).strftime("%d %b")
    for i in range(1, 5)
]

dashboard_data = {
    "skus": all_sku_data,
    "heatmap": heatmap_preds,
    "heatmap_weeks": heatmap_week_labels,
    "test_sizes": TEST_SIZES,
    "max_horizon": MAX_HORIZON,
}

elapsed = time.time() - t0
print(f"Computed {len(all_sku_data)} SKUs × {len(TEST_SIZES)} test sizes = {total_configs} configs in {elapsed:.1f}s")

# ── HTML template ──

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Demand Forecasting Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root { --slate-50:#f8fafc; --slate-100:#f1f5f9; --slate-200:#e2e8f0; --slate-300:#cbd5e1;
    --slate-400:#94a3b8; --slate-500:#64748b; --slate-600:#475569; --slate-700:#334155;
    --slate-800:#1e293b; --slate-900:#0f172a; --indigo:#6366f1; --green:#059669; --amber:#f59e0b; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    background:var(--slate-50); color:var(--slate-900); line-height:1.5; }
  .container { max-width:1200px; margin:0 auto; padding:1.5rem; }

  header { background:white; border-bottom:1px solid var(--slate-200); padding:1.2rem 0; margin-bottom:1.5rem; }
  header .container { display:flex; flex-direction:column; gap:1rem; }
  .header-top { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.75rem; }
  h1 { font-size:1.5rem; font-weight:700; }

  .controls { display:flex; gap:1rem; align-items:center; flex-wrap:wrap; }
  select { font-family:inherit; font-size:0.85rem; padding:0.5rem 0.9rem; border-radius:8px;
    border:1px solid var(--slate-200); background:white; min-width:260px; color:var(--slate-700); cursor:pointer; }
  select:focus { outline:none; border-color:var(--indigo); box-shadow:0 0 0 3px rgba(99,102,241,0.15); }
  button { font-family:inherit; font-size:0.85rem; padding:0.5rem 1.2rem; border-radius:8px; border:none;
    background:var(--green); color:white; font-weight:600; cursor:pointer; transition:background .15s; }
  button:hover { background:#047857; }

  .slider-group { display:flex; align-items:center; gap:0.4rem; }
  .slider-group label { font-size:0.8rem; color:var(--slate-600); white-space:nowrap; }
  .slider-group input[type=range] { width:110px; accent-color:var(--indigo); cursor:pointer; }
  .slider-group .slider-val { font-size:0.85rem; font-weight:700; color:var(--slate-800);
    min-width:28px; text-align:center; font-family:'SF Mono',SFMono-Regular,Consolas,monospace; }

  .kpi-row { display:flex; gap:0.75rem; margin:1rem 0 1.5rem; flex-wrap:wrap; }
  .kpi { flex:1; min-width:155px; background:white; border:1px solid var(--slate-200); border-radius:12px; padding:1rem 1.2rem; }
  .kpi-label { font-size:0.7rem; color:var(--slate-500); text-transform:uppercase; letter-spacing:0.05em; font-weight:500; }
  .kpi-value { font-size:1.35rem; font-weight:700; margin-top:0.15rem; font-family:'SF Mono',SFMono-Regular,Consolas,monospace; }
  .kpi-sub { font-size:0.75rem; color:var(--slate-500); margin-top:0.1rem; }

  .sku-header { margin-bottom:0.5rem; }
  .sku-header h2 { font-size:1.15rem; }
  .sku-header p { font-size:0.85rem; color:var(--slate-500); }

  .chart-card { background:white; border:1px solid var(--slate-200); border-radius:12px; margin-bottom:1rem; overflow:hidden; }
  .section-title { font-size:1rem; font-weight:600; margin:1.5rem 0 0.5rem; color:var(--slate-700); }

  table { width:100%; border-collapse:collapse; font-size:0.82rem; background:white;
    border:1px solid var(--slate-200); border-radius:12px; overflow:hidden; margin-bottom:1rem; }
  th { background:var(--slate-50); font-weight:600; color:var(--slate-600); text-transform:uppercase;
    font-size:0.7rem; letter-spacing:0.04em; }
  th, td { padding:0.6rem 0.9rem; text-align:left; border-bottom:1px solid var(--slate-100); }
  td { font-family:'SF Mono',SFMono-Regular,Consolas,monospace; font-size:0.8rem; }
  tr.best td { background:#f0fdf4; font-weight:600; }

  .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
  @media (max-width:768px) { .grid-2 { grid-template-columns:1fr; } .controls { flex-direction:column; align-items:stretch; } }
  .heatmap-section { margin-top:2rem; }
  .methodology { background:white; border:1px solid var(--slate-200); border-radius:12px; padding:1.5rem; margin-top:1.5rem;
    font-size:0.85rem; color:var(--slate-600); line-height:1.7; }
  .methodology h3 { color:var(--slate-800); margin-bottom:0.5rem; }
  .methodology ul { padding-left:1.2rem; }
  footer { text-align:center; padding:2rem 0; font-size:0.78rem; color:var(--slate-400); }
</style>
</head>
<body>

<header>
  <div class="container">
    <div class="header-top">
      <h1>📈 Demand Forecasting Dashboard</h1>
      <button onclick="renderSKU()">▶ Run Forecast</button>
    </div>
    <div class="controls">
      <select id="skuSelect"></select>
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
    </div>
  </div>
</header>

<div class="container">
  <div id="skuHeader" class="sku-header"></div>
  <div id="kpiRow" class="kpi-row"></div>
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

  <div class="heatmap-section">
    <h3 class="section-title">All-SKU Forecast Heatmap (Random Forest)</h3>
    <div class="chart-card"><div id="heatmapChart"></div></div>
  </div>

  <div class="methodology">
    <h3>Methodology</h3>
    <p><b>Data:</b> 44 SKUs × 98 weeks from the processed dataset with lagged prices, trend, month dummies, and one-hot categoricals.</p>
    <p><b>Train/test split:</b> Last N weeks held out (temporal split — no data leakage).</p>
    <p><b>Models:</b></p>
    <ul>
      <li><b>Linear Regression</b> — OLS baseline on raw features</li>
      <li><b>Random Forest</b> — 200 trees, max_depth=8, min_samples_leaf=4</li>
      <li><b>Neural Network (MLP)</b> — 2 hidden layers (64→32), scaled inputs, early stopping</li>
    </ul>
    <p><b>Confidence intervals:</b> Training residual σ × 1.96 × √(step) — widens with forecast horizon. 95% level.</p>
    <p><b>Best model selection:</b> Lowest MAE on the held-out test set.</p>
    <p style="margin-top:0.75rem;font-size:0.78rem;color:var(--slate-400);">
      Reference: Cohen, M.C., Gras, P.E., Pentecoste, A., &amp; Zhang, R. (2022).
      <i>Demand Prediction in Retail.</i> Springer SSCM 14.</p>
  </div>
</div>

<footer>Generated from demand_forecasting.ipynb · Models pre-computed with scikit-learn</footer>

<script>
const DATA = __DATA_PLACEHOLDER__;
const COLORS = {"Linear Regression":"#6366f1","Random Forest":"#059669","Neural Network (MLP)":"#f59e0b"};
const MODEL_NAMES = ["Linear Regression","Random Forest","Neural Network (MLP)"];
const plotCfg = {responsive:true, displayModeBar:false};

const sel = document.getElementById("skuSelect");
const horizonSlider = document.getElementById("horizonSlider");
const horizonVal = document.getElementById("horizonVal");
const testSlider = document.getElementById("testSlider");
const testVal = document.getElementById("testVal");

horizonSlider.addEventListener("input", () => { horizonVal.textContent = horizonSlider.value; });
testSlider.addEventListener("input", () => { testVal.textContent = testSlider.value; });

Object.keys(DATA.skus).sort((a,b)=>+a - +b).forEach(id => {
  const opt = document.createElement("option");
  opt.value = id;
  opt.textContent = `SKU ${id} — ${DATA.skus[id].meta.functionality}`;
  sel.appendChild(opt);
});

function addDays(dateStr, days) {
  const d = new Date(dateStr);
  d.setDate(d.getDate() + days);
  return d.toISOString().split("T")[0];
}

function renderSKU() {
  const id = sel.value;
  const s = DATA.skus[id];
  const m = s.meta;
  const testSize = parseInt(testSlider.value);
  const horizon = parseInt(horizonSlider.value);

  const cfg = s.by_test[String(testSize)];
  if (!cfg) {
    document.getElementById("skuHeader").innerHTML =
      `<h2>SKU ${id} — ${m.functionality}</h2><p style="color:#ef4444;">⚠ Insufficient data for test size ${testSize}. Try a smaller test size.</p>`;
    return;
  }

  const nWeeks = s.weeks.length;
  const wTrain = s.weeks.slice(0, nWeeks - testSize);
  const wTest  = s.weeks.slice(nWeeks - testSize);
  const yTrain = s.sales.slice(0, nWeeks - testSize);
  const yTest  = s.sales.slice(nWeeks - testSize);

  const best = cfg.best_model;
  const bRes = cfg.results[best];

  // Future weeks from last test week
  const lastTestDate = wTest[wTest.length - 1];
  const futureWeeks = [];
  for (let i = 1; i <= horizon; i++) futureWeeks.push(addDays(lastTestDate, i * 7));

  // Header
  document.getElementById("skuHeader").innerHTML =
    `<h2>SKU ${id} — ${m.functionality}</h2>
     <p>Color: ${m.color} · Avg price: $${m.avg_price} · Avg sales: ${m.avg_sales}/wk · Promo rate: ${m.feat_rate}%</p>`;

  // KPIs
  const avgHist = s.sales.slice(-12).reduce((a,b)=>a+b,0) / 12;
  const fcSlice = bRes.forecast.slice(0, horizon);
  const avgFc = fcSlice.reduce((a,b)=>a+b,0) / horizon;
  const delta = ((avgFc - avgHist) / Math.max(avgHist, 1)) * 100;
  const sym = delta >= 0 ? "▲" : "▼";
  const r2q = bRes.r2 > 0.5 ? "Good ✓" : "Moderate";
  document.getElementById("kpiRow").innerHTML = [
    kpi("Best Model", best.split("(")[0].trim(), "Lowest MAE on test"),
    kpi("Test MAE", bRes.mae.toFixed(1), "units/week"),
    kpi("Test R²", bRes.r2.toFixed(3), r2q),
    kpi(`Avg Forecast (${horizon}w)`, Math.round(avgFc), `${sym} ${Math.abs(delta).toFixed(1)}% vs recent`),
  ].join("");

  // ── Forecast chart ──
  const traces = [];
  traces.push({x:wTrain, y:yTrain, mode:"lines", name:"Historical", line:{color:"#cbd5e1",width:1.2},
    hovertemplate:"<b>%{x|%d %b %Y}</b><br>Sales: %{y:.0f} units<extra>Historical</extra>"});
  traces.push({x:wTest, y:yTest, mode:"lines+markers", name:"Actual (test)", line:{color:"#0f172a",width:2.2}, marker:{size:6},
    hovertemplate:"<b>%{x|%d %b %Y}</b><br>Actual: %{y:.0f} units<extra>Actual</extra>"});

  MODEL_NAMES.forEach(name => {
    const r = cfg.results[name];
    const isBest = name === best;
    traces.push({x:wTest, y:r.y_pred_test, mode:"lines", name:name+" (test)",
      line:{color:COLORS[name], width:isBest?2.5:1.5, dash:isBest?null:"dot"}, opacity:isBest?1:0.4,
      hovertemplate:"<b>%{x|%d %b %Y}</b><br>Predicted: %{y:.0f} units<extra>"+name+"</extra>"});
    traces.push({x:futureWeeks, y:r.forecast.slice(0,horizon), mode:"lines+markers", name:name+" (forecast)",
      line:{color:COLORS[name], width:isBest?2.5:1.5}, marker:{size:isBest?7:4, symbol:"diamond"}, opacity:isBest?1:0.4,
      hovertemplate:"<b>%{x|%d %b %Y}</b><br>Forecast: %{y:.0f} units<extra>"+name+"</extra>"});
    if (isBest) {
      const ciUp = r.ci_upper.slice(0,horizon);
      const ciLo = r.ci_lower.slice(0,horizon);
      traces.push({x:futureWeeks.concat([...futureWeeks].reverse()),
        y:ciUp.concat([...ciLo].reverse()),
        fill:"toself", fillcolor:"rgba(99,102,241,0.12)", line:{color:"rgba(0,0,0,0)"}, name:"95% CI",
        showlegend:true, hoverinfo:"skip"});
    }
  });

  Plotly.newPlot("forecastChart", traces, {
    title:`SKU ${id} — Demand Forecast`, height:460, hovermode:"closest",
    legend:{orientation:"h",y:-0.22,x:0.5,xanchor:"center",font:{size:11}},
    margin:{l:50,r:30,t:55,b:75},
    xaxis:{title:"Week", type:"date", tickformat:"%d %b %Y"},
    yaxis:{title:"Weekly sales (units)"},
    shapes:[{type:"line",x0:lastTestDate,x1:lastTestDate,y0:0,y1:1,yref:"paper",
      line:{width:1.5,dash:"dash",color:"#94a3b8"}}],
    annotations:[{x:lastTestDate,y:1,yref:"paper",text:"Forecast →",showarrow:false,
      font:{size:11,color:"#64748b"},xanchor:"left",yanchor:"top"}],
    hoverlabel:{bgcolor:"white",bordercolor:"#cbd5e1",font:{size:12}},
  }, plotCfg);

  // ── Forecast table ──
  let html = "<table><tr><th>Week</th>";
  MODEL_NAMES.forEach(n => html += `<th>${n}</th>`);
  html += "<th>95% CI (best)</th></tr>";
  for (let i = 0; i < horizon; i++) {
    html += `<tr><td>${futureWeeks[i]}</td>`;
    MODEL_NAMES.forEach(n => html += `<td>${Math.round(cfg.results[n].forecast[i])}</td>`);
    html += `<td>[${Math.round(bRes.ci_lower[i])}, ${Math.round(bRes.ci_upper[i])}]</td></tr>`;
  }
  html += "</table>";
  document.getElementById("forecastTable").innerHTML = html;

  // ── Model comparison bars ──
  const metrics = ["mae","rmse","r2"];
  const titles = ["MAE ↓","RMSE ↓","R² ↑"];
  const compTraces = metrics.map((met,i) => ({
    x: MODEL_NAMES, y: MODEL_NAMES.map(n => cfg.results[n][met]),
    type:"bar", marker:{color:MODEL_NAMES.map(n=>COLORS[n])}, showlegend:false,
    text:MODEL_NAMES.map(n=>cfg.results[n][met].toFixed(2)), textposition:"outside",
    textfont:{size:11,family:"monospace"}, xaxis:`x${i+1}`, yaxis:`y${i+1}`,
    hovertemplate:"<b>%{x}</b><br>"+titles[i]+": %{y:.3f}<extra></extra>",
  }));
  Plotly.newPlot("compChart", compTraces, {
    height:280, margin:{l:45,r:20,t:35,b:20}, grid:{rows:1,columns:3,pattern:"independent"},
    annotations:titles.map((t,i)=>({text:`<b>${t}</b>`,x:(i+0.5)/3,y:1.08,xref:"paper",yref:"paper",showarrow:false,font:{size:12}})),
  }, plotCfg);

  // ── Metrics table ──
  let mt = "<table><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE %</th><th>R²</th><th></th></tr>";
  MODEL_NAMES.forEach(n => {
    const r = cfg.results[n];
    const cls = n === best ? ' class="best"' : '';
    mt += `<tr${cls}><td>${n}</td><td>${r.mae.toFixed(2)}</td><td>${r.rmse.toFixed(2)}</td><td>${r.mape.toFixed(1)}</td><td>${r.r2.toFixed(3)}</td><td>${n===best?"★":""}</td></tr>`;
  });
  mt += "</table>";
  document.getElementById("metricsTable").innerHTML = mt;

  // ── Residuals ──
  const residuals = yTest.map((v,i) => +(v - bRes.y_pred_test[i]).toFixed(2));
  Plotly.newPlot("residChart", [{
    x:wTest, y:residuals, type:"bar",
    marker:{color:residuals.map(r => r>=0?"#6366f1":"#ef4444")},
    hovertemplate:"<b>%{x|%d %b %Y}</b><br>Residual: %{y:.1f} units<extra></extra>",
  }], {
    title:`Residuals — ${best}`, height:300, margin:{l:45,r:20,t:40,b:30},
    xaxis:{title:"Week", type:"date", tickformat:"%d %b %Y"}, yaxis:{title:"Actual − Predicted"},
    shapes:[{type:"line",x0:wTest[0],x1:wTest[wTest.length-1],y0:0,y1:0,line:{color:"#94a3b8",width:1}}],
    hoverlabel:{bgcolor:"white",bordercolor:"#cbd5e1",font:{size:12}},
  }, plotCfg);

  // ── Feature importance ──
  const fi = cfg.feature_importance;
  const fiSorted = Object.entries(fi).sort((a,b)=>a[1]-b[1]);
  Plotly.newPlot("featChart", [{
    y:fiSorted.map(e=>e[0]), x:fiSorted.map(e=>e[1]),
    type:"bar", orientation:"h", marker:{color:"#6366f1"},
    hovertemplate:"<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    text:fiSorted.map(e=>e[1].toFixed(3)), textposition:"outside", textfont:{size:10,family:"monospace"},
  }], {
    title:"Feature Importance (Random Forest)", height:300,
    margin:{l:150,r:60,t:40,b:30}, xaxis:{title:"Importance"},
  }, plotCfg);
}

function kpi(label, value, sub) {
  return `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-sub">${sub}</div></div>`;
}

function renderHeatmap() {
  const ids = Object.keys(DATA.heatmap).sort((a,b)=>+a - +b);
  const yLabels = ids.map(id => `SKU ${id}`);
  const z = ids.map(id => DATA.heatmap[id]);
  Plotly.newPlot("heatmapChart", [{
    z:z, x:DATA.heatmap_weeks, y:yLabels, type:"heatmap", colorscale:"Blues",
    text:z.map(row=>row.map(v=>Math.round(v))), texttemplate:"%{text}", textfont:{size:9},
    hovertemplate:"<b>%{y}</b><br>Week: %{x}<br>Forecast demand: %{z:.0f} units<extra></extra>",
  }], {
    title:"Forecasted Demand — All SKUs (Random Forest)",
    height: Math.max(500, ids.length * 22),
    margin:{l:70,r:20,t:50,b:30}, yaxis:{dtick:1, autorange:"reversed"},
    hoverlabel:{bgcolor:"white",bordercolor:"#cbd5e1",font:{size:12}},
  }, plotCfg);
}

renderSKU();
renderHeatmap();
</script>
</body>
</html>"""

test_step = TEST_SIZES[1] - TEST_SIZES[0]
html_out = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(dashboard_data))
html_out = html_out.replace("__TEST_MIN__", str(TEST_SIZES[0]))
html_out = html_out.replace("__TEST_MAX__", str(TEST_SIZES[-1]))
html_out = html_out.replace("__TEST_STEP__", str(test_step))

with open("../dashboards/demand_forecasting_dashboard.html", "w") as f:
    f.write(html_out)

print(f"Dashboard written to dashboards/demand_forecasting_dashboard.html ({len(html_out):,} bytes)")
