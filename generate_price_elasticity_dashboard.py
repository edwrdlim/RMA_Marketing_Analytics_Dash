"""
Generate a self-contained interactive HTML price elasticity dashboard.
Pre-computes all SKU elasticity results, curves, and scenarios so the
HTML dropdowns and controls work live with no Python backend.

Usage:  python generate_price_elasticity_dashboard.py
Output: price_elasticity_dashboard.html
"""

import json, time, warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings("ignore")

SCENARIOS = [-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30]
CURVE_POINTS = 300

t0 = time.time()

# ── Load data ─────────────────────────────────────────────────────────────────
raw  = pd.read_csv("data_raw.csv")
proc = pd.read_csv("data_processed.csv")
raw["week"]  = pd.to_datetime(raw["week"])
proc["week"] = pd.to_datetime(proc["week"])
raw["feat_main_page"] = raw["feat_main_page"].astype(str).str.lower().eq("true").astype(int)

MONTH_COLS = [f"month_{i}" for i in range(2, 13)]

sku_meta = raw.groupby("sku").agg(
    functionality=("functionality", "first"),
    color=("color",  "first"),
    vendor=("vendor", "first"),
    avg_price=("price", "mean"),
    avg_sales=("weekly_sales", "mean"),
    promo_rate=("feat_main_page", "mean"),
).reset_index()

sku_meta["category"] = sku_meta["functionality"].str.replace(
    r"^\d+\.", "", regex=True).str.strip()
sku_meta["display_name"] = sku_meta.apply(
    lambda r: f"{r['category']} ({r['color'].title()})", axis=1)
sku_meta["label"] = sku_meta.apply(
    lambda r: f"SKU {r['sku']} — {r['display_name']}", axis=1)

def meta(sku_id):
    row = sku_meta[sku_meta["sku"] == sku_id]
    return row.iloc[0] if len(row) > 0 else None

# ── Fit Scan*Pro per SKU ───────────────────────────────────────────────────────
def fit_scanpro(sku_id):
    df = proc[proc["sku"] == sku_id].sort_values("week").copy()
    df = df[df["weekly_sales"] > 0]
    if len(df) < 15 or df["feat_main_page"].nunique() < 2:
        return None

    y = np.log(df["weekly_sales"].values)
    X = pd.DataFrame(index=df.index)
    X["log_price"]      = np.log(df["price"].clip(lower=0.01))
    X["feat_main_page"] = df["feat_main_page"]
    X["trend"]          = df["trend"]
    for m in MONTH_COLS:
        if m in df.columns:
            X[m] = df[m]
    X = sm.add_constant(X, has_constant="add")

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    elast      = float(model.params.get("log_price", np.nan))
    elast_pval = float(model.pvalues.get("log_price", np.nan))
    promo_coef = float(model.params.get("feat_main_page", np.nan))
    promo_pval = float(model.pvalues.get("feat_main_page", np.nan))
    if np.isnan(elast):
        return None

    m  = meta(sku_id)
    P0 = float(df["price"].mean())
    Q0 = float(df["weekly_sales"].mean())
    R0 = P0 * Q0

    # Demand & revenue curves
    price_range   = np.linspace(P0 * 0.20, P0 * 3.0, CURVE_POINTS)
    demand_curve  = np.clip(Q0 * (price_range / P0) ** elast, 0, None)
    revenue_curve = price_range * demand_curve
    opt_idx       = int(np.argmax(revenue_curve))

    # Fitted vs actual for time-series chart
    X_full = X.copy()
    fitted = np.exp(model.predict(X_full)).tolist()

    # What-if scenarios
    scenarios = []
    for pct in SCENARIOS:
        P1 = P0 * (1 + pct)
        Q1 = max(Q0 * ((P1 / P0) ** elast), 0)
        R1 = P1 * Q1
        scenarios.append({
            "label"     : f"{'↑' if pct>0 else '↓'} {abs(int(pct*100))}%",
            "pct"       : round(pct, 2),
            "new_price" : round(P1, 2),
            "d_demand"  : round(Q1 - Q0, 1),
            "d_demand_pct": round((Q1-Q0)/Q0*100, 1) if Q0 > 0 else 0,
            "new_rev"   : round(R1, 2),
            "d_rev"     : round(R1 - R0, 2),
            "d_rev_pct" : round((R1-R0)/R0*100, 1) if R0 > 0 else 0,
        })

    return {
        "sku"            : int(sku_id),
        "label"          : str(m["label"]) if m is not None else f"SKU {sku_id}",
        "category"       : str(m["category"]) if m is not None else "Unknown",
        "color"          : str(m["color"]) if m is not None else "Unknown",
        "avg_price"      : round(P0, 2),
        "avg_sales"      : round(Q0, 1),
        "avg_revenue"    : round(R0, 2),
        "promo_rate"     : round(float(m["promo_rate"]) * 100, 1) if m is not None else 0,
        "elasticity"     : round(elast, 4),
        "elast_pval"     : round(elast_pval, 4),
        "elast_sig"      : bool(elast_pval < 0.05),
        "promo_coef"     : round(promo_coef, 4),
        "promo_pval"     : round(promo_pval, 4),
        "r2"             : round(float(model.rsquared), 4),
        "optimal_price"  : round(float(price_range[opt_idx]), 2),
        "optimal_revenue": round(float(revenue_curve[opt_idx]), 2),
        "price_range"    : [round(p, 2) for p in price_range.tolist()],
        "demand_curve"   : [round(q, 1) for q in demand_curve.tolist()],
        "revenue_curve"  : [round(r, 2) for r in revenue_curve.tolist()],
        "weeks"          : [pd.Timestamp(w).strftime("%Y-%m-%d") for w in df["week"].values],
        "actual"         : [round(float(v), 1) for v in df["weekly_sales"].values],
        "fitted"         : [round(float(v), 1) for v in fitted],
        "scenarios"      : scenarios,
    }

# ── Run all SKUs ───────────────────────────────────────────────────────────────
print("⏳ Fitting Scan*Pro for all SKUs...")
all_sku_data = {}
for sku_id in sorted(proc["sku"].unique()):
    r = fit_scanpro(sku_id)
    if r:
        all_sku_data[int(sku_id)] = r
        print(f"  SKU {sku_id:2d} — ε={r['elasticity']:+.3f}  "
              f"p={r['elast_pval']:.3f}  "
              f"{'★' if r['elast_sig'] else ' '}  R²={r['r2']:.3f}")

# ── Portfolio summary stats ────────────────────────────────────────────────────
elasticities = [r["elasticity"] for r in all_sku_data.values()]
n_elastic    = sum(1 for e in elasticities if e < -1)
n_inelastic  = sum(1 for e in elasticities if -1 <= e < 0)
n_sig        = sum(1 for r in all_sku_data.values() if r["elast_sig"])
avg_elast    = round(float(np.mean(elasticities)), 3)
most_sens    = min(all_sku_data.values(), key=lambda r: r["elasticity"])
least_sens   = max(all_sku_data.values(), key=lambda r: r["elasticity"])

# ── Heatmap matrix — all SKUs × all scenarios ──────────────────────────────────
heatmap_skus    = sorted(all_sku_data.keys())
heatmap_labels  = [all_sku_data[s]["label"].replace("SKU ", "") for s in heatmap_skus]
scenario_labels = [f"{'↑' if p>0 else '↓'}{abs(int(p*100))}%" for p in SCENARIOS]
heatmap_z, heatmap_text = [], []

for sku_id in heatmap_skus:
    r = all_sku_data[sku_id]
    row_z, row_t = [], []
    for sc in r["scenarios"]:
        row_z.append(sc["d_rev_pct"])
        row_t.append(f"{sc['d_rev_pct']:+.1f}%")
    heatmap_z.append(row_z)
    heatmap_text.append(row_t)

dashboard_data = {
    "skus"            : all_sku_data,
    "portfolio"       : {
        "n_skus"     : len(all_sku_data),
        "n_elastic"  : n_elastic,
        "n_inelastic": n_inelastic,
        "n_sig"      : n_sig,
        "avg_elast"  : avg_elast,
        "most_sens"  : {"sku": most_sens["sku"], "label": most_sens["label"],
                        "elasticity": most_sens["elasticity"]},
        "least_sens" : {"sku": least_sens["sku"], "label": least_sens["label"],
                        "elasticity": least_sens["elasticity"]},
    },
    "heatmap"         : {
        "z"              : heatmap_z,
        "text"           : heatmap_text,
        "sku_labels"     : heatmap_labels,
        "scenario_labels": scenario_labels,
    },
}

elapsed = time.time() - t0
print(f"\n {len(all_sku_data)} SKUs processed in {elapsed:.1f}s")

# ═════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Price Elasticity Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {
    --slate-50:#f8fafc; --slate-100:#f1f5f9; --slate-200:#e2e8f0;
    --slate-300:#cbd5e1; --slate-400:#94a3b8; --slate-500:#64748b;
    --slate-600:#475569; --slate-700:#334155; --slate-800:#1e293b;
    --slate-900:#0f172a;
    --blue:#2563EB; --blue-lt:rgba(37,99,235,0.10);
    --red:#DC2626;   --green:#16a34a;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    background:var(--slate-50); color:var(--slate-900); line-height:1.5; }
  .container { max-width:1200px; margin:0 auto; padding:1.5rem; }

  /* Header */
  header { background:white; border-bottom:1px solid var(--slate-200);
    padding:1.2rem 0; margin-bottom:1.5rem; }
  header .container { display:flex; flex-direction:column; gap:.8rem; }
  .header-top { display:flex; align-items:center;
    justify-content:space-between; flex-wrap:wrap; gap:.75rem; }
  h1 { font-size:1.5rem; font-weight:700; }
  .header-meta { font-size:.8rem; color:var(--slate-500); }

  /* Controls */
  .controls { display:flex; gap:1rem; align-items:center; flex-wrap:wrap; }
  select { font-family:inherit; font-size:.85rem; padding:.5rem .9rem;
    border-radius:8px; border:1px solid var(--slate-200); background:white;
    min-width:320px; color:var(--slate-700); cursor:pointer; }
  select:focus { outline:none; border-color:var(--blue);
    box-shadow:0 0 0 3px var(--blue-lt); }

  /* KPI cards */
  .kpi-row { display:flex; gap:.75rem; margin:1rem 0 1.5rem; flex-wrap:wrap; }
  .kpi { flex:1; min-width:155px; background:white;
    border:1px solid var(--slate-200); border-radius:12px; padding:1rem 1.2rem; }
  .kpi-label { font-size:.7rem; color:var(--slate-500); text-transform:uppercase;
    letter-spacing:.05em; font-weight:500; }
  .kpi-value { font-size:1.35rem; font-weight:700; margin-top:.15rem;
    font-family:'SF Mono',SFMono-Regular,Consolas,monospace; }
  .kpi-sub { font-size:.75rem; color:var(--slate-500); margin-top:.1rem; }

  /* SKU header */
  .sku-header h2 { font-size:1.15rem; font-weight:700; }
  .sku-header p  { font-size:.85rem; color:var(--slate-500); margin-top:.15rem; }

  /* Interpretation badge */
  .interp { border-left:3px solid var(--blue); background:#eff6ff;
    padding:.55rem 1rem; border-radius:0 6px 6px 0;
    font-size:.84rem; color:var(--slate-700);
    margin:.6rem 0 1rem; line-height:1.5; }

  /* Charts */
  .chart-card { background:white; border:1px solid var(--slate-200);
    border-radius:12px; margin-bottom:1rem; overflow:hidden; }
  .section-title { font-size:1rem; font-weight:600;
    margin:1.5rem 0 .5rem; color:var(--slate-700); }

  /* Tables */
  table { width:100%; border-collapse:collapse; font-size:.82rem;
    background:white; border:1px solid var(--slate-200);
    border-radius:12px; overflow:hidden; margin-bottom:1rem; }
  th { background:var(--slate-900); color:white; font-weight:500;
    text-transform:uppercase; font-size:.7rem; letter-spacing:.04em; }
  th, td { padding:.6rem .9rem; text-align:center;
    border-bottom:1px solid var(--slate-100); }
  td { font-family:'SF Mono',SFMono-Regular,Consolas,monospace; font-size:.8rem; }
  td.label-col { text-align:left; font-family:inherit; font-size:.82rem; }
  .pos { color:var(--green); font-weight:600; }
  .neg { color:var(--red);   font-weight:600; }
  .sig { color:var(--blue);  font-weight:700; }
  tr.row-up { background:#f0fdf4; }
  tr.row-dn { background:#fef9f9; }

  /* Grid */
  .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
  @media(max-width:768px) {
    .grid-2 { grid-template-columns:1fr; }
    .controls { flex-direction:column; align-items:stretch; }
    select { min-width:100%; }
  }

  /* Portfolio section */
  .portfolio-section { margin-top:2.5rem; }

  /* Methodology */
  .methodology { background:white; border:1px solid var(--slate-200);
    border-radius:12px; padding:1.5rem; margin-top:1.5rem;
    font-size:.85rem; color:var(--slate-600); line-height:1.7; }
  .methodology h3 { color:var(--slate-800); margin-bottom:.5rem; }
  .methodology ul  { padding-left:1.2rem; }
  footer { text-align:center; padding:2rem 0;
    font-size:.78rem; color:var(--slate-400); }
</style>
</head>
<body>

<!-- ═══════════════════════════════════ HEADER -->
<header>
  <div class="container">
    <div class="header-top">
      <h1> Price Elasticity &amp; Scenario Testing</h1>
    </div>
    <div class="controls">
      <select id="skuSelect" onchange="renderSKU()"></select>
    </div>
    <div class="header-meta" id="headerMeta"></div>
  </div>
</header>

<div class="container">

  <!-- SKU header + KPIs -->
  <div id="skuHeader" class="sku-header"></div>
  <div id="interpBox"  class="interp" style="display:none"></div>
  <div id="kpiRow"     class="kpi-row"></div>

  <!-- Demand + Revenue curves -->
  <div class="grid-2">
    <div class="chart-card"><div id="demandChart"></div></div>
    <div class="chart-card"><div id="revenueChart"></div></div>
  </div>

  <!-- Scan*Pro fit chart -->
  <h3 class="section-title">Scan*Pro Model Fit — Actual vs Fitted</h3>
  <div class="chart-card"><div id="fitChart"></div></div>

  <!-- Scenario waterfall -->
  <h3 class="section-title">Revenue Impact by Price Scenario</h3>
  <div class="chart-card"><div id="waterfallChart"></div></div>

  <!-- Scenario table -->
  <h3 class="section-title">What-If Scenario Table</h3>
  <div id="scenarioTable"></div>

  <!-- Portfolio overview -->
  <div class="portfolio-section">
    <h3 class="section-title">Portfolio Overview — All 44 SKUs</h3>
    <div id="portfolioKpis" class="kpi-row"></div>

    <h3 class="section-title">Elasticity Ranking</h3>
    <div class="chart-card"><div id="rankingChart"></div></div>

    <h3 class="section-title">Revenue Impact Heatmap — All SKUs × All Scenarios</h3>
    <div class="chart-card"><div id="heatmapChart"></div></div>

    <h3 class="section-title">All-SKU Price Strategy Summary</h3>
    <div id="strategyTable"></div>
  </div>

  <div class="methodology">
    <h3>Methodology</h3>
    <p><b>Data:</b> 44 SKUs × 98 weeks from <code>data_processed.csv</code>
      (lagged prices, trend, month dummies, one-hot categoricals).
      SKU names from <code>data_raw.csv</code>.</p>
    <p><b>Model:</b> Scan*Pro log-log OLS fitted per SKU:</p>
    <p style="margin:.4rem 0 .4rem 1.2rem;font-family:monospace;font-size:.82rem;">
      log(sales) = β₀ + β₁·log(price) + β₂·feat_main_page + β₃·trend + Σγₘ·monthₘ + ε</p>
    <p>The price coefficient <b>β₁ is the price elasticity directly</b>.</p>
    <p><b>Scenario simulation:</b>
      Q_new = Q₀ × (P_new/P₀)^ε &nbsp;·&nbsp; R_new = P_new × Q_new</p>
    <ul style="margin:.4rem 0 0 1.2rem;">
      <li><b>Elastic (|ε| &gt; 1):</b> Lowering price grows revenue</li>
      <li><b>Inelastic (|ε| &lt; 1):</b> Raising price grows revenue</li>
    </ul>
    <p style="margin-top:.75rem;font-size:.78rem;color:var(--slate-400);">
      Reference: Cohen, M.C., Gras, P.E., Pentecoste, A., &amp; Zhang, R. (2022).
      <i>Demand Prediction in Retail.</i> Springer SSCM 14.</p>
  </div>

</div>

<footer>Generated from price_elasticity.ipynb · Scan*Pro OLS via statsmodels</footer>

<script>
const DATA    = __DATA_PLACEHOLDER__;
const plotCfg = { responsive:true, displayModeBar:false };
const BLUE    = "#2563EB";
const SLATE   = "#0f172a";
const GREY_LG = "#cbd5e1";
const GREY_MD = "#94a3b8";
const RED     = "#DC2626";
const GREEN   = "#16a34a";

// ── Populate dropdown ──────────────────────────────────────────────────────
const sel = document.getElementById("skuSelect");
Object.keys(DATA.skus).sort((a,b)=>+a-+b).forEach(id => {
  const opt = document.createElement("option");
  opt.value = id;
  opt.textContent = DATA.skus[id].label;
  sel.appendChild(opt);
});

// ── Portfolio KPIs (static) ────────────────────────────────────────────────
function renderPortfolioKpis() {
  const p = DATA.portfolio;
  document.getElementById("portfolioKpis").innerHTML = [
    kpi("SKUs analysed",        p.n_skus,
        "with sufficient data"),
    kpi("Elastic  |ε| > 1",    `${p.n_elastic} / ${p.n_skus}`,
        "price-sensitive demand"),
    kpi("Inelastic  |ε| < 1",  `${p.n_inelastic} / ${p.n_skus}`,
        "price-insensitive demand"),
    kpi("Significant  p<0.05", `${p.n_sig} / ${p.n_skus}`,
        "statistically reliable ε"),
    kpi("Avg elasticity",       p.avg_elast,
        "across all SKUs"),
    kpi("Most sensitive",
        `SKU ${p.most_sens.sku}`,
        `ε = ${p.most_sens.elasticity.toFixed(3)}`),
    kpi("Least sensitive",
        `SKU ${p.least_sens.sku}`,
        `ε = ${p.least_sens.elasticity.toFixed(3)}`),
  ].join("");
}

// ── Elasticity ranking chart ───────────────────────────────────────────────
function renderRanking() {
  const rows  = Object.values(DATA.skus).sort((a,b)=>a.elasticity-b.elasticity);
  const yLbls = rows.map(r => r.label.replace("SKU ","") + (r.elast_sig?" ★":""));
  const xVals = rows.map(r => r.elasticity);
  const cols  = rows.map(r => r.elast_sig ? BLUE : GREY_MD);

  Plotly.newPlot("rankingChart", [{
    type:"bar", orientation:"h",
    y:yLbls, x:xVals, marker:{color:cols},
    text:xVals.map(v=>v.toFixed(2)),
    textposition:"outside", textfont:{size:9, color:GREY_MD},
    customdata:rows.map(r=>[r.elast_pval, r.avg_price, r.avg_sales,
                              r.r2, r.elast_sig?"Yes":"No"]),
    hovertemplate:"<b>%{y}</b><br>ε = <b>%{x:.4f}</b><br>" +
      "p-value: %{customdata[0]:.4f}  Sig: %{customdata[4]}<br>" +
      "Avg price: £%{customdata[1]:.2f}  Avg sales: %{customdata[2]:.0f}/wk<br>" +
      "R²: %{customdata[3]:.3f}<extra></extra>",
  }], {
    height:1000,
    margin:{l:10, r:80, t:30, b:30},
    xaxis:{title:"Price Elasticity (ε)", zeroline:true,
           zerolinecolor:GREY_LG, zerolinewidth:1,
           tickfont:{color:GREY_MD}},
    yaxis:{tickfont:{size:10, color:SLATE}},
    showlegend:false, plot_bgcolor:"white", paper_bgcolor:"white",
    shapes:[
      {type:"line", x0:-1, x1:-1, y0:0, y1:1, xref:"x", yref:"paper",
       line:{dash:"dash", color:GREY_MD, width:1.2}},
    ],
    annotations:[
      {x:-1, y:1, xref:"x", yref:"paper",
       text:"Unit elastic ε=−1", showarrow:false,
       font:{color:GREY_MD, size:10}, yanchor:"bottom", xanchor:"left"},
      {x:.99, y:.01, xref:"paper", yref:"paper",
       text:"★ = significant (p<0.05)  ·  Blue = significant  ·  Grey = not significant",
       showarrow:false, font:{color:GREY_MD, size:9},
       bgcolor:"white", align:"right"},
    ],
  }, plotCfg);
}

// ── Revenue heatmap ────────────────────────────────────────────────────────
function renderHeatmap() {
  const hm = DATA.heatmap;
  Plotly.newPlot("heatmapChart", [{
    type:"heatmap",
    z:hm.z, x:hm.scenario_labels, y:hm.sku_labels,
    text:hm.text, texttemplate:"%{text}", textfont:{size:8, color:"#334155"},
    colorscale:[
      [0.00, RED],   [0.40, "#FCA5A5"],
      [0.50, "#F8FAFC"],
      [0.60, "#86EFAC"], [1.00, GREEN],
    ],
    zmid:0,
    colorbar:{title:"ΔRevenue (%)", ticksuffix:"%", len:.7,
              tickfont:{size:10}},
    hovertemplate:"<b>%{y}</b><br>Scenario: %{x}<br>" +
      "Revenue change: <b>%{z:+.1f}%</b><extra></extra>",
  }], {
    height:1050,
    margin:{l:10, r:110, t:30, b:60},
    xaxis:{title:"Price Change Scenario", side:"top",
           tickfont:{size:11}},
    yaxis:{tickfont:{size:9}, autorange:"reversed"},
    paper_bgcolor:"white",
  }, plotCfg);
}

// ── Strategy table ─────────────────────────────────────────────────────────
function renderStrategyTable() {
  const rows = Object.values(DATA.skus).sort((a,b)=>a.elasticity-b.elasticity);
  let html = `<table>
    <tr><th>SKU</th><th>Product</th><th>Elasticity</th><th>Sig.</th>
    <th>Avg Price</th><th>ΔRev +10%</th><th>ΔRev −10%</th>
    <th>Optimal Price</th><th>Strategy</th></tr>`;

  rows.forEach(r => {
    const e   = r.elasticity;
    const P0  = r.avg_price, Q0 = r.avg_sales, R0 = r.avg_revenue;
    const Pup = P0*1.10, Pdn = P0*0.90;
    const Rup = Pup * Math.max(Q0 * Math.pow(Pup/P0, e), 0);
    const Rdn = Pdn * Math.max(Q0 * Math.pow(Pdn/P0, e), 0);
    const dUp = R0>0 ? ((Rup-R0)/R0*100).toFixed(1) : "0.0";
    const dDn = R0>0 ? ((Rdn-R0)/R0*100).toFixed(1) : "0.0";
    const strat = e < -1 ? "Consider lower price"
                : e <  0 ? "Consider higher price"
                : "Review data";
    const eCol = r.elast_sig ? "sig" : "";
    html += `<tr>
      <td class="label-col"><b>SKU ${r.sku}</b></td>
      <td class="label-col" style="font-size:.78rem">${r.label.split("—")[1]?.trim()||""}</td>
      <td class="${eCol}">${e.toFixed(4)}</td>
      <td>${r.elast_sig?"★":""}</td>
      <td>£${P0.toFixed(2)}</td>
      <td class="${parseFloat(dUp)>=0?'pos':'neg'}">${parseFloat(dUp)>=0?'+':''}${dUp}%</td>
      <td class="${parseFloat(dDn)>=0?'pos':'neg'}">${parseFloat(dDn)>=0?'+':''}${dDn}%</td>
      <td>£${r.optimal_price.toFixed(2)}</td>
      <td class="label-col" style="font-size:.78rem">${strat}</td>
    </tr>`;
  });
  html += "</table>";
  document.getElementById("strategyTable").innerHTML = html;
}

// ── SKU drill-down ─────────────────────────────────────────────────────────
function renderSKU() {
  const id = sel.value;
  const r  = DATA.skus[id];
  const e  = r.elasticity;
  const P0 = r.avg_price, Q0 = r.avg_sales, R0 = r.avg_revenue;

  // Header
  document.getElementById("skuHeader").innerHTML =
    `<h2>${r.label}</h2>
     <p>Category: ${r.category} &nbsp;·&nbsp;
        Colour: ${r.color} &nbsp;·&nbsp;
        Avg price: £${P0.toFixed(2)} &nbsp;·&nbsp;
        Avg sales: ${Q0.toFixed(0)} units/wk &nbsp;·&nbsp;
        Avg revenue: £${R0.toLocaleString("en-GB",{maximumFractionDigits:0})}/wk &nbsp;·&nbsp;
        Promo rate: ${r.promo_rate}%</p>`;

  // Interpretation
  let interp;
  if      (e < -1) interp = `Elastic — |ε| = ${Math.abs(e).toFixed(2)} > 1. Demand is price-sensitive. A 10% price rise causes ~${(Math.abs(e)*10).toFixed(0)}% demand fall. <b>Lowering price is likely to grow revenue.</b>`;
  else if (e <  0) interp = `Inelastic — |ε| = ${Math.abs(e).toFixed(2)} < 1. Demand is price-insensitive. A 10% price rise causes only ~${(Math.abs(e)*10).toFixed(0)}% demand fall. <b>Raising price is likely to grow revenue.</b>`;
  else             interp = `Unusual positive elasticity (ε = ${e.toFixed(2)}). Check model significance (p = ${r.elast_pval.toFixed(4)}).`;
  const box = document.getElementById("interpBox");
  box.innerHTML = interp;
  box.style.display = "block";

  // KPIs
  const sigStr = r.elast_sig ? "★ significant" : "not significant";
  document.getElementById("kpiRow").innerHTML = [
    kpi("Price elasticity", e.toFixed(4),
        `p = ${r.elast_pval.toFixed(4)}  ·  ${sigStr}`),
    kpi("Model R²",         r.r2.toFixed(4),
        "Scan*Pro goodness of fit"),
    kpi("Avg price",        `£${P0.toFixed(2)}`,
        `Avg sales: ${Q0.toFixed(0)} units/wk`),
    kpi("Avg revenue",      `£${R0.toLocaleString("en-GB",{maximumFractionDigits:0})}`,
        "per week at current price"),
    kpi("Revenue-max price",`£${r.optimal_price.toFixed(2)}`,
        `Peak £${r.optimal_revenue.toLocaleString("en-GB",{maximumFractionDigits:0})}/wk`),
  ].join("");

  // Demand curve
  Plotly.newPlot("demandChart", [
    {x:r.price_range, y:r.demand_curve, mode:"lines",
     line:{color:BLUE, width:2}, name:"Demand",
     hovertemplate:"Price: £%{x:.2f}<br>Demand: %{y:.0f} units<extra></extra>"},
    {x:[P0], y:[Q0], mode:"markers",
     marker:{size:11, color:SLATE, line:{color:"white",width:2}},
     name:"Current price",
     hovertemplate:`Current: £${P0.toFixed(2)} → ${Q0.toFixed(0)} units<extra></extra>`},
  ], {
    title:{text:`Demand Curve  (ε = ${e.toFixed(3)})`, font:{size:13,color:SLATE}},
    xaxis:{title:"Price (£)", tickfont:{color:GREY_MD}},
    yaxis:{title:"Weekly demand (units)", tickfont:{color:GREY_MD}},
    legend:{orientation:"h", y:-0.22},
    plot_bgcolor:"white", paper_bgcolor:"white",
    margin:{t:50,b:50,l:60,r:20}, height:340,
  }, plotCfg);

  // Revenue curve
  Plotly.newPlot("revenueChart", [
    {x:r.price_range, y:r.revenue_curve, mode:"lines",
     line:{color:GREY_LG, width:2}, name:"Revenue",
     hovertemplate:"Price: £%{x:.2f}<br>Revenue: £%{y:,.0f}<extra></extra>"},
    {x:[P0], y:[R0], mode:"markers",
     marker:{size:11, color:SLATE, line:{color:"white",width:2}},
     name:"Current",
     hovertemplate:`Current: £${P0.toFixed(2)} → £${R0.toLocaleString("en-GB",{maximumFractionDigits:0})}<extra></extra>`},
    {x:[r.optimal_price], y:[r.optimal_revenue], mode:"markers",
     marker:{size:14, color:BLUE, symbol:"star", line:{color:"white",width:1.5}},
     name:`Optimal £${r.optimal_price.toFixed(2)}`,
     hovertemplate:`Optimal: £${r.optimal_price.toFixed(2)} → £${r.optimal_revenue.toLocaleString("en-GB",{maximumFractionDigits:0})}/wk<extra></extra>`},
  ], {
    title:{text:`Revenue Curve  —  Optimal: £${r.optimal_price.toFixed(2)}`,
           font:{size:13,color:SLATE}},
    xaxis:{title:"Price (£)", tickfont:{color:GREY_MD}},
    yaxis:{title:"Weekly revenue (£)", tickfont:{color:GREY_MD}},
    legend:{orientation:"h", y:-0.22},
    plot_bgcolor:"white", paper_bgcolor:"white",
    margin:{t:50,b:50,l:60,r:20}, height:340,
  }, plotCfg);

  // Scan*Pro fit chart
  const nWeeks = r.weeks.length;
  Plotly.newPlot("fitChart", [
    {x:r.weeks, y:r.actual, mode:"lines", name:"Actual sales",
     line:{color:SLATE, width:1.8},
     hovertemplate:"<b>%{x|%d %b %Y}</b><br>Actual: %{y:.0f} units<extra>Actual</extra>"},
    {x:r.weeks, y:r.fitted, mode:"lines", name:"Scan*Pro fitted",
     line:{color:BLUE, width:1.5, dash:"dash"},
     hovertemplate:"<b>%{x|%d %b %Y}</b><br>Fitted: %{y:.0f} units<extra>Fitted</extra>"},
  ], {
    title:{text:`Scan*Pro Model Fit  (R² = ${r.r2.toFixed(3)})`,
           font:{size:13,color:SLATE}},
    xaxis:{title:"Week", type:"date", tickformat:"%b %Y",
           tickfont:{color:GREY_MD}},
    yaxis:{title:"Weekly sales (units)", tickfont:{color:GREY_MD}},
    legend:{orientation:"h", y:-0.2},
    plot_bgcolor:"white", paper_bgcolor:"white",
    height:320, margin:{t:50,b:60,l:60,r:20},
    hoverlabel:{bgcolor:"white", bordercolor:GREY_LG, font:{size:12}},
  }, plotCfg);

  // Waterfall
  const sc    = r.scenarios;
  const yVals = sc.map(s=>s.d_rev);
  Plotly.newPlot("waterfallChart", [{
    type:"bar",
    x:sc.map(s=>s.label),
    y:yVals,
    marker:{color:yVals.map(v=>v<0?RED:GREEN), line:{width:0}},
    text:yVals.map(v=>`£${v>=0?"+":""}${v.toLocaleString("en-GB",{maximumFractionDigits:0})}`),
    textposition:"outside", textfont:{size:10, color:GREY_MD},
    customdata:sc.map(s=>[s.new_price,s.d_demand_pct,s.new_rev,s.d_rev_pct]),
    hovertemplate:"<b>%{x}</b><br>" +
      "New price: £%{customdata[0]:.2f}<br>" +
      "Δ Demand: %{customdata[1]:+.1f}%<br>" +
      "New revenue: £%{customdata[2]:,.0f}<br>" +
      "Δ Revenue: %{customdata[3]:+.1f}%<extra></extra>",
  }], {
    title:{text:`Revenue Change by Price Scenario  (ε = ${e.toFixed(3)})`,
           font:{size:13,color:SLATE}},
    xaxis:{title:"Price scenario", tickfont:{color:GREY_MD}},
    yaxis:{title:"Δ Revenue vs baseline (£/week)", tickfont:{color:GREY_MD}},
    shapes:[{type:"line",x0:0,x1:1,y0:0,y1:0,xref:"paper",yref:"y",
             line:{color:GREY_LG,width:1}}],
    plot_bgcolor:"white", paper_bgcolor:"white",
    height:340, margin:{t:50,b:50,l:60,r:20},
  }, plotCfg);

  // Scenario table
  let html = `<table>
    <tr><th>Scenario</th><th>New Price</th>
    <th>Δ Demand (units)</th><th>Δ Demand (%)</th>
    <th>New Revenue</th><th>Δ Revenue (£)</th><th>Δ Revenue (%)</th></tr>`;
  sc.forEach(s => {
    const rowCls = s.pct > 0 ? "row-dn" : "row-up";
    const dc = s.d_demand >= 0 ? "pos":"neg";
    const rc = s.d_rev    >= 0 ? "pos":"neg";
    html += `<tr class="${rowCls}">
      <td><b>${s.label}</b></td>
      <td>£${s.new_price.toFixed(2)}</td>
      <td class="${dc}">${s.d_demand>=0?"+":""}${s.d_demand}</td>
      <td class="${dc}">${s.d_demand_pct>=0?"+":""}${s.d_demand_pct}%</td>
      <td>£${s.new_rev.toLocaleString("en-GB",{maximumFractionDigits:0})}</td>
      <td class="${rc}">${s.d_rev>=0?"£+":"£"}${s.d_rev.toLocaleString("en-GB",{maximumFractionDigits:0})}</td>
      <td class="${rc}">${s.d_rev_pct>=0?"+":""}${s.d_rev_pct}%</td>
    </tr>`;
  });
  html += "</table>";
  document.getElementById("scenarioTable").innerHTML = html;
}

// ── Helper ─────────────────────────────────────────────────────────────────
function kpi(label, value, sub) {
  return `<div class="kpi">
    <div class="kpi-label">${label}</div>
    <div class="kpi-value">${value}</div>
    <div class="kpi-sub">${sub}</div>
  </div>`;
}

// ── Init ───────────────────────────────────────────────────────────────────
renderSKU();
renderPortfolioKpis();
renderRanking();
renderHeatmap();
renderStrategyTable();
</script>
</body>
</html>"""

# Inject data
html_out = HTML.replace("__DATA_PLACEHOLDER__", json.dumps(dashboard_data))

with open("price_elasticity_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_out)

print(f" Dashboard written → price_elasticity_dashboard.html "
      f"({len(html_out):,} bytes / {len(html_out)//1024} KB)")
