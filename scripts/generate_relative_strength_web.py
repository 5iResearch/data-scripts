"""
Relative Strength Ladder (website page, PREVIEW): top-down long-term relative strength, market -> sector ->
industry -> stock, Canada first.

Every level is judged on the same measure, the ratio of one price to another (total-return closes):
  - Long-term relative trend: a straight line fitted through the log of the ratio (the same regression as
    generate_ratio_channel_screener.py); its slope, as a yearly rate, is how much faster or slower the asset has
    compounded than its benchmark, and R² is how steady that has been. Measured over 10 years and over 20 (a
    toggle on the page). Funds with less history use all they have (at least five years) and are marked.
  - Last 12 months: the ratio's change over the past year, and whether it sits above its 40-week average.
  - Status, from those two: Leader (long-term up, still up), Fading (long-term up, down this year), Turning up
    (long-term down, up this year), Laggard (down on both).

Sections per market: sectors vs the index (table + ratio charts), a sector-vs-sector grid (does the row beat the
column?), industries vs the index and vs their own sector (US), and the stocks leading each sector, measured
against their sector fund rather than the whole market.
Output: outputs/relative-strength/Relative_Strength.html
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress

from common_screening import load_sp500_sectors
from generate_index_rsi_web import FULLSCREEN_HTML, LOGO_B64

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "relative-strength")
CDN_CSV = os.path.join(REPO_ROOT, "data", "cdn_1w_rev_est_screener.csv")

WINDOWS = (10, 20)         # look-back choices on the page, in years; the first is the default
MIN_YEARS = 5
TOP_STOCKS = 6             # per sector
MIN_STOCK_R2 = 0.5         # a steady relative uptrend, not one lucky year
CDN_MIN_CAP = 1000         # $M, Canadian stock universe (rev CSV)

US_SECTORS = {"XLK": "Technology", "XLC": "Communication", "XLY": "Consumer Disc.", "XLF": "Financials",
              "XLV": "Health Care", "XLI": "Industrials", "XLE": "Energy", "XLB": "Materials",
              "XLRE": "Real Estate", "XLU": "Utilities", "XLP": "Consumer Staples"}
CDN_SECTORS = {"XIT.TO": "Technology", "XFN.TO": "Financials", "XEG.TO": "Energy", "XMA.TO": "Materials",
               "XGD.TO": "Gold Miners", "ZIN.TO": "Industrials", "XRE.TO": "REITs", "XUT.TO": "Utilities",
               "XST.TO": "Consumer Staples"}
# Industry fund -> (name, parent sector fund)
US_INDUSTRIES = {
    "SMH": ("Semiconductors", "XLK"), "IGV": ("Software", "XLK"), "CIBR": ("Cybersecurity", "XLK"),
    "SKYY": ("Cloud Computing", "XLK"), "FDN": ("Internet", "XLC"), "IBUY": ("Online Retail", "XLY"),
    "XRT": ("Retail", "XLY"), "ITB": ("Homebuilders", "XLY"), "XHB": ("Home Products", "XLY"),
    "IBB": ("Biotech", "XLV"), "IHI": ("Medical Devices", "XLV"), "XPH": ("Pharma", "XLV"),
    "IHF": ("Health Providers", "XLV"), "KRE": ("Regional Banks", "XLF"), "KBE": ("Banks", "XLF"),
    "KIE": ("Insurance", "XLF"), "IAI": ("Brokers & Exchanges", "XLF"), "IPAY": ("Payments", "XLF"),
    "XME": ("Metals & Mining", "XLB"), "GDX": ("Gold Miners", "XLB"), "COPX": ("Copper Miners", "XLB"),
    "SLX": ("Steel", "XLB"), "XOP": ("Oil & Gas E&P", "XLE"), "OIH": ("Oil Services", "XLE"),
    "AMLP": ("Pipelines (MLPs)", "XLE"), "URA": ("Uranium", "XLE"), "ITA": ("Aerospace & Defense", "XLI"),
    "IYT": ("Transports", "XLI"), "JETS": ("Airlines", "XLI"), "PAVE": ("Infrastructure", "XLI"),
    "TAN": ("Solar", "XLU"), "PHO": ("Water", "XLU"), "MOO": ("Agribusiness", "XLP"),
}
GICS_TO_US = {"Information Technology": "XLK", "Communication Services": "XLC", "Consumer Discretionary": "XLY",
              "Financials": "XLF", "Health Care": "XLV", "Industrials": "XLI", "Energy": "XLE",
              "Materials": "XLB", "Real Estate": "XLRE", "Utilities": "XLU", "Consumer Staples": "XLP"}
GICS_TO_CDN = {"Information Technology": "XIT.TO", "Financials": "XFN.TO", "Energy": "XEG.TO",
               "Materials": "XMA.TO", "Industrials": "ZIN.TO", "Real Estate": "XRE.TO", "Utilities": "XUT.TO",
               "Consumer Staples": "XST.TO"}   # Health Care, Discretionary, Communication: no TSX sector fund -> XIC


def download(tickers, chunk=200):
    out = []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        print(f"  downloading {i + 1}-{i + len(part)} of {len(tickers)}")
        raw = yf.download(part, period=f"{max(WINDOWS) + 1}y", auto_adjust=True, threads=True, progress=False)["Close"]
        out.append(raw.to_frame(part[0]) if isinstance(raw, pd.Series) else raw)
    df = pd.concat(out, axis=1)
    return df.loc[:, ~df.columns.duplicated()]


def score(a, b, years):
    """Relative trend of a vs b over the last `years`. None if under MIN_YEARS of shared history."""
    r = (a / b).dropna()
    r = r[r.index >= r.index[-1] - pd.DateOffset(years=years)] if len(r) else r
    if len(r) < MIN_YEARS * 250:
        return None
    y = np.log(r.values)
    x = np.arange(len(y))
    fit = linregress(x, y)
    trend = np.exp(fit.slope * 252) - 1
    one = r.iloc[-1] / r.iloc[-253] - 1 if len(r) > 253 else np.nan
    w = r.resample("W-FRI").last().dropna()
    ma40 = w.rolling(40).mean().iloc[-1]
    status = ("Leader" if one >= 0 else "Fading") if trend >= 0 else ("Turning up" if one >= 0 else "Laggard")
    high52 = r.iloc[-5:].max() >= r.iloc[-252:].max()
    return {"trend": round(trend * 100, 2), "r2": round(fit.rvalue ** 2, 2), "one": round(one * 100, 1),
            "above": bool(w.iloc[-1] > ma40), "status": status, "high": bool(high52),
            "years": round(len(r) / 252, 1), "start": w.index[0].strftime("%Y-%m-%d"),
            "w": [float(f"{v / w.iloc[0] * 100:.4g}") for v in w.values],
            "fit": [round(float(np.exp(fit.intercept - y[0]) * 100), 3),
                    round(float(np.exp(fit.intercept + fit.slope * (len(y) - 1) - y[0]) * 100), 3)]}


def grid(px, funds, years):
    """Row vs column: long-term relative trend (%/yr) and last-12-month change of row / column."""
    out = {"t10": [], "y1": []}
    for a in funds:
        t10, y1 = [], []
        for b in funds:
            s = None if a == b else score(px[a], px[b], years)
            t10.append(None if s is None else s["trend"])
            y1.append(None if s is None else s["one"])
        out["t10"].append(t10)
        out["y1"].append(y1)
    return out


def leaders(px, stocks, years):
    """stocks: list of (ticker, name, sector fund). Top stocks per sector by relative trend vs that sector fund."""
    by = {}
    for t, name, fund in stocks:
        if t not in px or fund not in px:
            continue
        s = score(px[t], px[fund], years)
        if s and s["trend"] > 0 and s["r2"] >= MIN_STOCK_R2:
            s.update(t=t.replace(".TO", ""), n=name)
            s["w"] = [round(v, 2) for v in s["w"]]
            by.setdefault(fund, []).append(s)
    return {f: sorted(v, key=lambda s: -s["trend"])[:TOP_STOCKS] for f, v in by.items()}


def build_market(px, bench, sectors, industries, stocks, years):
    rows = []
    for f, name in sectors.items():
        s = score(px[f], px[bench], years)
        if s:
            s.update(t=f.replace(".TO", ""), n=name, key=f)
            rows.append(s)
    rows.sort(key=lambda s: -s["trend"])
    order = [r["key"] for r in rows]
    ind = []
    for f, (name, parent) in (industries or {}).items():
        s, sp = score(px[f], px[bench], years), score(px[f], px[parent], years)
        if s:
            # also vs XLK, the long-run US leader: does anything outside tech beat tech itself?
            s.update(t=f, n=name, parent=parent, vs_parent=sp, vs_xlk=score(px[f], px["XLK"], years) if f != "XLK" else None)
            ind.append(s)
    ind.sort(key=lambda s: -s["trend"])
    lead = leaders(px, stocks, years)
    return {"sectors": rows, "grid": grid(px, order, years), "order": [px_label(f, sectors) for f in order],
            "industries": ind, "leaders": [{"fund": f.replace(".TO", ""), "n": sectors.get(f, "Other"),
                                            "rows": lead.get(f, [])} for f in order if f in lead]}


def px_label(f, sectors):
    return {"t": f.replace(".TO", ""), "n": sectors[f]}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sp = load_sp500_sectors()
    us_stocks = [(s.replace(".", "-"), n, GICS_TO_US.get(g)) for s, n, g in
                 zip(sp["Symbol"], sp["Security"], sp["GICS Sector"]) if GICS_TO_US.get(g)]
    cdn = pd.read_csv(CDN_CSV, converters={"Ticker": lambda v: str(v).strip()})
    cdn = cdn[pd.to_numeric(cdn["Market Cap"], errors="coerce") >= CDN_MIN_CAP]
    cdn_stocks = [(t.upper().replace(".", "-") + ".TO", n, GICS_TO_CDN.get(g)) for t, n, g in
                  zip(cdn["Ticker"], cdn["Name"], cdn["Sector"]) if GICS_TO_CDN.get(g)]

    funds = ["SPY", "XIC.TO"] + list(US_SECTORS) + list(CDN_SECTORS) + list(US_INDUSTRIES)
    print(f"Downloading {len(funds)} funds, {len(us_stocks)} US and {len(cdn_stocks)} Canadian stocks...")
    px = download(list(dict.fromkeys(funds + [s[0] for s in us_stocks + cdn_stocks])))
    data = {str(y): {
        "cdn": build_market(px, "XIC.TO", CDN_SECTORS, None, cdn_stocks, y),
        "us": build_market(px, "SPY", US_SECTORS, US_INDUSTRIES, us_stocks, y),
    } for y in WINDOWS}
    asof = px["SPY"].dropna().index[-1].strftime("%B %d, %Y")
    html = (PAGE.replace("%%DATA%%", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
            .replace("%%LOGO%%", LOGO_B64).replace("%%ASOF%%", asof).replace("%%FULLSCREEN%%", FULLSCREEN_HTML))
    out = os.path.join(OUTPUT_DIR, "Relative_Strength.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relative Strength Ladder</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { background: #FFFFFF; color: #363636; font-family: Arial, sans-serif; font-size: 16px; line-height: 1.55; margin: 0; padding: 0 0 32px; }
  header { padding: 18px 16px 12px; border-bottom: 1px solid #E6E6E6; }
  header h1 { margin: 0 0 6px; font-size: 28px; }
  header .meta { color: #555555; font-size: 15px; }
  .intro { padding: 12px 16px 0; }
  .intro p { margin: 0 0 8px; }
  .intro .legend { font-size: 15px; color: #555555; margin-top: 16px; }
  .tabs { display: flex; flex-wrap: wrap; gap: 8px 16px; padding: 14px 16px 0; }
  .tabs span { display: flex; }
  .tabs button { border: 1px solid #CFCFCF; background: #FFFFFF; color: #363636; padding: 7px 22px; font-size: 16px; cursor: pointer; }
  .tabs button:first-child { border-radius: 4px 0 0 4px; } .tabs button:last-child { border-radius: 0 4px 4px 0; border-left: none; }
  .tabs button.on { background: #1F79BE; border-color: #1F79BE; color: #FFFFFF; }
  .section { padding: 22px 16px 0; }
  .section h2 { margin: 0; font-size: 23px; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }
  .section .sub { margin: 6px 0 0; color: #555555; font-size: 15px; max-width: 1100px; }
  .tbl { padding: 12px 16px 4px; overflow-x: auto; }
  .tbl table { border-collapse: collapse; font-size: 14px; width: 100%; min-width: 720px; }
  .tbl th { text-align: left; font-size: 12px; color: #555555; text-transform: uppercase; letter-spacing: .03em;
            padding: 6px 8px; border-bottom: 2px solid #C67A29; white-space: nowrap; }
  .tbl td { padding: 5px 8px; border-bottom: 1px solid #E6E6E6; white-space: nowrap; }
  .tbl tbody tr:nth-child(even) { background: #F7F7F7; }
  .tbl td.tk { font-weight: bold; color: #1F79BE; }
  .tbl td.num { text-align: right; }
  .tbl th.num { text-align: right; }
  .up { color: #44A660; font-weight: bold; } .down { color: #A22A2A; font-weight: bold; }
  .pill { display: inline-block; padding: 1px 9px; border-radius: 10px; font-size: 12px; font-weight: bold; color: #FFFFFF; }
  .s-Leader { background: #44A660; } .s-Fading { background: #C67A29; } .s-Turning { background: #1F79BE; } .s-Laggard { background: #A22A2A; }
  .hi { color: #C67A29; font-size: 12px; font-weight: bold; margin-left: 6px; }
  .note { color: #9A9A9A; font-size: 12px; }
  .multi { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px 16px; padding: 8px 16px 0; }
  .multi > div { border: 1px solid #E6E6E6; border-radius: 4px; }
  .ihead { margin: 16px 16px 0; font-size: 17px; } .ihead span { color: #555555; font-weight: normal; font-size: 14px; }
  .gridwrap { padding: 12px 16px 4px; overflow-x: auto; }
  .gbar { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; padding: 10px 16px 0; font-size: 14px; color: #555555; }
  .toggle button { border: 1px solid #CFCFCF; background: #FFFFFF; color: #363636; padding: 4px 12px; font-size: 13px; cursor: pointer; }
  .toggle button:first-child { border-radius: 4px 0 0 4px; } .toggle button:last-child { border-radius: 0 4px 4px 0; border-left: none; }
  .toggle button.on { background: #1F79BE; border-color: #1F79BE; color: #FFFFFF; }
  table.grid { border-collapse: separate; border-spacing: 2px; font-size: 13px; }
  table.grid th { font-size: 12px; color: #555555; font-weight: bold; padding: 4px 6px; white-space: nowrap; }
  table.grid th.rowh { text-align: right; }
  table.grid td { width: 58px; height: 34px; text-align: center; border-radius: 3px; font-weight: bold; }
  table.grid td.self { background: #F2F2F2; }
  table.grid td.score { background: #FFFFFF; color: #363636; border-left: 2px solid #C67A29; }
  .lead { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 440px), 1fr)); gap: 12px 20px; padding: 12px 16px 0; }
  .lead .box h4 { margin: 0 0 4px; font-size: 16px; } .lead .box h4 span { color: #555555; font-weight: normal; font-size: 14px; }
  .lead table { border-collapse: collapse; width: 100%; font-size: 13px; table-layout: fixed; }
  .lead col.c1 { width: 52px; } .lead col.c3 { width: 64px; } .lead col.c4 { width: 58px; } .lead col.c5 { width: 96px; }
  .lead td { padding: 4px 6px; border-bottom: 1px solid #E6E6E6; white-space: nowrap; }
  .lead td.nm { overflow: hidden; text-overflow: ellipsis; color: #555555; }
  .lead td svg { display: block; }
  .lead th { text-align: left; font-size: 11px; color: #555555; text-transform: uppercase; padding: 4px 6px; border-bottom: 2px solid #C67A29; }
  .source { color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }
</style>
</head>
<body>
<header>
  <h1>Relative Strength Ladder</h1>
  <div class="meta">Prices as of %%ASOF%% &middot; sectors, industries and stocks against their markets and each other</div>
</header>
<div class="intro">
  <p>Some parts of the market have beaten the rest for years, and long-running leadership tends to persist more often
  than it reverses. This page works down from the market to individual stocks: which sectors have beaten the index
  over the past ten or twenty years, which beat each other, which industries lead within them, and which stocks lead within
  each sector. Everything is measured as a ratio of one price to another, so a rising line means outperformance,
  not just a rising price.</p>
  <p class="legend">Trend / yr: how much faster (or slower) it has compounded than its benchmark each year, from a
  trend line fitted through ten or twenty years of the ratio (your choice below). Steadiness (R²): how closely the ratio has followed that line,
  from 0 to 1. Last 12 mo: the ratio's change over the past year. Status: <span class="pill s-Leader">Leader</span>
  long-term up and still rising, <span class="pill s-Fading">Fading</span> long-term up but down this year,
  <span class="pill s-Turning">Turning up</span> long-term down but up this year, <span class="pill s-Laggard">Laggard</span>
  down on both. <span class="hi">&#9650; new high</span> marks a ratio at a 12-month high in the past week.</p>
</div>
<div class="tabs"><span id="mt"><button data-m="cdn" class="on">Canada</button><button data-m="us">US</button></span><span id="yt"><button data-y="10" class="on">10 years</button><button data-y="20">20 years</button></span></div>
<div id="page"></div>
<div class="source">Source: 5i Research, Yahoo Finance. Total returns (dividends reinvested).</div>
<script>
const DATA = %%DATA%%;
const LOGO = "%%LOGO%%";
const BLUE = "#1F79BE", ORANGE = "#C67A29", INK = "#363636", MUTED = "#555555", GRID = "#E6E6E6", GREEN = "#44A660", RED = "#A22A2A";
const SC = { "Leader": GREEN, "Fading": ORANGE, "Turning up": BLUE, "Laggard": RED };
const AX = { gridcolor: GRID, zeroline: false, linecolor: "#CFCFCF", tickfont: { size: 11, color: MUTED } };
const esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
const pct = function (v, d) { return v === null || v === undefined || isNaN(v) ? "&mdash;" : '<span class="' + (v >= 0 ? "up" : "down") + '">' + (v >= 0 ? "+" : "") + v.toFixed(d === undefined ? 1 : d) + "%</span>"; };
const pill = function (s) { return '<span class="pill s-' + s.split(" ")[0] + '">' + s + "</span>"; };
const weeks = function (start, n) { const out = [], d0 = Date.parse(start + "T00:00:00Z"); for (let i = 0; i < n; i++) out.push(new Date(d0 + i * 7 * 864e5).toISOString().slice(0, 10)); return out; };
function spark(v, color, width) {
  const W = width || 110, H = 26, lo = Math.min.apply(null, v), hi = Math.max.apply(null, v), sp = hi - lo || 1;
  const pts = v.map(function (y, i) { return (i / (v.length - 1) * W).toFixed(1) + "," + (H - 2 - (y - lo) / sp * (H - 4)).toFixed(1); });
  return '<svg width="' + W + '" height="' + H + '" style="vertical-align:middle"><polyline fill="none" stroke="' + color +
         '" stroke-width="1.5" points="' + pts.join(" ") + '"/></svg>';
}
let MKT = "cdn", YRS = "10";
function short(s) { return s.years < YRS - 0.5 ? ' <span class="note">since ' + s.start.slice(0, 4) + "</span>" : ""; }

function render() {
  const m = MKT, D = DATA[YRS][m], span = YRS === "10" ? "a decade" : "two decades", bench = m === "cdn" ? "XIC" : "SPY", where = m === "cdn" ? "Canada" : "US";
  let h = "";
  // 1. Sectors vs index
  h += '<div class="section"><h2>' + where + " sectors vs. " + bench + '</h2><p class="sub">Ranked by ' + YRS + "-year relative trend. " +
       "A sector at the top has compounded faster than the index for " + span + "; the charts below show each ratio, with its trend line (dotted) and 40-week average (dashed).</p></div>";
  h += '<div class="tbl"><table><thead><tr><th>Fund</th><th>Sector</th><th class="num">Trend / yr</th><th class="num">Steadiness</th>' +
       '<th class="num">Last 12 mo</th><th>vs 40-wk avg</th><th>Status</th><th>5 years</th></tr></thead><tbody>' +
       D.sectors.map(function (s) {
         return '<tr><td class="tk">' + s.t + "</td><td>" + esc(s.n) + short(s) + '</td><td class="num">' + pct(s.trend) +
           '</td><td class="num">' + s.r2.toFixed(2) + '</td><td class="num">' + pct(s.one) + "</td><td>" + (s.above ? "Above" : "Below") +
           "</td><td>" + pill(s.status) + (s.high ? '<span class="hi">&#9650; new high</span>' : "") + "</td><td>" + spark(s.w.slice(-261), SC[s.status]) + "</td></tr>";
       }).join("") + "</tbody></table></div>";
  h += '<div class="multi">' + D.sectors.map(function (s, i) { return '<div id="mc-' + m + i + '"></div>'; }).join("") + "</div>";
  // 2. Grid
  h += '<div class="section"><h2>Which sector beats which</h2><p class="sub">Each cell compares the row sector with the column sector: green means the row ' +
       "has outperformed the column, red that it has lagged, by how much per year. Read across a row to see what a sector beats; down a column to see what beats it.</p></div>";
  h += '<div class="gbar"><span class="toggle" id="gt-' + m + '"><button data-k="t10" class="on">' + YRS + '-year trend</button><button data-k="y1">Last 12 months</button></span>' +
       "<span>Row vs. column, % per year (12-month view: total % change)</span></div>";
  h += '<div class="gridwrap" id="g-' + m + '"></div>';
  // 3. Industries (US)
  if (D.industries.length) {
    h += '<div class="section"><h2>Industries vs. SPY and vs. their sector</h2><p class="sub">An industry that beats both the market and its own sector is where leadership is ' +
         "concentrated (semiconductors within technology, for example). Ranked by trend vs SPY.</p></div>";
    h += '<div class="tbl"><table><thead><tr><th>Fund</th><th>Industry</th><th>Sector</th><th class="num">Trend / yr vs SPY</th><th class="num">Steadiness</th>' +
         '<th class="num">Last 12 mo</th><th>Status vs SPY</th><th class="num">Trend / yr vs sector</th><th>Status vs sector</th><th>5 years vs SPY</th></tr></thead><tbody>' +
         D.industries.map(function (s) {
           const vp = s.vs_parent || {};
           return '<tr><td class="tk">' + s.t + "</td><td>" + esc(s.n) + short(s) + "</td><td>" + s.parent + '</td><td class="num">' + pct(s.trend) +
             '</td><td class="num">' + s.r2.toFixed(2) + '</td><td class="num">' + pct(s.one) + "</td><td>" + pill(s.status) +
             (s.high ? '<span class="hi">&#9650;</span>' : "") + '</td><td class="num">' + pct(vp.trend) + "</td><td>" + (vp.status ? pill(vp.status) : "") +
             "</td><td>" + spark(s.w.slice(-261), SC[s.status]) + "</td></tr>";
         }).join("") + "</tbody></table></div>";
    h += '<div class="gbar"><span class="toggle" id="it-' + m + '"><button data-k="sector" class="on">vs own sector</button>' +
         '<button data-k="xlk">vs XLK</button><button data-k="spy">vs SPY</button></span><span>Each industry\'s ratio, with its trend line (dotted) and 40-week average (dashed); ' +
         "strongest first within each sector</span></div>";
    h += '<div id="ic-' + m + '"></div>';
  }
  // 4. Leading stocks per sector
  h += '<div class="section"><h2>Leading stocks in each sector</h2><p class="sub">Within each sector, the stocks with the strongest steady ' + YRS + "-year uptrend against " +
       "their sector fund (not the whole market), so a bank is judged against banks. Sectors in the order of the table above; " +
       (m === "cdn" ? "Canadian stocks over $1 billion" : "S&amp;P 500 members") + ". Sparklines show the ratio to the sector fund over five years.</p></div>";
  h += '<div class="gbar"><span class="toggle" id="lt-' + m + '"><button data-k="table" class="on">Table</button><button data-k="charts">Charts</button></span>' +
       '<span>Charts: each stock\'s ratio to its sector fund over ' + (YRS === "10" ? "ten" : "twenty") + " years" + ', with its trend line (dotted) and 40-week average (dashed)</span></div>';
  h += '<div id="lc-' + m + '" style="display:none"></div>';
  h += '<div class="lead" id="lb-' + m + '">' + D.leaders.map(function (g) {
    return '<div class="box"><h4>' + esc(g.n) + " <span>vs " + g.fund + "</span></h4><table><colgroup><col class=\"c1\"><col><col class=\"c3\"><col class=\"c4\"><col class=\"c5\"></colgroup>" +
      "<thead><tr><th>Stock</th><th></th><th>Trend/yr</th><th>12 mo</th><th>5 years</th></tr></thead><tbody>" +
      g.rows.map(function (s) {
        return '<tr><td class="tk"><b style="color:#1F79BE">' + s.t + '</b></td><td class="nm" title="' + esc(s.n) + '">' + esc(s.n) + (s.years < YRS - 0.5 ? ' <span class="note">(' + Math.round(s.years) + ' yrs)</span>' : "") + "</td><td>" + pct(s.trend) +
          "</td><td>" + pct(s.one) + "</td><td>" + spark(s.w.slice(-261), SC[s.status], 90) + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }).join("") + "</div>";
  document.getElementById("page").innerHTML = h;

  D.sectors.forEach(function (s, i) { mini("mc-" + m + i, s, s.t, s.n, bench); });
  if (D.industries.length) {
    drawIndustries(m, "sector");
    document.querySelectorAll("#it-" + m + " button").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll("#it-" + m + " button").forEach(function (x) { x.classList.toggle("on", x === b); });
        drawIndustries(m, b.dataset.k);
      });
    });
  }
  document.querySelectorAll("#lt-" + m + " button").forEach(function (b) {
    b.addEventListener("click", function () {
      document.querySelectorAll("#lt-" + m + " button").forEach(function (x) { x.classList.toggle("on", x === b); });
      const charts = b.dataset.k === "charts";
      document.getElementById("lb-" + m).style.display = charts ? "none" : "";
      document.getElementById("lc-" + m).style.display = charts ? "" : "none";
      if (charts) drawLeaderCharts(m);
    });
  });
  drawGrid(m, "t10");
  document.querySelectorAll("#gt-" + m + " button").forEach(function (b) {
    b.addEventListener("click", function () {
      document.querySelectorAll("#gt-" + m + " button").forEach(function (x) { x.classList.toggle("on", x === b); });
      drawGrid(m, b.dataset.k);
    });
  });
}

// Industry ratio charts, grouped under their sector (sectors in table order), against the sector fund or SPY
function drawIndustries(m, vs) {
  const D = DATA[YRS][m], box = document.getElementById("ic-" + m);
  box.querySelectorAll(".js-plotly-plot").forEach(function (d) { Plotly.purge(d); });
  let h = "", jobs = [];
  D.sectors.forEach(function (sec) {
    const pick = function (x) { return vs === "spy" ? x : vs === "xlk" ? x.vs_xlk : x.vs_parent; };
    const against = vs === "spy" ? "SPY" : vs === "xlk" ? "XLK" : sec.t;
    const inds = D.industries.filter(function (x) { return x.parent === sec.t && pick(x); });
    if (!inds.length) return;
    inds.sort(function (a, b) { return pick(b).trend - pick(a).trend; });
    h += '<h4 class="ihead">' + esc(sec.n) + " <span>industries vs " + against + "</span></h4><div class=\"multi\">";
    inds.forEach(function (x) {
      const id = "ic-" + m + "-" + x.t;
      h += '<div id="' + id + '"></div>';
      jobs.push([id, pick(x), x.t, x.n, against]);
    });
    h += "</div>";
  });
  box.innerHTML = h;
  jobs.forEach(function (j) { mini.apply(null, j); });
}

// Stock ratio charts, one group per sector; drawn as they near the screen (up to ~70 charts per market)
function drawLeaderCharts(m) {
  const box = document.getElementById("lc-" + m);
  if (box.dataset.done) return;
  box.dataset.done = "1";
  const jobs = {};
  box.innerHTML = DATA[YRS][m].leaders.map(function (g) {
    return '<h4 class="ihead">' + esc(g.n) + " <span>leading stocks vs " + g.fund + '</span></h4><div class="multi">' +
      g.rows.map(function (s) {
        const id = "lc-" + m + "-" + s.t.replace(/[^A-Za-z0-9]/g, "_");
        jobs[id] = [id, s, s.t, s.n, g.fund];
        return '<div id="' + id + '" style="min-height:210px"></div>';
      }).join("") + "</div>";
  }).join("");
  const io = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { io.unobserve(e.target); mini.apply(null, jobs[e.target.id]); } });
  }, { rootMargin: "600px 0px" });
  Object.keys(jobs).forEach(function (id) { io.observe(document.getElementById(id)); });
}

// 1-2-5 steps on a log axis (Plotly's default labels every minor step once a ratio spans 10x: 2, 3, 4 ... 1000)
function logTicks(v) {
  const lo = Math.min.apply(null, v.filter(isFinite)), hi = Math.max.apply(null, v.filter(isFinite)), t = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    [1, 2, 5].forEach(function (m) { const x = m * Math.pow(10, e); if (x >= lo * 0.9 && x <= hi * 1.1) t.push(x); });
  if (t.length < 3) return {};   // narrow range: Plotly's own ticks read fine
  return { tickvals: t, ticktext: t.map(function (x) { return x.toLocaleString("en-US"); }) };
}

function mini(id, s, t, n, bench) {
  {
    const x = weeks(s.start, s.w.length), ma = s.w.map(function (_, j) {
      if (j < 39) return null; let t = 0; for (let k = j - 39; k <= j; k++) t += s.w[k]; return t / 40; });
    Plotly.newPlot(id, [
      { x: x, y: s.w, mode: "lines", line: { color: SC[s.status], width: 1.8 }, hovertemplate: "%{y:.1f}<extra></extra>" },
      { x: x, y: ma, mode: "lines", line: { color: "#9A9A9A", width: 1, dash: "dash" }, hoverinfo: "skip" },
      { x: [x[0], x[x.length - 1]], y: s.fit, mode: "lines", line: { color: INK, width: 1, dash: "dot" }, hoverinfo: "skip" },
    ], { height: 210, margin: { t: 30, b: 26, l: 40, r: 10 }, showlegend: false, paper_bgcolor: "#FFF", plot_bgcolor: "#FFF",
         font: { family: "Arial", color: INK, size: 12 }, hovermode: "x unified",
         title: { text: "<b>" + t + "</b> " + esc(n) + " / " + bench + "  <span style='color:" + SC[s.status] + "'>" + (s.trend >= 0 ? "+" : "") + s.trend.toFixed(1) + "%/yr</span>",
                  x: 0.03, font: { size: 13 } },
         xaxis: Object.assign({ type: "date" }, AX), yaxis: Object.assign({ type: "log" }, logTicks(s.w.concat(s.fit)), AX) },
      { responsive: true, displaylogo: false, displayModeBar: false });
  }
}

function drawGrid(m, k) {
  const D = DATA[YRS][m], g = D.grid[k], o = D.order, cap = k === "t10" ? 8 : 30;
  const color = function (v) {
    if (v === null) return "#F2F2F2";
    const a = Math.min(Math.abs(v) / cap, 1), c = v >= 0 ? [68, 166, 96] : [162, 42, 42];
    return "rgba(" + c.join(",") + "," + (0.12 + a * 0.78).toFixed(2) + ")";
  };
  let h = '<table class="grid"><thead><tr><th></th>' + o.map(function (c) { return "<th title=\"" + esc(c.n) + "\">" + c.t + "</th>"; }).join("") +
          '<th>Beats</th></tr></thead><tbody>';
  o.forEach(function (r, i) {
    const wins = g[i].filter(function (v) { return v !== null && v > 0; }).length, n = g[i].filter(function (v) { return v !== null; }).length;
    h += '<tr><th class="rowh">' + r.t + ' <span style="font-weight:normal">' + esc(r.n) + "</span></th>" + g[i].map(function (v, j) {
      if (i === j) return '<td class="self"></td>';
      const a = v === null ? 0 : Math.min(Math.abs(v) / cap, 1);
      return '<td style="background:' + color(v) + ";color:" + (a > 0.45 ? "#FFF" : INK) + '" title="' + r.t + " vs " + o[j].t + '">' +
             (v === null ? "" : (v > 0 ? "+" : "") + v.toFixed(k === "t10" ? 1 : 0)) + "</td>";
    }).join("") + '<td class="score">' + wins + "/" + n + "</td></tr>";
  });
  document.getElementById("g-" + m).innerHTML = h + "</tbody></table>";
}

document.querySelectorAll(".tabs button").forEach(function (b) {
  b.addEventListener("click", function () {
    b.parentNode.querySelectorAll("button").forEach(function (x) { x.classList.toggle("on", x === b); });
    if (b.dataset.m) MKT = b.dataset.m; else YRS = b.dataset.y;
    render();
  });
});
render();
</script>
%%FULLSCREEN%%
</body>
</html>
"""


if __name__ == "__main__":
    main()
