"""
Website-ready Stock Lookup: members type a ticker (TSX, S&P 500 or Nasdaq 100) and get, for that stock:
  1. price vs its 200-week moving average (with % above/below)
  2. analyst revenue estimate revisions (FY1E-FY3E over 1W/1M/3M/6M/1Y), from the revision CSVs
  3. where it sits against its market's efficient frontier (TSX for Canadian stocks, S&P 500 for US)
  4. growth of $10,000 vs its index (XIC or SPY) over the last 10 years, dividends included
  5. up to 5 complements: stocks that have smoothed the ride when held 50/50 with it (complement_engine, the
     Complement Finder notebook's method; same index: TSX for Canadian stocks, S&P 500 for US), from
     outputs/stock-lookup/complements.json, which generate_stock_complements.py refreshes weekly

GitHub Pages is static and browsers can't pull from Yahoo, so everything is precomputed: one small JSON file per
stock in outputs/stock-lookup/data/ (weekly closes, revisions, risk stats), plus the page with the ticker list,
both index series and both frontiers embedded. The page fetches data/<ticker>.json when a stock is chosen and
draws the charts in the browser. Link to a stock directly with ?t=RY.TO.

Universe and frontier logic are shared with the Efficient Frontier page (generate_stock_screener's universes,
generate_efficient_frontier_web.compute_frontier). TSX symbols that don't resolve as <symbol>.TO are retried as
TSX Venture / CSE / Cboe Canada listings and with their dropped punctuation restored (QSPUN -> QSP-UN.TO).
Output: outputs/stock-lookup/Stock_Lookup.html + outputs/stock-lookup/data/*.json
"""

import json
import os
import re
import shutil
from datetime import datetime

import pandas as pd

from common_screening import load_nasdaq100_table, load_sp500_sectors
from generate_efficient_frontier_web import compute_frontier
from generate_index_rsi_web import LOGO_B64
from generate_rev_revision_screener import CDN_CSV_PATH, US_CSV_PATH, load_rev_csv
from generate_stock_screener import (
    DATA_YEARS, RISK_FREE, compute_signals, download_prices, load_universes, price_window_for_as_of,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "stock-lookup")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
COMPLEMENTS_PATH = os.path.join(OUTPUT_DIR, "complements.json")

HISTORY_YEARS = 15          # weekly chart history: 10 years shown with a 200-week average from the start
BENCH = {"cdn": ("XIC.TO", "XIC (TSX)"), "us": ("SPY", "SPY (S&P 500)")}
REV_KEYS = [f"{fy}_{w}" for fy in ("fy1", "fy2", "fy3") for w in ("1w", "1m", "3m", "6m", "1y")]


# ── Universe & prices ──────────────────────────────────────────────────────────
def norm(ticker: str) -> str:
    """Ticker key for matching across sources: no exchange suffix, no punctuation (BBD-B.TO, BBD.B, BBDB -> BBDB)."""
    t = re.sub(r"\.(TO|V|CN|NE)$", "", str(ticker).upper())
    return re.sub(r"[^A-Z0-9]", "", t)


def tsx_candidates(symbol: str) -> list:
    """Yahoo symbols to try for a TSX list entry that didn't resolve as-is (see generate_rev_revision_web)."""
    base = symbol.upper().replace(".TO", "")
    flat = base.replace("-", "")
    cands = [f"{flat}{sfx}" for sfx in (".V", ".CN", ".NE")]
    for unit in ("UN", "U", "A", "B"):
        if len(flat) > len(unit) and flat.endswith(unit):
            cands += [f"{flat[:-len(unit)]}-{unit}{sfx}" for sfx in (".TO", ".V")]
    return [c for c in cands if c != symbol]


def load_prices():
    universes = load_universes([])
    # The TSX list drops punctuation (BEPUN); data/TSX.csv keeps it (BEP.UN), so add those forms too
    tsx_extra = pd.read_csv(os.path.join(REPO_ROOT, "data", "TSX.csv"), converters={"Symbol": str})["Symbol"]
    tsx = list(dict.fromkeys(universes["TSX"] + [s.strip().replace(".", "-") + ".TO" for s in tsx_extra if s.strip()]))
    us = list(dict.fromkeys(universes["S&P 500"] + universes["NASDAQ-100"]))
    benches = [b for b, _ in BENCH.values()]

    print(f"Downloading {len(tsx) + len(us)} tickers, {HISTORY_YEARS}yr daily...")
    prices = download_prices(tsx + us + benches, HISTORY_YEARS)
    prices = prices.loc[:, prices.notna().any()]

    # Recover TSX names that didn't resolve, trying each fallback form as one batch across all of them
    missing = [s for s in tsx if s not in prices.columns]
    resolved = {s: s for s in tsx if s in prices.columns}
    print(f"  {len(missing)} TSX symbols unresolved; trying other exchanges / restored punctuation...")
    cand_lists = {s: tsx_candidates(s) for s in missing}
    for i in range(max((len(c) for c in cand_lists.values()), default=0)):
        batch = {cands[i]: s for s, cands in cand_lists.items() if s not in resolved and len(cands) > i
                 and cands[i] not in prices.columns}
        if not batch:
            continue
        got = download_prices(list(batch), HISTORY_YEARS)
        got = got.loc[:, got.notna().any()] if not got.empty else got
        for sym in got.columns:
            if batch.get(sym) and batch[sym] not in resolved:
                resolved[batch[sym]] = sym
                prices[sym] = got[sym]
    tsx_final = list(dict.fromkeys(resolved.values()))
    # Duplicates from the two TSX lists (e.g. BEPUN.TO failing and BEP-UN.TO working) collapse to one symbol
    print(f"  TSX usable: {len(tsx_final)} (recovered {len(tsx_final) - len([s for s in tsx if s in prices.columns])})")
    market = {**{s: "cdn" for s in tsx_final}, **{s: "us" for s in us if s in prices.columns}}
    return prices, market, {"cdn": tsx_final, "us": [s for s in universes["S&P 500"] if s in prices.columns]}


# ── Names, sectors, revisions ──────────────────────────────────────────────────
def load_info() -> dict:
    """(market, norm ticker) -> dict(name, sector, industry, rev) from the revision CSVs, with Wikipedia /
    Koyfin as fallbacks for names and sectors."""
    info = {}
    for m, path in (("us", US_CSV_PATH), ("cdn", CDN_CSV_PATH)):
        raw = pd.read_csv(path, converters={"Ticker": lambda v: str(v).strip()})
        for _, r in raw.iterrows():
            info[(m, norm(r["Ticker"]))] = dict(name=r.get("Name", ""), sector=r.get("Sector", ""),
                                               industry=r.get("Industry", ""))
        rev = load_rev_csv(path)
        for _, r in rev.iterrows():
            vals = [None if pd.isna(r.get(k)) else round(float(r.get(k)) * 100, 2) for k in REV_KEYS]
            if any(v is not None for v in vals):
                info.setdefault((m, norm(r["ticker"])), {})["rev"] = vals
        kpath = os.path.join(REPO_ROOT, "data", f"koyfin_{m}.csv")
        if os.path.exists(kpath):
            for _, r in pd.read_csv(kpath, converters={"Ticker": lambda v: str(v).strip()}).iterrows():
                d = info.setdefault((m, norm(r["Ticker"])), {})
                for k, col in (("name", "Name"), ("sector", "Sector"), ("industry", "Industry")):
                    if not d.get(k) and isinstance(r.get(col), str):
                        d[k] = r[col]
    try:
        for _, r in load_sp500_sectors().iterrows():
            d = info.setdefault(("us", norm(r["Symbol"])), {})
            d.setdefault("name", r["Security"])
            d.setdefault("sector", r["GICS Sector"])
        for _, r in load_nasdaq100_table().iterrows():
            d = info.setdefault(("us", norm(r.iloc[0])), {})
            d.setdefault("name", r.iloc[1])
    except Exception as e:
        print(f"  Wikipedia name lookup failed: {e}")
    return info


# ── Build ──────────────────────────────────────────────────────────────────────
def weekly(series: pd.Series) -> pd.Series:
    w = series.dropna().resample("W-FRI").last().ffill()
    return w.dropna()


# Windows device names: a file called PRN.TO.json (Parkland) can't be read or deleted normally on Windows, which
# hangs GitHub Desktop for anyone pulling the repo. Such stocks get a leading underscore; the page applies the
# same rule in dataFile().
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def data_filename(sym: str) -> str:
    return ("_" if sym.split(".")[0].upper() in WINDOWS_RESERVED else "") + f"{sym}.json"


def compact(values) -> list:
    return [float(f"{v:.5g}") for v in values]


def build():
    prices, market, universes = load_prices()
    as_of = prices.dropna(how="all").index.max()
    prices10 = price_window_for_as_of(prices, as_of)
    signals = compute_signals(prices10)
    returns = prices10.pct_change(fill_method=None)
    info = load_info()
    comps = {}
    if os.path.exists(COMPLEMENTS_PATH):
        with open(COMPLEMENTS_PATH, encoding="utf-8") as f:
            comp_file = json.load(f)
        comps = comp_file["results"]
        print(f"  complements from {comp_file['asof']} for {sum('top' in v for v in comps.values())} stocks")
    else:
        print("  no complements.json yet (generate_stock_complements.py); complements section will be empty")

    # Frontiers (TSX for Canadian stocks, S&P 500 for US) and each market's grey dots
    frontiers = {}
    for m, universe in universes.items():
        d, _, _, ef_vol, ef_ret = compute_frontier(signals, returns, universe)
        frontiers[m] = dict(
            dots=[[round(r["Vol%"], 1), round(r["AnnRet%"], 1), t] for t, r in d.iterrows()],
            vol=[round(v, 2) for v in (ef_vol or [])], ret=[round(v, 2) for v in (ef_ret or [])],
            label="TSX" if m == "cdn" else "S&P 500")
        print(f"  {frontiers[m]['label']} frontier: {len(frontiers[m]['vol'])} points, {len(d)} stocks")

    benches = {}
    for m, (sym, label) in BENCH.items():
        w = weekly(prices[sym])
        s = signals.loc[sym] if sym in signals.index else None
        benches[m] = dict(sym=sym, label=label, start=w.index[0].strftime("%Y-%m-%d"), c=compact(w.tolist()),
                          risk=None if s is None else [round(s["Vol%"], 1), round(s["AnnRet%"], 1)])

    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)   # drop files for tickers that have left the universe
    os.makedirs(DATA_DIR)
    index, n_rev = [], 0
    for sym, m in market.items():
        w = weekly(prices[sym])
        if len(w) < 60:
            continue
        meta = info.get((m, norm(sym)), {})
        name = meta.get("name") or sym
        s = signals.loc[sym] if sym in signals.index else None
        doc = dict(
            t=sym, n=name, m=m, sector=meta.get("sector") or "", industry=meta.get("industry") or "",
            start=w.index[0].strftime("%Y-%m-%d"), c=compact(w.tolist()),
            last=round(float(prices[sym].dropna().iloc[-1]), 2),
            risk=None if s is None else [round(s["Vol%"], 1), round(s["AnnRet%"], 1), round(s["Sharpe"], 2),
                                         round(s["NObs"] / 252, 1)],
            rev=meta.get("rev"),
            comp=comps.get(sym),
        )
        n_rev += doc["rev"] is not None
        with open(os.path.join(DATA_DIR, data_filename(sym)), "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))
        index.append([sym, name, m])
    index.sort(key=lambda r: (r[2] != "cdn", r[0]))   # Canada first in the type-ahead list
    print(f"  wrote {len(index)} stock files ({n_rev} with revision data)")
    return dict(index=index, bench=benches, frontier=frontiers, asof=as_of.strftime("%Y-%m-%d"),
                build=datetime.now().strftime("%Y%m%d%H%M"),
                rf=RISK_FREE * 100, years=DATA_YEARS)


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stock Lookup</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { background: #FFFFFF; color: #363636; font-family: Arial, sans-serif; font-size: 16px; line-height: 1.55; margin: 0; padding: 0 0 32px; }
  header { padding: 18px 16px 12px; border-bottom: 1px solid #E6E6E6; }
  header h1 { margin: 0 0 6px; font-size: 28px; }
  header .meta { color: #555555; font-size: 15px; }
  .intro { padding: 12px 16px 0; }
  .intro p { margin: 0 0 8px; }
  .search { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 14px 0 4px; }
  .search input { flex: 0 1 340px; border: 1px solid #CFCFCF; border-radius: 4px; padding: 8px 12px; font-size: 16px; color: #363636; }
  .search button { background: #1F79BE; color: #FFFFFF; border: none; border-radius: 4px; padding: 9px 18px; font-size: 16px; cursor: pointer; }
  .chips { font-size: 14px; color: #555555; }
  .chips a { color: #1F79BE; cursor: pointer; margin-right: 10px; text-decoration: none; }
  .chips a:hover { text-decoration: underline; }
  #status { color: #A22A2A; font-size: 15px; min-height: 22px; }
  .card { margin: 8px 16px 0; padding: 14px 16px; border: 1px solid #E6E6E6; border-left: 4px solid #C67A29; border-radius: 4px; }
  .card h2 { margin: 0 0 2px; font-size: 24px; }
  .card .sub { color: #555555; font-size: 15px; }
  .stats { display: flex; flex-wrap: wrap; gap: 8px 28px; margin-top: 10px; }
  .stat .k { color: #555555; font-size: 13px; text-transform: uppercase; letter-spacing: .03em; }
  .stat .v { font-size: 19px; font-weight: bold; }
  .up { color: #44A660; } .down { color: #A22A2A; }
  .section { padding: 16px 16px 0; }
  .section h3 { margin: 0; font-size: 21px; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }
  .section p { margin: 8px 0 0; color: #555555; font-size: 15px; }
  .empty { color: #555555; padding: 16px; font-size: 15px; }
  #comp { padding: 12px 16px 8px; overflow-x: auto; }
  #comp table { border-collapse: collapse; font-size: 15px; min-width: 640px; width: 100%; max-width: 1000px; }
  #comp th { text-align: left; font-size: 12px; color: #555555; text-transform: uppercase; letter-spacing: .03em;
             padding: 6px 10px; border-bottom: 2px solid #C67A29; white-space: nowrap; }
  #comp th.grp { text-align: center; border-bottom: 1px solid #CFCFCF; }
  #comp td { padding: 7px 10px; border-bottom: 1px solid #E6E6E6; white-space: nowrap; }
  #comp td.num { text-align: center; font-weight: bold; }
  #comp tbody tr:nth-child(even) { background: #F7F7F7; }
  #comp a { color: #1F79BE; font-weight: bold; cursor: pointer; text-decoration: none; }
  #comp a:hover { text-decoration: underline; }
  #comp .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 7px; vertical-align: -1px; }
  #comp .note { color: #555555; font-size: 13px; margin: 10px 0 0; line-height: 1.5; max-width: 1000px; }
  .source { color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }
  #results { display: none; }
</style>
</head>
<body>
<header>
  <h1>Stock Lookup</h1>
  <div class="meta">Prices as of %%ASOF%% &middot; TSX, S&amp;P 500 and Nasdaq 100 stocks</div>
</header>
<div class="intro">
  <p>Enter a ticker or company name to see a stock's long-term trend, what analysts are doing with their revenue
  forecasts, how its risk and return compare with the rest of its market, and whether it has beaten simply owning
  the index.</p>
  <div class="search">
    <input id="q" list="tickers" placeholder="Ticker or company, e.g. RY, SHOP.TO, AAPL" autocomplete="off">
    <button onclick="lookup()">Look up</button>
    <datalist id="tickers"></datalist>
  </div>
  <div class="chips">Try: <a data-t="RY.TO">RY.TO</a><a data-t="SHOP.TO">SHOP.TO</a><a data-t="CNQ.TO">CNQ.TO</a><a data-t="AAPL">AAPL</a><a data-t="NVDA">NVDA</a><a data-t="JPM">JPM</a></div>
  <div id="status"></div>
</div>
<div id="results">
  <div class="card">
    <h2 id="s-name"></h2>
    <div class="sub" id="s-sub"></div>
    <div class="stats" id="s-stats"></div>
  </div>
  <div class="section"><h3>Price vs. 200-week moving average</h3>
    <p>The 200-week average (about four years) smooths out short-term swings and shows the long-term trend. Stocks in
    healthy uptrends tend to stay above it; the lower panel shows how far above or below it the stock is.</p></div>
  <div id="c-ma"></div>
  <div class="section"><h3>Analyst revenue revisions</h3>
    <p>How much analysts have changed their revenue forecasts for each of the next three fiscal years over the past
    week, month, three months, six months and year. Bars above zero mean estimates are rising.</p></div>
  <div id="c-rev"></div>
  <div class="section"><h3>Risk vs. return: <span id="ef-market"></span> efficient frontier</h3>
    <p>Every stock in the index plotted by annual return (up) against volatility (right) over the last %%YEARS%% years.
    The orange line is the efficient frontier, the best return that mixing the index's highest-Sharpe stocks has
    delivered for each level of risk. Closer to the line, or above it, is better. The chart focuses on the main body
    of the market; double-click it to zoom out to every stock.</p></div>
  <div id="c-ef"></div>
  <div class="section"><h3>Growth of $10,000 vs. the index</h3>
    <p>What $10,000 invested would be worth today in this stock versus the index ETF, with dividends reinvested.
    The simplest test of whether owning the stock has paid off compared with owning the market.</p></div>
  <div id="c-growth"></div>
  <div class="section"><h3>Stocks that have complemented <span id="comp-for"></span></h3>
    <p>High-quality stocks that, held 50/50 with this one, would have made for a smoother ride: lower volatility and
    smaller drops, without giving up much return. The chart below the table shows the difference, most visibly in the
    shaded sell-off. Each has beaten its own index, made money in every part of the
    period, moved differently from this stock, and held up better in most major sell-offs. One per sector, from the
    same index: TSX stocks for Canadian names, S&amp;P 500 stocks for US names. Click a ticker to look it up.</p></div>
  <div id="comp"></div>
  <div id="c-comp"></div>
</div>
<div class="source">Source: 5i Research, analyst revenue estimates, Yahoo Finance. For TSX, S&amp;P 500 and Nasdaq 100 members.</div>
<script>
const META = %%META%%;
const LOGO = "%%LOGO%%";
const BLUE = "#1F79BE", ORANGE = "#C67A29", GREEN = "#44A660", RED = "#A22A2A", INK = "#363636", MUTED = "#555555",
      GRID = "#E6E6E6", GOLD = "#E8B84B";
const CFG = { responsive: true, displaylogo: false };
const AX = { gridcolor: GRID, zeroline: false, linecolor: "#CFCFCF", tickfont: { size: 12, color: MUTED } };
// Same rule as data_filename() in the script: Windows device names (PRN, CON, AUX, NUL, COM1-9, LPT1-9) get "_"
const RESERVED = /^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$/;
// ?v= build stamp: browsers (and GitHub Pages) cache the data files, so without it a returning visitor can get
// yesterday's file, or one from before a new section was added, and that section silently stays empty
function dataFile(t) {
  return "data/" + (RESERVED.test(t.split(".")[0]) ? "_" : "") + encodeURIComponent(t) + ".json?v=" + META.build;
}
const BY_SYM = {};
META.index.forEach(function (r) { BY_SYM[r[0]] = r; });

// Type-ahead list: value is the ticker, label the company name
(function () {
  const dl = document.getElementById("tickers");
  dl.innerHTML = META.index.map(function (r) {
    return '<option value="' + r[0] + '">' + r[1].replace(/"/g, "&quot;") + '</option>';
  }).join("");
})();

function resolve(q) {
  q = q.trim();
  if (!q) return null;
  const up = q.toUpperCase().replace(/\s+/g, "");
  const dash = up.replace(/\./g, "-");
  const cands = [up, dash, up + ".TO", dash + ".TO", dash.replace(/-TO$/, ".TO"), dash.replace(/-TO$/, "") + ".TO"];
  for (let i = 0; i < cands.length; i++) if (BY_SYM[cands[i]]) return cands[i];
  // Company name: prefix match first, then anywhere
  const lq = q.toLowerCase();
  let hit = META.index.find(function (r) { return r[1].toLowerCase().indexOf(lq) === 0; }) ||
            META.index.find(function (r) { return r[1].toLowerCase().indexOf(lq) >= 0; });
  return hit ? hit[0] : null;
}

function dates(start, n) {
  const out = [], d0 = Date.parse(start + "T00:00:00Z");
  for (let i = 0; i < n; i++) out.push(new Date(d0 + i * 7 * 864e5).toISOString().slice(0, 10));
  return out;
}

// Logo top-right in the title row, ~34px tall whatever the chart height (plotH = height - top - bottom margins)
function logo(plotH, top) {
  return [{ source: LOGO, xref: "paper", yref: "paper", x: 1, y: 1 + (top - 10) / plotH, sizex: 0.16,
            sizey: 34 / plotH, xanchor: "right", yanchor: "top" }];
}

// 1-2-5 ticks for a log price axis (Plotly's defaults label every minor step: 3, 2, 100, 9, 8...)
function logTicks(lo, hi) {
  const t = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    [1, 2, 5].forEach(function (m) { const v = m * Math.pow(10, e); if (v >= lo * 0.9 && v <= hi * 1.1) t.push(v); });
  if (t.length < 3) {   // narrow range: plain steps instead
    const raw = (hi - lo) / 5, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag;
    t.length = 0;
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) t.push(v);
  }
  return t;
}
function nums(a) { return a.filter(function (v) { return v !== null && isFinite(v); }); }

function fmtPct(v, d) { return (v >= 0 ? "+" : "") + v.toFixed(d === undefined ? 1 : d) + "%"; }

function lookup(sym) {
  const q = sym || document.getElementById("q").value;
  const t = resolve(q);
  const st = document.getElementById("status");
  if (!t) {
    st.textContent = q.trim() ? '"' + q.trim() + '" isn\'t in our coverage (TSX, S&P 500 and Nasdaq 100 stocks). ' +
      "Try the ticker, e.g. RY or RY.TO, or pick from the list." : "";
    return;
  }
  st.textContent = "Loading " + t + "…";
  fetch(dataFile(t)).then(function (r) {
    if (!r.ok) throw new Error(r.status);
    return r.json();
  }).then(function (d) {
    st.textContent = "";
    document.getElementById("q").value = t;
    try { history.replaceState(null, "", "?t=" + encodeURIComponent(t)); } catch (e) {}
    render(d);
  }).catch(function () { st.textContent = "Couldn't load data for " + t + "."; });
}

function render(d) {
  document.getElementById("results").style.display = "block";
  const bench = META.bench[d.m], fr = META.frontier[d.m];
  const x = dates(d.start, d.c.length);

  // 200-week moving average and % vs it
  const ma = d.c.map(function (_, i) {
    if (i < 199) return null;
    let s = 0; for (let j = i - 199; j <= i; j++) s += d.c[j]; return s / 200;
  });
  const dist = d.c.map(function (v, i) { return ma[i] ? (v / ma[i] - 1) * 100 : null; });
  const lastDist = dist[dist.length - 1];

  // Summary card
  document.getElementById("s-name").textContent = d.n + " (" + d.t + ")";
  document.getElementById("s-sub").textContent = [d.sector, d.industry].filter(Boolean).join(" · ") ||
    (d.m === "cdn" ? "Canada" : "United States");
  const stats = [["Last price", "$" + d.last.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }), ""]];
  if (lastDist !== null) stats.push(["vs. 200-week avg", fmtPct(lastDist), lastDist >= 0 ? "up" : "down"]);
  if (d.risk) {
    const yrs = Math.min(d.risk[3], META.years);
    stats.push(["Return / yr (" + (yrs >= META.years - 0.5 ? META.years : yrs.toFixed(1)) + "y)", fmtPct(d.risk[1]), d.risk[1] >= 0 ? "up" : "down"]);
    stats.push(["Volatility", d.risk[0].toFixed(1) + "%", ""]);
    stats.push(["Sharpe ratio", d.risk[2].toFixed(2), ""]);
  }
  document.getElementById("s-stats").innerHTML = stats.map(function (s) {
    return '<div class="stat"><div class="k">' + s[0] + '</div><div class="v ' + s[2] + '">' + s[1] + "</div></div>";
  }).join("");

  // 1. Price vs 200-week MA
  const below = dist.map(function (v) { return v !== null && v < 0 ? 5 : 0; });
  const pv = nums(d.c.concat(ma)), dv = nums(dist);
  const pLo = Math.min.apply(null, pv), pHi = Math.max.apply(null, pv);
  const dLo = Math.min(0, Math.min.apply(null, dv)), dHi = Math.max(0, Math.max.apply(null, dv)), dPad = (dHi - dLo) * 0.08 || 5;
  Plotly.react("c-ma", [
    { x: x, y: d.c, mode: "lines+markers", name: "Weekly close", line: { color: BLUE, width: 2 },
      marker: { size: below, color: GREEN }, hovertemplate: "Close: $%{y:,.2f}<extra></extra>" },
    { x: x, y: ma, mode: "lines", name: "200-week average", line: { color: ORANGE, width: 2 },
      hovertemplate: "200-wk avg: $%{y:,.2f}<extra></extra>" },
    { x: x, y: dist, mode: "lines", yaxis: "y2", line: { color: "#7A7A7A", width: 1.6 }, showlegend: false,
      hovertemplate: "vs avg: %{y:+.1f}%<extra></extra>" },
  ], {
    height: 600, margin: { t: 60, b: 80, l: 70, r: 24 }, paper_bgcolor: "#FFF", plot_bgcolor: "#FFF",
    font: { family: "Arial", color: INK, size: 13 }, hovermode: "x unified",
    title: { text: "<b>" + d.t + "</b>  ·  price vs. 200-week average", x: 0.02, font: { size: 18 } },
    legend: { orientation: "h", x: 0, y: -0.1, yanchor: "top", font: { size: 12 } },
    xaxis: Object.assign({ anchor: "y2", type: "date" }, AX),
    yaxis: Object.assign({ domain: [0.34, 1], type: "log", tickvals: logTicks(pLo, pHi), tickformat: "$,.0f",
                           title: { text: "Price", font: { size: 13 } } }, AX),
    yaxis2: Object.assign({ domain: [0, 0.26], range: [dLo - dPad, dHi + dPad], ticksuffix: "%",
                            title: { text: "% vs avg", font: { size: 13 } } }, AX),
    shapes: [{ type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y2", y0: -1000, y1: 0, fillcolor: GREEN,
               opacity: 0.07, line: { width: 0 }, layer: "below" },
             { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y2", y0: 0, y1: 0, line: { color: GREEN, dash: "dash", width: 1.2 } }],
    images: logo(600 - 60 - 80, 60),
  }, CFG);

  // 2. Revisions
  const rev = document.getElementById("c-rev");
  if (d.rev) {
    const wins = ["1W", "1M", "3M", "6M", "1Y"], cols = [BLUE, "#4B8EA9", ORANGE];
    Plotly.react(rev, [0, 1, 2].map(function (k) {
      return { type: "bar", x: wins, y: d.rev.slice(k * 5, k * 5 + 5), name: "FY" + (k + 1) + "E",
               marker: { color: cols[k] }, hovertemplate: "%{x}: %{y:+.2f}%<extra>FY" + (k + 1) + "E</extra>" };
    }), {
      height: 420, margin: { t: 60, b: 80, l: 70, r: 24 }, paper_bgcolor: "#FFF", plot_bgcolor: "#FFF",
      font: { family: "Arial", color: INK, size: 13 }, barmode: "group",
      title: { text: "<b>" + d.t + "</b>  ·  revenue estimate revisions", x: 0.02, font: { size: 18 } },
      legend: { orientation: "h", x: 0, y: -0.12, yanchor: "top", traceorder: "normal" },
      xaxis: Object.assign({ type: "category" }, AX),
      yaxis: Object.assign({ ticksuffix: "%", zeroline: true, zerolinecolor: "#9A9A9A" }, AX),
      images: logo(420 - 60 - 80, 60),
    }, CFG);
  } else {
    Plotly.purge(rev);
    rev.innerHTML = '<div class="empty">No analyst revenue revision data for ' + d.t + ".</div>";
  }

  // 3. Efficient frontier
  document.getElementById("ef-market").textContent = fr.label;
  const efDiv = document.getElementById("c-ef");
  const tr = [
    { x: fr.dots.map(function (p) { return p[0]; }), y: fr.dots.map(function (p) { return p[1]; }), mode: "markers",
      name: fr.label + " stocks", marker: { color: "#B5B5B5", size: 6, opacity: 0.55 },
      text: fr.dots.map(function (p) { return p[2]; }),
      hovertemplate: "<b>%{text}</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%/yr<extra></extra>" },
  ];
  if (fr.vol.length) tr.push({ x: fr.vol, y: fr.ret, mode: "lines", name: "Efficient frontier",
    line: { color: ORANGE, width: 3 }, hovertemplate: "Frontier<br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%/yr<extra></extra>" });
  if (bench.risk) tr.push({ x: [bench.risk[0]], y: [bench.risk[1]], mode: "markers+text", name: bench.label,
    marker: { color: BLUE, size: 13, symbol: "square", line: { color: "#FFF", width: 1 } }, text: [bench.sym],
    textposition: "bottom center", textfont: { color: BLUE, size: 12 },
    hovertemplate: "<b>" + bench.label + "</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%/yr<extra></extra>" });
  if (d.risk) tr.push({ x: [d.risk[0]], y: [d.risk[1]], mode: "markers+text", name: d.t,
    marker: { color: GOLD, size: 18, symbol: "diamond", line: { color: INK, width: 1.2 } }, text: [d.t],
    textposition: "top center", textfont: { color: INK, size: 14 },
    hovertemplate: "<b>" + d.t + "</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%/yr<extra></extra>" });
  // Axes zoom on the main body of the market (95th percentile) plus the stock and the index. The frontier and a
  // few very volatile names (e.g. TSX junior miners at 200-350% volatility) run off the edge; double-click shows all.
  const vs = fr.dots.map(function (p) { return p[0]; }).sort(function (a, b) { return a - b; });
  const rs = fr.dots.map(function (p) { return p[1]; }).sort(function (a, b) { return a - b; });
  const q = function (a, p) { return a[Math.min(a.length - 1, Math.floor(p * a.length))]; };
  const xs = [q(vs, 0.95)].concat(bench.risk ? [bench.risk[0]] : [], d.risk ? [d.risk[0]] : []);
  const ys = [q(rs, 0.95)].concat(bench.risk ? [bench.risk[1]] : [], d.risk ? [d.risk[1]] : []);
  const ylo = Math.min(q(rs, 0.01), 0, d.risk ? d.risk[1] : 0), yhi = Math.max.apply(null, ys), pad = (yhi - ylo) * 0.07;
  Plotly.react(efDiv, tr, {
    height: 560, margin: { t: 60, b: 56, l: 70, r: 24 }, paper_bgcolor: "#FFF", plot_bgcolor: "#FFF",
    font: { family: "Arial", color: INK, size: 13 },
    title: { text: "<b>" + d.t + "</b>  ·  vs. the " + fr.label + " efficient frontier", x: 0.02, font: { size: 18 } },
    legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.85)", bordercolor: GRID, borderwidth: 1, font: { size: 12 } },
    // Explicit linear axes: left to guess, Plotly picked "category" here and scrambled the chart
    xaxis: Object.assign({ type: "linear", range: [0, Math.max.apply(null, xs) * 1.08], ticksuffix: "%",
                           title: { text: "Volatility (annualized)", font: { size: 13 } } }, AX),
    yaxis: Object.assign({ type: "linear", range: [ylo - pad, yhi + pad], ticksuffix: "%",
                           title: { text: "Return (annualized, last " + META.years + " years)", font: { size: 13 } } }, AX),
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: META.rf, y1: META.rf, line: { color: BLUE, dash: "dot", width: 1.2 } }],
    annotations: [{ xref: "paper", x: 1, y: META.rf, text: "Risk-free rate (" + META.rf.toFixed(1) + "%)", showarrow: false,
                    xanchor: "right", yanchor: "top", font: { size: 11, color: BLUE } }],
    images: logo(560 - 60 - 56, 60),
  }, CFG);

  renderComplements(d);

  // 4. Growth of $10,000 vs the index, last 10 years (or since the stock's data begins)
  const bx = dates(bench.start, bench.c.length), bmap = {};
  bx.forEach(function (dt, i) { bmap[dt] = bench.c[i]; });
  const cutoff = new Date(Date.parse(x[x.length - 1]) - 10 * 365.25 * 864e5).toISOString().slice(0, 10);
  let i0 = x.findIndex(function (dt) { return dt >= cutoff && bmap[dt] !== undefined; });
  if (i0 < 0) i0 = 0;
  const gx = [], gs = [], gb = [];
  for (let i = i0; i < x.length; i++) {
    if (bmap[x[i]] === undefined) continue;
    gx.push(x[i]); gs.push(10000 * d.c[i] / d.c[i0]); gb.push(10000 * bmap[x[i]] / bmap[x[i0]]);
  }
  const endS = gs[gs.length - 1], endB = gb[gb.length - 1];
  const money = function (v) { return "$" + Math.round(v).toLocaleString("en-US"); };
  Plotly.react("c-growth", [
    { x: gx, y: gs, mode: "lines", name: d.t + ": " + money(endS), line: { color: BLUE, width: 2.4 },
      hovertemplate: d.t + ": $%{y:,.0f}<extra></extra>" },
    { x: gx, y: gb, mode: "lines", name: bench.label + ": " + money(endB), line: { color: "#9A9A9A", width: 2 },
      hovertemplate: bench.sym + ": $%{y:,.0f}<extra></extra>" },
  ], {
    height: 440, margin: { t: 60, b: 40, l: 76, r: 24 }, paper_bgcolor: "#FFF", plot_bgcolor: "#FFF",
    font: { family: "Arial", color: INK, size: 13 }, hovermode: "x unified",
    title: { text: "<b>" + d.t + "</b>  ·  growth of $10,000 since " + gx[0].slice(0, 4) + " vs. " + bench.sym +
                   "  ·  <span style='color:" + (endS >= endB ? GREEN : RED) + "'><b>" +
                   (endS >= endB ? "ahead of" : "behind") + " the index</b></span>", x: 0.02, font: { size: 18 } },
    legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.85)", bordercolor: GRID, borderwidth: 1 },
    xaxis: Object.assign({ type: "date" }, AX),
    yaxis: Object.assign({ type: "log", tickvals: logTicks(Math.min.apply(null, gs.concat(gb)), Math.max.apply(null, gs.concat(gb))),
                           tickformat: "$,.0f" }, AX),
    images: logo(440 - 60 - 40, 60),
  }, CFG);
}

// 5. Complements table: change from holding the stock alone to a 50/50 blend, in percentage points
const COMP_COLORS = [BLUE, ORANGE, GREEN, "#8E6AC8", "#4B8EA9"];

// Growth of $10,000 alone vs each 50/50 blend (top), drawdowns (bottom). The top two blends show by default; the
// rest are a legend click away, since six lines at once is hard to read.
function renderComplementChart(d) {
  const div = document.getElementById("c-comp"), c = d.comp;
  if (!c || !c.top || !c.paths) { Plotly.purge(div); div.innerHTML = ""; return; }
  const p = c.paths, xs = dates(p.start, p.alone.length);
  const dd = function (v) { let m = -Infinity; return v.map(function (y) { m = Math.max(m, y); return (y / m - 1) * 100; }); };
  const series = [{ name: d.t + " alone", y: p.alone, color: INK, width: 2.8, vis: true }].concat(
    c.top.map(function (x, i) {
      return { name: "50/50 with " + x.t, y: p.blends[i], color: COMP_COLORS[i], width: 1.9, vis: i < 2 ? true : "legendonly" };
    }));
  const tr = [];
  series.forEach(function (s, i) {
    tr.push({ x: xs, y: s.y, mode: "lines", name: s.name, legendgroup: "g" + i, visible: s.vis,
              line: { color: s.color, width: s.width }, hovertemplate: s.name + ": $%{y:,.0f}<extra></extra>" });
    tr.push({ x: xs, y: dd(s.y), mode: "lines", yaxis: "y2", legendgroup: "g" + i, showlegend: false, visible: s.vis,
              line: { color: s.color, width: s.width * 0.8 }, hovertemplate: s.name + ": %{y:.0f}%<extra></extra>" });
  });
  const all = [].concat.apply([], series.map(function (s) { return s.y; }));
  const shapes = [], annotations = [];
  if (p.worst_dd) {
    shapes.push({ type: "rect", xref: "x", yref: "paper", x0: p.worst_dd[0], x1: p.worst_dd[1], y0: 0, y1: 1,
                  fillcolor: "#9A9A9A", opacity: 0.13, line: { width: 0 }, layer: "below" });
    annotations.push({ xref: "x", yref: "paper", x: p.worst_dd[0], y: 1, xanchor: "left", yanchor: "bottom", showarrow: false,
                       text: d.t + " worst drawdown", font: { size: 11, color: MUTED } });
  }
  const H = 620, T = 60, B = 90;
  Plotly.react(div, tr, {
    height: H, margin: { t: T, b: B, l: 76, r: 24 }, paper_bgcolor: "#FFF", plot_bgcolor: "#FFF",
    font: { family: "Arial", color: INK, size: 13 }, hovermode: "x unified",
    title: { text: "<b>" + d.t + "</b>  ·  growth of $10,000 alone vs. 50/50 with each complement", x: 0.02, font: { size: 18 } },
    legend: { orientation: "h", x: 0, y: -0.09, yanchor: "top", font: { size: 12 } },
    xaxis: Object.assign({ anchor: "y2", type: "date" }, AX),
    yaxis: Object.assign({ domain: [0.36, 1], type: "log", tickformat: "$,.0f",
                           tickvals: logTicks(Math.min.apply(null, all), Math.max.apply(null, all)) }, AX),
    yaxis2: Object.assign({ domain: [0, 0.28], ticksuffix: "%", title: { text: "Drawdown", font: { size: 13 } } }, AX),
    shapes: shapes, annotations: annotations, images: logo(H - T - B, T),
  }, CFG);
}

function renderComplements(d) {
  renderComplementChart(d);
  const box = document.getElementById("comp");
  document.getElementById("comp-for").textContent = d.n;
  const c = d.comp;
  if (!c) { box.innerHTML = '<div class="empty">Complements are not available for ' + d.t + ' yet.</div>'; return; }
  if (!c.top) {
    // Reasons are either "no stock passed all the tests" or "only 3.2 years of history (at least 5 needed)"
    box.innerHTML = '<div class="empty">' + (/^only/.test(c.reason || "")
      ? "Complements need at least five years of price history; " + d.t + " has " + c.reason.replace(/^only /, "").replace(/ \(.*$/, "") + "."
      : "No stock passed all of the complement tests for " + d.t + ".") + "</div>";
    return;
  }
  const FLAT = 0.005;
  const cell = function (delta, betterUp) {
    if (Math.abs(delta) < FLAT) return '<td class="num" style="color:' + MUTED + '">&asymp; same</td>';
    const good = (delta > 0) === betterUp;
    return '<td class="num" style="color:' + (good ? GREEN : RED) + '">' + (delta > 0 ? "&#9650; " : "&#9660; ") +
           (Math.abs(delta) * 100).toFixed(1) + " pts</td>";
  };
  const a = c.alone, pc = function (v) { return (v * 100).toFixed(1) + "%"; };
  const rows = c.top.map(function (x, i) {
    const link = BY_SYM[x.t] ? '<a data-t="' + x.t + '">' + x.t + "</a>" : x.t;
    return '<tr><td><span class="swatch" style="background:' + COMP_COLORS[i] + '"></span>' + x.n + " (" + link + ")</td><td>" + x.sector +
      '</td><td class="num" style="font-weight:normal">' + x.corr.toFixed(2) + "</td>" +
      cell(x.blend[0] - a[0], true) + cell(x.blend[1] - a[1], false) + cell(a[2] - x.blend[2], false) + "</tr>";
  }).join("");
  const span = c.start.slice(0, 4) + "&ndash;" + c.end.slice(0, 4);
  box.innerHTML = '<table><thead><tr><th rowspan="2">Complement</th><th rowspan="2">Sector</th>' +
    '<th rowspan="2">Correlation</th><th class="grp" colspan="3">50/50 blend vs. ' + d.t + ' alone</th></tr>' +
    "<tr><th>Return</th><th>Volatility</th><th>Max drawdown</th></tr></thead><tbody>" + rows + "</tbody></table>" +
    '<p class="note">' + d.t + " alone, " + span + ": " + pc(a[0]) + " a year, " + pc(a[1]) + " volatility, " + pc(a[2]) +
    " worst drawdown. The chart below shows " + d.t + " and its top two blends, in the colours above; click a name in its legend to add or hide the others " +
    "(weekly closes, so drops can look a little shallower than the daily figures in the table). Changes are in percentage points from holding " + d.t + " alone to a 50/50 mix rebalanced every quarter: " +
    "green is better (higher return, lower volatility, smaller worst drop), &asymp; is within 0.5 points. Correlation is the " +
    "highest of three sub-periods (lower means the two moved more independently). Total returns with dividends reinvested, in " +
    "Canadian dollars. Backtested and hypothetical, not a recommendation.</p>";
  box.querySelectorAll("a[data-t]").forEach(function (el) {
    el.addEventListener("click", function () { lookup(el.dataset.t); window.scrollTo({ top: 0, behavior: "smooth" }); });
  });
}

document.querySelectorAll(".chips a").forEach(function (a) { a.addEventListener("click", function () { lookup(a.dataset.t); }); });
document.getElementById("q").addEventListener("keydown", function (e) { if (e.key === "Enter") lookup(); });
(function () {
  const t = new URLSearchParams(location.search).get("t");
  if (t) lookup(t);
})();
</script>
</body>
</html>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta = build()
    html = (PAGE.replace("%%META%%", json.dumps(meta, separators=(",", ":")))
                .replace("%%LOGO%%", LOGO_B64)
                .replace("%%ASOF%%", datetime.strptime(meta["asof"], "%Y-%m-%d").strftime("%B %d, %Y"))
                .replace("%%YEARS%%", str(meta["years"])))
    out = os.path.join(OUTPUT_DIR, "Stock_Lookup.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
