"""
Ratio Channel Lookup — interactive, client-side companion to Channel Lookup
and the Ratio Channel Screener.

The Ratio Channel Screener (generate_ratio_channel_screener.py) assigns a
fixed benchmark per name (QQQ for Nasdaq-100 members, SPY for other US names,
XIC.TO for Canada) and screens the whole universe for names whose ratio to
THAT benchmark sits at a channel extreme. This page instead lets you pick
BOTH sides of the ratio yourself — any "base" ticker (e.g. SPY, or any other
stock/ETF) and any number of "compare" tickers (e.g. AAPL, MSFT, RY.TO) — and
renders a 10-year log(compare/base) regression channel for each pair on
demand, using the exact same methodology (log-ratio linregress, +/-2 sigma
bands, R^2, Z-score) as the screener, just computed in the browser instead of
pre-screened server-side against one fixed benchmark.

Since the regression is fit on a RATIO of two series, both tickers' price
histories need to line up date-for-date. Unlike Channel Lookup (which
downsamples each ticker's own history independently by row-stride, fine for a
standalone single-stock channel), this embeds every ticker's price on a
shared WEEKLY (Friday) calendar, so any two tickers' embedded date arrays are
directly comparable and their overlap - and the regression itself - can be
computed client-side with no server round-trip, no matter which pair you ask
for.

Universe: same combined US (S&P 500 + Nasdaq-100 + koyfin_us.csv +
us_1w_rev_est_screener.csv) + Canada (TSX + koyfin_cdn.csv +
cdn_1w_rev_est_screener.csv) universe as Channel Lookup and the Ratio Channel
Screener, merged into one flat namespace since either side of the ratio can
be any ticker.
"""

import base64
import json
import os
import sys
import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_screening import (
    load_nasdaq100_symbols,
    load_sp500_symbols,
    load_tsx_symbols,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "ratio-channel-lookup")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")
US_REV_PATH = os.path.join(REPO_ROOT, "data", "us_1w_rev_est_screener.csv")
CDN_REV_PATH = os.path.join(REPO_ROOT, "data", "cdn_1w_rev_est_screener.csv")
KOYFIN_US_PATH = os.path.join(REPO_ROOT, "data", "koyfin_us.csv")
KOYFIN_CDN_PATH = os.path.join(REPO_ROOT, "data", "koyfin_cdn.csv")

LOOKBACK_PERIOD = "10y"
MIN_WEEKS = 26  # bare minimum weekly bars to embed a ticker at all (~6 months) - a lookup tool,
                # not a screener, so this is deliberately permissive; per-pair overlap is checked
                # again client-side once you actually pick two tickers.
CHUNK_SIZE = 250  # yfinance batch download size
DEFAULT_BASE = "SPY"

# Broad-market ETFs that aren't in the individual-company universe below but are exactly the kind
# of ticker someone picks as a ratio's base/denominator - matches the fixed benchmarks used
# elsewhere in this repo (SPY/QQQ/XIC.TO) plus a couple more from the Industry RSI report.
BACKFILL_TICKERS = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "XIC.TO": "iShares S&P/TSX Capped Composite ETF",
    "IWM": "iShares Russell 2000 ETF",
    "RSP": "Invesco S&P 500 Equal Weight ETF",
    "QQQE": "Invesco Nasdaq-100 Equal Weight ETF",
}


def norm_us(ticker):
    return str(ticker).strip().upper().replace(".", "-")


def norm_cdn(ticker):
    t = str(ticker).strip().upper()
    if t.endswith(".TO"):
        t = t[:-3]
    return t.replace(".", "-") + ".TO"


def load_us_universe():
    sp500 = load_sp500_symbols()
    nasdaq100 = load_nasdaq100_symbols()
    rev_df = pd.read_csv(US_REV_PATH).dropna(subset=["Ticker"])
    koyfin_df = pd.read_csv(KOYFIN_US_PATH).dropna(subset=["Ticker"])

    tickers = {norm_us(t) for t in sp500} | {norm_us(t) for t in nasdaq100}
    tickers |= {norm_us(t) for t in rev_df["Ticker"]}
    tickers |= {norm_us(t) for t in koyfin_df["Ticker"]}

    name_map, sector_map = {}, {}
    for _, row in koyfin_df.iterrows():
        t = norm_us(row["Ticker"])
        name_map[t] = row.get("Name", t)
        sector_map[t] = (row.get("Sector", "") or "", row.get("Industry", "") or "")
    for _, row in rev_df.iterrows():
        t = norm_us(row["Ticker"])
        name_map.setdefault(t, row.get("Name", t))

    return sorted(t for t in tickers if t and t.strip()), name_map, sector_map


def load_cdn_universe():
    tsx = load_tsx_symbols()  # already ".TO"-suffixed, dash-normalized
    rev_df = pd.read_csv(CDN_REV_PATH).dropna(subset=["Ticker"])
    koyfin_df = pd.read_csv(KOYFIN_CDN_PATH).dropna(subset=["Ticker"])

    tickers = set(tsx)
    tickers |= {norm_cdn(t) for t in rev_df["Ticker"]}
    tickers |= {norm_cdn(t) for t in koyfin_df["Ticker"]}

    name_map, sector_map = {}, {}
    for _, row in koyfin_df.iterrows():
        t = norm_cdn(row["Ticker"])
        name_map[t] = row.get("Name", t)
        sector_map[t] = (row.get("Sector", "") or "", row.get("Industry", "") or "")
    for _, row in rev_df.iterrows():
        t = norm_cdn(row["Ticker"])
        name_map.setdefault(t, row.get("Name", t))

    return sorted(t for t in tickers if t and t.strip()), name_map, sector_map


def close_series_single(ticker, raw):
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty or "Close" not in raw:
        return None
    close = raw["Close"].dropna()
    return close if not close.empty else None


def batch_download_closes(tickers, period, chunk_size):
    close_map = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"  downloading {i + 1}-{i + len(chunk)} of {len(tickers)}...")
        try:
            raw = yf.download(chunk, period=period, group_by="ticker", auto_adjust=True, threads=True, progress=False)
        except Exception as exc:
            print(f"  batch error: {exc}")
            continue
        if len(chunk) == 1:
            close = close_series_single(chunk[0], raw)
            if close is not None:
                close_map[chunk[0]] = close
            continue
        for t in chunk:
            try:
                close = raw[t]["Close"].dropna()
                if not close.empty:
                    close_map[t] = close
            except (KeyError, TypeError):
                pass
    return close_map


def build_ticker_record(ticker, close, name, sector_industry):
    weekly = close.resample("W-FRI").last().dropna()
    if len(weekly) < MIN_WEEKS:
        return None
    return {
        "t": ticker,
        "n": str(name),
        "si": sector_industry,
        "d": [d.strftime("%Y-%m-%d") for d in weekly.index],
        "p": [round(float(v), 4) for v in weekly.values],
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()

    print("=== Universe ===")
    us_tickers, us_names, us_sectors = load_us_universe()
    cdn_tickers, cdn_names, cdn_sectors = load_cdn_universe()
    print(f"US: {len(us_tickers)} tickers, CDN: {len(cdn_tickers)} tickers")

    print(f"Downloading {len(us_tickers)} US tickers ({LOOKBACK_PERIOD})...")
    us_close_map = batch_download_closes(us_tickers, LOOKBACK_PERIOD, CHUNK_SIZE)
    print(f"Got price history for {len(us_close_map)} US tickers")

    print(f"Downloading {len(cdn_tickers)} CDN tickers ({LOOKBACK_PERIOD})...")
    cdn_close_map = batch_download_closes(cdn_tickers, LOOKBACK_PERIOD, CHUNK_SIZE)
    print(f"Got price history for {len(cdn_close_map)} CDN tickers")

    records = {}
    for ticker, close in us_close_map.items():
        sector, industry = us_sectors.get(ticker, ("", ""))
        si = " | ".join(s for s in (sector, industry) if s)
        rec = build_ticker_record(ticker, close.rename(ticker), us_names.get(ticker, ticker), si)
        if rec:
            records[ticker] = rec

    for ticker, close in cdn_close_map.items():
        sector, industry = cdn_sectors.get(ticker, ("", ""))
        si = " | ".join(s for s in (sector, industry) if s)
        rec = build_ticker_record(ticker, close.rename(ticker), cdn_names.get(ticker, ticker), si)
        if rec:
            records[ticker] = rec

    # Broad-market benchmark ETFs (the same ones used as fixed benchmarks elsewhere in this repo -
    # SPY/QQQ/XIC.TO - plus a couple more common "base" choices) aren't individual companies, so
    # they're not in the S&P 500/Nasdaq-100/revenue-screener/Koyfin universes above. Backfill them
    # explicitly since they're exactly the kind of ticker someone picks as the ratio's denominator.
    for bfill_ticker, bfill_name in BACKFILL_TICKERS.items():
        if bfill_ticker in records:
            continue
        try:
            close = close_series_single(bfill_ticker, yf.download(bfill_ticker, period=LOOKBACK_PERIOD, auto_adjust=True, progress=False))
            if close is not None:
                rec = build_ticker_record(bfill_ticker, close.rename(bfill_ticker), bfill_name, "")
                if rec:
                    records[bfill_ticker] = rec
        except Exception as exc:
            print(f"  could not backfill {bfill_ticker}: {exc}")

    print(f"Built {len(records)} ticker records")

    with open(LOGO_PATH, "rb") as f:
        logo_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()

    data_json = json.dumps(records, separators=(",", ":"), allow_nan=False)
    html = PAGE_TEMPLATE.format(
        date_str=today.strftime("%B %d, %Y"), data_json=data_json, count=len(records),
        logo_b64=logo_b64, default_base=DEFAULT_BASE,
    )
    out_path = os.path.join(OUTPUT_DIR, "Ratio_Channel_Lookup.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ratio Channel Lookup</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E8E8E8; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .controls {{ padding: 20px 32px; border-bottom: 1px solid #3A3A3C; display: flex; flex-wrap: wrap; align-items: flex-end; gap: 10px 20px; }}
  .controls .field {{ display: flex; flex-direction: column; gap: 4px; }}
  .controls label {{ color: #8E8E93; font-size: 13px; }}
  .controls input[type=text] {{ background: #2A2A2C; color: #E8E8E8; border: 1px solid #3A3A3C; border-radius: 4px;
    padding: 7px 10px; font-size: 14px; }}
  #baseInput {{ width: 140px; }}
  #compareInput {{ width: 380px; max-width: 100%; }}
  .controls button {{ background: #C67A29; color: #1C1C1E; border: none; border-radius: 4px; padding: 8px 16px;
    font-size: 14px; font-weight: 600; cursor: pointer; height: 34px; }}
  .controls button:hover {{ background: #d98b36; }}
  .not-found {{ padding: 0 32px; color: #A22A2A; font-size: 13px; min-height: 18px; }}
  .placeholder {{ padding: 40px 32px; color: #8E8E93; font-size: 14px; }}
  .chart-wrap {{ padding: 8px 12px; overflow-x: auto; }}
  .chart-note {{ padding: 4px 32px 0; color: #8E8E93; font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>Ratio Channel Lookup</h1>
  <div class="meta">Generated {date_str} &middot; {count} tickers available (US + Canada) &middot; pick any base ticker and any comparison ticker(s) for an on-demand 10-year log-ratio regression channel &mdash; same methodology as the Ratio Channel Screener, computed here for whichever pair you choose</div>
</header>
<div class="controls">
  <div class="field">
    <label for="baseInput">Base (denominator)</label>
    <input type="text" id="baseInput" value="{default_base}" placeholder="SPY">
  </div>
  <div class="field">
    <label for="compareInput">Compare against</label>
    <input type="text" id="compareInput" placeholder="AAPL, MSFT, RY.TO">
  </div>
  <button onclick="renderRatios()">Show</button>
</div>
<div id="notFound" class="not-found"></div>
<div id="results"><div class="placeholder">Enter a base ticker (defaults to SPY) and one or more comparison tickers, then press Enter or click Show. Each chart shows Compare/Base as a 10-year log-ratio regression channel.</div></div>

<script>
const DATA = {data_json};
const LOGO_B64 = "{logo_b64}";
const MIN_OVERLAP_WEEKS = 20;
const CHANNEL_SIGMA = 2;

['baseInput', 'compareInput'].forEach(function (id) {{
  document.getElementById(id).addEventListener('keydown', function (e) {{
    if (e.key === 'Enter') {{ e.preventDefault(); renderRatios(); }}
  }});
}});

function normalizeTicker(raw) {{
  var t = raw.trim().toUpperCase();
  if (!t) return '';
  if (t.endsWith('.TO')) {{
    return t.slice(0, -3).replace(/\\./g, '-') + '.TO';
  }}
  return t.replace(/\\./g, '-');
}}

function computeRegression(ratios) {{
  var n = ratios.length;
  var x = [], y = [];
  for (var i = 0; i < n; i++) {{ x.push(i); y.push(Math.log(ratios[i])); }}
  var meanX = x.reduce(function (a, b) {{ return a + b; }}, 0) / n;
  var meanY = y.reduce(function (a, b) {{ return a + b; }}, 0) / n;
  var ssXY = 0, ssXX = 0, ssYY = 0;
  for (var i = 0; i < n; i++) {{
    var dx = x[i] - meanX, dy = y[i] - meanY;
    ssXY += dx * dy; ssXX += dx * dx; ssYY += dy * dy;
  }}
  var slope = ssXY / ssXX;
  var intercept = meanY - slope * meanX;
  var ssRes = 0, resid = [];
  for (var i = 0; i < n; i++) {{
    var r = y[i] - (intercept + slope * x[i]);
    resid.push(r);
    ssRes += r * r;
  }}
  var std = Math.sqrt(ssRes / n);
  var r2 = ssYY > 0 ? (1 - ssRes / ssYY) : 0;
  return {{ slope: slope, intercept: intercept, std: std, r2: r2, z: resid[n - 1] / (std || 1) }};
}}

function buildOverlap(baseRec, cmpRec) {{
  var baseMap = {{}};
  for (var i = 0; i < baseRec.d.length; i++) {{ baseMap[baseRec.d[i]] = baseRec.p[i]; }}
  var dates = [], ratios = [];
  for (var i = 0; i < cmpRec.d.length; i++) {{
    var d = cmpRec.d[i], bp = baseMap[d], cp = cmpRec.p[i];
    if (bp && bp > 0 && cp && cp > 0) {{ dates.push(d); ratios.push(cp / bp); }}
  }}
  return {{ dates: dates, ratios: ratios }};
}}

function renderRatios() {{
  var baseTicker = normalizeTicker(document.getElementById('baseInput').value);
  var raw = document.getElementById('compareInput').value;
  var seen = {{}};
  var wanted = [];
  raw.split(/[,\\s]+/).forEach(function (piece) {{
    var t = normalizeTicker(piece);
    if (t && !seen[t]) {{ seen[t] = true; wanted.push(t); }}
  }});

  var container = document.getElementById('results');
  var notFoundEl = document.getElementById('notFound');
  container.innerHTML = '';

  if (!baseTicker) {{
    notFoundEl.textContent = 'Enter a base ticker.';
    container.innerHTML = '<div class="placeholder">Enter a base ticker (defaults to SPY) and one or more comparison tickers, then press Enter or click Show.</div>';
    return;
  }}
  if (!DATA[baseTicker]) {{
    notFoundEl.textContent = 'Base ticker not found in universe: ' + baseTicker;
    container.innerHTML = '<div class="placeholder">Try a different base ticker.</div>';
    return;
  }}
  if (!wanted.length) {{
    notFoundEl.textContent = '';
    container.innerHTML = '<div class="placeholder">Enter one or more comparison tickers, then press Enter or click Show.</div>';
    return;
  }}

  var missing = [];
  var skippedSelf = false;
  var i = 0;
  wanted.forEach(function (t) {{
    if (t === baseTicker) {{ skippedSelf = true; return; }}
    if (!DATA[t]) {{ missing.push(t); return; }}
    renderPair(DATA[baseTicker], DATA[t], i, container);
    i++;
  }});

  var msgs = [];
  if (missing.length) msgs.push('Not found in universe: ' + missing.join(', '));
  if (skippedSelf) msgs.push("Skipped comparing " + baseTicker + " to itself.");
  notFoundEl.textContent = msgs.join('  \\u2014  ');

  if (i === 0 && !container.innerHTML) {{
    container.innerHTML = '<div class="placeholder">Nothing to show.</div>';
  }}
}}

function renderPair(baseRec, cmpRec, i, container) {{
  var overlap = buildOverlap(baseRec, cmpRec);
  var wrap = document.createElement('div');
  wrap.id = 'chart-' + i;
  container.appendChild(wrap);

  if (overlap.ratios.length < MIN_OVERLAP_WEEKS) {{
    var note = document.createElement('div');
    note.className = 'chart-note';
    note.textContent = cmpRec.t + ' / ' + baseRec.t + ': only ' + overlap.ratios.length + ' weeks of overlapping history \\u2014 too little for a meaningful channel.';
    wrap.appendChild(note);
    wrap.className = 'chart-note';
    return;
  }}
  wrap.className = 'chart-wrap';

  var reg = computeRegression(overlap.ratios);
  var n = overlap.ratios.length;
  var xIdx = []; for (var k = 0; k < n; k++) xIdx.push(k);
  var center = xIdx.map(function (x) {{ return Math.exp(reg.intercept + reg.slope * x); }});
  var upper = xIdx.map(function (x) {{ return Math.exp(reg.intercept + reg.slope * x + CHANNEL_SIGMA * reg.std); }});
  var lower = xIdx.map(function (x) {{ return Math.exp(reg.intercept + reg.slope * x - CHANNEL_SIGMA * reg.std); }});

  var traces = [
    {{ x: overlap.dates, y: upper, mode: 'lines', line: {{ color: '#8E8E93', width: 1, dash: 'dot' }}, name: '+2\\u03c3', showlegend: false }},
    {{ x: overlap.dates, y: lower, mode: 'lines', line: {{ color: '#8E8E93', width: 1, dash: 'dot' }}, name: '-2\\u03c3', fill: 'tonexty', fillcolor: 'rgba(255,255,255,0.05)', showlegend: false }},
    {{ x: overlap.dates, y: center, mode: 'lines', line: {{ color: '#C67A29', width: 1.5, dash: 'dash' }}, name: 'Regression', showlegend: false }},
    {{ x: overlap.dates, y: overlap.ratios, mode: 'lines', line: {{ color: '#1F79BE', width: 1.8 }}, name: 'Ratio', showlegend: false }},
    {{ x: [overlap.dates[n - 1]], y: [overlap.ratios[n - 1]], mode: 'markers',
      marker: {{ color: '#A22A2A', size: 9, line: {{ color: '#E8E8E8', width: 1 }} }},
      name: 'Last', showlegend: false, hovertemplate: 'Z=' + reg.z.toFixed(2) + '\\u03c3<extra></extra>' }}
  ];

  var years = (n / 52.18).toFixed(1);
  var posLabel = reg.z <= -1.5 ? 'near bottom of channel' : (reg.z >= 1.5 ? 'near top of channel' : 'mid-channel');
  var titleText = '<b>' + cmpRec.t + ' / ' + baseRec.t + '</b>  \\u00b7  ' + cmpRec.n + ' vs ' + baseRec.n +
    '  \\u00b7  R\\u00b2=' + reg.r2.toFixed(2) + '  \\u00b7  Z=' + reg.z.toFixed(2) + '\\u03c3 (' + posLabel + ')  \\u00b7  ' + years + 'Y overlap' +
    (cmpRec.si ? ('<br><span style="font-size:12px;color:#8E8E93">' + cmpRec.si + '</span>') : '');

  Plotly.newPlot(wrap.id, traces, {{
    height: 480, width: 1200,
    paper_bgcolor: '#363636', plot_bgcolor: '#4A4A4A',
    font: {{ family: 'Arial, sans-serif', color: '#E8E8E8', size: 12 }},
    title: {{ text: titleText, font: {{ size: 15, color: '#E8E8E8' }}, x: 0.03, y: 0.96 }},
    margin: {{ t: cmpRec.si ? 90 : 75, b: 40, l: 60, r: 40 }},
    hovermode: 'x unified',
    xaxis: {{ gridcolor: '#555', gridwidth: 0.4 }},
    yaxis: {{ title: {{ text: cmpRec.t + ' / ' + baseRec.t + ' (log)' }}, type: 'log', gridcolor: '#555', gridwidth: 0.5, zeroline: false }},
    images: [{{ source: LOGO_B64, xref: 'paper', yref: 'paper', x: 1.0, y: 1.08, sizex: 0.12, sizey: 0.12, xanchor: 'right', yanchor: 'bottom', opacity: 0.90, layer: 'above' }}]
  }}, {{ responsive: false, displayModeBar: false }});
}}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
