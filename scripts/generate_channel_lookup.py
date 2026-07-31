"""
Channel Lookup — manual-entry version of the two automated regression-channel
screeners (generate_channel_screener.py / generate_uptrend_channel_screener.py
/ generate_channel_screener_cdn.py).

Instead of screening a universe for names sitting at a channel extreme, this
page embeds a lightweight 10-year regression-channel + revenue-revision
dataset for the full combined US + Canada universe, and lets you type any
comma/space-separated list of tickers to pull up their charts on demand —
client-side, no server, no gating on trend direction or channel position.
A sort control ranks whatever you typed by how far it currently sits from
its own regression line (most below / most above), instead of a fixed
top/bottom cutoff.

Universe:
  US:  S&P 500 + Nasdaq-100 + koyfin_us.csv + us_1w_rev_est_screener.csv
  CDN: TSX universe + koyfin_cdn.csv + cdn_1w_rev_est_screener.csv

Each ticker's chart data (downsampled close series + regression coefficients
+ revenue-revision %s) is embedded as JSON; Plotly charts are built in the
browser only for tickers actually looked up, so the page stays light no
matter how big the underlying universe gets.
"""

import base64
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_screening import (
    cap_tier,
    load_nasdaq100_symbols,
    load_sp500_symbols,
    load_tsx_symbols,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "channel-lookup")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")
US_REV_PATH = os.path.join(REPO_ROOT, "data", "us_1w_rev_est_screener.csv")
CDN_REV_PATH = os.path.join(REPO_ROOT, "data", "cdn_1w_rev_est_screener.csv")
KOYFIN_US_PATH = os.path.join(REPO_ROOT, "data", "koyfin_us.csv")
KOYFIN_CDN_PATH = os.path.join(REPO_ROOT, "data", "koyfin_cdn.csv")

US_BENCH = "QQQ"
CDN_BENCH = "XIC.TO"
LOOKBACK_PERIOD = "10y"
MIN_TRADING_DAYS = 60  # bare minimum to fit any kind of regression line
TARGET_POINTS = 420  # downsample target for the embedded price series
CHUNK_SIZE = 250  # yfinance batch download size

REV_COL_MAP = {
    "Ticker": "ticker",
    "Revenues Est Avg Rev % (FY1E - 1W)": "fy1_1w", "Revenues Est Avg Rev % (FY1E - 1M)": "fy1_1m",
    "Revenues Est Avg Rev % (FY1E - 3M)": "fy1_3m", "Revenues Est Avg Rev % (FY1E - 6M)": "fy1_6m",
    "Revenues Est Avg Rev % (FY1E - 1Y)": "fy1_1y",
    "Revenues Est Avg Rev % (FY2E - 1W)": "fy2_1w", "Revenues Est Avg Rev % (FY2E - 1M)": "fy2_1m",
    "Revenues Est Avg Rev % (FY2E - 3M)": "fy2_3m", "Revenues Est Avg Rev % (FY2E - 6M)": "fy2_6m",
    "Revenues Est Avg Rev % (FY2E - 1Y)": "fy2_1y",
    "Revenues Est Avg Rev % (FY3E - 1W)": "fy3_1w", "Revenues Est Avg Rev % (FY3E - 1M)": "fy3_1m",
    "Revenues Est Avg Rev % (FY3E - 3M)": "fy3_3m", "Revenues Est Avg Rev % (FY3E - 6M)": "fy3_6m",
    "Revenues Est Avg Rev % (FY3E - 1Y)": "fy3_1y",
}
REV_WIN_KEYS = ["1w", "1m", "3m", "6m", "1y"]


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

    name_map, sector_map, cap_map = {}, {}, {}
    for _, row in koyfin_df.iterrows():
        t = norm_us(row["Ticker"])
        name_map[t] = row.get("Name", t)
        sector_map[t] = (row.get("Sector", "") or "", row.get("Industry", "") or "")
        cap_map[t] = row.get("Market Cap")
    for _, row in rev_df.iterrows():
        t = norm_us(row["Ticker"])
        name_map.setdefault(t, row.get("Name", t))
        cap_map.setdefault(t, row.get("Market Cap"))

    return sorted(t for t in tickers if t and t.strip()), name_map, sector_map, cap_map


def load_cdn_universe():
    tsx = load_tsx_symbols()  # already ".TO"-suffixed, dash-normalized
    rev_df = pd.read_csv(CDN_REV_PATH).dropna(subset=["Ticker"])
    koyfin_df = pd.read_csv(KOYFIN_CDN_PATH).dropna(subset=["Ticker"])

    tickers = set(tsx)
    tickers |= {norm_cdn(t) for t in rev_df["Ticker"]}
    tickers |= {norm_cdn(t) for t in koyfin_df["Ticker"]}

    name_map, sector_map, cap_map = {}, {}, {}
    for _, row in koyfin_df.iterrows():
        t = norm_cdn(row["Ticker"])
        name_map[t] = row.get("Name", t)
        sector_map[t] = (row.get("Sector", "") or "", row.get("Industry", "") or "")
        cap_map[t] = row.get("Market Cap")
    for _, row in rev_df.iterrows():
        t = norm_cdn(row["Ticker"])
        name_map.setdefault(t, row.get("Name", t))
        cap_map.setdefault(t, row.get("Market Cap"))

    return sorted(t for t in tickers if t and t.strip()), name_map, sector_map, cap_map


def load_revision_map(path, norm_fn):
    raw = pd.read_csv(path).dropna(subset=["Ticker"])
    df = raw.rename(columns=REV_COL_MAP)
    keep = [c for c in REV_COL_MAP.values() if c in df.columns]
    df = df[keep].copy()
    for c in keep:
        if c != "ticker":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).map(norm_fn)
    return df.set_index("ticker").to_dict("index")


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
        chunk = tickers[i : i + chunk_size]
        print(f"  downloading {i + 1}-{i + len(chunk)} of {len(tickers)}...")
        try:
            raw = yf.download(
                chunk, period=period, group_by="ticker", auto_adjust=True, threads=True, progress=False
            )
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


def build_ticker_record(ticker, close, name, sector_industry, cap_bucket, revision_row, bench_label, bench_close):
    if len(close) < MIN_TRADING_DAYS:
        return None

    x_full = np.arange(len(close))
    y_full = np.log(close.values)
    slope, intercept, r_value, _, _ = linregress(x_full, y_full)
    if not np.isfinite(slope):
        return None

    fitted = intercept + slope * x_full
    resid = y_full - fitted
    std = float(resid.std())
    if std == 0:
        return None
    z_last = float(resid[-1] / std)
    r2 = float(r_value**2)

    stock_return_pct = float((close.iloc[-1] / close.iloc[0] - 1) * 100)
    outperf = None
    if bench_close is not None:
        aligned = pd.DataFrame({"stock": close, "bench": bench_close}).dropna()
        if len(aligned) >= MIN_TRADING_DAYS:
            bench_return_pct = float(aligned["bench"].iloc[-1] / aligned["bench"].iloc[0] - 1) * 100
            outperf = round(stock_return_pct - bench_return_pct, 1)

    n = len(close)
    stride = max(1, n // TARGET_POINTS)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    rev = None
    if revision_row:
        rev = {}
        for fy_idx, fy in enumerate(["fy1", "fy2", "fy3"], start=1):
            vals = [revision_row.get(f"{fy}_{w}") for w in REV_WIN_KEYS]
            rev[f"fy{fy_idx}"] = [round(v * 100, 3) if pd.notna(v) else None for v in vals]

    return {
        "t": ticker,
        "n": str(name),
        "si": sector_industry,
        "cap": cap_bucket,
        "r2": round(r2, 4),
        "z": round(z_last, 3),
        "sl": slope,
        "ic": intercept,
        "sd": std,
        "yrs": round(n / 252, 1),
        "ret": round(stock_return_pct, 1),
        "outp": outperf,
        "bench": bench_label,
        "x": [int(x_full[i]) for i in idx],
        "d": [close.index[i].strftime("%Y-%m-%d") for i in idx],
        "p": [round(float(close.iloc[i]), 3) for i in idx],
        "rev": rev,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()

    print("=== Universe ===")
    us_tickers, us_names, us_sectors, us_caps = load_us_universe()
    cdn_tickers, cdn_names, cdn_sectors, cdn_caps = load_cdn_universe()
    print(f"US: {len(us_tickers)} tickers, CDN: {len(cdn_tickers)} tickers")

    us_rev_map = load_revision_map(US_REV_PATH, norm_us)
    cdn_rev_map = load_revision_map(CDN_REV_PATH, norm_cdn)

    print(f"Downloading benchmarks ({US_BENCH}, {CDN_BENCH})...")
    qqq_close = close_series_single(US_BENCH, yf.download(US_BENCH, period=LOOKBACK_PERIOD, auto_adjust=True, progress=False))
    xic_close = close_series_single(CDN_BENCH, yf.download(CDN_BENCH, period=LOOKBACK_PERIOD, auto_adjust=True, progress=False))

    print(f"Downloading {len(us_tickers)} US tickers ({LOOKBACK_PERIOD})...")
    us_close_map = batch_download_closes(us_tickers, LOOKBACK_PERIOD, CHUNK_SIZE)
    print(f"Got price history for {len(us_close_map)} US tickers")

    print(f"Downloading {len(cdn_tickers)} CDN tickers ({LOOKBACK_PERIOD})...")
    cdn_close_map = batch_download_closes(cdn_tickers, LOOKBACK_PERIOD, CHUNK_SIZE)
    print(f"Got price history for {len(cdn_close_map)} CDN tickers")

    records = {}

    if qqq_close is not None:
        rec = build_ticker_record(US_BENCH, qqq_close.rename(US_BENCH), "Invesco QQQ Trust", "", "ETF", None, US_BENCH, None)
        if rec:
            records[US_BENCH] = rec
    if xic_close is not None:
        rec = build_ticker_record(CDN_BENCH, xic_close.rename(CDN_BENCH), "iShares S&P/TSX Capped Composite ETF", "", "ETF", None, CDN_BENCH, None)
        if rec:
            records[CDN_BENCH] = rec

    for ticker, close in us_close_map.items():
        sector, industry = us_sectors.get(ticker, ("", ""))
        si = " | ".join(s for s in (sector, industry) if s)
        try:
            rec = build_ticker_record(
                ticker, close.rename(ticker), us_names.get(ticker, ticker), si,
                cap_tier(us_caps.get(ticker)), us_rev_map.get(ticker), US_BENCH, qqq_close,
            )
        except Exception as exc:
            print(f"  error scoring {ticker}: {exc}")
            continue
        if rec:
            records[ticker] = rec

    for ticker, close in cdn_close_map.items():
        sector, industry = cdn_sectors.get(ticker, ("", ""))
        si = " | ".join(s for s in (sector, industry) if s)
        try:
            rec = build_ticker_record(
                ticker, close.rename(ticker), cdn_names.get(ticker, ticker), si,
                cap_tier(cdn_caps.get(ticker)), cdn_rev_map.get(ticker), CDN_BENCH, xic_close,
            )
        except Exception as exc:
            print(f"  error scoring {ticker}: {exc}")
            continue
        if rec:
            records[ticker] = rec

    print(f"Built {len(records)} ticker records")

    with open(LOGO_PATH, "rb") as f:
        logo_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()

    data_json = json.dumps(records, separators=(",", ":"), allow_nan=False)
    html = PAGE_TEMPLATE.format(date_str=today.strftime("%B %d, %Y"), data_json=data_json, count=len(records),
                                 logo_b64=logo_b64)
    out_path = os.path.join(OUTPUT_DIR, "Channel_Lookup.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Channel Lookup</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E8E8E8; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .controls {{ padding: 20px 32px; border-bottom: 1px solid #3A3A3C; display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; }}
  .controls label {{ color: #8E8E93; font-size: 13px; }}
  .controls input[type=text] {{ background: #2A2A2C; color: #E8E8E8; border: 1px solid #3A3A3C; border-radius: 4px;
    padding: 7px 10px; font-size: 14px; width: 360px; max-width: 100%; }}
  .controls select {{ background: #2A2A2C; color: #E8E8E8; border: 1px solid #3A3A3C; border-radius: 4px; padding: 6px 10px; font-size: 13px; }}
  .controls button {{ background: #C67A29; color: #1C1C1E; border: none; border-radius: 4px; padding: 8px 16px;
    font-size: 14px; font-weight: 600; cursor: pointer; }}
  .controls button:hover {{ background: #d98b36; }}
  .not-found {{ padding: 0 32px; color: #A22A2A; font-size: 13px; min-height: 18px; }}
  .placeholder {{ padding: 40px 32px; color: #8E8E93; font-size: 14px; }}
  .chart-wrap {{ padding: 8px 12px; overflow-x: auto; }}
</style>
</head>
<body>
<header>
  <h1>Channel Lookup</h1>
  <div class="meta">Generated {date_str} &middot; {count} tickers available (US + Canada) &middot; type any comma/space-separated list of tickers to pull up 10-year regression-channel + revenue-revision charts on demand, ranked wherever they currently sit in the channel &mdash; no top/bottom cutoff</div>
</header>
<div class="controls">
  <label for="tickerInput">Tickers:</label>
  <input type="text" id="tickerInput" placeholder="AAPL, MSFT, XIC.TO">
  <button onclick="renderTickers()">Show</button>
  <label for="sortSelect">Sort:</label>
  <select id="sortSelect" onchange="renderTickers()">
    <option value="entered">As entered</option>
    <option value="z_asc">Lowest in channel first</option>
    <option value="z_desc">Highest in channel first</option>
  </select>
</div>
<div id="notFound" class="not-found"></div>
<div id="results"><div class="placeholder">Type one or more tickers above and press Enter (or click Show).</div></div>

<script>
const DATA = {data_json};
const LOGO_B64 = "{logo_b64}";

document.getElementById('tickerInput').addEventListener('keydown', function (e) {{
  if (e.key === 'Enter') {{ e.preventDefault(); renderTickers(); }}
}});

function normalizeTicker(raw) {{
  var t = raw.trim().toUpperCase();
  if (!t) return '';
  if (t.endsWith('.TO')) {{
    return t.slice(0, -3).replace(/\\./g, '-') + '.TO';
  }}
  return t.replace(/\\./g, '-');
}}

function renderTickers() {{
  var raw = document.getElementById('tickerInput').value;
  var seen = {{}};
  var wanted = [];
  raw.split(/[,\\s]+/).forEach(function (piece) {{
    var t = normalizeTicker(piece);
    if (t && !seen[t]) {{ seen[t] = true; wanted.push(t); }}
  }});

  var found = [];
  var missing = [];
  wanted.forEach(function (t) {{
    if (DATA[t]) found.push(DATA[t]); else missing.push(t);
  }});

  var sortMode = document.getElementById('sortSelect').value;
  if (sortMode === 'z_asc') found.sort(function (a, b) {{ return a.z - b.z; }});
  else if (sortMode === 'z_desc') found.sort(function (a, b) {{ return b.z - a.z; }});

  document.getElementById('notFound').textContent = missing.length ? ('Not found in universe: ' + missing.join(', ')) : '';

  var container = document.getElementById('results');
  container.innerHTML = '';
  if (!found.length) {{
    container.innerHTML = '<div class="placeholder">Type one or more tickers above and press Enter (or click Show).</div>';
    return;
  }}
  found.forEach(function (rec, i) {{ renderCard(rec, i, container); }});
}}

function renderCard(rec, i, container) {{
  var wrap = document.createElement('div');
  wrap.className = 'chart-wrap';
  wrap.id = 'chart-' + i;
  container.appendChild(wrap);

  var center = rec.x.map(function (x) {{ return Math.exp(rec.ic + rec.sl * x); }});
  var upper = rec.x.map(function (x) {{ return Math.exp(rec.ic + rec.sl * x + 2 * rec.sd); }});
  var lower = rec.x.map(function (x) {{ return Math.exp(rec.ic + rec.sl * x - 2 * rec.sd); }});

  var traces = [
    {{ x: rec.d, y: upper, mode: 'lines', line: {{ color: '#8E8E93', width: 1, dash: 'dot' }}, name: '+2σ', showlegend: false, xaxis: 'x', yaxis: 'y' }},
    {{ x: rec.d, y: lower, mode: 'lines', line: {{ color: '#8E8E93', width: 1, dash: 'dot' }}, name: '-2σ', fill: 'tonexty', fillcolor: 'rgba(255,255,255,0.05)', showlegend: false, xaxis: 'x', yaxis: 'y' }},
    {{ x: rec.d, y: center, mode: 'lines', line: {{ color: '#C67A29', width: 1.5, dash: 'dash' }}, name: 'Regression', showlegend: false, xaxis: 'x', yaxis: 'y' }},
    {{ x: rec.d, y: rec.p, mode: 'lines', line: {{ color: '#1F79BE', width: 1.8 }}, name: 'Close', showlegend: false, xaxis: 'x', yaxis: 'y' }},
    {{ x: [rec.d[rec.d.length - 1]], y: [rec.p[rec.p.length - 1]], mode: 'markers',
      marker: {{ color: '#A22A2A', size: 9, line: {{ color: '#E8E8E8', width: 1 }} }},
      name: 'Last', showlegend: false, xaxis: 'x', yaxis: 'y',
      hovertemplate: 'Z=' + rec.z.toFixed(2) + 'σ<extra></extra>' }}
  ];

  var annotations = [
    {{ font: {{ size: 12, color: '#E8E8E8' }}, showarrow: false, text: rec.yrs + 'Y Regression Channel', x: 0.2821, xanchor: 'center', xref: 'paper', y: 1.0, yanchor: 'bottom', yref: 'paper' }},
    {{ font: {{ size: 12, color: '#E8E8E8' }}, showarrow: false, text: 'Analyst Revenue Revision Trend', x: 0.8271, xanchor: 'center', xref: 'paper', y: 1.0, yanchor: 'bottom', yref: 'paper' }}
  ];

  if (rec.rev) {{
    var winLabels = ['1W', '1M', '3M', '6M', '1Y'];
    var fyColors = ['#1F79BE', '#4B8EA9', '#C67A29'];
    [1, 2, 3].forEach(function (fy) {{
      traces.push({{ x: winLabels, y: rec.rev['fy' + fy], type: 'bar', name: 'FY' + fy + 'E',
        marker: {{ color: fyColors[fy - 1], opacity: 0.85 }}, xaxis: 'x2', yaxis: 'y2',
        hovertemplate: '%{{x}}: %{{y:+.2f}}%<extra>FY' + fy + 'E</extra>' }});
    }});
  }} else {{
    annotations.push({{ x: 0.81, y: 0.5, xref: 'paper', yref: 'paper', text: 'No revision data', showarrow: false, font: {{ size: 12, color: '#8E8E93' }} }});
  }}

  var posLabel = rec.z <= -1.5 ? 'near bottom of channel' : (rec.z >= 1.5 ? 'near top of channel' : 'mid-channel');
  var outpStr = (rec.outp === null || rec.outp === undefined) ? '\\u2014' : ((rec.outp >= 0 ? '+' : '') + rec.outp.toFixed(0) + '%');
  var titleText = '<b>' + rec.t + '</b>  ·  ' + rec.n + '  ·  R²=' + rec.r2.toFixed(2) +
    '  ·  Z=' + rec.z.toFixed(2) + 'σ (' + posLabel + ')  ·  ' + rec.yrs + 'Y Return: ' + rec.ret.toFixed(0) +
    '%  ·  vs ' + rec.bench + ': ' + outpStr +
    (rec.si ? ('<br><span style="font-size:12px;color:#8E8E93">' + rec.si + '</span>') : '');

  Plotly.newPlot(wrap.id, traces, {{
    height: 520, width: 1600,
    paper_bgcolor: '#363636', plot_bgcolor: '#4A4A4A',
    font: {{ family: 'Arial, sans-serif', color: '#E8E8E8', size: 12 }},
    title: {{ text: titleText, font: {{ size: 15, color: '#E8E8E8' }}, x: 0.03, y: 0.97 }},
    margin: {{ t: rec.si ? 105 : 90, b: 40, l: 60, r: 40 }},
    hovermode: 'x unified', barmode: 'group',
    legend: {{ orientation: 'h', y: 1.13, x: 0.64, font: {{ size: 10 }} }},
    xaxis: {{ domain: [0.0, 0.5642], gridcolor: '#555', gridwidth: 0.4 }},
    yaxis: {{ domain: [0.0, 1.0], title: {{ text: 'Price (log)' }}, type: 'log', gridcolor: '#555', gridwidth: 0.5, zeroline: false }},
    xaxis2: {{ domain: [0.6542, 1.0], anchor: 'y2', gridcolor: '#555', gridwidth: 0.4 }},
    yaxis2: {{ domain: [0.0, 1.0], anchor: 'x2', title: {{ text: 'Revenue Revision %' }}, ticksuffix: '%', gridcolor: '#555', gridwidth: 0.5, zeroline: true, zerolinecolor: '#555' }},
    annotations: annotations,
    images: [{{ source: LOGO_B64, xref: 'paper', yref: 'paper', x: 1.0, y: 1.16, sizex: 0.10, sizey: 0.10, xanchor: 'right', yanchor: 'bottom', opacity: 0.90, layer: 'above' }}]
  }}, {{ responsive: false, displayModeBar: false }});
}}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
