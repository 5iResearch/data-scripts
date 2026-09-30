"""
Website-ready 10-Year Regression Channel Screener: the same screens as generate_channel_screener_cdn.py (TSX, vs XIC)
and generate_channel_screener.py (S&P 500 + Nasdaq-100 + US revision CSV, vs QQQ), restyled to match the other
website pages (white background, 5i colours) and combined on one page, Canada first.

Scoring, universes, thresholds and data all come from those two scripts (their own score_ticker, loaders and
constants), so this page always lists the same names they do. The only differences are presentation:
  - a compact table per list (ranked by R², as in the originals), then a card per name;
  - the channel chart is drawn in the browser from weekly points (every fifth trading day, ending on the latest
    close) with the regression line and +/-2 sigma bands from the original daily fit, so the page is ~2 MB
    instead of ~30 MB and only draws the charts a visitor scrolls to;
  - beside it, in place of the originals' revision bars, the Stock Lookup's price vs EPS / revenue chart (median
    P/E scaling, actual prices, dashed estimates), so each card shows whether the business kept pace with the
    price. The revision signal stays as one header figure (next fiscal year's revenue estimate, 3-month change).
Sector / industry labels come from the revision CSVs' own columns, falling back to data/koyfin_*.csv.
Output: outputs/channel-screener-web/Channel_Screener.html
"""

import json
import os
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import yfinance as yf

import generate_channel_screener as us
import generate_channel_screener_cdn as ca
from common_screening import CAP_FILTER_CONTROL_HTML, CAP_FILTER_JS, cap_tier
from generate_index_rsi_web import FULLSCREEN_HTML, LOGO_B64
from generate_stock_lookup_web import download_both, load_info, norm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "channel-screener-web")
WEEK = 5   # trading days between chart points
FUND_YEARS = 12   # actual-price history for the price-vs-earnings chart: covers the -10FY year end with room to spare


def csv_labels(path, to_key):
    """Key -> (name, sector, industry) from a revision CSV. The Ticker converter keeps "NA" from reading as NaN."""
    df = pd.read_csv(path, converters={"Ticker": lambda v: str(v).strip()})
    df = df[df["Ticker"] != ""].fillna({"Name": "", "Sector": "", "Industry": ""})
    get = lambda c: df[c] if c in df else [""] * len(df)
    return {to_key(t): (n, s, i) for t, n, s, i in zip(df["Ticker"], get("Name"), get("Sector"), get("Industry"))}


def run_market(mod, key, label, bench_label, to_key, display, koyfin_sectors):
    """Runs one original screener's download + scoring (its own main(), minus the page) and returns its rows."""
    print(f"=== {label} ===")
    universe, _ = mod.load_universe()
    revisions, caps = mod.load_revision_map(), mod.load_cap_map()
    labels = csv_labels(mod.REV_SCREENER_PATH, to_key)
    print(f"Universe: {len(universe)} unique tickers")

    bench = mod.close_series_single(mod.BENCH_TICKER, yf.download(mod.BENCH_TICKER, period=mod.LOOKBACK_PERIOD,
                                                                  auto_adjust=True, progress=False))
    if bench is None:
        raise RuntimeError(f"Could not download {mod.BENCH_TICKER} close prices")
    closes = mod.batch_download_closes(universe, mod.LOOKBACK_PERIOD, mod.CHUNK_SIZE)
    print(f"Got price history for {len(closes)} tickers")

    rows = []
    for ticker, close in closes.items():
        try:
            row = mod.score_ticker(close.rename(ticker), bench)
        except Exception as exc:
            print(f"  error scoring {ticker}: {exc}")
            continue
        if row is None:
            continue
        name, sector, industry = labels.get(ticker, ("", "", ""))
        if not sector:
            sector, industry = koyfin_sectors.get(display(ticker), ("", ""))
        rows.append(pack(row, mod, display(ticker), name, sector, industry, cap_tier(caps.get(ticker)),
                         revisions.get(ticker)))
        rows[-1]["_sym"] = ticker
    rows.sort(key=lambda r: r["r2"], reverse=True)   # the originals' ranking (TOP_R2_FRACTION is 1.0: keep all)
    print(f"{len(rows)} names at a channel extreme "
          f"({sum(r['pos'] == 'Bottom' for r in rows)} bottom, {sum(r['pos'] == 'Top' for r in rows)} top)")
    return {"key": key, "label": label, "bench": bench_label, "rows": rows, "asof": str(bench.index[-1].date())}


def pack(row, mod, ticker, name, sector, industry, cap, rev):
    close = row["_close"]
    idx = np.arange(len(close) - 1, -1, -WEEK)[::-1]   # anchored on the latest close
    fit = np.exp(row["_intercept"] + row["_slope"] * idx)
    sig = lambda v: float(f"{v:.4g}")
    # One revision figure for the card header: next fiscal year's revenue estimate, change over three months
    rev3m = rev.get("fy1_3m") if rev else None
    return {
        "t": ticker, "n": name or ticker, "sector": sector, "industry": industry, "cap": cap, "pos": row["Position"],
        "r2": round(row["R2"], 3), "z": round(row["Z_Score"], 2), "ret": round(row["10Y_Return_%"]),
        "out": round(row["Outperformance_%"]), "trend": round(row["Annual_Trend_Return_%"], 1),
        "band": round(float(np.exp(mod.CHANNEL_SIGMA * row["_std"])), 4),
        "d": [close.index[i].strftime("%Y-%m-%d") for i in idx],
        "c": [sig(v) for v in close.values[idx]], "m": [sig(v) for v in fit],
        "rev3m": None if rev3m is None or pd.isna(rev3m) else round(rev3m * 100, 1),
    }


def add_fundamentals(markets):
    """EPS / revenue history and estimates (the Stock Lookup's load_info, from the revision CSVs) and weekly actual
    prices, for the price-vs-earnings chart. Actual rather than dividend-adjusted prices, as on the Stock Lookup:
    adjusting lowers past prices and would understate past P/Es."""
    print("=== Fundamentals ===")
    info = load_info()
    syms = [r["_sym"] for m in markets for r in m["rows"]]
    _, actual = download_both(syms, FUND_YEARS)
    for m in markets:
        for r in m["rows"]:
            sym = r.pop("_sym")
            r["fund"] = info.get((m["key"], norm(sym)), {}).get("fund")
            w = actual[sym].dropna().resample("W-FRI").last().dropna() if sym in actual else pd.Series(dtype=float)
            r["ps"] = w.index[0].strftime("%Y-%m-%d") if len(w) else None
            r["p"] = [float(f"{v:.5g}") for v in w.values] if len(w) else []
    rows = [r for m in markets for r in m["rows"]]
    print(f"  fundamentals for {sum(bool(r['fund']) for r in rows)}/{len(rows)}, "
          f"actual prices for {sum(bool(r['p']) for r in rows)}/{len(rows)}")


def table(rows, bench_label, sid):
    head = ("<thead><tr><th>Ticker</th><th>Name</th><th>Cap</th><th title=\"How cleanly the price has followed its "
            "trend line (1 = perfectly)\">Fit (R²)</th><th title=\"Standard deviations from the trend line\">Distance</th>"
            f"<th>10Y return</th><th>10Y vs {bench_label}</th><th>Trend / yr</th></tr></thead>")
    body = "".join(
        f'<tr><td class="tkr"><a href="#{sid}-{escape(r["t"])}">{escape(r["t"])}</a></td>'
        f'<td class="name" title="{escape(r["n"])}">{escape(r["n"])}</td><td>{r["cap"]}</td><td>{r["r2"]:.2f}</td>'
        f'<td class="{"down" if r["z"] < 0 else "up"}">{r["z"]:+.1f}σ</td><td>{r["ret"]:+,}%</td>'
        f'<td class="up">+{r["out"]:,}%</td><td>{r["trend"]:.1f}%</td></tr>'
        for r in rows)
    return f'<div class="tbl"><table>{head}<tbody>{body}</tbody></table></div>'


def build_body(markets):
    nav, parts = [], []
    for m in markets:
        for pos, title in (("Bottom", "bottom of its channel"), ("Top", "top of its channel")):
            rows = [r for r in m["rows"] if r["pos"] == pos]
            sid = f'{m["key"]}-{pos.lower()}'
            nav.append(f'<a href="#{sid}">{m["label"]}: {pos.lower()} ({len(rows)})</a>')
            parts.append(f'<div class="section" id="{sid}"><h2>{m["label"]}: at the {title}</h2>'
                         f'<p class="sub">{len(rows)} names, best trend fit first &middot; benchmark {m["bench"]}</p></div>')
            if not rows:
                parts.append('<p class="empty">No names passed every test today.</p>')
                continue
            parts.append(table(rows, m["bench"], sid))
            parts += [f'<div class="spot" id="{sid}-{escape(r["t"])}" data-cap="{r["cap"]}" data-m="{m["key"]}" '
                      f'data-i="{m["rows"].index(r)}"></div>' for r in rows]
    return '<div class="nav">' + "".join(nav) + "</div>" + "\n".join(parts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cdn = run_market(ca, "cdn", "Canada", "XIC", ca.to_yf_ticker, lambda t: t[:-3] if t.endswith(".TO") else t,
                     ca.load_sector_map())
    usa = run_market(us, "us", "US", us.BENCH_TICKER, lambda t: t.upper().replace(".", "-"), lambda t: t,
                     us.load_koyfin_sector_map(us.KOYFIN_US_PATH))
    markets = [cdn, usa]
    add_fundamentals(markets)
    data = {m["key"]: m["rows"] for m in markets}
    asof = max(m["asof"] for m in markets)
    html = (PAGE.replace("%%BODY%%", build_body(markets))
            .replace("%%DATA%%", json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"))
            .replace("%%LOGO%%", LOGO_B64).replace("%%CAP_JS%%", CAP_FILTER_JS)
            .replace("%%CAP_CONTROL%%", CAP_FILTER_CONTROL_HTML).replace("%%FULLSCREEN%%", FULLSCREEN_HTML)
            .replace("%%ASOF%%", datetime.strptime(asof, "%Y-%m-%d").strftime("%B %d, %Y"))
            .replace("%%SIGMA%%", f"{us.CHANNEL_SIGMA:.0f}").replace("%%ZTHRESH%%", f"{us.TOP_Z_THRESHOLD:g}"))
    out_path = os.path.join(OUTPUT_DIR, "Channel_Screener.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Regression Channel Screener</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
%%CAP_JS%%
<style>
  body { background: #FFFFFF; color: #363636; font-family: Arial, sans-serif; font-size: 16px; line-height: 1.55; margin: 0; padding: 0 0 32px; }
  header { padding: 18px 16px 12px; border-bottom: 1px solid #E6E6E6; }
  header h1 { margin: 0 0 6px; font-size: 28px; }
  header .meta { color: #555555; font-size: 15px; }
  .intro { padding: 12px 16px 0; }
  .intro p { margin: 0 0 8px; }
  .intro .legend { font-size: 15px; color: #555555; margin-top: 16px; }
  .cap-filter { margin-top: 12px; font-size: 15px; }
  .cap-filter label { color: #555555; margin-right: 6px; }
  .cap-filter select { background: #FFFFFF; color: #363636; border: 1px solid #CFCFCF; border-radius: 4px; padding: 4px 10px; font-size: 15px; }
  .nav { display: flex; flex-wrap: wrap; gap: 6px 18px; padding: 12px 16px 0; font-size: 15px; }
  .nav a, .tbl a { color: #1F79BE; text-decoration: none; }
  .nav a:hover, .tbl a:hover { text-decoration: underline; }
  .section { padding: 22px 16px 0; }
  .section h2 { margin: 0; font-size: 23px; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }
  .section .sub { margin: 6px 0 0; color: #555555; font-size: 15px; }
  .empty { color: #555555; padding: 8px 16px; }
  .tbl { padding: 12px 16px 8px; overflow-x: auto; }
  .tbl table { border-collapse: collapse; font-size: 14px; min-width: 700px; width: 100%; }
  .tbl th { text-align: left; font-size: 12px; color: #555555; text-transform: uppercase; letter-spacing: .03em;
            padding: 6px 8px; border-bottom: 2px solid #C67A29; white-space: nowrap; }
  .tbl td { padding: 5px 8px; border-bottom: 1px solid #E6E6E6; white-space: nowrap; }
  .tbl td.name { max-width: 260px; overflow: hidden; text-overflow: ellipsis; }
  .tbl td.tkr { font-weight: bold; }
  .tbl tbody tr:nth-child(even) { background: #F7F7F7; }
  .up { color: #44A660; font-weight: bold; } .down { color: #A22A2A; font-weight: bold; }
  .spot { border-top: 1px solid #E6E6E6; margin: 10px 16px 0; padding-top: 12px; min-height: 470px; scroll-margin-top: 8px; }
  .spot .hd .tk { font-size: 22px; font-weight: bold; color: #1F79BE; }
  .spot .hd .nm { font-size: 17px; margin-left: 6px; }
  .spot .hd .si { color: #555555; font-size: 14px; }
  .spot .stats { display: flex; flex-wrap: wrap; gap: 4px 22px; font-size: 14px; color: #555555; margin-top: 4px; }
  .spot .stats b { color: #363636; } .spot .stats b.up { color: #44A660; } .spot .stats b.down { color: #A22A2A; }
  .pair { display: flex; gap: 12px; align-items: flex-start; }
  .pair > div { flex: 1 1 0; min-width: 0; }
  @media (max-width: 900px) { .pair { display: block; } }
  .fund-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 14px; min-height: 32px; }
  .toggle button { border: 1px solid #CFCFCF; background: #FFFFFF; color: #363636; padding: 3px 12px; font-size: 13px; cursor: pointer; }
  .toggle button:first-child { border-radius: 4px 0 0 4px; } .toggle button:last-child { border-radius: 0 4px 4px 0; border-left: none; }
  .toggle button.on { background: #1F79BE; border-color: #1F79BE; color: #FFFFFF; }
  .toggle button:disabled { color: #B5B5B5; cursor: default; }
  .growth { font-size: 13px; color: #555555; } .growth b { color: #363636; }
  @media (max-width: 900px) { .fund-bar.pad { display: none; } }
  .norev { color: #555555; font-size: 14px; padding: 60px 16px; text-align: center; }
  .source { color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }
</style>
</head>
<body>
<header>
  <h1>Regression Channel Screener</h1>
  <div class="meta">Prices as of %%ASOF%% &middot; TSX, S&amp;P 500, Nasdaq 100 and other US stocks with analyst coverage</div>
</header>
<div class="intro">
  <p>These are stocks that have climbed in a steady, well-defined uptrend for the past 10 years, beaten their
  market over that time, and are now at an unusual point in that trend. A straight line fitted through ten years of
  prices (on a log scale) marks the trend; the shaded channel around it covers the stock's normal swings. Names near
  the <b>bottom</b> of their channel have pulled back within an intact uptrend, which can be a chance to buy a
  long-term winner on a dip. Names near the <b>top</b> have run well ahead of their trend, which can be a time to
  take some profits or wait for a better entry. Neither is a forecast: a trend can break.</p>
  <p class="legend">Fit (R²): how closely the price has tracked its trend line, from 0 to 1; higher means a cleaner,
  steadier trend, and each list is ranked by it. Distance: how far the price sits from the trend line, in standard
  deviations; every name listed is at least %%ZTHRESH%% away. 10Y vs benchmark: the stock's 10-year return minus
  XIC's for Canada, or QQQ's for the US. Trend / yr: the yearly growth rate of the trend line. The shaded channel on
  each chart is &plusmn;%%SIGMA%% standard deviations.</p>
  <p class="legend">The chart on the right asks whether the business has kept pace with the price. It plots the share
  price against earnings per share (or revenue) for the last eleven years, with analysts' estimates for the next
  three dashed. The blue line is scaled by the stock's median price-to-earnings (or price-to-sales) ratio, so a
  price above it means the stock trades above its usual valuation, and below it, cheaper than usual. A stock at the
  bottom of its channel with earnings still rising has become cheaper; one whose earnings fell with it may be
  breaking its trend.</p>
  %%CAP_CONTROL%%
</div>
%%BODY%%
<div class="source">Source: 5i Research, Koyfin, Yahoo Finance.</div>
<script>
const DATA = %%DATA%%;
const LOGO = "%%LOGO%%";
const BLUE = "#1F79BE", ORANGE = "#C67A29", INK = "#363636", MUTED = "#555555", GRID = "#E6E6E6", RED = "#A22A2A";
const AX = { gridcolor: GRID, zeroline: false, linecolor: "#CFCFCF", tickfont: { size: 12, color: MUTED } };
const CFG = { responsive: true, displaylogo: false };
const BASE = { paper_bgcolor: "#FFF", plot_bgcolor: "#FFF", font: { family: "Arial", color: INK, size: 13 },
               hoverlabel: { bgcolor: "#FFFFFF", bordercolor: GRID, font: { color: INK } } };
const esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };

function logTicks(lo, hi) {
  const t = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    [1, 2, 5].forEach(function (m) { const v = m * Math.pow(10, e); if (v >= lo * 0.9 && v <= hi * 1.1) t.push(v); });
  if (t.length < 3) {
    const raw = (hi - lo) / 5, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag;
    t.length = 0;
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) t.push(v);
  }
  return t;
}

// Log price axis with the currency written into each label: with tickvals set, Plotly drops tickprefix
function priceTicks(vals, cur) {
  const t = logTicks(Math.min.apply(null, vals), Math.max.apply(null, vals));
  return { tickvals: t, ticktext: t.map(function (v) { return cur + v.toLocaleString("en-US", { maximumFractionDigits: v < 10 ? 2 : 0 }); }) };
}

function render(el) {
  const m = el.dataset.m, r = DATA[m][+el.dataset.i], bench = m === "cdn" ? "XIC" : "QQQ";
  const info = [r.sector, r.industry].filter(Boolean).join(" · ");
  el.innerHTML = '<div class="hd"><span class="tk">' + esc(r.t) + '</span><span class="nm">' + esc(r.n) + "</span>" +
    (info ? '<div class="si">' + esc(info) + "</div>" : "") + '</div><div class="stats">' +
    "<span>Fit (R²) <b>" + r.r2.toFixed(2) + "</b></span><span>Distance <b class=\"" + (r.z < 0 ? "down" : "up") + "\">" +
    (r.z > 0 ? "+" : "") + r.z.toFixed(1) + "σ</b></span><span>10Y return <b>" + (r.ret > 0 ? "+" : "") +
    r.ret.toLocaleString("en-US") + "%</b></span><span>10Y vs " + bench + ' <b class="up">+' +
    r.out.toLocaleString("en-US") + "%</b></span><span>Trend <b>" + r.trend.toFixed(1) + "% / yr</b></span>" +
    (r.rev3m === null ? "" : '<span title="Change in the average analyst revenue estimate for the next fiscal year">' +
     'Revenue est., 3 mo <b class="' + (r.rev3m >= 0 ? "up" : "down") + '">' + (r.rev3m >= 0 ? "+" : "") + r.rev3m.toFixed(1) + "%</b></span>") +
    '</div><div class="pair"><div><div class="fund-bar pad"></div><div class="ch"></div></div>' +
    '<div><div class="fund-bar"><span class="toggle"><button data-k="eps">EPS</button><button data-k="rev">Revenue</button></span>' +
    '<span class="growth"></span></div><div class="fd"></div></div></div>';
  const ch = el.querySelector(".ch");
  const up = r.m.map(function (v) { return v * r.band; }), lo = r.m.map(function (v) { return v / r.band; });
  const last = r.c.length - 1, all = r.c.concat(up, lo);
  const H = 400, T = 44, B = 70;
  Plotly.newPlot(ch, [
    { x: r.d, y: up, mode: "lines", line: { color: "#9A9A9A", width: 1, dash: "dot" }, hoverinfo: "skip", showlegend: false },
    { x: r.d, y: lo, mode: "lines", line: { color: "#9A9A9A", width: 1, dash: "dot" }, fill: "tonexty",
      fillcolor: "rgba(31,121,190,0.06)", hoverinfo: "skip", showlegend: false },
    { x: r.d, y: r.m, mode: "lines", name: "Trend", line: { color: ORANGE, width: 1.6, dash: "dash" },
      hovertemplate: "Trend: $%{y:,.2f}<extra></extra>" },
    { x: r.d, y: r.c, mode: "lines", name: "Price", line: { color: BLUE, width: 1.8 },
      hovertemplate: "Price: $%{y:,.2f}<extra></extra>" },
    { x: [r.d[last]], y: [r.c[last]], mode: "markers", showlegend: false, hoverinfo: "skip",
      marker: { color: RED, size: 9, line: { color: "#FFFFFF", width: 1.5 } } },
  ], Object.assign({}, BASE, {
    height: H, margin: { t: T, b: B, l: 64, r: 16 }, hovermode: "x unified", showlegend: false,
    title: { text: "10-year trend channel (log scale)", x: 0.01, font: { size: 14, color: MUTED } },
    xaxis: Object.assign({ type: "date" }, AX),
    yaxis: Object.assign({ type: "log" }, priceTicks(all, m === "cdn" ? "C$" : "US$"), AX),
    images: [{ source: LOGO, xref: "paper", yref: "paper", x: 1, y: 1 + (T - 8) / (H - T - B), sizex: 0.2,
               sizey: 30 / (H - T - B), xanchor: "right", yanchor: "top" }],
  }), CFG);
  const bar = el.querySelectorAll(".fund-bar")[1];
  let key = r.fund && !r.fund.eps ? "rev" : "eps";
  bar.querySelectorAll("button").forEach(function (b) {
    b.addEventListener("click", function () { if (!b.disabled) { key = b.dataset.k; drawFund(el, r, key); } });
  });
  drawFund(el, r, key);
}

// Price vs EPS / revenue: the Stock Lookup's chart (renderFund there), drawn per card
function cagr(a, b, n) { return a > 0 && b > 0 ? Math.pow(b / a, 1 / n) - 1 : null; }
function nums(a) { return a.filter(function (v) { return v !== null && isFinite(v); }); }
function weeks(start, n) {
  const out = [], d0 = Date.parse(start + "T00:00:00Z");
  for (let i = 0; i < n; i++) out.push(new Date(d0 + i * 7 * 864e5).toISOString().slice(0, 10));
  return out;
}
function drawFund(el, r, key) {
  const div = el.querySelector(".fd"), g = el.querySelector(".growth"), f = r.fund;
  const btns = el.querySelectorAll(".toggle button");
  if (!f || (!f.eps && !f.rev) || !r.p.length) {
    div.innerHTML = '<div class="norev">No reported results or estimates on file for ' + esc(r.t) + "</div>";
    btns.forEach(function (b) { b.disabled = true; }); return;
  }
  btns.forEach(function (b) { b.disabled = !f[b.dataset.k]; b.classList.toggle("on", b.dataset.k === key); });
  const hist = f[key][0], est = f[key][1], eps = key === "eps", n = hist.length, last = hist[n - 1];

  const grow = [["10-yr", cagr(hist[0], last, 10)], ["5-yr", cagr(hist[n - 6], last, 5)], ["Next 3 yrs (est.)", cagr(last, est[2], 3)]];
  g.innerHTML = "Growth / yr: " + grow.map(function (x) {
    return x[0] + " <b>" + (x[1] === null ? "n/a" : (x[1] >= 0 ? "+" : "") + (x[1] * 100).toFixed(1) + "%") + "</b>";
  }).join(" &middot; ");

  // Fiscal-year-end dates: latest FY ends at f.fye (else assume December of the last full year)
  const fye = f.fye ? new Date(f.fye + "T00:00:00Z") : new Date(Date.UTC(new Date().getUTCFullYear() - 1, 11, 31));
  const fyDate = function (k) { return new Date(Date.UTC(fye.getUTCFullYear() + k, fye.getUTCMonth() + 1, 0)).toISOString().slice(0, 10); };
  const fyLabel = function (k) { return "FY" + (fye.getUTCFullYear() + k) + (k > 0 ? "E" : ""); };
  const hx = hist.map(function (_, i) { return fyDate(i - (n - 1)); }), ex = [1, 2, 3].map(fyDate);
  const hl = hist.map(function (_, i) { return fyLabel(i - (n - 1)); }), el2 = [1, 2, 3].map(fyLabel);

  // Weekly actual prices, from the first fiscal year on
  const px = weeks(r.ps, r.p.length), P = r.p;
  const priceAt = function (iso) { let v = null; for (let i = 0; i < px.length && px[i] <= iso; i++) v = P[i]; return v; };
  const p0 = px.findIndex(function (x) { return x >= hx[0]; });
  const pxX = px.slice(Math.max(0, p0 - 8)), pxY = P.slice(Math.max(0, p0 - 8));

  // Scale: median of price / value at each fiscal year end with a positive value (median P/E or P/"sales")
  const ratios = [];
  hist.forEach(function (v, i) { const pr = priceAt(hx[i]); if (v > 0 && pr) ratios.push(pr / v); });
  ratios.sort(function (a, b) { return a - b; });
  const k = ratios.length ? ratios[Math.floor(ratios.length / 2)] : null;
  const scaled = function (a) { return a.map(function (v) { return v === null || k === null ? null : v * k; }); };
  const hs = scaled(hist), es = scaled(est);
  const allPos = nums(hs.concat(es, pxY)).every(function (v) { return v > 0; });

  const unitName = eps ? "EPS" : "Revenue";
  const big = !eps && Math.max.apply(null, nums(hist.concat(est)).map(Math.abs)) >= 1000;
  const val = function (v) {
    return v === null ? "n/a" : eps ? f.cur + v.toFixed(2) : f.cur + (big ? (v / 1000).toFixed(1) + "B" : Math.round(v).toLocaleString("en-US") + "M");
  };
  // Hover: value plus year-over-year change (not annualized); no % from a loss
  const seq = hist.concat(est), seqLabels = hl.concat(el2);
  const yoy = function (i) {
    if (i === 0 || seq[i] === null || seq[i - 1] === null) return "";
    const a = seq[i - 1], b = seq[i], vs = " vs " + seqLabels[i - 1];
    if (a > 0) { const c = (b / a - 1) * 100; return " · " + (c >= 0 ? "+" : "") + c.toFixed(1) + "%" + vs; }
    return b > 0 ? " · back to a profit from a loss" : " · n/a (loss" + (a < 0 ? " both years)" : ")");
  };
  const hover = function (i) { return seqLabels[i] + ": " + val(seq[i]) + yoy(i); };
  const note = k === null ? "" : (eps ? "EPS × " + k.toFixed(1) + " (median P/E)" : "revenue scaled to price (median ratio)");
  const all = nums(hs.concat(es, pxY));
  Plotly.react(div, [
    { x: pxX, y: pxY, mode: "lines", name: "Share price", line: { color: INK, width: 1.5 },
      hovertemplate: "Price: " + f.cur + "%{y:,.2f}<extra></extra>" },
    { x: hx, y: hs, mode: "lines+markers", name: unitName + " (reported)", line: { color: BLUE, width: 3 },
      marker: { size: 7, color: BLUE }, customdata: hist.map(function (_, i) { return hover(i); }),
      hovertemplate: "%{customdata}<extra></extra>" },
    { x: [hx[n - 1]].concat(ex), y: [hs[n - 1]].concat(es), mode: "lines+markers", name: unitName + " (estimate)",
      line: { color: BLUE, width: 3, dash: "dash" }, marker: { size: [0, 7, 7, 7], color: "#FFFFFF", line: { color: BLUE, width: 2 } },
      customdata: [""].concat(est.map(function (_, i) { return hover(n + i); })), hovertemplate: "%{customdata}<extra></extra>" },
  ], Object.assign({}, BASE, {
    height: 400, margin: { t: 44, b: 70, l: 64, r: 16 }, hovermode: "closest",
    title: { text: "Price vs. " + (eps ? "earnings per share" : "revenue"), x: 0.01, font: { size: 14, color: MUTED } },
    legend: { orientation: "h", x: 0, y: -0.1, yanchor: "top", font: { size: 12 } },
    xaxis: Object.assign({ type: "date", range: [fyDate(-n), ex[2]] }, AX),
    yaxis: Object.assign({ type: allPos ? "log" : "linear" },
                         allPos ? priceTicks(all, f.cur) : { tickprefix: f.cur, tickformat: ",.0f" }, AX),
    annotations: note ? [{ xref: "paper", yref: "paper", x: 0.01, y: 0.99, xanchor: "left", yanchor: "top", showarrow: false,
                           text: "Blue line: " + note, font: { size: 11, color: MUTED }, bgcolor: "rgba(255,255,255,0.8)" }] : [],
  }), CFG);
}

// Draw each card when it nears the screen: a few hundred charts at once would stall the page
const spots = document.querySelectorAll(".spot");
if ("IntersectionObserver" in window) {
  const io = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { io.unobserve(e.target); render(e.target); } });
  }, { rootMargin: "800px 0px" });
  spots.forEach(function (s) { io.observe(s); });
} else spots.forEach(render);
</script>
%%FULLSCREEN%%
</body>
</html>
"""


if __name__ == "__main__":
    main()
