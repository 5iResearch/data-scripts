"""
Sector Head-to-Heads (website page): inside each sector, which stocks have consistently beaten which.

Every stock is compared with every other stock in its sector over rolling 12-month windows, one starting at each
month-end for the last ten years (total-return closes). For each pair: the share of windows in which the row stock
had the higher 12-month return. A stock's "beat the sector" score is the average of its row, i.e. its average
head-to-head win rate against its sector peers (50% = average).

Same method as the local study in Final Versions/Sector Industry Analysis/Sector Dominance Study (core.py), which
also back-tested whether a high score predicts the next 12 months (2011-2025: weakly in the US, not in Canada).

Universes: Canada = the revision-screener CSV plus the Koyfin Canada export (which adds interlisted names such as
SHOP, AEM, WCN), over C$1B, with their sectors; US = S&P 500 (GICS sectors) plus Nasdaq-100 names outside it (ICB industry mapped to a GICS sector).
Output: outputs/sector-dominance/Sector_Dominance.html
"""

import json
import os

import numpy as np
import pandas as pd
import yfinance as yf

from common_screening import _sp500_table, load_nasdaq100_table
from generate_index_rsi_web import FULLSCREEN_HTML, LOGO_B64

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "sector-dominance")
CDN_CSV = os.path.join(REPO_ROOT, "data", "cdn_1w_rev_est_screener.csv")
KOYFIN_CSV = os.path.join(REPO_ROOT, "data", "koyfin_cdn.csv")

YEARS = 10
WIN = 12              # months per head-to-head window
MIN_WIN = 36          # a pair needs this many shared windows; a stock needs this many windows to be scored
DOM = 0.70            # "dominates" a peer = beat it in at least 70% of shared windows
SHORT = 60            # fewer windows than this = flagged as short history on the page
CDN_MIN_CAP = 1000    # C$ millions
CDN_SKIP = {"IMAX"}   # IMAX.TO is a Hamilton ETF, not IMAX Corp
US_DUP = {"GOOG", "FOX", "NWS"}   # second share classes: keep GOOGL, FOXA, NWSA so a company isn't its own peer
ICB_TO_GICS = {"Technology": "Information Technology", "Telecommunications": "Communication Services",
               "Health Care": "Health Care", "Consumer Discretionary": "Consumer Discretionary",
               "Consumer Staples": "Consumer Staples", "Industrials": "Industrials", "Energy": "Energy",
               "Utilities": "Utilities", "Financials": "Financials", "Real Estate": "Real Estate",
               "Basic Materials": "Materials"}
SECTOR_SHORT = {"Information Technology": "Technology", "Communication Services": "Communication",
                "Consumer Discretionary": "Consumer Disc."}


def download(tickers, chunk=200):
    out = []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        print(f"  downloading {i + 1}-{i + len(part)} of {len(tickers)}")
        raw = yf.download(part, period=f"{YEARS + 2}y", auto_adjust=True, threads=True, progress=False)["Close"]
        out.append(raw.to_frame(part[0]) if isinstance(raw, pd.Series) else raw)
    df = pd.concat(out, axis=1)
    return df.loc[:, ~df.columns.duplicated()]


def cdn_candidates(t):
    """The revision CSV drops exchange and punctuation (BBDB = BBD.B, PMZUN = PMZ.UN): TSX first, then the dashed
    class / unit forms Yahoo uses, then TSX Venture."""
    out = [f"{t}.TO"]
    for unit in ("UN", "U", "A", "B", "X"):
        if len(t) > len(unit) + 1 and t.endswith(unit):
            out.append(f"{t[:-len(unit)]}-{unit}.TO")
    return out + [f"{t}.V"]


def universe():
    sp = _sp500_table()
    us = pd.DataFrame({"t": sp["Symbol"].str.replace(".", "-", regex=False), "n": sp["Security"],
                       "sector": sp["GICS Sector"], "sub": sp["GICS Sub-Industry"]})
    ndx = load_nasdaq100_table()
    ndx = ndx[~ndx["Ticker"].str.replace(".", "-", regex=False).isin(us["t"])]
    us = pd.concat([us, pd.DataFrame({"t": ndx["Ticker"].str.replace(".", "-", regex=False), "n": ndx["Company"],
                                      "sector": ndx["Sector"].map(ICB_TO_GICS), "sub": ndx["Sector"]})])
    us = us[~us["t"].isin(US_DUP)].dropna(subset=["sector"]).drop_duplicates("t")
    us["y"] = us["t"]

    # The revision CSV leaves out some big interlisted names (SHOP, AEM, WCN, OTEX); the Koyfin Canada export has them
    c = pd.concat([pd.read_csv(f, converters={"Ticker": lambda v: str(v).strip()}, keep_default_na=False)
                   for f in (CDN_CSV, KOYFIN_CSV)])
    c["Market Cap"] = pd.to_numeric(c["Market Cap"], errors="coerce")
    c = c[(c["Market Cap"] >= CDN_MIN_CAP) & ~c["Ticker"].isin(CDN_SKIP) & (c["Sector"] != "")].drop_duplicates("Ticker")
    cdn = pd.DataFrame({"t": c["Ticker"].str.upper(), "n": c["Name"], "sector": c["Sector"], "sub": c["Industry"],
                        "ref": pd.to_numeric(c["Last Price"], errors="coerce")})
    return us, cdn


def resolve_cdn(cdn):
    """Pick the first Yahoo symbol with recent prices for each Canadian ticker; show it TSX-style (BBD.B)."""
    cands = {t: cdn_candidates(t) for t in cdn["t"]}
    recent = yf.download(sorted({s for v in cands.values() for s in v}), period="1mo", auto_adjust=True,
                         threads=True, progress=False)["Close"]
    ref = dict(zip(cdn["t"], cdn["ref"]))

    def same_stock(t, close):
        """The symbol must trade near the CSV's last price (in C$, or US$ for interlisted names): LULU.TO is a
        $4 receipt, not lululemon."""
        r = close / ref[t] if ref.get(t) else 1.0
        return 0.55 <= r <= 2.3

    ok = {s for s in recent.columns if recent[s].notna().sum() >= 5}
    last = {s: recent[s].dropna().iloc[-1] for s in ok}
    pick = {t: next((s for s in v if s in ok and same_stock(t, last[s])), None) for t, v in cands.items()}
    for t in [t for t, s in pick.items() if s is None]:   # big batches sometimes drop names: retry one at a time
        for sym in cands[t]:
            hit = yf.download(sym, period="1mo", auto_adjust=True, progress=False)["Close"].squeeze()
            if isinstance(hit, pd.Series) and hit.notna().sum() >= 5 and same_stock(t, hit.dropna().iloc[-1]):
                pick[t] = sym
                break
    missing = [t for t, s in pick.items() if s is None]
    if missing:
        print(f"  no matching Yahoo price for {len(missing)} Canadian tickers (US-only listings etc.): {', '.join(missing)}")
    cdn = cdn[cdn["t"].map(pick).notna()].copy()
    cdn["y"] = cdn["t"].map(pick)
    cdn["t"] = cdn["y"].str.replace(r"\.(TO|V)$", "", regex=True).str.replace("-", ".")
    return cdn


def despike(px, jump=0.8, back=0.25, days=5):
    """Blank one-off Yahoo glitches: a close >80% away from the prior close that is back within 25% of it inside
    five sessions."""
    px = px.copy()
    for c in px.columns:
        s = px[c].dropna()
        if len(s) < days + 2:
            continue
        v = s.values.copy()
        for i in np.where(np.abs(np.log(v[1:] / v[:-1])) > np.log(1 + jump))[0] + 1:
            for j in range(i + 1, min(i + days, len(v))):
                if abs(np.log(v[j] / v[i - 1])) < np.log(1 + back):
                    v[i:j] = np.nan
                    break
        px.loc[s.index, c] = v
    return px


def restart(px, jump=1.5, penny=2.0):
    """A >150% one-day rise from under $2 that sticks is a shell / reverse takeover / unadjusted consolidation: the
    history before it belongs to a different business, so the series starts that day."""
    r = px.pct_change(fill_method=None).where(px.ffill().shift() < penny)
    for c in px.columns:
        d = r.index[r[c] > jump]
        if len(d):
            print(f"  {c}: history restarted {d[-1]:%Y-%m-%d}")
            px.loc[:d[-1], c] = np.nan
    return px


def pairwise(R):
    a = R.values
    ok = ~np.isnan(a)
    both = ok[:, :, None] & ok[:, None, :]
    n = both.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        W = ((a[:, :, None] > a[:, None, :]) & both).sum(0) / n
    W[n < MIN_WIN] = np.nan
    np.fill_diagonal(W, np.nan)
    return W, n


def r1(v, d=1):
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def build_market(meta, px):
    M = px.resample("ME").last()
    M.index = M.index.where(M.index < px.index[-1], px.index[-1])   # current month stamped at the last close
    R12 = (M / M.shift(WIN) - 1).iloc[-(YEARS * 12 - WIN + 1):]
    out = []
    for sec, g in meta.groupby("sector"):
        cols = [t for t in g["t"] if t in R12.columns and R12[t].notna().sum() >= MIN_WIN]
        R = R12[cols]
        W, N = pairwise(R)
        keep = np.sum(~np.isnan(W), axis=1) >= 3
        cols = [c for c, k in zip(cols, keep) if k]
        if len(cols) < 4:
            continue
        idx = [i for i, k in enumerate(keep) if k]
        W, N, R = W[np.ix_(idx, idx)], N[np.ix_(idx, idx)], R[cols]
        score = np.nanmean(W, axis=1)
        order = np.argsort(-score)
        W, N, cols, score = W[np.ix_(order, order)], N[np.ix_(order, order)], [cols[i] for i in order], score[order]
        med = R.median(axis=1)
        rel = R.sub(med, axis=0)
        info = g.set_index("t")
        rows = []
        for i, t in enumerate(cols):
            m = M[t].dropna()
            cagr = (m.iloc[-1] / m.iloc[-(YEARS * 12 + 1)]) ** (1 / YEARS) - 1 if len(m) > YEARS * 12 else np.nan
            rows.append({"t": t, "n": str(info.loc[t, "n"]), "sub": str(info.loc[t, "sub"]),
                         "s": r1(score[i] * 100), "med": r1(R[t].gt(med).where(R[t].notna()).mean() * 100, 0),
                         "dom": int(np.sum(W[i] >= DOM)), "by": int(np.sum(W[i] <= 1 - DOM)),
                         "peers": int(np.sum(~np.isnan(W[i]))), "win": int(R[t].notna().sum()),
                         "mrel": r1(rel[t].median() * 100, 0), "wrel": r1(rel[t].min() * 100, 0),
                         "cagr": r1(cagr * 100), "r12": r1(R[t].iloc[-1] * 100, 0)})
        out.append({"sector": sec, "short": SECTOR_SHORT.get(sec, sec), "rows": rows,
                    "W": [[None if np.isnan(v) else int(round(v * 100)) for v in row] for row in W],
                    "N": [[int(v) for v in row] for row in N]})
        print(f"  {sec}: {len(cols)} stocks, top {', '.join(cols[:3])}")
    out.sort(key=lambda s: -len(s["rows"]))
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    us, cdn = universe()
    print(f"Universe: {len(us)} US, {len(cdn)} Canadian candidates")
    cdn = resolve_cdn(cdn)
    print(f"Downloading {len(us)} US and {len(cdn)} Canadian stocks...")
    px = download(list(dict.fromkeys(list(us["y"]) + list(cdn["y"]))))
    px = px[px.index < px.index[-1]] if px.iloc[-1].notna().mean() < 0.5 else px   # drop a half-filled "today" row
    # Separate tables per market: TSX-style tickers can match US ones once the suffix is gone (T = Telus vs AT&T)
    cdn_px = restart(despike(px[[y for y in cdn["y"] if y in px]].rename(columns=dict(zip(cdn["y"], cdn["t"])))))
    us_px = restart(despike(px[[y for y in us["y"] if y in px]]))
    data = {"cdn": build_market(cdn, cdn_px), "us": build_market(us, us_px),
            "years": YEARS, "short": SHORT, "dom": int(DOM * 100)}
    asof = px.index[-1].strftime("%B %d, %Y")
    html = (PAGE.replace("%%DATA%%", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
            .replace("%%LOGO%%", LOGO_B64).replace("%%ASOF%%", asof).replace("%%FULLSCREEN%%", FULLSCREEN_HTML))
    out = os.path.join(OUTPUT_DIR, "Sector_Dominance.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sector Head-to-Heads</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { background: #FFFFFF; color: #363636; font-family: Arial, sans-serif; font-size: 16px; line-height: 1.55; margin: 0; padding: 0 0 32px; }
  header { padding: 18px 16px 12px; border-bottom: 1px solid #E6E6E6; display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
  header h1 { margin: 0 0 6px; font-size: 28px; }
  header .meta { color: #555555; font-size: 15px; }
  .intro { padding: 12px 16px 0; }
  .intro p { margin: 0 0 8px; }
  .intro .legend { font-size: 15px; color: #555555; margin-top: 16px; }
  .tabs { display: flex; flex-wrap: wrap; gap: 8px 16px; padding: 14px 16px 0; }
  .tabs button { border: 1px solid #CFCFCF; background: #FFFFFF; color: #363636; padding: 7px 22px; font-size: 16px; cursor: pointer; }
  .tabs button:first-child { border-radius: 4px 0 0 4px; } .tabs button:last-child { border-radius: 0 4px 4px 0; border-left: none; }
  .tabs button.on { background: #1F79BE; border-color: #1F79BE; color: #FFFFFF; }
  .pills { display: flex; flex-wrap: wrap; gap: 6px; padding: 12px 16px 0; }
  .pills button { border: 1px solid #CFCFCF; background: #FFFFFF; color: #363636; border-radius: 16px; padding: 5px 13px; font-size: 14px; cursor: pointer; }
  .pills button span { color: #9A9A9A; font-size: 12px; margin-left: 3px; }
  .pills button.on { background: #1F79BE; border-color: #1F79BE; color: #FFFFFF; } .pills button.on span { color: #DCEBF7; }
  .section { padding: 22px 16px 0; }
  .section h2 { margin: 0; font-size: 23px; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }
  .section .sub { margin: 6px 0 0; color: #555555; font-size: 15px; max-width: 1100px; }
  .cols { display: grid; grid-template-columns: minmax(0, 3fr) minmax(280px, 2fr); gap: 8px 20px; padding: 0 16px; align-items: start; }
  @media (max-width: 900px) { .cols { grid-template-columns: minmax(0, 1fr); } }
  .pick { padding: 10px 0 0; font-size: 14px; color: #555555; }
  .pick select { font: inherit; font-size: 15px; padding: 4px 6px; border: 1px solid #CFCFCF; border-radius: 4px; color: #363636; max-width: 100%; }
  .card { border: 1px solid #E6E6E6; border-radius: 4px; padding: 10px 12px; margin-top: 10px; }
  .card h3 { margin: 0; font-size: 18px; } .card h3 span { color: #555555; font-weight: normal; font-size: 14px; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 10px 0 2px; }
  .stats div { background: #F7F7F7; border-radius: 4px; padding: 6px 8px; }
  .stats b { display: block; font-size: 20px; } .stats small { color: #555555; font-size: 12px; line-height: 1.3; display: block; }
  .tbl { padding: 12px 16px 4px; overflow-x: auto; }
  .tbl table { border-collapse: collapse; font-size: 14px; width: 100%; min-width: 860px; }
  .tbl th { text-align: left; font-size: 12px; color: #555555; text-transform: uppercase; letter-spacing: .03em; padding: 6px 8px;
            border-bottom: 2px solid #C67A29; vertical-align: bottom; cursor: pointer; user-select: none; }
  .tbl th.num, .tbl td.num { text-align: right; }
  .tbl td { padding: 5px 8px; border-bottom: 1px solid #E6E6E6; white-space: nowrap; }
  .tbl tbody tr:nth-child(even) { background: #F7F7F7; }
  .tbl tbody tr { cursor: pointer; } .tbl tbody tr.sel { outline: 2px solid #1F79BE; outline-offset: -2px; }
  .tbl td.tk { font-weight: bold; color: #1F79BE; }
  .tbl td.nm { color: #555555; max-width: 240px; overflow: hidden; text-overflow: ellipsis; }
  .bar { display: inline-block; width: 80px; height: 8px; background: #E6E6E6; border-radius: 4px; overflow: hidden; vertical-align: middle; margin-left: 6px; }
  .bar i { display: block; height: 100%; }
  .up { color: #44A660; font-weight: bold; } .down { color: #A22A2A; font-weight: bold; }
  .flag { color: #C67A29; font-size: 11px; font-weight: bold; margin-left: 4px; }
  .source { color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }
</style>
</head>
<body>
<header>
  <div><h1>Sector Head-to-Heads</h1>
  <div class="meta">Prices as of %%ASOF%% &middot; which stocks have consistently beaten which, inside each sector</div></div>
</header>
<div class="intro">
  <p>A stock can lead its sector over ten years on the strength of one great run. This page asks a stricter question:
  how often has it come out ahead? Every stock is matched against every other stock in its sector, once for each
  12-month window starting at a month-end over the past ten years, and we count who had the better return. A name
  that wins most of those match-ups has outperformed its peers steadily, not just once.</p>
  <p>Treat it as a record of the past, not a forecast. When we tested it back to 2011, a high score did little to
  predict the next 12 months in the US and nothing in Canada, where mining and energy leaders tend to swap places
  with each commodity cycle.</p>
  <p class="legend">Beat the sector: the stock's average win rate against its sector peers (50% is average).
  Dominates: peers it beat in at least 70% of windows; Dominated by: peers that beat it that often.
  Median / worst 12 mo vs sector: its 12-month return minus the sector median, typical and worst window.
  <span class="flag">short</span> marks stocks with under five years of history, judged on fewer windows.</p>
</div>
<div class="tabs"><span id="mt"><button data-m="cdn" class="on">Canada</button><button data-m="us">US</button></span></div>
<div class="pills" id="pills"></div>
<div id="page"></div>
<div class="source">Source: 5i Research, Yahoo Finance. Total returns (dividends reinvested). Canada: TSX stocks over C$1 billion;
US: S&amp;P 500 and Nasdaq-100. Today's members only, so stocks that dropped out over the decade are missing.</div>
<script>
const DATA = %%DATA%%;
const BLUE = "#1F79BE", ORANGE = "#C67A29", INK = "#363636", MUTED = "#555555", GRID = "#E6E6E6";
const esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
const pct = function (v, d) { return v === null || v === undefined ? "&mdash;" : '<span class="' + (v >= 0 ? "up" : "down") + '">' + (v >= 0 ? "+" : "") + v.toFixed(d || 0) + "%</span>"; };
const scol = function (s) { return s >= 60 ? BLUE : s <= 40 ? ORANGE : "#9A9A9A"; };
let MKT = "cdn", SEC = 0, SEL = 0, SORT = null;
try { const q = JSON.parse(localStorage.getItem("sdom") || "{}"); if (q.m) MKT = q.m; if (q.s !== undefined) SEC = q.s; } catch (e) {}

function save() { try { localStorage.setItem("sdom", JSON.stringify({ m: MKT, s: SEC })); } catch (e) {} }

function pills() {
  const D = DATA[MKT];
  if (SEC >= D.length) SEC = 0;
  document.getElementById("pills").innerHTML = D.map(function (s, i) {
    return '<button data-i="' + i + '"' + (i === SEC ? ' class="on"' : "") + ">" + esc(s.sector) + "<span>" + s.rows.length + "</span></button>";
  }).join("");
  document.querySelectorAll("#pills button").forEach(function (b) {
    b.addEventListener("click", function () { SEC = +b.dataset.i; SEL = 0; SORT = null; save(); pills(); render(); });
  });
}

function render() {
  const S = DATA[MKT][SEC], R = S.rows, top = R[0];
  let h = '<div class="section"><h2>' + esc(S.sector) + ": " + R.length + ' stocks</h2><p class="sub">Rows and columns run from the most consistent ' +
    "stock to the least. A blue cell means the row stock beat the column stock in most 12-month windows; orange, that it mostly lost. " +
    "A mostly blue row is a stock that has out-run nearly every peer, most of the time. Hover a cell for the record; click a row to see one stock's match-ups.</p></div>";
  h += '<div class="cols"><div id="heat"></div><div><div class="pick">Stock: <select id="sel">' +
       R.map(function (r, i) { return '<option value="' + i + '">' + r.t + " &middot; " + esc(r.n) + "</option>"; }).join("") +
       '</select></div><div id="card"></div></div></div>';
  h += '<div class="section"><h2>Ranking</h2><p class="sub">Click a column to sort; click a row to see that stock\'s match-ups above.</p></div>';
  h += '<div class="tbl" id="tbl"></div>';
  document.getElementById("page").innerHTML = h;
  heat();
  table();
  card();
  document.getElementById("sel").addEventListener("change", function (e) { choose(+e.target.value); });
}

function choose(i) {
  SEL = i;
  document.getElementById("sel").value = i;
  card();
  document.querySelectorAll("#tbl tbody tr").forEach(function (tr) { tr.classList.toggle("sel", +tr.dataset.i === i); });
}

function heat() {
  const S = DATA[MKT][SEC], t = S.rows.map(function (r) { return r.t; }), n = t.length, fs = n > 45 ? 9 : n > 25 ? 10 : 11;
  const text = S.W.map(function (row, i) { return row.map(function (v, j) {
    return v === null ? (i === j ? "" : t[i] + " vs " + t[j] + ": not enough shared history")
      : "<b>" + t[i] + "</b> beat <b>" + t[j] + "</b><br>in " + v + "% of " + S.N[i][j] + " shared 12-month windows"; }); });
  const el = document.getElementById("heat"), w = el.clientWidth || 700, side = Math.min(w, Math.max(360, n * 15 + 90));
  Plotly.newPlot(el, [{ type: "heatmap", z: S.W, x: t, y: t, zmin: 0, zmax: 100, xgap: 1, ygap: 1, text: text,
    hovertemplate: "%{text}<extra></extra>", colorscale: [[0, ORANGE], [0.5, "#F7F7F7"], [1, BLUE]],
    colorbar: { thickness: 10, len: 0.6, ticksuffix: "%", tickfont: { size: 11, color: MUTED }, title: { text: "row<br>beat<br>column", font: { size: 11, color: MUTED } } } }],
    { height: side + 30, margin: { t: 10, b: 60, l: 60, r: 10 }, paper_bgcolor: "#FFF", plot_bgcolor: "#FFF",
      font: { family: "Arial", color: INK, size: 12 },
      xaxis: { tickangle: -90, tickfont: { size: fs }, showgrid: false, ticks: "", fixedrange: true, side: "bottom" },
      yaxis: { autorange: "reversed", tickfont: { size: fs }, showgrid: false, ticks: "", fixedrange: true } },
    { responsive: true, displaylogo: false, displayModeBar: false });
  el.on("plotly_click", function (e) { if (e.points && e.points.length) choose(t.indexOf(e.points[0].y)); });
}

function card() {
  const S = DATA[MKT][SEC], r = S.rows[SEL], row = S.W[SEL], t = S.rows.map(function (x) { return x.t; });
  const all = row.map(function (v, j) { return { t: t[j], v: v, n: S.N[SEL][j] }; }).filter(function (p) { return p.v !== null; })
                 .sort(function (a, b) { return b.v - a.v; });
  const cut = all.length > 24, peers = cut ? all.slice(0, 10).concat([{ t: "⋮", v: null, n: 0 }], all.slice(-10)) : all;
  let h = '<div class="card"><h3>' + r.t + " <span>" + esc(r.n) + (r.win < DATA.short ? '<span class="flag">short</span>' : "") + "</span></h3>" +
    '<div class="note" style="color:#555;font-size:13px">' + esc(r.sub) + "</div>" +
    '<div class="stats"><div><b style="color:' + scol(r.s) + '">' + r.s.toFixed(0) + "%</b><small>beat the sector</small></div>" +
    "<div><b>" + r.dom + " / " + r.peers + "</b><small>peers it dominates</small></div>" +
    "<div><b>" + r.by + "</b><small>peers that dominate it</small></div></div>" +
    '<div id="peerbar"></div>' + (cut ? '<div style="color:#555;font-size:12px">The 10 peers it beat most often and the 10 it beat least often, of ' +
    all.length + '. Hover the heatmap row for the rest.</div>' : "") + "</div>";
  document.getElementById("card").innerHTML = h;
  const ht = Math.max(240, peers.length * 16 + 60);
  Plotly.newPlot("peerbar", [{ type: "bar", orientation: "h", y: peers.map(function (p) { return p.t; }), x: peers.map(function (p) { return p.v; }),
      marker: { color: peers.map(function (p) { return p.v >= 50 ? BLUE : ORANGE; }) },
      customdata: peers.map(function (p) { return p.v === null ? "" : "won " + p.v + "% of " + p.n + " windows"; }), textposition: "none",
      hovertemplate: r.t + " vs %{y}: %{customdata}<extra></extra>" }],
    { height: ht, margin: { t: 26, b: 30, l: 56, r: 10 }, paper_bgcolor: "#FFF", plot_bgcolor: "#FFF", font: { family: "Arial", color: INK, size: 11 },
      title: { text: r.t + " vs each peer: share of windows won", x: 0, font: { size: 13 } },
      xaxis: { range: [0, 100], ticksuffix: "%", gridcolor: GRID, fixedrange: true },
      yaxis: { autorange: "reversed", tickfont: { size: 11 }, fixedrange: true, type: "category" },
      shapes: [{ type: "line", x0: 50, x1: 50, yref: "paper", y0: 0, y1: 1, line: { color: INK, width: 1, dash: "dot" } }] },
    { responsive: true, displaylogo: false, displayModeBar: false });
}

const COLS = [["#", null, 1], ["Stock", "t", 0], ["Company", "n", 0], ["Industry", "sub", 0], ["Beat the sector", "s", 1], ["Beat sector median", "med", 1],
              ["Dominates", "dom", 1], ["Dominated by", "by", 1], ["Median 12 mo vs sector", "mrel", 1], ["Worst 12 mo vs sector", "wrel", 1],
              ["10-yr return / yr", "cagr", 1], ["Last 12 mo", "r12", 1]];

function table() {
  const S = DATA[MKT][SEC];
  let idx = S.rows.map(function (_, i) { return i; });
  if (SORT) idx.sort(function (a, b) {
    const x = S.rows[a][SORT.k], y = S.rows[b][SORT.k];
    if (x === null) return 1; if (y === null) return -1;
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * SORT.d;
  });
  let h = "<table><thead><tr>" + COLS.map(function (c) {
    return '<th class="' + (c[2] ? "num" : "") + '" data-k="' + (c[1] || "") + '">' + c[0] + (SORT && SORT.k === c[1] ? (SORT.d > 0 ? " &#9650;" : " &#9660;") : "") + "</th>";
  }).join("") + "</tr></thead><tbody>";
  idx.forEach(function (i) {
    const r = S.rows[i];
    h += '<tr data-i="' + i + '"' + (i === SEL ? ' class="sel"' : "") + '><td class="num">' + (i + 1) + '</td><td class="tk">' + r.t +
      (r.win < DATA.short ? '<span class="flag">short</span>' : "") + '</td><td class="nm" title="' + esc(r.n) + '">' + esc(r.n) +
      '</td><td class="nm" title="' + esc(r.sub) + '">' + esc(r.sub) + '</td><td class="num"><b style="color:' + scol(r.s) + '">' + r.s.toFixed(0) +
      '%</b><span class="bar"><i style="width:' + r.s + "%;background:" + scol(r.s) + '"></i></span></td><td class="num">' + r.med + '%</td><td class="num">' +
      r.dom + '</td><td class="num">' + r.by + '</td><td class="num">' + pct(r.mrel) + '</td><td class="num">' + pct(r.wrel) +
      '</td><td class="num">' + pct(r.cagr, 1) + '</td><td class="num">' + pct(r.r12) + "</td></tr>";
  });
  document.getElementById("tbl").innerHTML = h + "</tbody></table>";
  document.querySelectorAll("#tbl th").forEach(function (th) {
    th.addEventListener("click", function () {
      const k = th.dataset.k;
      if (!k) { SORT = null; } else { SORT = SORT && SORT.k === k ? { k: k, d: -SORT.d } : { k: k, d: (k === "t" || k === "n" || k === "sub" || k === "by") ? 1 : -1 }; }
      table();
    });
  });
  document.querySelectorAll("#tbl tbody tr").forEach(function (tr) {
    tr.addEventListener("click", function () { choose(+tr.dataset.i); document.getElementById("heat").scrollIntoView({ block: "start" }); });
  });
}

document.querySelectorAll("#mt button").forEach(function (b) {
  b.classList.toggle("on", b.dataset.m === MKT);
  b.addEventListener("click", function () {
    document.querySelectorAll("#mt button").forEach(function (x) { x.classList.toggle("on", x === b); });
    MKT = b.dataset.m; SEC = 0; SEL = 0; SORT = null; save(); pills(); render();
  });
});
pills();
render();
</script>
%%FULLSCREEN%%
</body>
</html>
"""


if __name__ == "__main__":
    main()
