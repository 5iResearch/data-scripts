"""
Website-ready Revenue Revision Screener: the same screen and ranking as generate_rev_revision_screener.py,
restyled to match the other website pages (white background, 5i colours) with a slimmer table.

Sections, Canada first: Canada (vs XIC), US S&P 500 (vs SPY), All-US (~2,000 names, vs QQQ).
Each shows its top 40 as a compact table (Rank, Ticker, Name, Cap, 10Y vs benchmark) split into three
side-by-side columns that read top-to-bottom, left-to-right, followed by a spotlight chart per name
(3-year price and the revenue revision bars by fiscal year).

Scoring, gates and data (the manually refreshed revision CSVs in data/) all come from
generate_rev_revision_screener.py, so both pages always rank the same names. Sector / industry labels come from
the revision CSVs' own Sector and Industry columns (not data/koyfin_*.csv).
Output: outputs/rev-revision-web/Rev_Revision_Screener.html
"""

import math
import os
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from common_screening import CAP_FILTER_CONTROL_HTML, CAP_FILTER_JS, load_sp500_symbols
from generate_index_rsi_web import BLUE, GREEN, GRID, INK, MUTED, ORANGE, RED
from generate_rev_revision_screener import (
    ALL_WIN_KEYS, ALL_WIN_LABELS, CDN_CSV_PATH, CHART_CUTOFF, PRICE_YEARS, SPOTLIGHT_TOP_N, US_CSV_PATH,
    compute_vs_bench_10y, download_bench_series, download_spotlight_prices, load_rev_csv, rank_by_revisions,
    score_row,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "rev-revision-web")

TABLE_COLUMNS = 3          # a full top 40 is split into up to this many side-by-side tables...
ROWS_PER_TABLE = 14        # ...of about this many rows, so a short list (e.g. 8 Canadian names) stays one table
FY_COLORS = [BLUE, "#4B8EA9", ORANGE]


def load_sector_map(path: str) -> dict:
    """Ticker -> (Sector, Industry) from the revision CSV itself (Sector / Industry columns added Sep 2026), so this
    page no longer needs data/koyfin_*.csv. Every name on the page comes from these same files, so coverage is total.
    The Ticker converter keeps National Bank's "NA" from being read as missing."""
    df = pd.read_csv(path, converters={"Ticker": lambda v: str(v).strip()})
    if not {"Sector", "Industry"} <= set(df.columns):
        print(f"  {os.path.basename(path)} has no Sector/Industry columns; labels will be blank")
        return {}
    df = df.fillna({"Sector": "", "Industry": ""})
    return {t.upper(): (sec, ind) for t, sec, ind in zip(df["Ticker"], df["Sector"], df["Industry"])}


def with_revisions(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, df.apply(score_row, axis=1)], axis=1)


def vs_bench_cell(val) -> str:
    if val is None or pd.isna(val):
        return '<td class="na">&mdash;</td>'
    cls, mark = ("up", "&#10003;") if val >= 0 else ("down", "&#10007;")
    return f'<td class="{cls}">{mark} {val:+.0f}%</td>'


def build_tables(spotlight: pd.DataFrame, bench_label: str) -> str:
    """Top-N table split into TABLE_COLUMNS tables; ranks continue at the top of the next one."""
    rows = [
        f'<tr><td class="rank">{int(r["rank"])}</td><td class="tkr">{escape(str(r["ticker"]))}</td>'
        f'<td class="name" title="{escape(str(r["name"]))}">{escape(str(r["name"]))}</td>'
        f'<td>{escape(str(r["cap_bucket"]))}</td>{vs_bench_cell(r["vs_bench_10y"])}</tr>'
        for _, r in spotlight.iterrows()
    ]
    head = ('<colgroup><col class="c-rank"><col class="c-tkr"><col><col class="c-cap"><col class="c-vs"></colgroup>'
            f"<thead><tr><th>#</th><th>Ticker</th><th>Name</th><th>Cap</th>"
            f"<th>10Y vs {escape(bench_label)}</th></tr></thead>")
    per = math.ceil(len(rows) / min(TABLE_COLUMNS, math.ceil(len(rows) / ROWS_PER_TABLE)))
    tables = [f"<table>{head}<tbody>{''.join(rows[i:i + per])}</tbody></table>" for i in range(0, len(rows), per)]
    return f'<div class="tables">{"".join(tables)}</div>'


def make_spotlight(row, price_series, bench_label: str, sector_map: dict) -> go.Figure:
    ticker, name, rank = row["ticker"], row["name"], int(row["rank"])
    sector, industry = sector_map.get(str(ticker).upper(), ("", ""))
    chart = price_series[price_series.index >= CHART_CUTOFF] if price_series is not None else None

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.10,
                        subplot_titles=[f"{PRICE_YEARS}-year price", "Revenue estimate revisions"])
    if chart is not None and len(chart) > 5:
        # ISO strings and plain lists: see plot_price_rsi() in generate_index_rsi_web.py
        dates, pv = chart.index.strftime("%Y-%m-%d"), chart.astype(float).tolist()
        ret = (pv[-1] / pv[0] - 1) * 100
        color = GREEN if ret >= 0 else RED
        fill = "rgba(68,166,96,0.10)" if ret >= 0 else "rgba(162,42,42,0.10)"
        fig.add_trace(go.Scatter(x=dates, y=[pv[0]] * len(pv), mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=pv, mode="lines", line=dict(color=color, width=1.8), fill="tonexty",
                                 fillcolor=fill, showlegend=False,
                                 hovertemplate="%{x|%b %d %Y}<br>$%{y:,.2f}<extra></extra>"), row=1, col=1)
        fig.add_annotation(x=0.02, y=0.97, xref="x domain", yref="y domain", text=f"{PRICE_YEARS}Y: <b>{ret:+.1f}%</b>",
                           showarrow=False, font=dict(size=13, color=color), bgcolor="#FFFFFF",
                           bordercolor=color, borderwidth=1, row=1, col=1)
    else:
        fig.add_annotation(x=0.27, y=0.5, xref="paper", yref="paper", text="No price data", showarrow=False,
                           font=dict(size=13, color=MUTED))

    for i, fy in enumerate(["fy1", "fy2", "fy3"]):
        vals = [row.get(f"{fy}_{w}", np.nan) * 100 for w in ALL_WIN_KEYS]
        fig.add_trace(go.Bar(x=ALL_WIN_LABELS, y=[None if pd.isna(v) else v for v in vals], name=f"FY{i + 1}E",
                             marker_color=FY_COLORS[i],
                             hovertemplate="%{x}: %{y:+.2f}%<extra>FY" + str(i + 1) + "E</extra>"), row=1, col=2)

    vs = row["vs_bench_10y"]
    if vs is None or pd.isna(vs):
        vs_html = f'<span style="color:{MUTED}">10Y vs {bench_label}: —</span>'
    else:
        vs_html = (f'<span style="color:{GREEN if vs >= 0 else RED}">10Y vs {bench_label}: '
                   f'{"✓" if vs >= 0 else "✗"} {vs:+.0f}%</span>')
    # Plain characters, not HTML entities: Plotly titles don't decode named entities like &middot;
    info = " · ".join(s for s in (sector, industry) if s)
    sub = f"Rank #{rank}" + (f"  ·  {info}" if info else "") + f"  ·  {vs_html}"

    fig.update_layout(
        title=dict(text=(f'<b><span style="font-size:22px;color:{BLUE}">{escape(str(ticker))}</span></b>'
                         f'<span style="font-size:17px;color:{INK}">  {escape(str(name))}</span><br>'
                         f'<span style="font-size:13px;color:{MUTED}">{sub}</span>'),
                   x=0.01, xanchor="left", y=0.96, yanchor="top"),
        height=400, autosize=True, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color=INK, size=13),
        margin=dict(l=64, r=30, t=96, b=70), barmode="group",
        # Legend under the bar chart; above it, it collided with the subplot title
        legend=dict(orientation="h", x=1.0, xanchor="right", y=-0.1, yanchor="top", font=dict(size=12),
                    traceorder="normal"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=GRID, font=dict(color=INK)),
    )
    axis = dict(gridcolor=GRID, linecolor="#CFCFCF", tickfont=dict(size=12, color=MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(zeroline=False, **axis)
    fig.update_yaxes(title_text="Price ($)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Revision", ticksuffix="%", zeroline=True, zerolinecolor="#9A9A9A", row=1, col=2)
    for ann in fig.layout.annotations:
        if ann.text in (f"{PRICE_YEARS}-year price", "Revenue estimate revisions"):
            ann.font = dict(size=13, color=MUTED)
    return fig


def chart_div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True, "displaylogo": False})


def cdn_symbol_candidates(ticker: str) -> list:
    """Yahoo symbols to try for a Canadian ticker from the revision CSV, which drops both the exchange and any
    punctuation: TSX first, then TSX Venture, CSE and Cboe Canada, then the dotted forms Yahoo writes with a
    dash (trust units DUN -> D-UN.TO, share classes CCLB -> CCL-B.TO)."""
    t = ticker.upper().replace(".TO", "")
    cands = [f"{t}{sfx}" for sfx in (".TO", ".V", ".CN", ".NE")]
    for unit in ("UN", "U", "A", "B"):
        if len(t) > len(unit) and t.endswith(unit):
            cands += [f"{t[:-len(unit)]}-{unit}{sfx}" for sfx in (".TO", ".V")]
    return cands


def fetch_cdn_prices(tickers: list) -> dict:
    """Ticker -> price series. One batch download on .TO, then the other candidates one at a time for the few
    names that come back empty (CSE / TSX Venture listings, trust units, share classes)."""
    found = download_spotlight_prices([cdn_symbol_candidates(t)[0] for t in tickers], label="Canada")
    prices = {t: found[cdn_symbol_candidates(t)[0]] for t in tickers if cdn_symbol_candidates(t)[0] in found}
    for t in (t for t in tickers if t not in prices):
        for sym in cdn_symbol_candidates(t)[1:]:
            hit = download_spotlight_prices([sym], label="Canada")
            if sym in hit:
                print(f"  {t}: found on Yahoo as {sym}")
                prices[t] = hit[sym]
                break
        else:
            print(f"  {t}: no Yahoo price under any of {cdn_symbol_candidates(t)}")
    return prices


def fetch_us_prices(tickers: list) -> dict:
    return download_spotlight_prices(list(tickers), label="US")


def build_section(ranked, title, bench_series, bench_label, sector_map, fetch_prices=fetch_us_prices) -> str:
    parts = [f'<div class="section"><h2>{title}</h2></div>']
    if ranked.empty:
        parts.append('<p class="empty">No names passed the screen today.</p>')
        return "\n".join(parts)

    spotlight = ranked.head(SPOTLIGHT_TOP_N).copy()
    print(f"  downloading prices for {len(spotlight)} names...")
    prices = fetch_prices(list(spotlight["ticker"]))
    spotlight["vs_bench_10y"] = [compute_vs_bench_10y(prices.get(t), bench_series) for t in spotlight["ticker"]]

    parts.append(build_tables(spotlight, bench_label))
    for _, row in spotlight.iterrows():
        fig = make_spotlight(row, prices.get(row["ticker"]), bench_label, sector_map)
        parts.append(f'<div class="spot" data-cap="{row["cap_bucket"]}">{chart_div(fig)}</div>')
    return "\n".join(parts)


def build_report() -> str:
    print("=== Benchmarks ===")
    qqq, spy, xic = (download_bench_series(t) for t in ("QQQ", "SPY", "XIC.TO"))
    us_sectors, cdn_sectors = load_sector_map(US_CSV_PATH), load_sector_map(CDN_CSV_PATH)

    # Same universes and gates as generate_rev_revision_screener.main()
    print("=== Canada ===")
    cdn = with_revisions(load_rev_csv(CDN_CSV_PATH))
    cdn_ranked = rank_by_revisions(cdn, extra_gate=lambda d: (d["fy1_1m"] > 0) & (d["fy2_1m"] > 0))

    print("=== US ===")
    us = with_revisions(load_rev_csv(US_CSV_PATH))
    sp500 = {s.replace(".", "-") for s in load_sp500_symbols()}
    sp500_ranked = rank_by_revisions(us[us["ticker"].isin(sp500)].reset_index(drop=True))
    all_us_ranked = rank_by_revisions(us)

    return "\n".join([
        build_section(cdn_ranked, "Canada", xic, "XIC", cdn_sectors, fetch_cdn_prices),
        build_section(sp500_ranked, "US: S&amp;P 500", spy, "SPY", us_sectors),
        build_section(all_us_ranked, "US: All (~2,000 names)", qqq, "QQQ", us_sectors),
    ])


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revenue Revision Screener</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
{cap_js}
<style>
  body {{ background: #FFFFFF; color: #363636; font-family: Arial, sans-serif; font-size: 16px; line-height: 1.55; margin: 0; padding: 0 0 32px; }}
  header {{ padding: 18px 16px 12px; border-bottom: 1px solid #E6E6E6; }}
  header h1 {{ margin: 0 0 6px; font-size: 28px; color: #363636; }}
  header .meta {{ color: #555555; font-size: 15px; }}
  .intro {{ padding: 12px 16px 0; }}
  .intro p {{ margin: 0 0 8px; }}
  .intro .legend {{ font-size: 15px; color: #555555; margin-top: 16px; }}
  .cap-filter {{ margin-top: 12px; font-size: 15px; }}
  .cap-filter label {{ color: #555555; margin-right: 6px; }}
  .cap-filter select {{ background: #FFFFFF; color: #363636; border: 1px solid #CFCFCF; border-radius: 4px; padding: 4px 10px; font-size: 15px; }}
  .section {{ padding: 16px 16px 0; }}
  .section h2 {{ margin: 0; font-size: 23px; color: #363636; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .tables {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px 24px; padding: 14px 16px 8px; align-items: start; }}
  @media (max-width: 1000px) {{ .tables {{ grid-template-columns: minmax(0, 1fr); }} }}
  @media (max-width: 480px) {{ .tables table {{ font-size: 13px; }} .tables td, .tables th {{ padding: 5px 4px; }} }}
  /* Fixed layout: every column keeps its width and only Name shrinks (with an ellipsis), so three tables fit
     side by side on a laptop and one fits a phone without cutting off the 10Y column */
  .tables table {{ border-collapse: collapse; font-size: 14px; width: 100%; table-layout: fixed; }}
  .tables col.c-rank {{ width: 2.6em; }} .tables col.c-tkr {{ width: 4.6em; }} .tables col.c-cap {{ width: 3.8em; }} .tables col.c-vs {{ width: 6.6em; }}
  .tables th {{ text-align: left; font-size: 12px; font-weight: bold; color: #555555; text-transform: uppercase; letter-spacing: .03em;
               padding: 6px 6px; border-bottom: 2px solid #C67A29; white-space: nowrap; overflow: hidden; }}
  .tables td {{ padding: 5px 6px; border-bottom: 1px solid #E6E6E6; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .tables tbody tr:nth-child(even) {{ background: #F7F7F7; }}
  .tables td.rank {{ color: #555555; text-align: right; }}
  .tables td.tkr {{ font-weight: bold; color: #1F79BE; }}
  .tables td.up {{ color: #44A660; font-weight: bold; }}
  .tables td.down {{ color: #A22A2A; font-weight: bold; }}
  .tables td.na {{ color: #9A9A9A; }}
  .spot {{ border-top: 1px solid #E6E6E6; margin: 8px 16px 0; }}
  .empty {{ color: #555555; padding: 8px 16px; }}
  .source {{ color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }}
</style>
</head>
<body>
<header>
  <h1>Revenue Revision Screener</h1>
  <div class="meta">Updated {date_str} &middot; Canada, S&amp;P 500 and all-US top 40</div>
</header>
<div class="intro">
  <p>These are the stocks where analysts have been raising their revenue forecasts the most, and most consistently.
  Each company is scored on how much its revenue estimates for the next three fiscal years have risen over the past
  week, and whether those increases have been building over the past month, quarter, six months and year. Only
  companies whose estimates for the next two years both rose in the past week make the list (for Canada, over the
  past month as well). This is a screen for business momentum, not price: it does not look at how the stock has
  traded.</p>
  <p class="legend">10Y vs benchmark: the stock's total return over the past 10 years (or since listing, if at least
  two years) minus the benchmark's: XIC for Canada, SPY for the S&amp;P 500 and QQQ for all-US. <span style="color:#44A660;font-weight:bold">&#10003;</span>
  beat the benchmark, <span style="color:#A22A2A;font-weight:bold">&#10007;</span> lagged it. Charts below each table show
  the 3-year price and the revenue estimate revisions for each of the next three fiscal years.</p>
  {cap_control}
</div>
{body}
<div class="source">Source: 5i Research, analyst revenue estimates, Yahoo Finance.</div>
</body>
</html>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    html = PAGE_TEMPLATE.format(date_str=datetime.now().strftime("%B %d, %Y"), body=build_report(),
                                cap_js=CAP_FILTER_JS, cap_control=CAP_FILTER_CONTROL_HTML)
    out_path = os.path.join(OUTPUT_DIR, "Rev_Revision_Screener.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
