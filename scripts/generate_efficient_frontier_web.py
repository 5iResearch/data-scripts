"""
Website-ready Efficient Frontier page: the risk/return + efficient frontier charts from
generate_stock_screener.py for the TSX, S&P 500 and NASDAQ-100 (not its combined chart), today only, styled to
match the other website pages (white background, 5i colours).

Left out of this page on purpose: the "view as of" snapshots, the Sharpe leaderboard / sector / screen
tables and the movers chart. Kept: the client-side "highlight a ticker" search, and the custom watchlist
(WATCHLIST / EXTRA_TICKERS, see generate_stock_screener.py), which is forced into every frontier.

Signals, the frontier optimisation and the universes all come from generate_stock_screener.py, so the
charts match that report's "Today" view.
Output: outputs/efficient-frontier/Efficient_Frontier.html
"""

import json
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from generate_index_rsi_web import BLUE, GREEN, GRID, INK, LOGO_B64, MUTED, ORANGE, RED
from generate_stock_screener import (
    DATA_YEARS, MIN_FRONTIER_HISTORY, RISK_FREE, TOP_N, compute_signals, download_prices, efficient_frontier,
    get_extra_tickers, load_universes,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "efficient-frontier")

GOLD = "#E8B84B"
# Top-25 / frontier eligibility: the full DATA_YEARS window (95% of it, for holidays / halts). Two reasons:
#  - the screener's own 1-year minimum lets recent listings with a short hot run (e.g. SNDK: ~300%/yr over
#    1.6 years) top the Sharpe ranking;
#  - efficient_frontier() only uses dates where every basket member traded, so one newer member (e.g. PLTR,
#    listed 2020) shrinks the frontier's window to its history while the dots are measured over 10 years,
#    and the frontier floats far above every stock.
# Newer stocks still show as grey dots and can be found with the search.
FULL_HISTORY_SHARE = 0.95
AXIS_QUANTILES = (0.01, 0.99)   # axes fit this share of the dots (plus top 25 and frontier); double-click shows all
SHARPE_SCALE = [[0, RED], [0.5, "#E8C46A"], [1, GREEN]]


def make_frontier_chart(signals, returns, universe, extra_tickers, title) -> go.Figure:
    """Same selection logic as generate_stock_screener.make_scatter_frontier(), light theme."""
    members = [t for t in universe if t in signals.index]
    d = signals.loc[members].dropna(subset=["Vol%", "AnnRet%", "Sharpe"])
    eligible = d[d["NObs"] >= FULL_HISTORY_SHARE * signals["NObs"].max()]
    top = eligible.nlargest(TOP_N, "Sharpe").index.tolist()
    custom = [t for t in extra_tickers if t in signals.index]
    custom_frontier = [t for t in custom if signals.loc[t, "NObs"] >= MIN_FRONTIER_HISTORY]
    ef_vol = ef_ret = None

    d_all, d_top = d.reset_index(), d.loc[[t for t in top if t not in custom]].reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d_all["Vol%"].tolist(), y=d_all["AnnRet%"].tolist(), mode="markers",
        marker=dict(color="#B5B5B5", size=6, opacity=0.6),
        customdata=d_all[["Ticker", "Sharpe"]].values.tolist(), name="All stocks in the index",
        hovertemplate="<b>%{customdata[0]}</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%/yr"
                      "<br>Sharpe: %{customdata[1]:.2f}<extra></extra>",
    ))
    if not d_top.empty:
        fig.add_trace(go.Scatter(
            x=d_top["Vol%"].tolist(), y=d_top["AnnRet%"].tolist(), mode="markers+text",
            marker=dict(color=d_top["Sharpe"].tolist(), colorscale=SHARPE_SCALE,
                        # Scale across the top 25 themselves; on the whole index they'd all be the same green
                        cmin=float(d_top["Sharpe"].min()), cmax=float(d_top["Sharpe"].max()),
                        size=11, line=dict(color="#FFFFFF", width=1),
                        colorbar=dict(title=dict(text="Sharpe", font=dict(size=12, color=INK)), thickness=12,
                                      tickfont=dict(size=11, color=MUTED), len=0.6, y=0.45)),
            text=d_top["Ticker"].tolist(), textposition="top center", textfont=dict(size=10, color=INK),
            name=f"Top {TOP_N} by Sharpe",
            hovertemplate="<b>%{text}</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%/yr"
                          "<br>Sharpe: %{marker.color:.2f}<extra></extra>",
        ))
    if custom:
        d_c = signals.loc[custom].reset_index()
        fig.add_trace(go.Scatter(
            x=d_c["Vol%"].tolist(), y=d_c["AnnRet%"].tolist(), mode="markers+text",
            marker=dict(color=GOLD, size=15, symbol="diamond", line=dict(color=INK, width=1)),
            text=d_c["Ticker"].tolist(), textposition="bottom center", textfont=dict(size=11, color=INK),
            name="Watchlist", customdata=d_c["Sharpe"].tolist(),
            hovertemplate="<b>%{text}</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%/yr"
                          "<br>Sharpe: %{customdata:.2f}<extra></extra>",
        ))

    basket = [t for t in dict.fromkeys(top + custom_frontier) if t in returns.columns]
    if len(basket) >= 2:
        ef_vol, ef_ret = efficient_frontier(returns, basket)
        if ef_vol:
            # Keep only the efficient (upper) half: from the minimum-volatility point up
            i_min = ef_vol.index(min(ef_vol))
            ef_vol, ef_ret = ef_vol[i_min:], ef_ret[i_min:]
            fig.add_trace(go.Scatter(
                x=ef_vol, y=ef_ret, mode="lines", line=dict(color=ORANGE, width=3),
                name="Efficient frontier",
                hovertemplate="Efficient frontier<br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%/yr<extra></extra>",
            ))
        else:
            print(f"  [{title}] frontier optimisation returned no points")

    fig.add_hline(y=0, line_color="#9A9A9A", line_width=1)
    fig.add_hline(y=RISK_FREE * 100, line_dash="dot", line_color=BLUE, line_width=1.4,
                  annotation_text=f"Risk-free rate ({RISK_FREE * 100:.1f}%)", annotation_position="bottom right",
                  annotation_font=dict(color=BLUE, size=12))

    fig.update_layout(
        height=660, autosize=True, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color=INK, size=14),
        title=dict(text=f"<b>{title}</b>  ·  risk vs. return", font=dict(size=20, color=INK), x=0.02, y=0.97),
        margin=dict(t=80, b=60, l=72, r=30),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)", bordercolor=GRID, borderwidth=1,
                    font=dict(size=12)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=GRID, font=dict(color=INK, size=13)),
    )
    axis = dict(gridcolor=GRID, gridwidth=0.6, zeroline=False, linecolor="#CFCFCF",
                tickfont=dict(size=13, color=MUTED), title_font=dict(size=14, color=INK), ticksuffix="%")
    # Axis ranges fit the bulk of the dots plus everything highlighted, so a single broken or extreme
    # ticker (e.g. ELE.TO's split-distorted +2,000%/yr) can't squash the chart into a corner
    lo_q, hi_q = AXIS_QUANTILES
    xs = [d["Vol%"].quantile(hi_q)] + d_top["Vol%"].tolist() + list(ef_vol or [])
    ys_hi = [d["AnnRet%"].quantile(hi_q)] + d_top["AnnRet%"].tolist() + list(ef_ret or [])
    y_lo = min(d["AnnRet%"].quantile(lo_q), 0)
    if custom:
        xs += signals.loc[custom, "Vol%"].tolist()
        ys_hi += signals.loc[custom, "AnnRet%"].tolist()
        y_lo = min(y_lo, signals.loc[custom, "AnnRet%"].min())
    x_hi, y_hi = max(xs) * 1.08, max(ys_hi)
    y_pad = (y_hi - y_lo) * 0.06
    fig.update_xaxes(title_text="Volatility (annualized)", range=[0, x_hi], **axis)
    fig.update_yaxes(title_text=f"Return (annualized, last {DATA_YEARS} years)", range=[y_lo - y_pad, y_hi + y_pad],
                     **axis)
    fig.add_layout_image(dict(source=LOGO_B64, xref="paper", yref="paper", x=1.0, y=1.02, sizex=0.16, sizey=0.07,
                              xanchor="right", yanchor="bottom", layer="above"))
    return fig


def search_html(signals: pd.DataFrame, div_ids: list) -> str:
    """Client-side highlighter: every screened ticker's volatility / return / Sharpe is embedded at build time,
    so typing one adds a marker to every chart without a rerun."""
    payload = {t: [r["Vol%"], r["AnnRet%"], round(r["Sharpe"], 2)]
               for t, r in signals[["Vol%", "AnnRet%", "Sharpe"]].dropna().iterrows()}
    return f"""
<div class="search">
  <label for="tickerSearch">Find a stock on the charts:</label>
  <input type="text" id="tickerSearch" placeholder="e.g. AAPL, RY.TO, SHOP.TO" autocomplete="off">
  <button onclick="addTickers()">Show</button>
  <button onclick="clearTickers()" class="secondary">Clear</button>
  <span id="searchStatus"></span>
</div>
<script>
const SIGNALS = {json.dumps(payload, separators=(",", ":"))};
const CHARTS = {json.dumps(div_ids)};
const added = {{}};
function addTickers() {{
  const tickers = document.getElementById('tickerSearch').value.split(/[,\\s]+/).map(function (t) {{ return t.trim().toUpperCase(); }}).filter(Boolean);
  const found = [], missing = [];
  tickers.forEach(function (t) {{
    if (!(t in SIGNALS)) {{ missing.push(t); return; }}
    found.push(t);
    const v = SIGNALS[t];
    const trace = {{ x: [v[0]], y: [v[1]], mode: 'markers+text', type: 'scatter', name: t, showlegend: true,
      marker: {{ color: '{GOLD}', size: 15, symbol: 'diamond', line: {{ color: '{INK}', width: 1 }} }},
      text: [t], textposition: 'bottom center', textfont: {{ size: 12, color: '{INK}' }},
      hovertemplate: '<b>' + t + '</b><br>Volatility: ' + v[0].toFixed(1) + '%<br>Return: ' + v[1].toFixed(1) + '%/yr<br>Sharpe: ' + v[2].toFixed(2) + '<extra></extra>' }};
    CHARTS.forEach(function (id) {{
      const el = document.getElementById(id);
      if (!el) return;
      if (!(id in added)) added[id] = {{ base: el.data.length, n: 0 }};
      Plotly.addTraces(id, trace); added[id].n++;
    }});
  }});
  const msg = [];
  if (found.length) msg.push('Showing ' + found.join(', '));
  if (missing.length) msg.push('Not in the TSX, S&P 500 or Nasdaq 100: ' + missing.join(', '));
  document.getElementById('searchStatus').textContent = msg.join('. ');
  document.getElementById('tickerSearch').value = '';
}}
function clearTickers() {{
  Object.keys(added).forEach(function (id) {{
    const a = added[id], idx = [];
    for (let i = a.base; i < a.base + a.n; i++) idx.push(i);
    if (idx.length) Plotly.deleteTraces(id, idx);
    a.n = 0;
  }});
  document.getElementById('searchStatus').textContent = '';
}}
document.getElementById('tickerSearch').addEventListener('keydown', function (e) {{ if (e.key === 'Enter') addTickers(); }});
</script>"""


def build_report():
    extra = get_extra_tickers()
    universes = load_universes(extra)
    tickers = list(dict.fromkeys(universes["TSX"] + universes["S&P 500"] + universes["NASDAQ-100"] + extra))
    print(f"Downloading {len(tickers)} tickers, {DATA_YEARS}yr history...")
    prices = download_prices(tickers, DATA_YEARS)
    signals = compute_signals(prices)
    returns = prices.pct_change(fill_method=None)
    print(f"{len(signals)} stocks with usable signals")

    charts = [
        ("TSX", universes["TSX"], "frontier-tsx"),
        ("S&amp;P 500", universes["S&P 500"], "frontier-sp500"),
        ("Nasdaq 100", universes["NASDAQ-100"], "frontier-nasdaq100"),
    ]
    parts, ids = [], []
    for title, universe, div_id in charts:
        if not universe:
            continue
        print(f"  {title}...")
        fig = make_frontier_chart(signals, returns, universe, extra, title.replace("&amp;", "&"))
        parts.append(f'<div class="section"><h2>{title}</h2></div>')
        parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id,
                                 config={"responsive": True, "displaylogo": False}))
        ids.append(div_id)
    return search_html(signals, ids), "\n".join(parts), len(signals), extra


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Efficient Frontier</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #FFFFFF; color: #363636; font-family: Arial, sans-serif; font-size: 16px; line-height: 1.55; margin: 0; padding: 0 0 32px; }}
  header {{ padding: 18px 16px 12px; border-bottom: 1px solid #E6E6E6; }}
  header h1 {{ margin: 0 0 6px; font-size: 28px; color: #363636; }}
  header .meta {{ color: #555555; font-size: 15px; }}
  .intro {{ padding: 12px 16px 0; }}
  .intro p {{ margin: 0 0 8px; }}
  .intro .legend {{ font-size: 15px; color: #555555; margin-top: 16px; }}
  .search {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 14px; font-size: 15px; }}
  .search label {{ color: #555555; }}
  .search input {{ flex: 0 1 260px; border: 1px solid #CFCFCF; border-radius: 4px; padding: 6px 10px; font-size: 15px; color: #363636; }}
  .search button {{ background: #1F79BE; color: #FFFFFF; border: none; border-radius: 4px; padding: 7px 16px; font-size: 15px; cursor: pointer; }}
  .search button.secondary {{ background: #FFFFFF; color: #363636; border: 1px solid #CFCFCF; }}
  #searchStatus {{ color: #555555; font-size: 14px; }}
  .section {{ padding: 16px 16px 0; }}
  .section h2 {{ margin: 0; font-size: 23px; color: #363636; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .source {{ color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }}
</style>
</head>
<body>
<header>
  <h1>Efficient Frontier</h1>
  <div class="meta">Updated {date_str} &middot; TSX, S&amp;P 500 and Nasdaq 100 &middot; last {years} years of daily prices</div>
</header>
<div class="intro">
  <p>Each dot is a stock, plotted by how much it has returned per year over the last {years} years (up) against how
  bumpy the ride has been (right). The best risk/reward sits toward the top-left: high return for the volatility
  taken on. The {top_n} stocks with the highest <b>Sharpe ratio</b>, meaning return above the risk-free rate per unit of
  volatility, are labelled and coloured from red (lowest of the group) to green (highest). To keep the comparison fair, only
  stocks with the full {years} years of history qualify for the top {top_n} and the frontier.</p>
  <p>The orange <b>efficient frontier</b> shows the best trade-off available by combining those {top_n} stocks: for each
  level of volatility, the highest return a long-only mix of them has delivered. Points on the line beat any single
  stock at the same risk, which is the case for diversification. It describes the past, not a recommended portfolio.</p>
  <p class="legend">Grey dots are every other stock in the index (newer listings are measured since they began
  trading). A few extreme outliers sit off the chart; double-click a chart to zoom out and see them. The dotted blue line is the risk-free rate
  ({rf:.1f}%); stocks below it have not paid for their risk.{custom_note}</p>
  {search}
</div>
{charts}
<div class="source">Source: 5i Research, Yahoo Finance. {n_stocks:,} stocks with price history, current index members.</div>
</body>
</html>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    search, charts, n_stocks, extra = build_report()
    custom_note = f" Gold diamonds are our watchlist: {', '.join(extra)}." if extra else ""
    html = PAGE_TEMPLATE.format(date_str=datetime.now().strftime("%B %d, %Y"), years=DATA_YEARS, top_n=TOP_N,
                                rf=RISK_FREE * 100, custom_note=custom_note, search=search, charts=charts,
                                n_stocks=n_stocks)
    out_path = os.path.join(OUTPUT_DIR, "Efficient_Frontier.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
