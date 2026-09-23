"""
Website-ready 200-week moving average report for the four headline indices
(S&P 500, Nasdaq 100, TSX, Russell 2000), styled to match the Index RSI page.

For each index, one chart over the full available history:
  top     weekly close with its 200-week simple moving average (log scale)
  bottom  % distance of the close above/below the 200-week average
Weeks that close below the average are marked with green dots.

Same light theme, timeframe buttons and axis refitting as generate_index_rsi_web.py.
Output: outputs/index-200wma/Index_200WMA_Report.html
"""

import math
import os
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generate_index_rsi_web import (
    BLUE, GREEN, GRID, INDEX_SYMBOLS, INK, LOGO_B64, MUTED, ORANGE,
    fetch_all, fig_to_div, log_ticks, timeframe_ranges,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "index-200wma")

MA_WEEKS = 200


def weekly_vs_ma(daily_df):
    df = daily_df[["Close"]].resample("W").last().dropna()
    df["MA"] = df["Close"].rolling(MA_WEEKS).mean()
    df["Dist"] = (df["Close"] / df["MA"] - 1) * 100
    return df.dropna()


def dist_axis_range(df, pad_frac: float = 0.07) -> list:
    """Fit the % distance panel to the data (always including 0), rounded out to the nearest 5."""
    lo, hi = min(df["Dist"].min(), 0), max(df["Dist"].max(), 0)
    pad = (hi - lo) * pad_frac
    return [math.floor((lo - pad) / 5) * 5, math.ceil((hi + pad) / 5) * 5]


def plot_price_ma(df, label: str, name: str) -> go.Figure:
    below = df["Dist"] < 0
    dot_sizes = [6 if b else 0 for b in below]
    # ISO strings and plain lists: see plot_price_rsi() in generate_index_rsi_web.py
    dates = df.index.strftime("%Y-%m-%d")
    windows = timeframe_ranges(df.index)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=dates, y=df["Close"].tolist(), mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(color=GREEN, size=dot_sizes, line=dict(width=0)),
        name="Level", hovertemplate="Level: %{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=df["MA"].tolist(), mode="lines",
        line=dict(color=ORANGE, width=2),
        name="200-week average", hovertemplate="200-week avg: %{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=df["Dist"].tolist(), mode="lines+markers",
        line=dict(color="#7A7A7A", width=1.6), marker=dict(color=GREEN, size=dot_sizes, line=dict(width=0)),
        name="vs average", hovertemplate="vs 200-week avg: %{y:+.1f}%<extra></extra>",
    ), row=2, col=1)

    fig.add_hrect(y0=-100, y1=0, row=2, col=1, fillcolor=GREEN, opacity=0.07, line_width=0)
    fig.add_hline(y=0, line_dash="dash", line_color=GREEN, line_width=1.4, row=2, col=1)

    fig.update_layout(
        height=700, autosize=True,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color=INK, size=14),
        title=dict(text=f"<b>{name}</b> ({label})  ·  200-Week Moving Average",
                   font=dict(size=20, color=INK), x=0.02, y=0.975),
        showlegend=True,
        legend=dict(orientation="h", x=1.0, xanchor="right", y=1.02, yanchor="bottom",
                    font=dict(size=13, color=INK), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=104, b=44, l=72, r=30),
        meta=dict(full_range=windows[-1][1]),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=GRID, font=dict(color=INK, size=14)),
    )
    # Legend only for the two top-panel lines
    fig.data[2].showlegend = False

    axis_style = dict(zeroline=False, linecolor="#CFCFCF", tickfont=dict(size=13, color=MUTED),
                      title_font=dict(size=14, color=INK), showgrid=True, gridcolor=GRID, gridwidth=0.6)
    lo, hi = min(df["Close"].min(), df["MA"].min()), max(df["Close"].max(), df["MA"].max())
    fig.update_yaxes(title_text="Level", row=1, col=1, type="log", tickformat=",.0f",
                     tickvals=log_ticks(lo, hi), **axis_style)
    fig.update_yaxes(title_text="% vs average", row=2, col=1, range=dist_axis_range(df),
                     ticksuffix="%", **axis_style)
    fig.update_xaxes(range=windows[-1][1])
    fig.update_xaxes(type="date", hoverformat="%b %d %Y", showticklabels=True, **axis_style)

    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="right", showactive=True, active=len(windows) - 1,
        x=0, xanchor="left", y=1.02, yanchor="bottom", pad=dict(l=0, r=0, t=0, b=0),
        bgcolor="#F2F2F2", bordercolor="#CFCFCF", borderwidth=1, font=dict(size=13, color=INK),
        buttons=[dict(label=lbl, method="relayout", args=[{"xaxis.range": rng, "xaxis2.range": rng}])
                 for lbl, rng in windows],
    )])

    fig.add_layout_image(dict(
        source=LOGO_B64, xref="paper", yref="paper", x=1.0, y=1.174,
        sizex=0.16, sizey=0.062, xanchor="right", yanchor="top", layer="above",
    ))
    return fig


def build_report() -> str:
    print("Downloading index data...")
    data = fetch_all(list(INDEX_SYMBOLS.keys()))

    charts = []
    for ticker, cfg in INDEX_SYMBOLS.items():
        try:
            df = weekly_vs_ma(data[ticker])
        except Exception as e:
            print(f"Error {ticker}: {e}")
            continue
        charts.append(f'<div class="section"><h2>{cfg["name"]}</h2></div>')
        charts.append(fig_to_div(plot_price_ma(df, cfg["label"], cfg["name"])))

    return "\n".join(charts)


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Index 200-Week Average</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
// Refit the price (log) and % distance axes to the visible x range after timeframe buttons,
// drag-zoom or double-click. Padding matches dist_axis_range().
function niceTicks(lo, hi) {{
  if (hi / lo > 3) {{
    var t = [];
    for (var e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
      [1, 2, 5].forEach(function (m) {{ var v = m * Math.pow(10, e); if (v >= lo && v <= hi) t.push(v); }});
    return t;
  }}
  var raw = (hi - lo) / 5, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
  var step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag, ticks = [];
  for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  return ticks;
}}
function fitAxes(id) {{
  var gd = document.getElementById(id);
  var xs = gd.data[0].x.map(function (d) {{ return Date.parse(d); }});
  var toMs = function (v) {{ return typeof v === "number" ? v : Date.parse(String(v).replace(" ", "T")); }};
  var span = gd.layout.meta.full_range;
  function fit() {{
    var r = gd.layout.xaxis.range, a = toMs(r[0]), b = toMs(r[1]);
    var pLo = Infinity, pHi = -Infinity, dLo = 0, dHi = 0, any = false;
    for (var i = 0; i < xs.length; i++) {{
      if (xs[i] < a || xs[i] > b) continue;
      any = true;
      var p = Math.min(gd.data[0].y[i], gd.data[1].y[i]), P = Math.max(gd.data[0].y[i], gd.data[1].y[i]);
      var d = gd.data[2].y[i];
      if (p < pLo) pLo = p; if (P > pHi) pHi = P;
      if (d < dLo) dLo = d; if (d > dHi) dHi = d;
    }}
    if (!any) return;
    var pad = (dHi - dLo) * 0.07;
    var lpad = (Math.log10(pHi) - Math.log10(pLo)) * 0.05 || 0.01;
    // Only y keys here, so the handler below ignores the event this fires
    Plotly.relayout(gd, {{
      "yaxis.range": [Math.log10(pLo) - lpad, Math.log10(pHi) + lpad],
      "yaxis.tickvals": niceTicks(pLo, pHi),
      "yaxis2.range": [Math.floor((dLo - pad) / 5) * 5, Math.ceil((dHi + pad) / 5) * 5]
    }});
  }}
  gd.on("plotly_relayout", function (ev) {{
    if (ev["xaxis.autorange"] || ev["xaxis2.autorange"]) {{
      // Deferred: relayouting inside this event gets overwritten when Plotly finishes the autorange.
      setTimeout(function () {{ Plotly.relayout(gd, {{"xaxis.autorange": false, "xaxis.range": span.slice(),
                                                "xaxis2.autorange": false, "xaxis2.range": span.slice()}}); }}, 0);
      return;
    }}
    if (Object.keys(ev).some(function (k) {{ return k.indexOf("xaxis") === 0; }})) fit();
  }});
  fit();
}}
</script>
<style>
  body {{ background: #FFFFFF; color: #363636; font-family: Arial, sans-serif; font-size: 16px; line-height: 1.55; margin: 0; padding: 0 0 32px; }}
  header {{ padding: 18px 16px 12px; border-bottom: 1px solid #E6E6E6; }}
  header h1 {{ margin: 0 0 6px; font-size: 28px; color: #363636; }}
  header .meta {{ color: #555555; font-size: 15px; }}
  .intro {{ padding: 12px 16px 0; }}
  .intro p {{ margin: 0 0 8px; }}
  .intro .legend {{ font-size: 15px; color: #555555; margin-top: 16px; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 4px 0 2px; vertical-align: middle; }}
  .section {{ padding: 16px 16px 0; }}
  .section h2 {{ margin: 0; font-size: 23px; color: #363636; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .source {{ color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }}
</style>
</head>
<body>
<header>
  <h1>Index 200-Week Moving Average</h1>
  <div class="meta">Updated {date_str} &middot; S&amp;P 500, Nasdaq 100, S&amp;P/TSX Composite, Russell 2000</div>
</header>
<div class="intro">
  <p>These charts compare each major index with its 200-week moving average: the simple average of its last
  200 weekly closes, or roughly four years. It smooths out short-term swings and shows the long-term trend.
  In a healthy bull market the index stays well above this line. Weekly closes below it are uncommon, about
  15&ndash;20% of weeks in each index's history, and have mostly come during bear markets such as 2002, 2009 and 2022.</p>
  <p>The top panel shows the index and its 200-week average. The bottom panel shows how far the index sits
  above or below that average, in percent. The further above zero, the more stretched the index is relative
  to its long-term trend.</p>
  <p class="legend"><span class="dot" style="background:#44A660"></span>Week closed below its 200-week average
  &nbsp;&nbsp;Charts show each index's full available history. The current week uses the latest close until the week ends.</p>
</div>
{charts}
<div class="source">Source: 5i Research, Yahoo Finance.</div>
</body>
</html>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    charts = build_report()
    html = PAGE_TEMPLATE.format(date_str=datetime.now().strftime("%B %d, %Y"), charts=charts)

    out_path = os.path.join(OUTPUT_DIR, "Index_200WMA_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
