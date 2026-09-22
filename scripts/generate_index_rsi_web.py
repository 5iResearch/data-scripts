"""
Website-ready Index RSI report: a light-theme version of the Industry RSI report,
limited to the four headline indices (S&P 500, Nasdaq 100, TSX, Russell 2000).

For each index it renders two charts over the full available history, price on top
and RSI below:
  Weekly RSI-14
  Weekly RSI-52 (1-Year)
Oversold/overbought levels are set per index (see INDEX_SYMBOLS).

The page has a white background and uses the 5i corporate colours so it can be
embedded directly on the 5i website. Charts are responsive (no fixed width) so
they fit the site's content column. Output: outputs/index-rsi/Index_RSI_Report.html
"""

import math
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from generate_industry_rsi_report import LOGO_B64, fetch_all, get_weekly_rsi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "index-rsi")

# ── Corporate colours, light theme ───────────────────────────────────────────
ORANGE = "#C67A29"
BLUE = "#1F79BE"
INK = "#363636"
MUTED = "#555555"
GRID = "#E6E6E6"
GREEN = "#44A660"
RED = "#A22A2A"

SIGNAL_COLORS = {"buy": GREEN, "hold": "#9A9A9A", "sell": RED}

# Index levels rather than ETFs, so the charts show the benchmarks by name.
# (oversold, overbought) per index for the weekly RSI-14 and the 1-year RSI-52. Defaults are
# 40/75 and 45/65; S&P 500 1-year and both Russell 2000 views are tightened because they
# rarely/never reached the defaults after 2006. The tuned levels each flag roughly 4-7% of weeks
# since 2006, in line with the defaults on the other indices.
INDEX_SYMBOLS = {
    "^GSPC": dict(name="S&P 500", label="SPX", weekly=(40, 75), yearly=(47, 63)),
    "^NDX": dict(name="Nasdaq 100", label="NDX", weekly=(40, 75), yearly=(45, 65)),
    "^GSPTSE": dict(name="S&P/TSX Composite", label="TSX", weekly=(40, 75), yearly=(45, 65)),
    "^RUT": dict(name="Russell 2000", label="RUT", weekly=(40, 70), yearly=(46, 61)),
}


# Gap after the last data point, as a fraction of the visible date range
RIGHT_PAD = 0.03


def timeframe_ranges(index) -> list:
    """(label, [start, end]) for each timeframe button, ending RIGHT_PAD of the window past the last point."""
    first, last = index[0], index[-1]
    starts = [("YTD", pd.Timestamp(year=last.year, month=1, day=1))]
    starts += [(f"{n}Y", last - pd.DateOffset(years=n)) for n in (1, 3, 5, 10)]
    starts = [(lbl, st) for lbl, st in starts if st > first] + [("All", first)]
    return [(lbl, [st.strftime("%Y-%m-%d"),
                   (last + max((last - st) * RIGHT_PAD, pd.Timedelta(days=2))).strftime("%Y-%m-%d")])
            for lbl, st in starts]


def rsi_axis_range(df, thresholds, pad_frac: float = 0.07) -> list:
    """Fit the RSI panel to the data (and both threshold lines), rounded out to the nearest 5.
    Padding scales with the data's spread, so the slower 1-year RSI gets as snug a fit as the weekly one."""
    lo = min(df["RSI"].min(), thresholds[0])
    hi = max(df["RSI"].max(), thresholds[1])
    pad = (hi - lo) * pad_frac
    lo, hi = lo - pad, hi + pad
    return [max(0, math.floor(lo / 5) * 5), min(100, math.ceil(hi / 5) * 5)]


def log_ticks(lo: float, hi: float) -> list:
    """1-2-5 tick values spanning [lo, hi], so the log price axis doesn't crowd its labels."""
    ticks = []
    for exp in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10 ** exp
            if lo * 0.8 <= v <= hi * 1.25:
                ticks.append(v)
    return ticks


def plot_price_rsi(df, label: str, name: str, subtitle: str, thresholds) -> go.Figure:
    # Only oversold/overbought weeks get a dot; neutral weeks are just the line,
    # otherwise grey dots bury the brand-coloured lines on a white background.
    dot_colors = [SIGNAL_COLORS[s] for s in df["Signal"]]
    dot_sizes = [0 if s == "hold" else 6 for s in df["Signal"]]
    # ISO strings, not the DatetimeIndex: this plotly/pandas combo serializes datetime64 as raw ints
    dates = df.index.strftime("%Y-%m-%d")
    windows = timeframe_ranges(df.index)
    # y values go in as plain lists (below) for the same reason: plotly 6 base64-encodes numpy
    # arrays, and fitAxes() in the page script reads gd.data[i].y as numbers.

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.66, 0.34], vertical_spacing=0.06)

    fig.add_trace(
        go.Scatter(
            x=dates, y=df["Close"].tolist(), mode="lines+markers",
            line=dict(color=BLUE, width=2),
            marker=dict(color=dot_colors, size=dot_sizes, line=dict(width=0)),
            name="Price",
            hovertemplate="Level: %{y:,.2f}<extra></extra>",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=df["RSI"].tolist(), mode="lines+markers",
            line=dict(color=ORANGE, width=2),
            marker=dict(color=dot_colors, size=dot_sizes, line=dict(width=0)),
            name="RSI",
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=2, col=1,
    )

    oversold, overbought = thresholds
    fig.add_hrect(y0=overbought, y1=100, row=2, col=1, fillcolor=RED, opacity=0.07, line_width=0)
    fig.add_hrect(y0=0, y1=oversold, row=2, col=1, fillcolor=GREEN, opacity=0.07, line_width=0)
    for y, color in [(oversold, GREEN), (overbought, RED)]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=1.4, row=2, col=1)

    fig.update_layout(
        height=640, autosize=True,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color=INK, size=14),
        title=dict(
            text=f"<b>{name}</b> ({label})  ·  {subtitle}",
            font=dict(size=20, color=INK), x=0.02, y=0.975,
        ),
        showlegend=False,
        margin=dict(t=104, b=44, l=72, r=30),
        # Read by fitAxes() in the page script to refit the RSI axis after a range change
        meta=dict(thresholds=list(thresholds), full_range=windows[-1][1]),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=GRID, font=dict(color=INK, size=14)),
    )

    axis_style = dict(zeroline=False, linecolor="#CFCFCF", tickfont=dict(size=13, color=MUTED),
                      title_font=dict(size=14, color=INK))
    fig.update_yaxes(title_text="Level", row=1, col=1, type="log", tickformat=",.0f",
                     showgrid=True, gridcolor=GRID, gridwidth=0.6,
                     tickvals=log_ticks(df["Close"].min(), df["Close"].max()), **axis_style)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=rsi_axis_range(df, thresholds),
                     showgrid=True, gridcolor=GRID, gridwidth=0.6, **axis_style)
    # Explicit ranges, not autorange (which pads by years because of the markers). Starts on "All".
    fig.update_xaxes(range=windows[-1][1])
    fig.update_xaxes(type="date", hoverformat="%b %d %Y", showticklabels=True,
                     showgrid=True, gridcolor=GRID, gridwidth=0.6, **axis_style)

    # Timeframe buttons with fixed ranges rather than Plotly's rangeselector: the rangeselector counts
    # back from the current axis end, which breaks once the end carries the right-hand gap.
    # fitAxes() in the page script refits the y axes whenever the x range changes.
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="right", showactive=True, active=len(windows) - 1,
        x=0, xanchor="left", y=1.02, yanchor="bottom", pad=dict(l=0, r=0, t=0, b=0),
        bgcolor="#F2F2F2", bordercolor="#CFCFCF", borderwidth=1, font=dict(size=13, color=INK),
        buttons=[dict(label=lbl, method="relayout", args=[{"xaxis.range": rng, "xaxis2.range": rng}])
                 for lbl, rng in windows],
    )])

    fig.add_layout_image(dict(
        source=LOGO_B64, xref="paper", yref="paper", x=1.0, y=1.195,
        sizex=0.16, sizey=0.07, xanchor="right", yanchor="top", layer="above",
    ))
    return fig


def fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"responsive": True, "displaylogo": False},
                       post_script="fitAxes('{plot_id}');")


def build_report() -> str:
    print("Downloading index data...")
    fetch_all(list(INDEX_SYMBOLS.keys()))

    charts = []
    for ticker, cfg in INDEX_SYMBOLS.items():
        name, label, weekly, yearly = cfg["name"], cfg["label"], cfg["weekly"], cfg["yearly"]
        try:
            df_w = get_weekly_rsi(ticker, rsi_length=14, thresholds=weekly)
            df_52 = get_weekly_rsi(ticker, rsi_length=52, thresholds=yearly)
        except Exception as e:
            print(f"Error {ticker}: {e}")
            continue

        charts.append(f'<div class="section"><h2>{name}</h2></div>')
        charts.append(fig_to_div(plot_price_rsi(df_w, label, name, "Weekly RSI (14)", weekly)))
        charts.append(fig_to_div(plot_price_rsi(df_52, label, name, "1-Year RSI (52-week)", yearly)))

    return "\n".join(charts)


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Index RSI Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
// Plotly doesn't rescale y when the x range changes (range buttons, drag-zoom), so refit
// the price (log) and RSI axes to whatever is visible. RSI padding matches rsi_axis_range().
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
    var pLo = Infinity, pHi = -Infinity, rLo = Infinity, rHi = -Infinity;
    for (var i = 0; i < xs.length; i++) {{
      if (xs[i] < a || xs[i] > b) continue;
      var p = gd.data[0].y[i], q = gd.data[1].y[i];
      if (p < pLo) pLo = p; if (p > pHi) pHi = p;
      if (q < rLo) rLo = q; if (q > rHi) rHi = q;
    }}
    if (!isFinite(pLo)) return;
    var th = gd.layout.meta.thresholds;
    var lo = Math.min(rLo, th[0]), hi = Math.max(rHi, th[1]), pad = (hi - lo) * 0.07;
    var lpad = (Math.log10(pHi) - Math.log10(pLo)) * 0.05 || 0.01;
    // Only y keys here, so the handler below ignores the event this fires
    Plotly.relayout(gd, {{
      "yaxis.range": [Math.log10(pLo) - lpad, Math.log10(pHi) + lpad],
      "yaxis.tickvals": niceTicks(pLo, pHi),
      "yaxis2.range": [Math.max(0, Math.floor((lo - pad) / 5) * 5), Math.min(100, Math.ceil((hi + pad) / 5) * 5)]
    }});
  }}
  gd.on("plotly_relayout", function (ev) {{
    if (ev["xaxis.autorange"] || ev["xaxis2.autorange"]) {{
      // Deferred: relayouting inside this event gets overwritten when Plotly finishes the autorange.
      // The x change re-enters this handler, which then fits y.
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
  header {{ padding: 24px 16px 18px; border-bottom: 1px solid #E6E6E6; }}
  header h1 {{ margin: 0 0 6px; font-size: 28px; color: #363636; }}
  header .meta {{ color: #555555; font-size: 15px; }}
  .intro {{ padding: 18px 16px 4px; max-width: 900px; }}
  .intro p {{ margin: 0 0 12px; }}
  .intro .legend {{ font-size: 15px; color: #555555; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 4px 0 2px; vertical-align: middle; }}
  .section {{ padding: 28px 16px 0; }}
  .section h2 {{ margin: 0; font-size: 23px; color: #363636; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .source {{ color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }}
</style>
</head>
<body>
<header>
  <h1>Index RSI Report</h1>
  <div class="meta">Updated {date_str} &middot; S&amp;P 500, Nasdaq 100, S&amp;P/TSX Composite, Russell 2000</div>
</header>
<div class="intro">
  <p>These charts show how stretched each major index is compared with its own recent history, using the
  Relative Strength Index (RSI). RSI runs from 0 to 100 and compares the size of recent gains with recent losses.
  High readings mean the index has run up hard and may be overbought. Low readings mean it has sold off sharply
  and may be oversold.</p>
  <p>Each index has two views. The <b>weekly RSI</b> uses the last 14 weeks and picks up shorter swings.
  The <b>1-year RSI</b> uses the last 52 weeks and tracks the longer trend. It moves more slowly, so its
  oversold and overbought levels sit closer together. Each index has its own levels, marked by the dashed lines,
  because some indices run hotter or calmer than others. For example, the weekly S&amp;P 500 is oversold below 40
  and overbought above 75.</p>
  <p class="legend"><span class="dot" style="background:#44A660"></span>Oversold week
  &nbsp;&nbsp;<span class="dot" style="background:#A22A2A"></span>Overbought week
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

    out_path = os.path.join(OUTPUT_DIR, "Index_RSI_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
