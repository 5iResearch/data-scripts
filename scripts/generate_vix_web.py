"""
Website-ready VIX page, styled to match the Index RSI and 200-week MA pages.

Two charts on daily closes, full available history:
  1. S&P 500 (top) with the VIX (bottom), the VIX line coloured by zone: <= 20 favourable,
     20-30 unfavourable, > 30 highly favourable (contrarian: fear spikes have tended to be good
     times to buy).
  2. S&P 500 (top) with the VIX/VIX3M ratio (bottom). A ratio above 1 (inverted VIX term
     structure: near-term fear above 3-month) is marked on both panels as a potential
     intermediate-term buying opportunity. Starts in 2006, when VIX3M data begins.

Same light theme, timeframe buttons and axis refitting as generate_index_rsi_web.py.
Output: outputs/vix/VIX_Report.html
"""

import math
import os
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generate_index_rsi_web import (
    BLUE, GREEN, GRID, INK, LOGO_B64, MUTED, ORANGE, RED,
    fetch_all, fig_to_div, log_ticks, timeframe_ranges,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "vix")

VIX_ZONES = (20, 30)   # favourable <= 20 < unfavourable <= 30 < highly favourable
ZONE_COLORS = (GREEN, RED, BLUE)
SPX_COLOR = "#4A4A4A"
SPX_MODE = "highlight"   # how the S&P 500 line is coloured on the VIX chart: "highlight" or "mirror"
RATIO_SIGNAL = 1.0

AXIS_STYLE = dict(zeroline=False, linecolor="#CFCFCF", tickfont=dict(size=13, color=MUTED),
                  title_font=dict(size=14, color=INK), showgrid=True, gridcolor=GRID, gridwidth=0.6)


def fit_range(values, include=(), step: float = 5, pad_frac: float = 0.07, floor=None) -> list:
    """Fit a linear axis to the values (plus any levels that must stay visible), rounded out to `step`.
    Mirrors fitRange() in the page script, which redoes this for the visible window."""
    lo, hi = min(min(values), *include), max(max(values), *include)
    pad = (hi - lo) * pad_frac
    lo, hi = math.floor((lo - pad) / step) * step, math.ceil((hi + pad) / step) * step
    return [max(lo, floor) if floor is not None else lo, hi]


def base_layout(fig: go.Figure, title: str, windows, fit, xaxes, height: int, top: int = 104):
    """Shared styling, timeframe buttons and logo. `fit` / `xaxes` are read by fitAxes() in the page."""
    fig.update_layout(
        height=height, autosize=True,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color=INK, size=14),
        title=dict(text=title, font=dict(size=20, color=INK), x=0.02, y=1 - 16 / height),
        showlegend=False,
        margin=dict(t=top, b=44, l=72, r=30),
        meta=dict(full_range=windows[-1][1], fit=fit, xaxes=xaxes),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=GRID, font=dict(color=INK, size=14)),
    )
    fig.update_xaxes(range=windows[-1][1], type="date", hoverformat="%b %d %Y", showticklabels=True, **AXIS_STYLE)
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="right", showactive=True, active=len(windows) - 1,
        x=0, xanchor="left", y=1.02, yanchor="bottom", pad=dict(l=0, r=0, t=0, b=0),
        bgcolor="#F2F2F2", bordercolor="#CFCFCF", borderwidth=1, font=dict(size=13, color=INK),
        buttons=[dict(label=lbl, method="relayout", args=[{f"{ax}.range": rng for ax in xaxes}])
                 for lbl, rng in windows],
    )])
    # Logo top-right, level with the title, whatever the plot height
    plot_h = height - top - 44
    fig.add_layout_image(dict(
        source=LOGO_B64, xref="paper", yref="paper", x=1.0, y=1 + (top - 10) / plot_h,
        sizex=0.16, sizey=34 / plot_h, xanchor="right", yanchor="top", layer="above",
    ))


def zone_segments(vix, thresholds=VIX_ZONES, values=None) -> list:
    """Split a line into one (x, y) piece per VIX zone, cutting exactly where the VIX crosses a
    threshold so the coloured pieces join up. None in y breaks a zone's line where the VIX leaves it.
    `values` (default: the VIX itself) is the series drawn, e.g. the S&P 500 coloured by the VIX's
    zone; it's linearly interpolated at each crossing."""
    values = vix if values is None else values
    zone = lambda v: sum(v > t for t in thresholds)
    fmt = lambda ts: ts.strftime("%Y-%m-%d %H:%M:%S")
    out = [([], []) for _ in range(len(thresholds) + 1)]
    xs, vs, ys = list(vix.index), vix.tolist(), values.tolist()
    z = zone(vs[0])
    out[z][0].append(fmt(xs[0]))
    out[z][1].append(ys[0])
    for x0, v0, y0, x1, v1, y1 in zip(xs, vs, ys, xs[1:], vs[1:], ys[1:]):
        z1 = zone(v1)
        # Thresholds between the two zones, in the order the line crosses them
        crossed = thresholds[z:z1] if z1 > z else thresholds[z1:z][::-1]
        step = 1 if z1 > z else -1
        for t in crossed:
            frac = (t - v0) / (v1 - v0)
            xt, yt = fmt(x0 + (x1 - x0) * frac), y0 + (y1 - y0) * frac
            out[z][0].extend([xt, xt])
            out[z][1].extend([yt, None])
            z += step
            out[z][0].append(xt)
            out[z][1].append(yt)
        out[z][0].append(fmt(x1))
        out[z][1].append(y1)
    return out


def plot_vix(vix, spx, spx_mode: str = SPX_MODE) -> go.Figure:
    # ISO strings and plain lists: see plot_price_rsi() in generate_index_rsi_web.py
    dates = vix.index.strftime("%Y-%m-%d")
    windows = timeframe_ranges(vix.index)
    fav, unfav = VIX_ZONES

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55], vertical_spacing=0.06)
    # Trace 0: invisible full S&P series for hover and fitAxes(); the visible line is drawn in
    # coloured pieces below (same approach as the VIX, trace 1).
    fig.add_trace(go.Scatter(
        x=dates, y=spx.tolist(), mode="markers", marker=dict(size=4, opacity=0),
        name="S&P 500", hovertemplate="S&P 500: %{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    # Trace 1: invisible full VIX series that carries the hover and feeds fitAxes(). The visible line is
    # drawn as one segment trace per zone (hover skipped), so unified hover never shows a stray
    # value from a neighbouring zone's trace.
    fig.add_trace(go.Scatter(
        x=dates, y=vix.tolist(), mode="markers", marker=dict(size=4, opacity=0),
        name="VIX", hovertemplate="VIX: %{y:.2f}<extra></extra>",
    ), row=2, col=1)
    for (xs, ys), color in zip(zone_segments(vix), ZONE_COLORS):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.8),
                                 hoverinfo="skip", connectgaps=False), row=2, col=1)
    # S&P 500 line: "mirror" = coloured by all three VIX zones; "highlight" = grey, blue while VIX > 30
    thresholds, colors = (VIX_ZONES, ZONE_COLORS) if spx_mode == "mirror" else (VIX_ZONES[-1:], (SPX_COLOR, BLUE))
    for (xs, ys), color in zip(zone_segments(vix, thresholds, spx), colors):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.6),
                                 hoverinfo="skip", connectgaps=False), row=1, col=1)
    for (y0, y1), color in zip([(0, fav), (fav, unfav), (unfav, 200)], ZONE_COLORS):
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=0.06, line_width=0, layer="below", row=2, col=1)
    for y in VIX_ZONES:
        fig.add_hline(y=y, line_dash="dash", line_color="#9A9A9A", line_width=1.2, row=2, col=1)

    base_layout(fig, "<b>CBOE Volatility Index</b> (VIX)  ·  with the S&P 500", windows,
                fit=[dict(ax="yaxis", tr=[0], log=True),
                     dict(ax="yaxis2", tr=[1], inc=list(VIX_ZONES), step=5, floor=0)],
                xaxes=["xaxis", "xaxis2"], height=720)
    fig.update_yaxes(title_text="S&P 500", row=1, col=1, type="log", tickformat=",.0f",
                     tickvals=log_ticks(spx.min(), spx.max()), **AXIS_STYLE)
    fig.update_yaxes(title_text="VIX", row=2, col=1, range=fit_range(vix, VIX_ZONES, floor=0), **AXIS_STYLE)
    return fig


def plot_ratio(spx, ratio) -> go.Figure:
    dates = ratio.index.strftime("%Y-%m-%d")
    windows = timeframe_ranges(ratio.index)
    signal = ratio > RATIO_SIGNAL
    dot_sizes = [5 if s else 0 for s in signal]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(
        x=dates, y=spx.tolist(), mode="lines+markers",
        line=dict(color=SPX_COLOR, width=1.6), marker=dict(color=GREEN, size=dot_sizes, line=dict(width=0)),
        name="S&P 500", hovertemplate="S&P 500: %{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=ratio.tolist(), mode="lines+markers",
        line=dict(color=ORANGE, width=1.6), marker=dict(color=GREEN, size=dot_sizes, line=dict(width=0)),
        name="VIX/VIX3M", hovertemplate="VIX/VIX3M: %{y:.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hrect(y0=RATIO_SIGNAL, y1=5, row=2, col=1, fillcolor=GREEN, opacity=0.10, line_width=0, layer="below")
    fig.add_hline(y=RATIO_SIGNAL, line_dash="dash", line_color=GREEN, line_width=1.4, row=2, col=1)

    base_layout(fig, "<b>VIX / VIX3M Ratio</b>  ·  with the S&P 500", windows,
                fit=[dict(ax="yaxis", tr=[0], log=True),
                     dict(ax="yaxis2", tr=[1], inc=[RATIO_SIGNAL], step=0.1)],
                xaxes=["xaxis", "xaxis2"], height=680)
    fig.update_yaxes(title_text="S&P 500", row=1, col=1, type="log", tickformat=",.0f",
                     tickvals=log_ticks(spx.min(), spx.max()), **AXIS_STYLE)
    fig.update_yaxes(title_text="VIX/VIX3M", row=2, col=1, tickformat=".1f",
                     range=fit_range(ratio, [RATIO_SIGNAL], step=0.1), **AXIS_STYLE)
    return fig


def build_report(spx_mode: str = SPX_MODE) -> str:
    print("Downloading VIX, VIX3M and S&P 500...")
    data = fetch_all(["^VIX", "^VIX3M", "^GSPC"])
    vix = data["^VIX"]["Close"].dropna()
    vix3m = data["^VIX3M"]["Close"].dropna()
    spx = data["^GSPC"]["Close"].dropna()

    spx_vix = spx.reindex(vix.index).ffill()
    ratio = (vix / vix3m).dropna()
    spx_ratio = spx.reindex(ratio.index).ffill()

    return "\n".join([
        '<div class="section"><h2>VIX</h2></div>',
        fig_to_div(plot_vix(vix, spx_vix, spx_mode)),
        '<div class="section"><h2>VIX / VIX3M</h2></div>',
        fig_to_div(plot_ratio(spx_ratio, ratio)),
    ])


# Page script shared with generate_offense_defense_web.py: refits each y axis to the visible x range.
# Inserted into PAGE_TEMPLATE via format(), so it's kept out of the template to avoid doubled braces.
FIT_AXES_JS = """<script>
// Refit each y axis to the visible x range after timeframe buttons, drag-zoom or double-click.
// Which traces feed which axis comes from layout.meta.fit (set in base_layout()).
function niceTicks(lo, hi) {
  if (hi / lo > 3) {
    var t = [];
    for (var e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
      [1, 2, 5].forEach(function (m) { var v = m * Math.pow(10, e); if (v >= lo && v <= hi) t.push(v); });
    return t;
  }
  var raw = (hi - lo) / 5, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
  var step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag, ticks = [];
  for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  return ticks;
}
function fitAxes(id) {
  var gd = document.getElementById(id), meta = gd.layout.meta;
  var toMs = function (v) { return typeof v === "number" ? v : Date.parse(String(v).replace(" ", "T")); };
  var xs = gd.data.map(function (t) { return t.x.map(toMs); });
  function fit() {
    var r = gd.layout.xaxis.range, a = toMs(r[0]), b = toMs(r[1]), upd = {};
    meta.fit.forEach(function (f) {
      var lo = Infinity, hi = -Infinity;
      f.tr.forEach(function (t) {
        for (var i = 0; i < xs[t].length; i++) {
          if (xs[t][i] < a || xs[t][i] > b) continue;
          var y = gd.data[t].y[i];
          if (y < lo) lo = y; if (y > hi) hi = y;
        }
      });
      if (!isFinite(lo)) return;
      (f.inc || []).forEach(function (v) { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      if (f.log) {
        var lpad = (Math.log10(hi) - Math.log10(lo)) * 0.05 || 0.01;
        upd[f.ax + ".range"] = [Math.log10(lo) - lpad, Math.log10(hi) + lpad];
        var ticks = niceTicks(lo, hi);
        upd[f.ax + ".tickvals"] = ticks;
        // Plotly drops ticksuffix on log axes with explicit tickvals, so write the labels when one is wanted
        if (f.suffix) upd[f.ax + ".ticktext"] = ticks.map(function (v) { return v.toLocaleString("en-US") + f.suffix; });
        return;
      }
      var pad = (hi - lo) * 0.07, s = f.step;
      lo = Math.floor((lo - pad) / s) * s; hi = Math.ceil((hi + pad) / s) * s;
      if (f.floor !== undefined && f.floor !== null) lo = Math.max(lo, f.floor);
      upd[f.ax + ".range"] = [lo, hi];
    });
    // Only y keys here, so the handler below ignores the event this fires
    Plotly.relayout(gd, upd);
  }
  gd.on("plotly_relayout", function (ev) {
    if (meta.xaxes.some(function (ax) { return ev[ax + ".autorange"]; })) {
      // Deferred: relayouting inside this event gets overwritten when Plotly finishes the autorange.
      var reset = {};
      meta.xaxes.forEach(function (ax) { reset[ax + ".autorange"] = false; reset[ax + ".range"] = meta.full_range.slice(); });
      setTimeout(function () { Plotly.relayout(gd, reset); }, 0);
      return;
    }
    if (Object.keys(ev).some(function (k) { return k.indexOf("xaxis") === 0; })) fit();
  });
  fit();
}
</script>"""


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VIX Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
{fit_js}
<style>
  body {{ background: #FFFFFF; color: #363636; font-family: Arial, sans-serif; font-size: 16px; line-height: 1.55; margin: 0; padding: 0 0 32px; }}
  header {{ padding: 18px 16px 12px; border-bottom: 1px solid #E6E6E6; }}
  header h1 {{ margin: 0 0 6px; font-size: 28px; color: #363636; }}
  header .meta {{ color: #555555; font-size: 15px; }}
  .intro {{ padding: 12px 16px 0; }}
  .intro p {{ margin: 0 0 8px; }}
  .intro .legend {{ font-size: 15px; color: #555555; margin-top: 16px; }}
  .intro .legend + p {{ margin-top: 16px; }}
  .bar {{ display: inline-block; width: 22px; height: 4px; border-radius: 2px; margin: 0 6px 0 2px; vertical-align: middle; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 4px 0 2px; vertical-align: middle; }}
  .section {{ padding: 16px 16px 0; }}
  .section h2 {{ margin: 0; font-size: 23px; color: #363636; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .source {{ color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }}
</style>
</head>
<body>
<header>
  <h1>VIX Report</h1>
  <div class="meta">Updated {date_str} &middot; CBOE Volatility Index and VIX/VIX3M ratio</div>
</header>
<div class="intro">
  <p>The <b>VIX</b> measures how much volatility options traders expect in the S&amp;P 500 over the next 30 days,
  and is often called the market's fear gauge. At 20 or below, markets are calm, which has generally been a
  favourable backdrop. Between 20 and 30, stress is building, which has tended to be less favourable. Above 30,
  fear is extreme. Those spikes have historically come near market lows, so we treat readings above 30 as a highly
  favourable time to be buying.</p>
  <p class="legend"><span class="bar" style="background:#44A660"></span>Favourable (20 and below)
  &nbsp;&nbsp;<span class="bar" style="background:#A22A2A"></span>Unfavourable (20&ndash;30)
  &nbsp;&nbsp;<span class="bar" style="background:#1F79BE"></span>Highly favourable (above 30)
  <br>The S&amp;P 500 line above the VIX turns blue on days the VIX closed above 30.</p>
  <p>The <b>VIX/VIX3M ratio</b> compares expected volatility over the next month with the next three months.
  Normally the ratio sits below 1, because investors pay more to protect against the unknown further out.
  When it rises above 1, near-term fear has overtaken longer-term fear. That usually happens during sharp
  sell-offs and has often marked a good intermediate-term buying opportunity.</p>
  <p class="legend"><span class="dot" style="background:#44A660"></span>Day the ratio closed above 1
  &nbsp;&nbsp;Charts use daily closes and show the full available history. VIX3M data begins in 2006.</p>
</div>
{charts}
<div class="source">Source: 5i Research, Yahoo Finance.</div>
</body>
</html>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    charts = build_report()
    html = PAGE_TEMPLATE.format(fit_js=FIT_AXES_JS, date_str=datetime.now().strftime("%B %d, %Y"), charts=charts)

    out_path = os.path.join(OUTPUT_DIR, "VIX_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
