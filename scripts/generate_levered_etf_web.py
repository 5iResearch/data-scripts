"""
Website-ready Levered ETF Sentiment page: the levered inverse/long notional volume ratios from the
Inverse ETF Sentiment report, for the Nasdaq 100, S&P 500 and Russell 2000 only, styled to match the
Index RSI, 200-week MA, VIX and Offense vs Defense pages.

Ratio = dollar volume (adjusted close x volume) traded in the 3x inverse ETF as a % of the 3x long ETF:
  Nasdaq 100 SQQQ / TQQQ, S&P 500 SPXU / UPRO, Russell 2000 TZA / TNA.
Above the upper line = heavy hedging / fear (contrarian bullish, blue); below the lower line =
complacency (red). The underlying ETF price is on top, with dots on those days.

Lines (THRESHOLD_MODE): "adaptive" (default) uses the 5th / 95th percentile of the ratio over the trailing
two years, so they follow structural shifts in how these funds trade; "fixed" uses the per-pair levels in
PAIRS (QQQ 90/14 and IWM 130/20 from the Inverse ETF Sentiment report; SPY 110/25 set to match QQQ's hit rate).
Runs off the last completed session: today's partial bar is dropped (see completed_sessions()).

Uses the shared page script, layout and segment helpers from generate_vix_web.py.
Output: outputs/levered-etf-sentiment/Levered_ETF_Sentiment.html
"""

import math
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generate_index_rsi_web import BLUE, RED, fetch_all, fig_to_div, log_ticks, timeframe_ranges
from generate_vix_web import AXIS_STYLE, FIT_AXES_JS, SPX_COLOR, base_layout, zone_segments

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "levered-etf-sentiment")

PAIRS = [
    dict(name="Nasdaq 100", etf="QQQ", inverse="SQQQ", long="TQQQ", low=14, high=90),
    dict(name="S&P 500", etf="SPY", inverse="SPXU", long="UPRO", low=25, high=110),
    dict(name="Russell 2000", etf="IWM", inverse="TZA", long="TNA", low=20, high=130),
]
NORMAL = "#9A9A9A"
ZONE_COLORS = (RED, NORMAL, BLUE)   # below low / between / above high

# "fixed": the per-pair low/high levels above. "adaptive": the 5th / 95th percentile of each ratio over
# the trailing ADAPTIVE_DAYS, so the lines follow structural drift in how these funds are traded
# (e.g. TZA/TNA's median rising from ~54% in 2024 to ~144% in 2026).
THRESHOLD_MODE = "adaptive"
ADAPTIVE_DAYS = 504          # ~2 years of sessions
ADAPTIVE_QUANTILES = (0.05, 0.95)


def completed_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop today's row (New York date). During market hours yfinance returns a partial bar whose
    half-day volume skews the ratio, so the page always runs off the previous session's close."""
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    return df[df.index < today]


def notional_ratio(data: dict, inverse: str, long: str):
    dollars = {t: data[t]["Close"] * data[t]["Volume"] for t in (inverse, long)}
    ratio = (dollars[inverse] / dollars[long] * 100).dropna()
    return ratio[(ratio > 0) & (ratio < float("inf"))]


def adaptive_bands(ratio: pd.Series) -> pd.DataFrame:
    """Rolling low/high lines from the ratio's own trailing history (needs a year before it starts)."""
    roll = ratio.rolling(ADAPTIVE_DAYS, min_periods=ADAPTIVE_DAYS // 2)
    lo_q, hi_q = ADAPTIVE_QUANTILES
    return pd.DataFrame({"low": roll.quantile(lo_q), "high": roll.quantile(hi_q)}).dropna()


def plot_pair(price, ratio, cfg, mode: str = THRESHOLD_MODE) -> go.Figure:
    if mode == "adaptive":
        bands = adaptive_bands(ratio)
        ratio, price = ratio.loc[bands.index], price.loc[bands.index]
        low_s, high_s = bands["low"], bands["high"]
    else:
        low_s = pd.Series(cfg["low"], index=ratio.index)
        high_s = pd.Series(cfg["high"], index=ratio.index)
    # ISO strings and plain lists: see plot_price_rsi() in generate_index_rsi_web.py
    dates = ratio.index.strftime("%Y-%m-%d")
    windows = timeframe_ranges(ratio.index)
    zone = [(r > lo) + (r > hi) for r, lo, hi in zip(ratio, low_s, high_s)]
    # Position of the ratio within its band (low -> 0, high -> 1), so zone_segments() can cut the line
    # at constant thresholds 0 and 1 even when the lines move. Computed in log space, like the axis.
    lr, ll, lh = (s.apply(math.log) for s in (ratio, low_s, high_s))
    position = (lr - ll) / (lh - ll)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(
        x=dates, y=price.tolist(), mode="lines+markers", line=dict(color=SPX_COLOR, width=1.4),
        marker=dict(color=[ZONE_COLORS[z] for z in zone], size=[0 if z == 1 else 5 for z in zone], line=dict(width=0)),
        name=cfg["etf"], hovertemplate=f"{cfg['etf']}: %{{y:,.2f}}<extra></extra>",
    ), row=1, col=1)
    # Trace 1: invisible full ratio series for hover and fitAxes(); visible line is drawn per zone below
    fig.add_trace(go.Scatter(
        x=dates, y=ratio.tolist(), mode="markers", marker=dict(size=4, opacity=0),
        name="Ratio", hovertemplate=f"{cfg['inverse']}/{cfg['long']}: %{{y:.0f}}%<extra></extra>",
    ), row=2, col=1)
    # Traces 2 and 3: the low / high lines (moving in adaptive mode), also fed to fitAxes() so they stay in view
    for s, color, label in [(low_s, RED, "Complacency line"), (high_s, BLUE, "Fear line")]:
        fig.add_trace(go.Scatter(x=dates, y=s.tolist(), mode="lines", line=dict(color=color, width=1.2, dash="dash"),
                                 name=label, hovertemplate=f"{label}: %{{y:.0f}}%<extra></extra>"), row=2, col=1)
    for (xs, ys), color in zip(zone_segments(position, (0, 1), ratio), ZONE_COLORS):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.3),
                                 hoverinfo="skip", connectgaps=False), row=2, col=1)
    if mode != "adaptive":
        fig.add_hrect(y0=0.01, y1=cfg["low"], fillcolor=RED, opacity=0.06, line_width=0, layer="below", row=2, col=1)
        fig.add_hrect(y0=cfg["high"], y1=100000, fillcolor=BLUE, opacity=0.06, line_width=0, layer="below", row=2, col=1)

    base_layout(fig, f"<b>{cfg['name']}</b> ({cfg['etf']})  ·  {cfg['inverse']} / {cfg['long']} ratio", windows,
                fit=[dict(ax="yaxis", tr=[0], log=True), dict(ax="yaxis2", tr=[1, 2, 3], log=True, suffix="%")],
                xaxes=["xaxis", "xaxis2"], height=720)
    fig.update_yaxes(title_text=cfg["etf"], row=1, col=1, type="log", tickformat=",.0f",
                     tickvals=log_ticks(price.min(), price.max()), **AXIS_STYLE)
    # Log scale: the ratio is heavily skewed (most days 20-100%, spikes past 300%). Ticks set by fitAxes().
    fig.update_yaxes(title_text=f"{cfg['inverse']} as % of {cfg['long']}", row=2, col=1, type="log",
                     tickformat=",.0f", ticksuffix="%", **AXIS_STYLE)
    return fig


def build_report(mode: str = THRESHOLD_MODE) -> tuple[pd.Timestamp, str]:
    tickers = [t for p in PAIRS for t in (p["etf"], p["inverse"], p["long"])]
    print("Downloading " + ", ".join(tickers) + "...")
    data = {t: completed_sessions(df) for t, df in fetch_all(tickers).items()}

    parts, as_of = [], None
    for cfg in PAIRS:
        try:
            ratio = notional_ratio(data, cfg["inverse"], cfg["long"])
            price = data[cfg["etf"]]["Close"].reindex(ratio.index).ffill()
        except Exception as e:
            print(f"Error {cfg['etf']}: {e}")
            continue
        parts.append(f'<div class="section"><h2>{cfg["name"]}</h2></div>')
        parts.append(fig_to_div(plot_pair(price, ratio, cfg, mode)))
        as_of = ratio.index[-1] if as_of is None else max(as_of, ratio.index[-1])
    return as_of, "\n".join(parts)


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Levered ETF Sentiment</title>
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
  .bar {{ display: inline-block; width: 22px; height: 4px; border-radius: 2px; margin: 0 6px 0 2px; vertical-align: middle; }}
  .section {{ padding: 16px 16px 0; }}
  .section h2 {{ margin: 0; font-size: 23px; color: #363636; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .source {{ color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }}
</style>
</head>
<body>
<header>
  <h1>Levered ETF Sentiment</h1>
  <div class="meta">Data as of the {as_of_str} close &middot; Nasdaq 100, S&amp;P 500, Russell 2000</div>
</header>
<div class="intro">
  <p>Traders who want to make a big, short-term bet on the market often use triple-leveraged ETFs: one that moves
  three times the index (for example TQQQ for the Nasdaq 100) and one that moves three times the opposite way
  (SQQQ). Comparing the dollar value traded in the two is a real-time read on how traders are positioned.</p>
  <p>The bottom panel shows the dollar volume in the inverse (bearish) fund as a percentage of the long (bullish) fund.
  When it spikes, traders are rushing to bet against the market. That kind of fear has often come near short-term
  lows, so we treat it as a contrarian positive. When it drops very low, almost everyone is betting on further gains,
  a sign of complacency that has tended to come before pullbacks.</p>
  <p class="legend"><span class="bar" style="background:#1F79BE"></span>Heavy hedging / fear (above the upper line)
  &nbsp;&nbsp;<span class="bar" style="background:#A22A2A"></span>Complacency (below the lower line)
  <br>Dots on the price chart mark those days. {levels_text}</p>
</div>
{charts}
<div class="source">Source: 5i Research, Yahoo Finance.</div>
</body>
</html>
"""


LEVELS_TEXT = {
    "fixed": ("Levels: Nasdaq 100 (SQQQ/TQQQ) 90% and 14%; S&amp;P 500 (SPXU/UPRO) 110% and 25%; Russell 2000 "
              "(TZA/TNA) 130% and 20%. Daily data since each fund pair launched (2008&ndash;2010)."),
    "adaptive": ("The lines adjust over time: the upper line is the top 5% of that ratio's readings over the past two "
                 "years and the lower line is the bottom 5%, so a day only counts as fear or complacency if it stands "
                 "out from recent trading. Daily data; each chart starts a year after its fund pair launched "
                 "(2009&ndash;2011)."),
}


def render(mode: str = THRESHOLD_MODE) -> str:
    as_of, charts = build_report(mode)
    return PAGE_TEMPLATE.format(fit_js=FIT_AXES_JS, as_of_str=as_of.strftime("%B %d, %Y"),
                                levels_text=LEVELS_TEXT[mode], charts=charts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    html = render()

    out_path = os.path.join(OUTPUT_DIR, "Levered_ETF_Sentiment.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
