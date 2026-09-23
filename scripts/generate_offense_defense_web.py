"""
Website-ready Offense vs Defense page: a risk-on / risk-off gauge built from US sector SPDRs,
styled to match the Index RSI, 200-week MA and VIX pages.

  Offense  XLK, XLY, XLC, XLF, XLI   (cyclical / growth sectors)
  Defense  XLP, XLU, XLV             (defensive sectors)

Each basket is an equal-weight, daily-rebalanced total-return index of its members (a sector joins
its basket when it starts trading, e.g. XLC in 2018, so history runs back to the SPDRs' 1998 launch).
The ratio is offense / defense. Above its 200-day average = risk-on (green), below = risk-off (blue).

One chart: S&P 500 (top, grey, blue in risk-off) and, by default, an oscillator of the ratio's %
distance from its 200-day average (bottom, green above zero / blue below). VIEW = "ratio" shows the
ratio itself with its average instead.
Uses the shared page script, layout and segment helpers from generate_vix_web.py.
Output: outputs/offense-defense/Offense_Defense_Report.html
"""

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generate_index_rsi_web import BLUE, GREEN, fetch_all, fig_to_div, log_ticks, timeframe_ranges
from generate_vix_web import AXIS_STYLE, FIT_AXES_JS, SPX_COLOR, base_layout, zone_segments

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "offense-defense")

OFFENSE = {"XLK": "Technology", "XLY": "Consumer Discretionary", "XLC": "Communication Services",
           "XLF": "Financials", "XLI": "Industrials"}
DEFENSE = {"XLP": "Consumer Staples", "XLU": "Utilities", "XLV": "Health Care"}
MA_DAYS = 200
MA_COLOR = "#9A9A9A"
RISK_OFF = BLUE
VIEW = "oscillator"   # bottom panel: "ratio" (ratio + 200-day average) or "oscillator" (% from the average)


def basket_index(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight, daily-rebalanced index of the columns; each member counts from its first price."""
    rets = prices.pct_change(fill_method=None).mean(axis=1, skipna=True)
    return (1 + rets.fillna(0)).cumprod()


def offense_defense_ratio(data: dict) -> pd.DataFrame:
    prices = pd.DataFrame({t: data[t]["Close"] for t in [*OFFENSE, *DEFENSE]}).dropna(how="all")
    df = pd.DataFrame({"Ratio": basket_index(prices[list(OFFENSE)]) / basket_index(prices[list(DEFENSE)])})
    df["MA"] = df["Ratio"].rolling(MA_DAYS).mean()
    return df.dropna()


def plot_offense_defense(df, spx, view: str = VIEW) -> go.Figure:
    # ISO strings and plain lists: see plot_price_rsi() in generate_index_rsi_web.py
    dates = df.index.strftime("%Y-%m-%d")
    windows = timeframe_ranges(df.index)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55], vertical_spacing=0.06)
    # Traces 0 and 1: invisible full S&P and ratio series that carry the hover and feed fitAxes(); the
    # visible lines are drawn in regime-coloured pieces below (hover skipped), as on the VIX chart.
    fig.add_trace(go.Scatter(
        x=dates, y=spx.tolist(), mode="markers", marker=dict(size=4, opacity=0),
        name="S&P 500", hovertemplate="S&P 500: %{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    gap = df["Ratio"] - df["MA"]
    if view == "oscillator":
        # % distance of the ratio from its 200-day average: zero crossings are the regime changes
        osc = (df["Ratio"] / df["MA"] - 1) * 100
        fig.add_trace(go.Scatter(
            x=dates, y=osc.tolist(), mode="markers", marker=dict(size=4, opacity=0),
            name="vs 200-day avg", hovertemplate="vs 200-day avg: %{y:+.1f}%<extra></extra>",
        ), row=2, col=1)
        for (xs, ys), color, fill in zip(zone_segments(osc, (0,)), (RISK_OFF, GREEN),
                                         ("rgba(31,121,190,0.22)", "rgba(68,166,96,0.22)")):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.4),
                                     fill="tozeroy", fillcolor=fill, hoverinfo="skip", connectgaps=False),
                          row=2, col=1)
        fig.add_hline(y=0, line_color="#9A9A9A", line_width=1.2, row=2, col=1)
        bottom_fit = dict(ax="yaxis2", tr=[1], inc=[0], step=5)
    else:
        fig.add_trace(go.Scatter(
            x=dates, y=df["Ratio"].tolist(), mode="markers", marker=dict(size=4, opacity=0),
            name="Offense/Defense", hovertemplate="Offense/Defense: %{y:.3f}<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=df["MA"].tolist(), mode="lines", line=dict(color=MA_COLOR, width=1.4, dash="dot"),
            name="200-day avg", hovertemplate="200-day avg: %{y:.3f}<extra></extra>",
        ), row=2, col=1)
        # Ratio pieces cut where it crosses its average: blue risk-off / green risk-on
        for (xs, ys), color in zip(zone_segments(gap, (0,), df["Ratio"]), (RISK_OFF, GREEN)):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.8),
                                     hoverinfo="skip", connectgaps=False), row=2, col=1)
        bottom_fit = dict(ax="yaxis2", tr=[1, 2], log=True)
    # S&P 500: blue risk-off / grey otherwise
    for (xs, ys), color in zip(zone_segments(gap, (0,), spx), (RISK_OFF, SPX_COLOR)):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.6),
                                 hoverinfo="skip", connectgaps=False), row=1, col=1)

    base_layout(fig, "<b>Offense vs Defense</b>  ·  with the S&P 500", windows,
                fit=[dict(ax="yaxis", tr=[0], log=True), bottom_fit],
                xaxes=["xaxis", "xaxis2"], height=720)
    fig.update_yaxes(title_text="S&P 500", row=1, col=1, type="log", tickformat=",.0f",
                     tickvals=log_ticks(spx.min(), spx.max()), **AXIS_STYLE)
    # Bottom axis range / ticks are set on load by fitAxes() in the page script
    if view == "oscillator":
        fig.update_yaxes(title_text="% vs 200-day avg", row=2, col=1, ticksuffix="%", **AXIS_STYLE)
    else:
        fig.update_yaxes(title_text="Offense / Defense", row=2, col=1, type="log", tickformat=".2f", **AXIS_STYLE)
    return fig


def build_report(view: str = VIEW) -> tuple[str, str]:
    print("Downloading sector SPDRs and S&P 500...")
    data = fetch_all([*OFFENSE, *DEFENSE, "^GSPC"])
    df = offense_defense_ratio(data)
    spx = data["^GSPC"]["Close"].reindex(df.index).ffill()

    on = df["Ratio"].iloc[-1] > df["MA"].iloc[-1]
    # First day of the current regime: the day after it last closed the other way
    same = (df["Ratio"] > df["MA"]) == on
    since = same.index[0] if same.all() else same.index[same.index.get_loc(same[~same].index[-1]) + 1]
    status = (f'<span class="pill {"on" if on else "off"}">{"Risk-on" if on else "Risk-off"}</span> '
              f'since {since.strftime("%B %d, %Y")}: the ratio is {abs(df["Ratio"].iloc[-1] / df["MA"].iloc[-1] - 1) * 100:.1f}% '
              f'{"above" if on else "below"} its 200-day average.')

    charts = "\n".join([
        '<div class="section"><h2>Offense vs Defense</h2></div>',
        fig_to_div(plot_offense_defense(df, spx, view)),
    ])
    return status, charts


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Offense vs Defense</title>
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
  .status {{ font-size: 17px; }}
  .pill {{ display: inline-block; padding: 1px 10px; border-radius: 12px; font-size: 15px; font-weight: bold; color: #FFFFFF; }}
  .pill.on {{ background: #44A660; }} .pill.off {{ background: #1F79BE; }}
  .bar {{ display: inline-block; width: 22px; height: 4px; border-radius: 2px; margin: 0 6px 0 2px; vertical-align: middle; }}
  .bar.dotted {{ height: 0; border-top: 3px dotted #9A9A9A; border-radius: 0; }}
  .section {{ padding: 16px 16px 0; }}
  .section h2 {{ margin: 0; font-size: 23px; color: #363636; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .source {{ color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }}
</style>
</head>
<body>
<header>
  <h1>Offense vs Defense</h1>
  <div class="meta">Updated {date_str} &middot; US sector risk-on / risk-off gauge</div>
</header>
<div class="intro">
  <p class="status">{status}</p>
  <p>This chart compares the market's <b>offensive</b> sectors with its <b>defensive</b> ones. Offensive sectors
  (technology, consumer discretionary, communication services, financials and industrials) do best when investors
  are confident and the economy is growing. Defensive sectors (consumer staples, utilities and health care) hold up
  better when investors are worried, because people keep buying groceries, paying utility bills and seeing doctors
  in any economy.</p>
  <p>We track the offensive group's performance relative to the defensive group's. When offense is outperforming,
  investors are leaning into risk. When defense is outperforming, they are seeking safety. The bottom panel shows
  how far that relationship sits above or below its own 200-day average. Above zero signals a <b>risk-on</b>
  market, below zero a <b>risk-off</b> market, and the further from zero, the stronger the lean.</p>
  <p class="legend"><span class="bar" style="background:#44A660"></span>Risk-on (above zero)
  &nbsp;&nbsp;<span class="bar" style="background:#1F79BE"></span>Risk-off (below zero)
  <br>The S&amp;P 500 line turns blue during risk-off periods.
  <br>Each group is an equal-weighted basket of the Select Sector SPDR ETFs (XLK, XLY, XLC, XLF, XLI versus XLP, XLU,
  XLV), including dividends. Daily closes since 1998; communication services (XLC) joins in 2018.</p>
</div>
{charts}
<div class="source">Source: 5i Research, Yahoo Finance.</div>
</body>
</html>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    status, charts = build_report()
    html = PAGE_TEMPLATE.format(fit_js=FIT_AXES_JS, date_str=datetime.now().strftime("%B %d, %Y"),
                                status=status, charts=charts)

    out_path = os.path.join(OUTPUT_DIR, "Offense_Defense_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
