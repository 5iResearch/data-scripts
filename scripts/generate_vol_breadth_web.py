"""
Website-ready Volatility Breadth page: two gauges built from eight CBOE volatility indexes, four on stocks
(VIX, VXN, RVX, VXD) and four on commodities (OVX, GVZ, VXSLV, VXGDX).

  1. Fear breadth: how many indexes are in their own "fear" zone today, charted against the S&P 500.
  2. Where the fear is: whether stock volatility, commodity volatility, both or neither are "uneasy",
     with the S&P 500 line coloured by that regime.

Each index's zones are set at the same percentiles of its own history as VIX 20 / 30 (VIX closed below
20 on 62.5% of days since 1990 and below 30 on 91.7%), so "fear" always means "as rare as VIX above 30" and
"uncertain" means "as rare as VIX above 20". Cutoffs and the historical returns quoted on the page come from
the Oct 2026 volatility index study (Mar 2011 - Oct 2026) and are fixed here, not recomputed daily.

CBOE paused VXSLV (Feb 2022 - May 2025) and VXGDX (Feb 2022 - Sep 2025), so counts are over the indexes
available each day: eight normally, six during that gap.

Data: CBOE daily history files (Yahoo no longer carries RVX, VXSLV or VXGDX). S&P 500 from Yahoo.
Output: outputs/vol-breadth/Vol_Breadth_Report.html
"""

import html
import os
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

from generate_index_rsi_web import (
    BLUE, GREEN, ORANGE, RED, fetch_all, fig_to_div, log_ticks, timeframe_ranges, with_fullscreen,
)
from generate_vix_web import AXIS_STYLE, FIT_AXES_JS, base_layout, zone_segments

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "vol-breadth")

# symbol: (what it tracks, uncertain cutoff (VIX-20 equivalent), fear cutoff (VIX-30 equivalent))
EQUITY = {
    "VIX":   ("S&P 500", 20.0, 30.0),
    "VXN":   ("Nasdaq-100", 24.2, 45.7),
    "RVX":   ("Russell 2000", 23.9, 34.6),
    "VXD":   ("Dow", 17.6, 27.5),
}
COMMODITY = {
    "OVX":   ("Crude oil", 39.5, 57.5),
    "GVZ":   ("Gold", 19.1, 27.8),
    "VXSLV": ("Silver", 32.5, 52.0),
    "VXGDX": ("Gold miners", 41.1, 50.9),
}
INDEXES = EQUITY | COMMODITY
CORE = ["VIX", "VXN", "RVX", "VXD", "OVX", "GVZ"]   # never paused; a day needs all six
HISTORY_START = "2011-03-16"   # first day all eight indexes have data

GREY = "#9A9A9A"
# Fear breadth: (lowest count, label, colour). Grey = normal, green = favourable, blue = highly favourable.
BANDS = [(0, "Calm", GREY), (1, "Isolated fear", GREY), (2, "Spreading fear", GREEN),
         (4, "Broad fear", BLUE), (6, "Fear everywhere", BLUE)]
# Where the fear is: regime code -> (label, colour, S&P 500 avg next-year return, % of days up a year later)
REGIMES = {
    0: ("Neither uneasy", GREY, 13.1, 89),
    1: ("Stocks uneasy, commodities calm", RED, 10.2, 71),
    2: ("Commodities uneasy, stocks calm", ORANGE, 15.2, 94),
    3: ("Both uneasy", GREEN, 22.2, 92),
}
SPX_COLOR = "#4A4A4A"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def band(count: int):
    return [b for b in BANDS if count >= b[0]][-1]


def fetch_cboe(symbol: str) -> pd.Series:
    """Daily closes from CBOE's history CSV, falling back to its chart-data JSON."""
    try:
        r = requests.get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv",
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.upper() for c in df.columns]
        col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
        s = pd.Series(df[col].astype(float).values, index=pd.to_datetime(df["DATE"]))
    except Exception as e:
        print(f"  {symbol}: CSV failed ({e}), trying JSON")
        r = requests.get(f"https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_{symbol}.json",
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json()["data"])
        s = pd.Series(df["close"].astype(float).values, index=pd.to_datetime(df["date"]))
    s = s[s > 0].sort_index()
    return s[~s.index.duplicated(keep="last")]


def load_data():
    print("Downloading CBOE volatility indexes...")
    vols = pd.DataFrame({s: fetch_cboe(s) for s in INDEXES}).loc[HISTORY_START:]
    vols = vols.ffill(limit=3)   # fills holiday mismatches only; the 2022-25 VXSLV/VXGDX gap stays empty
    vols = vols[vols[CORE].notna().all(axis=1)]
    print("Downloading S&P 500...")
    # Two tickers: the shared fetch_all() mishandles a single-ticker download
    spx = fetch_all(["^GSPC", "^VIX"])["^GSPC"]["Close"].dropna()
    spx.index = spx.index.tz_localize(None).normalize()
    vols = vols[vols.index.isin(spx.index)]
    return vols, spx.reindex(vols.index)


def signals(vols: pd.DataFrame):
    level = lambda group, i: pd.DataFrame({s: vols[s] >= group[s][i] for s in group}).where(vols[list(group)].notna())
    fear = level(INDEXES, 2)
    count = fear.sum(axis=1).astype(int)
    avail = vols.notna().sum(axis=1)
    names = fear.apply(lambda r: ", ".join(r.index[r == True]) or "none", axis=1)
    eq_unc, cm_unc = level(EQUITY, 1).sum(axis=1), level(COMMODITY, 1).sum(axis=1)
    cm_avail = vols[list(COMMODITY)].notna().sum(axis=1)
    # Uneasy = at least 2 of the 4 stock indexes / at least half the available commodity indexes at "uncertain"
    eq_on, cm_on = eq_unc >= 2, cm_unc >= np.ceil(cm_avail / 2)
    regime = eq_on.astype(int) + 2 * cm_on.astype(int)
    return count, avail, names, regime, eq_unc.astype(int), cm_unc.astype(int), cm_avail


def plot_breadth(count, avail, names, spx) -> go.Figure:
    dates = count.index.strftime("%Y-%m-%d")
    windows = timeframe_ranges(count.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.58, 0.42], vertical_spacing=0.06)
    # Trace 0: invisible full S&P series for hover and fitAxes(); visible line drawn in pieces below
    fig.add_trace(go.Scatter(x=dates, y=spx.tolist(), mode="markers", marker=dict(size=4, opacity=0),
                             hovertemplate="S&P 500: %{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(
        x=dates, y=count.tolist(), marker=dict(color=[band(c)[2] for c in count], line=dict(width=0)),
        customdata=np.column_stack([avail.tolist(), names.tolist()]).tolist(),
        hovertemplate="In fear zone: %{y} of %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
    ), row=2, col=1)
    # S&P 500 grey, blue on days 4+ indexes were in their fear zone
    for (xs, ys), color in zip(zone_segments(count.astype(float), (3.5,), spx), (SPX_COLOR, BLUE)):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.6),
                                 hoverinfo="skip", connectgaps=False), row=1, col=1)
    base_layout(fig, "<b>Fear Breadth</b>  ·  indexes in their fear zone, with the S&P 500", windows,
                fit=[dict(ax="yaxis", tr=[0], log=True)], xaxes=["xaxis", "xaxis2"], height=680)
    fig.update_layout(bargap=0)
    fig.update_yaxes(title_text="S&P 500", row=1, col=1, type="log", tickformat=",.0f",
                     tickvals=log_ticks(spx.min(), spx.max()), **AXIS_STYLE)
    fig.update_yaxes(title_text="Indexes in fear", row=2, col=1, range=[0, 8.3], tickvals=[0, 2, 4, 6, 8],
                     **AXIS_STYLE)
    return fig


def plot_regime(regime, eq_unc, cm_unc, cm_avail, spx) -> go.Figure:
    dates = regime.index.strftime("%Y-%m-%d")
    windows = timeframe_ranges(regime.index)
    labels = [REGIMES[r][0] for r in regime]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=spx.tolist(), mode="markers", marker=dict(size=4, opacity=0),
        customdata=np.column_stack([labels, eq_unc.tolist(), cm_unc.tolist(), cm_avail.tolist()]).tolist(),
        hovertemplate=("S&P 500: %{y:,.2f}<br>%{customdata[0]}<br>"
                       "Stocks uneasy: %{customdata[1]} of 4 · Commodities: %{customdata[2]} of %{customdata[3]}<extra></extra>"),
    ))
    for (xs, ys), code in zip(zone_segments(regime.astype(float), (0.5, 1.5, 2.5), spx), REGIMES):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=REGIMES[code][1], width=1.8),
                                 hoverinfo="skip", connectgaps=False))
    base_layout(fig, "<b>Where the Fear Is</b>  ·  S&P 500 coloured by regime", windows,
                fit=[dict(ax="yaxis", tr=[0], log=True)], xaxes=["xaxis"], height=520)
    fig.update_yaxes(title_text="S&P 500", type="log", tickformat=",.0f",
                     tickvals=log_ticks(spx.min(), spx.max()), **AXIS_STYLE)
    return fig


def tiles_html(last: pd.Series) -> str:
    def group(title, members):
        out = f'<div class="grp">{title}</div><div class="tiles">'
        for s, (name, unc, fear) in members.items():
            v = last[s]
            zone, zc = ("Fear", BLUE) if v >= fear else ("Uncertain", ORANGE) if v >= unc else ("Calm", GREY)
            out += (f'<div class="tile" style="border-top-color:{zc}"><div class="hd"><b>{s}</b> {html.escape(name)}</div>'
                    f'<div class="lv">{v:.1f} <span style="color:{zc}">{zone}</span></div>'
                    f'<div class="th">Uncertain {unc:g} &middot; Fear {fear:g}</div></div>')
        return out + "</div>"
    return group("Stock volatility", EQUITY) + group("Commodity volatility", COMMODITY)


def breadth_gauge(date, n: int, avail: int) -> str:
    _, label, color = band(n)
    cells = "".join(f'<span style="background:{band(i + 1)[2] if i < n else "#E6E6E6"}"></span>' for i in range(avail))
    return f"""<div class="gauge">
  <div class="big"><span class="num" style="color:{color}">{n}</span><span class="of">of {avail}</span></div>
  <div class="read"><div class="lbl" style="color:{color}">{label}</div>
    <div class="cells">{cells}</div>
    <div class="sub">volatility indexes in their fear zone on {date}</div></div>
</div>"""


def regime_gauge(code: int, eq: int, cm: int, cm_avail: int) -> str:
    def cell(c):
        label, color, ret, up = REGIMES[c]
        on = c == code
        style = f"border-color:{color};background:{color}1A" if on else ""
        now = '<div class="now">Today</div>' if on else ""
        return (f'<div class="q{" on" if on else ""}" style="{style}">{now}<div class="ql" style="color:{color}">{label}</div>'
                f'<div class="qs">S&amp;P 500 next year: <b>{ret:+.0f}%</b> avg, up {up}% of the time</div></div>')
    return f"""<div class="quad">
  <div class="axis-c"></div><div class="axis-c">Commodities calm</div><div class="axis-c">Commodities uneasy</div>
  <div class="axis-r">Stocks calm</div>{cell(0)}{cell(2)}
  <div class="axis-r">Stocks uneasy</div>{cell(1)}{cell(3)}
</div>
<p class="sub pad">Today: {eq} of 4 stock indexes and {cm} of {cm_avail} commodity indexes at &ldquo;uncertain&rdquo; or higher.</p>"""


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Volatility Breadth</title>
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
  .sub {{ color: #555555; font-size: 15px; }}
  .pad {{ padding: 0 16px; margin: 6px 0 0; }}
  .grp {{ padding: 14px 16px 0; font-weight: 700; font-size: 15px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; padding: 6px 16px 0; }}
  .tile {{ border: 1px solid #E6E6E6; border-top: 4px solid #9A9A9A; border-radius: 4px; padding: 8px 12px; }}
  .tile .hd {{ font-size: 14px; color: #555555; }}
  .tile .hd b {{ color: #363636; margin-right: 4px; }}
  .tile .lv {{ font-size: 26px; font-weight: 700; line-height: 1.3; }}
  .tile .lv span {{ font-size: 15px; margin-left: 4px; }}
  .tile .th {{ color: #777777; font-size: 12px; white-space: nowrap; }}
  .gauge {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px 26px; padding: 12px 16px 6px; }}
  .gauge .big {{ white-space: nowrap; }}
  .gauge .num {{ font-size: 72px; font-weight: 700; line-height: 1; }}
  .gauge .of {{ font-size: 22px; color: #555555; margin-left: 8px; }}
  .gauge .read {{ flex: 1; min-width: 240px; }}
  .gauge .lbl {{ font-size: 24px; font-weight: 700; }}
  .gauge .cells {{ display: flex; gap: 5px; margin: 8px 0 6px; max-width: 420px; }}
  .gauge .cells span {{ flex: 1; height: 14px; border-radius: 3px; }}
  .quad {{ display: grid; grid-template-columns: 110px minmax(0, 1fr) minmax(0, 1fr); gap: 8px; padding: 12px 16px 0; max-width: 820px; }}
  .quad .axis-c {{ font-size: 14px; font-weight: 700; color: #555555; text-align: center; align-self: end; }}
  .quad .axis-r {{ font-size: 14px; font-weight: 700; color: #555555; align-self: center; }}
  .quad .q {{ border: 2px solid #E6E6E6; border-radius: 6px; padding: 10px 12px; opacity: 0.55; }}
  .quad .q.on {{ opacity: 1; border-width: 3px; }}
  .quad .ql {{ font-weight: 700; font-size: 16px; }}
  .quad .qs {{ font-size: 13px; color: #555555; margin-top: 4px; }}
  .quad .now {{ display: inline-block; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em;
               color: #FFFFFF; background: #363636; border-radius: 3px; padding: 1px 6px; margin-bottom: 4px; }}
  .section {{ padding: 26px 16px 0; }}
  .section h2 {{ margin: 0; font-size: 23px; color: #363636; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .source {{ color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }}
  @media (max-width: 700px) {{
    .tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .quad {{ grid-template-columns: 70px minmax(0, 1fr) minmax(0, 1fr); }}
    .quad .axis-c, .quad .axis-r {{ font-size: 12px; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Volatility Breadth</h1>
  <div class="meta">Updated {date_str} &middot; Eight CBOE volatility indexes across stocks and commodities</div>
</header>
<div class="intro">
  <p>A spike in one fear gauge can be a passing scare. When fear shows up across stocks, oil, gold and silver
  at the same time, it has usually come near a major low. This page tracks eight volatility indexes, each with
  its own <b>uncertain</b> and <b>fear</b> levels, set at the readings that are as rare for that index as a VIX above
  20 and above 30.</p>
</div>
<div class="section"><h2>Today's readings</h2></div>
{tiles}
<div class="section"><h2>Fear breadth</h2></div>
<div class="intro">
  <p>How many of the indexes are in their fear zone. Since 2011, the S&amp;P 500's average return over the
  following year has been about +14% with none or one, +22% with two or three, +28% with four or five and +49%
  with six or more.</p>
  <p class="legend"><span class="bar" style="background:{grey}"></span>0&ndash;1: normal
  &nbsp;&nbsp;<span class="bar" style="background:{green}"></span>2&ndash;3: favourable
  &nbsp;&nbsp;<span class="bar" style="background:{blue}"></span>4 or more: highly favourable
  <br>The S&amp;P 500 line turns blue on days four or more indexes were in their fear zone.</p>
</div>
{breadth_gauge}
{breadth_chart}
<div class="section"><h2>Where the fear is</h2></div>
<div class="intro">
  <p>Whether the unease is in stocks, commodities or both. A group counts as uneasy when at least two of its
  indexes are at their uncertain level or higher. Nervous stocks with calm commodities has been the weakest
  backdrop for the S&amp;P 500: the market worrying about itself, as in late 2021 and early 2022. Unease in both at
  once has been the strongest.</p>
</div>
{regime_gauge}
<div class="intro"><p class="legend"><span class="bar" style="background:{grey}"></span>Neither uneasy
  &nbsp;&nbsp;<span class="bar" style="background:{red}"></span>Stocks uneasy only
  &nbsp;&nbsp;<span class="bar" style="background:{orange}"></span>Commodities uneasy only
  &nbsp;&nbsp;<span class="bar" style="background:{green}"></span>Both uneasy
  <br>S&amp;P 500 returns in the boxes are averages since 2011 for days in each regime.</p></div>
{regime_chart}
<div class="source">Source: 5i Research, CBOE, Yahoo Finance. CBOE paused the silver and gold-miner indexes from
Feb 2022 to mid-2025; counts in that period use the six indexes available.</div>
</body>
</html>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vols, spx = load_data()
    count, avail, names, regime, eq_unc, cm_unc, cm_avail = signals(vols)
    date = vols.index[-1].strftime("%B %d, %Y").replace(" 0", " ")
    page = PAGE_TEMPLATE.format(
        fit_js=FIT_AXES_JS, date_str=datetime.now().strftime("%B %d, %Y"),
        grey=GREY, green=GREEN, blue=BLUE, red=RED, orange=ORANGE,
        tiles=tiles_html(vols.iloc[-1]),
        breadth_gauge=breadth_gauge(date, int(count.iloc[-1]), int(avail.iloc[-1])),
        breadth_chart=fig_to_div(plot_breadth(count, avail, names, spx)),
        regime_gauge=regime_gauge(int(regime.iloc[-1]), int(eq_unc.iloc[-1]), int(cm_unc.iloc[-1]), int(cm_avail.iloc[-1])),
        regime_chart=fig_to_div(plot_regime(regime, eq_unc, cm_unc, cm_avail, spx)),
    )
    out_path = os.path.join(OUTPUT_DIR, "Vol_Breadth_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(with_fullscreen(page))
    print(f"Saved: {out_path}  (latest {vols.index[-1].date()}: {int(count.iloc[-1])} of {int(avail.iloc[-1])} in fear, "
          f"regime: {REGIMES[int(regime.iloc[-1])][0]}; 4+ in fear by year: "
          f"{count[count >= 4].groupby(count[count >= 4].index.year).size().to_dict()})")


if __name__ == "__main__":
    main()
